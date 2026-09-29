//! A run's write-ahead log: one file per (run, worker), `wal-<worker>.jsonl` in the run's
//! directory. Each line is one [`Event`] in canonical JSON, chained to the one before it by
//! hash and fsync'd before the step that caused it continues. Opening or reading a log verifies
//! every line and refuses the whole log at the first that does not hold; nothing is skipped.

use crate::{canonical, Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// One logged event. `event_id` is the `sha256:` hash of the event without it;
/// `previous_event_hash` is the `event_id` before it in the same (run, worker) log, null first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub run_id: String,
    pub worker_id: String,
    pub sequence: u64,
    pub kind: String,
    pub created_at: String,
    pub previous_event_hash: Option<String>,
    pub payload_hash: String,
    pub payload: Value,
    pub event_id: String,
}

impl Event {
    /// The `sha256:` hash of this event without `event_id`.
    pub fn id(&self) -> Result<String> {
        let mut v = serde_json::to_value(self)?;
        v.as_object_mut()
            .expect("an event is an object")
            .remove("event_id");
        Ok(canonical::hash(&v)?)
    }
}

/// The open log of one (run, worker), held exclusively by this process.
#[derive(Debug)]
pub struct Wal {
    run: String,
    worker: String,
    path: PathBuf,
    file: File,
    sequence: u64,
    last: Option<String>,
    broken: bool,
}

/// A run or worker id: 1–128 of `[A-Za-z0-9._:@+-]`, not starting with `.`.
fn id(what: &str, s: &str) -> Result<()> {
    let ok = !s.is_empty()
        && s.len() <= 128
        && !s.starts_with('.')
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:@+-".contains(&b));
    if ok {
        Ok(())
    } else {
        Err(Error::Invalid(format!(
            "{what} id {s:?}: 1-128 of [A-Za-z0-9._:@+-], not starting with '.'"
        )))
    }
}

impl Wal {
    /// The log of `worker` in run `run` under `dir`: verified to its last line when it exists,
    /// created when it does not, and locked against every other writer.
    pub fn open(dir: &Path, run: &str, worker: &str) -> Result<Wal> {
        id("run", run)?;
        id("worker", worker)?;
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("wal-{worker}.jsonl"));
        let fresh = !path.exists();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        // A child process between fork and exec holds a copy of the descriptor, and with it the
        // lock, for a moment; another writer holds it for as long as it runs.
        let mut tries = 0;
        while let Err(e) = file.try_lock() {
            tries += 1;
            if tries == 40 {
                return Err(Error::Invalid(format!(
                    "{}: held by another writer ({e})",
                    path.display()
                )));
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        if fresh {
            File::open(dir)?.sync_all()?;
        }
        let events = read(&path, run, worker)?;
        Ok(Wal {
            run: run.into(),
            worker: worker.into(),
            sequence: events.len() as u64,
            last: events.last().map(|e| e.event_id.clone()),
            path,
            file,
            broken: false,
        })
    }

    /// Append one event and fsync it. After a failed append the log takes no more.
    pub fn append<T: Serialize + ?Sized>(&mut self, kind: &str, payload: &T) -> Result<Event> {
        if self.broken {
            return Err(Error::Invalid(format!(
                "{}: an earlier append failed; the log takes no more",
                self.path.display()
            )));
        }
        let wrote = self.write(kind, payload);
        self.broken = wrote.is_err();
        wrote
    }

    fn write<T: Serialize + ?Sized>(&mut self, kind: &str, payload: &T) -> Result<Event> {
        let payload = serde_json::from_slice(&canonical::to_vec(payload)?)?;
        let mut e = Event {
            run_id: self.run.clone(),
            worker_id: self.worker.clone(),
            sequence: self.sequence + 1,
            kind: kind.into(),
            created_at: crate::wire::now(),
            previous_event_hash: self.last.clone(),
            payload_hash: canonical::hash(&payload)?,
            payload,
            event_id: String::new(),
        };
        e.event_id = e.id()?;
        let mut line = canonical::to_vec(&e)?;
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.file.sync_data()?;
        self.sequence = e.sequence;
        self.last = Some(e.event_id.clone());
        Ok(e)
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Every event of the log at `path`, which must be `run`'s and `worker`'s. Refused, with the
/// line, when a line is torn, not canonical, not the next in sequence, not chained to the one
/// before it, of another run or worker, or does not hash to its `payload_hash` and `event_id`.
pub fn read(path: &Path, run: &str, worker: &str) -> Result<Vec<Event>> {
    let bytes = std::fs::read(path)?;
    let bad = |n: usize, why: String| Error::Invalid(format!("{}:{n}: {why}", path.display()));
    let mut lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    // A complete log ends with a newline, so the split's last piece is empty.
    if lines.pop() != Some(&[][..]) {
        return Err(bad(lines.len() + 1, "torn line: no newline".into()));
    }
    let mut out: Vec<Event> = Vec::new();
    for (i, line) in lines.into_iter().enumerate() {
        let n = i + 1;
        let value = canonical::parse(line).map_err(|e| bad(n, e.to_string()))?;
        let e: Event = serde_json::from_value(value).map_err(|e| bad(n, e.to_string()))?;
        if canonical::to_vec(&e).map_err(|e| bad(n, e.to_string()))? != line {
            return Err(bad(n, "not canonical JSON".into()));
        }
        if e.run_id != run {
            return Err(bad(n, format!("run {:?}, this log is {run:?}'s", e.run_id)));
        }
        if e.worker_id != worker {
            return Err(bad(
                n,
                format!("worker {:?}, this log is {worker:?}'s", e.worker_id),
            ));
        }
        if e.sequence != n as u64 {
            return Err(bad(n, format!("sequence {}, expected {n}", e.sequence)));
        }
        let previous = out.last().map(|p| p.event_id.clone());
        if e.previous_event_hash != previous {
            return Err(bad(
                n,
                "previous_event_hash does not chain to the line before".into(),
            ));
        }
        if canonical::hash(&e.payload)? != e.payload_hash {
            return Err(bad(n, "payload does not hash to payload_hash".into()));
        }
        if e.id()? != e.event_id {
            return Err(bad(n, "event does not hash to event_id".into()));
        }
        out.push(e);
    }
    Ok(out)
}

/// Every worker's log of `run` in `dir`, each verified, merged by [`merge`].
pub fn gather(dir: &Path, run: &str) -> Result<Vec<Event>> {
    let mut logs = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if let Some(worker) = name
            .strip_prefix("wal-")
            .and_then(|n| n.strip_suffix(".jsonl"))
        {
            logs.push(read(&dir.join(&*name), run, worker)?);
        }
    }
    merge(logs)
}

/// The run's events in one order that depends on nothing but the events: by `sequence`, then
/// `worker_id`. Refused when the logs are of different runs or one (worker, sequence) repeats.
pub fn merge(logs: Vec<Vec<Event>>) -> Result<Vec<Event>> {
    let mut all: Vec<Event> = logs.into_iter().flatten().collect();
    if let Some(first) = all.first() {
        let run = first.run_id.clone();
        if let Some(e) = all.iter().find(|e| e.run_id != run) {
            return Err(Error::Invalid(format!(
                "runs {run:?} and {:?} in one merge",
                e.run_id
            )));
        }
    }
    all.sort_by(|a, b| {
        a.sequence
            .cmp(&b.sequence)
            .then_with(|| a.worker_id.cmp(&b.worker_id))
    });
    if let Some(w) = all
        .windows(2)
        .find(|w| w[0].sequence == w[1].sequence && w[0].worker_id == w[1].worker_id)
    {
        return Err(Error::Invalid(format!(
            "worker {:?} logs sequence {} twice",
            w[0].worker_id, w[0].sequence
        )));
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("wal-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn appends_chain_and_reopen() {
        let d = dir("chain");
        let mut w = Wal::open(&d, "run-1", "lead").unwrap();
        let a = w
            .append("round", &json!({"round": 1, "loss": 1.25}))
            .unwrap();
        let b = w
            .append("round", &json!({"round": 2, "loss": 1.1}))
            .unwrap();
        assert_eq!((a.sequence, b.sequence), (1, 2));
        assert_eq!(a.previous_event_hash, None);
        assert_eq!(b.previous_event_hash.as_deref(), Some(a.event_id.as_str()));
        assert!(a.event_id.starts_with("sha256:") && a.event_id != b.event_id);
        assert!(
            Wal::open(&d, "run-1", "lead").is_err(),
            "a second writer is refused"
        );
        drop(w);
        let mut w = Wal::open(&d, "run-1", "lead").unwrap();
        assert_eq!(w.sequence(), 2);
        let c = w.append("checkpoint", &json!({"round": 2})).unwrap();
        assert_eq!(c.previous_event_hash.as_deref(), Some(b.event_id.as_str()));
        assert_eq!(read(w.path(), "run-1", "lead").unwrap(), vec![a, b, c]);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn another_run_or_worker_is_refused() {
        let d = dir("ids");
        let mut w = Wal::open(&d, "run-1", "lead").unwrap();
        w.append("round", &json!({"round": 1})).unwrap();
        drop(w);
        assert!(Wal::open(&d, "run-2", "lead").is_err());
        std::fs::rename(d.join("wal-lead.jsonl"), d.join("wal-other.jsonl")).unwrap();
        assert!(Wal::open(&d, "run-1", "other").is_err());
        assert!(Wal::open(&d, "../x", "lead").is_err());
        assert!(Wal::open(&d, "run-1", "a/b").is_err());
        assert!(w_append_nan(&d).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    fn w_append_nan(d: &Path) -> Result<Event> {
        let mut w = Wal::open(d, "run-3", "lead")?;
        w.append("round", &vec![f64::NAN])
    }

    /// Every single-byte change anywhere in the log refuses it.
    #[test]
    fn every_byte_flip_is_refused() {
        let d = dir("flip");
        let mut w = Wal::open(&d, "run-1", "lead").unwrap();
        for r in 0..3 {
            w.append(
                "round",
                &json!({"round": r, "loss": 0.5 + r as f64, "note": "é"}),
            )
            .unwrap();
        }
        drop(w);
        let path = d.join("wal-lead.jsonl");
        let good = std::fs::read(&path).unwrap();
        assert_eq!(read(&path, "run-1", "lead").unwrap().len(), 3);
        for i in 0..good.len() {
            for flip in [0x01u8, 0x20, 0x80] {
                let mut bad = good.clone();
                bad[i] ^= flip;
                std::fs::write(&path, &bad).unwrap();
                assert!(
                    read(&path, "run-1", "lead").is_err(),
                    "byte {i} ^ {flip:#x} accepted"
                );
            }
        }
        std::fs::write(&path, &good[..good.len() - 1]).unwrap();
        assert!(read(&path, "run-1", "lead").is_err(), "torn tail accepted");
        let lines: Vec<&[u8]> = good
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .collect();
        let swapped = [lines[1], lines[0], lines[2]].join(&b'\n');
        std::fs::write(&path, [swapped, b"\n".to_vec()].concat()).unwrap();
        assert!(
            read(&path, "run-1", "lead").is_err(),
            "reordered lines accepted"
        );
        let dropped = [lines[0], lines[2]].join(&b'\n');
        std::fs::write(&path, [dropped, b"\n".to_vec()].concat()).unwrap();
        assert!(
            read(&path, "run-1", "lead").is_err(),
            "a dropped middle line accepted"
        );
        std::fs::write(&path, &good).unwrap();
        assert!(read(&path, "run-1", "lead").is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Merging orders by (sequence, worker) whatever order the logs arrive in.
    #[test]
    fn merge_order() {
        let d = dir("merge");
        let mut logs = Vec::new();
        for (worker, n) in [("w2", 2), ("lead", 3), ("w1", 1)] {
            let mut w = Wal::open(&d, "run-1", worker).unwrap();
            for s in 0..n {
                w.append("round", &json!({"worker": worker, "step": s}))
                    .unwrap();
            }
            logs.push(read(w.path(), "run-1", worker).unwrap());
        }
        let merged = merge(logs.clone()).unwrap();
        let order: Vec<(u64, &str)> = merged
            .iter()
            .map(|e| (e.sequence, e.worker_id.as_str()))
            .collect();
        assert_eq!(
            order,
            [
                (1, "lead"),
                (1, "w1"),
                (1, "w2"),
                (2, "lead"),
                (2, "w2"),
                (3, "lead")
            ]
        );
        let mut reversed = logs.clone();
        reversed.reverse();
        assert_eq!(merge(reversed).unwrap(), merged);
        let mut shuffled: Vec<Vec<Event>> = logs
            .iter()
            .map(|l| l.iter().rev().cloned().collect())
            .collect();
        shuffled.rotate_left(1);
        assert_eq!(merge(shuffled).unwrap(), merged);
        assert_eq!(gather(&d, "run-1").unwrap(), merged);
        let mut twice = logs.clone();
        twice.push(logs[0].clone());
        assert!(merge(twice).is_err());
        let mut other = logs[0][0].clone();
        other.run_id = "run-2".into();
        assert!(merge(vec![logs[1].clone(), vec![other]]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
