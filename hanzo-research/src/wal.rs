//! Write-Ahead Log (WAL) for durable, tamper-evident research telemetry.
//!
//! Enforces WAL-first logging:
//! 1. Verify full hash chain on open after mending only a torn final line.
//! 2. Advisory file lock (`<wal>.lock`) serializing multi-process writers.
//! 3. Immediate `sync_data()` (fsync) before execution proceeds.
//! 4. Idempotent append via RFC 8785 `event_id` deduplication.

use crate::canonical::rfc8785_digest;
use crate::lineage::ResearchEvent;
use crate::wire::now;
use serde::Serialize;
use serde_json::json;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

struct FileLock {
    file: File,
}

impl FileLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if rc != 0 {
                return Err(io::Error::last_os_error());
            }
        }
        Ok(FileLock { file })
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

/// Compute canonical RFC 8785 idempotent `event_id`, matching Python `wal.default_event_id`.
pub fn default_event_id<T: Serialize>(
    run_id: &str,
    worker_id: &str,
    event_type: &str,
    payload: &T,
) -> io::Result<String> {
    let payload_val =
        serde_json::to_value(payload).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let body = json!({
        "run_id": run_id,
        "worker_id": worker_id,
        "type": event_type,
        "payload": payload_val,
    });
    rfc8785_digest(&body).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Mend only a torn final line left by an abrupt crash. Interior lines are never modified.
pub fn mend(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(());
    }
    let mut data = Vec::new();
    File::open(path)?.read_to_end(&mut data)?;
    if data.is_empty() || data.ends_with(b"\n") {
        return Ok(());
    }
    let end = data
        .iter()
        .rposition(|&b| b == b'\n')
        .map_or(0, |idx| idx + 1);
    if serde_json::from_slice::<serde_json::Value>(&data[end..]).is_err() {
        let f = OpenOptions::new().write(true).open(path)?;
        f.set_len(end as u64)?;
        f.sync_data()?;
        return Ok(());
    }
    let mut f = OpenOptions::new().append(true).open(path)?;
    f.write_all(b"\n")?;
    f.sync_data()?;
    Ok(())
}

/// Read all events in file order; errors on any malformed line.
pub fn read(path: impl AsRef<Path>) -> io::Result<Vec<ResearchEvent>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<ResearchEvent>(&line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: line {} is not a valid ResearchEvent: {e}", path.display(), idx + 1),
            )
        })?;
        out.push(event);
    }
    Ok(out)
}

/// Verify sequence monotonicity, hash chain links, and RFC 8785 payload hashes.
pub fn verify(path: impl AsRef<Path>) -> io::Result<String> {
    let path = path.as_ref();
    let events = read(path)?;
    let mut last = GENESIS.to_string();
    for (i, ev) in events.iter().enumerate() {
        let want_seq = (i + 1) as u64;
        if ev.sequence != want_seq {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: line {} has sequence {}, expected {want_seq}", path.display(), i + 1, ev.sequence),
            ));
        }
        if ev.previous_event_hash != last {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: line {} previous_event_hash mismatch", path.display(), i + 1),
            ));
        }
        let recomputed_payload = rfc8785_digest(&ev.payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if recomputed_payload != ev.payload_hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: line {} payload_hash mismatch", path.display(), i + 1),
            ));
        }
        last = ev
            .digest()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    }
    Ok(last)
}

/// WAL manager for a research run and worker.
#[derive(Debug)]
pub struct ResearchWal {
    run_id: String,
    worker_id: String,
    wal_path: PathBuf,
    lock_path: PathBuf,
    sequence: u64,
    last_event_hash: String,
}

impl ResearchWal {
    /// Open or create a WAL file under `run_dir/wal.jsonl` for default worker `worker-0`.
    pub fn open(run_dir: &Path, run_id: &str) -> io::Result<Self> {
        Self::open_worker(run_dir, run_id, "worker-0")
    }

    /// Open or create a WAL file for a specific worker, mending torn tail and verifying chain.
    pub fn open_worker(run_dir: &Path, run_id: &str, worker_id: &str) -> io::Result<Self> {
        std::fs::create_dir_all(run_dir)?;
        let wal_path = if worker_id == "worker-0" || worker_id.is_empty() {
            run_dir.join("wal.jsonl")
        } else {
            run_dir.join(format!("wal-{worker_id}.jsonl"))
        };
        let lock_path = PathBuf::from(format!("{}.lock", wal_path.display()));
        let _guard = FileLock::acquire(&lock_path)?;

        mend(&wal_path)?;
        let last_event_hash = verify(&wal_path)?;
        let events = read(&wal_path)?;
        let sequence = events.last().map_or(0, |e| e.sequence);

        let _ = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        Ok(ResearchWal {
            run_id: run_id.to_string(),
            worker_id: worker_id.to_string(),
            wal_path,
            lock_path,
            sequence,
            last_event_hash,
        })
    }

    /// Append an event to the WAL under advisory flock, fsync to disk, and return the envelope.
    pub fn append<T: Serialize>(
        &mut self,
        event_type: &str,
        payload: &T,
    ) -> io::Result<ResearchEvent> {
        let _guard = FileLock::acquire(&self.lock_path)?;
        mend(&self.wal_path)?;
        let json_val = serde_json::to_value(payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let events = read(&self.wal_path)?;
        let (seq, prev) = match events.last() {
            Some(last) => {
                let h = last
                    .digest()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                (last.sequence + 1, h)
            }
            None => (1, GENESIS.to_string()),
        };

        let event_id = format!(
            "{}-{}-{}-{}",
            self.run_id, self.worker_id, seq, event_type
        );
        let event = ResearchEvent::new(
            self.run_id.clone(),
            self.worker_id.clone(),
            event_id,
            seq,
            prev,
            json_val,
            now(),
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let event_hash = event
            .digest()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let line = serde_json::to_string(&event)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)?;
        file.seek(SeekFrom::End(0))?;
        writeln!(file, "{line}")?;
        file.sync_data()?;

        self.sequence = seq;
        self.last_event_hash = event_hash;
        Ok(event)
    }

    /// Append idempotently by `event_id` (defaults to RFC 8785 digest of run, worker, type, payload).
    pub fn append_idempotent<T: Serialize>(
        &mut self,
        event_id: Option<&str>,
        event_type: &str,
        payload: &T,
    ) -> io::Result<ResearchEvent> {
        let _guard = FileLock::acquire(&self.lock_path)?;
        mend(&self.wal_path)?;
        let json_val = serde_json::to_value(payload)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let eid = match event_id.filter(|s| !s.is_empty()) {
            Some(s) => s.to_string(),
            None => default_event_id(&self.run_id, &self.worker_id, event_type, &json_val)?,
        };
        let want_payload_hash = rfc8785_digest(&json_val)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let events = read(&self.wal_path)?;
        for existing in &events {
            if existing.event_id == eid {
                if existing.payload_hash != want_payload_hash || existing.run_id != self.run_id {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("idempotent event_id collision with different payload: {eid}"),
                    ));
                }
                if let Some(last) = events.last() {
                    self.sequence = last.sequence;
                    self.last_event_hash = last
                        .digest()
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                }
                return Ok(existing.clone());
            }
        }

        let (seq, prev) = match events.last() {
            Some(last) => {
                let h = last
                    .digest()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                (last.sequence + 1, h)
            }
            None => (1, GENESIS.to_string()),
        };

        let event = ResearchEvent::new(
            self.run_id.clone(),
            self.worker_id.clone(),
            eid,
            seq,
            prev,
            json_val,
            now(),
        )
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let event_hash = event
            .digest()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let line = serde_json::to_string(&event)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.wal_path)?;
        file.seek(SeekFrom::End(0))?;
        writeln!(file, "{line}")?;
        file.sync_data()?;

        self.sequence = seq;
        self.last_event_hash = event_hash;
        Ok(event)
    }

    /// Merge events from multiple workers into a canonical run-level event ordering.
    pub fn canonical_merged_ordering(mut events: Vec<ResearchEvent>) -> Vec<ResearchEvent> {
        events.sort_by(|a, b| {
            a.sequence
                .cmp(&b.sequence)
                .then_with(|| a.created_at.cmp(&b.created_at))
                .then_with(|| a.worker_id.cmp(&b.worker_id))
                .then_with(|| a.payload_hash.cmp(&b.payload_hash))
        });
        events
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn last_event_hash(&self) -> &str {
        &self.last_event_hash
    }

    pub fn wal_path(&self) -> &Path {
        &self.wal_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_append_verify_mend_and_idempotent() {
        let temp = std::env::temp_dir().join(format!("wal-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp);
        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();

        let payload1 = serde_json::json!({"step": 1, "loss": 1.25});
        let e1 = wal.append("step", &payload1).unwrap();
        assert_eq!(e1.sequence, 1);
        assert_eq!(e1.previous_event_hash, GENESIS);

        let e2 = wal
            .append_idempotent(None, "ckpt", &serde_json::json!({"a": 1, "b": 2}))
            .unwrap();
        let e2_dup = wal
            .append_idempotent(None, "ckpt", &serde_json::json!({"b": 2, "a": 1}))
            .unwrap();
        assert_eq!(e2.sequence, 2);
        assert_eq!(e2_dup.sequence, 2);
        assert_eq!(e2.digest().unwrap(), e2_dup.digest().unwrap());

        // Torn tail mend check
        {
            let mut f = OpenOptions::new().append(true).open(wal.wal_path()).unwrap();
            f.write_all(b"{\"run_id\":\"torn").unwrap();
        }
        let wal_recovered = ResearchWal::open(&temp, "run-test").unwrap();
        assert_eq!(wal_recovered.sequence(), 2);

        // Interior corruption must fail open & verify
        let raw = std::fs::read_to_string(wal.wal_path()).unwrap();
        std::fs::write(wal.wal_path(), raw.replacen("\"loss\":1.25", "\"loss\":9.99", 1)).unwrap();
        assert!(ResearchWal::open(&temp, "run-test").is_err());

        let _ = std::fs::remove_dir_all(&temp);
    }
}
