//! Write-Ahead Log (WAL) for durable, tamper-evident research telemetry.
//!
//! Enforces WAL-first logging:
//! 1. Local append with monotonic sequence number and SHA-256 hash chaining.
//! 2. Immediate `sync_data()` (fsync) before training proceeds.
//! 3. Background/idempotent spooling to `/v1/research`.
//!
//! Verification: `open_worker` and `verify` check that sequence = line number,
//! `previous_event_hash` chains correctly, and `payload_hash` = rfc8785_digest(payload).
//! A torn final line (not valid JSON, no trailing newline) is mended by truncation,
//! matching Python `hanzo_research.wal._mend`. Any other corruption returns an error.

use crate::canonical::rfc8785_digest;
use crate::lineage::ResearchEvent;
use crate::wire::now;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::fd::AsRawFd;

/// Advisory lock on the WAL file (flock on Unix, no-op elsewhere).
#[cfg(unix)]
fn lock_exclusive(file: &File) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    // LOCK_EX = 2, LOCK_NB = 4 (non-blocking). We block until acquired.
    let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
fn unlock(file: &File) -> std::io::Result<()> {
    let fd = file.as_raw_fd();
    let ret = unsafe { libc::flock(fd, libc::LOCK_UN) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &File) -> std::io::Result<()> {
    // No-op on non-Unix platforms (documented limitation)
    Ok(())
}

#[cfg(not(unix))]
fn unlock(_file: &File) -> std::io::Result<()> {
    Ok(())
}

/// Mend a torn final line in the WAL file, matching Python `_mend`.
/// Returns true if the file was modified.
fn mend_wal(path: &Path) -> std::io::Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let data = std::fs::read(path)?;
    if data.is_empty() || data.ends_with(b"\n") {
        return Ok(false);
    }
    // Find the last newline
    let last_newline = data.iter().rposition(|&b| b == b'\n').map(|i| i + 1).unwrap_or(0);
    let tail = &data[last_newline..];
    // Try to parse tail as JSON
    if serde_json::from_slice::<serde_json::Value>(tail).is_err() {
        // Torn line - truncate at last newline
        let mut file = OpenOptions::new().write(true).open(path)?;
        file.set_len(last_newline as u64)?;
        file.sync_data()?;
        return Ok(true);
    }
    // Valid JSON but missing newline - add it
    let mut file = OpenOptions::new().append(true).open(path)?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(true)
}

/// Verify a WAL file completely: sequence = line number, hash chain, payload hashes.
/// Returns the last event's digest on success.
pub fn verify(path: &Path) -> std::io::Result<String> {
    if !path.exists() {
        return Ok("0".repeat(64));
    }

    // First mend any torn tail
    mend_wal(path)?;

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut last_hash = "0".repeat(64);

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let line_num = i + 1;
        if line.trim().is_empty() {
            continue;
        }

        let event: ResearchEvent = serde_json::from_str(&line).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("WAL line {} is not valid JSON: {}", line_num, e),
            )
        })?;

        // Verify sequence = line number
        if event.sequence != line_num as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WAL line {} has sequence {}, expected {}",
                    line_num, event.sequence, line_num
                ),
            ));
        }

        // Verify hash chain
        if event.previous_event_hash != last_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WAL line {} does not chain to previous event (expected {}, got {})",
                    line_num, last_hash, event.previous_event_hash
                ),
            ));
        }

        // Verify payload hash = rfc8785_digest(payload)
        let recomputed = ResearchEvent::new(
            event.run_id.clone(),
            event.worker_id.clone(),
            event.event_id.clone(),
            event.sequence,
            last_hash.clone(),
            event.payload.clone(),
            event.created_at.clone(),
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if recomputed.payload_hash != event.payload_hash {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WAL line {} payload hash mismatch (expected {}, got {})",
                    line_num, recomputed.payload_hash, event.payload_hash
                ),
            ));
        }

        last_hash = event.digest().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
    }

    Ok(last_hash)
}

/// WAL manager for a single research run and worker.
#[derive(Debug)]
pub struct ResearchWal {
    run_id: String,
    worker_id: String,
    wal_path: PathBuf,
    file: File,
    sequence: u64,
    last_event_hash: String,
}

impl ResearchWal {
    /// Open or create a WAL file under `run_dir/wal.jsonl` for default worker `worker-0`.
    pub fn open(run_dir: &Path, run_id: &str) -> std::io::Result<Self> {
        Self::open_worker(run_dir, run_id, "worker-0")
    }

    /// Open or create a WAL file for a specific worker.
    /// Verifies the existing log: sequence = line number, previous_event_hash chains,
    /// payload_hash = rfc8785_digest(payload). A torn final line is mended.
    /// Any other corruption returns an error.
    pub fn open_worker(run_dir: &Path, run_id: &str, worker_id: &str) -> std::io::Result<Self> {
        std::fs::create_dir_all(run_dir)?;
        let wal_path = if worker_id == "worker-0" || worker_id.is_empty() {
            run_dir.join("wal.jsonl")
        } else {
            run_dir.join(format!("wal-{}.jsonl", worker_id))
        };

        // Mend any torn tail before reading
        mend_wal(&wal_path)?;

        let mut sequence = 0u64;
        let mut last_event_hash = "0".repeat(64);

        if wal_path.exists() {
            // Open read-only first to verify
            let file = File::open(&wal_path)?;
            let reader = BufReader::new(file);

            for (i, line) in reader.lines().enumerate() {
                let line = line?;
                let line_num = i + 1;
                if line.trim().is_empty() {
                    continue;
                }

                let event: ResearchEvent = serde_json::from_str(&line).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("WAL line {} is not valid JSON: {}", line_num, e),
                    )
                })?;

                // Verify sequence = line number
                if event.sequence != line_num as u64 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "WAL line {} has sequence {}, expected {}",
                            line_num, event.sequence, line_num
                        ),
                    ));
                }

                // Verify hash chain
                if event.previous_event_hash != last_event_hash {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "WAL line {} does not chain to previous event",
                            line_num
                        ),
                    ));
                }

                // Verify payload hash
                let recomputed = ResearchEvent::new(
                    event.run_id.clone(),
                    event.worker_id.clone(),
                    event.event_id.clone(),
                    event.sequence,
                    last_event_hash.clone(),
                    event.payload.clone(),
                    event.created_at.clone(),
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                if recomputed.payload_hash != event.payload_hash {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("WAL line {} payload hash mismatch", line_num),
                    ));
                }

                sequence = event.sequence;
                last_event_hash = event.digest().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
            }
        }

        // Open for append with exclusive lock
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

        lock_exclusive(&file)?;

        // Re-read the tail sequence while holding the lock to handle concurrent writers
        if wal_path.exists() {
            let file_read = File::open(&wal_path)?;
            let reader = BufReader::new(file_read);
            sequence = 0;
            last_event_hash = "0".repeat(64);

            for (i, line) in reader.lines().enumerate() {
                let line = line?;
                let line_num = i + 1;
                if line.trim().is_empty() {
                    continue;
                }

                let event: ResearchEvent = serde_json::from_str(&line).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("WAL line {} is not valid JSON: {}", line_num, e),
                    )
                })?;

                if event.sequence != line_num as u64 {
                    unlock(&file)?;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "WAL line {} has sequence {}, expected {}",
                            line_num, event.sequence, line_num
                        ),
                    ));
                }

                if event.previous_event_hash != last_event_hash {
                    unlock(&file)?;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "WAL line {} does not chain to previous event",
                            line_num
                        ),
                    ));
                }

                let recomputed = ResearchEvent::new(
                    event.run_id.clone(),
                    event.worker_id.clone(),
                    event.event_id.clone(),
                    event.sequence,
                    last_event_hash.clone(),
                    event.payload.clone(),
                    event.created_at.clone(),
                )
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                if recomputed.payload_hash != event.payload_hash {
                    unlock(&file)?;
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("WAL line {} payload hash mismatch", line_num),
                    ));
                }

                sequence = event.sequence;
                last_event_hash = event.digest().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
            }
        }

        Ok(ResearchWal {
            run_id: run_id.to_string(),
            worker_id: worker_id.to_string(),
            wal_path,
            file,
            sequence,
            last_event_hash,
        })
    }

    /// Append an event to the WAL, fsync to disk, and return the envelope.
    pub fn append<T: Serialize>(
        &mut self,
        event_type: &str,
        payload: &T,
    ) -> std::io::Result<ResearchEvent> {
        let json_val = serde_json::to_value(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.sequence += 1;
        let event_id = format!(
            "{}-{}-{}-{}",
            self.run_id, self.worker_id, self.sequence, event_type
        );
        let created_at = now();

        let event = ResearchEvent::new(
            self.run_id.clone(),
            self.worker_id.clone(),
            event_id,
            self.sequence,
            self.last_event_hash.clone(),
            json_val,
            created_at,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let event_hash = event
            .digest()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Format single-line JSON
        let line = serde_json::to_string(&event)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        writeln!(self.file, "{}", line)?;
        self.file.sync_data()?;

        self.last_event_hash = event_hash;
        Ok(event)
    }

    /// Idempotent append: if the event_id already exists with the identical payload,
    /// return the existing event without writing. If it exists with a different payload,
    /// return an error. Default event_id is rfc8785_digest of {run_id, worker_id, event_type, payload}.
    pub fn append_idempotent<T: Serialize>(
        &mut self,
        event_id: Option<&str>,
        event_type: &str,
        payload: &T,
    ) -> std::io::Result<ResearchEvent> {
        let json_val = serde_json::to_value(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Compute default event_id if not provided: rfc8785_digest of {run_id, worker_id, event_type, payload}
        let id = if let Some(id) = event_id {
            id.to_string()
        } else {
            let id_payload = serde_json::json!({
                "run_id": self.run_id,
                "worker_id": self.worker_id,
                "event_type": event_type,
                "payload": json_val
            });
            rfc8785_digest(&id_payload).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?
        };

        // Check if this event_id already exists in the WAL
        if self.wal_path.exists() {
            let file = File::open(&self.wal_path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(existing) = serde_json::from_str::<ResearchEvent>(&line) {
                    if existing.event_id == id {
                        // Event ID exists - check if payload matches
                        if existing.payload == json_val {
                            // Identical payload - return existing event
                            return Ok(existing);
                        } else {
                            // Different payload - error
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::AlreadyExists,
                                format!(
                                    "Event ID {} already exists with different payload",
                                    id
                                ),
                            ));
                        }
                    }
                }
            }
        }

        // Not found - append normally
        self.sequence += 1;
        let created_at = now();

        let event = ResearchEvent::new(
            self.run_id.clone(),
            self.worker_id.clone(),
            id,
            self.sequence,
            self.last_event_hash.clone(),
            json_val,
            created_at,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let event_hash = event
            .digest()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let line = serde_json::to_string(&event)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        writeln!(self.file, "{}", line)?;
        self.file.sync_data()?;

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

impl Drop for ResearchWal {
    fn drop(&mut self) {
        let _ = unlock(&self.file);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::process::Command;

    #[test]
    fn test_wal_append_and_recover() {
        let temp = std::env::temp_dir().join(format!("wal-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();

        let payload1 = json!({"step": 1, "loss": 1.25});
        let e1 = wal.append("step", &payload1).unwrap();
        assert_eq!(e1.sequence, 1);
        assert_eq!(
            e1.previous_event_hash,
            "0000000000000000000000000000000000000000000000000000000000000000"
        );

        let payload2 = json!({"step": 2, "loss": 1.10});
        let e2 = wal.append("step", &payload2).unwrap();
        assert_eq!(e2.sequence, 2);
        assert_eq!(e2.previous_event_hash, e1.digest().unwrap());

        // Reopen to verify recovery
        let wal_recovered = ResearchWal::open(&temp, "run-test").unwrap();
        assert_eq!(wal_recovered.sequence(), 2);
        assert_eq!(wal_recovered.last_event_hash(), e2.digest().unwrap());

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_verify_valid_wal() {
        let temp = std::env::temp_dir().join(format!("wal-verify-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();
        wal.append("step", &json!({"step": 1})).unwrap();
        wal.append("step", &json!({"step": 2})).unwrap();
        drop(wal);

        let digest = verify(&temp.join("wal.jsonl")).unwrap();
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn test_verify_detects_tampered_sequence() {
        let temp = std::env::temp_dir().join(format!("wal-tamper-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();
        wal.append("step", &json!({"step": 1})).unwrap();
        wal.append("step", &json!({"step": 2})).unwrap();
        drop(wal);

        // Tamper: change sequence in line 2
        let content = fs::read_to_string(&temp.join("wal.jsonl")).unwrap();
        let tampered = content.replace("\"sequence\":2", "\"sequence\":99");
        fs::write(&temp.join("wal.jsonl"), tampered).unwrap();

        let err = verify(&temp.join("wal.jsonl")).unwrap_err();
        assert!(err.to_string().contains("sequence"));
    }

    #[test]
    fn test_verify_detects_broken_chain() {
        let temp = std::env::temp_dir().join(format!("wal-chain-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();
        wal.append("step", &json!({"step": 1})).unwrap();
        wal.append("step", &json!({"step": 2})).unwrap();
        drop(wal);

        // Tamper: break the hash chain
        let content = fs::read_to_string(&temp.join("wal.jsonl")).unwrap();
        let tampered = content.replace(
            "previous_event_hash\":\"0000000000000000000000000000000000000000000000000000000000000000",
            "previous_event_hash\":\"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );
        fs::write(&temp.join("wal.jsonl"), tampered).unwrap();

        let err = verify(&temp.join("wal.jsonl")).unwrap_err();
        assert!(err.to_string().contains("chain"));
    }

    #[test]
    fn test_verify_detects_payload_hash_mismatch() {
        let temp = std::env::temp_dir().join(format!("wal-payload-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();
        wal.append("step", &json!({"step": 1})).unwrap();
        drop(wal);

        // Tamper: change payload but not payload_hash
        let content = fs::read_to_string(&temp.join("wal.jsonl")).unwrap();
        let tampered = content.replace("\"step\":1", "\"step\":999");
        fs::write(&temp.join("wal.jsonl"), tampered).unwrap();

        let err = verify(&temp.join("wal.jsonl")).unwrap_err();
        assert!(err.to_string().contains("payload hash"));
    }

    #[test]
    fn test_verify_mends_torn_tail() {
        let temp = std::env::temp_dir().join(format!("wal-torn-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();
        wal.append("step", &json!({"step": 1})).unwrap();
        drop(wal);

        // Corrupt: truncate the last line (remove newline and partial JSON)
        let mut content = fs::read(&temp.join("wal.jsonl")).unwrap();
        // Remove last newline and a few chars to simulate torn write
        if let Some(pos) = content.iter().rposition(|&b| b == b'\n') {
            content.truncate(pos + 1); // Keep newline
            content.extend_from_slice(b"{\"incomplete");
            fs::write(&temp.join("wal.jsonl"), content).unwrap();

            // Verify should mend and succeed
            let digest = verify(&temp.join("wal.jsonl")).unwrap();
            assert_eq!(digest.len(), 64);

            // The torn line should be gone
            let fixed = fs::read_to_string(&temp.join("wal.jsonl")).unwrap();
            assert!(!fixed.contains("incomplete"));
        }
    }

    #[test]
    fn test_verify_rejects_corrupted_middle_line() {
        let temp = std::env::temp_dir().join(format!("wal-corrupt-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();
        wal.append("step", &json!({"step": 1})).unwrap();
        wal.append("step", &json!({"step": 2})).unwrap();
        drop(wal);

        // Corrupt a middle line (not valid JSON)
        let content = fs::read_to_string(&temp.join("wal.jsonl")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        let mut new_lines = lines.clone();
        // Corrupt line 1
        new_lines[0] = "not valid json at all";
        fs::write(&temp.join("wal.jsonl"), new_lines.join("\n") + "\n").unwrap();

        let err = verify(&temp.join("wal.jsonl")).unwrap_err();
        assert!(err.to_string().contains("valid JSON"));
    }

    #[test]
    fn test_concurrent_append_two_processes() {
        let temp = std::env::temp_dir().join(format!("wal-concurrent-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        // Spawn two threads that each append 500 events
        let run_dir = temp.to_string_lossy().to_string();
        let run_id = "run-concurrent";

        let handle1 = std::thread::spawn({
            let run_dir = run_dir.clone();
            let run_id = run_id.to_string();
            move || {
                let mut wal = ResearchWal::open_worker(
                    Path::new(&run_dir),
                    &run_id,
                    "worker-1",
                ).unwrap();
                for i in 1..=500 {
                    wal.append("step", &json!({"worker": 1, "step": i})).unwrap();
                }
            }
        });

        let handle2 = std::thread::spawn({
            let run_dir = run_dir.clone();
            let run_id = run_id.to_string();
            move || {
                let mut wal = ResearchWal::open_worker(
                    Path::new(&run_dir),
                    &run_id,
                    "worker-2",
                ).unwrap();
                for i in 1..=500 {
                    wal.append("step", &json!({"worker": 2, "step": i})).unwrap();
                }
            }
        });

        handle1.join().unwrap();
        handle2.join().unwrap();

        // Verify both WALs are valid
        let wal1_path = Path::new(&run_dir).join("wal-worker-1.jsonl");
        let wal2_path = Path::new(&run_dir).join("wal-worker-2.jsonl");

        let d1 = verify(&wal1_path).unwrap();
        let d2 = verify(&wal2_path).unwrap();
        assert_eq!(d1.len(), 64);
        assert_eq!(d2.len(), 64);

        // Each should have exactly 500 events
        let content1 = fs::read_to_string(&wal1_path).unwrap();
        let content2 = fs::read_to_string(&wal2_path).unwrap();
        assert_eq!(content1.lines().count(), 500);
        assert_eq!(content2.lines().count(), 500);

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_append_idempotent_same_payload() {
        let temp = std::env::temp_dir().join(format!("wal-idempotent-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();

        let payload = json!({"step": 1, "loss": 0.5});

        // First append
        let e1 = wal.append_idempotent(None, "step", &payload).unwrap();
        // Second append with same payload (uses default event_id)
        let e2 = wal.append_idempotent(None, "step", &payload).unwrap();

        // Should return the same event (same sequence, same digest)
        assert_eq!(e1.sequence, e2.sequence);
        assert_eq!(e1.digest().unwrap(), e2.digest().unwrap());

        // Only one line in WAL
        let content = fs::read_to_string(&temp.join("wal.jsonl")).unwrap();
        assert_eq!(content.lines().count(), 1);

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_append_idempotent_different_payload_error() {
        let temp = std::env::temp_dir().join(format!("wal-idempotent-err-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();

        let payload1 = json!({"step": 1, "loss": 0.5});
        let payload2 = json!({"step": 1, "loss": 0.6}); // Different!

        // First append
        wal.append_idempotent(Some("custom-id"), "step", &payload1).unwrap();
        // Second append with same event_id but different payload
        let err = wal.append_idempotent(Some("custom-id"), "step", &payload2).unwrap_err();

        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(err.to_string().contains("different payload"));

        let _ = fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_append_idempotent_explicit_id() {
        let temp = std::env::temp_dir().join(format!("wal-idempotent-explicit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp);
        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();

        let payload = json!({"step": 1, "loss": 0.5});

        // First append with explicit ID
        let e1 = wal.append_idempotent(Some("my-custom-id"), "step", &payload).unwrap();
        // Second append with same explicit ID
        let e2 = wal.append_idempotent(Some("my-custom-id"), "step", &payload).unwrap();

        assert_eq!(e1.sequence, e2.sequence);
        assert_eq!(e1.event_id, "my-custom-id");
        assert_eq!(e2.event_id, "my-custom-id");

        let _ = fs::remove_dir_all(&temp);
    }
}
