//! Write-Ahead Log (WAL) for durable, tamper-evident research telemetry.
//!
//! Enforces WAL-first logging:
//! 1. Local append with monotonic sequence number and SHA-256 hash chaining.
//! 2. Immediate `sync_data()` (fsync) before training proceeds.
//! 3. Background/idempotent spooling to `/v1/research`.

use crate::lineage::ResearchEvent;
use crate::wire::now;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

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
    pub fn open_worker(run_dir: &Path, run_id: &str, worker_id: &str) -> std::io::Result<Self> {
        std::fs::create_dir_all(run_dir)?;
        let wal_path = if worker_id == "worker-0" || worker_id.is_empty() {
            run_dir.join("wal.jsonl")
        } else {
            run_dir.join(format!("wal-{worker_id}.jsonl"))
        };

        let mut sequence = 0u64;
        let mut last_event_hash = "0000000000000000000000000000000000000000000000000000000000000000".to_string();

        if wal_path.exists() {
            let file = File::open(&wal_path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(event) = serde_json::from_str::<ResearchEvent>(&line) {
                    sequence = event.sequence;
                    if let Ok(hash) = event.digest() {
                        last_event_hash = hash;
                    }
                }
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&wal_path)?;

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
    pub fn append<T: Serialize>(&mut self, event_type: &str, payload: &T) -> std::io::Result<ResearchEvent> {
        let json_val = serde_json::to_value(payload)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        self.sequence += 1;
        let event_id = format!("{}-{}-{}-{}", self.run_id, self.worker_id, self.sequence, event_type);
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

        writeln!(self.file, "{line}")?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_append_and_recover() {
        let temp = std::env::temp_dir().join(format!("wal-test-{}", std::process::id()));
        let mut wal = ResearchWal::open(&temp, "run-test").unwrap();

        let payload1 = serde_json::json!({"step": 1, "loss": 1.25});
        let e1 = wal.append("step", &payload1).unwrap();
        assert_eq!(e1.sequence, 1);
        assert_eq!(e1.previous_event_hash, "0000000000000000000000000000000000000000000000000000000000000000");

        let payload2 = serde_json::json!({"step": 2, "loss": 1.10});
        let e2 = wal.append("step", &payload2).unwrap();
        assert_eq!(e2.sequence, 2);
        assert_eq!(e2.previous_event_hash, e1.digest().unwrap());

        // Reopen to verify recovery
        let wal_recovered = ResearchWal::open(&temp, "run-test").unwrap();
        assert_eq!(wal_recovered.sequence(), 2);
        assert_eq!(wal_recovered.last_event_hash(), e2.digest().unwrap());

        let _ = std::fs::remove_dir_all(&temp);
    }
}
