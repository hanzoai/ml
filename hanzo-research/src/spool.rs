//! Durable WAL-to-cloud spooler with fsynced cursor and Idempotency-Key deduplication.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::wal::{self, GENESIS};
use crate::wire::{now, Evidence, Source};
use crate::{Error, Research, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpoolCursor {
    pub run_id: String,
    pub committed_seq: u64,
    pub last_digest: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpoolReport {
    pub drained: usize,
    pub committed_seq: u64,
    pub pending: usize,
    pub offline: bool,
    pub error: String,
}

fn write_cursor_fsync(path: &Path, cursor: &SpoolCursor) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut raw = serde_json::to_vec_pretty(cursor)?;
    raw.push(b'\n');
    let mut fp = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp)?;
    fp.write_all(&raw)?;
    fp.sync_data()?;
    fs::rename(tmp, path)?;
    Ok(())
}

pub struct Spooler;

impl Spooler {
    pub fn load_cursor(cursor_path: impl AsRef<Path>) -> Result<SpoolCursor> {
        let p = cursor_path.as_ref();
        if !p.is_file() {
            return Ok(SpoolCursor {
                run_id: String::new(),
                committed_seq: 0,
                last_digest: GENESIS.to_string(),
                updated_at: now(),
            });
        }
        Ok(serde_json::from_slice(&fs::read(p)?)?)
    }

    pub fn drain(
        client: &Research,
        wal_path: impl AsRef<Path>,
        cursor_path: impl AsRef<Path>,
        max_retries: u32,
        base_backoff_ms: u64,
    ) -> Result<SpoolReport> {
        let wpath = wal_path.as_ref();
        let cpath = cursor_path.as_ref();
        wal::verify(wpath)?;
        let events = wal::read(wpath)?;
        let mut cursor = Self::load_cursor(cpath)?;
        let uncommitted: Vec<_> = events
            .into_iter()
            .filter(|e| e.sequence > cursor.committed_seq)
            .collect();
        if uncommitted.is_empty() {
            return Ok(SpoolReport {
                drained: 0,
                committed_seq: cursor.committed_seq,
                pending: 0,
                offline: false,
                error: String::new(),
            });
        }

        let mut drained = 0usize;
        let total = uncommitted.len();
        for (idx, ev) in uncommitted.into_iter().enumerate() {
            let ev_digest = ev.digest().map_err(Error::Json)?;
            let record = Evidence {
                uri: format!("wal://{}/{}", ev.run_id, ev.sequence),
                hash: ev_digest.clone(),
                kind: "element".into(),
                source: Source {
                    system: "hanzo-research-wal".into(),
                    revision: ev.worker_id.clone(),
                },
                content: Some(serde_json::to_value(&ev)?),
                ..Evidence::default()
            };
            let mut attempt = 0u32;
            loop {
                match client.evidence(&record, Some(&ev_digest)) {
                    Ok(_) => break,
                    Err(err) => {
                        attempt += 1;
                        if attempt > max_retries {
                            return Ok(SpoolReport {
                                drained,
                                committed_seq: cursor.committed_seq,
                                pending: total - idx,
                                offline: true,
                                error: err.to_string(),
                            });
                        }
                        if base_backoff_ms > 0 {
                            let mult = 1u64 << (attempt - 1).min(10);
                            thread::sleep(Duration::from_millis(base_backoff_ms * mult));
                        }
                    }
                }
            }
            cursor = SpoolCursor {
                run_id: ev.run_id,
                committed_seq: ev.sequence,
                last_digest: ev_digest,
                updated_at: now(),
            };
            write_cursor_fsync(cpath, &cursor)?;
            drained += 1;
        }

        Ok(SpoolReport {
            drained,
            committed_seq: cursor.committed_seq,
            pending: 0,
            offline: false,
            error: String::new(),
        })
    }
}
