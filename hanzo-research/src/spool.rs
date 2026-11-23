//! Idempotent spooler: drains a WAL and posts unsent events to `/v1/research` in order.
//!
//! Each event is sent exactly once — an Idempotency-Key header equals the event's digest.
//! A cursor file (`<wal>.cursor`) records the last sent sequence plus its digest so restarts
//! resume cleanly without re-sending.

use crate::lineage::ResearchEvent;
use crate::{Api, Research};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// Cursor persisted after each successful batch: the last sent event's sequence and its digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpoolCursor {
    pub sequence: u64,
    pub digest: String,
}

/// Spooler drains a WAL file and posts unsent events to the research API.
#[derive(Debug)]
pub struct Spooler {
    wal_path: PathBuf,
    cursor_path: PathBuf,
}

impl Spooler {
    /// Create a spooler for the given WAL. The cursor lives at `<wal>.cursor`.
    pub fn new(wal_path: &Path) -> Self {
        let cursor_path = wal_path.with_extension("cursor");
        Spooler {
            wal_path: wal_path.to_path_buf(),
            cursor_path,
        }
    }

    /// Load the persisted cursor, if any.
    fn load_cursor(&self) -> std::io::Result<Option<SpoolCursor>> {
        if !self.cursor_path.exists() {
            return Ok(None);
        }
        let file = File::open(&self.cursor_path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(cursor) = serde_json::from_str::<SpoolCursor>(&line) {
                return Ok(Some(cursor));
            }
        }
        Ok(None)
    }

    /// Persist the cursor atomically (write to temp, fsync, rename).
    fn save_cursor(&self, cursor: &SpoolCursor) -> std::io::Result<()> {
        let tmp_path = self.cursor_path.with_extension("cursor.tmp");
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)?;
        let line = serde_json::to_string(cursor).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        writeln!(file, "{}", line)?;
        file.sync_data()?;
        std::fs::rename(&tmp_path, &self.cursor_path)?;
        Ok(())
    }

    /// Drain the WAL, posting unsent events in batches through the Research client.
    /// Returns the number of events sent.
    pub fn drain(&self, research: &Research) -> std::io::Result<usize> {
        // Load the cursor to know where to resume
        let cursor = self.load_cursor()?;
        let start_sequence = cursor.map(|c| c.sequence).unwrap_or(0);

        // Read all events from the WAL
        let file = File::open(&self.wal_path)?;
        let reader = BufReader::new(file);

        let mut events_to_send = Vec::new();
        let mut last_event_hash = "0".repeat(64);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let sequence = i as u64 + 1;
            if sequence <= start_sequence {
                // Already sent, but verify chain integrity
                let event: ResearchEvent = serde_json::from_str(&line).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
                if event.sequence != sequence {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "WAL line {} has sequence {}, expected {}",
                            sequence, event.sequence, sequence
                        ),
                    ));
                }
                if event.previous_event_hash != last_event_hash {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "WAL line {} does not chain to previous event",
                            sequence
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
                        format!("WAL line {} payload hash mismatch", sequence),
                    ));
                }
                last_event_hash = event.digest().map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                })?;
                continue;
            }

            // This event hasn't been sent yet
            let event: ResearchEvent = serde_json::from_str(&line).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;

            // Verify chain integrity for unsent events too
            if event.sequence != sequence {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "WAL line {} has sequence {}, expected {}",
                        sequence, event.sequence, sequence
                    ),
                ));
            }
            if event.previous_event_hash != last_event_hash {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "WAL line {} does not chain to previous event",
                        sequence
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
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("WAL line {} payload hash mismatch", sequence),
                ));
            }

            events_to_send.push(event.clone());
            last_event_hash = event.digest().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;
        }

        // Send events one by one through the Research client's API
        // Using the evidence endpoint with the event as JSON payload
        // The Idempotency-Key is the event digest
        let mut sent_count = 0;
        for event in events_to_send {
            let event_digest = event.digest().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;

            // Build a simple evidence record from the WAL event
            // We'll post to /v1/research/evidence with the event as content
            let evidence_payload = serde_json::json!({
                "uri": format!("wal://{}/{}/{}", event.run_id, event.worker_id, event.sequence),
                "kind": "wal-event",
                "source": { "system": "wal-spooler" },
                "content": {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "worker_id": event.worker_id,
                    "sequence": event.sequence,
                    "event_type": event.event_type().unwrap_or_else(|| "unknown".into()),
                    "payload": event.payload,
                    "created_at": event.created_at,
                    "previous_event_hash": event.previous_event_hash,
                    "payload_hash": event.payload_hash,
                    "digest": event_digest.clone()
                }
            });

            let api = Api::new(&research.base, &research.project, research.key.clone())
                .map_err(std::io::Error::other)?;

            // Post with Idempotency-Key header
            let url = format!("{}/v1/research/evidence", api.base());
            let body = serde_json::to_vec(&evidence_payload).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?;

            let req = api
                .agent_for_spooler()
                .post(&url)
                .set("Content-Type", "application/json")
                .set("Authorization", &format!("Bearer {}", api.token()))
                .set("X-Project-Id", &api.project())
                .set("Idempotency-Key", &event_digest);

            let resp = req.send(&*body);
            match resp {
                Ok(response) => {
                    let _status = response.status();
                    // Read and discard the response body
                    let mut resp_reader = response.into_reader();
                    let mut _buf = Vec::new();
                    resp_reader.read_to_end(&mut _buf).ok();

                    // Update cursor after successful send
                    let new_cursor = SpoolCursor {
                        sequence: event.sequence,
                        digest: event_digest,
                    };
                    self.save_cursor(&new_cursor)?;
                    sent_count += 1;
                }
                Err(e) => {
                    // Network error - stop cleanly, cursor not updated
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionRefused,
                        format!("Failed to send event {}: {}", event.sequence, e),
                    ));
                }
            }
        }

        Ok(sent_count)
    }
}

/// Extension trait to get event_type from ResearchEvent
trait EventTypeExt {
    fn event_type(&self) -> Option<String>;
}

impl EventTypeExt for ResearchEvent {
    fn event_type(&self) -> Option<String> {
        // Extract event type from event_id (format: run-worker-seq-type)
        self.event_id
            .rsplit_once('-')
            .map(|(_, t)| t.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ResearchWal;
    use serde_json::json;
    use std::io::Write;
    use std::net::{TcpListener};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_spool_cursor_serialization() {
        let cursor = SpoolCursor {
            sequence: 42,
            digest: "abc123".to_string(),
        };
        let json = serde_json::to_string(&cursor).unwrap();
        let parsed: SpoolCursor = serde_json::from_str(&json).unwrap();
        assert_eq!(cursor, parsed);
    }

    #[test]
    fn test_spooler_drain_at_least_once_delivery() {
        // Create a fake HTTP server that tracks what it receives
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = received.clone();

        thread::spawn(move || {
            for stream in listener.incoming() {
                let stream = stream.unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request = Vec::new();
                loop {
                    let mut buf = [0u8; 1024];
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            request.extend_from_slice(&buf[..n]);
                            if request.windows(4).any(|w| w == b"\r\n\r\n") {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                let request_str = String::from_utf8_lossy(&request);
                let idempotency_key = request_str
                    .lines()
                    .find(|l| l.to_lowercase().starts_with("idempotency-key:"))
                    .map(|l| l.split(':').nth(1).unwrap_or("").trim())
                    .unwrap_or("")
                    .to_string();

                if !idempotency_key.is_empty() {
                    received_clone.lock().unwrap().push(idempotency_key);
                }

                // Send 200 OK
                let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let mut stream = stream.try_clone().unwrap();
                stream.write_all(response).ok();
                stream.flush().ok();
            }
        });

        // Create a WAL with some events
        let temp_dir = std::env::temp_dir().join(format!("spool-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let mut wal = ResearchWal::open(&temp_dir, "test-run").unwrap();
        for i in 1..=5 {
            wal.append("step", &json!({"step": i, "loss": 1.0 / i as f64}))
                .unwrap();
        }

        let wal_path = wal.wal_path().to_path_buf();
        drop(wal);

        // Create spooler and drain
        let spooler = Spooler::new(&wal_path);
        let research = Research::new(
            &format!("http://{}", addr),
            "test-key",
            "test-project",
            ".",
            &[],
        );

        let sent = spooler.drain(&research).unwrap();
        assert_eq!(sent, 5);

        // Verify all events were received (at least once)
        thread::sleep(Duration::from_millis(100));
        let received_keys = received.lock().unwrap();
        assert_eq!(received_keys.len(), 5);

        // Restart spooler - should not re-send (cursor persists)
        let spooler2 = Spooler::new(&wal_path);
        let sent2 = spooler2.drain(&research).unwrap();
        assert_eq!(sent2, 0); // No new events

        thread::sleep(Duration::from_millis(100));
        let received_keys2 = received.lock().unwrap();
        assert_eq!(received_keys2.len(), 5); // Still 5, no re-send

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
