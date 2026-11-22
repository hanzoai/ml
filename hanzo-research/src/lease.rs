//! Cross-machine cloud lease manager with monotonic fencing tokens over `/v1/research/evidence`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::canonical::rfc8785_digest;
use crate::provenance;
use crate::wal::ResearchWal;
use crate::wire::{now, Evidence, Source};
use crate::{Error, Research, Result};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Lease {
    pub uri: String,
    pub campaign: String,
    pub opportunity_id: String,
    pub fencing_token: i64,
    pub owner_worker_id: String,
    pub host: String,
    pub acquired_at_unix: i64,
    pub expires_at_unix: i64,
    pub reclaimed_from: String,
    pub offline_local: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LeaseOutcome {
    pub status: String,
    pub lease: Option<Lease>,
    pub holder_worker_id: String,
    pub expires_at_unix: i64,
    pub fencing_token: i64,
}

fn sanitize_segment(seg: &str) -> String {
    seg.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn latest_version(rows: &[Evidence]) -> Option<&Evidence> {
    rows.iter().max_by_key(|e| e.version)
}

fn default_clock() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

pub struct CloudLeaseManager {
    pub client: Option<Research>,
    pub worker_id: String,
    pub commit: String,
    pub host: String,
    pub clock_skew_grace_sec: i64,
    pub local_dir: Option<PathBuf>,
    clock_override: Option<i64>,
}

impl CloudLeaseManager {
    pub fn new(
        client: Option<Research>,
        worker_id: &str,
        commit: &str,
        local_dir: Option<&Path>,
    ) -> Self {
        let host = provenance::host().hostname;
        if let Some(dir) = local_dir {
            let _ = fs::create_dir_all(dir.join("leases"));
        }
        Self {
            client,
            worker_id: worker_id.to_string(),
            commit: commit.to_string(),
            host,
            clock_skew_grace_sec: 5,
            local_dir: local_dir.map(Path::to_path_buf),
            clock_override: None,
        }
    }

    pub fn set_clock(&mut self, unix_ts: i64) {
        self.clock_override = Some(unix_ts);
    }

    pub fn lease_uri(campaign: &str, opportunity_id: &str) -> String {
        format!(
            "research://leases/{}/{}",
            sanitize_segment(campaign),
            sanitize_segment(opportunity_id)
        )
    }

    fn now_unix(&self) -> i64 {
        self.clock_override.unwrap_or_else(default_clock)
    }

    fn local_path(&self, uri: &str) -> Option<PathBuf> {
        let dir = self.local_dir.as_ref()?;
        let key = rfc8785_digest(&uri).unwrap_or_else(|_| sanitize_segment(uri));
        Some(dir.join("leases").join(format!("{}.json", &key[..24.min(key.len())])))
    }

    fn acquire_offline(
        &self,
        uri: &str,
        campaign: &str,
        opportunity_id: &str,
        ttl_sec: i64,
    ) -> Result<LeaseOutcome> {
        let now_u = self.now_unix();
        let mut prev_ver = 0i64;
        let mut reclaimed_from = String::new();
        let path = self.local_path(uri);
        if let Some(p) = &path {
            if p.is_file() {
                let cur: Value = serde_json::from_slice(&fs::read(p)?)?;
                prev_ver = cur["fencing_token"].as_i64().unwrap_or(0);
                let holder = cur["owner_worker_id"].as_str().unwrap_or("");
                let state = cur["state"].as_str().unwrap_or("");
                let exp_u = cur["expires_at_unix"].as_i64().unwrap_or(0);
                if state == "held"
                    && holder != self.worker_id
                    && now_u < exp_u + self.clock_skew_grace_sec
                {
                    return Ok(LeaseOutcome {
                        status: "busy".into(),
                        lease: None,
                        holder_worker_id: holder.into(),
                        expires_at_unix: exp_u,
                        fencing_token: prev_ver,
                    });
                }
                if state == "held" && holder != self.worker_id {
                    reclaimed_from = holder.into();
                }
            }
        }
        let fencing = prev_ver + 1;
        let lease = Lease {
            uri: uri.into(),
            campaign: campaign.into(),
            opportunity_id: opportunity_id.into(),
            fencing_token: fencing,
            owner_worker_id: self.worker_id.clone(),
            host: self.host.clone(),
            acquired_at_unix: now_u,
            expires_at_unix: now_u + ttl_sec,
            reclaimed_from,
            offline_local: true,
        };
        if let Some(p) = &path {
            let mut payload = serde_json::to_value(&lease)?;
            payload["state"] = json!("held");
            payload["updated_at"] = json!(now());
            let tmp = p.with_file_name(format!(
                "{}.tmp",
                p.file_name().unwrap_or_default().to_string_lossy()
            ));
            fs::write(&tmp, serde_json::to_vec_pretty(&payload)?)?;
            fs::rename(tmp, p)?;
        }
        if let Some(dir) = &self.local_dir {
            let mut wal = ResearchWal::open_worker(&dir.join("wal"), "lease-offline", &self.worker_id)?;
            wal.append("lease_acquired_offline", &lease)?;
        }
        Ok(LeaseOutcome {
            status: "acquired".into(),
            holder_worker_id: self.worker_id.clone(),
            expires_at_unix: lease.expires_at_unix,
            fencing_token: fencing,
            lease: Some(lease),
        })
    }

    pub fn acquire_lease(
        &self,
        campaign: &str,
        opportunity_id: &str,
        ttl_sec: i64,
    ) -> Result<LeaseOutcome> {
        let uri = Self::lease_uri(campaign, opportunity_id);
        let Some(client) = &self.client else {
            return self.acquire_offline(&uri, campaign, opportunity_id, ttl_sec);
        };
        let now_u = self.now_unix();
        let existing = match client.evidence_versions(&uri) {
            Ok(v) => v,
            Err(_) if self.local_dir.is_some() => {
                return self.acquire_offline(&uri, campaign, opportunity_id, ttl_sec);
            }
            Err(e) => return Err(e),
        };

        let mut prev_ver = 0i64;
        let mut reclaimed_from = String::new();
        if let Some(latest) = latest_version(&existing) {
            prev_ver = latest.version;
            if let Some(c) = &latest.content {
                let state = c["state"].as_str().unwrap_or("");
                let holder = c["owner_worker_id"].as_str().unwrap_or("");
                let exp_u = c["expires_at_unix"].as_i64().unwrap_or(0);
                if state == "held"
                    && holder != self.worker_id
                    && now_u < exp_u + self.clock_skew_grace_sec
                {
                    return Ok(LeaseOutcome {
                        status: "busy".into(),
                        lease: None,
                        holder_worker_id: holder.into(),
                        expires_at_unix: exp_u,
                        fencing_token: latest.version,
                    });
                }
                if state == "held"
                    && holder != self.worker_id
                    && now_u >= exp_u + self.clock_skew_grace_sec
                {
                    reclaimed_from = holder.into();
                }
            }
        }

        let content = json!({
            "campaign": campaign,
            "opportunity_id": opportunity_id,
            "owner_worker_id": self.worker_id,
            "host": self.host,
            "acquired_at_unix": now_u,
            "expires_at_unix": now_u + ttl_sec,
            "ttl_sec": ttl_sec,
            "expected_prev_version": prev_ver,
            "reclaimed_from": reclaimed_from,
            "state": "held",
        });
        let ev = Evidence {
            uri: uri.clone(),
            hash: rfc8785_digest(&content)?,
            kind: "element".into(),
            source: Source {
                system: "hanzo-research-lease".into(),
                revision: self.commit.clone(),
            },
            content: Some(content),
            ..Evidence::default()
        };
        client.evidence(&ev, None)?;

        let verified = client.evidence_versions(&uri)?;
        let winner = latest_version(&verified)
            .ok_or_else(|| Error::Message(format!("lease verification missing for {uri}")))?;
        let w_content = winner.content.as_ref().unwrap_or(&Value::Null);
        let w_holder = w_content["owner_worker_id"].as_str().unwrap_or("");
        let w_exp = w_content["expires_at_unix"].as_i64().unwrap_or(0);
        let w_state = w_content["state"].as_str().unwrap_or("");
        if w_holder != self.worker_id || w_state != "held" {
            return Ok(LeaseOutcome {
                status: "busy".into(),
                lease: None,
                holder_worker_id: w_holder.into(),
                expires_at_unix: w_exp,
                fencing_token: winner.version,
            });
        }

        let lease = Lease {
            uri,
            campaign: campaign.into(),
            opportunity_id: opportunity_id.into(),
            fencing_token: winner.version,
            owner_worker_id: self.worker_id.clone(),
            host: self.host.clone(),
            acquired_at_unix: now_u,
            expires_at_unix: w_exp,
            reclaimed_from: reclaimed_from.clone(),
            offline_local: false,
        };
        if !reclaimed_from.is_empty() {
            if let Some(dir) = &self.local_dir {
                let mut wal = ResearchWal::open_worker(&dir.join("wal"), campaign, &self.worker_id)?;
                wal.append("lease_reclaimed", &lease)?;
            }
        }
        Ok(LeaseOutcome {
            status: "acquired".into(),
            holder_worker_id: self.worker_id.clone(),
            expires_at_unix: w_exp,
            fencing_token: winner.version,
            lease: Some(lease),
        })
    }

    pub fn renew_lease(&self, lease: &Lease, ttl_sec: i64) -> Result<Lease> {
        let now_u = self.now_unix();
        if lease.offline_local || self.client.is_none() {
            let path = self.local_path(&lease.uri);
            if let Some(p) = &path {
                if p.is_file() {
                    let cur: Value = serde_json::from_slice(&fs::read(p)?)?;
                    if cur["owner_worker_id"].as_str() != Some(&self.worker_id)
                        || cur["fencing_token"].as_i64() != Some(lease.fencing_token)
                    {
                        return Err(Error::Message(format!(
                            "lease fenced offline on {}",
                            lease.uri
                        )));
                    }
                }
            }
            let renewed = Lease {
                fencing_token: lease.fencing_token + 1,
                expires_at_unix: now_u + ttl_sec,
                ..lease.clone()
            };
            if let Some(p) = &path {
                let mut payload = serde_json::to_value(&renewed)?;
                payload["state"] = json!("held");
                payload["updated_at"] = json!(now());
                let tmp = p.with_file_name(format!(
                    "{}.tmp",
                    p.file_name().unwrap_or_default().to_string_lossy()
                ));
                fs::write(&tmp, serde_json::to_vec_pretty(&payload)?)?;
                fs::rename(tmp, p)?;
            }
            return Ok(renewed);
        }

        let client = self.client.as_ref().unwrap();
        let rows = client.evidence_versions(&lease.uri)?;
        let latest = latest_version(&rows)
            .ok_or_else(|| Error::Message(format!("lease missing on {}", lease.uri)))?;
        let lc = latest.content.as_ref().unwrap_or(&Value::Null);
        if lc["owner_worker_id"].as_str() != Some(&self.worker_id)
            || latest.version != lease.fencing_token
            || lc["state"].as_str() != Some("held")
        {
            return Err(Error::Message(format!(
                "lease fenced on {}: expected token {}",
                lease.uri, lease.fencing_token
            )));
        }

        let content = json!({
            "campaign": lease.campaign,
            "opportunity_id": lease.opportunity_id,
            "owner_worker_id": self.worker_id,
            "host": self.host,
            "acquired_at_unix": lease.acquired_at_unix,
            "expires_at_unix": now_u + ttl_sec,
            "ttl_sec": ttl_sec,
            "expected_prev_version": lease.fencing_token,
            "reclaimed_from": lease.reclaimed_from,
            "state": "held",
        });
        client.evidence(
            &Evidence {
                uri: lease.uri.clone(),
                hash: rfc8785_digest(&content)?,
                kind: "element".into(),
                source: Source {
                    system: "hanzo-research-lease".into(),
                    revision: self.commit.clone(),
                },
                content: Some(content),
                ..Evidence::default()
            },
            None,
        )?;
        let after_rows = client.evidence_versions(&lease.uri)?;
        let after = latest_version(&after_rows)
            .ok_or_else(|| Error::Message(format!("lease missing after renew on {}", lease.uri)))?;
        let ac = after.content.as_ref().unwrap_or(&Value::Null);
        if ac["owner_worker_id"].as_str() != Some(&self.worker_id) {
            return Err(Error::Message(format!(
                "lease lost during renewal race on {}",
                lease.uri
            )));
        }
        Ok(Lease {
            fencing_token: after.version,
            expires_at_unix: now_u + ttl_sec,
            ..lease.clone()
        })
    }

    pub fn verify_fencing(&self, lease: &Lease) -> Result<()> {
        if lease.offline_local || self.client.is_none() {
            if let Some(p) = self.local_path(&lease.uri) {
                if p.is_file() {
                    let cur: Value = serde_json::from_slice(&fs::read(&p)?)?;
                    if cur["owner_worker_id"].as_str() != Some(&self.worker_id)
                        || cur["fencing_token"].as_i64() != Some(lease.fencing_token)
                        || cur["state"].as_str() != Some("held")
                    {
                        return Err(Error::Message(format!(
                            "offline fencing token superseded on {}",
                            lease.uri
                        )));
                    }
                }
            }
            return Ok(());
        }
        let client = self.client.as_ref().unwrap();
        let rows = client.evidence_versions(&lease.uri)?;
        let latest = latest_version(&rows)
            .ok_or_else(|| Error::Message(format!("lease missing on {}", lease.uri)))?;
        let lc = latest.content.as_ref().unwrap_or(&Value::Null);
        if lc["owner_worker_id"].as_str() != Some(&self.worker_id)
            || latest.version != lease.fencing_token
            || lc["state"].as_str() != Some("held")
        {
            return Err(Error::Message(format!(
                "fencing token {} superseded on {}",
                lease.fencing_token, lease.uri
            )));
        }
        Ok(())
    }

    pub fn reclaim_stale(
        &self,
        campaign: &str,
        opportunity_id: &str,
        ttl_sec: i64,
    ) -> Result<LeaseOutcome> {
        self.acquire_lease(campaign, opportunity_id, ttl_sec)
    }

    pub fn release_lease(&self, lease: &Lease, final_state: &str) -> Result<Option<Evidence>> {
        if final_state != "released" && final_state != "completed" {
            return Err(Error::Message(
                "final_state must be released or completed".into(),
            ));
        }
        let now_u = self.now_unix();
        if lease.offline_local || self.client.is_none() {
            if let Some(p) = self.local_path(&lease.uri) {
                if p.is_file() {
                    let mut cur: Value = serde_json::from_slice(&fs::read(&p)?)?;
                    if cur["owner_worker_id"].as_str() == Some(&self.worker_id) {
                        cur["state"] = json!(final_state);
                        cur["expires_at_unix"] = json!(now_u);
                        cur["fencing_token"] = json!(lease.fencing_token + 1);
                        let tmp = p.with_file_name(format!(
                            "{}.tmp",
                            p.file_name().unwrap_or_default().to_string_lossy()
                        ));
                        fs::write(&tmp, serde_json::to_vec_pretty(&cur)?)?;
                        fs::rename(tmp, p)?;
                    }
                }
            }
            return Ok(None);
        }
        let client = self.client.as_ref().unwrap();
        let rows = client.evidence_versions(&lease.uri)?;
        let Some(latest) = latest_version(&rows) else {
            return Ok(None);
        };
        let lc = latest.content.as_ref().unwrap_or(&Value::Null);
        if lc["owner_worker_id"].as_str() != Some(&self.worker_id) {
            return Ok(None);
        }
        let content = json!({
            "campaign": lease.campaign,
            "opportunity_id": lease.opportunity_id,
            "owner_worker_id": self.worker_id,
            "host": self.host,
            "acquired_at_unix": lease.acquired_at_unix,
            "expires_at_unix": now_u,
            "ttl_sec": 0,
            "expected_prev_version": latest.version,
            "reclaimed_from": lease.reclaimed_from,
            "state": final_state,
        });
        let filed = client.evidence(
            &Evidence {
                uri: lease.uri.clone(),
                hash: rfc8785_digest(&content)?,
                kind: "element".into(),
                source: Source {
                    system: "hanzo-research-lease".into(),
                    revision: self.commit.clone(),
                },
                content: Some(content),
                ..Evidence::default()
            },
            None,
        )?;
        Ok(Some(filed))
    }
}
