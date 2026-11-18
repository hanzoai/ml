//! Content-addressed, block-deduplicated training checkpoints across workers and rounds
//! (AdaCheck, HIP-1349).
//!
//! DiLoCo checkpoints in [`super::coordinator`] persist `theta` (`f32[N]`), outer `momentum`
//! (`f32[N]`), outer `carry` (`f32[N]`), `merged` (`bool[limit]`), and every joined worker's
//! inner AdamW state (`f32[3 * N]` plus step `t`). Across successive checkpoints and
//! heterogeneous workers, three sources of byte-level redundancy arise:
//! 1. Straggler, paused, or dropped workers whose `3 * N` inner state does not advance
//!    between rounds,
//! 2. Zero-initialized or sparse chunks (`carry`, early `momentum`, untouched parameter slices),
//! 3. Shared initial or synchronized blocks across workers.
//!
//! [`AdaCheckStore`] splits flat `f32` buffers into aligned chunks, hashes their little-endian
//! IEEE-754 bytes with SHA-256 in parallel, stores each unique chunk once under `blobs/`, and
//! verifies every chunk's SHA-256 digest on restore so bit-identical DiLoCo resume is preserved.

use super::coordinator::{Inner, Start};
use super::net::Params;
use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Default chunk granularity: 16,384 `f32` elements (64 KiB).
pub const DEFAULT_CHUNK_ELEMS: usize = 16_384;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChunkedVectorManifest {
    pub len_elems: usize,
    pub chunks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerInnerManifest {
    pub t: u64,
    pub state: ChunkedVectorManifest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaCheckManifest {
    pub format_version: u32,
    pub chunk_elems: usize,
    pub round: u64,
    pub merged: Vec<u8>,
    pub theta: ChunkedVectorManifest,
    pub momentum: ChunkedVectorManifest,
    pub carry: ChunkedVectorManifest,
    pub inner: BTreeMap<String, WorkerInnerManifest>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdaCheckStats {
    pub logical_bytes: u64,
    pub unique_bytes_written: u64,
    pub reused_bytes_intra: u64,
    pub reused_bytes_temporal: u64,
    pub total_chunks: usize,
    pub unique_chunks_written: usize,
    pub save_ms: f64,
}

impl AdaCheckStats {
    /// Compression/deduplication factor of this checkpoint (`logical / newly_written`).
    pub fn incremental_ratio(&self) -> f64 {
        if self.unique_bytes_written == 0 {
            f64::INFINITY
        } else {
            self.logical_bytes as f64 / self.unique_bytes_written as f64
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdaCheckRestoreStats {
    pub logical_bytes: u64,
    pub unique_blobs_read: usize,
    pub restore_ms: f64,
}

pub struct AdaCheckStore {
    root: PathBuf,
    chunk_elems: usize,
}

fn validate_worker_name(name: &str) -> Result<()> {
    ensure!(!name.is_empty(), "worker name must not be empty");
    ensure!(
        !name.contains("..")
            && !name.bytes().any(|b| matches!(b, b'/' | b'\\' | 0)),
        "invalid worker name in checkpoint: {name:?}"
    );
    Ok(())
}

fn validate_digest_hex(hex: &str) -> Result<()> {
    ensure!(
        hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "invalid sha256 chunk digest: {hex:?}"
    );
    Ok(())
}

fn f32_slice_to_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

fn le_bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure!(bytes.len() % 4 == 0, "corrupt f32 chunk byte length {}", bytes.len());
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        out.push(f32::from_bits(bits));
    }
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("path lacks parent")?;
    fs::create_dir_all(parent)?;
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = parent.join(format!(
        ".tmp.{}.{}.{}",
        std::process::id(),
        seq,
        path.file_name().and_then(|n| n.to_str()).unwrap_or("blob")
    ));
    {
        let mut f = fs::File::create(&tmp)
            .with_context(|| format!("create temp file {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) if path.exists() => {
            let _ = fs::remove_file(&tmp);
            let _ = e;
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e).with_context(|| format!("atomic rename into {}", path.display()))
        }
    }
}

impl AdaCheckStore {
    pub fn new(root: impl Into<PathBuf>, chunk_elems: usize) -> Self {
        Self {
            root: root.into(),
            chunk_elems: chunk_elems.max(256),
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn blob_path(&self, digest: &str) -> PathBuf {
        self.root.join("blobs").join(&digest[..2]).join(format!("{digest}.bin"))
    }

    /// Total physical blob bytes currently stored under `root/blobs`.
    pub fn physical_blob_bytes(&self) -> Result<u64> {
        let blobs = self.root.join("blobs");
        if !blobs.exists() {
            return Ok(0);
        }
        let mut total = 0u64;
        for entry in fs::read_dir(&blobs)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                for sub in fs::read_dir(entry.path())? {
                    let sub = sub?;
                    let name = sub.file_name();
                    if let Some(s) = name.to_str() {
                        if s.ends_with(".bin") && !s.starts_with(".tmp.") {
                            total += sub.metadata()?.len();
                        }
                    }
                }
            }
        }
        Ok(total)
    }

    fn chunk_bytes(&self, raw: &[u8]) -> Vec<(String, Vec<u8>)> {
        let step = self.chunk_elems * 4;
        raw.par_chunks(step)
            .map(|slice| (sha256_hex(slice), slice.to_vec()))
            .collect()
    }

    /// Save a complete DiLoCo checkpoint (`theta`, outer optimizer state, and all workers'
    /// inner AdamW states) into `dir/adacheck_manifest.json` backed by deduplicated blobs.
    pub fn save_checkpoint(
        &self,
        dir: &Path,
        theta: &[f32],
        momentum: &[f32],
        carry: &[f32],
        merged: &[bool],
        round: u64,
        inner: &HashMap<String, Inner>,
    ) -> Result<AdaCheckStats> {
        let t0 = Instant::now();
        ensure!(
            theta.len() == momentum.len() && theta.len() == carry.len(),
            "theta, momentum, and carry lengths must match"
        );
        for (name, w) in inner {
            validate_worker_name(name)?;
            ensure!(
                w.data.len() == theta.len() * 12,
                "worker {name} inner byte length {} does not match 3 * {} f32s",
                w.data.len(),
                theta.len()
            );
        }

        let mut stats = AdaCheckStats::default();
        let mut seen_this_ckpt: HashSet<String> = HashSet::new();

        let mut ingest = |chunks: Vec<(String, Vec<u8>)>, len_elems: usize| -> Result<ChunkedVectorManifest> {
            let mut digests = Vec::with_capacity(chunks.len());
            for (digest, bytes) in chunks {
                let b_len = bytes.len() as u64;
                stats.logical_bytes += b_len;
                stats.total_chunks += 1;
                if !seen_this_ckpt.insert(digest.clone()) {
                    stats.reused_bytes_intra += b_len;
                } else {
                    let bp = self.blob_path(&digest);
                    if bp.exists() {
                        stats.reused_bytes_temporal += b_len;
                    } else {
                        atomic_write(&bp, &bytes)?;
                        stats.unique_bytes_written += b_len;
                        stats.unique_chunks_written += 1;
                    }
                }
                digests.push(digest);
            }
            Ok(ChunkedVectorManifest {
                len_elems,
                chunks: digests,
            })
        };

        let theta_m = ingest(self.chunk_bytes(&f32_slice_to_le_bytes(theta)), theta.len())?;
        let mom_m = ingest(self.chunk_bytes(&f32_slice_to_le_bytes(momentum)), momentum.len())?;
        let carry_m = ingest(self.chunk_bytes(&f32_slice_to_le_bytes(carry)), carry.len())?;

        let mut sorted_names: Vec<&String> = inner.keys().collect();
        sorted_names.sort();
        let mut inner_manifests = BTreeMap::new();
        for name in sorted_names {
            let w = &inner[name];
            let st_m = ingest(self.chunk_bytes(&w.data), w.data.len() / 4)?;
            inner_manifests.insert(
                name.clone(),
                WorkerInnerManifest {
                    t: w.t,
                    state: st_m,
                },
            );
        }

        let flags: Vec<u8> = merged.iter().map(|&b| u8::from(b)).collect();
        stats.logical_bytes += flags.len() as u64 + 8;

        let manifest = AdaCheckManifest {
            format_version: 1,
            chunk_elems: self.chunk_elems,
            round,
            merged: flags,
            theta: theta_m,
            momentum: mom_m,
            carry: carry_m,
            inner: inner_manifests,
        };

        fs::create_dir_all(dir)?;
        let json_bytes = serde_json::to_vec_pretty(&manifest)?;
        atomic_write(&dir.join("adacheck_manifest.json"), &json_bytes)?;
        stats.save_ms = t0.elapsed().as_secs_f64() * 1_000.0;
        Ok(stats)
    }

    /// Restore `Start` bit-for-bit from `dir/adacheck_manifest.json`, verifying every
    /// referenced blob's SHA-256 digest before reconstructing tensors.
    pub fn restore_checkpoint(
        &self,
        dir: &Path,
        params: &Params,
        limit: usize,
    ) -> Result<(Start, AdaCheckRestoreStats)> {
        let t0 = Instant::now();
        let raw_manifest = fs::read(dir.join("adacheck_manifest.json"))
            .with_context(|| format!("read {}/adacheck_manifest.json", dir.display()))?;
        let manifest: AdaCheckManifest = serde_json::from_slice(&raw_manifest)?;
        ensure!(
            manifest.format_version == 1,
            "unsupported adacheck format version {}",
            manifest.format_version
        );
        ensure!(
            manifest.theta.len_elems == params.len
                && manifest.momentum.len_elems == params.len
                && manifest.carry.len_elems == params.len,
            "checkpoint parameter length {} does not match model length {}",
            manifest.theta.len_elems,
            params.len
        );
        ensure!(
            manifest.merged.len() <= limit,
            "checkpoint covers {} batches, run limit is {limit}",
            manifest.merged.len()
        );

        let mut blob_cache: HashMap<String, Vec<u8>> = HashMap::new();
        let mut load_bytes = |vec_m: &ChunkedVectorManifest| -> Result<Vec<u8>> {
            let mut out = Vec::with_capacity(vec_m.len_elems * 4);
            for digest in &vec_m.chunks {
                validate_digest_hex(digest)?;
                if !blob_cache.contains_key(digest) {
                    let bp = self.blob_path(digest);
                    let data = fs::read(&bp)
                        .with_context(|| format!("missing blob {}", bp.display()))?;
                    let actual = sha256_hex(&data);
                    if &actual != digest {
                        bail!(
                            "adacheck blob integrity failure for {}: expected {digest}, got {actual}",
                            bp.display()
                        );
                    }
                    blob_cache.insert(digest.clone(), data);
                }
                out.extend_from_slice(&blob_cache[digest]);
            }
            ensure!(
                out.len() == vec_m.len_elems * 4,
                "reconstructed byte length {} != expected {}",
                out.len(),
                vec_m.len_elems * 4
            );
            Ok(out)
        };

        let theta = le_bytes_to_f32_vec(&load_bytes(&manifest.theta)?)?;
        let momentum = le_bytes_to_f32_vec(&load_bytes(&manifest.momentum)?)?;
        let carry = le_bytes_to_f32_vec(&load_bytes(&manifest.carry)?)?;

        let mut inner = HashMap::with_capacity(manifest.inner.len());
        for (name, wm) in &manifest.inner {
            validate_worker_name(name)?;
            ensure!(
                wm.state.len_elems == 3 * params.len,
                "worker {name} state length {} != {}",
                wm.state.len_elems,
                3 * params.len
            );
            let raw = load_bytes(&wm.state)?;
            inner.insert(
                name.clone(),
                Inner {
                    t: wm.t,
                    data: Arc::new(raw),
                },
            );
        }

        let mut merged: Vec<bool> = manifest.merged.iter().map(|&b| b != 0).collect();
        merged.resize(limit, false);

        let logical_bytes = (params.len as u64 * 12)
            + (inner.len() as u64 * params.len as u64 * 12)
            + manifest.merged.len() as u64
            + 8;
        let stats = AdaCheckRestoreStats {
            logical_bytes,
            unique_blobs_read: blob_cache.len(),
            restore_ms: t0.elapsed().as_secs_f64() * 1_000.0,
        };
        Ok((
            Start {
                theta,
                momentum,
                carry,
                merged,
                round: manifest.round,
                inner,
            },
            stats,
        ))
    }
}
