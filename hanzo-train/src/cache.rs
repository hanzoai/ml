//! Reader for the DeepSpec target-cache **v2** on-disk format.
//!
//! Layout (see the cache `manifest.json` + `scripts/data/convert_engine_dump.py`):
//! * `manifest.json` — `hidden_size`, `target_layer_ids`, `num_samples`, `shards`.
//! * `samples.idx` — dense records of [`REC`] bytes each, little-endian:
//!   `sample_id u64, shard_id u32, seq_len u32`, then 5 `u64` **absolute** byte
//!   offsets into the shard for `[input_ids, attention_mask, loss_mask,
//!   target_hidden_states, target_last_hidden_states]`.
//! * `shard-*.bin` — raw payloads at those offsets:
//!   `input_ids i32[seq]`, `attention_mask u8[seq]`, `loss_mask u8[seq]`,
//!   `target_hidden_states bf16[seq, n_fused*hidden]`,
//!   `target_last_hidden_states bf16[seq, hidden]`.

use hanzo_ml::Result;
use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// bytes per `samples.idx` record.
pub const REC: usize = 56;

#[derive(Debug, Deserialize)]
pub struct ShardEntry {
    pub file_name: String,
    pub shard_id: u32,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub hidden_size: usize,
    pub target_layer_ids: Vec<i64>,
    pub num_samples: usize,
    pub shards: Vec<ShardEntry>,
}

/// bf16 (upper 16 bits of an f32) → f32: place the bits in the high half.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[derive(Debug, Clone, Copy)]
pub struct IndexRecord {
    pub sample_id: u64,
    pub shard_id: u32,
    pub seq_len: u32,
    pub off_input_ids: u64,
    pub off_attn: u64,
    pub off_loss: u64,
    pub off_hidden: u64,
    pub off_last: u64,
}

/// One decoded training sample. All hidden states are converted to f32.
pub struct Sample {
    pub seq_len: usize,
    pub input_ids: Vec<i32>,
    pub attention_mask: Vec<u8>,
    pub loss_mask: Vec<u8>,
    /// `[seq_len * n_fused * hidden]`, row-major over `[seq, n_fused*hidden]`.
    pub target_hidden: Vec<f32>,
    /// `[seq_len * hidden]`.
    pub target_last_hidden: Vec<f32>,
}

pub struct Cache {
    pub manifest: Manifest,
    pub records: Vec<IndexRecord>,
    pub n_fused: usize,
    shard: File,
}

#[inline]
fn rd_u64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
#[inline]
fn rd_u32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn io<E: std::fmt::Display>(e: E) -> hanzo_ml::Error {
    hanzo_ml::Error::msg(e.to_string())
}

impl Cache {
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir = dir.as_ref();
        let mtext = std::fs::read_to_string(dir.join("manifest.json")).map_err(io)?;
        let manifest: Manifest = serde_json::from_str(&mtext).map_err(io)?;
        if manifest.shards.len() != 1 {
            hanzo_ml::bail!(
                "MVP cache reader supports a single shard, got {}",
                manifest.shards.len()
            );
        }

        let idx = std::fs::read(dir.join("samples.idx")).map_err(io)?;
        if idx.len() % REC != 0 {
            hanzo_ml::bail!(
                "samples.idx length {} is not a multiple of record size {REC}",
                idx.len()
            );
        }
        let n = idx.len() / REC;
        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let o = i * REC;
            records.push(IndexRecord {
                sample_id: rd_u64(&idx, o),
                shard_id: rd_u32(&idx, o + 8),
                seq_len: rd_u32(&idx, o + 12),
                off_input_ids: rd_u64(&idx, o + 16),
                off_attn: rd_u64(&idx, o + 24),
                off_loss: rd_u64(&idx, o + 32),
                off_hidden: rd_u64(&idx, o + 40),
                off_last: rd_u64(&idx, o + 48),
            });
        }

        let shard = File::open(dir.join(&manifest.shards[0].file_name)).map_err(io)?;
        let n_fused = manifest.target_layer_ids.len();
        Ok(Self {
            manifest,
            records,
            n_fused,
            shard,
        })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    fn read_at(&self, off: u64, len: usize) -> Result<Vec<u8>> {
        let mut f = &self.shard;
        f.seek(SeekFrom::Start(off)).map_err(io)?;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf).map_err(io)?;
        Ok(buf)
    }

    pub fn read_sample(&self, i: usize) -> Result<Sample> {
        let r = self.records[i];
        let seq = r.seq_len as usize;
        let h = self.manifest.hidden_size;
        let nf = self.n_fused;

        let raw = self.read_at(r.off_input_ids, seq * 4)?;
        let input_ids: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let attention_mask = self.read_at(r.off_attn, seq)?;
        let loss_mask = self.read_at(r.off_loss, seq)?;

        let hraw = self.read_at(r.off_hidden, seq * nf * h * 2)?;
        let target_hidden: Vec<f32> = hraw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect();

        let lraw = self.read_at(r.off_last, seq * h * 2)?;
        let target_last_hidden: Vec<f32> = lraw
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect();

        Ok(Sample {
            seq_len: seq,
            input_ids,
            attention_mask,
            loss_mask,
            target_hidden,
            target_last_hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI300: &str = "/home/z/work/zen/hf/v4-dspark-cache/mini300_cache";

    #[test]
    fn bf16_roundtrip() {
        // 1.0f32 == 0x3F800000; bf16 upper half == 0x3F80.
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
    }

    #[test]
    fn read_record_1_shapes_and_finite() -> Result<()> {
        let dir = Path::new(MINI300);
        if !dir.join("samples.idx").exists() {
            eprintln!("skipping: mini300 cache not present at {}", dir.display());
            return Ok(());
        }
        let cache = Cache::open(dir)?;
        assert_eq!(
            cache.n_fused, 5,
            "target_layer_ids [1,11,21,31,41] => 5 fused"
        );
        assert_eq!(cache.manifest.hidden_size, 4096);

        // Record 1 has seq_len 274 (verified against the idx layout).
        let s = cache.read_sample(1)?;
        assert_eq!(s.seq_len, 274);
        assert_eq!(s.input_ids.len(), s.seq_len);
        assert_eq!(s.attention_mask.len(), s.seq_len);
        assert_eq!(s.loss_mask.len(), s.seq_len);
        assert_eq!(
            s.target_hidden.len(),
            s.seq_len * cache.n_fused * cache.manifest.hidden_size
        );
        assert_eq!(
            s.target_last_hidden.len(),
            s.seq_len * cache.manifest.hidden_size
        );
        assert!(
            s.target_hidden.iter().all(|v| v.is_finite()),
            "hidden states must be finite"
        );
        assert!(s.target_last_hidden.iter().all(|v| v.is_finite()));
        // loss_mask is "ones_except_first": position 0 disabled, rest enabled.
        assert_eq!(s.loss_mask[0], 0);
        assert!(s.loss_mask[1..].iter().all(|&m| m == 1));
        Ok(())
    }
}
