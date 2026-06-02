//! BitDelta: 1-bit-per-weight compression of a fine-tune relative to its base.
//!
//! For each tensor we form the delta `Δ = finetuned − base`, keep a single f32
//! scale `α = mean(|Δ|)`, and store the sign of every element as one packed bit.
//! Reconstruction is `finetuned ≈ base + α · sign(Δ)`. This is the scheme from
//! the BitDelta paper (<https://arxiv.org/abs/2402.10193>); it is lossy in
//! magnitude but exactly preserves the sign/direction of every weight update,
//! shrinking a stored fine-tune delta from ~16/32 bits per weight to ~1.
//!
//! # On-disk `.bitdelta` format (little-endian)
//!
//! ```text
//! magic      : [u8; 8]   = b"BITDELTA"
//! version    : u32       = BITDELTA_VERSION
//! n_tensors  : u32
//! repeated n_tensors times, in the file's tensor order:
//!   name_len : u32
//!   name     : [u8; name_len]   (UTF-8)
//!   rank     : u32
//!   dims     : [u64; rank]
//!   scale    : f32              (mean |Δ| for this tensor)
//!   n_words  : u32              (= ceil(numel / 32))
//!   bits     : [u32; n_words]   (bit i of word w => element 32*w + i; 1 = Δ ≥ 0)
//! ```
//!
//! Only the keys shared between `base` and `finetuned` are encoded; any base
//! tensor the fine-tune does not change is left to the (unchanged) base weights
//! and simply copied through on `decode_apply`.

use super::{fmt_shape, load_checkpoint, save_checkpoint, DEVICE};
use crate::{bail, DType, Result, Tensor};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// Magic bytes at the start of every `.bitdelta` file.
pub const BITDELTA_MAGIC: &[u8; 8] = b"BITDELTA";
/// On-disk format version.
pub const BITDELTA_VERSION: u32 = 1;

/// Per-tensor metadata recovered from a `.bitdelta` file.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDeltaHeader {
    /// Tensor name (must match a key in the base checkpoint to be applied).
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Per-tensor scale `α = mean(|Δ|)`.
    pub scale: f32,
}

/// File-level metadata recovered from a `.bitdelta` file.
#[derive(Debug, Clone, PartialEq)]
pub struct BitDeltaHeader {
    /// Format version (see [`BITDELTA_VERSION`]).
    pub version: u32,
    /// Per-tensor headers, in file order.
    pub tensors: Vec<TensorDeltaHeader>,
}

/// A fully-decoded `.bitdelta`: headers plus the packed sign words for each tensor.
///
/// This is the in-memory representation produced by [`BitDelta::read`] and
/// consumed by [`BitDelta::write`]; most callers want [`encode`]/[`decode_apply`]
/// instead and never touch this directly.
#[derive(Debug, Clone)]
pub struct BitDelta {
    /// File-level header.
    pub header: BitDeltaHeader,
    /// Packed sign words per tensor, aligned 1:1 with `header.tensors`.
    pub packed: Vec<Vec<u32>>,
}

#[inline]
fn numel(shape: &[usize]) -> usize {
    // saturate on overflow so a crafted `.bitdelta` shape can't panic/wrap; the
    // downstream words_len check then rejects the mismatch gracefully.
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .unwrap_or(usize::MAX)
}

#[inline]
fn n_words(numel: usize) -> usize {
    numel.div_ceil(32)
}

/// Pack the signs of `delta` (length `numel`) into u32 words, returning the words
/// and `mean(|delta|)`. Bit `i` is set iff `delta[i] >= 0` (so the zero delta maps
/// to `+scale`, keeping decode deterministic).
fn pack_signs(delta: &[f32]) -> (Vec<u32>, f32) {
    let mut words = vec![0u32; n_words(delta.len())];
    let mut abs_sum = 0f64;
    for (i, &d) in delta.iter().enumerate() {
        abs_sum += d.abs() as f64;
        if d >= 0.0 {
            words[i / 32] |= 1u32 << (i % 32);
        }
    }
    let scale = if delta.is_empty() {
        0.0
    } else {
        (abs_sum / delta.len() as f64) as f32
    };
    (words, scale)
}

/// Reconstruct a signed `±scale` delta of length `numel` from packed sign words.
fn unpack_signed(words: &[u32], numel: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; numel];
    for (i, slot) in out.iter_mut().enumerate() {
        let bit = (words[i / 32] >> (i % 32)) & 1;
        *slot = if bit == 1 { scale } else { -scale };
    }
    out
}

impl BitDelta {
    /// Serialize to the `.bitdelta` binary format.
    pub fn write<W: Write>(&self, mut w: W) -> Result<()> {
        let to_io = |e: std::io::Error| crate::Error::Msg(format!("bitdelta write: {e}"));
        w.write_all(BITDELTA_MAGIC).map_err(to_io)?;
        w.write_all(&self.header.version.to_le_bytes())
            .map_err(to_io)?;
        let n = self.header.tensors.len() as u32;
        w.write_all(&n.to_le_bytes()).map_err(to_io)?;
        for (meta, words) in self.header.tensors.iter().zip(self.packed.iter()) {
            let name = meta.name.as_bytes();
            w.write_all(&(name.len() as u32).to_le_bytes())
                .map_err(to_io)?;
            w.write_all(name).map_err(to_io)?;
            w.write_all(&(meta.shape.len() as u32).to_le_bytes())
                .map_err(to_io)?;
            for &d in &meta.shape {
                w.write_all(&(d as u64).to_le_bytes()).map_err(to_io)?;
            }
            w.write_all(&meta.scale.to_le_bytes()).map_err(to_io)?;
            w.write_all(&(words.len() as u32).to_le_bytes())
                .map_err(to_io)?;
            for &word in words {
                w.write_all(&word.to_le_bytes()).map_err(to_io)?;
            }
        }
        Ok(())
    }

    /// Deserialize from the `.bitdelta` binary format, validating the magic,
    /// version, and per-tensor word counts against the declared shapes.
    pub fn read<R: Read>(mut r: R) -> Result<Self> {
        let to_io = |e: std::io::Error| crate::Error::Msg(format!("bitdelta read: {e}"));
        let mut u32_buf = [0u8; 4];
        let mut u64_buf = [0u8; 8];

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(to_io)?;
        if &magic != BITDELTA_MAGIC {
            bail!("bitdelta: bad magic {magic:?}, not a .bitdelta file");
        }
        r.read_exact(&mut u32_buf).map_err(to_io)?;
        let version = u32::from_le_bytes(u32_buf);
        if version != BITDELTA_VERSION {
            bail!("bitdelta: unsupported version {version} (expected {BITDELTA_VERSION})");
        }
        r.read_exact(&mut u32_buf).map_err(to_io)?;
        let n_tensors = u32::from_le_bytes(u32_buf) as usize;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut packed = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            r.read_exact(&mut u32_buf).map_err(to_io)?;
            let name_len = u32::from_le_bytes(u32_buf) as usize;
            let mut name_bytes = vec![0u8; name_len];
            r.read_exact(&mut name_bytes).map_err(to_io)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| crate::Error::Msg(format!("bitdelta: non-utf8 tensor name: {e}")))?;

            r.read_exact(&mut u32_buf).map_err(to_io)?;
            let rank = u32::from_le_bytes(u32_buf) as usize;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                r.read_exact(&mut u64_buf).map_err(to_io)?;
                shape.push(u64::from_le_bytes(u64_buf) as usize);
            }

            r.read_exact(&mut u32_buf).map_err(to_io)?;
            let scale = f32::from_le_bytes(u32_buf);

            r.read_exact(&mut u32_buf).map_err(to_io)?;
            let words_len = u32::from_le_bytes(u32_buf) as usize;
            let expected = n_words(numel(&shape));
            if words_len != expected {
                bail!(
                    "bitdelta: tensor '{name}' shape {} needs {expected} words but file has {words_len}",
                    fmt_shape(&shape)
                );
            }
            let mut words = vec![0u32; words_len];
            for word in words.iter_mut() {
                r.read_exact(&mut u32_buf).map_err(to_io)?;
                *word = u32::from_le_bytes(u32_buf);
            }

            tensors.push(TensorDeltaHeader { name, shape, scale });
            packed.push(words);
        }

        Ok(BitDelta {
            header: BitDeltaHeader { version, tensors },
            packed,
        })
    }

    /// Read just the header (and packed words) of a `.bitdelta` file from disk.
    pub fn from_file(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path)
            .map_err(|e| crate::Error::Msg(format!("bitdelta: open {}: {e}", path.display())))?;
        Self::read(std::io::BufReader::new(f))
    }
}

/// Encode the fine-tune `finetuned` as a 1-bit delta against `base`, writing a
/// `.bitdelta` file to `out`.
///
/// For every tensor shared by the two checkpoints we store `sign(finetuned−base)`
/// (1 bit/element) plus `mean(|finetuned−base|)` (one f32). A fine-tune tensor
/// whose shape disagrees with the base, or that is missing from the base, is an
/// error; base-only tensors are ignored (decode restores them from the base).
pub fn encode(base: &Path, finetuned: &Path, out: &Path) -> Result<()> {
    let base_map = load_checkpoint(base)?;
    let ft_map = load_checkpoint(finetuned)?;

    // Encode in a stable order so the file is deterministic across runs.
    let mut names: Vec<&String> = ft_map.keys().collect();
    names.sort();

    let mut tensors = Vec::new();
    let mut packed = Vec::new();
    for name in names {
        let ft = &ft_map[name];
        let Some(base_t) = base_map.get(name) else {
            bail!(
                "bitdelta encode: tensor '{name}' in {} is absent from base {}",
                finetuned.display(),
                base.display()
            );
        };
        if ft.dims() != base_t.dims() {
            bail!(
                "bitdelta encode: tensor '{name}' shape mismatch: base {} has {} but {} has {}",
                base.display(),
                fmt_shape(base_t.dims()),
                finetuned.display(),
                fmt_shape(ft.dims())
            );
        }
        let delta: Vec<f32> = ft
            .to_dtype(DType::F32)?
            .sub(&base_t.to_dtype(DType::F32)?)?
            .flatten_all()?
            .to_vec1()?;
        let (words, scale) = pack_signs(&delta);
        tensors.push(TensorDeltaHeader {
            name: name.clone(),
            shape: ft.dims().to_vec(),
            scale,
        });
        packed.push(words);
    }

    let bd = BitDelta {
        header: BitDeltaHeader {
            version: BITDELTA_VERSION,
            tensors,
        },
        packed,
    };
    let f = std::fs::File::create(out)
        .map_err(|e| crate::Error::Msg(format!("bitdelta: create {}: {e}", out.display())))?;
    bd.write(std::io::BufWriter::new(f))
}

/// Reconstruct a fine-tune from `base` plus a `.bitdelta` file, writing the
/// approximate `finetuned ≈ base + scale·sign(Δ)` checkpoint to `out`.
///
/// Tensors named in the `.bitdelta` are reconstructed and overwrite the base;
/// every other base tensor is passed through unchanged. A `.bitdelta` tensor
/// whose shape no longer matches the base, or that the base lacks, is an error.
pub fn decode_apply(base: &Path, bitdelta: &Path, out: &Path) -> Result<()> {
    let bd = BitDelta::from_file(bitdelta)?;
    let base_map = load_checkpoint(base)?;

    let mut out_map: HashMap<String, Tensor> = HashMap::with_capacity(base_map.len());
    // Start from a copy of the base so untouched tensors survive verbatim.
    for (name, tensor) in base_map.iter() {
        out_map.insert(name.clone(), tensor.clone());
    }

    for (meta, words) in bd.header.tensors.iter().zip(bd.packed.iter()) {
        let Some(base_t) = base_map.get(&meta.name) else {
            bail!(
                "bitdelta decode: tensor '{}' is absent from base {}",
                meta.name,
                base.display()
            );
        };
        if base_t.dims() != meta.shape.as_slice() {
            bail!(
                "bitdelta decode: tensor '{}' shape mismatch: base {} has {} but delta has {}",
                meta.name,
                base.display(),
                fmt_shape(base_t.dims()),
                fmt_shape(&meta.shape)
            );
        }
        let n = numel(&meta.shape);
        let signed = unpack_signed(words, n, meta.scale);
        let delta = Tensor::from_vec(signed, meta.shape.clone(), &DEVICE)?;
        let dtype = base_t.dtype();
        let recon = base_t.to_dtype(DType::F32)?.add(&delta)?.to_dtype(dtype)?;
        out_map.insert(meta.name.clone(), recon);
    }

    save_checkpoint(&out_map, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("hanzo_bitdelta_{pid}_{nanos}_{name}"));
        p
    }

    fn write_ckpt(path: &Path, tensors: &[(&str, Tensor)]) {
        let map: HashMap<String, Tensor> =
            tensors.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        save_checkpoint(&map, path).unwrap();
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        (dot / (na * nb)) as f32
    }

    #[test]
    fn pack_unpack_roundtrip_signs_exact() {
        // Exercise the bit-packing primitives directly across a word boundary.
        let delta: Vec<f32> = (0..70)
            .map(|i| if i % 3 == 0 { 1.5 } else { -0.5 } * (i as f32 + 1.0))
            .collect();
        let (words, scale) = pack_signs(&delta);
        assert_eq!(words.len(), super::n_words(70));
        assert!(scale > 0.0);
        let signed = unpack_signed(&words, delta.len(), scale);
        for (d, s) in delta.iter().zip(signed.iter()) {
            assert_eq!(d.is_sign_positive(), *s > 0.0, "sign mismatch for {d}");
        }
    }

    #[test]
    fn roundtrip_random_256x256() {
        use rand::Rng;
        let mut rng = rand::rng();

        // base and a fine-tune that differs from it by a random delta.
        let shape = [256usize, 256];
        let n = shape[0] * shape[1];
        let base_v: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();

        // BitDelta reconstructs every element as ±mean(|Δ|), so the recovered-delta
        // cosine is exactly mean(|Δ|)/rms(Δ) — it is high precisely when magnitudes
        // cluster around one characteristic scale, which is the regime BitDelta
        // targets (one scale per tensor). We model a realistic structured fine-tune
        // update: a random sign times a dominant step `mu` plus small noise, so
        // magnitudes are tightly clustered (not the pathological uniform-around-zero
        // case whose cosine is only sqrt(3)/2 ~ 0.866).
        let mu = 0.05f32;
        let delta_v: Vec<f32> = (0..n)
            .map(|_| {
                let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                let noise = rng.random_range(-0.01..0.01);
                sign * (mu + noise)
            })
            .collect();
        let ft_v: Vec<f32> = base_v.iter().zip(&delta_v).map(|(b, d)| b + d).collect();

        let base = tmp("rt_base.safetensors");
        let ft = tmp("rt_ft.safetensors");
        let bdfile = tmp("rt.bitdelta");
        let recon = tmp("rt_recon.safetensors");

        write_ckpt(
            &base,
            &[("w", Tensor::from_slice(&base_v, &shape, &Device::Cpu).unwrap())],
        );
        write_ckpt(
            &ft,
            &[("w", Tensor::from_slice(&ft_v, &shape, &Device::Cpu).unwrap())],
        );

        encode(&base, &ft, &bdfile).unwrap();
        decode_apply(&base, &bdfile, &recon).unwrap();

        let recon_map = load_checkpoint(&recon).unwrap();
        let recon_w: Vec<f32> = recon_map["w"].flatten_all().unwrap().to_vec1().unwrap();

        // Reconstructed delta vs the true delta.
        let recon_delta: Vec<f32> = recon_w.iter().zip(&base_v).map(|(r, b)| r - b).collect();

        // 1) sign recovery is exact (BitDelta's defining guarantee).
        for (td, rd) in delta_v.iter().zip(&recon_delta) {
            assert_eq!(
                td.is_sign_positive(),
                rd.is_sign_positive(),
                "sign mismatch: true {td} recon {rd}"
            );
        }

        // 2) the reconstructed delta points the same way as the true delta.
        let cos = cosine(&delta_v, &recon_delta);
        assert!(cos > 0.9, "cosine similarity too low: {cos}");

        // 3) the scale is the mean abs delta and every recon delta has that magnitude.
        let true_scale = delta_v.iter().map(|x| x.abs() as f64).sum::<f64>() / n as f64;
        let bd = BitDelta::from_file(&bdfile).unwrap();
        let stored_scale = bd.header.tensors[0].scale;
        assert!(
            (stored_scale as f64 - true_scale).abs() < 1e-4,
            "scale {stored_scale} vs expected {true_scale}"
        );
        for rd in &recon_delta {
            assert!((rd.abs() - stored_scale).abs() < 1e-4, "magnitude {rd}");
        }

        for p in [base, ft, bdfile, recon] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn uniform_delta_hits_theoretical_cosine_bound() {
        // Sanity check on the lossy-ness: for a delta that is uniform on [-a, a],
        // BitDelta replaces each element by ±mean(|Δ|), and the recovered-delta
        // cosine converges to mean(|x|)/rms(x) = (a/2)/(a/sqrt 3) = sqrt(3)/2 ~ 0.866.
        // This documents the worst-case direction loss (and exercises a 512x512 tensor).
        use rand::Rng;
        let mut rng = rand::rng();
        let shape = [512usize, 512];
        let n = shape[0] * shape[1];
        let base_v: Vec<f32> = vec![0.0; n];
        let delta_v: Vec<f32> = (0..n).map(|_| rng.random_range(-0.1f32..0.1)).collect();
        let ft_v: Vec<f32> = delta_v.clone();

        let base = tmp("ub_base.safetensors");
        let ft = tmp("ub_ft.safetensors");
        let bdfile = tmp("ub.bitdelta");
        let recon = tmp("ub_recon.safetensors");
        write_ckpt(
            &base,
            &[("w", Tensor::from_slice(&base_v, &shape, &Device::Cpu).unwrap())],
        );
        write_ckpt(
            &ft,
            &[("w", Tensor::from_slice(&ft_v, &shape, &Device::Cpu).unwrap())],
        );

        encode(&base, &ft, &bdfile).unwrap();
        decode_apply(&base, &bdfile, &recon).unwrap();
        let recon_map = load_checkpoint(&recon).unwrap();
        let recon_w: Vec<f32> = recon_map["w"].flatten_all().unwrap().to_vec1().unwrap();
        let recon_delta: Vec<f32> = recon_w.iter().zip(&base_v).map(|(r, b)| r - b).collect();

        // Signs are still exact even in the worst case.
        for (td, rd) in delta_v.iter().zip(&recon_delta) {
            assert_eq!(td.is_sign_positive(), rd.is_sign_positive());
        }
        let cos = cosine(&delta_v, &recon_delta);
        let theory = (3f32).sqrt() / 2.0; // ~0.866
        assert!(
            (cos - theory).abs() < 0.03,
            "uniform cosine {cos} should sit near the sqrt(3)/2 bound {theory}"
        );

        for p in [base, ft, bdfile, recon] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn header_survives_serialization() {
        let base = tmp("hd_base.safetensors");
        let ft = tmp("hd_ft.safetensors");
        let bdfile = tmp("hd.bitdelta");

        write_ckpt(
            &base,
            &[
                ("a", Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap()),
                ("b", Tensor::zeros((5,), DType::F32, &Device::Cpu).unwrap()),
            ],
        );
        write_ckpt(
            &ft,
            &[
                ("a", Tensor::ones((2, 3), DType::F32, &Device::Cpu).unwrap()),
                ("b", Tensor::ones((5,), DType::F32, &Device::Cpu).unwrap()),
            ],
        );

        encode(&base, &ft, &bdfile).unwrap();
        let bd = BitDelta::from_file(&bdfile).unwrap();
        assert_eq!(bd.header.version, BITDELTA_VERSION);
        assert_eq!(bd.header.tensors.len(), 2);
        // Names are encoded in sorted order: "a" then "b".
        assert_eq!(bd.header.tensors[0].name, "a");
        assert_eq!(bd.header.tensors[0].shape, vec![2, 3]);
        assert_eq!(bd.header.tensors[1].name, "b");
        assert_eq!(bd.header.tensors[1].shape, vec![5]);
        // delta is all +1 -> scale = 1.0, all signs positive.
        assert!((bd.header.tensors[0].scale - 1.0).abs() < 1e-6);

        for p in [base, ft, bdfile] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn decode_preserves_dtype_and_untouched_tensors() {
        let base = tmp("dp_base.safetensors");
        let ft = tmp("dp_ft.safetensors");
        let bdfile = tmp("dp.bitdelta");
        let recon = tmp("dp_recon.safetensors");

        let base_w = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let ft_w = Tensor::from_slice(&[1.5f32, 1.0, 3.5, 3.0], &[2, 2], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let frozen = Tensor::from_slice(&[9.0f32, 8.0], &[2], &Device::Cpu).unwrap();

        write_ckpt(&base, &[("w", base_w), ("frozen", frozen)]);
        write_ckpt(&ft, &[("w", ft_w)]);

        encode(&base, &ft, &bdfile).unwrap();
        decode_apply(&base, &bdfile, &recon).unwrap();

        let recon_map = load_checkpoint(&recon).unwrap();
        // dtype preserved on the reconstructed tensor.
        assert_eq!(recon_map["w"].dtype(), DType::BF16);
        // untouched tensor copied through unchanged.
        let frozen_v: Vec<f32> = recon_map["frozen"].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(frozen_v, vec![9.0, 8.0]);

        for p in [base, ft, bdfile, recon] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn encode_errors_on_shape_mismatch() {
        let base = tmp("em_base.safetensors");
        let ft = tmp("em_ft.safetensors");
        let bdfile = tmp("em.bitdelta");
        write_ckpt(
            &base,
            &[("w", Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap())],
        );
        write_ckpt(
            &ft,
            &[("w", Tensor::zeros((3, 3), DType::F32, &Device::Cpu).unwrap())],
        );

        let err = encode(&base, &ft, &bdfile).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"), "got: {err}");

        for p in [base, ft, bdfile] {
            let _ = std::fs::remove_file(p);
        }
    }

    #[test]
    fn read_rejects_bad_magic() {
        let bytes = b"NOTADELTA\x01\x00\x00\x00";
        let err = BitDelta::read(&bytes[..]).unwrap_err();
        assert!(err.to_string().contains("magic"), "got: {err}");
    }
}
