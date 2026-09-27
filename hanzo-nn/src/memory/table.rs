//! A table on the host: shards in safetensors files, mapped and read row by row.
//!
//! The table under `prefix` is the tensors `{prefix}.shard_{i}.weight`, each `(rows, dim)`, row
//! `r` in shard `r / per` at `r % per` (`per` the first shard's rows; every shard but the last
//! holds that many). They may sit in one file or several, and anywhere in a file: each is read
//! from its own header range, never from a stride, and at any byte alignment. Rows are F32, BF16
//! or F8E4M3; an F8E4M3 table carries one scale, `{prefix}.weight_scale` (BF16 or F32), and a row
//! reads as `f32(fp8) · scale`. The mapping is advised random access; only rows a pass reads are
//! paged in, and nothing of the table is uploaded but the rows [`super::Lookup`] gathers.

use hanzo_ml::{bail, DType, Device, Result, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Largest finite E4M3 magnitude: the scale maps a table's largest entry here.
const E4M3: f32 = 448.0;

/// One tensor's place in a safetensors file.
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A safetensors file's header, parsed: every tensor's dtype, shape and absolute byte range.
fn header(path: &Path) -> Result<HashMap<String, Entry>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(hanzo_ml::Error::wrap)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(hanzo_ml::Error::wrap)?;
    let n = u64::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(hanzo_ml::Error::wrap)?;
    let json: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&buf).map_err(hanzo_ml::Error::wrap)?;
    let mut out = HashMap::new();
    for (name, v) in json {
        let (Some(dtype), Some(shape), Some(off)) = (
            v["dtype"].as_str(),
            v["shape"].as_array(),
            v["data_offsets"].as_array(),
        ) else {
            continue;
        };
        let at = |i: usize| {
            off.get(i)
                .and_then(|o| o.as_u64())
                .map(|o| 8 + n + o as usize)
        };
        let (Some(start), Some(end)) = (at(0), at(1)) else {
            bail!("{}: {name} has no data offsets", path.display());
        };
        out.insert(
            name,
            Entry {
                dtype: dtype.to_string(),
                shape: shape
                    .iter()
                    .map(|d| d.as_u64().unwrap_or(0) as usize)
                    .collect(),
                start,
                end,
            },
        );
    }
    Ok(out)
}

fn dtype(name: &str) -> Result<DType> {
    Ok(match name {
        "F32" => DType::F32,
        "BF16" => DType::BF16,
        "F8_E4M3" => DType::F8E4M3,
        d => bail!("a table of {d} rows"),
    })
}

/// A shard: the mapping holding it, where its rows start, and how many it has.
struct Shard {
    map: usize,
    at: usize,
    rows: usize,
}

/// A mapped table.
pub struct Table {
    maps: Vec<Arc<memmap2::Mmap>>,
    shards: Vec<Shard>,
    /// Rows of every shard but the last.
    per: usize,
    pub dim: usize,
    /// F32, BF16 or F8E4M3.
    pub dtype: DType,
    /// The F8E4M3 scale; 1 otherwise.
    pub scale: f32,
    /// Rows in all.
    pub rows: usize,
}

impl Table {
    /// The table under `prefix` in `files`: headers parsed, every file holding a shard mapped,
    /// and each shard's range advised random access.
    pub fn open(files: &[PathBuf], prefix: &str) -> Result<Table> {
        let mut found: HashMap<usize, (usize, Entry)> = HashMap::new();
        let mut scale = None;
        let mut maps = Vec::new();
        for path in files {
            let mut head = header(path)?;
            let mut mine = Vec::new();
            let names: Vec<String> = head.keys().cloned().collect();
            for name in names {
                let Some(rest) = name.strip_prefix(prefix) else {
                    continue;
                };
                if let Some(i) = rest
                    .strip_prefix(".shard_")
                    .and_then(|r| r.strip_suffix(".weight"))
                    .and_then(|i| i.parse::<usize>().ok())
                {
                    mine.push((i, head.remove(&name).expect("listed")));
                } else if rest == ".weight_scale" {
                    scale = Some((maps.len(), head.remove(&name).expect("listed")));
                }
            }
            let holds = !mine.is_empty() || scale.as_ref().is_some_and(|(m, _)| *m == maps.len());
            if !holds {
                continue;
            }
            let f = std::fs::File::open(path).map_err(hanzo_ml::Error::wrap)?;
            // SAFETY: the file is opened read-only and the table's rows are only ever read.
            let map = unsafe { memmap2::Mmap::map(&f).map_err(hanzo_ml::Error::wrap)? };
            for (i, e) in mine {
                if e.end > map.len() {
                    bail!("{}: {prefix}.shard_{i} runs past the file", path.display());
                }
                if found.insert(i, (maps.len(), e)).is_some() {
                    bail!("{prefix}.shard_{i} is in two files");
                }
            }
            maps.push(Arc::new(map));
        }
        let Some((_, first)) = found.get(&0) else {
            bail!("no file holds {prefix}.shard_0.weight");
        };
        if first.shape.len() != 2 {
            bail!("{prefix}.shard_0 is {:?}, expected 2-D", first.shape);
        }
        let (per, dim, kind) = (first.shape[0], first.shape[1], first.dtype.clone());
        let dt = dtype(&kind)?;
        let mut shards = Vec::with_capacity(found.len());
        for i in 0..found.len() {
            let Some((map, e)) = found.remove(&i) else {
                bail!(
                    "{prefix}.shard_{i} is missing: shards run 0..{}",
                    shards.len()
                );
            };
            let last = found.is_empty();
            let ok = e.dtype == kind
                && e.shape.len() == 2
                && e.shape[1] == dim
                && (e.shape[0] == per || last && e.shape[0] <= per)
                && e.end - e.start == e.shape[0] * dim * dt.size_in_bytes();
            if !ok {
                bail!(
                    "{prefix}.shard_{i} is {} {:?}; shard 0 is {kind} {:?}",
                    e.dtype,
                    e.shape,
                    [per, dim]
                );
            }
            shards.push(Shard {
                map,
                at: e.start,
                rows: e.shape[0],
            });
        }
        let scale = match (dt, scale) {
            (DType::F8E4M3, Some((m, e))) => {
                let raw = &maps[m][e.start..e.end];
                match (e.dtype.as_str(), raw.len()) {
                    ("BF16", 2) => half::bf16::from_le_bytes([raw[0], raw[1]]).to_f32(),
                    ("F32", 4) => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                    (d, n) => bail!("{prefix}.weight_scale is {d} of {n} bytes"),
                }
            }
            (DType::F8E4M3, None) => bail!("{prefix}.weight_scale is missing beside the table"),
            _ => 1.0,
        };
        #[cfg(unix)]
        for s in &shards {
            let map = &maps[s.map];
            let page = 4096;
            let lo = s.at / page * page;
            let hi = s.at + s.rows * dim * dt.size_in_bytes();
            // SAFETY: the range lies inside the mapping; advice changes no contents.
            unsafe {
                libc::madvise(
                    map.as_ptr().add(lo) as *mut libc::c_void,
                    hi - lo,
                    libc::MADV_RANDOM,
                );
            }
        }
        Ok(Table {
            rows: shards.iter().map(|s| s.rows).sum(),
            maps,
            shards,
            per,
            dim,
            dtype: dt,
            scale,
        })
    }

    /// Bytes per row.
    pub fn width(&self) -> usize {
        self.dim * self.dtype.size_in_bytes()
    }

    /// Row `r`'s raw bytes.
    pub fn row(&self, r: u32) -> Result<&[u8]> {
        let r = r as usize;
        let Some(s) = self.shards.get(r / self.per.max(1)) else {
            bail!("row {r} is past the table's {}", self.rows);
        };
        let i = r % self.per.max(1);
        if i >= s.rows {
            bail!("row {r} is past the table's {}", self.rows);
        }
        let w = self.width();
        let at = s.at + i * w;
        Ok(&self.maps[s.map][at..at + w])
    }

    /// The raw bytes of `rows`, [`Table::width`] each, in order.
    pub fn bytes(&self, rows: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(rows.len() * self.width());
        for &r in rows {
            out.extend_from_slice(self.row(r)?);
        }
        Ok(out)
    }

    /// Row `r` as F32 into `out` (`dim` long): an F8E4M3 row times the scale, rounded once to
    /// F32, exactly as [`Table::gather`] computes it before its BF16 rounding.
    pub fn read(&self, r: u32, out: &mut [f32]) -> Result<()> {
        let raw = self.row(r)?;
        match self.dtype {
            DType::F32 => {
                for (o, b) in out.iter_mut().zip(raw.chunks_exact(4)) {
                    *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            DType::BF16 => {
                for (o, b) in out.iter_mut().zip(raw.chunks_exact(2)) {
                    *o = half::bf16::from_le_bytes([b[0], b[1]]).to_f32();
                }
            }
            _ => {
                for (o, &b) in out.iter_mut().zip(raw) {
                    *o = float8::F8E4M3::from_bits(b).to_f32() * self.scale;
                }
            }
        }
        Ok(())
    }

    /// `rows` as a `(rows, dim)` tensor on `dev`: F32 and BF16 rows as stored; F8E4M3 rows as
    /// `bf16(f32(fp8) · f32(scale))`, dequantized on the host (the served dequant kernel stores
    /// exactly that), so only the rows asked for are read and uploaded.
    pub fn gather(&self, rows: &[u32], dev: &Device) -> Result<Tensor> {
        let bytes = self.bytes(rows)?;
        let t = Tensor::from_raw_buffer(&bytes, self.dtype, &[rows.len(), self.dim], &Device::Cpu)?;
        let t = match self.dtype {
            DType::F8E4M3 => t
                .to_dtype(DType::F32)?
                .affine(f64::from(self.scale), 0.0)?
                .to_dtype(DType::BF16)?,
            _ => t,
        };
        t.to_device(dev)
    }

    /// Write `data` (`rows × dim` F32, row-major) as the table under `prefix` into `dir`,
    /// `shard_{i}.safetensors` of `per` rows each (the last may hold fewer), stored as `dtype`
    /// (F32, BF16, or F8E4M3 under one F32 scale that maps the largest magnitude to 448, written
    /// into shard 0). Returns the files.
    pub fn write(
        dir: &Path,
        prefix: &str,
        data: &[f32],
        dim: usize,
        dtype: DType,
        per: usize,
    ) -> Result<Vec<PathBuf>> {
        if dim == 0 || data.len() % dim != 0 || per == 0 {
            bail!(
                "a table of {} values in rows of {dim}, {per} a shard",
                data.len()
            );
        }
        std::fs::create_dir_all(dir).map_err(hanzo_ml::Error::wrap)?;
        let top = data.iter().fold(0f32, |m, x| m.max(x.abs()));
        let scale = if top > 0.0 { top / E4M3 } else { 1.0 };
        let rows = data.len() / dim;
        let mut files = Vec::new();
        for (i, start) in (0..rows).step_by(per).enumerate() {
            let n = per.min(rows - start);
            let part = &data[start * dim..(start + n) * dim];
            let t = Tensor::from_slice(part, (n, dim), &Device::Cpu)?;
            let t = match dtype {
                DType::F32 => t,
                DType::BF16 => t.to_dtype(DType::BF16)?,
                DType::F8E4M3 => {
                    let q: Vec<float8::F8E4M3> = part
                        .iter()
                        .map(|&x| float8::F8E4M3::from_f32(x / scale))
                        .collect();
                    Tensor::from_vec(q, (n, dim), &Device::Cpu)?
                }
                d => bail!("a table of {d:?} rows"),
            };
            let mut ts = HashMap::from([(format!("{prefix}.shard_{i}.weight"), t)]);
            if i == 0 && dtype == DType::F8E4M3 {
                ts.insert(
                    format!("{prefix}.weight_scale"),
                    Tensor::from_slice(&[scale], 1, &Device::Cpu)?,
                );
            }
            let path = dir.join(format!("shard_{i}.safetensors"));
            hanzo_ml::safetensors::save(&ts, &path)?;
            files.push(path);
        }
        Ok(files)
    }
}
