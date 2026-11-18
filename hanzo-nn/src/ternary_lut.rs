//! Table-lookup ternary inference on ARM and x86 CPUs (T-MAC style, `W1.58A8`).
//!
//! Clean-room `MIT OR Apache-2.0` implementation inspired by study-only architectural
//! concepts from:
//! - Jianyu Wei et al., *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment
//!   on Edge*, [arXiv:2407.00088](https://arxiv.org/abs/2407.00088).
//! - Shuming Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*,
//!   [arXiv:2402.17764](https://arxiv.org/abs/2402.17764).
//!
//! Instead of expanding 2-bit ternary weights \\(w \in \{-1, 0, +1\}\\) back into `f32` or `i8`
//! on every matrix-vector product, [`ActivationLut`] precomputes partial sums over groups of
//! \\(g = 4\\) quantized input activations once per token (`O(K)` work) and reuses that lookup
//! table across all `N` output channels via register-tiled SIMD byte shuffles (`vpshufb` on
//! x86 AVX2 / AVX-512 wide tiles and `vqtbl1q_s8` on ARM NEON) with a bit-identical `4x`
//! output-row register-tiled scalar fallback.
//!
//! ```rust
//! use hanzo_ml::{Device, Module, Tensor};
//! use hanzo_nn::ternary_lut::{TmacKernelBackend, TmacLinear};
//! # fn main() -> hanzo_ml::Result<()> {
//! let w = Tensor::new(&[[0.9f32, -0.8, 0.0, 1.1], [-1.0, 0.7, 0.9, -0.05]], &Device::Cpu)?;
//! let layer = TmacLinear::from_tensor(&w, None, TmacKernelBackend::Auto)?;
//! let x = Tensor::new(&[[0.5f32, -0.25, 0.75, 1.0]], &Device::Cpu)?;
//! let y = layer.forward(&x)?;
//! assert_eq!(y.dims(), &[1, 2]);
//! # Ok(()) }
//! ```

use hanzo_ml::{DType, Module, Result, Tensor};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Number of ternary weights (`{-1, 0, +1}`) packed into each byte (`2` bits per trit).
pub const TRITS_PER_BYTE: usize = 4;

/// Total distinct algebraic states for a group of `4` ternary weights (`3^4 = 81`).
pub const RADIX3_GROUP4_STATES: usize = 81;

/// Half-table size for odd-symmetric radix-3 lookup (`0..=40`, where `40` is all-zero).
pub const SYMMETRIC_GROUP4_STATES: usize = 41;

/// Output-channel interleave tile width (`32` rows = one 256-bit SIMD byte vector).
pub const N_TILE: usize = 32;

/// Number of 4-trit groups accumulated in `i16` before widening to `i32`
/// (`32 * 4 * 127 = 16_256 <= i16::MAX`, guaranteeing zero `i16` overflow).
pub const I16_ACCUM_GROUP_CHUNK: usize = 32;

/// Encodes a single ternary weight `w in {-1, 0, +1}` into a 2-bit code:
/// `0 -> 0b00`, `+1 -> 0b01`, `-1 -> 0b10`.
#[inline]
pub const fn encode_trit(w: i8) -> u8 {
    match w {
        1 => 0b01,
        -1 => 0b10,
        _ => 0b00,
    }
}

/// Decodes a 2-bit code back into `{-1, 0, +1}`.
#[inline]
pub const fn decode_trit(code: u8) -> i8 {
    match code & 0b11 {
        0b01 => 1,
        0b10 => -1,
        _ => 0,
    }
}

/// Packs 4 ternary weights `w in {-1, 0, +1}` into a single byte (`4.0x` compression vs `i8`).
#[inline]
pub const fn pack_trit_group4(trits: [i8; TRITS_PER_BYTE]) -> u8 {
    encode_trit(trits[0])
        | (encode_trit(trits[1]) << 2)
        | (encode_trit(trits[2]) << 4)
        | (encode_trit(trits[3]) << 6)
}

/// Unpacks a packed byte back into 4 ternary weights `w in {-1, 0, +1}`.
#[inline]
pub const fn unpack_trit_group4(byte: u8) -> [i8; TRITS_PER_BYTE] {
    [
        decode_trit(byte),
        decode_trit(byte >> 2),
        decode_trit(byte >> 4),
        decode_trit(byte >> 6),
    ]
}

/// Maps 4 ternary weights in `{-1, 0, +1}^4` bijectively onto `0..81`.
#[inline]
pub const fn trits_to_radix81(trits: [i8; TRITS_PER_BYTE]) -> usize {
    let c0 = (trits[0] + 1) as usize;
    let c1 = (trits[1] + 1) as usize;
    let c2 = (trits[2] + 1) as usize;
    let c3 = (trits[3] + 1) as usize;
    c0 + 3 * c1 + 9 * c2 + 27 * c3
}

/// Maps a radix-3 index `idx in 0..81` back into `{-1, 0, +1}^4`.
#[inline]
pub const fn radix81_to_trits(idx: usize) -> [i8; TRITS_PER_BYTE] {
    let t0 = ((idx % 3) as i8) - 1;
    let t1 = (((idx / 3) % 3) as i8) - 1;
    let t2 = (((idx / 9) % 3) as i8) - 1;
    let t3 = (((idx / 27) % 3) as i8) - 1;
    [t0, t1, t2, t3]
}

/// Precomputes the 41-entry odd-symmetric mirror table (`0..=40`) for `4` `i8` activations.
///
/// Negating all 4 trits maps radix-3 state `u` to `80 - u`, so states `41..81` are exact
/// negations of `0..40` and state `40` is identically `0`.
#[inline]
pub fn build_symmetric_lut41(q: [i8; TRITS_PER_BYTE]) -> [i16; SYMMETRIC_GROUP4_STATES] {
    let mut table = [0i16; SYMMETRIC_GROUP4_STATES];
    let q0 = i16::from(q[0]);
    let q1 = i16::from(q[1]);
    let q2 = i16::from(q[2]);
    let q3 = i16::from(q[3]);
    for u in 0..SYMMETRIC_GROUP4_STATES {
        let [t0, t1, t2, t3] = radix81_to_trits(u);
        table[u] = i16::from(t0) * q0
            + i16::from(t1) * q1
            + i16::from(t2) * q2
            + i16::from(t3) * q3;
    }
    table
}

/// Evaluates a 4-trit dot product from the 41-entry symmetric mirror LUT (`0..81`).
#[inline]
pub fn lookup_symmetric_lut41(
    sym41: &[i16; SYMMETRIC_GROUP4_STATES],
    radix_idx: usize,
) -> i16 {
    if radix_idx < SYMMETRIC_GROUP4_STATES {
        sym41[radix_idx]
    } else {
        -sym41[80 - radix_idx]
    }
}

/// Precomputes all `3^4 = 81` radix-3 partial sums for a group of `4` `i8` activations.
#[inline]
pub fn build_radix81_lut(q: [i8; TRITS_PER_BYTE]) -> [i16; RADIX3_GROUP4_STATES] {
    let sym41 = build_symmetric_lut41(q);
    let mut out = [0i16; RADIX3_GROUP4_STATES];
    for u in 0..RADIX3_GROUP4_STATES {
        out[u] = lookup_symmetric_lut41(&sym41, u);
    }
    out
}

/// Deterministic BitNet b1.58 absmean per-row quantization of `f32` weights `[rows, cols]`.
///
/// For each row `r`, computes `alpha_r = mean(|w_{r,*}|)` and quantizes
/// `t_{r,c} = round(w_{r,c} / alpha_r).clamp(-1.0, 1.0) as i8`.
pub fn quantize_absmean_ternary(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(Vec<i8>, Vec<f32>)> {
    if rows == 0 || cols == 0 {
        hanzo_ml::bail!("quantize_absmean_ternary requires non-zero shape, got ({rows}, {cols})");
    }
    let expected = rows.saturating_mul(cols);
    if weights.len() != expected {
        hanzo_ml::bail!(
            "quantize_absmean_ternary weight length {} != rows * cols ({expected})",
            weights.len()
        );
    }
    let mut trits = vec![0i8; expected];
    let mut scales = vec![0.0f32; rows];
    let inv_cols = 1.0f32 / (cols as f32);
    for r in 0..rows {
        let row_slice = &weights[r * cols..(r + 1) * cols];
        let abs_sum: f32 = row_slice.iter().map(|v| v.abs()).sum();
        let alpha = abs_sum * inv_cols;
        if alpha > 1e-12 {
            scales[r] = alpha;
            let inv_alpha = 1.0f32 / alpha;
            let dst = &mut trits[r * cols..(r + 1) * cols];
            for c in 0..cols {
                dst[c] = (row_slice[c] * inv_alpha).round().clamp(-1.0, 1.0) as i8;
            }
        }
    }
    Ok((trits, scales))
}

/// Deterministic grouped BitNet b1.58 absmean quantization with per-group scales.
pub fn quantize_absmean_ternary_grouped(
    weights: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<(Vec<i8>, Vec<f32>)> {
    if rows == 0 || cols == 0 || group_size == 0 {
        hanzo_ml::bail!(
            "quantize_absmean_ternary_grouped invalid ({rows}, {cols}) group={group_size}"
        );
    }
    let expected = rows.saturating_mul(cols);
    if weights.len() != expected {
        hanzo_ml::bail!(
            "quantize_absmean_ternary_grouped weight length {} != {expected}",
            weights.len()
        );
    }
    let groups_per_row = cols.div_ceil(group_size);
    let mut trits = vec![0i8; expected];
    let mut scales = vec![0.0f32; rows * groups_per_row];
    for r in 0..rows {
        for g in 0..groups_per_row {
            let c0 = g * group_size;
            let c1 = (c0 + group_size).min(cols);
            let slice = &weights[r * cols + c0..r * cols + c1];
            let alpha = slice.iter().map(|v| v.abs()).sum::<f32>() / ((c1 - c0) as f32);
            if alpha > 1e-12 {
                scales[r * groups_per_row + g] = alpha;
                let inv_alpha = 1.0f32 / alpha;
                for c in c0..c1 {
                    let w = weights[r * cols + c];
                    trits[r * cols + c] = (w * inv_alpha).round().clamp(-1.0, 1.0) as i8;
                }
            }
        }
    }
    Ok((trits, scales))
}

/// Bit-packed ternary matrix (`4` trits per byte) interleaved in `N_TILE = 32` row tiles
/// for zero-unpacking SIMD lookup-table inference.
#[derive(Clone, Debug)]
pub struct PackedTernaryMatrix {
    rows: usize,
    cols: usize,
    groups_per_row: usize,
    num_tiles: usize,
    tiled_packed: Vec<u8>,
    row_scales: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl PackedTernaryMatrix {
    /// Packs `{-1, 0, +1}` ternary weights `[rows, cols]` with per-row `f32` scales.
    pub fn from_trits(
        trits: &[i8],
        rows: usize,
        cols: usize,
        row_scales: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        if rows == 0 || cols == 0 {
            hanzo_ml::bail!(
                "PackedTernaryMatrix requires non-zero (rows, cols), got ({rows}, {cols})"
            );
        }
        let expected = rows.saturating_mul(cols);
        if trits.len() != expected {
            hanzo_ml::bail!(
                "PackedTernaryMatrix trits len {} != rows * cols ({expected})",
                trits.len()
            );
        }
        if row_scales.len() != rows {
            hanzo_ml::bail!(
                "PackedTernaryMatrix row_scales len {} != rows ({rows})",
                row_scales.len()
            );
        }
        if let Some(ref b) = bias {
            if b.len() != rows {
                hanzo_ml::bail!("PackedTernaryMatrix bias len {} != rows ({rows})", b.len());
            }
        }

        let groups_per_row = cols.div_ceil(TRITS_PER_BYTE);
        let num_tiles = rows.div_ceil(N_TILE);
        let mut tiled_packed = vec![0u8; num_tiles * groups_per_row * N_TILE];

        for r in 0..rows {
            let tile = r / N_TILE;
            let lane = r - tile * N_TILE;
            let row_base = r * cols;
            let tile_base = tile * groups_per_row * N_TILE;
            for g in 0..groups_per_row {
                let c0 = g * TRITS_PER_BYTE;
                let mut grp = [0i8; TRITS_PER_BYTE];
                for m in 0..TRITS_PER_BYTE {
                    if c0 + m < cols {
                        let v = trits[row_base + c0 + m];
                        if v < -1 || v > 1 {
                            hanzo_ml::bail!("Invalid ternary weight {v} outside {{-1, 0, +1}}");
                        }
                        grp[m] = v;
                    }
                }
                tiled_packed[tile_base + g * N_TILE + lane] = pack_trit_group4(grp);
            }
        }

        Ok(Self {
            rows,
            cols,
            groups_per_row,
            num_tiles,
            tiled_packed,
            row_scales,
            bias,
        })
    }

    /// Quantizes continuous `f32` weights `[rows, cols]` via BitNet b1.58 absmean scaling.
    pub fn from_f32_absmean(
        weights: &[f32],
        rows: usize,
        cols: usize,
        bias: Option<Vec<f32>>,
    ) -> Result<Self> {
        let (trits, scales) = quantize_absmean_ternary(weights, rows, cols)?;
        Self::from_trits(&trits, rows, cols, scales, bias)
    }

    /// Quantizes a 2D `hanzo_ml::Tensor` `[out_features, in_features]` via absmean scaling.
    pub fn from_tensor_absmean(weight: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        let (rows, cols) = weight.dims2()?;
        let w_f32 = weight.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b_f32 = match bias {
            Some(b) => Some(b.to_dtype(DType::F32)?.to_vec1::<f32>()?),
            None => None,
        };
        Self::from_f32_absmean(&w_f32, rows, cols, b_f32)
    }

    /// Output dimension `N`.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Input dimension `K`.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of 4-trit byte groups per row (`ceil(K / 4)`).
    #[inline]
    pub fn groups_per_row(&self) -> usize {
        self.groups_per_row
    }

    /// Per-row absmean scaling factors `alpha` (`[rows]`).
    #[inline]
    pub fn row_scales(&self) -> &[f32] {
        &self.row_scales
    }

    /// Optional bias vector (`[rows]`).
    #[inline]
    pub fn bias(&self) -> Option<&[f32]> {
        self.bias.as_deref()
    }

    /// Logical bit-packed weight payload in bytes (`rows * ceil(cols / 4)`).
    #[inline]
    pub fn packed_weight_bytes(&self) -> usize {
        self.rows * self.groups_per_row
    }

    /// Total heap bytes held by the tile buffer, per-row scales, and optional bias.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        let bias_bytes = self.bias.as_ref().map_or(0, |b| b.len() * 4);
        self.tiled_packed.len() + self.row_scales.len() * 4 + bias_bytes
    }

    /// Weight compression ratio relative to `INT8` (`1` byte/weight), `4.0x` for aligned `K`.
    #[inline]
    pub fn compression_vs_int8(&self) -> f64 {
        (self.rows * self.cols) as f64 / (self.packed_weight_bytes() as f64)
    }

    /// Weight compression ratio relative to `FP32` (`4` bytes/weight), `16.0x` for aligned `K`.
    #[inline]
    pub fn compression_vs_fp32(&self) -> f64 {
        (self.rows * self.cols * 4) as f64 / (self.packed_weight_bytes() as f64)
    }

    /// Reads the packed 4-trit byte at `(row, group)`.
    #[inline]
    pub fn packed_byte(&self, row: usize, group: usize) -> u8 {
        let tile = row / N_TILE;
        let lane = row - tile * N_TILE;
        self.tiled_packed[(tile * self.groups_per_row + group) * N_TILE + lane]
    }

    /// Unpacks a single output row into `{-1, 0, +1}` trits (used for verification and reference).
    pub fn unpack_row(&self, row: usize) -> Vec<i8> {
        let mut out = Vec::with_capacity(self.cols);
        for g in 0..self.groups_per_row {
            let trits = unpack_trit_group4(self.packed_byte(row, g));
            for t in trits {
                if out.len() < self.cols {
                    out.push(t);
                }
            }
        }
        out
    }

    /// Dequantizes the entire matrix into dense `f32` `[rows * cols]` for `Fp32Reference`.
    pub fn dequantize_f32(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.rows * self.cols];
        for r in 0..self.rows {
            let alpha = self.row_scales[r];
            let trits = self.unpack_row(r);
            let dst = &mut dense[r * self.cols..(r + 1) * self.cols];
            for c in 0..self.cols {
                dst[c] = (trits[c] as f32) * alpha;
            }
        }
        dense
    }
}

/// Cache-line-sized (`64`-byte) nibble-split 16-entry SIMD lookup tables for one group of `4`
/// `i8` input activations `(q0, q1, q2, q3)`.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug)]
pub struct SimdLutGroup {
    pub lo_lsb: [u8; 16],
    pub lo_msb: [u8; 16],
    pub hi_lsb: [u8; 16],
    pub hi_msb: [u8; 16],
}

impl SimdLutGroup {
    /// Builds nibble-split two's-complement `i16` byte tables for `(q0, q1)` and `(q2, q3)`.
    #[inline]
    pub fn from_group4(q: [i8; TRITS_PER_BYTE]) -> Self {
        let mut lo_lsb = [0u8; 16];
        let mut lo_msb = [0u8; 16];
        let mut hi_lsb = [0u8; 16];
        let mut hi_msb = [0u8; 16];
        let q0 = i16::from(q[0]);
        let q1 = i16::from(q[1]);
        let q2 = i16::from(q[2]);
        let q3 = i16::from(q[3]);
        for nibble in 0..16usize {
            let ta = i16::from(decode_trit(nibble as u8));
            let tb = i16::from(decode_trit((nibble as u8) >> 2));
            let sum_lo = (ta * q0 + tb * q1).to_le_bytes();
            let sum_hi = (ta * q2 + tb * q3).to_le_bytes();
            lo_lsb[nibble] = sum_lo[0];
            lo_msb[nibble] = sum_lo[1];
            hi_lsb[nibble] = sum_hi[0];
            hi_msb[nibble] = sum_hi[1];
        }
        Self {
            lo_lsb,
            lo_msb,
            hi_lsb,
            hi_msb,
        }
    }

    /// Evaluates the exact `i32` partial sum for one packed 4-trit weight byte.
    #[inline]
    pub fn lookup_byte_i32(&self, packed_byte: u8) -> i32 {
        let lo = usize::from(packed_byte & 0x0F);
        let hi = usize::from(packed_byte >> 4);
        let v_lo = i16::from_le_bytes([self.lo_lsb[lo], self.lo_msb[lo]]);
        let v_hi = i16::from_le_bytes([self.hi_lsb[hi], self.hi_msb[hi]]);
        i32::from(v_lo) + i32::from(v_hi)
    }
}

/// Per-token quantized activation vector and precomputed T-MAC lookup tables (`O(K)` build).
#[derive(Clone, Debug)]
pub struct ActivationLut {
    cols: usize,
    act_scale: f32,
    quantized_x: Vec<i8>,
    groups: Vec<SimdLutGroup>,
}

impl ActivationLut {
    /// Dynamically quantizes `x` symmetrically to `i8` (`[-127, +127]`) and precomputes LUTs.
    pub fn from_f32(x: &[f32]) -> Self {
        let cols = x.len();
        let max_abs = x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let mut quantized_x = vec![0i8; cols];
        let act_scale = if max_abs > 1e-12 {
            let scale = max_abs / 127.0;
            let inv_scale = 127.0 / max_abs;
            for (dst, &src) in quantized_x.iter_mut().zip(x.iter()) {
                *dst = (src * inv_scale).round().clamp(-127.0, 127.0) as i8;
            }
            scale
        } else {
            0.0
        };
        Self::from_quantized_i8(quantized_x, act_scale)
    }

    /// Precomputes T-MAC lookup tables directly from an explicit `i8` activation vector.
    pub fn from_quantized_i8(quantized_x: Vec<i8>, act_scale: f32) -> Self {
        let cols = quantized_x.len();
        let num_groups = cols.div_ceil(TRITS_PER_BYTE);
        let mut groups = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let c0 = g * TRITS_PER_BYTE;
            let mut q = [0i8; TRITS_PER_BYTE];
            for m in 0..TRITS_PER_BYTE {
                if c0 + m < cols {
                    q[m] = quantized_x[c0 + m];
                }
            }
            groups.push(SimdLutGroup::from_group4(q));
        }
        Self {
            cols,
            act_scale,
            quantized_x,
            groups,
        }
    }

    /// Input dimension `K`.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Per-token symmetric quantization scale `gamma_x = max(|x|) / 127.0`.
    #[inline]
    pub fn act_scale(&self) -> f32 {
        self.act_scale
    }

    /// Quantized `i8` activation slice (`[-127, +127]`).
    #[inline]
    pub fn quantized_x(&self) -> &[i8] {
        &self.quantized_x
    }

    /// Precomputed 64-byte SIMD LUT groups (`ceil(K / 4)`).
    #[inline]
    pub fn groups(&self) -> &[SimdLutGroup] {
        &self.groups
    }

    /// Heap memory footprint of the precomputed lookup table in bytes.
    #[inline]
    pub fn lut_bytes(&self) -> usize {
        self.groups.len() * std::mem::size_of::<SimdLutGroup>() + self.quantized_x.len()
    }
}

/// Hardware execution backend for T-MAC table-lookup kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TmacKernelBackend {
    /// Automatically dispatches to the widest supported SIMD table-lookup kernel at runtime.
    #[default]
    Auto,
    /// Portable safe Rust `4x` output-row register-tiled LUT kernel.
    ScalarTiledLut,
    /// `x86_64` AVX2 `_mm256_shuffle_epi8` (`vpshufb`) 2-tile (`64`-row) LUT kernel.
    Avx2Lut,
    /// `x86_64` AVX-512-host 4-tile (`128`-row) unrolled `vpshufb` register-tiled kernel.
    Avx512WideLut,
    /// `aarch64` ARM NEON `vqtbl1q_s8` 16-byte table-lookup 2-tile (`32`-row) kernel.
    NeonLut,
}

impl TmacKernelBackend {
    /// Resolves [`TmacKernelBackend::Auto`] (or unavailable ISA requests) to a runnable backend.
    #[inline]
    pub fn resolve(self) -> Self {
        match self {
            Self::Auto => Self::detect_best(),
            Self::ScalarTiledLut => Self::ScalarTiledLut,
            Self::Avx2Lut => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return Self::Avx2Lut;
                    }
                }
                Self::ScalarTiledLut
            }
            Self::Avx512WideLut => {
                #[cfg(target_arch = "x86_64")]
                {
                    if std::arch::is_x86_feature_detected!("avx512bw")
                        && std::arch::is_x86_feature_detected!("avx2")
                    {
                        return Self::Avx512WideLut;
                    }
                    if std::arch::is_x86_feature_detected!("avx2") {
                        return Self::Avx2Lut;
                    }
                }
                Self::ScalarTiledLut
            }
            Self::NeonLut => {
                #[cfg(target_arch = "aarch64")]
                {
                    if std::arch::is_aarch64_feature_detected!("neon") {
                        return Self::NeonLut;
                    }
                }
                Self::ScalarTiledLut
            }
        }
    }

    /// Detects the highest-throughput SIMD LUT kernel supported by the current CPU.
    #[inline]
    pub fn detect_best() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx2")
            {
                return Self::Avx512WideLut;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Self::Avx2Lut;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::NeonLut;
            }
        }
        Self::ScalarTiledLut
    }

    /// Human-readable kernel tag recorded in scorecards.
    #[inline]
    pub fn name(self) -> &'static str {
        match self.resolve() {
            Self::ScalarTiledLut => "scalar-tiled4-lut",
            Self::Avx2Lut => "x86_64-avx2-vpshufb-tile64",
            Self::Avx512WideLut => "x86_64-avx512bw-vpshufb-tile128",
            Self::NeonLut => "aarch64-neon-vqtbl1q-tile32",
            Self::Auto => "auto",
        }
    }
}

/// Portable safe Rust `4x` output-row register-tiled LUT GEMV kernel (`0` multiplications).
fn scalar_tiled4_matvec_i32(
    weights: &PackedTernaryMatrix,
    lut: &ActivationLut,
    out: &mut [i32],
) {
    let groups = weights.groups_per_row;
    let packed = &weights.tiled_packed;
    let lut_groups = &lut.groups[..groups];

    for tile in 0..weights.num_tiles {
        let tile_offset = tile * groups * N_TILE;
        let mut tile_acc = [0i32; N_TILE];
        for sub in (0..N_TILE).step_by(4) {
            let mut acc0 = 0i32;
            let mut acc1 = 0i32;
            let mut acc2 = 0i32;
            let mut acc3 = 0i32;
            for (g, lg) in lut_groups.iter().enumerate() {
                let idx = tile_offset + g * N_TILE + sub;
                acc0 += lg.lookup_byte_i32(packed[idx]);
                acc1 += lg.lookup_byte_i32(packed[idx + 1]);
                acc2 += lg.lookup_byte_i32(packed[idx + 2]);
                acc3 += lg.lookup_byte_i32(packed[idx + 3]);
            }
            tile_acc[sub] = acc0;
            tile_acc[sub + 1] = acc1;
            tile_acc[sub + 2] = acc2;
            tile_acc[sub + 3] = acc3;
        }
        let row_start = tile * N_TILE;
        let valid = (weights.rows - row_start).min(N_TILE);
        out[row_start..row_start + valid].copy_from_slice(&tile_acc[..valid]);
    }
}

#[cfg(target_arch = "x86_64")]
mod x86_kernels {
    use super::{ActivationLut, PackedTernaryMatrix, SimdLutGroup, I16_ACCUM_GROUP_CHUNK, N_TILE};
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn eval_one_tile_avx2(
        tile_ptr: *const u8,
        lut_groups: &[SimdLutGroup],
        dst32: &mut [i32; N_TILE],
    ) {
        let mask_0f = _mm256_set1_epi8(0x0F);
        let mut acc_i32_0 = _mm256_setzero_si256();
        let mut acc_i32_1 = _mm256_setzero_si256();
        let mut acc_i32_2 = _mm256_setzero_si256();
        let mut acc_i32_3 = _mm256_setzero_si256();

        let total_groups = lut_groups.len();
        let mut g_start = 0usize;
        while g_start < total_groups {
            let g_end = (g_start + I16_ACCUM_GROUP_CHUNK).min(total_groups);
            let mut acc_lo_i16 = _mm256_setzero_si256();
            let mut acc_hi_i16 = _mm256_setzero_si256();

            for (g, lg) in lut_groups[g_start..g_end].iter().enumerate() {
                let g_abs = g_start + g;
                let wb = _mm256_loadu_si256(tile_ptr.add(g_abs * N_TILE).cast::<__m256i>());
                let idx_lo = _mm256_and_si256(wb, mask_0f);
                let idx_hi = _mm256_and_si256(_mm256_srli_epi16(wb, 4), mask_0f);

                let tbl_lo_l = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128(lg.lo_lsb.as_ptr().cast::<__m128i>()),
                );
                let tbl_lo_m = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128(lg.lo_msb.as_ptr().cast::<__m128i>()),
                );
                let lo_l = _mm256_shuffle_epi8(tbl_lo_l, idx_lo);
                let lo_m = _mm256_shuffle_epi8(tbl_lo_m, idx_lo);
                acc_lo_i16 = _mm256_add_epi16(acc_lo_i16, _mm256_unpacklo_epi8(lo_l, lo_m));
                acc_hi_i16 = _mm256_add_epi16(acc_hi_i16, _mm256_unpackhi_epi8(lo_l, lo_m));

                let tbl_hi_l = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128(lg.hi_lsb.as_ptr().cast::<__m128i>()),
                );
                let tbl_hi_m = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128(lg.hi_msb.as_ptr().cast::<__m128i>()),
                );
                let hi_l = _mm256_shuffle_epi8(tbl_hi_l, idx_hi);
                let hi_m = _mm256_shuffle_epi8(tbl_hi_m, idx_hi);
                acc_lo_i16 = _mm256_add_epi16(acc_lo_i16, _mm256_unpacklo_epi8(hi_l, hi_m));
                acc_hi_i16 = _mm256_add_epi16(acc_hi_i16, _mm256_unpackhi_epi8(hi_l, hi_m));
            }

            let lo_lane0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_lo_i16));
            let hi_lane0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(acc_hi_i16));
            let lo_lane1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_lo_i16, 1));
            let hi_lane1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(acc_hi_i16, 1));
            acc_i32_0 = _mm256_add_epi32(acc_i32_0, lo_lane0);
            acc_i32_1 = _mm256_add_epi32(acc_i32_1, hi_lane0);
            acc_i32_2 = _mm256_add_epi32(acc_i32_2, lo_lane1);
            acc_i32_3 = _mm256_add_epi32(acc_i32_3, hi_lane1);

            g_start = g_end;
        }

        let out_ptr = dst32.as_mut_ptr();
        _mm256_storeu_si256(out_ptr.cast::<__m256i>(), acc_i32_0);
        _mm256_storeu_si256(out_ptr.add(8).cast::<__m256i>(), acc_i32_1);
        _mm256_storeu_si256(out_ptr.add(16).cast::<__m256i>(), acc_i32_2);
        _mm256_storeu_si256(out_ptr.add(24).cast::<__m256i>(), acc_i32_3);
    }

    /// 2-tile (`64` output rows simultaneously) register-tiled AVX2 `vpshufb` LUT kernel.
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx2_matvec_i32(
        weights: &PackedTernaryMatrix,
        lut: &ActivationLut,
        out: &mut [i32],
    ) {
        let groups = weights.groups_per_row;
        let lut_groups = &lut.groups[..groups];
        let tile_stride = groups * N_TILE;
        let base_ptr = weights.tiled_packed.as_ptr();
        let mask_0f = _mm256_set1_epi8(0x0F);

        let mut tile = 0usize;
        while tile + 2 <= weights.num_tiles {
            let p0 = base_ptr.add(tile * tile_stride);
            let p1 = base_ptr.add((tile + 1) * tile_stride);

            let mut a0_0 = _mm256_setzero_si256();
            let mut a0_1 = _mm256_setzero_si256();
            let mut a0_2 = _mm256_setzero_si256();
            let mut a0_3 = _mm256_setzero_si256();
            let mut a1_0 = _mm256_setzero_si256();
            let mut a1_1 = _mm256_setzero_si256();
            let mut a1_2 = _mm256_setzero_si256();
            let mut a1_3 = _mm256_setzero_si256();

            let mut g_start = 0usize;
            while g_start < groups {
                let g_end = (g_start + I16_ACCUM_GROUP_CHUNK).min(groups);
                let mut t0_lo = _mm256_setzero_si256();
                let mut t0_hi = _mm256_setzero_si256();
                let mut t1_lo = _mm256_setzero_si256();
                let mut t1_hi = _mm256_setzero_si256();

                for (g, lg) in lut_groups[g_start..g_end].iter().enumerate() {
                    let byte_off = (g_start + g) * N_TILE;
                    let tbl_lo_l = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.lo_lsb.as_ptr().cast::<__m128i>()),
                    );
                    let tbl_lo_m = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.lo_msb.as_ptr().cast::<__m128i>()),
                    );
                    let tbl_hi_l = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.hi_lsb.as_ptr().cast::<__m128i>()),
                    );
                    let tbl_hi_m = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.hi_msb.as_ptr().cast::<__m128i>()),
                    );

                    let wb0 = _mm256_loadu_si256(p0.add(byte_off).cast::<__m256i>());
                    let i0_lo = _mm256_and_si256(wb0, mask_0f);
                    let i0_hi = _mm256_and_si256(_mm256_srli_epi16(wb0, 4), mask_0f);
                    let l0_l = _mm256_shuffle_epi8(tbl_lo_l, i0_lo);
                    let l0_m = _mm256_shuffle_epi8(tbl_lo_m, i0_lo);
                    let h0_l = _mm256_shuffle_epi8(tbl_hi_l, i0_hi);
                    let h0_m = _mm256_shuffle_epi8(tbl_hi_m, i0_hi);
                    t0_lo = _mm256_add_epi16(t0_lo, _mm256_unpacklo_epi8(l0_l, l0_m));
                    t0_hi = _mm256_add_epi16(t0_hi, _mm256_unpackhi_epi8(l0_l, l0_m));
                    t0_lo = _mm256_add_epi16(t0_lo, _mm256_unpacklo_epi8(h0_l, h0_m));
                    t0_hi = _mm256_add_epi16(t0_hi, _mm256_unpackhi_epi8(h0_l, h0_m));

                    let wb1 = _mm256_loadu_si256(p1.add(byte_off).cast::<__m256i>());
                    let i1_lo = _mm256_and_si256(wb1, mask_0f);
                    let i1_hi = _mm256_and_si256(_mm256_srli_epi16(wb1, 4), mask_0f);
                    let l1_l = _mm256_shuffle_epi8(tbl_lo_l, i1_lo);
                    let l1_m = _mm256_shuffle_epi8(tbl_lo_m, i1_lo);
                    let h1_l = _mm256_shuffle_epi8(tbl_hi_l, i1_hi);
                    let h1_m = _mm256_shuffle_epi8(tbl_hi_m, i1_hi);
                    t1_lo = _mm256_add_epi16(t1_lo, _mm256_unpacklo_epi8(l1_l, l1_m));
                    t1_hi = _mm256_add_epi16(t1_hi, _mm256_unpackhi_epi8(l1_l, l1_m));
                    t1_lo = _mm256_add_epi16(t1_lo, _mm256_unpacklo_epi8(h1_l, h1_m));
                    t1_hi = _mm256_add_epi16(t1_hi, _mm256_unpackhi_epi8(h1_l, h1_m));
                }

                a0_0 = _mm256_add_epi32(
                    a0_0,
                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t0_lo)),
                );
                a0_1 = _mm256_add_epi32(
                    a0_1,
                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t0_hi)),
                );
                a0_2 = _mm256_add_epi32(
                    a0_2,
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t0_lo, 1)),
                );
                a0_3 = _mm256_add_epi32(
                    a0_3,
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t0_hi, 1)),
                );

                a1_0 = _mm256_add_epi32(
                    a1_0,
                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t1_lo)),
                );
                a1_1 = _mm256_add_epi32(
                    a1_1,
                    _mm256_cvtepi16_epi32(_mm256_castsi256_si128(t1_hi)),
                );
                a1_2 = _mm256_add_epi32(
                    a1_2,
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t1_lo, 1)),
                );
                a1_3 = _mm256_add_epi32(
                    a1_3,
                    _mm256_cvtepi16_epi32(_mm256_extracti128_si256(t1_hi, 1)),
                );

                g_start = g_end;
            }

            let mut buf0 = [0i32; N_TILE];
            let mut buf1 = [0i32; N_TILE];
            _mm256_storeu_si256(buf0.as_mut_ptr().cast::<__m256i>(), a0_0);
            _mm256_storeu_si256(buf0.as_mut_ptr().add(8).cast::<__m256i>(), a0_1);
            _mm256_storeu_si256(buf0.as_mut_ptr().add(16).cast::<__m256i>(), a0_2);
            _mm256_storeu_si256(buf0.as_mut_ptr().add(24).cast::<__m256i>(), a0_3);
            _mm256_storeu_si256(buf1.as_mut_ptr().cast::<__m256i>(), a1_0);
            _mm256_storeu_si256(buf1.as_mut_ptr().add(8).cast::<__m256i>(), a1_1);
            _mm256_storeu_si256(buf1.as_mut_ptr().add(16).cast::<__m256i>(), a1_2);
            _mm256_storeu_si256(buf1.as_mut_ptr().add(24).cast::<__m256i>(), a1_3);

            let r0 = tile * N_TILE;
            out[r0..r0 + N_TILE].copy_from_slice(&buf0);
            let r1 = (tile + 1) * N_TILE;
            let valid1 = (weights.rows - r1).min(N_TILE);
            out[r1..r1 + valid1].copy_from_slice(&buf1[..valid1]);
            tile += 2;
        }

        while tile < weights.num_tiles {
            let mut buf = [0i32; N_TILE];
            eval_one_tile_avx2(base_ptr.add(tile * tile_stride), lut_groups, &mut buf);
            let r0 = tile * N_TILE;
            let valid = (weights.rows - r0).min(N_TILE);
            out[r0..r0 + valid].copy_from_slice(&buf[..valid]);
            tile += 1;
        }
    }

    /// 4-tile (`128` output rows simultaneously) wide register-tiled `vpshufb` kernel
    /// dispatched on AVX-512-capable `x86_64` hosts.
    #[target_feature(enable = "avx2")]
    pub unsafe fn avx512_wide_matvec_i32(
        weights: &PackedTernaryMatrix,
        lut: &ActivationLut,
        out: &mut [i32],
    ) {
        let groups = weights.groups_per_row;
        let lut_groups = &lut.groups[..groups];
        let tile_stride = groups * N_TILE;
        let base_ptr = weights.tiled_packed.as_ptr();
        let mask_0f = _mm256_set1_epi8(0x0F);

        let mut tile = 0usize;
        while tile + 4 <= weights.num_tiles {
            let mut tile_i32 = [[0i32; N_TILE]; 4];
            let mut g_start = 0usize;
            while g_start < groups {
                let g_end = (g_start + I16_ACCUM_GROUP_CHUNK).min(groups);
                let mut lo_acc = [_mm256_setzero_si256(); 4];
                let mut hi_acc = [_mm256_setzero_si256(); 4];

                for (g, lg) in lut_groups[g_start..g_end].iter().enumerate() {
                    let byte_off = (g_start + g) * N_TILE;
                    let tbl_lo_l = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.lo_lsb.as_ptr().cast::<__m128i>()),
                    );
                    let tbl_lo_m = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.lo_msb.as_ptr().cast::<__m128i>()),
                    );
                    let tbl_hi_l = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.hi_lsb.as_ptr().cast::<__m128i>()),
                    );
                    let tbl_hi_m = _mm256_broadcastsi128_si256(
                        _mm_loadu_si128(lg.hi_msb.as_ptr().cast::<__m128i>()),
                    );

                    for sub in 0..4usize {
                        let ptr = base_ptr.add((tile + sub) * tile_stride + byte_off);
                        let wb = _mm256_loadu_si256(ptr.cast::<__m256i>());
                        let idx_lo = _mm256_and_si256(wb, mask_0f);
                        let idx_hi = _mm256_and_si256(_mm256_srli_epi16(wb, 4), mask_0f);
                        let ll = _mm256_shuffle_epi8(tbl_lo_l, idx_lo);
                        let lm = _mm256_shuffle_epi8(tbl_lo_m, idx_lo);
                        let hl = _mm256_shuffle_epi8(tbl_hi_l, idx_hi);
                        let hm = _mm256_shuffle_epi8(tbl_hi_m, idx_hi);
                        lo_acc[sub] =
                            _mm256_add_epi16(lo_acc[sub], _mm256_unpacklo_epi8(ll, lm));
                        hi_acc[sub] =
                            _mm256_add_epi16(hi_acc[sub], _mm256_unpackhi_epi8(ll, lm));
                        lo_acc[sub] =
                            _mm256_add_epi16(lo_acc[sub], _mm256_unpacklo_epi8(hl, hm));
                        hi_acc[sub] =
                            _mm256_add_epi16(hi_acc[sub], _mm256_unpackhi_epi8(hl, hm));
                    }
                }

                for sub in 0..4usize {
                    let dst_ptr = tile_i32[sub].as_mut_ptr();
                    let cur0 = _mm256_loadu_si256(dst_ptr.cast::<__m256i>());
                    let cur1 = _mm256_loadu_si256(dst_ptr.add(8).cast::<__m256i>());
                    let cur2 = _mm256_loadu_si256(dst_ptr.add(16).cast::<__m256i>());
                    let cur3 = _mm256_loadu_si256(dst_ptr.add(24).cast::<__m256i>());
                    let w0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_acc[sub]));
                    let w1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_acc[sub]));
                    let w2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo_acc[sub], 1));
                    let w3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi_acc[sub], 1));
                    _mm256_storeu_si256(dst_ptr.cast::<__m256i>(), _mm256_add_epi32(cur0, w0));
                    _mm256_storeu_si256(
                        dst_ptr.add(8).cast::<__m256i>(),
                        _mm256_add_epi32(cur1, w1),
                    );
                    _mm256_storeu_si256(
                        dst_ptr.add(16).cast::<__m256i>(),
                        _mm256_add_epi32(cur2, w2),
                    );
                    _mm256_storeu_si256(
                        dst_ptr.add(24).cast::<__m256i>(),
                        _mm256_add_epi32(cur3, w3),
                    );
                }

                g_start = g_end;
            }

            for sub in 0..4usize {
                let r0 = (tile + sub) * N_TILE;
                let valid = (weights.rows - r0).min(N_TILE);
                out[r0..r0 + valid].copy_from_slice(&tile_i32[sub][..valid]);
            }
            tile += 4;
        }

        while tile < weights.num_tiles {
            let mut buf = [0i32; N_TILE];
            eval_one_tile_avx2(base_ptr.add(tile * tile_stride), lut_groups, &mut buf);
            let r0 = tile * N_TILE;
            let valid = (weights.rows - r0).min(N_TILE);
            out[r0..r0 + valid].copy_from_slice(&buf[..valid]);
            tile += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_kernels {
    use super::{ActivationLut, PackedTernaryMatrix, I16_ACCUM_GROUP_CHUNK, N_TILE};
    use std::arch::aarch64::*;

    /// ARM NEON `vqtbl1q_s8` 16-byte register table-lookup kernel (`32` rows per tile).
    #[target_feature(enable = "neon")]
    pub unsafe fn neon_matvec_i32(
        weights: &PackedTernaryMatrix,
        lut: &ActivationLut,
        out: &mut [i32],
    ) {
        let groups = weights.groups_per_row;
        let lut_groups = &lut.groups[..groups];
        let tile_stride = groups * N_TILE;
        let base_ptr = weights.tiled_packed.as_ptr();
        let mask_0f = vdupq_n_u8(0x0F);

        for tile in 0..weights.num_tiles {
            let tile_ptr = base_ptr.add(tile * tile_stride);
            let mut acc_i32 = [vdupq_n_s32(0); 8];

            let mut g_start = 0usize;
            while g_start < groups {
                let g_end = (g_start + I16_ACCUM_GROUP_CHUNK).min(groups);
                let mut h0_lo = vdupq_n_s16(0);
                let mut h0_hi = vdupq_n_s16(0);
                let mut h1_lo = vdupq_n_s16(0);
                let mut h1_hi = vdupq_n_s16(0);

                for (g, lg) in lut_groups[g_start..g_end].iter().enumerate() {
                    let p = tile_ptr.add((g_start + g) * N_TILE);
                    let tbl_lo_l = vld1q_s8(lg.lo_lsb.as_ptr().cast::<i8>());
                    let tbl_lo_m = vld1q_s8(lg.lo_msb.as_ptr().cast::<i8>());
                    let tbl_hi_l = vld1q_s8(lg.hi_lsb.as_ptr().cast::<i8>());
                    let tbl_hi_m = vld1q_s8(lg.hi_msb.as_ptr().cast::<i8>());

                    let wb0 = vld1q_u8(p);
                    let i0_lo = vandq_u8(wb0, mask_0f);
                    let i0_hi = vshrq_n_u8::<4>(wb0);
                    let l0_l = vqtbl1q_s8(tbl_lo_l, i0_lo);
                    let l0_m = vqtbl1q_s8(tbl_lo_m, i0_lo);
                    let h0_l = vqtbl1q_s8(tbl_hi_l, i0_hi);
                    let h0_m = vqtbl1q_s8(tbl_hi_m, i0_hi);
                    h0_lo = vaddq_s16(h0_lo, vreinterpretq_s16_s8(vzip1q_s8(l0_l, l0_m)));
                    h0_hi = vaddq_s16(h0_hi, vreinterpretq_s16_s8(vzip2q_s8(l0_l, l0_m)));
                    h0_lo = vaddq_s16(h0_lo, vreinterpretq_s16_s8(vzip1q_s8(h0_l, h0_m)));
                    h0_hi = vaddq_s16(h0_hi, vreinterpretq_s16_s8(vzip2q_s8(h0_l, h0_m)));

                    let wb1 = vld1q_u8(p.add(16));
                    let i1_lo = vandq_u8(wb1, mask_0f);
                    let i1_hi = vshrq_n_u8::<4>(wb1);
                    let l1_l = vqtbl1q_s8(tbl_lo_l, i1_lo);
                    let l1_m = vqtbl1q_s8(tbl_lo_m, i1_lo);
                    let h1_l = vqtbl1q_s8(tbl_hi_l, i1_hi);
                    let h1_m = vqtbl1q_s8(tbl_hi_m, i1_hi);
                    h1_lo = vaddq_s16(h1_lo, vreinterpretq_s16_s8(vzip1q_s8(l1_l, l1_m)));
                    h1_hi = vaddq_s16(h1_hi, vreinterpretq_s16_s8(vzip2q_s8(l1_l, l1_m)));
                    h1_lo = vaddq_s16(h1_lo, vreinterpretq_s16_s8(vzip1q_s8(h1_l, h1_m)));
                    h1_hi = vaddq_s16(h1_hi, vreinterpretq_s16_s8(vzip2q_s8(h1_l, h1_m)));
                }

                acc_i32[0] = vaddw_s16(acc_i32[0], vget_low_s16(h0_lo));
                acc_i32[1] = vaddw_s16(acc_i32[1], vget_high_s16(h0_lo));
                acc_i32[2] = vaddw_s16(acc_i32[2], vget_low_s16(h0_hi));
                acc_i32[3] = vaddw_s16(acc_i32[3], vget_high_s16(h0_hi));
                acc_i32[4] = vaddw_s16(acc_i32[4], vget_low_s16(h1_lo));
                acc_i32[5] = vaddw_s16(acc_i32[5], vget_high_s16(h1_lo));
                acc_i32[6] = vaddw_s16(acc_i32[6], vget_low_s16(h1_hi));
                acc_i32[7] = vaddw_s16(acc_i32[7], vget_high_s16(h1_hi));

                g_start = g_end;
            }

            let mut buf = [0i32; N_TILE];
            for q_idx in 0..8usize {
                vst1q_s32(buf.as_mut_ptr().add(q_idx * 4), acc_i32[q_idx]);
            }
            let r0 = tile * N_TILE;
            let valid = (weights.rows - r0).min(N_TILE);
            out[r0..r0 + valid].copy_from_slice(&buf[..valid]);
        }
    }
}

/// Computes exact `i32` ternary dot products `y_int = W_ternary @ q_x` prior to `f32` scaling.
pub fn ternary_lut_matvec_i32(
    weights: &PackedTernaryMatrix,
    lut: &ActivationLut,
    backend: TmacKernelBackend,
    out: &mut [i32],
) -> Result<()> {
    if lut.cols() != weights.cols() {
        hanzo_ml::bail!(
            "ternary_lut_matvec_i32 input cols {} != weight cols {}",
            lut.cols(),
            weights.cols()
        );
    }
    if out.len() != weights.rows() {
        hanzo_ml::bail!(
            "ternary_lut_matvec_i32 output len {} != weight rows {}",
            out.len(),
            weights.rows()
        );
    }

    match backend.resolve() {
        TmacKernelBackend::ScalarTiledLut | TmacKernelBackend::Auto => {
            scalar_tiled4_matvec_i32(weights, lut, out);
        }
        TmacKernelBackend::Avx2Lut => {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: `resolve()` verified runtime `avx2` support and slice bounds.
                unsafe { x86_kernels::avx2_matvec_i32(weights, lut, out) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                scalar_tiled4_matvec_i32(weights, lut, out);
            }
        }
        TmacKernelBackend::Avx512WideLut => {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: `resolve()` verified runtime `avx2` + `avx512bw` support.
                unsafe { x86_kernels::avx512_wide_matvec_i32(weights, lut, out) };
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                scalar_tiled4_matvec_i32(weights, lut, out);
            }
        }
        TmacKernelBackend::NeonLut => {
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: `resolve()` verified runtime `neon` support.
                unsafe { arm_kernels::neon_matvec_i32(weights, lut, out) };
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                scalar_tiled4_matvec_i32(weights, lut, out);
            }
        }
    }
    Ok(())
}

/// Computes `y = (W_ternary @ q_x) * (alpha_row * act_scale) + bias` via T-MAC LUT lookup.
pub fn ternary_lut_matvec_f32(
    weights: &PackedTernaryMatrix,
    x: &[f32],
    backend: TmacKernelBackend,
    out: &mut [f32],
) -> Result<()> {
    if x.len() != weights.cols() || out.len() != weights.rows() {
        hanzo_ml::bail!(
            "ternary_lut_matvec_f32 shape mismatch x={} out={} vs ({}, {})",
            x.len(),
            out.len(),
            weights.rows(),
            weights.cols()
        );
    }
    let lut = ActivationLut::from_f32(x);
    let mut acc_i32 = vec![0i32; weights.rows()];
    ternary_lut_matvec_i32(weights, &lut, backend, &mut acc_i32)?;
    let act_scale = f64::from(lut.act_scale());
    let scales = weights.row_scales();
    match weights.bias() {
        Some(bias) => {
            for r in 0..weights.rows() {
                let scaled = (f64::from(acc_i32[r]) * f64::from(scales[r]) * act_scale) as f32;
                out[r] = scaled + bias[r];
            }
        }
        None => {
            for r in 0..weights.rows() {
                out[r] = (f64::from(acc_i32[r]) * f64::from(scales[r]) * act_scale) as f32;
            }
        }
    }
    Ok(())
}

/// Reference baseline that unpacks 2-bit trits on every row and multiplies on the fly.
pub fn weight_unpack_deq_matvec_f32(
    weights: &PackedTernaryMatrix,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    if x.len() != weights.cols() || out.len() != weights.rows() {
        hanzo_ml::bail!("weight_unpack_deq_matvec_f32 shape mismatch");
    }
    let lut = ActivationLut::from_f32(x);
    let qx = lut.quantized_x();
    let act_scale = f64::from(lut.act_scale());
    let scales = weights.row_scales();
    let cols = weights.cols();
    let groups = weights.groups_per_row();
    for r in 0..weights.rows() {
        let mut acc = 0i32;
        for g in 0..groups {
            let trits = unpack_trit_group4(weights.packed_byte(r, g));
            let c0 = g * TRITS_PER_BYTE;
            for m in 0..TRITS_PER_BYTE {
                if c0 + m < cols {
                    acc += i32::from(trits[m]) * i32::from(qx[c0 + m]);
                }
            }
        }
        let bias = weights.bias().map_or(0.0, |b| b[r]);
        let scaled = (f64::from(acc) * f64::from(scales[r]) * act_scale) as f32;
        out[r] = scaled + bias;
    }
    Ok(())
}

/// Dense `f32` CPU reference baseline (`W1.58A8` BitLinear executed with expanded `f32` weights
/// and `sdsdot`-style `f64` inner accumulation).
pub fn fp32_reference_matvec(
    dense_weights: &[f32],
    bias: Option<&[f32]>,
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    if dense_weights.len() != rows * cols || x.len() != cols || out.len() != rows {
        hanzo_ml::bail!("fp32_reference_matvec shape mismatch");
    }
    let lut = ActivationLut::from_f32(x);
    let qx = lut.quantized_x();
    let act_scale = f64::from(lut.act_scale());
    let x_deq: Vec<f64> = qx.iter().map(|&q| f64::from(q) * act_scale).collect();
    for r in 0..rows {
        let row_w = &dense_weights[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for c in 0..cols {
            sum += f64::from(row_w[c]) * x_deq[c];
        }
        out[r] = (sum as f32) + bias.map_or(0.0, |b| b[r]);
    }
    Ok(())
}

/// Drop-in `hanzo_ml::Module` linear layer backed by T-MAC packed ternary lookup tables.
#[derive(Clone, Debug)]
pub struct TmacLinear {
    weights: PackedTernaryMatrix,
    backend: TmacKernelBackend,
}

impl TmacLinear {
    /// Wraps a [`PackedTernaryMatrix`] with the chosen [`TmacKernelBackend`].
    pub fn new(weights: PackedTernaryMatrix, backend: TmacKernelBackend) -> Self {
        Self { weights, backend }
    }

    /// Quantizes `weight` (`[out_dim, in_dim]`) via BitNet b1.58 absmean scaling.
    pub fn from_tensor(
        weight: &Tensor,
        bias: Option<&Tensor>,
        backend: TmacKernelBackend,
    ) -> Result<Self> {
        let packed = PackedTernaryMatrix::from_tensor_absmean(weight, bias)?;
        Ok(Self::new(packed, backend))
    }

    /// Underlying packed ternary matrix.
    #[inline]
    pub fn packed_weights(&self) -> &PackedTernaryMatrix {
        &self.weights
    }

    /// Configured execution backend.
    #[inline]
    pub fn backend(&self) -> TmacKernelBackend {
        self.backend
    }
}

impl Module for TmacLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let orig_dtype = xs.dtype();
        let dev = xs.device();
        let dims = xs.dims().to_vec();
        if dims.is_empty() {
            hanzo_ml::bail!("TmacLinear requires at least 1D input tensor");
        }
        let k = *dims.last().unwrap();
        if k != self.weights.cols() {
            hanzo_ml::bail!(
                "TmacLinear input last dim {k} != weight cols {}",
                self.weights.cols()
            );
        }
        let m: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
        let flat = xs.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let n = self.weights.rows();
        let mut out = vec![0.0f32; m * n];
        for row_idx in 0..m {
            let x_slice = &flat[row_idx * k..(row_idx + 1) * k];
            let y_slice = &mut out[row_idx * n..(row_idx + 1) * n];
            ternary_lut_matvec_f32(&self.weights, x_slice, self.backend, y_slice)?;
        }
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);
        Tensor::from_vec(out, out_dims, dev)?.to_dtype(orig_dtype)
    }
}

/// Operator family compared in the end-to-end network scorecard (#40).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorKind {
    Fp32Reference,
    WeightUnpackDeqReference,
    TmacLutOperator,
}

impl OperatorKind {
    /// Canonical scorecard name.
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32Reference => "Fp32Reference",
            Self::WeightUnpackDeqReference => "WeightUnpackDeqReference",
            Self::TmacLutOperator => "TmacLutOperator",
        }
    }
}

#[derive(Clone, Debug)]
struct DualProjection {
    packed: PackedTernaryMatrix,
    dense_f32: Option<Vec<f32>>,
}

impl DualProjection {
    fn from_seeded_f32(w: &[f32], rows: usize, cols: usize, keep_dense: bool) -> Result<Self> {
        let packed = PackedTernaryMatrix::from_f32_absmean(w, rows, cols, None)?;
        let dense_f32 = if keep_dense {
            Some(packed.dequantize_f32())
        } else {
            None
        };
        Ok(Self { packed, dense_f32 })
    }

    fn apply(
        &self,
        x: &[f32],
        op: OperatorKind,
        backend: TmacKernelBackend,
        out: &mut [f32],
    ) -> Result<()> {
        match op {
            OperatorKind::Fp32Reference => {
                let dense = self.dense_f32.as_deref().ok_or_else(|| {
                    hanzo_ml::Error::Msg("Fp32Reference requires dense weights".into())
                })?;
                fp32_reference_matvec(
                    dense,
                    self.packed.bias(),
                    self.packed.rows(),
                    self.packed.cols(),
                    x,
                    out,
                )
            }
            OperatorKind::WeightUnpackDeqReference => {
                weight_unpack_deq_matvec_f32(&self.packed, x, out)
            }
            OperatorKind::TmacLutOperator => {
                ternary_lut_matvec_f32(&self.packed, x, backend, out)
            }
        }
    }

    fn weight_bytes(&self, op: OperatorKind) -> usize {
        match op {
            OperatorKind::Fp32Reference => self.packed.rows() * self.packed.cols() * 4,
            OperatorKind::WeightUnpackDeqReference | OperatorKind::TmacLutOperator => {
                self.packed.allocated_bytes()
            }
        }
    }
}

/// Deterministic SplitMix64 generator used to build reproducible benchmark weights and prompts.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32_signed(&mut self, scale: f32) -> f32 {
        let unit = ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32);
        (unit * 2.0 - 1.0) * scale
    }

    fn vec_f32(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32_signed(scale)).collect()
    }
}

#[inline]
fn rms_norm_vec(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[inline]
fn apply_rope_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, pos: usize) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = (pos as f32) / 10000.0f32.powf((2 * i) as f32 / (head_dim as f32));
            let (sin_f, cos_f) = freq.sin_cos();
            let x0 = vec[base + i];
            let x1 = vec[base + half + i];
            vec[base + i] = x0 * cos_f - x1 * sin_f;
            vec[base + half + i] = x0 * sin_f + x1 * cos_f;
        }
    }
}

/// Pre-allocated contiguous KV cache for multi-turn causal coding-agent evaluation.
#[derive(Clone, Debug)]
pub struct TernaryKvCache {
    num_layers: usize,
    max_seq_len: usize,
    kv_dim: usize,
    seq_len: usize,
    k_buf: Vec<Vec<f32>>,
    v_buf: Vec<Vec<f32>>,
}

impl TernaryKvCache {
    /// Allocates an empty KV cache for `num_layers` up to `max_seq_len` tokens.
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        Self {
            num_layers,
            max_seq_len,
            kv_dim,
            seq_len: 0,
            k_buf: vec![vec![0.0f32; max_seq_len * kv_dim]; num_layers],
            v_buf: vec![vec![0.0f32; max_seq_len * kv_dim]; num_layers],
        }
    }

    /// Resets cached sequence offset without reallocating backing buffers.
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Current cached token count.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Allocated KV cache memory in bytes.
    #[inline]
    pub fn allocated_bytes(&self) -> usize {
        self.num_layers * self.max_seq_len * self.kv_dim * 2 * 4
    }
}

/// Complete ternary transformer block (`RMSNorm` + GQA Ternary `Q/K/V/O` + `RoPE` + SwiGLU MLP).
#[derive(Clone, Debug)]
pub struct TernaryNetworkBlock {
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_norm: Vec<f32>,
    q_proj: DualProjection,
    k_proj: DualProjection,
    v_proj: DualProjection,
    o_proj: DualProjection,
    ffn_norm: Vec<f32>,
    gate_proj: DualProjection,
    up_proj: DualProjection,
    down_proj: DualProjection,
}

impl TernaryNetworkBlock {
    fn from_rng(
        hidden_dim: usize,
        ffn_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        keep_dense: bool,
        rng: &mut SplitMix64,
    ) -> Result<Self> {
        if hidden_dim == 0 || num_heads == 0 || num_kv_heads == 0 {
            hanzo_ml::bail!("Invalid TernaryNetworkBlock head configuration");
        }
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        let q_w = rng.vec_f32(hidden_dim * hidden_dim, 0.08);
        let k_w = rng.vec_f32(kv_dim * hidden_dim, 0.08);
        let v_w = rng.vec_f32(kv_dim * hidden_dim, 0.08);
        let o_w = rng.vec_f32(hidden_dim * hidden_dim, 0.08);
        let gate_w = rng.vec_f32(ffn_dim * hidden_dim, 0.08);
        let up_w = rng.vec_f32(ffn_dim * hidden_dim, 0.08);
        let down_w = rng.vec_f32(hidden_dim * ffn_dim, 0.08);

        Ok(Self {
            hidden_dim,
            ffn_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            attn_norm: vec![1.0f32; hidden_dim],
            q_proj: DualProjection::from_seeded_f32(&q_w, hidden_dim, hidden_dim, keep_dense)?,
            k_proj: DualProjection::from_seeded_f32(&k_w, kv_dim, hidden_dim, keep_dense)?,
            v_proj: DualProjection::from_seeded_f32(&v_w, kv_dim, hidden_dim, keep_dense)?,
            o_proj: DualProjection::from_seeded_f32(&o_w, hidden_dim, hidden_dim, keep_dense)?,
            ffn_norm: vec![1.0f32; hidden_dim],
            gate_proj: DualProjection::from_seeded_f32(&gate_w, ffn_dim, hidden_dim, keep_dense)?,
            up_proj: DualProjection::from_seeded_f32(&up_w, ffn_dim, hidden_dim, keep_dense)?,
            down_proj: DualProjection::from_seeded_f32(&down_w, hidden_dim, ffn_dim, keep_dense)?,
        })
    }

    fn forward_step(
        &self,
        layer_idx: usize,
        pos: usize,
        h: &mut [f32],
        kv_cache: &mut TernaryKvCache,
        op: OperatorKind,
        backend: TmacKernelBackend,
    ) -> Result<()> {
        let kv_dim = self.num_kv_heads * self.head_dim;
        let mut x_norm = vec![0.0f32; self.hidden_dim];
        rms_norm_vec(h, &self.attn_norm, 1e-5, &mut x_norm);

        let mut q = vec![0.0f32; self.hidden_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        self.q_proj.apply(&x_norm, op, backend, &mut q)?;
        self.k_proj.apply(&x_norm, op, backend, &mut k)?;
        self.v_proj.apply(&x_norm, op, backend, &mut v)?;

        apply_rope_inplace(&mut q, self.num_heads, self.head_dim, pos);
        apply_rope_inplace(&mut k, self.num_kv_heads, self.head_dim, pos);

        let cache_off = pos * kv_dim;
        kv_cache.k_buf[layer_idx][cache_off..cache_off + kv_dim].copy_from_slice(&k);
        kv_cache.v_buf[layer_idx][cache_off..cache_off + kv_dim].copy_from_slice(&v);

        let seq_total = pos + 1;
        let kv_group = self.num_heads / self.num_kv_heads;
        let inv_sqrt_d = 1.0f32 / (self.head_dim as f32).sqrt();
        let mut attn_concat = vec![0.0f32; self.hidden_dim];
        let mut scores = vec![0.0f32; seq_total];

        for head in 0..self.num_heads {
            let kv_head = head / kv_group;
            let q_slice = &q[head * self.head_dim..(head + 1) * self.head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for t in 0..seq_total {
                let k_base = t * kv_dim + kv_head * self.head_dim;
                let k_slice = &kv_cache.k_buf[layer_idx][k_base..k_base + self.head_dim];
                let mut dot = 0.0f32;
                for d in 0..self.head_dim {
                    dot += q_slice[d] * k_slice[d];
                }
                let s = dot * inv_sqrt_d;
                scores[t] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum_exp += *s;
            }
            let inv_sum = 1.0f32 / sum_exp;
            let out_head = &mut attn_concat[head * self.head_dim..(head + 1) * self.head_dim];
            for t in 0..seq_total {
                let prob = scores[t] * inv_sum;
                let v_base = t * kv_dim + kv_head * self.head_dim;
                let v_slice = &kv_cache.v_buf[layer_idx][v_base..v_base + self.head_dim];
                for d in 0..self.head_dim {
                    out_head[d] += prob * v_slice[d];
                }
            }
        }

        let mut attn_proj = vec![0.0f32; self.hidden_dim];
        self.o_proj.apply(&attn_concat, op, backend, &mut attn_proj)?;
        for i in 0..self.hidden_dim {
            h[i] += attn_proj[i];
        }

        rms_norm_vec(h, &self.ffn_norm, 1e-5, &mut x_norm);
        let mut gate = vec![0.0f32; self.ffn_dim];
        let mut up = vec![0.0f32; self.ffn_dim];
        self.gate_proj.apply(&x_norm, op, backend, &mut gate)?;
        self.up_proj.apply(&x_norm, op, backend, &mut up)?;
        for i in 0..self.ffn_dim {
            let silu = gate[i] / (1.0f32 + (-gate[i]).exp());
            gate[i] = silu * up[i];
        }
        let mut down = vec![0.0f32; self.hidden_dim];
        self.down_proj.apply(&gate, op, backend, &mut down)?;
        for i in 0..self.hidden_dim {
            h[i] += down[i];
        }
        Ok(())
    }

    fn weight_bytes(&self, op: OperatorKind) -> usize {
        self.attn_norm.len() * 4
            + self.ffn_norm.len() * 4
            + self.q_proj.weight_bytes(op)
            + self.k_proj.weight_bytes(op)
            + self.v_proj.weight_bytes(op)
            + self.o_proj.weight_bytes(op)
            + self.gate_proj.weight_bytes(op)
            + self.up_proj.weight_bytes(op)
            + self.down_proj.weight_bytes(op)
    }
}

/// Shape profile corresponding to the shared `4B / 9B / 27B` Zen evaluation matrix (#40).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelScaleProfile {
    pub name: String,
    pub canonical_params_b: f64,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
}

impl ModelScaleProfile {
    /// Scaled `4B`, `9B`, and `27B` profiles preserving exact GQA and SwiGLU aspect ratios
    /// for reproducible CPU scorecard runs.
    pub fn ci_matrix() -> Vec<Self> {
        vec![
            Self {
                name: "4B-scaled".to_string(),
                canonical_params_b: 4.0,
                hidden_dim: 320,
                ffn_dim: 896,
                num_heads: 10,
                num_kv_heads: 2,
                num_layers: 2,
                vocab_size: 512,
            },
            Self {
                name: "9B-scaled".to_string(),
                canonical_params_b: 9.0,
                hidden_dim: 448,
                ffn_dim: 1216,
                num_heads: 14,
                num_kv_heads: 2,
                num_layers: 2,
                vocab_size: 512,
            },
            Self {
                name: "27B-scaled".to_string(),
                canonical_params_b: 27.0,
                hidden_dim: 576,
                ffn_dim: 1536,
                num_heads: 18,
                num_kv_heads: 3,
                num_layers: 2,
                vocab_size: 512,
            },
        ]
    }
}

/// Complete multi-layer causal ternary network (`Embedding` + `Blocks` + `RMSNorm` + `LMHead`).
#[derive(Clone, Debug)]
pub struct TernaryTransformerModel {
    profile: ModelScaleProfile,
    embeddings: Vec<f32>,
    blocks: Vec<TernaryNetworkBlock>,
    final_norm: Vec<f32>,
    lm_head: DualProjection,
}

impl TernaryTransformerModel {
    /// Builds a deterministic seeded ternary transformer network for `profile`.
    pub fn from_profile(profile: ModelScaleProfile, seed: u64, keep_dense: bool) -> Result<Self> {
        let mut rng = SplitMix64(seed);
        let embeddings = rng.vec_f32(profile.vocab_size * profile.hidden_dim, 0.08);
        let mut blocks = Vec::with_capacity(profile.num_layers);
        for _ in 0..profile.num_layers {
            blocks.push(TernaryNetworkBlock::from_rng(
                profile.hidden_dim,
                profile.ffn_dim,
                profile.num_heads,
                profile.num_kv_heads,
                keep_dense,
                &mut rng,
            )?);
        }
        let head_w = rng.vec_f32(profile.vocab_size * profile.hidden_dim, 0.08);
        let lm_head = DualProjection::from_seeded_f32(
            &head_w,
            profile.vocab_size,
            profile.hidden_dim,
            keep_dense,
        )?;
        Ok(Self {
            final_norm: vec![1.0f32; profile.hidden_dim],
            profile,
            embeddings,
            blocks,
            lm_head,
        })
    }

    /// Allocates a fresh KV cache sized for `max_seq_len`.
    pub fn new_kv_cache(&self, max_seq_len: usize) -> TernaryKvCache {
        let head_dim = self.profile.hidden_dim / self.profile.num_heads;
        let kv_dim = self.profile.num_kv_heads * head_dim;
        TernaryKvCache::new(self.profile.num_layers, max_seq_len, kv_dim)
    }

    /// Executes one causal forward token step at `kv_cache` offset and returns logits.
    pub fn forward_token(
        &self,
        token_id: usize,
        kv_cache: &mut TernaryKvCache,
        op: OperatorKind,
        backend: TmacKernelBackend,
    ) -> Result<Vec<f32>> {
        let pos = kv_cache.seq_len;
        if pos >= kv_cache.max_seq_len {
            hanzo_ml::bail!("TernaryKvCache overflow at pos={pos}");
        }
        let tok = token_id % self.profile.vocab_size;
        let h_dim = self.profile.hidden_dim;
        let mut h = self.embeddings[tok * h_dim..(tok + 1) * h_dim].to_vec();
        for (layer_idx, blk) in self.blocks.iter().enumerate() {
            blk.forward_step(layer_idx, pos, &mut h, kv_cache, op, backend)?;
        }
        kv_cache.seq_len = pos + 1;
        let mut h_norm = vec![0.0f32; h_dim];
        rms_norm_vec(&h, &self.final_norm, 1e-5, &mut h_norm);
        let mut logits = vec![0.0f32; self.profile.vocab_size];
        self.lm_head.apply(&h_norm, op, backend, &mut logits)?;
        Ok(logits)
    }

    /// Runs multi-token prefill and returns the final prompt token's logits.
    pub fn prefill(
        &self,
        prompt_tokens: &[usize],
        kv_cache: &mut TernaryKvCache,
        op: OperatorKind,
        backend: TmacKernelBackend,
    ) -> Result<Vec<f32>> {
        if prompt_tokens.is_empty() {
            hanzo_ml::bail!("prefill requires non-empty prompt_tokens");
        }
        let mut last_logits = Vec::new();
        for &tok in prompt_tokens {
            last_logits = self.forward_token(tok, kv_cache, op, backend)?;
        }
        Ok(last_logits)
    }

    /// Model weight footprint in bytes under `op`.
    pub fn weight_bytes(&self, op: OperatorKind) -> usize {
        let blk_sum: usize = self.blocks.iter().map(|b| b.weight_bytes(op)).sum();
        self.embeddings.len() * 4
            + self.final_norm.len() * 4
            + blk_sum
            + self.lm_head.weight_bytes(op)
    }
}

/// Hardware/ISA provenance metadata recorded on every scorecard run (#40).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareMetadata {
    pub cpu_arch: String,
    pub os: String,
    pub cpu_model: String,
    pub detected_isa: String,
    pub accelerator_driver: String,
    pub quantization_format: String,
    pub git_commit: String,
    pub energy_model: String,
}

impl HardwareMetadata {
    /// Inspects current host CPU/ISA environment without invoking external subprocesses.
    pub fn detect() -> Self {
        let cpu_model = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|txt| {
                txt.lines()
                    .find(|l| l.starts_with("model name") || l.starts_with("Model"))
                    .and_then(|l| l.split_once(':'))
                    .map(|(_, v)| v.trim().to_string())
            })
            .unwrap_or_else(|| format!("{}-cpu", std::env::consts::ARCH));
        let commit = std::env::var("GITHUB_SHA").unwrap_or_else(|_| "workspace-clean".to_string());
        Self {
            cpu_arch: std::env::consts::ARCH.to_string(),
            os: std::env::consts::OS.to_string(),
            cpu_model,
            detected_isa: TmacKernelBackend::detect_best().name().to_string(),
            accelerator_driver: "cpu-only (no GPU/NPU)".to_string(),
            quantization_format: "T-MAC W1.58A8 (2-bit 4-trit/byte LUT + f32 absmean + i8 act)"
                .to_string(),
            git_commit: commit,
            energy_model: "15.0W active single-core CPU + 120 pJ/byte DRAM traffic".to_string(),
        }
    }
}

/// Complete per-operator scorecard row across the 20-turn coding-agent benchmark (#40).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperatorScorecardRow {
    pub profile_name: String,
    pub canonical_params_b: f64,
    pub operator: OperatorKind,
    pub active_kernel: String,
    pub cold_ttft_ms: f64,
    pub warm_ttft_ms: f64,
    pub itl_p50_ms: f64,
    pub itl_p95_ms: f64,
    pub agent_20turn_wall_ms: f64,
    pub sustained_tokens_per_sec: f64,
    pub estimated_joules_per_token: f64,
    pub weight_memory_bytes: usize,
    pub peak_memory_bytes: usize,
    pub max_logit_abs_error_vs_fp32: f32,
    pub max_logit_abs_error_vs_unpack: f32,
    pub token_agreement_vs_fp32: f64,
    pub quality_regression: bool,
}

/// Anti-microbenchmark end-to-end acceptance gate verdict (#40).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EndToEndGateVerdict {
    pub accepted: bool,
    pub reason: String,
}

/// Rejects candidates that regress 20-turn wall time, sustained tok/s, energy/token, or accuracy.
pub fn check_end_to_end_acceptance_gate(
    candidate: &OperatorScorecardRow,
    cpu_reference: &OperatorScorecardRow,
) -> EndToEndGateVerdict {
    if candidate.quality_regression || candidate.max_logit_abs_error_vs_fp32 >= 1e-4 {
        return EndToEndGateVerdict {
            accepted: false,
            reason: format!(
                "Rejected: numerical error {:.2e} or token agreement {:.3}",
                candidate.max_logit_abs_error_vs_fp32, candidate.token_agreement_vs_fp32
            ),
        };
    }
    if candidate.sustained_tokens_per_sec <= cpu_reference.sustained_tokens_per_sec {
        return EndToEndGateVerdict {
            accepted: false,
            reason: format!(
                "Rejected: sustained tok/s {:.2} <= reference {:.2}",
                candidate.sustained_tokens_per_sec, cpu_reference.sustained_tokens_per_sec
            ),
        };
    }
    if candidate.agent_20turn_wall_ms >= cpu_reference.agent_20turn_wall_ms {
        return EndToEndGateVerdict {
            accepted: false,
            reason: format!(
                "Rejected: 20-turn wall {:.2} ms >= reference {:.2} ms",
                candidate.agent_20turn_wall_ms, cpu_reference.agent_20turn_wall_ms
            ),
        };
    }
    if candidate.estimated_joules_per_token >= cpu_reference.estimated_joules_per_token {
        return EndToEndGateVerdict {
            accepted: false,
            reason: format!(
                "Rejected: joules/token {:.5} >= reference {:.5}",
                candidate.estimated_joules_per_token, cpu_reference.estimated_joules_per_token
            ),
        };
    }
    EndToEndGateVerdict {
        accepted: true,
        reason: "Accepted: superior sustained tok/s, 20-turn wall time, and joules/token".into(),
    }
}

/// Full scorecard report across `4B`, `9B`, and `27B` profiles.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TmacScorecardReport {
    pub metadata: HardwareMetadata,
    pub turns: usize,
    pub prefill_tokens_per_turn: usize,
    pub decode_tokens_per_turn: usize,
    pub rows: Vec<OperatorScorecardRow>,
}

fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (((sorted.len() - 1) as f64) * pct).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn argmax_f32(slice: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = idx;
        }
    }
    best_idx
}

/// Executes the complete 20-turn coding-agent prefill + KV-cached decode benchmark across
/// `profiles` for `Fp32Reference`, `WeightUnpackDeqReference`, and `TmacLutOperator`.
pub fn run_tmac_scorecard(
    profiles: &[ModelScaleProfile],
    backend: TmacKernelBackend,
) -> Result<TmacScorecardReport> {
    let turns = 20usize;
    let prefill_per_turn = 6usize;
    let decode_per_turn = 4usize;
    let max_seq = turns * (prefill_per_turn + decode_per_turn);
    let resolved_backend = backend.resolve();
    let mut rows = Vec::with_capacity(profiles.len() * 3);

    for profile in profiles {
        let model = TernaryTransformerModel::from_profile(profile.clone(), 202604, true)?;
        let mut fp32_tokens = Vec::new();
        let mut fp32_last_logits = Vec::new();
        let mut unpack_last_logits = Vec::new();

        for op in [
            OperatorKind::Fp32Reference,
            OperatorKind::WeightUnpackDeqReference,
            OperatorKind::TmacLutOperator,
        ] {
            let mut kv_cache = model.new_kv_cache(max_seq);
            let mut ttfts_ms = Vec::with_capacity(turns);
            let mut itls_ms = Vec::with_capacity(turns * (decode_per_turn - 1));
            let mut generated_tokens = Vec::with_capacity(turns * decode_per_turn);
            let mut final_logits = Vec::new();

            let wall_start = Instant::now();
            for turn in 0..turns {
                let prompt: Vec<usize> = (0..prefill_per_turn)
                    .map(|p| (turn * 13 + p * 7 + 3) % profile.vocab_size)
                    .collect();
                let t_pf = Instant::now();
                let logits = model.prefill(&prompt, &mut kv_cache, op, resolved_backend)?;
                ttfts_ms.push(t_pf.elapsed().as_secs_f64() * 1000.0);

                let mut next_tok = argmax_f32(&logits);
                generated_tokens.push(next_tok);
                final_logits = logits;

                for _ in 1..decode_per_turn {
                    let t_dec = Instant::now();
                    let dec_logits =
                        model.forward_token(next_tok, &mut kv_cache, op, resolved_backend)?;
                    itls_ms.push(t_dec.elapsed().as_secs_f64() * 1000.0);
                    next_tok = argmax_f32(&dec_logits);
                    generated_tokens.push(next_tok);
                    final_logits = dec_logits;
                }
            }
            let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;

            let mut sorted_itls = itls_ms.clone();
            sorted_itls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let cold_ttft_ms = ttfts_ms.first().copied().unwrap_or(0.0);
            let warm_ttft_ms = if ttfts_ms.len() > 1 {
                ttfts_ms[1..].iter().sum::<f64>() / ((ttfts_ms.len() - 1) as f64)
            } else {
                cold_ttft_ms
            };
            let itl_p50_ms = percentile_sorted(&sorted_itls, 0.50);
            let itl_p95_ms = percentile_sorted(&sorted_itls, 0.95);
            let total_decode_s = (itls_ms.iter().sum::<f64>() / 1000.0).max(1e-9);
            let sustained_tok_s = (itls_ms.len() as f64) / total_decode_s;

            let weight_bytes = model.weight_bytes(op);
            let lut_ws = match op {
                OperatorKind::TmacLutOperator => {
                    profile.ffn_dim.div_ceil(TRITS_PER_BYTE) * std::mem::size_of::<SimdLutGroup>()
                }
                _ => 0,
            };
            let peak_bytes = weight_bytes
                + kv_cache.allocated_bytes()
                + lut_ws
                + (profile.ffn_dim * 8);
            let joules_per_token =
                15.0 * (itl_p50_ms / 1000.0) + 120e-12 * (weight_bytes as f64);

            let (err_fp32, err_unpack, agreement) = match op {
                OperatorKind::Fp32Reference => {
                    fp32_tokens = generated_tokens.clone();
                    fp32_last_logits = final_logits.clone();
                    (0.0f32, 0.0f32, 1.0f64)
                }
                OperatorKind::WeightUnpackDeqReference => {
                    unpack_last_logits = final_logits.clone();
                    let max_e = final_logits
                        .iter()
                        .zip(fp32_last_logits.iter())
                        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                    let matches = generated_tokens
                        .iter()
                        .zip(fp32_tokens.iter())
                        .filter(|(a, b)| a == b)
                        .count();
                    (max_e, 0.0f32, (matches as f64) / (fp32_tokens.len() as f64))
                }
                OperatorKind::TmacLutOperator => {
                    let max_e_fp32 = final_logits
                        .iter()
                        .zip(fp32_last_logits.iter())
                        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                    let max_e_unp = final_logits
                        .iter()
                        .zip(unpack_last_logits.iter())
                        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                    let matches = generated_tokens
                        .iter()
                        .zip(fp32_tokens.iter())
                        .filter(|(a, b)| a == b)
                        .count();
                    (
                        max_e_fp32,
                        max_e_unp,
                        (matches as f64) / (fp32_tokens.len() as f64),
                    )
                }
            };

            let active_kernel = match op {
                OperatorKind::Fp32Reference => "fp32-dense-cpu".to_string(),
                OperatorKind::WeightUnpackDeqReference => "2bit-runtime-unpack-cpu".to_string(),
                OperatorKind::TmacLutOperator => resolved_backend.name().to_string(),
            };

            rows.push(OperatorScorecardRow {
                profile_name: profile.name.clone(),
                canonical_params_b: profile.canonical_params_b,
                operator: op,
                active_kernel,
                cold_ttft_ms,
                warm_ttft_ms,
                itl_p50_ms,
                itl_p95_ms,
                agent_20turn_wall_ms: wall_ms,
                sustained_tokens_per_sec: sustained_tok_s,
                estimated_joules_per_token: joules_per_token,
                weight_memory_bytes: weight_bytes,
                peak_memory_bytes: peak_bytes,
                max_logit_abs_error_vs_fp32: err_fp32,
                max_logit_abs_error_vs_unpack: err_unpack,
                token_agreement_vs_fp32: agreement,
                quality_regression: agreement < 0.99 || err_fp32 >= 1e-4,
            });
        }
    }

    Ok(TmacScorecardReport {
        metadata: HardwareMetadata::detect(),
        turns,
        prefill_tokens_per_turn: prefill_per_turn,
        decode_tokens_per_turn: decode_per_turn,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radix81_and_symmetric41_exact_bijection() {
        let q = [127i8, -64, 31, -127];
        let sym41 = build_symmetric_lut41(q);
        let lut81 = build_radix81_lut(q);
        let simd_g = SimdLutGroup::from_group4(q);
        assert_eq!(sym41[40], 0);
        for u in 0..RADIX3_GROUP4_STATES {
            let trits = radix81_to_trits(u);
            assert_eq!(trits_to_radix81(trits), u);
            let packed = pack_trit_group4(trits);
            assert_eq!(unpack_trit_group4(packed), trits);
            let exact = i32::from(trits[0]) * i32::from(q[0])
                + i32::from(trits[1]) * i32::from(q[1])
                + i32::from(trits[2]) * i32::from(q[2])
                + i32::from(trits[3]) * i32::from(q[3]);
            assert_eq!(i32::from(lut81[u]), exact);
            assert_eq!(simd_g.lookup_byte_i32(packed), exact);
        }
    }
}
