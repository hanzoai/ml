//! BITCOS zero-density-aware ternary packing and profile-guided layout selector.
//!
//! Stores ternary weights in {-1, 0, +1} with per-group f16 scales (group 64 or 128)
//! as a 1-bit presence bitmap, a compacted 1-bit sign stream over non-zero entries,
//! and a CSR-style u32 sign-bit offset index per superblock so rows and warps decode
//! independently in parallel with 0.0 ULP error relative to PQ2_0 and PTQ1_0.

use super::iq_quants::{BlockPQ2_0, BlockPTQ1_0, QK_PQ2_0, QK_PTQ1_0};
use super::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use crate::Result;
use half::f16;

pub const BITCOS_WORD_BITS: usize = 64;
pub const BITCOS_GROUP_64: usize = 64;
pub const BITCOS_GROUP_128: usize = 128;
pub const DEFAULT_SUPERBLOCK_ELEMS: usize = 512;
pub const PQ2_0_BPW: f64 = 2.125;
pub const PTQ1_0_BPW: f64 = 1.75;
pub const ML_TERNARY_LAYOUT_ENV: &str = "ML_TERNARY_LAYOUT";

const BITS_PER_BYTE: usize = 8;
const SCALE_F16_BITS: usize = 16;
const OFFSET_U32_BITS: usize = 32;
const PQ2_CODE_MASK: u8 = 0x03;
const PQ2_CODE_ZERO: u8 = 1;
const PQ2_CODE_MAX_TERNARY: u8 = 2;
const PTQ1_QS_STAGE0_BYTES: usize = 16;
const PTQ1_QS_STAGE1_BYTES: usize = 8;
const PTQ1_QH_BYTES: usize = 2;
const PTQ1_POW3: [u8; 5] = [1, 3, 9, 27, 81];
const TERNARY_BYTE_STATES: usize = 6561; // 3^8 states for 8 ternary weights
const BYTE_LUT_ENTRIES: usize = 256;
const TINY_MATRIX_ELEMS: usize = 65_536;
const BITCOS_PTQ1_MARGIN_BPW: f64 = 0.02;
const PARALLEL_ROW_THRESHOLD: usize = 64;

const fn build_byte_tables() -> ([u32; BYTE_LUT_ENTRIES], [u64; TERNARY_BYTE_STATES]) {
    let mut meta = [0u32; BYTE_LUT_ENTRIES];
    let mut trits = [0u64; TERNARY_BYTE_STATES];
    let mut base = 0usize;
    let mut b = 0usize;
    while b < BYTE_LUT_ENTRIES {
        let pop = (b as u8).count_ones();
        let combos = 1usize << pop;
        let mask = (combos - 1) as u32;
        meta[b] = (base as u32) | (mask << 16) | (pop << 24);
        let mut s = 0usize;
        while s < combos {
            let mut word = 0u64;
            let mut k = 0usize;
            let mut lane = 0usize;
            while lane < BITS_PER_BYTE {
                if ((b >> lane) & 1) != 0 {
                    let sign_neg = ((s >> k) & 1) != 0;
                    let byte_val: u8 = if sign_neg { 0xFF } else { 0x01 };
                    word |= (byte_val as u64) << (lane * BITS_PER_BYTE);
                    k += 1;
                }
                lane += 1;
            }
            trits[base + s] = word;
            s += 1;
        }
        base += combos;
        b += 1;
    }
    assert!(base == TERNARY_BYTE_STATES);
    (meta, trits)
}

const BYTE_TABLES: ([u32; BYTE_LUT_ENTRIES], [u64; TERNARY_BYTE_STATES]) = build_byte_tables();

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitcosGroupSize {
    G64 = 64,
    G128 = 128,
}

impl BitcosGroupSize {
    #[inline]
    pub const fn elems(self) -> usize {
        match self {
            Self::G64 => BITCOS_GROUP_64,
            Self::G128 => BITCOS_GROUP_128,
        }
    }

    #[inline]
    pub const fn words_per_group(self) -> usize {
        self.elems() / BITCOS_WORD_BITS
    }

    #[inline]
    pub const fn scale_bpw(self) -> f64 {
        (SCALE_F16_BITS as f64) / (self.elems() as f64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitcosConfig {
    pub group_size: BitcosGroupSize,
    pub superblock_elems: usize,
}

impl Default for BitcosConfig {
    fn default() -> Self {
        Self {
            group_size: BitcosGroupSize::G128,
            superblock_elems: DEFAULT_SUPERBLOCK_ELEMS,
        }
    }
}

impl BitcosConfig {
    pub const fn new(group_size: BitcosGroupSize, superblock_elems: usize) -> Self {
        Self {
            group_size,
            superblock_elems,
        }
    }

    fn validate(self, cols: usize) -> Result<usize> {
        let g = self.group_size.elems();
        if cols == 0 || !cols.is_multiple_of(g) {
            crate::bail!("bitcos: cols {cols} must be a non-zero multiple of group size {g}");
        }
        if self.superblock_elems == 0 || !self.superblock_elems.is_multiple_of(BITCOS_WORD_BITS) {
            crate::bail!(
                "bitcos: superblock_elems {} must be a non-zero multiple of {BITCOS_WORD_BITS}",
                self.superblock_elems
            );
        }
        Ok(cols.div_ceil(self.superblock_elems))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernaryLayout {
    Pq2_0,
    Ptq1_0,
    BitcosG128,
    BitcosG64,
}

impl TernaryLayout {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pq2_0 => "PQ2_0",
            Self::Ptq1_0 => "PTQ1_0",
            Self::BitcosG128 => "Bitcos(G128)",
            Self::BitcosG64 => "Bitcos(G64)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TernarySelectionPolicy {
    Balanced,
    MemoryFirst,
    UpgradePq2Only,
}

/// Exact analytical effective bits-per-weight for BITCOS including presence bitmap,
/// u64-padded compacted sign stream, per-group f16 scales, and CSR u32 sign offsets.
pub fn estimate_bitcos_bpw(
    zero_fraction: f64,
    rows: usize,
    cols: usize,
    config: BitcosConfig,
) -> f64 {
    let total_elems = rows.saturating_mul(cols);
    if total_elems == 0 {
        return 0.0;
    }
    let z = zero_fraction.clamp(0.0, 1.0);
    let est_nnz = ((1.0 - z) * (total_elems as f64)).round() as usize;
    let presence_bits = total_elems.div_ceil(BITCOS_WORD_BITS) * BITCOS_WORD_BITS;
    let sign_bits = est_nnz.div_ceil(BITCOS_WORD_BITS) * BITCOS_WORD_BITS;
    let groups_per_row = cols.div_ceil(config.group_size.elems());
    let scale_bits = rows * groups_per_row * SCALE_F16_BITS;
    let sb_elems = config.superblock_elems.max(BITCOS_WORD_BITS);
    let sb_per_row = cols.div_ceil(sb_elems);
    let offset_bits = (rows * sb_per_row + 1) * OFFSET_U32_BITS;
    (presence_bits + sign_bits + scale_bits + offset_bits) as f64 / (total_elems as f64)
}

fn parse_env_layout() -> Option<TernaryLayout> {
    let val = std::env::var(ML_TERNARY_LAYOUT_ENV).ok()?;
    match val.trim().to_ascii_lowercase().as_str() {
        "pq2" | "pq2_0" => Some(TernaryLayout::Pq2_0),
        "ptq1" | "ptq1_0" => Some(TernaryLayout::Ptq1_0),
        "bitcos" | "bitcos_g128" | "bitcos128" => Some(TernaryLayout::BitcosG128),
        "bitcos_g64" | "bitcos64" => Some(TernaryLayout::BitcosG64),
        _ => None,
    }
}

/// Profile-guided per-tensor layout selector balancing byte savings vs CPU decode cost.
///
/// Decision regimes for group-128 aligned matrices:
/// 1. Small tensors (< 65,536 weights): `PQ2_0` minimizes decode instruction count when
///    absolute byte savings are negligible (< 3 KB).
/// 2. Sparse ternary regime (`b_BITCOS + 0.02 <= 1.75`, i.e. z >= ~0.46 at sb=512):
///    `BitcosG128` strictly beats both `PQ2_0` and `PTQ1_0` in memory footprint while
///    sparse sub-block skipping accelerates matvec.
/// 3. Dense / Hadamard-rotated ternary regime (z < ~0.46, including Bonsai-2 z ~ 0.328):
///    `PTQ1_0` achieves 1.75 bpw (near log2(3) = 1.585 payload limit) vs Bitcos ~1.86 bpw
///    and PQ2_0 2.125 bpw.
/// 4. Group-64-only odd column shapes (`cols % 128 != 0` and `cols % 64 == 0`): `BitcosG64`.
pub fn choose_ternary_layout(zero_fraction: f64, rows: usize, cols: usize) -> TernaryLayout {
    if let Some(forced) = parse_env_layout() {
        return forced;
    }
    choose_ternary_layout_with_policy(zero_fraction, rows, cols, TernarySelectionPolicy::Balanced)
}

pub fn choose_ternary_layout_with_policy(
    zero_fraction: f64,
    rows: usize,
    cols: usize,
    policy: TernarySelectionPolicy,
) -> TernaryLayout {
    if !cols.is_multiple_of(BITCOS_GROUP_128) {
        return TernaryLayout::BitcosG64;
    }
    let total_elems = rows.saturating_mul(cols);
    let bitcos_bpw = estimate_bitcos_bpw(zero_fraction, rows, cols, BitcosConfig::default());
    match policy {
        TernarySelectionPolicy::MemoryFirst => {
            if bitcos_bpw < PTQ1_0_BPW {
                TernaryLayout::BitcosG128
            } else {
                TernaryLayout::Ptq1_0
            }
        }
        TernarySelectionPolicy::UpgradePq2Only => {
            if total_elems >= TINY_MATRIX_ELEMS && bitcos_bpw + BITCOS_PTQ1_MARGIN_BPW < PQ2_0_BPW {
                TernaryLayout::BitcosG128
            } else {
                TernaryLayout::Pq2_0
            }
        }
        TernarySelectionPolicy::Balanced => {
            if total_elems < TINY_MATRIX_ELEMS {
                return TernaryLayout::Pq2_0;
            }
            if bitcos_bpw + BITCOS_PTQ1_MARGIN_BPW <= PTQ1_0_BPW {
                TernaryLayout::BitcosG128
            } else {
                TernaryLayout::Ptq1_0
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TernaryTensorProfile {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub zero_fraction: f64,
    pub recommended_layout: TernaryLayout,
    pub effective_bpw: f64,
    pub estimated_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct TernaryFormatManifest {
    pub policy: TernarySelectionPolicy,
    pub tensors: Vec<TernaryTensorProfile>,
    pub total_weights: u64,
    pub total_bytes: u64,
    pub aggregate_bpw: f64,
}

impl TernaryFormatManifest {
    pub fn from_tensor_stats(
        entries: &[(String, usize, usize, f64)],
        policy: TernarySelectionPolicy,
    ) -> Self {
        let mut tensors = Vec::with_capacity(entries.len());
        let mut total_weights = 0u64;
        let mut total_bytes = 0u64;
        for (name, rows, cols, z) in entries {
            let layout = choose_ternary_layout_with_policy(*z, *rows, *cols, policy);
            let elems = rows.saturating_mul(*cols);
            let bpw = match layout {
                TernaryLayout::Pq2_0 => PQ2_0_BPW,
                TernaryLayout::Ptq1_0 => PTQ1_0_BPW,
                TernaryLayout::BitcosG128 => {
                    estimate_bitcos_bpw(*z, *rows, *cols, BitcosConfig::default())
                }
                TernaryLayout::BitcosG64 => estimate_bitcos_bpw(
                    *z,
                    *rows,
                    *cols,
                    BitcosConfig::new(BitcosGroupSize::G64, DEFAULT_SUPERBLOCK_ELEMS),
                ),
            };
            let bytes = ((elems as f64) * bpw / (BITS_PER_BYTE as f64)).ceil() as usize;
            total_weights += elems as u64;
            total_bytes += bytes as u64;
            tensors.push(TernaryTensorProfile {
                name: name.clone(),
                rows: *rows,
                cols: *cols,
                zero_fraction: *z,
                recommended_layout: layout,
                effective_bpw: bpw,
                estimated_bytes: bytes,
            });
        }
        let aggregate_bpw = if total_weights > 0 {
            (total_bytes as f64 * BITS_PER_BYTE as f64) / (total_weights as f64)
        } else {
            0.0
        };
        Self {
            policy,
            tensors,
            total_weights,
            total_bytes,
            aggregate_bpw,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ZeroDensityMetrics {
    pub total_weights: u64,
    pub zero_weights: u64,
    pub pos_weights: u64,
    pub neg_weights: u64,
    pub non_ternary_weights: u64,
    pub zero_blocks_16: u64,
    pub total_blocks_16: u64,
    pub zero_blocks_32: u64,
    pub total_blocks_32: u64,
    pub zero_blocks_64: u64,
    pub total_blocks_64: u64,
    pub zero_scale_blocks_128: u64,
    pub total_scale_blocks_128: u64,
}

impl ZeroDensityMetrics {
    pub fn merge(&mut self, other: &Self) {
        self.total_weights += other.total_weights;
        self.zero_weights += other.zero_weights;
        self.pos_weights += other.pos_weights;
        self.neg_weights += other.neg_weights;
        self.non_ternary_weights += other.non_ternary_weights;
        self.zero_blocks_16 += other.zero_blocks_16;
        self.total_blocks_16 += other.total_blocks_16;
        self.zero_blocks_32 += other.zero_blocks_32;
        self.total_blocks_32 += other.total_blocks_32;
        self.zero_blocks_64 += other.zero_blocks_64;
        self.total_blocks_64 += other.total_blocks_64;
        self.zero_scale_blocks_128 += other.zero_scale_blocks_128;
        self.total_scale_blocks_128 += other.total_scale_blocks_128;
    }

    pub fn zero_fraction(&self) -> f64 {
        if self.total_weights == 0 {
            0.0
        } else {
            self.zero_weights as f64 / self.total_weights as f64
        }
    }

    pub fn zero_block16_fraction(&self) -> f64 {
        if self.total_blocks_16 == 0 {
            0.0
        } else {
            self.zero_blocks_16 as f64 / self.total_blocks_16 as f64
        }
    }

    pub fn zero_block32_fraction(&self) -> f64 {
        if self.total_blocks_32 == 0 {
            0.0
        } else {
            self.zero_blocks_32 as f64 / self.total_blocks_32 as f64
        }
    }

    pub fn zero_block64_fraction(&self) -> f64 {
        if self.total_blocks_64 == 0 {
            0.0
        } else {
            self.zero_blocks_64 as f64 / self.total_blocks_64 as f64
        }
    }

    pub fn shannon_entropy_bpw(&self) -> f64 {
        if self.total_weights == 0 {
            return 0.0;
        }
        let n = self.total_weights as f64;
        let mut h = 0.0;
        for &c in &[
            self.neg_weights,
            self.zero_weights,
            self.pos_weights,
            self.non_ternary_weights,
        ] {
            if c > 0 {
                let p = (c as f64) / n;
                h -= p * p.log2();
            }
        }
        h
    }

    #[inline]
    fn record_presence_u64(&mut self, m64: u64) {
        self.total_blocks_64 += 1;
        self.total_blocks_32 += 2;
        self.total_blocks_16 += 4;
        if m64 == 0 {
            self.zero_blocks_64 += 1;
            self.zero_blocks_32 += 2;
            self.zero_blocks_16 += 4;
            return;
        }
        if (m64 as u32) == 0 {
            self.zero_blocks_32 += 1;
        }
        if (m64 >> 32) == 0 {
            self.zero_blocks_32 += 1;
        }
        if (m64 & 0xFFFF) == 0 {
            self.zero_blocks_16 += 1;
        }
        if ((m64 >> 16) & 0xFFFF) == 0 {
            self.zero_blocks_16 += 1;
        }
        if ((m64 >> 32) & 0xFFFF) == 0 {
            self.zero_blocks_16 += 1;
        }
        if (m64 >> 48) == 0 {
            self.zero_blocks_16 += 1;
        }
    }
}

/// Compact the 32 even-indexed bits (0, 2, 4, ..., 62) of `x` into a contiguous u32.
#[inline]
fn compact_even_bits_u64(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// Zero-copy SWAR scan of raw `BlockPQ2_0` bytes (34 B/block).
pub fn scan_pq2_0_bytes(raw: &[u8]) -> Result<ZeroDensityMetrics> {
    const BLOCK_BYTES: usize = std::mem::size_of::<BlockPQ2_0>();
    if !raw.len().is_multiple_of(BLOCK_BYTES) {
        crate::bail!(
            "scan_pq2_0_bytes: len {} not divisible by {BLOCK_BYTES}",
            raw.len()
        );
    }
    let nb = raw.len() / BLOCK_BYTES;
    let mut m = ZeroDensityMetrics {
        total_weights: (nb * QK_PQ2_0) as u64,
        total_scale_blocks_128: nb as u64,
        ..Default::default()
    };
    for bi in 0..nb {
        let blk = &raw[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
        let d_bits = u16::from_le_bytes([blk[0], blk[1]]);
        if f16::from_bits(d_bits).to_f32() == 0.0 {
            m.zero_scale_blocks_128 += 1;
        }
        for half64 in 0..2 {
            let base = 2 + half64 * 16;
            let w_lo = u64::from_le_bytes(blk[base..base + 8].try_into().unwrap());
            let w_hi = u64::from_le_bytes(blk[base + 8..base + 16].try_into().unwrap());
            let mut pres64 = 0u64;
            for (half32, w) in [(0, w_lo), (1, w_hi)] {
                let b0 = w & 0x5555_5555_5555_5555;
                let b1 = (w >> 1) & 0x5555_5555_5555_5555;
                let is_neg = (!b1) & (!b0) & 0x5555_5555_5555_5555;
                let is_zero = (!b1) & b0;
                let is_pos = b1 & (!b0);
                let is_p2 = b1 & b0;
                m.neg_weights += is_neg.count_ones() as u64;
                m.zero_weights += is_zero.count_ones() as u64;
                m.pos_weights += is_pos.count_ones() as u64;
                m.non_ternary_weights += is_p2.count_ones() as u64;
                let nz_even = (b1 | (!b0)) & 0x5555_5555_5555_5555;
                let p32 = compact_even_bits_u64(nz_even) as u64;
                pres64 |= p32 << (half32 * 32);
            }
            m.record_presence_u64(pres64);
        }
    }
    Ok(m)
}

#[inline]
fn unpack_ptq1_0_trits(blk: &BlockPTQ1_0, out: &mut [i8; QK_PTQ1_0]) {
    let mut idx = 0usize;
    for &p in &PTQ1_POW3 {
        for m in 0..PTQ1_QS_STAGE0_BYTES {
            let xi = (((blk.qs[m].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
            out[idx + m] = xi;
        }
        idx += PTQ1_QS_STAGE0_BYTES;
    }
    for &p in &PTQ1_POW3 {
        for m in 0..PTQ1_QS_STAGE1_BYTES {
            let xi =
                (((blk.qs[PTQ1_QS_STAGE0_BYTES + m].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
            out[idx + m] = xi;
        }
        idx += PTQ1_QS_STAGE1_BYTES;
    }
    for &p in &PTQ1_POW3[..4] {
        for m in 0..PTQ1_QH_BYTES {
            out[idx + m] = (((blk.qh[m].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
        }
        idx += PTQ1_QH_BYTES;
    }
}

/// Zero-copy scan of raw `BlockPTQ1_0` bytes (28 B/block).
pub fn scan_ptq1_0_bytes(raw: &[u8]) -> Result<ZeroDensityMetrics> {
    const BLOCK_BYTES: usize = std::mem::size_of::<BlockPTQ1_0>();
    if !raw.len().is_multiple_of(BLOCK_BYTES) {
        crate::bail!(
            "scan_ptq1_0_bytes: len {} not divisible by {BLOCK_BYTES}",
            raw.len()
        );
    }
    let nb = raw.len() / BLOCK_BYTES;
    let mut m = ZeroDensityMetrics {
        total_weights: (nb * QK_PTQ1_0) as u64,
        total_scale_blocks_128: nb as u64,
        ..Default::default()
    };
    let mut trits = [0i8; QK_PTQ1_0];
    for bi in 0..nb {
        let s = &raw[bi * BLOCK_BYTES..(bi + 1) * BLOCK_BYTES];
        let mut qs = [0u8; 24];
        qs.copy_from_slice(&s[0..24]);
        let qh = [s[24], s[25]];
        let d = f16::from_bits(u16::from_le_bytes([s[26], s[27]]));
        if d.to_f32() == 0.0 {
            m.zero_scale_blocks_128 += 1;
        }
        let blk = BlockPTQ1_0 { qs, qh, d };
        unpack_ptq1_0_trits(&blk, &mut trits);
        for half64 in 0..2 {
            let mut pres64 = 0u64;
            for j in 0..BITCOS_WORD_BITS {
                let t = trits[half64 * BITCOS_WORD_BITS + j];
                if t < 0 {
                    m.neg_weights += 1;
                    pres64 |= 1u64 << j;
                } else if t > 0 {
                    m.pos_weights += 1;
                    pres64 |= 1u64 << j;
                } else {
                    m.zero_weights += 1;
                }
            }
            m.record_presence_u64(pres64);
        }
    }
    Ok(m)
}

/// Lossless zero-density-aware ternary matrix representation (BITCOS).
#[derive(Debug, Clone)]
pub struct BitcosMatrix {
    rows: usize,
    cols: usize,
    group_size: BitcosGroupSize,
    superblock_elems: usize,
    superblocks_per_row: usize,
    nnz: usize,
    presence: Vec<u64>,
    signs: Vec<u64>,
    scales: Vec<f16>,
    sign_offsets: Vec<u32>,
}

impl BitcosMatrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn group_size(&self) -> BitcosGroupSize {
        self.group_size
    }

    pub fn superblock_elems(&self) -> usize {
        self.superblock_elems
    }

    pub fn superblocks_per_row(&self) -> usize {
        self.superblocks_per_row
    }

    pub fn nnz(&self) -> usize {
        self.nnz
    }

    pub fn zero_fraction(&self) -> f64 {
        let total = self.rows.saturating_mul(self.cols);
        if total == 0 {
            0.0
        } else {
            1.0 - (self.nnz as f64) / (total as f64)
        }
    }

    pub fn presence_words(&self) -> &[u64] {
        &self.presence
    }

    pub fn sign_words(&self) -> &[u64] {
        &self.signs
    }

    pub fn scales(&self) -> &[f16] {
        &self.scales
    }

    pub fn sign_offsets(&self) -> &[u32] {
        &self.sign_offsets
    }

    pub fn presence_bytes(&self) -> usize {
        self.presence.len() * std::mem::size_of::<u64>()
    }

    pub fn sign_bytes(&self) -> usize {
        self.signs.len() * std::mem::size_of::<u64>()
    }

    pub fn scale_bytes(&self) -> usize {
        self.scales.len() * std::mem::size_of::<f16>()
    }

    pub fn offset_bytes(&self) -> usize {
        self.sign_offsets.len() * std::mem::size_of::<u32>()
    }

    pub fn total_bytes(&self) -> usize {
        self.presence_bytes() + self.sign_bytes() + self.scale_bytes() + self.offset_bytes()
    }

    pub fn effective_bpw(&self) -> f64 {
        let total_elems = self.rows.saturating_mul(self.cols);
        if total_elems == 0 {
            0.0
        } else {
            (self.total_bytes() as f64 * BITS_PER_BYTE as f64) / (total_elems as f64)
        }
    }

    /// Encode ternary weights `trits` in {-1, 0, +1} with per-group `f16` scales.
    pub fn encode_from_trits(
        trits: &[i8],
        scales: &[f16],
        rows: usize,
        cols: usize,
        config: BitcosConfig,
    ) -> Result<Self> {
        let sb_per_row = config.validate(cols)?;
        let total_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| crate::Error::Msg("bitcos: rows * cols overflow".into()))?;
        if trits.len() != total_elems {
            crate::bail!(
                "bitcos: trits len {} != rows * cols ({total_elems})",
                trits.len()
            );
        }
        let groups_per_row = cols / config.group_size.elems();
        let expected_scales = rows * groups_per_row;
        if scales.len() != expected_scales {
            crate::bail!(
                "bitcos: scales len {} != expected {expected_scales}",
                scales.len()
            );
        }
        let words_per_row = cols / BITCOS_WORD_BITS;
        let sb_words = config.superblock_elems / BITCOS_WORD_BITS;
        let mut presence = vec![0u64; rows * words_per_row];
        let mut sign_offsets = Vec::with_capacity(rows * sb_per_row + 1);
        let mut signs = Vec::with_capacity(total_elems.div_ceil(BITCOS_WORD_BITS));
        let mut sign_cursor = 0usize;

        for r in 0..rows {
            let row_trits = &trits[r * cols..(r + 1) * cols];
            let row_pres = &mut presence[r * words_per_row..(r + 1) * words_per_row];
            for sb in 0..sb_per_row {
                if sign_cursor > u32::MAX as usize {
                    crate::bail!("bitcos: total non-zero count exceeds u32 sign offset limit");
                }
                sign_offsets.push(sign_cursor as u32);
                let w_start = sb * sb_words;
                let w_end = (w_start + sb_words).min(words_per_row);
                for w_idx in w_start..w_end {
                    let chunk =
                        &row_trits[w_idx * BITCOS_WORD_BITS..(w_idx + 1) * BITCOS_WORD_BITS];
                    let mut p_word = 0u64;
                    for j in 0..BITCOS_WORD_BITS {
                        let t = chunk[j];
                        match t {
                            0 => {}
                            1 => {
                                p_word |= 1u64 << j;
                                if (sign_cursor >> 6) >= signs.len() {
                                    signs.push(0u64);
                                }
                                sign_cursor += 1;
                            }
                            -1 => {
                                p_word |= 1u64 << j;
                                let sw = sign_cursor >> 6;
                                let sb_bit = sign_cursor & 63;
                                if sw >= signs.len() {
                                    signs.push(0u64);
                                }
                                signs[sw] |= 1u64 << sb_bit;
                                sign_cursor += 1;
                            }
                            other => {
                                crate::bail!("bitcos: non-ternary weight {other} at row {r}");
                            }
                        }
                    }
                    row_pres[w_idx] = p_word;
                }
            }
        }
        if sign_cursor > u32::MAX as usize {
            crate::bail!("bitcos: total non-zero count exceeds u32 sign offset limit");
        }
        sign_offsets.push(sign_cursor as u32);
        signs.truncate(sign_cursor.div_ceil(BITCOS_WORD_BITS));

        Ok(Self {
            rows,
            cols,
            group_size: config.group_size,
            superblock_elems: config.superblock_elems,
            superblocks_per_row: sb_per_row,
            nnz: sign_cursor,
            presence,
            signs,
            scales: scales.to_vec(),
            sign_offsets,
        })
    }

    /// Losslessly pack `BlockPQ2_0` blocks into `BitcosMatrix` with default G128 config.
    pub fn encode_from_pq2_0(blocks: &[BlockPQ2_0], rows: usize, cols: usize) -> Result<Self> {
        Self::encode_from_pq2_0_with_config(blocks, rows, cols, BitcosConfig::default())
    }

    /// Losslessly pack `BlockPQ2_0` blocks into `BitcosMatrix` with custom config (G128 or G64).
    pub fn encode_from_pq2_0_with_config(
        blocks: &[BlockPQ2_0],
        rows: usize,
        cols: usize,
        config: BitcosConfig,
    ) -> Result<Self> {
        if !cols.is_multiple_of(QK_PQ2_0) {
            crate::bail!("encode_from_pq2_0: cols {cols} must be divisible by {QK_PQ2_0}");
        }
        let sb_per_row = config.validate(cols)?;
        let blocks_per_row = cols / QK_PQ2_0;
        if blocks.len() != rows * blocks_per_row {
            crate::bail!(
                "encode_from_pq2_0: blocks len {} != expected {}",
                blocks.len(),
                rows * blocks_per_row
            );
        }
        let words_per_row = cols / BITCOS_WORD_BITS;
        let sb_words = config.superblock_elems / BITCOS_WORD_BITS;
        let groups_per_row = cols / config.group_size.elems();
        let mut presence = vec![0u64; rows * words_per_row];
        let mut scales = Vec::with_capacity(rows * groups_per_row);
        let mut sign_offsets = Vec::with_capacity(rows * sb_per_row + 1);
        let mut signs = Vec::with_capacity((rows * cols).div_ceil(BITCOS_WORD_BITS));
        let mut sign_cursor = 0usize;

        for r in 0..rows {
            let row_blocks = &blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
            for blk in row_blocks {
                match config.group_size {
                    BitcosGroupSize::G128 => scales.push(blk.d),
                    BitcosGroupSize::G64 => {
                        scales.push(blk.d);
                        scales.push(blk.d);
                    }
                }
            }
            let row_pres = &mut presence[r * words_per_row..(r + 1) * words_per_row];
            for sb in 0..sb_per_row {
                if sign_cursor > u32::MAX as usize {
                    crate::bail!("encode_from_pq2_0: non-zero count overflows u32");
                }
                sign_offsets.push(sign_cursor as u32);
                let w_start = sb * sb_words;
                let w_end = (w_start + sb_words).min(words_per_row);
                for w_idx in w_start..w_end {
                    let blk = &row_blocks[w_idx / 2];
                    let qs16 = &blk.qs[(w_idx % 2) * 16..(w_idx % 2) * 16 + 16];
                    let mut p_word = 0u64;
                    for j in 0..BITCOS_WORD_BITS {
                        let code = (qs16[j / 4] >> ((j % 4) * 2)) & PQ2_CODE_MASK;
                        if code > PQ2_CODE_MAX_TERNARY {
                            crate::bail!(
                                "encode_from_pq2_0: non-ternary +2 code (3) at row {r}, col {}",
                                w_idx * BITCOS_WORD_BITS + j
                            );
                        }
                        if code != PQ2_CODE_ZERO {
                            p_word |= 1u64 << j;
                            let sw = sign_cursor >> 6;
                            if sw >= signs.len() {
                                signs.push(0u64);
                            }
                            if code == 0 {
                                signs[sw] |= 1u64 << (sign_cursor & 63);
                            }
                            sign_cursor += 1;
                        }
                    }
                    row_pres[w_idx] = p_word;
                }
            }
        }
        if sign_cursor > u32::MAX as usize {
            crate::bail!("encode_from_pq2_0: non-zero count overflows u32");
        }
        sign_offsets.push(sign_cursor as u32);
        signs.truncate(sign_cursor.div_ceil(BITCOS_WORD_BITS));

        Ok(Self {
            rows,
            cols,
            group_size: config.group_size,
            superblock_elems: config.superblock_elems,
            superblocks_per_row: sb_per_row,
            nnz: sign_cursor,
            presence,
            signs,
            scales,
            sign_offsets,
        })
    }

    /// Losslessly pack `BlockPTQ1_0` blocks into `BitcosMatrix` with default G128 config.
    pub fn encode_from_ptq1_0(blocks: &[BlockPTQ1_0], rows: usize, cols: usize) -> Result<Self> {
        Self::encode_from_ptq1_0_with_config(blocks, rows, cols, BitcosConfig::default())
    }

    /// Losslessly pack `BlockPTQ1_0` blocks into `BitcosMatrix` with custom config (G128 or G64).
    pub fn encode_from_ptq1_0_with_config(
        blocks: &[BlockPTQ1_0],
        rows: usize,
        cols: usize,
        config: BitcosConfig,
    ) -> Result<Self> {
        if !cols.is_multiple_of(QK_PTQ1_0) {
            crate::bail!("encode_from_ptq1_0: cols {cols} must be divisible by {QK_PTQ1_0}");
        }
        let sb_per_row = config.validate(cols)?;
        let blocks_per_row = cols / QK_PTQ1_0;
        if blocks.len() != rows * blocks_per_row {
            crate::bail!(
                "encode_from_ptq1_0: blocks len {} != expected {}",
                blocks.len(),
                rows * blocks_per_row
            );
        }
        let words_per_row = cols / BITCOS_WORD_BITS;
        let sb_words = config.superblock_elems / BITCOS_WORD_BITS;
        let groups_per_row = cols / config.group_size.elems();
        let mut presence = vec![0u64; rows * words_per_row];
        let mut scales = Vec::with_capacity(rows * groups_per_row);
        let mut sign_offsets = Vec::with_capacity(rows * sb_per_row + 1);
        let mut signs = Vec::with_capacity((rows * cols).div_ceil(BITCOS_WORD_BITS));
        let mut sign_cursor = 0usize;
        let mut row_trits = vec![0i8; cols];
        let mut blk_trits = [0i8; QK_PTQ1_0];

        for r in 0..rows {
            let row_blocks = &blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
            for (bi, blk) in row_blocks.iter().enumerate() {
                match config.group_size {
                    BitcosGroupSize::G128 => scales.push(blk.d),
                    BitcosGroupSize::G64 => {
                        scales.push(blk.d);
                        scales.push(blk.d);
                    }
                }
                unpack_ptq1_0_trits(blk, &mut blk_trits);
                row_trits[bi * QK_PTQ1_0..(bi + 1) * QK_PTQ1_0].copy_from_slice(&blk_trits);
            }
            let row_pres = &mut presence[r * words_per_row..(r + 1) * words_per_row];
            for sb in 0..sb_per_row {
                if sign_cursor > u32::MAX as usize {
                    crate::bail!("encode_from_ptq1_0: non-zero count overflows u32");
                }
                sign_offsets.push(sign_cursor as u32);
                let w_start = sb * sb_words;
                let w_end = (w_start + sb_words).min(words_per_row);
                for w_idx in w_start..w_end {
                    let chunk =
                        &row_trits[w_idx * BITCOS_WORD_BITS..(w_idx + 1) * BITCOS_WORD_BITS];
                    let mut p_word = 0u64;
                    for j in 0..BITCOS_WORD_BITS {
                        let t = chunk[j];
                        if t != 0 {
                            p_word |= 1u64 << j;
                            let sw = sign_cursor >> 6;
                            if sw >= signs.len() {
                                signs.push(0u64);
                            }
                            if t < 0 {
                                signs[sw] |= 1u64 << (sign_cursor & 63);
                            }
                            sign_cursor += 1;
                        }
                    }
                    row_pres[w_idx] = p_word;
                }
            }
        }
        if sign_cursor > u32::MAX as usize {
            crate::bail!("encode_from_ptq1_0: non-zero count overflows u32");
        }
        sign_offsets.push(sign_cursor as u32);
        signs.truncate(sign_cursor.div_ceil(BITCOS_WORD_BITS));

        Ok(Self {
            rows,
            cols,
            group_size: config.group_size,
            superblock_elems: config.superblock_elems,
            superblocks_per_row: sb_per_row,
            nnz: sign_cursor,
            presence,
            signs,
            scales,
            sign_offsets,
        })
    }

    /// Decode a single `(row, sb)` superblock independently using only `sign_offsets`,
    /// proving warp-level / superblock-level random-access parallel decodability.
    pub fn decode_superblock_to_f32(&self, row: usize, sb: usize, dst: &mut [f32]) -> Result<()> {
        let words_per_row = self.cols / BITCOS_WORD_BITS;
        let sb_words = self.superblock_elems / BITCOS_WORD_BITS;
        let w_start = sb * sb_words;
        let w_end = (w_start + sb_words).min(words_per_row);
        let expected_len = (w_end - w_start) * BITCOS_WORD_BITS;
        if dst.len() != expected_len {
            crate::bail!(
                "decode_superblock_to_f32: dst len {} != expected {expected_len}",
                dst.len()
            );
        }
        let sb_idx = row * self.superblocks_per_row + sb;
        let mut bit_cursor = self.sign_offsets[sb_idx] as usize;
        let words_per_group = self.group_size.words_per_group();
        let groups_per_row = self.cols / self.group_size.elems();

        for (local_w, w_idx) in (w_start..w_end).enumerate() {
            let group_idx = w_idx / words_per_group;
            let d = self.scales[row * groups_per_row + group_idx].to_f32();
            let zero_val = 0.0f32 * d;
            let mask = self.presence[row * words_per_row + w_idx];
            let out_slice = &mut dst[local_w * BITCOS_WORD_BITS..(local_w + 1) * BITCOS_WORD_BITS];
            if mask == 0 {
                out_slice.fill(zero_val);
                continue;
            }
            let sbits = read_sign_bits_u64(&self.signs, bit_cursor);
            let mut local_s = 0u32;
            for j in 0..BITCOS_WORD_BITS {
                if ((mask >> j) & 1) == 0 {
                    out_slice[j] = zero_val;
                } else {
                    let is_neg = ((sbits >> local_s) & 1) != 0;
                    out_slice[j] = if is_neg { -d } else { d };
                    local_s += 1;
                }
            }
            bit_cursor += local_s as usize;
        }
        debug_assert_eq!(bit_cursor, self.sign_offsets[sb_idx + 1] as usize);
        Ok(())
    }

    /// Decode entire `[rows, cols]` matrix to `f32` (0.0 ULP vs PQ2_0 / PTQ1_0).
    pub fn decode_to_f32(&self, out: &mut [f32]) -> Result<()> {
        let total = self.rows * self.cols;
        if out.len() != total {
            crate::bail!("decode_to_f32: out len {} != expected {total}", out.len());
        }
        let sb_elems = self.superblock_elems;
        if self.rows >= PARALLEL_ROW_THRESHOLD {
            let out_addr = out.as_mut_ptr() as usize;
            let cols = self.cols;
            let sb_per_row = self.superblocks_per_row;
            crate::utils::barrier_pool().execute_chunked(self.rows, |range| {
                let out_ptr = out_addr as *mut f32;
                for r in range {
                    let row_slice =
                        unsafe { std::slice::from_raw_parts_mut(out_ptr.add(r * cols), cols) };
                    for sb in 0..sb_per_row {
                        let start = sb * sb_elems;
                        let end = (start + sb_elems).min(cols);
                        self.decode_superblock_to_f32(r, sb, &mut row_slice[start..end])
                            .expect("valid superblock slice");
                    }
                }
            });
        } else {
            for r in 0..self.rows {
                let row_slice = &mut out[r * self.cols..(r + 1) * self.cols];
                for sb in 0..self.superblocks_per_row {
                    let start = sb * sb_elems;
                    let end = (start + sb_elems).min(self.cols);
                    self.decode_superblock_to_f32(r, sb, &mut row_slice[start..end])?;
                }
            }
        }
        Ok(())
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        self.decode_to_f32(&mut out)?;
        Ok(out)
    }

    /// Compute a single row dot product against `BlockQ8_0`-quantized activation `ys`.
    #[inline]
    pub fn row_dot_q8_0(&self, row: usize, ys: &[BlockQ8_0]) -> f32 {
        debug_assert!(row < self.rows);
        debug_assert_eq!(ys.len(), self.cols / QK8_0);
        let words_per_row = self.cols / BITCOS_WORD_BITS;
        let row_pres = &self.presence[row * words_per_row..(row + 1) * words_per_row];
        let groups_per_row = self.cols / self.group_size.elems();
        let row_scales = &self.scales[row * groups_per_row..(row + 1) * groups_per_row];
        let bit_start = self.sign_offsets[row * self.superblocks_per_row] as usize;

        #[cfg(target_arch = "aarch64")]
        {
            if has_aarch64_dotprod() {
                return unsafe {
                    row_dot_q8_0_neon_dotprod(
                        row_pres,
                        &self.signs,
                        row_scales,
                        self.group_size,
                        bit_start,
                        ys,
                    )
                };
            }
            return unsafe {
                row_dot_q8_0_neon_base(
                    row_pres,
                    &self.signs,
                    row_scales,
                    self.group_size,
                    bit_start,
                    ys,
                )
            };
        }
        #[allow(unreachable_code)]
        row_dot_q8_0_scalar(
            row_pres,
            &self.signs,
            row_scales,
            self.group_size,
            bit_start,
            ys,
        )
    }

    /// Single-row scalar reference dot product against `BlockQ8_0` (exposed for parity tests).
    pub fn row_dot_q8_0_scalar_ref(&self, row: usize, ys: &[BlockQ8_0]) -> f32 {
        let words_per_row = self.cols / BITCOS_WORD_BITS;
        let row_pres = &self.presence[row * words_per_row..(row + 1) * words_per_row];
        let groups_per_row = self.cols / self.group_size.elems();
        let row_scales = &self.scales[row * groups_per_row..(row + 1) * groups_per_row];
        let bit_start = self.sign_offsets[row * self.superblocks_per_row] as usize;
        row_dot_q8_0_scalar(
            row_pres,
            &self.signs,
            row_scales,
            self.group_size,
            bit_start,
            ys,
        )
    }

    /// Parallel CPU `matvec` (`y = W * x`) using Q8_0 activation quantization, matching
    /// the CPU inference contract of `BlockPQ2_0` and `BlockPTQ1_0`.
    pub fn matvec_q8_0(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.cols {
            crate::bail!("matvec_q8_0: x len {} != cols {}", x.len(), self.cols);
        }
        if out.len() != self.rows {
            crate::bail!("matvec_q8_0: out len {} != rows {}", out.len(), self.rows);
        }
        let n_q8 = self.cols / QK8_0;
        let mut x_q8 = vec![BlockQ8_0::zeros(); n_q8];
        BlockQ8_0::from_float(x, &mut x_q8);
        self.matvec_prequant_q8_0(&x_q8, out)
    }

    /// Compute 4 consecutive row dot products against `BlockQ8_0` activations in lockstep,
    /// amortizing `x_q8` vector loads across 4 output rows just like `GgmlType::vec_dot_4`.
    #[inline]
    pub fn row_dot_4_q8_0(&self, row_base: usize, ys: &[BlockQ8_0]) -> (f32, f32, f32, f32) {
        debug_assert!(row_base + 4 <= self.rows);
        #[cfg(target_arch = "aarch64")]
        {
            if has_aarch64_dotprod() {
                return unsafe { self.row_dot_4_q8_0_neon_dotprod(row_base, ys) };
            }
        }
        (
            self.row_dot_q8_0(row_base, ys),
            self.row_dot_q8_0(row_base + 1, ys),
            self.row_dot_q8_0(row_base + 2, ys),
            self.row_dot_q8_0(row_base + 3, ys),
        )
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon,dotprod")]
    unsafe fn row_dot_4_q8_0_neon_dotprod(
        &self,
        row_base: usize,
        ys: &[BlockQ8_0],
    ) -> (f32, f32, f32, f32) {
        use std::arch::aarch64::*;
        let words_per_row = self.cols / BITCOS_WORD_BITS;
        let groups_per_row = self.cols / self.group_size.elems();
        let words_per_group = self.group_size.words_per_group();

        let p0 = self.presence.as_ptr().add(row_base * words_per_row);
        let p1 = p0.add(words_per_row);
        let p2 = p1.add(words_per_row);
        let p3 = p2.add(words_per_row);

        let sc0 = self.scales.as_ptr().add(row_base * groups_per_row);
        let sc1 = sc0.add(groups_per_row);
        let sc2 = sc1.add(groups_per_row);
        let sc3 = sc2.add(groups_per_row);

        let mut c0 = self.sign_offsets[row_base * self.superblocks_per_row] as usize;
        let mut c1 = self.sign_offsets[(row_base + 1) * self.superblocks_per_row] as usize;
        let mut c2 = self.sign_offsets[(row_base + 2) * self.superblocks_per_row] as usize;
        let mut c3 = self.sign_offsets[(row_base + 3) * self.superblocks_per_row] as usize;

        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;
        let zero_acc = vdupq_n_s32(0);

        for g_idx in 0..groups_per_row {
            let mut g0 = 0.0f32;
            let mut g1 = 0.0f32;
            let mut g2 = 0.0f32;
            let mut g3 = 0.0f32;
            for sub_w in 0..words_per_group {
                let w_idx = g_idx * words_per_group + sub_w;
                let yb_lo = ys.get_unchecked(w_idx * 2);
                let yb_hi = ys.get_unchecked(w_idx * 2 + 1);
                let dy_lo = yb_lo.d.to_f32();
                let dy_hi = yb_hi.d.to_f32();
                let y0_lo = vld1q_s8(yb_lo.qs.as_ptr());
                let y1_lo = vld1q_s8(yb_lo.qs.as_ptr().add(16));
                let y0_hi = vld1q_s8(yb_hi.qs.as_ptr());
                let y1_hi = vld1q_s8(yb_hi.qs.as_ptr().add(16));

                for (pres_ptr, cur, g_acc) in [
                    (p0, &mut c0, &mut g0),
                    (p1, &mut c1, &mut g1),
                    (p2, &mut c2, &mut g2),
                    (p3, &mut c3, &mut g3),
                ] {
                    let mask = *pres_ptr.add(w_idx);
                    if mask != 0 {
                        let sbits64 = read_sign_bits_u64(&self.signs, *cur);
                        let m_lo = mask as u32;
                        let m_hi = (mask >> 32) as u32;
                        let pop_lo = if m_lo != 0 {
                            let (q0, q1, p_lo) = unpack_m32_to_neon(m_lo, sbits64);
                            let acc = vdotq_s32(vdotq_s32(zero_acc, q0, y0_lo), q1, y1_lo);
                            *g_acc += dy_lo * (vaddvq_s32(acc) as f32);
                            p_lo
                        } else {
                            0
                        };
                        let pop_hi = if m_hi != 0 {
                            let (q0, q1, p_hi) = unpack_m32_to_neon(m_hi, sbits64 >> pop_lo);
                            let acc = vdotq_s32(vdotq_s32(zero_acc, q0, y0_hi), q1, y1_hi);
                            *g_acc += dy_hi * (vaddvq_s32(acc) as f32);
                            p_hi
                        } else {
                            0
                        };
                        *cur += (pop_lo + pop_hi) as usize;
                    }
                }
            }
            sum0 += (*sc0.add(g_idx)).to_f32() * g0;
            sum1 += (*sc1.add(g_idx)).to_f32() * g1;
            sum2 += (*sc2.add(g_idx)).to_f32() * g2;
            sum3 += (*sc3.add(g_idx)).to_f32() * g3;
        }
        (sum0, sum1, sum2, sum3)
    }

    /// Parallel CPU `matvec` over a pre-quantized `BlockQ8_0` activation vector.
    pub fn matvec_prequant_q8_0(&self, x_q8: &[BlockQ8_0], out: &mut [f32]) -> Result<()> {
        if x_q8.len() != self.cols / QK8_0 {
            crate::bail!(
                "matvec_prequant_q8_0: x_q8 len {} != expected {}",
                x_q8.len(),
                self.cols / QK8_0
            );
        }
        if out.len() != self.rows {
            crate::bail!(
                "matvec_prequant_q8_0: out len {} != rows {}",
                out.len(),
                self.rows
            );
        }
        let n_quad = self.rows & !3;
        let quads_total = n_quad / 4;
        if self.rows >= PARALLEL_ROW_THRESHOLD && quads_total > 0 {
            let (main, tail) = out.split_at_mut(n_quad);
            let main_addr = main.as_mut_ptr() as usize;
            crate::utils::barrier_pool().execute_chunked(quads_total, |range| {
                let main_ptr = main_addr as *mut f32;
                for q in range {
                    let r = q * 4;
                    let (d0, d1, d2, d3) = self.row_dot_4_q8_0(r, x_q8);
                    unsafe {
                        let dst = main_ptr.add(r);
                        *dst = d0;
                        *dst.add(1) = d1;
                        *dst.add(2) = d2;
                        *dst.add(3) = d3;
                    }
                }
            });
            for (i, dst) in tail.iter_mut().enumerate() {
                *dst = self.row_dot_q8_0(n_quad + i, x_q8);
            }
        } else {
            for q in 0..quads_total {
                let r = q * 4;
                let (d0, d1, d2, d3) = self.row_dot_4_q8_0(r, x_q8);
                out[r] = d0;
                out[r + 1] = d1;
                out[r + 2] = d2;
                out[r + 3] = d3;
            }
            for r in n_quad..self.rows {
                out[r] = self.row_dot_q8_0(r, x_q8);
            }
        }
        Ok(())
    }

    /// Direct exact `f32` `matvec` (`y = W * x`) with zero activation quantization error.
    pub fn matvec_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.cols {
            crate::bail!("matvec_f32: x len {} != cols {}", x.len(), self.cols);
        }
        if out.len() != self.rows {
            crate::bail!("matvec_f32: out len {} != rows {}", out.len(), self.rows);
        }
        if self.rows >= PARALLEL_ROW_THRESHOLD {
            let out_addr = out.as_mut_ptr() as usize;
            crate::utils::barrier_pool().execute_chunked(self.rows, |range| {
                let out_ptr = out_addr as *mut f32;
                for r in range {
                    let v = self.row_dot_f32(r, x);
                    unsafe { *out_ptr.add(r) = v };
                }
            });
        } else {
            for (r, dst) in out.iter_mut().enumerate() {
                *dst = self.row_dot_f32(r, x);
            }
        }
        Ok(())
    }

    #[inline]
    pub fn row_dot_f32(&self, row: usize, x: &[f32]) -> f32 {
        let words_per_row = self.cols / BITCOS_WORD_BITS;
        let row_pres = &self.presence[row * words_per_row..(row + 1) * words_per_row];
        let groups_per_row = self.cols / self.group_size.elems();
        let row_scales = &self.scales[row * groups_per_row..(row + 1) * groups_per_row];
        let words_per_group = self.group_size.words_per_group();
        let mut bit_cursor = self.sign_offsets[row * self.superblocks_per_row] as usize;
        let mut total = 0.0f32;

        for (g_idx, &scale_f16) in row_scales.iter().enumerate() {
            let d = scale_f16.to_f32();
            let mut group_acc = 0.0f32;
            for sub_w in 0..words_per_group {
                let w_idx = g_idx * words_per_group + sub_w;
                let mut mask = row_pres[w_idx];
                if mask == 0 {
                    continue;
                }
                let sbits = read_sign_bits_u64(&self.signs, bit_cursor);
                let x_sub = &x[w_idx * BITCOS_WORD_BITS..(w_idx + 1) * BITCOS_WORD_BITS];
                let mut s_cur = 0u32;
                while mask != 0 {
                    let lane = mask.trailing_zeros() as usize;
                    let xv = unsafe { *x_sub.get_unchecked(lane) };
                    let sign_neg = ((sbits >> s_cur) & 1) != 0;
                    group_acc += if sign_neg { -xv } else { xv };
                    s_cur += 1;
                    mask &= mask - 1;
                }
                bit_cursor += s_cur as usize;
            }
            total += d * group_acc;
        }
        total
    }
}

#[inline(always)]
fn read_sign_bits_u64(signs: &[u64], bit_cursor: usize) -> u64 {
    let word_idx = bit_cursor >> 6;
    let bit_off = bit_cursor & 63;
    if word_idx >= signs.len() {
        return 0;
    }
    let lo = unsafe { *signs.get_unchecked(word_idx) };
    if bit_off == 0 {
        return lo;
    }
    let hi = if word_idx + 1 < signs.len() {
        unsafe { *signs.get_unchecked(word_idx + 1) }
    } else {
        0
    };
    (lo >> bit_off) | (hi << (64 - bit_off))
}

/// Branchless sparse dot product for scalar reference verification.
#[inline(always)]
fn sparse_dot32_s8(mut m32: u32, sbits: u64, qs: &[i8; QK8_0]) -> i32 {
    let mut isum = 0i32;
    let mut k = 0u32;
    while m32 != 0 {
        let idx = m32.trailing_zeros() as usize;
        let yv = unsafe { *qs.get_unchecked(idx) } as i32;
        let neg = ((sbits >> k) & 1) as i32;
        isum += (yv ^ -neg) + neg;
        k += 1;
        m32 &= m32 - 1;
    }
    isum
}

fn row_dot_q8_0_scalar(
    row_pres: &[u64],
    signs: &[u64],
    row_scales: &[f16],
    group_size: BitcosGroupSize,
    mut bit_cursor: usize,
    ys: &[BlockQ8_0],
) -> f32 {
    let words_per_group = group_size.words_per_group();
    let mut sumf = 0.0f32;
    for (g_idx, &scale_f16) in row_scales.iter().enumerate() {
        let d0 = scale_f16.to_f32();
        let mut sumi = 0.0f32;
        for sub_w in 0..words_per_group {
            let w_idx = g_idx * words_per_group + sub_w;
            let mask = row_pres[w_idx];
            if mask == 0 {
                continue;
            }
            let sbits64 = read_sign_bits_u64(signs, bit_cursor);
            let m_lo = mask as u32;
            let m_hi = (mask >> 32) as u32;
            let pop_lo = m_lo.count_ones();
            let pop_hi = m_hi.count_ones();
            if m_lo != 0 {
                let yb = &ys[w_idx * 2];
                let isum = sparse_dot32_s8(m_lo, sbits64, &yb.qs);
                sumi += yb.d.to_f32() * (isum as f32);
            }
            if m_hi != 0 {
                let yb = &ys[w_idx * 2 + 1];
                let isum = sparse_dot32_s8(m_hi, sbits64 >> pop_lo, &yb.qs);
                sumi += yb.d.to_f32() * (isum as f32);
            }
            bit_cursor += (pop_lo + pop_hi) as usize;
        }
        sumf += d0 * sumi;
    }
    sumf
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn has_aarch64_dotprod() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0);
    let v = CACHED.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
    let detected = std::arch::is_aarch64_feature_detected!("dotprod");
    CACHED.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
    detected
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_m32_to_neon(
    m32: u32,
    sbits: u64,
) -> (
    std::arch::aarch64::int8x16_t,
    std::arch::aarch64::int8x16_t,
    u32,
) {
    use std::arch::aarch64::*;
    // 4 byte lookups per 32 weights; popcnt comes from BYTE_TABLES.0 >> 24 in GPRs.
    let b0 = (m32 & 0xFF) as usize;
    let b1 = ((m32 >> 8) & 0xFF) as usize;
    let b2 = ((m32 >> 16) & 0xFF) as usize;
    let b3 = (m32 >> 24) as usize;

    let meta_ptr = BYTE_TABLES.0.as_ptr();
    let trits_ptr = BYTE_TABLES.1.as_ptr();

    let m0 = *meta_ptr.add(b0);
    let m1 = *meta_ptr.add(b1);
    let m2 = *meta_ptr.add(b2);
    let m3 = *meta_ptr.add(b3);

    let p0 = m0 >> 24;
    let p1 = m1 >> 24;
    let p2 = m2 >> 24;
    let p3 = m3 >> 24;

    let off1 = p0;
    let off2 = p0 + p1;
    let off3 = off2 + p2;
    let pop_total = off3 + p3;

    let s0 = (sbits as usize) & (((m0 >> 16) & 0xFF) as usize);
    let s1 = ((sbits >> off1) as usize) & (((m1 >> 16) & 0xFF) as usize);
    let s2 = ((sbits >> off2) as usize) & (((m2 >> 16) & 0xFF) as usize);
    let s3 = ((sbits >> off3) as usize) & (((m3 >> 16) & 0xFF) as usize);

    let w0 = *trits_ptr.add(((m0 & 0xFFFF) as usize) + s0);
    let w1 = *trits_ptr.add(((m1 & 0xFFFF) as usize) + s1);
    let w2 = *trits_ptr.add(((m2 & 0xFFFF) as usize) + s2);
    let w3 = *trits_ptr.add(((m3 & 0xFFFF) as usize) + s3);

    let q0 = vcombine_s8(vcreate_s8(w0), vcreate_s8(w1));
    let q1 = vcombine_s8(vcreate_s8(w2), vcreate_s8(w3));
    (q0, q1, pop_total)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn row_dot_q8_0_neon_dotprod(
    row_pres: &[u64],
    signs: &[u64],
    row_scales: &[f16],
    group_size: BitcosGroupSize,
    mut bit_cursor: usize,
    ys: &[BlockQ8_0],
) -> f32 {
    use std::arch::aarch64::*;
    let words_per_group = group_size.words_per_group();
    let mut sumf = 0.0f32;
    let zero_acc = vdupq_n_s32(0);

    for (g_idx, &scale_f16) in row_scales.iter().enumerate() {
        let d0 = scale_f16.to_f32();
        let mut sumi = 0.0f32;
        for sub_w in 0..words_per_group {
            let w_idx = g_idx * words_per_group + sub_w;
            let mask = *row_pres.get_unchecked(w_idx);
            if mask == 0 {
                continue;
            }
            let sbits64 = read_sign_bits_u64(signs, bit_cursor);
            let m_lo = mask as u32;
            let m_hi = (mask >> 32) as u32;

            let pop_lo = if m_lo != 0 {
                let yb = ys.get_unchecked(w_idx * 2);
                let (q0, q1, p_lo) = unpack_m32_to_neon(m_lo, sbits64);
                let y0 = vld1q_s8(yb.qs.as_ptr());
                let y1 = vld1q_s8(yb.qs.as_ptr().add(16));
                let acc = vdotq_s32(vdotq_s32(zero_acc, q0, y0), q1, y1);
                sumi += yb.d.to_f32() * (vaddvq_s32(acc) as f32);
                p_lo
            } else {
                0
            };
            let pop_hi = if m_hi != 0 {
                let yb = ys.get_unchecked(w_idx * 2 + 1);
                let (q0, q1, p_hi) = unpack_m32_to_neon(m_hi, sbits64 >> pop_lo);
                let y0 = vld1q_s8(yb.qs.as_ptr());
                let y1 = vld1q_s8(yb.qs.as_ptr().add(16));
                let acc = vdotq_s32(vdotq_s32(zero_acc, q0, y0), q1, y1);
                sumi += yb.d.to_f32() * (vaddvq_s32(acc) as f32);
                p_hi
            } else {
                0
            };
            bit_cursor += (pop_lo + pop_hi) as usize;
        }
        sumf += d0 * sumi;
    }
    sumf
}

#[cfg(target_arch = "aarch64")]
unsafe fn row_dot_q8_0_neon_base(
    row_pres: &[u64],
    signs: &[u64],
    row_scales: &[f16],
    group_size: BitcosGroupSize,
    mut bit_cursor: usize,
    ys: &[BlockQ8_0],
) -> f32 {
    use std::arch::aarch64::*;
    let words_per_group = group_size.words_per_group();
    let mut sumf = 0.0f32;

    for (g_idx, &scale_f16) in row_scales.iter().enumerate() {
        let d0 = scale_f16.to_f32();
        let mut sumi = 0.0f32;
        for sub_w in 0..words_per_group {
            let w_idx = g_idx * words_per_group + sub_w;
            let mask = *row_pres.get_unchecked(w_idx);
            if mask == 0 {
                continue;
            }
            let sbits64 = read_sign_bits_u64(signs, bit_cursor);
            let m_lo = mask as u32;
            let m_hi = (mask >> 32) as u32;

            let pop_lo = if m_lo != 0 {
                let yb = ys.get_unchecked(w_idx * 2);
                let (q0, q1, p_lo) = unpack_m32_to_neon(m_lo, sbits64);
                let y0 = vld1q_s8(yb.qs.as_ptr());
                let y1 = vld1q_s8(yb.qs.as_ptr().add(16));
                let s0 = vaddq_s16(
                    vmull_s8(vget_low_s8(q0), vget_low_s8(y0)),
                    vmull_high_s8(q0, y0),
                );
                let s1 = vaddq_s16(
                    vmull_s8(vget_low_s8(q1), vget_low_s8(y1)),
                    vmull_high_s8(q1, y1),
                );
                let isum = vaddlvq_s16(vaddq_s16(s0, s1));
                sumi += yb.d.to_f32() * (isum as f32);
                p_lo
            } else {
                0
            };
            let pop_hi = if m_hi != 0 {
                let yb = ys.get_unchecked(w_idx * 2 + 1);
                let (q0, q1, p_hi) = unpack_m32_to_neon(m_hi, sbits64 >> pop_lo);
                let y0 = vld1q_s8(yb.qs.as_ptr());
                let y1 = vld1q_s8(yb.qs.as_ptr().add(16));
                let s0 = vaddq_s16(
                    vmull_s8(vget_low_s8(q0), vget_low_s8(y0)),
                    vmull_high_s8(q0, y0),
                );
                let s1 = vaddq_s16(
                    vmull_s8(vget_low_s8(q1), vget_low_s8(y1)),
                    vmull_high_s8(q1, y1),
                );
                let isum = vaddlvq_s16(vaddq_s16(s0, s1));
                sumi += yb.d.to_f32() * (isum as f32);
                p_hi
            } else {
                0
            };
            bit_cursor += (pop_lo + pop_hi) as usize;
        }
        sumf += d0 * sumi;
    }
    sumf
}

/// Pack ternary trits in {-1, 0, +1} and per-128-element `f16` scales into `BlockPQ2_0`.
pub fn pack_trits_to_pq2_0(trits: &[i8], scales: &[f16]) -> Result<Vec<BlockPQ2_0>> {
    if !trits.len().is_multiple_of(QK_PQ2_0) {
        crate::bail!(
            "pack_trits_to_pq2_0: trits len {} not divisible by {QK_PQ2_0}",
            trits.len()
        );
    }
    let nb = trits.len() / QK_PQ2_0;
    if scales.len() != nb {
        crate::bail!(
            "pack_trits_to_pq2_0: scales len {} != expected {nb}",
            scales.len()
        );
    }
    let mut blocks = Vec::with_capacity(nb);
    for bi in 0..nb {
        let mut qs = [0u8; QK_PQ2_0 / 4];
        for j in 0..QK_PQ2_0 {
            let t = trits[bi * QK_PQ2_0 + j];
            if !(-1..=1).contains(&t) {
                crate::bail!("pack_trits_to_pq2_0: invalid trit {t}");
            }
            let code = (t + 1) as u8;
            qs[j / 4] |= (code & PQ2_CODE_MASK) << ((j % 4) * 2);
        }
        blocks.push(BlockPQ2_0 { d: scales[bi], qs });
    }
    Ok(blocks)
}

/// Pack ternary trits in {-1, 0, +1} and per-128-element `f16` scales into `BlockPTQ1_0`.
pub fn pack_trits_to_ptq1_0(trits: &[i8], scales: &[f16]) -> Result<Vec<BlockPTQ1_0>> {
    if !trits.len().is_multiple_of(QK_PTQ1_0) {
        crate::bail!(
            "pack_trits_to_ptq1_0: trits len {} not divisible by {QK_PTQ1_0}",
            trits.len()
        );
    }
    let nb = trits.len() / QK_PTQ1_0;
    if scales.len() != nb {
        crate::bail!(
            "pack_trits_to_ptq1_0: scales len {} != expected {nb}",
            scales.len()
        );
    }
    let mut blocks = Vec::with_capacity(nb);
    for bi in 0..nb {
        let src = &trits[bi * QK_PTQ1_0..(bi + 1) * QK_PTQ1_0];
        for &t in src {
            if !(-1..=1).contains(&t) {
                crate::bail!("pack_trits_to_ptq1_0: invalid trit {t}");
            }
        }
        let mut qs = [0u8; 24];
        for m in 0..PTQ1_QS_STAGE0_BYTES {
            let mut val = 0u32;
            for n in 0..5 {
                val = val * 3 + (src[n * PTQ1_QS_STAGE0_BYTES + m] + 1) as u32;
            }
            qs[m] = ((val * 256).div_ceil(243)) as u8;
        }
        for m in 0..PTQ1_QS_STAGE1_BYTES {
            let mut val = 0u32;
            for n in 0..5 {
                val = val * 3 + (src[80 + n * PTQ1_QS_STAGE1_BYTES + m] + 1) as u32;
            }
            qs[PTQ1_QS_STAGE0_BYTES + m] = ((val * 256).div_ceil(243)) as u8;
        }
        let mut qh = [0u8; PTQ1_QH_BYTES];
        for m in 0..PTQ1_QH_BYTES {
            let mut val = 0u32;
            for n in 0..4 {
                val = val * 3 + (src[120 + n * PTQ1_QH_BYTES + m] + 1) as u32;
            }
            qh[m] = ((val * 3 * 256).div_ceil(243)) as u8;
        }
        blocks.push(BlockPTQ1_0 {
            qs,
            qh,
            d: scales[bi],
        });
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_next(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *state
    }

    fn make_synthetic_trits(rows: usize, cols: usize, zero_frac: f64, seed: u64) -> Vec<i8> {
        let mut st = seed;
        let z_thresh = (zero_frac.clamp(0.0, 1.0) * (u32::MAX as f64)) as u32;
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            let r = (lcg_next(&mut st) >> 32) as u32;
            if r < z_thresh {
                out.push(0);
            } else if (r & 1) == 0 {
                out.push(1);
            } else {
                out.push(-1);
            }
        }
        out
    }

    #[test]
    fn pq2_0_roundtrip_zero_ulp_all_densities() {
        let rows = 96;
        let cols = 512;
        for (idx, &z) in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0].iter().enumerate() {
            let trits = make_synthetic_trits(rows, cols, z, 0xBEEF_0000 + idx as u64);
            let nb = (rows * cols) / QK_PQ2_0;
            let scales: Vec<f16> = (0..nb)
                .map(|i| {
                    let sign = if i % 7 == 0 { -1.0 } else { 1.0 };
                    let mag = if i % 19 == 0 {
                        0.0
                    } else {
                        0.013 * ((i % 23) as f32 + 1.0)
                    };
                    f16::from_f32(sign * mag)
                })
                .collect();
            let pq2 = pack_trits_to_pq2_0(&trits, &scales).unwrap();
            let mut want = vec![0.0f32; rows * cols];
            BlockPQ2_0::to_float(&pq2, &mut want);

            for cfg in [
                BitcosConfig::new(BitcosGroupSize::G128, 512),
                BitcosConfig::new(BitcosGroupSize::G128, 128),
                BitcosConfig::new(BitcosGroupSize::G64, 256),
                BitcosConfig::new(BitcosGroupSize::G64, 64),
            ] {
                let bm = BitcosMatrix::encode_from_pq2_0_with_config(&pq2, rows, cols, cfg)
                    .expect("encode pq2_0");
                let got = bm.to_vec_f32().expect("decode bitcos");
                assert_eq!(got.len(), want.len());
                for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "0.0 ULP mismatch at z={z}, cfg={cfg:?}, elem={k}: got={g} ({:#x}) want={w} ({:#x})",
                        g.to_bits(),
                        w.to_bits()
                    );
                }
            }
        }
    }

    #[test]
    fn ptq1_0_roundtrip_zero_ulp_all_densities() {
        let rows = 80;
        let cols = 512;
        for (idx, &z) in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0].iter().enumerate() {
            let trits = make_synthetic_trits(rows, cols, z, 0xCAFE_1000 + idx as u64);
            let nb = (rows * cols) / QK_PTQ1_0;
            let scales: Vec<f16> = (0..nb)
                .map(|i| {
                    let sign = if i % 5 == 0 { -1.0 } else { 1.0 };
                    let mag = if i % 17 == 0 {
                        0.0
                    } else {
                        0.021 * ((i % 31) as f32 + 1.0)
                    };
                    f16::from_f32(sign * mag)
                })
                .collect();
            let ptq1 = pack_trits_to_ptq1_0(&trits, &scales).unwrap();
            let mut want = vec![0.0f32; rows * cols];
            BlockPTQ1_0::to_float(&ptq1, &mut want);

            for cfg in [
                BitcosConfig::new(BitcosGroupSize::G128, 512),
                BitcosConfig::new(BitcosGroupSize::G64, 128),
            ] {
                let bm = BitcosMatrix::encode_from_ptq1_0_with_config(&ptq1, rows, cols, cfg)
                    .expect("encode ptq1_0");
                let got = bm.to_vec_f32().expect("decode bitcos");
                for (k, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        w.to_bits(),
                        "0.0 ULP mismatch on PTQ1_0 at z={z}, elem={k}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_non_ternary_pq2_code() {
        let mut blk = BlockPQ2_0 {
            d: f16::from_f32(0.5),
            qs: [0x55; 32],
        };
        blk.qs[7] |= 0x03; // inject code 3 (+2)
        let err = BitcosMatrix::encode_from_pq2_0(&[blk], 1, 128);
        assert!(err.is_err());
    }

    #[test]
    fn matvec_q8_0_and_f32_parity_vs_pq2_and_ptq1() {
        let rows = 128;
        let cols = 1024;
        let mut st = 0x1234_5678_9ABC_DEF0u64;
        let x: Vec<f32> = (0..cols)
            .map(|_| ((lcg_next(&mut st) >> 40) as i32 - 8_388_608) as f32 / 4_194_304.0)
            .collect();

        for &z in &[0.20, 0.35, 0.50, 0.65, 0.80] {
            let trits = make_synthetic_trits(rows, cols, z, 0x7777 + (z * 100.0) as u64);
            let nb = (rows * cols) / QK_PQ2_0;
            let scales: Vec<f16> = (0..nb)
                .map(|i| f16::from_f32(0.03 * ((i % 13) as f32 + 1.0)))
                .collect();
            let pq2 = pack_trits_to_pq2_0(&trits, &scales).unwrap();
            let bm128 = BitcosMatrix::encode_from_pq2_0(&pq2, rows, cols).unwrap();
            let bm64 = BitcosMatrix::encode_from_pq2_0_with_config(
                &pq2,
                rows,
                cols,
                BitcosConfig::new(BitcosGroupSize::G64, 256),
            )
            .unwrap();

            let mut x_q8 = vec![BlockQ8_0::zeros(); cols / QK8_0];
            BlockQ8_0::from_float(&x, &mut x_q8);

            let mut out_bitcos128 = vec![0.0f32; rows];
            let mut out_bitcos64 = vec![0.0f32; rows];
            bm128.matvec_q8_0(&x, &mut out_bitcos128).unwrap();
            bm64.matvec_q8_0(&x, &mut out_bitcos64).unwrap();

            for r in 0..rows {
                let pq2_row = &pq2[r * (cols / QK_PQ2_0)..(r + 1) * (cols / QK_PQ2_0)];
                let want_pq2 = BlockPQ2_0::vec_dot(cols, pq2_row, &x_q8);
                let scalar_ref = bm128.row_dot_q8_0_scalar_ref(r, &x_q8);
                assert_eq!(
                    out_bitcos128[r].to_bits(),
                    want_pq2.to_bits(),
                    "Q8_0 row {r} bitcos128 != pq2_0 at z={z}: {} vs {}",
                    out_bitcos128[r],
                    want_pq2
                );
                assert_eq!(
                    out_bitcos128[r].to_bits(),
                    scalar_ref.to_bits(),
                    "NEON vs scalar mismatch at row {r}, z={z}"
                );
                assert!(
                    (out_bitcos64[r] - want_pq2).abs() <= 1e-4 * want_pq2.abs().max(1.0),
                    "G64 vs G128 drift at row {r}, z={z}: {} vs {}",
                    out_bitcos64[r],
                    want_pq2
                );
            }

            let wf32 = bm128.to_vec_f32().unwrap();
            let mut out_f32 = vec![0.0f32; rows];
            bm128.matvec_f32(&x, &mut out_f32).unwrap();
            for r in 0..rows {
                let mut exact = 0.0f32;
                for c in 0..cols {
                    exact += wf32[r * cols + c] * x[c];
                }
                let diff = (out_f32[r] - exact).abs();
                assert!(
                    diff <= 1e-4 * exact.abs().max(1.0),
                    "matvec_f32 mismatch at row {r}, z={z}: got={} want={exact}",
                    out_f32[r]
                );
            }
        }
    }

    #[test]
    fn swar_scanner_matches_exact_counts() {
        let rows = 16;
        let cols = 512;
        let trits = make_synthetic_trits(rows, cols, 0.55, 0x9999);
        // Inject structured all-zero 64-weight and 32-weight blocks in row 0.
        let mut trits = trits;
        trits[0..128].fill(0);
        trits[128..160].fill(0);
        let nb = (rows * cols) / QK_PQ2_0;
        let mut scales: Vec<f16> = (0..nb).map(|_| f16::from_f32(0.25)).collect();
        scales[0] = f16::from_f32(0.0);

        let pq2 = pack_trits_to_pq2_0(&trits, &scales).unwrap();
        let ptq1 = pack_trits_to_ptq1_0(&trits, &scales).unwrap();
        let pq2_bytes = unsafe {
            std::slice::from_raw_parts(
                pq2.as_ptr() as *const u8,
                pq2.len() * std::mem::size_of::<BlockPQ2_0>(),
            )
        };
        let ptq1_bytes = unsafe {
            std::slice::from_raw_parts(
                ptq1.as_ptr() as *const u8,
                ptq1.len() * std::mem::size_of::<BlockPTQ1_0>(),
            )
        };
        let m_pq2 = scan_pq2_0_bytes(pq2_bytes).unwrap();
        let m_ptq1 = scan_ptq1_0_bytes(ptq1_bytes).unwrap();
        let want_z = trits.iter().filter(|&&t| t == 0).count() as u64;
        let want_p = trits.iter().filter(|&&t| t == 1).count() as u64;
        let want_n = trits.iter().filter(|&&t| t == -1).count() as u64;
        assert_eq!(m_pq2.zero_weights, want_z);
        assert_eq!(m_pq2.pos_weights, want_p);
        assert_eq!(m_pq2.neg_weights, want_n);
        assert_eq!(m_pq2.non_ternary_weights, 0);
        assert_eq!(m_pq2.zero_scale_blocks_128, 1);
        assert_eq!(m_ptq1.zero_weights, want_z);
        assert_eq!(m_ptq1.pos_weights, want_p);
        assert_eq!(m_ptq1.neg_weights, want_n);
        assert_eq!(m_pq2.zero_blocks_64, m_ptq1.zero_blocks_64);
        assert_eq!(m_pq2.zero_blocks_32, m_ptq1.zero_blocks_32);
        assert_eq!(m_pq2.zero_blocks_16, m_ptq1.zero_blocks_16);
        assert!(m_pq2.zero_blocks_64 >= 2);
        assert!(m_pq2.zero_blocks_32 >= 5);
    }

    #[test]
    fn profile_guided_selector_and_manifest() {
        // Tiny matrix chooses PQ2_0.
        assert_eq!(choose_ternary_layout(0.33, 64, 256), TernaryLayout::Pq2_0);
        // Bonsai-2 empirical z ~ 0.3276 on large matrix chooses PTQ1_0 (1.75 bpw < 1.86 bpw).
        assert_eq!(
            choose_ternary_layout(0.3276, 4096, 12288),
            TernaryLayout::Ptq1_0
        );
        // High zero density z = 0.65 chooses BitcosG128 (~1.54 bpw < 1.75 bpw).
        assert_eq!(
            choose_ternary_layout(0.65, 4096, 12288),
            TernaryLayout::BitcosG128
        );
        // Group-64-only column count chooses BitcosG64.
        assert_eq!(
            choose_ternary_layout(0.50, 1024, 192),
            TernaryLayout::BitcosG64
        );
        // UpgradePq2Only upgrades z = 0.3276 PQ2_0 weights to BitcosG128 (saves ~12.5% vs PQ2_0).
        assert_eq!(
            choose_ternary_layout_with_policy(
                0.3276,
                4096,
                12288,
                TernarySelectionPolicy::UpgradePq2Only
            ),
            TernaryLayout::BitcosG128
        );

        let manifest = TernaryFormatManifest::from_tensor_stats(
            &[
                ("blk.0.ffn_gate.weight".into(), 6144, 2048, 0.3276),
                ("sparse_head.weight".into(), 4096, 4096, 0.65),
            ],
            TernarySelectionPolicy::Balanced,
        );
        assert_eq!(manifest.tensors.len(), 2);
        assert_eq!(
            manifest.tensors[0].recommended_layout,
            TernaryLayout::Ptq1_0
        );
        assert_eq!(
            manifest.tensors[1].recommended_layout,
            TernaryLayout::BitcosG128
        );
        assert!(manifest.aggregate_bpw < PTQ1_0_BPW);
    }

    #[test]
    fn qtensor_to_bitcos_matrix_roundtrip() {
        let rows = 32;
        let cols = 256;
        let trits = make_synthetic_trits(rows, cols, 0.45, 0x4242);
        let scales: Vec<f16> = (0..(rows * cols / QK_PQ2_0))
            .map(|i| f16::from_f32(0.05 * (i as f32 + 1.0)))
            .collect();
        let pq2 = pack_trits_to_pq2_0(&trits, &scales).unwrap();
        let raw = unsafe {
            std::slice::from_raw_parts(
                pq2.as_ptr() as *const u8,
                pq2.len() * std::mem::size_of::<BlockPQ2_0>(),
            )
        };
        let qt = crate::quantized::ggml_file::qtensor_from_ggml(
            crate::quantized::GgmlDType::PQ2_0,
            raw,
            vec![rows, cols],
            &crate::Device::Cpu,
        )
        .unwrap();
        let bm = qt.to_bitcos_matrix().unwrap();
        let got = bm.to_vec_f32().unwrap();
        let want = qt
            .dequantize(&crate::Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits());
        }
    }
}
