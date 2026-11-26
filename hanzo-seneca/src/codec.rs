//! Pure-Rust (`std`-only) byte compression, integrity checks, content hashing,
//! and deterministic PRNG utilities for `hanzo-seneca`.
//!
//! No external crates and no `unsafe` code are used anywhere in this module.

use std::fmt;

/// Magic header for Seneca clean-room LZ/RLE compressed shard frames.
const FRAME_MAGIC: [u8; 4] = *b"SNCA";
/// Header length: 4B magic + 4B uncompressed length + 4B IEEE CRC32.
const HEADER_LEN: usize = 12;

/// Maximum uncompressed payload supported by the frame format (64 MiB).
const MAX_FRAME_LEN: usize = 64 * 1024 * 1024;

/// Errors returned when decoding or validating a compressed shard frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    TruncatedHeader,
    InvalidMagic,
    PayloadTooLarge(usize),
    UnexpectedEof,
    InvalidBackReference { offset: usize, output_len: usize },
    LengthMismatch { expected: usize, actual: usize },
    ChecksumMismatch { expected: u32, actual: u32 },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TruncatedHeader => write!(f, "compressed frame header is truncated"),
            Self::InvalidMagic => write!(f, "compressed frame magic mismatch"),
            Self::PayloadTooLarge(n) => write!(f, "frame uncompressed length {n} exceeds limit"),
            Self::UnexpectedEof => write!(f, "unexpected end of compressed stream"),
            Self::InvalidBackReference { offset, output_len } => write!(
                f,
                "invalid LZ back-reference offset {offset} at output position {output_len}"
            ),
            Self::LengthMismatch { expected, actual } => {
                write!(
                    f,
                    "decompressed length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::ChecksumMismatch { expected, actual } => write!(
                f,
                "CRC32 checksum mismatch: expected 0x{expected:08x}, got 0x{actual:08x}"
            ),
        }
    }
}

impl std::error::Error for CodecError {}

/// Compile-time IEEE 802.3 CRC32 lookup table (polynomial 0xEDB88320).
const CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0usize;
        while bit < 8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Computes the standard IEEE 802.3 CRC32 checksum of `data`.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        let idx = ((crc ^ u32::from(b)) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[idx];
    }
    !crc
}

/// Computes a deterministic 128-bit domain-separated content hash (two independent
/// 64-bit FNV-1a lanes with SplitMix64 avalanche finalization).
pub fn content_hash_128(domain: &[u8], data: &[u8]) -> u128 {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325;
    let mut h2: u64 = 0x8422_2325_cbf2_9ce4 ^ 0x9e37_79b9_7f4a_7c15;

    for &b in domain {
        h1 ^= u64::from(b);
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
        h2 ^= u64::from(b).wrapping_add(0x55);
        h2 = h2.wrapping_mul(0x0000_0100_0000_01b3).rotate_left(7);
    }
    // Domain separator byte.
    h1 ^= 0xFF;
    h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
    h2 ^= 0xAA;
    h2 = h2.wrapping_mul(0x0000_0100_0000_01b3);

    for (idx, &b) in data.iter().enumerate() {
        h1 ^= u64::from(b);
        h1 = h1.wrapping_mul(0x0000_0100_0000_01b3);
        h2 ^= u64::from(b) ^ ((idx as u64).wrapping_mul(0x9e37_79b9));
        h2 = h2.wrapping_mul(0x0000_0100_0000_01b3).rotate_left(5);
    }

    let lo = splitmix64_mix(h1 ^ (data.len() as u64));
    let hi = splitmix64_mix(h2 ^ ((data.len() as u64).rotate_left(32)));
    (u128::from(hi) << 64) | u128::from(lo)
}

/// Computes a deterministic 64-bit hash suitable for transform profile tags.
#[inline]
pub fn hash_u64(domain: &[u8], data: &[u8]) -> u64 {
    let h = content_hash_128(domain, data);
    (h as u64) ^ ((h >> 64) as u64)
}

#[inline]
fn splitmix64_mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Deterministic `SplitMix64` PRNG for reproducible shuffles and stochastic augmentations.
#[derive(Debug, Clone)]
pub struct Rng64 {
    state: u64,
}

impl Rng64 {
    /// Creates a new deterministic PRNG seeded with `seed`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next pseudo-random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        splitmix64_mix(self.state)
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Generates a uniform `usize` in `[0, upper)` (returns `0` if `upper == 0`).
    #[inline]
    pub fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }

    /// Performs an in-place deterministic Fisher-Yates shuffle.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        if slice.len() <= 1 {
            return;
        }
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(i + 1);
            slice.swap(i, j);
        }
    }
}

/// Compresses `input` with a clean-room LZ77 + RLE byte codec and frames it
/// with magic bytes, uncompressed length, and an IEEE CRC32 checksum.
///
/// Token encoding:
/// - `0x00..=0x7F`: literal run of `(tag + 1)` bytes (`1..=128`)
/// - `0x80..=0xBF`: RLE repeat of single byte, count `((tag & 0x3F) + 3)` (`3..=66`)
/// - `0xC0..=0xFF`: LZ back-reference, length `((tag & 0x3F) + 4)` (`4..=67`),
///   followed by `u16` little-endian backward offset (`1..=65535`)
pub fn compress_lz_rle(input: &[u8]) -> Vec<u8> {
    let checksum = crc32(input);
    let mut out = Vec::with_capacity(HEADER_LEN + input.len() / 2 + 16);
    out.extend_from_slice(&FRAME_MAGIC);
    out.extend_from_slice(&(input.len() as u32).to_le_bytes());
    out.extend_from_slice(&checksum.to_le_bytes());

    if input.is_empty() {
        return out;
    }

    const HASH_BITS: usize = 12;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    let mut head = [u32::MAX; HASH_SIZE];

    let mut pos = 0usize;
    let mut lit_start = 0usize;
    let len = input.len();

    let flush_literals = |out: &mut Vec<u8>, input: &[u8], start: usize, end: usize| {
        let mut cur = start;
        while cur < end {
            let chunk = (end - cur).min(128);
            out.push((chunk - 1) as u8);
            out.extend_from_slice(&input[cur..cur + chunk]);
            cur += chunk;
        }
    };

    while pos < len {
        // 1. Check for RLE run >= 3 identical bytes.
        let b = input[pos];
        let max_rle = (len - pos).min(66);
        let mut rle_len = 1usize;
        while rle_len < max_rle && input[pos + rle_len] == b {
            rle_len += 1;
        }
        if rle_len >= 3 {
            flush_literals(&mut out, input, lit_start, pos);
            out.push(0x80 | ((rle_len - 3) as u8));
            out.push(b);
            pos += rle_len;
            lit_start = pos;
            continue;
        }

        // 2. Check for 4-byte+ LZ back-reference within 65535-byte window.
        if pos + 4 <= len {
            let key = lz_hash4(&input[pos..pos + 4]);
            let cand = head[key];
            head[key] = pos as u32;

            if cand != u32::MAX {
                let cand_pos = cand as usize;
                let dist = pos - cand_pos;
                if dist > 0 && dist <= u16::MAX as usize {
                    let max_match = (len - pos).min(67);
                    let mut mlen = 0usize;
                    while mlen < max_match && input[cand_pos + mlen] == input[pos + mlen] {
                        mlen += 1;
                    }
                    if mlen >= 4 {
                        flush_literals(&mut out, input, lit_start, pos);
                        out.push(0xC0 | ((mlen - 4) as u8));
                        out.extend_from_slice(&(dist as u16).to_le_bytes());
                        // Index interior positions for better subsequent matches.
                        let step_end = (pos + mlen).min(len.saturating_sub(3));
                        let mut p = pos + 1;
                        while p < step_end {
                            let k = lz_hash4(&input[p..p + 4]);
                            head[k] = p as u32;
                            p += 1;
                        }
                        pos += mlen;
                        lit_start = pos;
                        continue;
                    }
                }
            }
        }

        pos += 1;
    }

    flush_literals(&mut out, input, lit_start, len);
    out
}

/// Decompresses a frame produced by [`compress_lz_rle`] and verifies both
/// exact length and IEEE CRC32 integrity.
pub fn decompress_lz_rle(input: &[u8]) -> Result<Vec<u8>, CodecError> {
    if input.len() < HEADER_LEN {
        return Err(CodecError::TruncatedHeader);
    }
    if input[0..4] != FRAME_MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let expected_len = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;
    if expected_len > MAX_FRAME_LEN {
        return Err(CodecError::PayloadTooLarge(expected_len));
    }
    let expected_crc = u32::from_le_bytes([input[8], input[9], input[10], input[11]]);

    let mut out = Vec::with_capacity(expected_len);
    let mut pos = HEADER_LEN;

    while pos < input.len() {
        let tag = input[pos];
        pos += 1;

        if tag < 0x80 {
            let count = (tag as usize) + 1;
            if pos + count > input.len() {
                return Err(CodecError::UnexpectedEof);
            }
            if out.len() + count > expected_len {
                return Err(CodecError::LengthMismatch {
                    expected: expected_len,
                    actual: out.len() + count,
                });
            }
            out.extend_from_slice(&input[pos..pos + count]);
            pos += count;
        } else if tag < 0xC0 {
            let count = ((tag & 0x3F) as usize) + 3;
            if pos >= input.len() {
                return Err(CodecError::UnexpectedEof);
            }
            if out.len() + count > expected_len {
                return Err(CodecError::LengthMismatch {
                    expected: expected_len,
                    actual: out.len() + count,
                });
            }
            let b = input[pos];
            pos += 1;
            out.resize(out.len() + count, b);
        } else {
            let count = ((tag & 0x3F) as usize) + 4;
            if pos + 2 > input.len() {
                return Err(CodecError::UnexpectedEof);
            }
            let offset = u16::from_le_bytes([input[pos], input[pos + 1]]) as usize;
            pos += 2;
            if offset == 0 || offset > out.len() {
                return Err(CodecError::InvalidBackReference {
                    offset,
                    output_len: out.len(),
                });
            }
            if out.len() + count > expected_len {
                return Err(CodecError::LengthMismatch {
                    expected: expected_len,
                    actual: out.len() + count,
                });
            }
            let start = out.len() - offset;
            for i in 0..count {
                let byte = out[start + i];
                out.push(byte);
            }
        }
    }

    if out.len() != expected_len {
        return Err(CodecError::LengthMismatch {
            expected: expected_len,
            actual: out.len(),
        });
    }
    let actual_crc = crc32(&out);
    if actual_crc != expected_crc {
        return Err(CodecError::ChecksumMismatch {
            expected: expected_crc,
            actual: actual_crc,
        });
    }
    Ok(out)
}

#[inline]
fn lz_hash4(window: &[u8]) -> usize {
    let v = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
    ((v.wrapping_mul(2654435761u32)) >> 20) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // Standard IEEE 802.3 CRC32 check vector for ASCII "123456789".
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn lz_rle_roundtrip_and_compression() {
        let mut payload = Vec::new();
        for i in 0..256u32 {
            payload.extend_from_slice(b"fn train_step(batch: &[f32]) -> f32 { ");
            payload.extend_from_slice(&[0u8; 16]);
            payload.push((i & 0xFF) as u8);
            payload.extend_from_slice(b" }\n");
        }
        let compressed = compress_lz_rle(&payload);
        assert!(
            compressed.len() * 4 < payload.len(),
            "expected >=4x compression, got {} -> {}",
            payload.len(),
            compressed.len()
        );
        let decoded = decompress_lz_rle(&compressed).expect("roundtrip must succeed");
        assert_eq!(decoded, payload);
    }

    #[test]
    fn lz_rle_empty_and_incompressible() {
        let empty = compress_lz_rle(&[]);
        assert_eq!(decompress_lz_rle(&empty).unwrap(), Vec::<u8>::new());

        let mut rng = Rng64::new(42);
        let noisy: Vec<u8> = (0..1024).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
        let comp = compress_lz_rle(&noisy);
        assert_eq!(decompress_lz_rle(&comp).unwrap(), noisy);
    }

    #[test]
    fn lz_rle_detects_corruption() {
        let payload =
            b"hanzo-seneca integrity verification test payload repeated repeated repeated";
        let mut comp = compress_lz_rle(payload);
        let last = comp.len() - 1;
        comp[last] ^= 0x01;
        let err = decompress_lz_rle(&comp).unwrap_err();
        assert!(matches!(
            err,
            CodecError::ChecksumMismatch { .. }
                | CodecError::LengthMismatch { .. }
                | CodecError::UnexpectedEof
                | CodecError::InvalidBackReference { .. }
        ));
    }

    #[test]
    fn content_hash_is_domain_separated() {
        let a = content_hash_128(b"dataset", b"shard-0");
        let b = content_hash_128(b"transform", b"shard-0");
        assert_ne!(a, b);
    }
}
