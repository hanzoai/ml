//! Equivalence proof for [`quant_format!`] (Cut 2, "decomplect quant support").
//!
//! We re-express three EXISTING, stable formats through `quant_format!` under fresh
//! `*Demo` names and assert, on sample GGUF bytes, that the macro-generated impl is
//! byte-for-byte identical to the hand-written impl already in the tree:
//!
//! * `BlockQ4_0`  — the simplest "real" type: f16 `d` + packed 4-bit nibbles.
//! * `BlockQ8_0`  — f16 `d` + 32 signed int8 (also the universal `VecDotType`).
//! * `BlockIQ4nl` — a genuine *decode-only* type (panic `from_float`,
//!   dequant-then-dot `vec_dot`) — i.e. exactly the shape of the 11
//!   IQ types the macro is meant to absorb.
//!
//! The demo decode bodies are the existing `to_float` bodies pasted verbatim. The test
//! reinterprets ONE buffer of fixed pseudo-random bytes as both the real block and the
//! demo block (both `#[repr(C)]`, identical layout) and checks:
//!   1. `size_of::<Demo>() == size_of::<Real>()` (the macro's compile-time assert already
//!      pins the absolute byte size; this pins it *to the real struct*), and
//!   2. `Demo::to_float(bytes) == Real::to_float(bytes)` bit-for-bit.
//!
//! If the macro expansion drifted from the hand-written semantics, (2) fails. Nothing
//! here touches the real structs, so it cannot conflict with the concurrent
//! `iq_quants.rs` work.

// `quant_format!` is in textual scope: this file is `mod proof;` declared *after* the
// `macro_rules! quant_format` definition in the parent `quant_format` module.
use crate::quantized::k_quants::{
    BlockIQ4nl, BlockQ4_0, BlockQ8_0, GgmlType, KVALUES_IQ4NL, QK4_0, QK4_NL, QK8_0,
};
use half::f16;

// --- Macro-generated re-expressions of three stable formats -----------------------

quant_format! {
    /// Macro twin of `BlockQ4_0` (proof-only).
    name: BlockQ4_0Demo,
    dtype: Q4_0,
    block_elems: QK4_0,
    byte_size: 18,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; QK4_0 / 2],
    },
    // Verbatim from BlockQ4_0::to_float (k_quants.rs).
    decode: |xs, ys| {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        debug_assert!(k.is_multiple_of(qk), "dequantize_row_q4_0: {k} is not divisible by {qk}");
        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;
                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
    },
}

quant_format! {
    /// Macro twin of `BlockQ8_0` (proof-only).
    name: BlockQ8_0Demo,
    dtype: Q8_0,
    block_elems: QK8_0,
    byte_size: 34,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [i8; QK8_0],
    },
    // Verbatim from BlockQ8_0::to_float (k_quants.rs).
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK8_0), "dequantize_row_q8_0: {k} is not divisible by {QK8_0}");
        let nb = k / QK8_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
    },
}

quant_format! {
    /// Macro twin of the decode-only `BlockIQ4nl` (proof-only).
    name: BlockIQ4nlDemo,
    dtype: IQ4_NL,
    block_elems: QK4_NL,
    byte_size: 18,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; QK4_NL / 2],
    },
    // Verbatim from BlockIQ4nl::to_float (k_quants.rs).
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK4_NL), "dequantize_row_iq4_nl: {k} is not divisible by {QK4_NL}");
        let nb = k / QK4_NL;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK4_NL / 2 {
                let q = xs[i].qs[j];
                ys[i * QK4_NL + j] = d * KVALUES_IQ4NL[(q & 0x0f) as usize] as f32;
                ys[i * QK4_NL + j + QK4_NL / 2] = d * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
            }
        }
    },
}

// --- Test harness -----------------------------------------------------------------

/// Deterministic, reproducible "random" bytes (xorshift) so the proof is stable.
fn sample_bytes(n: usize) -> Vec<u8> {
    let mut state: u32 = 0x9E37_79B9;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xff) as u8
        })
        .collect()
}

/// Reinterpret a byte buffer as `&[T]` (both demo and real blocks are `#[repr(C)]`).
/// `bytes.len()` must be a multiple of `size_of::<T>()` and the buffer suitably aligned;
/// we allocate via `Vec<u128>` to guarantee 16-byte alignment, which covers f16/i8/u8
/// blocks.
fn as_blocks<T>(bytes: &[u8]) -> &[T] {
    let sz = std::mem::size_of::<T>();
    assert_eq!(bytes.len() % sz, 0);
    assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, bytes.len() / sz) }
}

/// Allocate `nbytes` of 16-byte-aligned, deterministically-filled storage.
fn aligned_sample(nbytes: usize) -> Vec<u8> {
    let words = nbytes.div_ceil(16);
    let mut backing: Vec<u128> = vec![0u128; words];
    let raw =
        unsafe { std::slice::from_raw_parts_mut(backing.as_mut_ptr() as *mut u8, words * 16) };
    let src = sample_bytes(nbytes);
    raw[..nbytes].copy_from_slice(&src);
    // Move the bytes into a fresh Vec<u8> that keeps the same 16-aligned allocation by
    // leaking the u128 backing (test-only; process-lifetime).
    let ptr = backing.as_ptr() as *const u8;
    std::mem::forget(backing);
    unsafe { Vec::from_raw_parts(ptr as *mut u8, nbytes, words * 16) }
}

/// Generic equivalence check: same bytes, same byte size, bit-identical dequant.
fn assert_dequant_eq<Real, Demo>(n_blocks: usize, elems_per_block: usize)
where
    Real: GgmlType,
    Demo: GgmlType,
{
    assert_eq!(
        std::mem::size_of::<Demo>(),
        std::mem::size_of::<Real>(),
        "macro struct size differs from hand-written struct"
    );
    let nbytes = n_blocks * std::mem::size_of::<Real>();
    let buf = aligned_sample(nbytes);

    let real: &[Real] = as_blocks(&buf);
    let demo: &[Demo] = as_blocks(&buf);

    let n = n_blocks * elems_per_block;
    let mut out_real = vec![0f32; n];
    let mut out_demo = vec![0f32; n];
    Real::to_float(real, &mut out_real);
    Demo::to_float(demo, &mut out_demo);

    for (i, (r, d)) in out_real.iter().zip(out_demo.iter()).enumerate() {
        assert_eq!(
            r.to_bits(),
            d.to_bits(),
            "dequant mismatch at element {i}: real={r} demo={d}"
        );
    }
}

#[test]
fn q4_0_macro_matches_handwritten() {
    assert_dequant_eq::<BlockQ4_0, BlockQ4_0Demo>(8, QK4_0);
}

#[test]
fn q8_0_macro_matches_handwritten() {
    assert_dequant_eq::<BlockQ8_0, BlockQ8_0Demo>(8, QK8_0);
}

#[test]
fn iq4nl_macro_matches_handwritten() {
    assert_dequant_eq::<BlockIQ4nl, BlockIQ4nlDemo>(8, QK4_NL);
}

/// The macro-generated `vec_dot` (dequant-then-dot vs Q8_0) must equal a hand-rolled
/// reference dot for the decode-only IQ4_NL twin, proving the generated matmul building
/// block is correct, not just the dequant.
#[test]
fn iq4nl_macro_vec_dot_matches_reference() {
    let n_blocks = 4usize;
    let xbytes = aligned_sample(n_blocks * std::mem::size_of::<BlockIQ4nlDemo>());
    let ybytes = aligned_sample(n_blocks * std::mem::size_of::<BlockQ8_0>());
    let xs: &[BlockIQ4nlDemo] = as_blocks(&xbytes);
    let ys: &[BlockQ8_0] = as_blocks(&ybytes);

    let n = n_blocks * QK4_NL;
    let got = BlockIQ4nlDemo::vec_dot(n, xs, ys);

    // Reference: dequantize xs to f32, dequantize ys (Q8_0) to f32, plain dot.
    let mut xf = vec![0f32; n];
    BlockIQ4nlDemo::to_float(xs, &mut xf);
    let mut want = 0f32;
    for (b, y) in ys.iter().enumerate() {
        let dy = y.d.to_f32();
        for j in 0..QK8_0 {
            want += xf[b * QK8_0 + j] * (y.qs[j] as f32 * dy);
        }
    }
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "vec_dot mismatch: {got} vs {want}"
    );
}
