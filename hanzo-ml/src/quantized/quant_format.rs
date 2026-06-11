//! Declarative GGUF quant-type formatter ("decomplect quant support", Cut 2).
//!
//! A GGUF *decode-only* quant type is, mechanically, always the same five things:
//!
//!   1. a `#[repr(C)] #[derive(Clone, Copy)]` block struct whose field layout is a
//!      bit-for-bit replica of llama.cpp's `block_<t>` in `ggml-common.h`,
//!   2. a compile-time `assert!(size_of::<Block>() == N)` pinning that layout to the
//!      on-disk GGUF byte size,
//!   3. `to_float` — the *only* part that actually differs per type: the dequant
//!      (`dequantize_row_<t>` ported from `ggml-quants.c`),
//!   4. `from_float` — a panic stub (these types are decode-only; we never quantize
//!      *to* them), and
//!   5. `vec_dot` / `vec_dot_unopt` — the *identical* "dequantize the super-block to
//!      f32, then dot it against the Q8_0-quantized lhs" pattern for every type
//!      (only the block size changes; see [`vec_dot_dequant_q8_0`]).
//!
//! Today each type spells all five out by hand (≈55-75 lines), *plus* ~6 match arms
//! in `mod.rs` (`from_u32`, `to_u32`, `cpu_zeros`, `from_data`, `type_size`,
//! `block_size`) *plus* one in `ggml_file.rs` — seven independent places that must
//! agree or the type silently miswires. [`quant_format!`] collapses (1)-(5) into one
//! declaration whose only freeform part is the decode body, and [`for_each_quant!`] /
//! the [`quant_table!`]-generated arms collapse the mod.rs wiring into one table.
//!
//! Adding a quant type becomes: one `quant_format!` block + one row in the table.
//! Not seven hand-kept-in-sync sites.
//!
//! This module is **purely additive** — it introduces the machinery and a `#[cfg(test)]`
//! equivalence proof against the existing hand-written impls. The real migration of the
//! 11 IQ types in `iq_quants.rs` happens separately (see `quant-macro-migration.md`); it
//! deletes the boilerplate and replaces it with `quant_format!` calls.

#![allow(unused_macros)]
#![allow(unused_imports)]

use super::k_quants::{GgmlType, BlockQ8_0, QK8_0};
use half::f16;

/// The one shared matmul building block behind every decode-only type's `vec_dot`.
///
/// Dequantizes each block of `xs` to f32 (via `T::to_float`) and dots it against the
/// Q8_0-quantized lhs. Generic over the block size: a `BLOCK` of `B` elements spans
/// `B / QK8_0` Q8_0 blocks (every decode-only quant type has `B` a multiple of 32).
///
/// This is the generalization of `iq_quants::vec_dot_qk_q8_0` (which hard-codes QK_K)
/// to *any* block size, so the macro can use one helper for QK_K, Q1_0 (128) and
/// NVFP4 (64) alike. The body is byte-for-byte the loop the hand-written impls inline.
#[inline]
pub fn vec_dot_dequant_q8_0<const B: usize, T>(xs: &[T], ys: &[BlockQ8_0]) -> f32
where
    T: GgmlType<VecDotType = BlockQ8_0>,
{
    let mut sumf = 0f32;
    let mut tmp = [0f32; B];
    let q8_per_block = B / QK8_0;
    for (bx, x) in xs.iter().enumerate() {
        T::to_float(std::slice::from_ref(x), &mut tmp);
        for sub in 0..q8_per_block {
            let y = &ys[bx * q8_per_block + sub];
            let dy = f16::to_f32(y.d);
            for j in 0..QK8_0 {
                sumf += tmp[sub * QK8_0 + j] * (y.qs[j] as f32 * dy);
            }
        }
    }
    sumf
}

/// Declare ONE decode-only GGUF quant type: its block struct, byte-size assertion, and
/// full `impl GgmlType` (decode-from-body, `from_float` panic stub, and the standard
/// dequant-then-dot-against-Q8_0 `vec_dot`/`vec_dot_unopt`).
///
/// # Example
///
/// ```ignore
/// quant_format! {
///     /// IQ4_XS block: f16 super-block scale, 6-bit sub-block scales split across
///     /// scales_h/scales_l, then 128 bytes of packed 4-bit codebook indices.
///     name: BlockIQ4xs,
///     dtype: IQ4_XS,          // a GgmlDType variant
///     block_elems: QK_K,      // elements per block (BLCK_SIZE)
///     byte_size: 136,         // on-disk size_of::<Block>(), asserted at compile time
///     vec_dot: BlockQ8_0,     // VecDotType
///     fields: {
///         d: f16,
///         scales_h: u16,
///         scales_l: [u8; QK_K / 64],
///         qs: [u8; QK_K / 2],
///     },
///     // The ONLY freeform part: dequantize_row_iq4_xs, verbatim from ggml-quants.c.
///     // `xs: &[Self]` in, `ys: &mut [f32]` out — write `ys.len()` elements.
///     decode: |xs, ys| {
///         let nb = ys.len() / QK_K;
///         for i in 0..nb { /* ... fill ys ... */ }
///     },
/// }
/// ```
///
/// `name`, `dtype`, `block_elems`, `byte_size`, `vec_dot` and the `fields` are pure
/// declaration. `decode` is the existing `to_float` body, dropped in unchanged. Field
/// visibility is `pub(crate)` to match the existing hand-written blocks.
macro_rules! quant_format {
    (
        $(#[$meta:meta])*
        name: $name:ident,
        dtype: $dtype:ident,
        block_elems: $blck:expr,
        byte_size: $bytes:expr,
        vec_dot: $vecdot:ty,
        fields: { $( $(#[$fmeta:meta])* $fname:ident : $fty:ty ),+ $(,)? },
        decode: |$xs:ident, $ys:ident| $decode:block $(,)?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq)]
        #[repr(C)]
        pub struct $name {
            $( $(#[$fmeta])* pub(crate) $fname : $fty ),+
        }

        // Pin the in-memory layout to the GGUF on-disk byte size. If the fields don't
        // add up to `byte_size`, this fails to compile — the miswire is impossible.
        const _: () = assert!(
            ::std::mem::size_of::<$name>() == $bytes,
            concat!(stringify!($name), " on-disk byte size mismatch"),
        );

        impl $crate::quantized::k_quants::GgmlType for $name {
            const DTYPE: $crate::quantized::GgmlDType = $crate::quantized::GgmlDType::$dtype;
            const BLCK_SIZE: usize = $blck;
            type VecDotType = $vecdot;

            // The dequant: the supplied `decode` body, with the exact `to_float` signature.
            fn to_float($xs: &[Self], $ys: &mut [f32]) $decode

            // Decode-only: we read these from GGUF but never quantize *to* them.
            fn from_float(_xs: &[f32], _ys: &mut [Self]) {
                panic!(
                    concat!(stringify!($dtype), " quantize (from_float) is not supported")
                )
            }

            // No SIMD specialization for decode-only types: dequant-then-dot is the path.
            fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                Self::vec_dot_unopt(n, xs, ys)
            }

            fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                debug_assert!(
                    n % <Self as $crate::quantized::k_quants::GgmlType>::BLCK_SIZE == 0,
                    concat!(
                        "vec_dot ", stringify!($name), ": {} not divisible by block size"
                    ),
                    n,
                );
                $crate::quantized::quant_format::vec_dot_dequant_q8_0::<{ $blck }, Self>(xs, ys)
            }
        }
    };
}

pub(crate) use quant_format;

/// A single declarative table of every quant type, driving the `mod.rs` wiring so that
/// `from_u32` / `to_u32` / `cpu_zeros` / `from_data` / `type_size` / `block_size` are
/// generated from ONE list instead of six hand-maintained `match` blocks.
///
/// Each row is `Variant => Block @ ggml_id`. Call it with a callback macro that takes the
/// whole list; see [`impl_dtype_wiring!`] for the concrete expansion used by `mod.rs`, and
/// the `#[cfg(test)]` `for_each_quant_smoke` test below for how a consumer iterates it.
///
/// (Left as the canonical *source of truth* the migration points `mod.rs` at; wiring the
/// real `GgmlDType` impl to it is the final, mechanical migration step — it must land in
/// the same commit that removes the six hand-written match blocks, and is intentionally
/// NOT done here to avoid colliding with the concurrent edits to `mod.rs`.)
#[macro_export]
macro_rules! for_each_quant {
    ($cb:ident) => {
        $cb! {
            // Variant         Block            ggml type id
            Q4_0     => BlockQ4_0     @ 2,
            Q4_1     => BlockQ4_1     @ 3,
            Q5_0     => BlockQ5_0     @ 6,
            Q5_1     => BlockQ5_1     @ 7,
            Q8_0     => BlockQ8_0     @ 8,
            Q8_1     => BlockQ8_1     @ 9,
            Q2K      => BlockQ2K      @ 10,
            Q3K      => BlockQ3K      @ 11,
            Q4K      => BlockQ4K      @ 12,
            Q5K      => BlockQ5K      @ 13,
            Q6K      => BlockQ6K      @ 14,
            Q8K      => BlockQ8K      @ 15,
            IQ2_XXS  => BlockIQ2xxs   @ 16,
            IQ2_XS   => BlockIQ2xs    @ 17,
            IQ3_XXS  => BlockIQ3xxs   @ 18,
            IQ1_S    => BlockIQ1s     @ 19,
            IQ4_NL   => BlockIQ4nl    @ 20,
            IQ3_S    => BlockIQ3s     @ 21,
            IQ2_S    => BlockIQ2s     @ 22,
            IQ4_XS   => BlockIQ4xs    @ 23,
            IQ1_M    => BlockIQ1m     @ 29,
            MXFP4    => BlockMXFP4    @ 39,
            TQ1_0    => BlockTQ1_0    @ 34,
            TQ2_0    => BlockTQ2_0    @ 35,
            NVFP4    => BlockNVFP4    @ 40,
            Q1_0     => BlockQ1_0     @ 41,
        }
    };
}

// `quant_format.rs` is a leaf file module, so its `proof` child would normally live at
// `quant_format/proof.rs`; point at the sibling `proof.rs` instead to keep both files flat.
#[cfg(test)]
#[path = "proof.rs"]
mod proof;
