//! kernel-gen: regenerate (or drift-check) the committed DSL backend artifacts hanzo-ml dispatches.
//!
//!   kernel-gen --target metal --out ../hanzo-ml/src/metal/dsl          # write artifacts
//!   kernel-gen --target metal --check ../hanzo-ml/src/metal/dsl        # fail on any drift
//!
//! Each artifact is captured from the runtime's own compilation log (see `hanzo_kernel::gen`), so the
//! committed text is exactly what that backend compiled. Run on a host that has the target device.

#[cfg(any(feature = "metal", feature = "cuda", feature = "rocm"))]
use hanzo_kernel::gen::capture;
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
use hanzo_kernel::gen::Spec;
use hanzo_kernel::prelude::*;
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: kernel-gen --target metal|cuda|rocm|vulkan (--out DIR | --check DIR)");
    std::process::exit(2);
}

/// One PQ2_0 matvec specialization, launched on a 3-row x 256-col tensor (odd block count, so the
/// half-word path and the clamped tail are both compiled in).
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! pq2_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $conv:expr, $nsg:literal, $nr0:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (rows, k) = (3usize, 256usize);
                let (codes, d) = pq2_0_synth(rows, k, 1);
                let words = pq2_0_gguf_words(&pq2_0_gguf_bytes(&codes, &d, rows, k));
                let x: Vec<$e> = (0..k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matvec_pq2_0_gguf_run_as::<$e, $r>(c, &words, &x, rows, k, $nsg, $nr0);
            },
        }
    };
}

/// One PQ2_0 prefill GEMM specialization (`f32` or `bf16`, `bm = 32` or `64`), launched on a small ragged tile.
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! pq2_gemm_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $conv:expr, $bm:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (m, n, k) = (17usize, 73usize, 256usize);
                let (codes, d) = pq2_0_synth(n, k, 1);
                let words = pq2_0_gguf_words(&pq2_0_gguf_bytes(&codes, &d, n, k));
                let x: Vec<$e> = (0..m * k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matmul_pq2_0_gguf_bm_run_as::<$e, $r>(c, &words, &x, m, n, k, $bm);
            },
        }
    };
}

/// One PTQ1_0 matvec specialization (`28 B` GGUF blocks).
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! ptq1_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $conv:expr, $nsg:literal, $nr0:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (rows, k) = (3usize, 256usize);
                let (codes, d) = ptq1_0_synth(rows, k, 1);
                let words = ptq1_0_gguf_words(&ptq1_0_gguf_bytes(&codes, &d, rows, k));
                let x: Vec<$e> = (0..k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matvec_ptq1_0_gguf_run_as::<$e, $r>(c, &words, &x, rows, k, $nsg, $nr0);
            },
        }
    };
}

/// One PTQ1_0 prefill GEMM specialization (`f32` or `bf16`, `bm = 32` or `64`).
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! ptq1_gemm_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $conv:expr, $bm:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (m, n, k) = (17usize, 73usize, 256usize);
                let (codes, d) = ptq1_0_synth(n, k, 1);
                let words = ptq1_0_gguf_words(&ptq1_0_gguf_bytes(&codes, &d, n, k));
                let x: Vec<$e> = (0..m * k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matmul_ptq1_0_gguf_bm_run_as::<$e, $r>(c, &words, &x, m, n, k, $bm);
            },
        }
    };
}

/// All 32 ternary (PQ2_0 + PTQ1_0) decode matvec and prefill GEMM specializations for runtime `R`.
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
fn ternary_specs<R: Runtime>() -> Vec<Spec<R>> {
    vec![
        // PQ2_0 decode matvec over native GGUF blocks: one artifact per (NSG, NR0) schedule x dtype.
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg2_nr2_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2,
            2
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg2_nr4_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2,
            4
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg2_nr8_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2,
            8
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg2_nr2_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2,
            2
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg2_nr4_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2,
            4
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg2_nr8_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2,
            8
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg4_nr2_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4,
            2
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg4_nr4_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4,
            4
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg4_nr8_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4,
            8
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg4_nr2_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4,
            2
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg4_nr4_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4,
            4
        ),
        pq2_spec!(
            R,
            "matvec_pq2_0_gguf_nsg4_nr8_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4,
            8
        ),
        // PQ2_0 prefill GEMM (M > 1) over native GGUF blocks.
        pq2_gemm_spec!(
            R,
            "matmul_pq2_0_gguf_bm32_f32",
            "matmul_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            32
        ),
        pq2_gemm_spec!(
            R,
            "matmul_pq2_0_gguf_bm32_bf16",
            "matmul_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            32
        ),
        pq2_gemm_spec!(
            R,
            "matmul_pq2_0_gguf_bm64_f32",
            "matmul_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            64
        ),
        pq2_gemm_spec!(
            R,
            "matmul_pq2_0_gguf_bm64_bf16",
            "matmul_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            64
        ),
        // PTQ1_0 decode matvec + prefill GEMM over native 28-byte GGUF blocks.
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg2_nr2_f32",
            "matvec_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2,
            2
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg2_nr4_f32",
            "matvec_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2,
            4
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg2_nr8_f32",
            "matvec_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2,
            8
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg2_nr2_bf16",
            "matvec_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2,
            2
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg2_nr4_bf16",
            "matvec_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2,
            4
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg2_nr8_bf16",
            "matvec_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2,
            8
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg4_nr2_f32",
            "matvec_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4,
            2
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg4_nr4_f32",
            "matvec_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4,
            4
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg4_nr8_f32",
            "matvec_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4,
            8
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg4_nr2_bf16",
            "matvec_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4,
            2
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg4_nr4_bf16",
            "matvec_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4,
            4
        ),
        ptq1_spec!(
            R,
            "matvec_ptq1_0_gguf_nsg4_nr8_bf16",
            "matvec_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4,
            8
        ),
        ptq1_gemm_spec!(
            R,
            "matmul_ptq1_0_gguf_bm32_f32",
            "matmul_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            32
        ),
        ptq1_gemm_spec!(
            R,
            "matmul_ptq1_0_gguf_bm32_bf16",
            "matmul_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            32
        ),
        ptq1_gemm_spec!(
            R,
            "matmul_ptq1_0_gguf_bm64_f32",
            "matmul_ptq1_0_gguf_f_f32",
            f32,
            std::convert::identity,
            64
        ),
        ptq1_gemm_spec!(
            R,
            "matmul_ptq1_0_gguf_bm64_bf16",
            "matmul_ptq1_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            64
        ),
    ]
}

/// Production GDN (stateful prefill, single-token decode step, typed gating, typed lane-split
/// recurrence) and Paged/Contiguous FlashAttention (`d = 64, 128, 256`) across `f32`, `f16`, `bf16`.
/// `d256_br` is `8` on Metal (`< 32 KB` threadgroup memory) and `16` on CUDA/ROCm/Vulkan (`16x16` CMMA).
#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! gdn_conv_step_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $zero:expr) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_step_run_as::<$e, $r>(
                    c, &vec![$zero; 4], &vec![$zero; 16], &vec![$zero; 16], 1, 4, 4,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! gdn_conv_stateful_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $zero:expr) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_stateful_run_as::<$e, $r>(
                    c, &vec![$zero; 16], &vec![$zero; 16], &vec![$zero; 16], 1, 4, 4, true, 4,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! gdn_gating_io_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $zero:expr) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_gating_io_run_as::<$e, $r>(
                    c, &vec![$zero; 4], &vec![$zero; 4], &vec![0.0f32; 4], &vec![0.0f32; 4], 4,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! gdn_scan_io_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $zero:expr, $kd:literal, $vd:literal, $lanes:literal, $blk:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_io_run_as::<$e, $r>(
                    c,
                    &vec![$zero; $kd],
                    &vec![$zero; $kd],
                    &vec![$zero; $vd],
                    &[$zero],
                    &[$zero],
                    &vec![0.0f32; $kd * $vd],
                    1,
                    1,
                    $kd,
                    $vd,
                    $lanes,
                    $blk,
                    false,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! flash_paged_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $zero:expr, $d:literal, $br:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_paged_run_as::<$e, $r>(
                    c,
                    &vec![$zero; $d],
                    &vec![$zero; 16 * $d],
                    &vec![$zero; 16 * $d],
                    &[0u32],
                    &[1u32],
                    1,
                    1,
                    1,
                    1,
                    $d,
                    $br,
                    16,
                    1,
                    16,
                    1,
                    0,
                    4,
                    0,
                    0,
                    1.0 / ($d as f32).sqrt(),
                    0.0,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! flash_paged_split_kv_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $zero:expr, $d:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_paged_split_kv_run_as::<$e, $r>(
                    c,
                    &vec![$zero; $d],
                    &vec![$zero; 16 * $d],
                    &vec![$zero; 16 * $d],
                    &[0u32],
                    &[1u32],
                    1,
                    1,
                    1,
                    1,
                    $d,
                    1,
                    1,
                    16,
                    1,
                    0,
                    4,
                    0,
                    0,
                    1.0 / ($d as f32).sqrt(),
                    0.0,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
macro_rules! flash_split_kv_reduce_spec {
    ($r:ty, $artifact:literal, $entry:literal, $e:ty, $d:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_split_kv_reduce_run_as::<$e, $r>(
                    c,
                    &vec![0.0f32; 2 * ($d + 2)],
                    1,
                    2,
                    $d,
                );
            },
        }
    };
}

#[cfg(any(
    feature = "metal",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan"
))]
fn gdn_and_flash_prod_specs<R: Runtime>(is_metal: bool) -> Vec<Spec<R>> {
    let mut specs = vec![
        // Single-token GDN conv1d decode step (in-place state shift + SiLU).
        gdn_conv_step_spec!(
            R,
            "gdn_conv1d_step_k4_f32",
            "gdn_conv1d_step_f_f32",
            f32,
            0.0f32
        ),
        gdn_conv_step_spec!(
            R,
            "gdn_conv1d_step_k4_f16",
            "gdn_conv1d_step_f_",
            half::f16,
            half::f16::from_f32(0.0)
        ),
        gdn_conv_step_spec!(
            R,
            "gdn_conv1d_step_k4_bf16",
            "gdn_conv1d_step_f_",
            half::bf16,
            half::bf16::from_f32(0.0)
        ),
        // Stateful GDN conv1d prefill / chunked prefill + trailing state handoff.
        gdn_conv_stateful_spec!(
            R,
            "gdn_conv1d_stateful_k4_f32",
            "gdn_conv1d_stateful_f_f32",
            f32,
            0.0f32
        ),
        gdn_conv_stateful_spec!(
            R,
            "gdn_conv1d_stateful_k4_f16",
            "gdn_conv1d_stateful_f_",
            half::f16,
            half::f16::from_f32(0.0)
        ),
        gdn_conv_stateful_spec!(
            R,
            "gdn_conv1d_stateful_k4_bf16",
            "gdn_conv1d_stateful_f_",
            half::bf16,
            half::bf16::from_f32(0.0)
        ),
        // Typed GDN gating (f32 internal exp/softplus).
        gdn_gating_io_spec!(R, "gdn_gating_io_f32", "gdn_gating_io_f_f32", f32, 0.0f32),
        gdn_gating_io_spec!(
            R,
            "gdn_gating_io_f16",
            "gdn_gating_io_f_",
            half::f16,
            half::f16::from_f32(0.0)
        ),
        gdn_gating_io_spec!(
            R,
            "gdn_gating_io_bf16",
            "gdn_gating_io_f_",
            half::bf16,
            half::bf16::from_f32(0.0)
        ),
        // Typed GDN lane-split recurrence scan (f32 in-place state).
        gdn_scan_io_spec!(
            R,
            "gdn_scan_lanes_io_k128_v128_l4_f32",
            "gdn_scan_lanes_io_f_f32",
            f32,
            0.0f32,
            128,
            128,
            4,
            64
        ),
        gdn_scan_io_spec!(
            R,
            "gdn_scan_lanes_io_k128_v128_l4_f16",
            "gdn_scan_lanes_io_f_",
            half::f16,
            half::f16::from_f32(0.0),
            128,
            128,
            4,
            64
        ),
        gdn_scan_io_spec!(
            R,
            "gdn_scan_lanes_io_k128_v128_l4_bf16",
            "gdn_scan_lanes_io_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            128,
            128,
            4,
            64
        ),
        gdn_scan_io_spec!(
            R,
            "gdn_scan_lanes_io_k64_v64_l4_f32",
            "gdn_scan_lanes_io_f_f32",
            f32,
            0.0f32,
            64,
            64,
            4,
            64
        ),
        gdn_scan_io_spec!(
            R,
            "gdn_scan_lanes_io_k64_v64_l4_f16",
            "gdn_scan_lanes_io_f_",
            half::f16,
            half::f16::from_f32(0.0),
            64,
            64,
            4,
            64
        ),
        gdn_scan_io_spec!(
            R,
            "gdn_scan_lanes_io_k64_v64_l4_bf16",
            "gdn_scan_lanes_io_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            64,
            64,
            4,
            64
        ),
        // Production Paged & Contiguous FlashAttention (d=64, d=128).
        flash_paged_spec!(
            R,
            "flash_attn_paged_d64_f32",
            "flash_attn_paged_f_f32",
            f32,
            0.0f32,
            64,
            16
        ),
        flash_paged_spec!(
            R,
            "flash_attn_paged_d64_f16",
            "flash_attn_paged_f_",
            half::f16,
            half::f16::from_f32(0.0),
            64,
            16
        ),
        flash_paged_spec!(
            R,
            "flash_attn_paged_d64_bf16",
            "flash_attn_paged_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            64,
            16
        ),
        flash_paged_spec!(
            R,
            "flash_attn_paged_d128_f32",
            "flash_attn_paged_f_f32",
            f32,
            0.0f32,
            128,
            16
        ),
        flash_paged_spec!(
            R,
            "flash_attn_paged_d128_f16",
            "flash_attn_paged_f_",
            half::f16,
            half::f16::from_f32(0.0),
            128,
            16
        ),
        flash_paged_spec!(
            R,
            "flash_attn_paged_d128_bf16",
            "flash_attn_paged_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            128,
            16
        ),
    ];
    if is_metal {
        // Metal uses br=8 for d=256 to stay under the 32 KB threadgroup memory ceiling.
        specs.extend([
            flash_paged_spec!(
                R,
                "flash_attn_paged_d256_f32",
                "flash_attn_paged_f_f32",
                f32,
                0.0f32,
                256,
                8
            ),
            flash_paged_spec!(
                R,
                "flash_attn_paged_d256_f16",
                "flash_attn_paged_f_",
                half::f16,
                half::f16::from_f32(0.0),
                256,
                8
            ),
            flash_paged_spec!(
                R,
                "flash_attn_paged_d256_bf16",
                "flash_attn_paged_f_",
                half::bf16,
                half::bf16::from_f32(0.0),
                256,
                8
            ),
        ]);
    } else {
        specs.extend([
            flash_paged_spec!(
                R,
                "flash_attn_paged_d256_f32",
                "flash_attn_paged_f_f32",
                f32,
                0.0f32,
                256,
                16
            ),
            flash_paged_spec!(
                R,
                "flash_attn_paged_d256_f16",
                "flash_attn_paged_f_",
                half::f16,
                half::f16::from_f32(0.0),
                256,
                16
            ),
            flash_paged_spec!(
                R,
                "flash_attn_paged_d256_bf16",
                "flash_attn_paged_f_",
                half::bf16,
                half::bf16::from_f32(0.0),
                256,
                16
            ),
        ]);
    }
    specs.extend([
        // Canonical DSL Split-KV paged & contiguous FlashAttention decode (d=64, 128, 256).
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d64_f32",
            "flash_attn_paged_split_kv_f_f32",
            f32,
            0.0f32,
            64
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d64_f16",
            "flash_attn_paged_split_kv_f_",
            half::f16,
            half::f16::from_f32(0.0),
            64
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d64_bf16",
            "flash_attn_paged_split_kv_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            64
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d128_f32",
            "flash_attn_paged_split_kv_f_f32",
            f32,
            0.0f32,
            128
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d128_f16",
            "flash_attn_paged_split_kv_f_",
            half::f16,
            half::f16::from_f32(0.0),
            128
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d128_bf16",
            "flash_attn_paged_split_kv_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            128
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d256_f32",
            "flash_attn_paged_split_kv_f_f32",
            f32,
            0.0f32,
            256
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d256_f16",
            "flash_attn_paged_split_kv_f_",
            half::f16,
            half::f16::from_f32(0.0),
            256
        ),
        flash_paged_split_kv_spec!(
            R,
            "flash_attn_paged_split_kv_d256_bf16",
            "flash_attn_paged_split_kv_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            256
        ),
        // Second-pass online-softmax merge for Split-KV FlashAttention (d=64, 128, 256).
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d64_f32",
            "flash_attn_split_kv_reduce_f_f32",
            f32,
            64
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d64_f16",
            "flash_attn_split_kv_reduce_f_",
            half::f16,
            64
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d64_bf16",
            "flash_attn_split_kv_reduce_f_",
            half::bf16,
            64
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d128_f32",
            "flash_attn_split_kv_reduce_f_f32",
            f32,
            128
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d128_f16",
            "flash_attn_split_kv_reduce_f_",
            half::f16,
            128
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d128_bf16",
            "flash_attn_split_kv_reduce_f_",
            half::bf16,
            128
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d256_f32",
            "flash_attn_split_kv_reduce_f_f32",
            f32,
            256
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d256_f16",
            "flash_attn_split_kv_reduce_f_",
            half::f16,
            256
        ),
        flash_split_kv_reduce_spec!(
            R,
            "flash_attn_split_kv_reduce_d256_bf16",
            "flash_attn_split_kv_reduce_f_",
            half::bf16,
            256
        ),
    ]);
    macro_rules! fwht_spec {
        ($artifact:literal, $entry:literal, $e:ty, $zero:expr, $b:expr) => {
            Spec {
                artifact: $artifact,
                entry: $entry,
                launch: |c| {
                    let _ = hanzo_kernel::fwht::fwht_hadamard_run_as::<$e, R>(
                        c,
                        &vec![$zero; $b],
                        &vec![1.0f32; $b],
                        $b,
                        true,
                        false,
                    );
                },
            }
        };
    }
    specs.extend([
        fwht_spec!(
            "fwht_hadamard_b64_f32",
            "fwht_hadamard_f_f32",
            f32,
            0.0f32,
            64
        ),
        fwht_spec!(
            "fwht_hadamard_b64_f16",
            "fwht_hadamard_f_",
            half::f16,
            half::f16::from_f32(0.0),
            64
        ),
        fwht_spec!(
            "fwht_hadamard_b64_bf16",
            "fwht_hadamard_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            64
        ),
        fwht_spec!(
            "fwht_hadamard_b128_f32",
            "fwht_hadamard_f_f32",
            f32,
            0.0f32,
            128
        ),
        fwht_spec!(
            "fwht_hadamard_b128_f16",
            "fwht_hadamard_f_",
            half::f16,
            half::f16::from_f32(0.0),
            128
        ),
        fwht_spec!(
            "fwht_hadamard_b128_bf16",
            "fwht_hadamard_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            128
        ),
        fwht_spec!(
            "fwht_hadamard_b256_f32",
            "fwht_hadamard_f_f32",
            f32,
            0.0f32,
            256
        ),
        fwht_spec!(
            "fwht_hadamard_b256_f16",
            "fwht_hadamard_f_",
            half::f16,
            half::f16::from_f32(0.0),
            256
        ),
        fwht_spec!(
            "fwht_hadamard_b256_bf16",
            "fwht_hadamard_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            256
        ),
        fwht_spec!(
            "fwht_hadamard_b512_f32",
            "fwht_hadamard_f_f32",
            f32,
            0.0f32,
            512
        ),
        fwht_spec!(
            "fwht_hadamard_b512_f16",
            "fwht_hadamard_f_",
            half::f16,
            half::f16::from_f32(0.0),
            512
        ),
        fwht_spec!(
            "fwht_hadamard_b512_bf16",
            "fwht_hadamard_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            512
        ),
        fwht_spec!(
            "fwht_hadamard_b1024_f32",
            "fwht_hadamard_f_f32",
            f32,
            0.0f32,
            1024
        ),
        fwht_spec!(
            "fwht_hadamard_b1024_f16",
            "fwht_hadamard_f_",
            half::f16,
            half::f16::from_f32(0.0),
            1024
        ),
        fwht_spec!(
            "fwht_hadamard_b1024_bf16",
            "fwht_hadamard_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            1024
        ),
        fwht_spec!(
            "fwht_hadamard_b2048_f32",
            "fwht_hadamard_f_f32",
            f32,
            0.0f32,
            2048
        ),
        fwht_spec!(
            "fwht_hadamard_b2048_f16",
            "fwht_hadamard_f_",
            half::f16,
            half::f16::from_f32(0.0),
            2048
        ),
        fwht_spec!(
            "fwht_hadamard_b2048_bf16",
            "fwht_hadamard_f_",
            half::bf16,
            half::bf16::from_f32(0.0),
            2048
        ),
    ]);
    specs
}

/// Every Metal artifact hanzo-ml commits under `src/metal/dsl/`. Adding a DSL kernel to the Metal
/// path = one entry here + one `kernel_msl` arm in hanzo-ml.
#[cfg(feature = "metal")]
fn metal_specs() -> Vec<Spec<cubecl::wgpu::WgpuRuntime>> {
    use cubecl::wgpu::WgpuRuntime as R;
    let mut specs = vec![
        Spec {
            artifact: "rms_norm_blk",
            entry: "rms_norm_blk_f_f32",
            launch: |c| {
                let (rows, n) = (2usize, 128usize);
                let x = vec![0.5f32; rows * n];
                let w = vec![1.0f32; n];
                let _ =
                    hanzo_kernel::norm::rms_norm_blk_run::<f32, R>(c, &x, &w, rows, n, 1e-5, 128);
            },
        },
        Spec {
            artifact: "gemv",
            entry: "gemv_f_f32",
            launch: |c| {
                let (n, k) = (4usize, 64usize);
                let _ = hanzo_kernel::quant::gemv_run::<R>(
                    c,
                    &vec![1.0; n * k],
                    &vec![1.0; k],
                    n,
                    k,
                    32,
                );
            },
        },
    ];
    specs.extend(ternary_specs::<R>());
    specs.extend([
        // GDN linear-attention ops (zen6 / qwen3_5_moe): conv1d, gating, and lane-split recurrence scan.
        Spec {
            artifact: "gdn_conv1d_k4",
            entry: "gdn_conv1d_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_run::<R>(
                    c,
                    &vec![0.0; 16],
                    &vec![0.0; 16],
                    1,
                    4,
                    4,
                    4,
                );
            },
        },
        Spec {
            artifact: "gdn_gating",
            entry: "gdn_gating_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_gating_run::<R>(
                    c,
                    &vec![0.0; 4],
                    &vec![0.0; 4],
                    &vec![0.0; 4],
                    &vec![0.0; 4],
                    4,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l4",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run::<R>(
                    c,
                    &vec![0.0; 128],
                    &vec![0.0; 128],
                    &vec![0.0; 128],
                    &[0.0],
                    &[0.0],
                    &vec![0.0; 128 * 128],
                    1,
                    1,
                    128,
                    128,
                    4,
                    64,
                    false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l8",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run::<R>(
                    c,
                    &vec![0.0; 128],
                    &vec![0.0; 128],
                    &vec![0.0; 128],
                    &[0.0],
                    &[0.0],
                    &vec![0.0; 128 * 128],
                    1,
                    1,
                    128,
                    128,
                    8,
                    128,
                    false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k64_v64_l4",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run::<R>(
                    c,
                    &vec![0.0; 64],
                    &vec![0.0; 64],
                    &vec![0.0; 64],
                    &[0.0],
                    &[0.0],
                    &vec![0.0; 64 * 64],
                    1,
                    1,
                    64,
                    64,
                    4,
                    64,
                    false,
                );
            },
        },
        // Tiled online-softmax FlashAttention (`8x8` simdgroup_matrix on Metal, plane=32).
        Spec {
            artifact: "flash_attn_d64_p32",
            entry: "flash_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_run::<R>(
                    c,
                    &vec![0.0; 64],
                    &vec![0.0; 64],
                    &vec![0.0; 64],
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    64,
                    false,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d128_p32",
            entry: "flash_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_run::<R>(
                    c,
                    &vec![0.0; 128],
                    &vec![0.0; 128],
                    &vec![0.0; 128],
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    128,
                    false,
                );
            },
        },
    ]);
    specs.extend(gdn_and_flash_prod_specs::<R>(true));
    specs
}

#[cfg(any(feature = "metal", feature = "cuda", feature = "rocm"))]
fn write_or_check(dir: &Path, check: bool, ext: &str, items: &[(&str, String)]) -> bool {
    let mut ok = true;
    for (stem, raw_src) in items {
        let src = raw_src.replace("\r\n", "\n");
        let path = dir.join(format!("{stem}.{ext}"));
        if check {
            match std::fs::read_to_string(&path) {
                Ok(cur) if cur.replace("\r\n", "\n") == src => {
                    eprintln!("ok     {}", path.display())
                }
                Ok(cur) => {
                    let cur = cur.replace("\r\n", "\n");
                    ok = false;
                    let first = cur
                        .lines()
                        .zip(src.lines())
                        .position(|(a, b)| a != b)
                        .unwrap_or(cur.lines().count().min(src.lines().count()));
                    eprintln!(
                        "DRIFT  {} (first differing line {}; committed {} lines, generated {} lines)",
                        path.display(),
                        first + 1,
                        cur.lines().count(),
                        src.lines().count()
                    );
                }
                Err(e) => {
                    ok = false;
                    eprintln!("MISSING {}: {e}", path.display());
                }
            }
        } else {
            std::fs::create_dir_all(dir).expect("create out dir");
            std::fs::write(&path, &src).expect("write artifact");
            eprintln!("wrote  {}", path.display());
        }
    }
    ok
}

/// Rewrite every `OpEntryPoint` (opcode 15) in a little-endian SPIR-V module to `"main"`, so
/// `hanzo-ml`'s `VulkanDevice::pipeline` can bind it directly via `c"main"` without external
/// `spirv-dis`/`spirv-as` tools.
#[cfg(all(feature = "vulkan", feature = "spirv-dump"))]
fn rename_spv_entry_to_main(bytes: &[u8]) -> Vec<u8> {
    assert!(
        bytes.len() >= 20 && bytes.len() % 4 == 0,
        "invalid SPIR-V byte length"
    );
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(words[0], 0x0723_0203, "invalid SPIR-V magic");

    let mut out: Vec<u32> = Vec::with_capacity(words.len());
    out.extend_from_slice(&words[..5]);
    let mut i = 5usize;
    let mut renamed = 0usize;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let op = (header & 0xffff) as u16;
        assert!(
            wc >= 1 && i + wc <= words.len(),
            "corrupt SPIR-V instruction at word {i}"
        );
        if op == 15 {
            // OpEntryPoint: [header, ExecModel, EntryPointId, Name (null-term literal), Interface...]
            assert!(wc >= 4, "truncated OpEntryPoint");
            let mut name_words = 0usize;
            while 3 + name_words < wc {
                let w = words[i + 3 + name_words].to_le_bytes();
                name_words += 1;
                if w.contains(&0) {
                    break;
                }
            }
            let iface = &words[i + 3 + name_words..i + wc];
            // "main\0" packed into 2 little-endian u32 words: [b"main", 0].
            let new_wc = 3 + 2 + iface.len();
            out.push(((new_wc as u32) << 16) | 15u32);
            out.push(words[i + 1]);
            out.push(words[i + 2]);
            out.push(u32::from_le_bytes(*b"main"));
            out.push(0u32);
            out.extend_from_slice(iface);
            renamed += 1;
        } else {
            out.extend_from_slice(&words[i..i + wc]);
        }
        i += wc;
    }
    assert_eq!(
        renamed, 1,
        "expected exactly 1 OpEntryPoint in SPIR-V module, got {renamed}"
    );
    let mut out_bytes = Vec::with_capacity(out.len() * 4);
    for w in out {
        out_bytes.extend_from_slice(&w.to_le_bytes());
    }
    out_bytes
}

/// Structural SPIR-V fingerprint: `(byte_len, entry_name, opcode_counts)`.
/// CubeCL's SPIR-V emitter iterates Rust `HashSet`/`HashMap` for capabilities and shared-memory
/// offsets, so two runs on the same IR can permute header/offset order while keeping exact size,
/// entry point, and instruction opcode counts.
#[cfg(all(feature = "vulkan", feature = "spirv-dump"))]
fn spv_fingerprint(
    bytes: &[u8],
) -> Option<(usize, String, std::collections::BTreeMap<u16, usize>)> {
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        return None;
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words[0] != 0x0723_0203 {
        return None;
    }
    let mut counts = std::collections::BTreeMap::new();
    let mut entry = String::new();
    let mut i = 5usize;
    while i < words.len() {
        let wc = (words[i] >> 16) as usize;
        let op = (words[i] & 0xffff) as u16;
        if wc == 0 || i + wc > words.len() {
            return None;
        }
        *counts.entry(op).or_insert(0) += 1;
        if op == 15 && wc >= 4 {
            let mut raw = Vec::new();
            for &w in &words[i + 3..i + wc] {
                let bs = w.to_le_bytes();
                if let Some(pos) = bs.iter().position(|&b| b == 0) {
                    raw.extend_from_slice(&bs[..pos]);
                    break;
                }
                raw.extend_from_slice(&bs);
            }
            entry = String::from_utf8_lossy(&raw).into_owned();
        }
        i += wc;
    }
    Some((bytes.len(), entry, counts))
}

#[cfg(all(feature = "vulkan", feature = "spirv-dump"))]
fn write_or_check_spv(dir: &Path, check: bool, items: &[(String, Vec<u8>)]) -> bool {
    let mut ok = true;
    for (stem, bytes) in items {
        let path = dir.join(format!("{stem}.spv"));
        if check {
            match std::fs::read(&path) {
                Ok(cur) => {
                    let cur_fp = spv_fingerprint(&cur);
                    let gen_fp = spv_fingerprint(bytes);
                    if cur_fp.is_some() && cur_fp == gen_fp {
                        eprintln!("ok     {} ({} bytes)", path.display(), cur.len());
                    } else {
                        ok = false;
                        eprintln!(
                            "DRIFT  {} (committed {} bytes, generated {} bytes)",
                            path.display(),
                            cur.len(),
                            bytes.len()
                        );
                    }
                }
                Err(e) => {
                    ok = false;
                    eprintln!("MISSING {}: {e}", path.display());
                }
            }
        } else {
            std::fs::create_dir_all(dir).expect("create out dir");
            std::fs::write(&path, bytes).expect("write artifact");
            eprintln!("wrote  {} ({} bytes)", path.display(), bytes.len());
        }
    }
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut target = None;
    let mut dir: Option<(PathBuf, bool)> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--target" => target = it.next().cloned(),
            "--out" => dir = it.next().map(|d| (PathBuf::from(d), false)),
            "--check" => dir = it.next().map(|d| (PathBuf::from(d), true)),
            _ => usage(),
        }
    }
    let (Some(target), Some((dir, check))) = (target, dir) else {
        usage()
    };

    // Must be set before the first client exists: cubecl reads its config once.
    let log = std::env::temp_dir().join(format!("hanzo-kernel-gen-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    std::env::set_var("CUBECL_DEBUG_LOG", &log);

    let ok = match target.as_str() {
        #[cfg(feature = "metal")]
        "metal" => {
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
            let client = WgpuRuntime::client(&WgpuDevice::default());
            let name = WgpuRuntime::name(&client);
            if !name.contains("msl") {
                eprintln!("runtime is `{name}`, not wgpu<msl>: refusing to write Metal artifacts");
                std::process::exit(1);
            }
            let specs = metal_specs();
            let got = capture(&client, &specs, &log).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let items: Vec<(&str, String)> = got
                .into_iter()
                .map(|(i, s)| (specs[i].artifact, s))
                .collect();
            write_or_check(&dir, check, "metal", &items)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use cubecl::cuda::{CudaDevice, CudaRuntime};
            let client = CudaRuntime::client(&CudaDevice::new(0));
            let mut specs = ternary_specs::<CudaRuntime>();
            specs.extend(gdn_and_flash_prod_specs::<CudaRuntime>(false));
            let got = capture(&client, &specs, &log).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let items: Vec<(&str, String)> = got
                .into_iter()
                .map(|(i, s)| (specs[i].artifact, s))
                .collect();
            write_or_check(&dir, check, "cu", &items)
        }
        #[cfg(feature = "rocm")]
        "rocm" => {
            use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
            let client = HipRuntime::client(&AmdDevice::default());
            let mut specs = ternary_specs::<HipRuntime>();
            specs.extend(gdn_and_flash_prod_specs::<HipRuntime>(false));
            let got = capture(&client, &specs, &log).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let items: Vec<(&str, String)> = got
                .into_iter()
                .map(|(i, s)| (specs[i].artifact, s))
                .collect();
            write_or_check(&dir, check, "hip", &items)
        }
        #[cfg(all(feature = "vulkan", feature = "spirv-dump"))]
        "vulkan" => {
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
            let spv_tmp =
                std::env::temp_dir().join(format!("hanzo-kernel-gen-spv-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&spv_tmp);
            std::fs::create_dir_all(&spv_tmp).expect("create spv temp dir");
            std::env::set_var("CUBECL_DEBUG_SPIRV", &spv_tmp);

            let client = WgpuRuntime::client(&WgpuDevice::default());
            let name = WgpuRuntime::name(&client);
            if !name.contains("spirv") {
                eprintln!("runtime is `{name}`, not wgpu<spirv>: refusing to write Vulkan SPIR-V");
                std::process::exit(1);
            }
            let hw_plane = client.properties().hardware.plane_size_max;
            let force_p64 = std::env::var_os("ML_KERNEL_GEN_P64")
                .or_else(|| std::env::var_os("HANZO_KERNEL_GEN_P64"))
                .is_some();
            let plane = if force_p64 { 64 } else { hw_plane };
            let mut all_specs = ternary_specs::<WgpuRuntime>();
            all_specs.extend(gdn_and_flash_prod_specs::<WgpuRuntime>(false));
            let specs: Vec<_> = all_specs
                .into_iter()
                .filter(|s| {
                    if force_p64 && hw_plane != 64 {
                        return s.artifact.ends_with("_f32")
                            && (s.artifact.starts_with("flash_attn_paged_split_kv_")
                                || s.artifact.starts_with("flash_attn_split_kv_"));
                    }
                    // Emit f32 ternary + GDN + paged/split-KV FlashAttention + FWHT kernels.
                    s.artifact.ends_with("_f32")
                        && (plane == 32
                            || s.artifact.starts_with("matmul_")
                            || s.artifact.starts_with("gdn_")
                            || s.artifact.starts_with("flash_attn_paged_")
                            || s.artifact.starts_with("flash_attn_split_kv_")
                            || s.artifact.starts_with("fwht_"))
                })
                .collect();
            let mut items: Vec<(String, Vec<u8>)> = Vec::with_capacity(specs.len());
            for spec in &specs {
                for ent in std::fs::read_dir(&spv_tmp)
                    .expect("read spv temp dir")
                    .flatten()
                {
                    let _ = std::fs::remove_file(ent.path());
                }
                (spec.launch)(&client);
                let spvs: Vec<PathBuf> = std::fs::read_dir(&spv_tmp)
                    .expect("read spv temp dir")
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("spv"))
                    .collect();
                if spvs.len() != 1 {
                    eprintln!(
                        "vulkan kernel-gen: expected 1 .spv for `{}`, found {}",
                        spec.artifact,
                        spvs.len()
                    );
                    std::process::exit(1);
                }
                let raw = std::fs::read(&spvs[0]).expect("read dumped spv");
                let bytes = rename_spv_entry_to_main(&raw);
                let stem = if plane == 64 {
                    format!("{}_p64", spec.artifact)
                } else {
                    spec.artifact.to_string()
                };
                items.push((stem, bytes));
            }
            let _ = std::fs::remove_dir_all(&spv_tmp);
            write_or_check_spv(&dir, check, &items)
        }
        t => {
            eprintln!("target `{t}` not built in (enable its cargo feature) or not supported yet");
            std::process::exit(2);
        }
    };
    let _ = std::fs::remove_file(&log);
    std::process::exit(if ok { 0 } else { 1 });
}
