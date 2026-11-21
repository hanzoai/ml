//! kernel-gen: regenerate (or drift-check) the committed DSL backend artifacts hanzo-ml dispatches.
//!
//!   kernel-gen --target metal --out ../hanzo-ml/src/metal/dsl          # write artifacts
//!   kernel-gen --target metal --check ../hanzo-ml/src/metal/dsl        # fail on any drift
//!
//! Each artifact is captured from the runtime's own compilation log (see `hanzo_kernel::gen`), so the
//! committed text is exactly what that backend compiled. Run on a host that has the target device.

use hanzo_kernel::gen::{capture, Spec};
use hanzo_kernel::prelude::*;
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: kernel-gen --target metal (--out DIR | --check DIR)");
    std::process::exit(2);
}

/// One PQ2_0 matvec specialization, launched on a 3-row x 256-col tensor (odd block count, so the
/// half-word path and the clamped tail are both compiled in).
#[cfg(feature = "metal")]
macro_rules! pq2_spec {
    ($artifact:literal, $entry:literal, $e:ty, $conv:expr, $nsg:literal, $nr0:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (rows, k) = (3usize, 256usize);
                let (codes, d) = pq2_0_synth(rows, k, 1);
                let words = pq2_0_gguf_words(&pq2_0_gguf_bytes(&codes, &d, rows, k));
                let x: Vec<$e> = (0..k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matvec_pq2_0_gguf_run_as::<$e, cubecl::wgpu::WgpuRuntime>(
                    c, &words, &x, rows, k, $nsg, $nr0,
                );
            },
        }
    };
}

/// One PQ2_0 prefill GEMM specialization (`f32` or `bf16`, `bm = 32` or `64`), launched on a small ragged tile.
#[cfg(feature = "metal")]
macro_rules! pq2_gemm_spec {
    ($artifact:literal, $entry:literal, $e:ty, $conv:expr, $bm:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (m, n, k) = (17usize, 73usize, 256usize);
                let (codes, d) = pq2_0_synth(n, k, 1);
                let words = pq2_0_gguf_words(&pq2_0_gguf_bytes(&codes, &d, n, k));
                let x: Vec<$e> = (0..m * k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matmul_pq2_0_gguf_bm_run_as::<$e, cubecl::wgpu::WgpuRuntime>(
                    c, &words, &x, m, n, k, $bm,
                );
            },
        }
    };
}

/// One PTQ1_0 matvec specialization (`28 B` GGUF blocks).
#[cfg(feature = "metal")]
macro_rules! ptq1_spec {
    ($artifact:literal, $entry:literal, $e:ty, $conv:expr, $nsg:literal, $nr0:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (rows, k) = (3usize, 256usize);
                let (codes, d) = ptq1_0_synth(rows, k, 1);
                let words = ptq1_0_gguf_words(&ptq1_0_gguf_bytes(&codes, &d, rows, k));
                let x: Vec<$e> = (0..k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matvec_ptq1_0_gguf_run_as::<$e, cubecl::wgpu::WgpuRuntime>(
                    c, &words, &x, rows, k, $nsg, $nr0,
                );
            },
        }
    };
}

/// One PTQ1_0 prefill GEMM specialization (`f32` or `bf16`, `bm = 32` or `64`).
#[cfg(feature = "metal")]
macro_rules! ptq1_gemm_spec {
    ($artifact:literal, $entry:literal, $e:ty, $conv:expr, $bm:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (m, n, k) = (17usize, 73usize, 256usize);
                let (codes, d) = ptq1_0_synth(n, k, 1);
                let words = ptq1_0_gguf_words(&ptq1_0_gguf_bytes(&codes, &d, n, k));
                let x: Vec<$e> = (0..m * k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matmul_ptq1_0_gguf_bm_run_as::<$e, cubecl::wgpu::WgpuRuntime>(
                    c, &words, &x, m, n, k, $bm,
                );
            },
        }
    };
}

/// Every Metal artifact hanzo-ml commits under `src/metal/dsl/`. Adding a DSL kernel to the Metal
/// path = one entry here + one `kernel_msl` arm in hanzo-ml.
#[cfg(feature = "metal")]
fn metal_specs() -> Vec<Spec<cubecl::wgpu::WgpuRuntime>> {
    use cubecl::wgpu::WgpuRuntime as R;
    vec![
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
        // PQ2_0 decode matvec over native GGUF blocks: one artifact per (NSG, NR0) schedule x dtype.
        // The entry name is the same for every schedule (cubecl names it after kernel + elem), so
        // the ARTIFACT name is the pipeline identity hanzo-ml caches on.
        pq2_spec!("matvec_pq2_0_gguf_nsg2_nr2_f32",  "matvec_pq2_0_gguf_f_f32", f32,        std::convert::identity, 2, 2),
        pq2_spec!("matvec_pq2_0_gguf_nsg2_nr4_f32",  "matvec_pq2_0_gguf_f_f32", f32,        std::convert::identity, 2, 4),
        pq2_spec!("matvec_pq2_0_gguf_nsg2_nr8_f32",  "matvec_pq2_0_gguf_f_f32", f32,        std::convert::identity, 2, 8),
        pq2_spec!("matvec_pq2_0_gguf_nsg2_nr2_bf16", "matvec_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 2, 2),
        pq2_spec!("matvec_pq2_0_gguf_nsg2_nr4_bf16", "matvec_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 2, 4),
        pq2_spec!("matvec_pq2_0_gguf_nsg2_nr8_bf16", "matvec_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 2, 8),
        pq2_spec!("matvec_pq2_0_gguf_nsg4_nr2_f32",  "matvec_pq2_0_gguf_f_f32", f32,        std::convert::identity, 4, 2),
        pq2_spec!("matvec_pq2_0_gguf_nsg4_nr4_f32",  "matvec_pq2_0_gguf_f_f32", f32,        std::convert::identity, 4, 4),
        pq2_spec!("matvec_pq2_0_gguf_nsg4_nr8_f32",  "matvec_pq2_0_gguf_f_f32", f32,        std::convert::identity, 4, 8),
        pq2_spec!("matvec_pq2_0_gguf_nsg4_nr2_bf16", "matvec_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 4, 2),
        pq2_spec!("matvec_pq2_0_gguf_nsg4_nr4_bf16", "matvec_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 4, 4),
        pq2_spec!("matvec_pq2_0_gguf_nsg4_nr8_bf16", "matvec_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 4, 8),
        // PQ2_0 prefill GEMM (M > 1) over native GGUF blocks (`8x8x8` simdgroup_matrix on Metal).
        pq2_gemm_spec!("matmul_pq2_0_gguf_bm32_f32",  "matmul_pq2_0_gguf_f_f32", f32,        std::convert::identity, 32),
        pq2_gemm_spec!("matmul_pq2_0_gguf_bm32_bf16", "matmul_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 32),
        pq2_gemm_spec!("matmul_pq2_0_gguf_bm64_f32",  "matmul_pq2_0_gguf_f_f32", f32,        std::convert::identity, 64),
        pq2_gemm_spec!("matmul_pq2_0_gguf_bm64_bf16", "matmul_pq2_0_gguf_f_",    half::bf16, half::bf16::from_f32, 64),
        // PTQ1_0 decode matvec + prefill GEMM over native 28-byte GGUF blocks.
        ptq1_spec!("matvec_ptq1_0_gguf_nsg2_nr2_f32",  "matvec_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 2, 2),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg2_nr4_f32",  "matvec_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 2, 4),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg2_nr8_f32",  "matvec_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 2, 8),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg2_nr2_bf16", "matvec_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 2, 2),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg2_nr4_bf16", "matvec_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 2, 4),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg2_nr8_bf16", "matvec_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 2, 8),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg4_nr2_f32",  "matvec_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 4, 2),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg4_nr4_f32",  "matvec_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 4, 4),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg4_nr8_f32",  "matvec_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 4, 8),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg4_nr2_bf16", "matvec_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 4, 2),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg4_nr4_bf16", "matvec_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 4, 4),
        ptq1_spec!("matvec_ptq1_0_gguf_nsg4_nr8_bf16", "matvec_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 4, 8),
        ptq1_gemm_spec!("matmul_ptq1_0_gguf_bm32_f32",  "matmul_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 32),
        ptq1_gemm_spec!("matmul_ptq1_0_gguf_bm32_bf16", "matmul_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 32),
        ptq1_gemm_spec!("matmul_ptq1_0_gguf_bm64_f32",  "matmul_ptq1_0_gguf_f_f32", f32,        std::convert::identity, 64),
        ptq1_gemm_spec!("matmul_ptq1_0_gguf_bm64_bf16", "matmul_ptq1_0_gguf_f_",    half::bf16, half::bf16::from_f32, 64),
        // GDN linear-attention ops (`F32`, `F16`, `BF16`): conv1d, conv1d_step, gating, scan_lanes.
        Spec {
            artifact: "gdn_conv1d_k4",
            entry: "gdn_conv1d_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_run_as::<f32, R>(c, &vec![0.0f32; 16], &vec![0.0f32; 16], 1, 4, 4, 4);
            },
        },
        Spec {
            artifact: "gdn_conv1d_k4_f16",
            entry: "gdn_conv1d_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_run_as::<half::f16, R>(c, &vec![half::f16::ZERO; 16], &vec![half::f16::ZERO; 16], 1, 4, 4, 4);
            },
        },
        Spec {
            artifact: "gdn_conv1d_k4_bf16",
            entry: "gdn_conv1d_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_run_as::<half::bf16, R>(c, &vec![half::bf16::ZERO; 16], &vec![half::bf16::ZERO; 16], 1, 4, 4, 4);
            },
        },
        Spec {
            artifact: "gdn_conv1d_step_k4_f32",
            entry: "gdn_conv1d_step_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_step_run_as::<f32, R>(c, &vec![0.0f32; 4], &vec![0.0f32; 16], &vec![0.0f32; 16], 1, 4, 4);
            },
        },
        Spec {
            artifact: "gdn_conv1d_step_k4_f16",
            entry: "gdn_conv1d_step_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_step_run_as::<half::f16, R>(c, &vec![half::f16::ZERO; 4], &vec![half::f16::ZERO; 16], &vec![half::f16::ZERO; 16], 1, 4, 4);
            },
        },
        Spec {
            artifact: "gdn_conv1d_step_k4_bf16",
            entry: "gdn_conv1d_step_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_conv1d_step_run_as::<half::bf16, R>(c, &vec![half::bf16::ZERO; 4], &vec![half::bf16::ZERO; 16], &vec![half::bf16::ZERO; 16], 1, 4, 4);
            },
        },
        Spec {
            artifact: "gdn_gating",
            entry: "gdn_gating_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_gating_run_as::<f32, R>(c, &vec![0.0f32; 4], &vec![0.0f32; 4], &vec![0.0f32; 4], &vec![0.0f32; 4], 4);
            },
        },
        Spec {
            artifact: "gdn_gating_f16",
            entry: "gdn_gating_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_gating_run_as::<half::f16, R>(c, &vec![half::f16::ZERO; 4], &vec![half::f16::ZERO; 4], &vec![0.0f32; 4], &vec![0.0f32; 4], 4);
            },
        },
        Spec {
            artifact: "gdn_gating_bf16",
            entry: "gdn_gating_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_gating_run_as::<half::bf16, R>(c, &vec![half::bf16::ZERO; 4], &vec![half::bf16::ZERO; 4], &vec![0.0f32; 4], &vec![0.0f32; 4], 4);
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l4",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<f32, R>(
                    c, &vec![0.0f32; 128], &vec![0.0f32; 128], &vec![0.0f32; 128], &[0.0f32], &[0.0f32],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 4, 64, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l4_f16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &[half::f16::ZERO], &[half::f16::ZERO],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 4, 64, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l4_bf16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &[half::bf16::ZERO], &[half::bf16::ZERO],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 4, 64, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l8",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<f32, R>(
                    c, &vec![0.0f32; 128], &vec![0.0f32; 128], &vec![0.0f32; 128], &[0.0f32], &[0.0f32],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 8, 128, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l8_f16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &[half::f16::ZERO], &[half::f16::ZERO],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 8, 128, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l8_bf16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &[half::bf16::ZERO], &[half::bf16::ZERO],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 8, 128, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l16_f32",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<f32, R>(
                    c, &vec![0.0f32; 128], &vec![0.0f32; 128], &vec![0.0f32; 128], &[0.0f32], &[0.0f32],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 16, 128, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l16_f16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &[half::f16::ZERO], &[half::f16::ZERO],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 16, 128, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k128_v128_l16_bf16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &[half::bf16::ZERO], &[half::bf16::ZERO],
                    &vec![0.0f32; 128 * 128], 1, 1, 128, 128, 16, 128, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k64_v64_l4",
            entry: "gdn_scan_lanes_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<f32, R>(
                    c, &vec![0.0f32; 64], &vec![0.0f32; 64], &vec![0.0f32; 64], &[0.0f32], &[0.0f32],
                    &vec![0.0f32; 64 * 64], 1, 1, 64, 64, 4, 64, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k64_v64_l4_f16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64], &[half::f16::ZERO], &[half::f16::ZERO],
                    &vec![0.0f32; 64 * 64], 1, 1, 64, 64, 4, 64, false,
                );
            },
        },
        Spec {
            artifact: "gdn_scan_lanes_k64_v64_l4_bf16",
            entry: "gdn_scan_lanes_f_",
            launch: |c| {
                let _ = hanzo_kernel::gdn::gdn_scan_lanes_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64], &[half::bf16::ZERO], &[half::bf16::ZERO],
                    &vec![0.0f32; 64 * 64], 1, 1, 64, 64, 4, 64, false,
                );
            },
        },
        // Tiled online-softmax FlashAttention (`8x8` simdgroup_matrix on Metal, plane=32).
        Spec {
            artifact: "flash_attn_d64_p32",
            entry: "flash_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<f32, R>(
                    c, &vec![0.0f32; 64], &vec![0.0f32; 64], &vec![0.0f32; 64], 1, 1, 1, 1, 1, 1, 64, 0, 16, 16, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d64_p32_f16",
            entry: "flash_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64], 1, 1, 1, 1, 1, 1, 64, 0, 16, 16, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d64_p32_bf16",
            entry: "flash_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64], 1, 1, 1, 1, 1, 1, 64, 0, 16, 16, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d128_p32",
            entry: "flash_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<f32, R>(
                    c, &vec![0.0f32; 128], &vec![0.0f32; 128], &vec![0.0f32; 128], 1, 1, 1, 1, 1, 1, 128, 0, 16, 16, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d128_p32_f16",
            entry: "flash_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128], 1, 1, 1, 1, 1, 1, 128, 0, 16, 16, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d128_p32_bf16",
            entry: "flash_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128], 1, 1, 1, 1, 1, 1, 128, 0, 16, 16, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d256_p32_f32",
            entry: "flash_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<f32, R>(
                    c, &vec![0.0f32; 256], &vec![0.0f32; 256], &vec![0.0f32; 256], 1, 1, 1, 1, 1, 1, 256, 0, 8, 8, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d256_p32_f16",
            entry: "flash_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 256], &vec![half::f16::ZERO; 256], &vec![half::f16::ZERO; 256], 1, 1, 1, 1, 1, 1, 256, 0, 8, 8, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        Spec {
            artifact: "flash_attn_d256_p32_bf16",
            entry: "flash_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::flash_attn_launch_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 256], &vec![half::bf16::ZERO; 256], &vec![half::bf16::ZERO; 256], 1, 1, 1, 1, 1, 1, 256, 0, 8, 8, 32, hanzo_kernel::prelude::Target::Metal,
                );
            },
        },
        // Paged decode attention + KV block writer (`F32` x=4, `F16`/`BF16` x=8; runtime block_size).
        Spec {
            artifact: "paged_attn_d64_x4_f32",
            entry: "paged_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<f32, R>(
                    c, &vec![0.0f32; 64], &vec![0.0f32; 64 * 16], &vec![0.0f32; 64 * 16], &[0], &[1], 1, 1, 1, 64, 16, 4, 1, 0.125, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d64_x8_f16",
            entry: "paged_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64 * 16], &vec![half::f16::ZERO; 64 * 16], &[0], &[1], 1, 1, 1, 64, 16, 8, 1, 0.125, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d64_x8_bf16",
            entry: "paged_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64 * 16], &vec![half::bf16::ZERO; 64 * 16], &[0], &[1], 1, 1, 1, 64, 16, 8, 1, 0.125, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d128_x4_f32",
            entry: "paged_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<f32, R>(
                    c, &vec![0.0f32; 128], &vec![0.0f32; 128 * 16], &vec![0.0f32; 128 * 16], &[0], &[1], 1, 1, 1, 128, 16, 4, 1, 0.08838835, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d128_x8_f16",
            entry: "paged_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 128], &vec![half::f16::ZERO; 128 * 16], &vec![half::f16::ZERO; 128 * 16], &[0], &[1], 1, 1, 1, 128, 16, 8, 1, 0.08838835, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d128_x8_bf16",
            entry: "paged_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 128], &vec![half::bf16::ZERO; 128 * 16], &vec![half::bf16::ZERO; 128 * 16], &[0], &[1], 1, 1, 1, 128, 16, 8, 1, 0.08838835, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d256_x4_f32",
            entry: "paged_attn_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<f32, R>(
                    c, &vec![0.0f32; 256], &vec![0.0f32; 256 * 16], &vec![0.0f32; 256 * 16], &[0], &[1], 1, 1, 1, 256, 16, 4, 1, 0.0625, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d256_x8_f16",
            entry: "paged_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 256], &vec![half::f16::ZERO; 256 * 16], &vec![half::f16::ZERO; 256 * 16], &[0], &[1], 1, 1, 1, 256, 16, 8, 1, 0.0625, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "paged_attn_d256_x8_bf16",
            entry: "paged_attn_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::paged_attn_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 256], &vec![half::bf16::ZERO; 256 * 16], &vec![half::bf16::ZERO; 256 * 16], &[0], &[1], 1, 1, 1, 256, 16, 8, 1, 0.0625, 0.0, 0,
                );
            },
        },
        Spec {
            artifact: "reshape_and_cache_x4_f32",
            entry: "reshape_and_cache_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::flash::reshape_and_cache_run_as::<f32, R>(
                    c, &vec![0.0f32; 64], &vec![0.0f32; 64], &vec![0.0f32; 64 * 16], &vec![0.0f32; 64 * 16], &[0i64], 1, 1, 64, 16, 4,
                );
            },
        },
        Spec {
            artifact: "reshape_and_cache_x8_f16",
            entry: "reshape_and_cache_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::reshape_and_cache_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64], &vec![half::f16::ZERO; 64 * 16], &vec![half::f16::ZERO; 64 * 16], &[0i64], 1, 1, 64, 16, 8,
                );
            },
        },
        Spec {
            artifact: "reshape_and_cache_x8_bf16",
            entry: "reshape_and_cache_f_",
            launch: |c| {
                let _ = hanzo_kernel::flash::reshape_and_cache_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64], &vec![half::bf16::ZERO; 64 * 16], &vec![half::bf16::ZERO; 64 * 16], &[0i64], 1, 1, 64, 16, 8,
                );
            },
        },
        // Fused normalized FWHT with optional explicit ±1 signs (warp plane_shuffle_xor + smem).
        Spec {
            artifact: "hadamard_fwht_b64_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 64], None, 64, 64, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b64_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 64], None, 64, 64, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b64_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 64], None, 64, 64, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b128_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 128], None, 128, 128, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b128_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 128], None, 128, 128, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b128_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 128], None, 128, 128, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b256_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 256], None, 256, 256, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b256_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 256], None, 256, 256, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b256_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 256], None, 256, 256, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b512_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 512], None, 512, 512, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b512_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 512], None, 512, 512, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b512_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 512], None, 512, 512, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b1024_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 1024], None, 1024, 1024, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b1024_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 1024], None, 1024, 1024, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b1024_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 1024], None, 1024, 1024, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b2048_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 2048], None, 2048, 2048, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b2048_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 2048], None, 2048, 2048, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b2048_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 2048], None, 2048, 2048, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b4096_f32",
            entry: "hadamard_fwht_f_f32",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<f32, R>(
                    c, &vec![0.0f32; 4096], None, 4096, 4096, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b4096_f16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::f16, R>(
                    c, &vec![half::f16::ZERO; 4096], None, 4096, 4096, false, true,
                );
            },
        },
        Spec {
            artifact: "hadamard_fwht_b4096_bf16",
            entry: "hadamard_fwht_f_",
            launch: |c| {
                let _ = hanzo_kernel::fwht::hadamard_fwht_run_as::<half::bf16, R>(
                    c, &vec![half::bf16::ZERO; 4096], None, 4096, 4096, false, true,
                );
            },
        },
    ]
}

fn write_or_check(dir: &Path, check: bool, ext: &str, items: &[(&str, String)]) -> bool {
    let mut ok = true;
    for (stem, src) in items {
        let path = dir.join(format!("{stem}.{ext}"));
        if check {
            match std::fs::read_to_string(&path) {
                Ok(cur) if cur == *src => eprintln!("ok     {}", path.display()),
                Ok(cur) => {
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
            std::fs::write(&path, src).expect("write artifact");
            eprintln!("wrote  {}", path.display());
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
        t => {
            eprintln!("target `{t}` not built in (enable its cargo feature) or not supported yet");
            std::process::exit(2);
        }
    };
    let _ = std::fs::remove_file(&log);
    std::process::exit(if ok { 0 } else { 1 });
}
