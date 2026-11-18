// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shape-specialised kernel autotuning laboratory and end-to-end promotion gate (`HIP-1345` / `#41`).
//!
//! Searches tile dimensions (`tile_m`, `tile_n`, `tile_k`), warp/wavefront layouts,
//! quantization group sizes, epilogue fusion modes, and batch-size (`M`) dispatch
//! thresholds across `CUDA`, `HIP` (`ROCm`), and `Metal` targets for the shared
//! `Zen-4B` / `Zen-9B` / `Zen-27B` model matrix.
//!
//! Critical invariant (`#41`): configurations that win isolated operator microbenchmarks
//! (for example, high `split_k` atomic/workspace reductions that saturate L2 or add extra
//! kernel launches across a 20-turn coding-agent loop) are **rejected** by
//! [`evaluate_end_to_end_gate`] unless they also improve full-network end-to-end wall time
//! without violating numerical accuracy or quality budgets.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Target accelerator family for shape-specialised kernel synthesis and tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorBackend {
    Cuda,
    Hip,
    Metal,
}

/// Hardware execution profile capturing physical resource limits and roofline parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareTargetProfile {
    pub backend: AcceleratorBackend,
    pub arch_name: String,
    pub driver_version: String,
    pub compute_units: u32,
    pub lane_width: u32,
    pub max_shared_mem_bytes: u32,
    pub max_registers_per_block: u32,
    pub peak_hbm_gbps: f64,
    pub peak_tensor_tflops: f64,
    pub kernel_launch_overhead_us: f64,
    pub l2_cache_bytes: usize,
    pub tdp_watts: f64,
}

impl HardwareTargetProfile {
    /// Canonical NVIDIA Hopper / Ada-class CUDA target profile.
    pub fn cuda_sm90_reference() -> Self {
        Self {
            backend: AcceleratorBackend::Cuda,
            arch_name: "sm_90a".to_string(),
            driver_version: "CUDA-12.6-r560".to_string(),
            compute_units: 132,
            lane_width: 32,
            max_shared_mem_bytes: 163_840,
            max_registers_per_block: 65_536,
            peak_hbm_gbps: 3_350.0,
            peak_tensor_tflops: 989.0,
            kernel_launch_overhead_us: 3.8,
            l2_cache_bytes: 50 * 1024 * 1024,
            tdp_watts: 700.0,
        }
    }

    /// Canonical AMD Strix Halo / RDNA3.5 + CDNA HIP target profile.
    pub fn hip_gfx1151_reference() -> Self {
        Self {
            backend: AcceleratorBackend::Hip,
            arch_name: "gfx1151".to_string(),
            driver_version: "ROCm-6.4.0".to_string(),
            compute_units: 40,
            lane_width: 32,
            max_shared_mem_bytes: 65_536,
            max_registers_per_block: 65_536,
            peak_hbm_gbps: 256.0,
            peak_tensor_tflops: 59.4,
            kernel_launch_overhead_us: 5.4,
            l2_cache_bytes: 32 * 1024 * 1024,
            tdp_watts: 120.0,
        }
    }

    /// Canonical Apple Silicon Metal unified-memory GPU target profile.
    pub fn metal_apple_gpu_reference() -> Self {
        Self {
            backend: AcceleratorBackend::Metal,
            arch_name: "apple_m4_max_gpu40".to_string(),
            driver_version: "Metal-3.2".to_string(),
            compute_units: 40,
            lane_width: 32,
            max_shared_mem_bytes: 32_768,
            max_registers_per_block: 32_768,
            peak_hbm_gbps: 546.0,
            peak_tensor_tflops: 36.8,
            kernel_launch_overhead_us: 4.9,
            l2_cache_bytes: 24 * 1024 * 1024,
            tdp_watts: 78.0,
        }
    }
}

/// Quantization layout pinned into every benchmark scorecard and exported backend plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationFormat {
    Bf16,
    Int8Group128,
    Int4Group128,
    Int4Group64,
}

impl QuantizationFormat {
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            Self::Bf16 => 2.0,
            Self::Int8Group128 => 1.015625,
            Self::Int4Group128 => 0.515625,
            Self::Int4Group64 => 0.53125,
        }
    }

    pub fn group_size(self) -> Option<u32> {
        match self {
            Self::Bf16 => None,
            Self::Int8Group128 | Self::Int4Group128 => Some(128),
            Self::Int4Group64 => Some(64),
        }
    }
}

/// Shared Zen model scale matrix required by the serving & kernel scorecard (`4B` / `9B` / `27B`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZenModelScale {
    pub model_tier: String,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_kv_heads: u32,
}

impl ZenModelScale {
    pub fn zen_4b() -> Self {
        Self {
            model_tier: "zen-4b".to_string(),
            num_layers: 36,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_attention_heads: 32,
            num_kv_heads: 8,
        }
    }

    pub fn zen_9b() -> Self {
        Self {
            model_tier: "zen-9b".to_string(),
            num_layers: 42,
            hidden_size: 3584,
            intermediate_size: 14336,
            num_attention_heads: 28,
            num_kv_heads: 4,
        }
    }

    pub fn zen_27b() -> Self {
        Self {
            model_tier: "zen-27b".to_string(),
            num_layers: 46,
            hidden_size: 4608,
            intermediate_size: 24576,
            num_attention_heads: 36,
            num_kv_heads: 8,
        }
    }

    pub fn shared_matrix() -> Vec<Self> {
        vec![Self::zen_4b(), Self::zen_9b(), Self::zen_27b()]
    }
}

/// Epilogue fusion choices considered by the shape-specialised autotuner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FusionChoice {
    /// Separate dequantize + matmul + activation + residual launches (unfused baseline).
    Unfused,
    /// Fuse groupwise dequantization + GEMM epilogue bias/residual writeback.
    DequantGemmResidual,
    /// Fuse gate+up SwiGLU projection, activation, and RMSNorm residual epilogue.
    FusedSwigluRmsNorm,
}

/// Candidate kernel schedule in the discrete autotuning search space.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelScheduleCandidate {
    pub schedule_id: String,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
    pub waves_per_block: u32,
    pub software_stages: u32,
    pub split_k: u32,
    pub quant_group_size: u32,
    pub fusion: FusionChoice,
    /// Maximum token batch `M` before switching from narrow decode kernel to tiled prefill kernel.
    pub decode_dispatch_max_m: u32,
}

impl KernelScheduleCandidate {
    /// Default untuned vendor/generic schedule used as the baseline comparator.
    pub fn generic_baseline(quant: QuantizationFormat) -> Self {
        Self {
            schedule_id: "baseline_generic_64x64x32_unfused".to_string(),
            tile_m: 64,
            tile_n: 64,
            tile_k: 32,
            waves_per_block: 4,
            software_stages: 2,
            split_k: 1,
            quant_group_size: quant.group_size().unwrap_or(128),
            fusion: FusionChoice::Unfused,
            decode_dispatch_max_m: 1,
        }
    }

    /// Computes dynamic shared-memory / threadgroup staging bytes required per workgroup.
    pub fn required_shared_mem_bytes(&self) -> u32 {
        let a_tile_bytes = self.tile_m * self.tile_k * 2;
        let b_tile_bytes = self.tile_n * self.tile_k * 2;
        (a_tile_bytes + b_tile_bytes) * self.software_stages
    }

    /// Estimates 32-bit registers required per threadblock.
    pub fn required_registers_per_block(&self, lane_width: u32) -> u32 {
        let threads = self.waves_per_block * lane_width;
        let accum_per_thread = ((self.tile_m * self.tile_n) / threads.max(1)).max(8);
        threads * (accum_per_thread + 24)
    }
}

/// Hardware feasibility rejection reason during search-space pruning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningReason {
    SharedMemoryExceeded { required: u32, limit: u32 },
    RegisterFileExceeded { required: u32, limit: u32 },
    MisalignedQuantGroup { tile_k: u32, group_size: u32 },
    InvalidSplitKZero,
}

/// Validates whether a candidate schedule satisfies target hardware resource constraints.
pub fn check_hardware_feasibility(
    candidate: &KernelScheduleCandidate,
    hw: &HardwareTargetProfile,
) -> std::result::Result<(), PruningReason> {
    if candidate.split_k == 0 {
        return Err(PruningReason::InvalidSplitKZero);
    }
    if candidate.tile_k < 16 || (candidate.quant_group_size % candidate.tile_k != 0 && candidate.tile_k % candidate.quant_group_size != 0) {
        return Err(PruningReason::MisalignedQuantGroup {
            tile_k: candidate.tile_k,
            group_size: candidate.quant_group_size,
        });
    }
    let smem = candidate.required_shared_mem_bytes();
    if smem > hw.max_shared_mem_bytes {
        return Err(PruningReason::SharedMemoryExceeded {
            required: smem,
            limit: hw.max_shared_mem_bytes,
        });
    }
    let regs = candidate.required_registers_per_block(hw.lane_width);
    if regs > hw.max_registers_per_block {
        return Err(PruningReason::RegisterFileExceeded {
            required: regs,
            limit: hw.max_registers_per_block,
        });
    }
    Ok(())
}

/// Generates a rich candidate grid spanning decode-specialised, balanced hybrid,
/// prefill-tiled, and adversarial `split_k` microbenchmark-trap schedules.
pub fn generate_candidate_search_space(quant: QuantizationFormat) -> Vec<KernelScheduleCandidate> {
    let g = quant.group_size().unwrap_or(128);
    let mut out = vec![KernelScheduleCandidate::generic_baseline(quant)];

    let tile_ms = [16_u32, 32, 64, 128];
    let tile_ns = [64_u32, 128];
    let tile_ks = [32_u32, 64];
    let waves = [4_u32, 8];
    let stages = [2_u32, 3];
    let split_ks = [1_u32, 2, 8];
    let fusions = [
        FusionChoice::Unfused,
        FusionChoice::DequantGemmResidual,
        FusionChoice::FusedSwigluRmsNorm,
    ];
    let thresholds = [4_u32, 8, 16];

    for &tm in &tile_ms {
        for &tn in &tile_ns {
            for &tk in &tile_ks {
                for &w in &waves {
                    for &st in &stages {
                        for &sk in &split_ks {
                            for &fu in &fusions {
                                if sk == 8 && fu != FusionChoice::DequantGemmResidual {
                                    continue;
                                }
                                if sk < 8 && st == 3 && tm == 128 && tn == 128 {
                                    continue;
                                }
                                let thresh = thresholds[((tm + tn + w) as usize) % thresholds.len()];
                                let fu_tag = match fu {
                                    FusionChoice::Unfused => "unfused",
                                    FusionChoice::DequantGemmResidual => "deq_res",
                                    FusionChoice::FusedSwigluRmsNorm => "swiglu_rms",
                                };
                                out.push(KernelScheduleCandidate {
                                    schedule_id: format!(
                                        "t{tm}x{tn}x{tk}_w{w}_s{st}_sk{sk}_{fu_tag}_m{thresh}"
                                    ),
                                    tile_m: tm,
                                    tile_n: tn,
                                    tile_k: tk,
                                    waves_per_block: w,
                                    software_stages: st,
                                    split_k: sk,
                                    quant_group_size: g,
                                    fusion: fu,
                                    decode_dispatch_max_m: thresh,
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Complete mandatory benchmark scorecard for every candidate evaluation (`#41`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelRunScorecard {
    pub schedule_id: String,
    pub model_tier: String,
    pub arch_name: String,
    pub driver_version: String,
    pub quant_format: QuantizationFormat,
    pub git_commit: String,
    /// Isolated single-layer GEMM microbenchmark latency (microseconds).
    pub isolated_microbench_us: f64,
    /// Cold Time-To-First-Token on initial 512-token prompt (milliseconds).
    pub cold_ttft_ms: f64,
    /// Warm prefix-cached Time-To-First-Token on subsequent turns (milliseconds).
    pub warm_ttft_ms: f64,
    /// Median inter-token decode latency across 20-turn coding-agent generation (milliseconds).
    pub p50_itl_ms: f64,
    /// 95th-percentile inter-token decode latency across long-context turns (milliseconds).
    pub p95_itl_ms: f64,
    /// Full 20-turn coding-agent benchmark wall-clock completion time (milliseconds).
    pub agent_20turn_wall_ms: f64,
    /// Peak resident HBM / unified memory including KV cache + split-K workspaces (bytes).
    pub peak_memory_bytes: usize,
    /// Mean energy consumed per generated token across the 20-turn loop (joules/token).
    pub joules_per_token: f64,
    /// Maximum absolute output logit deviation vs FP32 reference (`<= 5e-3` gate).
    pub max_abs_output_error: f64,
    /// Perplexity delta on held-out coding-agent validation traces (`<= 0.015` gate).
    pub perplexity_regression_delta: f64,
    pub output_correct: bool,
}

/// Promotion decision enforcing end-to-end improvement over isolated microbenchmark traps.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum PromotionVerdict {
    Promoted {
        microbench_speedup: f64,
        end_to_end_speedup: f64,
        energy_reduction_pct: f64,
    },
    RejectedMicrobenchOnlyTrap {
        microbench_speedup: f64,
        end_to_end_ratio: f64,
        reason: String,
    },
    RejectedEndToEndRegression {
        end_to_end_ratio: f64,
    },
    RejectedNumericalOrQuality {
        max_abs_output_error: f64,
        perplexity_regression_delta: f64,
    },
}

/// Evaluates a feasible schedule on a `(HardwareTargetProfile, ZenModelScale, QuantizationFormat)`
/// triple using roof-line + launch/workspace/cache interference physics across both isolated
/// GEMM and the full 20-turn coding-agent loop (`20` turns x `256` prefill delta + `96` decode tokens).
pub fn simulate_schedule_scorecard(
    candidate: &KernelScheduleCandidate,
    hw: &HardwareTargetProfile,
    model: &ZenModelScale,
    quant: QuantizationFormat,
    git_commit: &str,
) -> KernelRunScorecard {
    let h = model.hidden_size as f64;
    let inter = model.intermediate_size as f64;
    let layers = model.num_layers as f64;
    let bpw = quant.bytes_per_weight();

    // Per-layer parameter count (QKV + O + Gate/Up/Down projections)
    let qkv_o_weights = 4.0 * h * h;
    let mlp_weights = 3.0 * h * inter;
    let layer_weights = qkv_o_weights + mlp_weights;
    let total_model_weight_bytes = (layer_weights * layers * bpw) as usize;

    // 1. Isolated single-GEMM microbenchmark (`M=1`, `K=hidden`, `N=intermediate`, warm L2)
    let micro_bytes = h * inter * bpw;
    let tile_m_waste = (candidate.tile_m as f64 / 16.0).max(1.0).powf(0.38);
    let stage_eff = if candidate.software_stages >= 3 { 0.94 } else { 0.86 };
    let wave_occ = if candidate.waves_per_block == 4 { 0.95 } else { 0.88 };
    // In an isolated single-op microbenchmark, high split_k parallelizes across idle SMs/CUs!
    let micro_splitk_boost = match candidate.split_k {
        1 => 0.76,
        2 => 0.93,
        _ => 1.08,
    };
    let effective_micro_bw_gbps =
        hw.peak_hbm_gbps * stage_eff * wave_occ * micro_splitk_boost / tile_m_waste;
    let isolated_microbench_us = (micro_bytes / (effective_micro_bw_gbps * 1e3))
        + hw.kernel_launch_overhead_us;

    // 2. End-to-End Full Network Decode (`M=1`) & Prefill (`M=256..512`) across all layers!
    // Unfused schedules pay 9 kernel launches/layer; fused SwiGLU+RMSNorm cuts to 4 launches/layer.
    let base_launches_per_layer = match candidate.fusion {
        FusionChoice::Unfused => 9.0,
        FusionChoice::DequantGemmResidual => 6.0,
        FusionChoice::FusedSwigluRmsNorm => 4.0,
    };
    // `split_k > 1` adds an extra atomic/workspace reduction barrier launch per linear op AND
    // allocates FP32 partial-sum buffers that thrash L2 cache across back-to-back layers!
    let splitk_extra_launches = if candidate.split_k > 1 {
        4.0 * (candidate.split_k as f64 / 2.0)
    } else {
        0.0
    };
    let splitk_workspace_bytes = if candidate.split_k > 1 {
        (candidate.split_k as usize) * (model.intermediate_size as usize) * 4 * 16
    } else {
        0
    };
    let l2_thrash_penalty = if candidate.split_k >= 8 {
        1.42 // Severe L2 eviction & reduction barrier stall across 36-46 stacked layers!
    } else if candidate.split_k == 2 {
        0.96
    } else {
        1.0
    };

    let fusion_traffic_factor = match candidate.fusion {
        FusionChoice::Unfused => 1.24, // Extra HBM round-trips for unfused dequant + SwiGLU intermediates
        FusionChoice::DequantGemmResidual => 1.06,
        FusionChoice::FusedSwigluRmsNorm => 0.91,
    };

    // Shape-adaptive dispatch threshold bonus: narrow tile_m (16/32) wins decode while switching cleanly on prefill
    let decode_tile_eff = match candidate.tile_m {
        16 => 0.96,
        32 => 0.92,
        64 => 0.81,
        _ => 0.68,
    };
    let dispatch_adapt_bonus = if (4..=16).contains(&candidate.decode_dispatch_max_m) {
        0.95
    } else {
        1.03
    };

    let layer_decode_mem_ms = ((layer_weights * bpw * fusion_traffic_factor * l2_thrash_penalty)
        / (hw.peak_hbm_gbps * decode_tile_eff * stage_eff * wave_occ * 1e6))
        * dispatch_adapt_bonus;
    let layer_launch_ms =
        ((base_launches_per_layer + splitk_extra_launches) * hw.kernel_launch_overhead_us) / 1e3;
    let p50_itl_ms = (layer_decode_mem_ms + layer_launch_ms) * layers;
    let p95_itl_ms = p50_itl_ms * (1.11 + if candidate.split_k >= 8 { 0.14 } else { 0.0 });

    // Prefill compute + memory time (`512` cold tokens, `256` warm delta tokens per turn)
    let prefill_flops_per_tok = 2.0 * layer_weights * layers;
    let prefill_tile_eff = match (candidate.tile_m, candidate.tile_n) {
        (64, 128) | (128, 128) => 0.88,
        (32, 128) | (64, 64) => 0.82,
        (16, 128) => if candidate.decode_dispatch_max_m >= 8 { 0.86 } else { 0.72 },
        _ => 0.74,
    };
    let prefill_512_compute_ms =
        ((prefill_flops_per_tok * 512.0) / (hw.peak_tensor_tflops * prefill_tile_eff * 1e9))
            * fusion_traffic_factor;
    let cold_compile_warmup_ms = 14.5 + (candidate.software_stages as f64) * 1.5;
    let cold_ttft_ms = prefill_512_compute_ms + p50_itl_ms + cold_compile_warmup_ms;
    let warm_ttft_ms = (prefill_512_compute_ms * 0.48) + p50_itl_ms * 0.65;

    // 20-turn coding-agent benchmark: 1 cold turn + 19 warm turns, 96 generated tokens/turn = 1,920 tokens
    let total_generated_tokens = 20.0 * 96.0;
    let agent_20turn_wall_ms =
        cold_ttft_ms + 19.0 * warm_ttft_ms + total_generated_tokens * p50_itl_ms;

    let kv_cache_bytes = (model.num_layers as usize)
        * (model.num_kv_heads as usize)
        * 128
        * 2
        * 6144
        * 2;
    let peak_memory_bytes = total_model_weight_bytes + kv_cache_bytes + splitk_workspace_bytes;

    let power_draw_factor = match candidate.fusion {
        FusionChoice::Unfused => 0.88,
        FusionChoice::DequantGemmResidual => 0.83,
        FusionChoice::FusedSwigluRmsNorm => 0.79,
    } * if candidate.split_k >= 8 { 1.12 } else { 1.0 };
    let total_energy_joules = (agent_20turn_wall_ms / 1_000.0) * hw.tdp_watts * power_draw_factor;
    let joules_per_token = total_energy_joules / total_generated_tokens;

    // Numerical parity & quality tracking (`split_k >= 8` non-associative atomic reduction adds FP rounding drift)
    let base_quant_err = match quant {
        QuantizationFormat::Bf16 => 0.00045,
        QuantizationFormat::Int8Group128 => 0.00095,
        QuantizationFormat::Int4Group128 => 0.00280,
        QuantizationFormat::Int4Group64 => 0.00210,
    };
    let max_abs_output_error = base_quant_err + (candidate.split_k.saturating_sub(1) as f64) * 0.00025;
    let perplexity_regression_delta = match quant {
        QuantizationFormat::Bf16 => 0.0005,
        QuantizationFormat::Int8Group128 => 0.0021,
        QuantizationFormat::Int4Group128 => 0.0084,
        QuantizationFormat::Int4Group64 => 0.0062,
    };
    let output_correct = max_abs_output_error <= 0.005 && perplexity_regression_delta <= 0.015;

    KernelRunScorecard {
        schedule_id: candidate.schedule_id.clone(),
        model_tier: model.model_tier.clone(),
        arch_name: hw.arch_name.clone(),
        driver_version: hw.driver_version.clone(),
        quant_format: quant,
        git_commit: git_commit.to_string(),
        isolated_microbench_us: round4(isolated_microbench_us),
        cold_ttft_ms: round4(cold_ttft_ms),
        warm_ttft_ms: round4(warm_ttft_ms),
        p50_itl_ms: round4(p50_itl_ms),
        p95_itl_ms: round4(p95_itl_ms),
        agent_20turn_wall_ms: round4(agent_20turn_wall_ms),
        peak_memory_bytes,
        joules_per_token: round4(joules_per_token),
        max_abs_output_error: round6(max_abs_output_error),
        perplexity_regression_delta: round6(perplexity_regression_delta),
        output_correct,
    }
}

/// Strictly gates candidate promotion on end-to-end 20-turn inference AND numerical/quality parity.
/// Any configuration that wins isolated GEMM microbenchmarks (`microbench_speedup > 1.02`) while
/// failing to beat the baseline end-to-end (`end_to_end_speedup <= 1.01`) is explicitly rejected!
pub fn evaluate_end_to_end_gate(
    baseline: &KernelRunScorecard,
    candidate: &KernelRunScorecard,
) -> PromotionVerdict {
    if !candidate.output_correct
        || candidate.max_abs_output_error > 0.005
        || candidate.perplexity_regression_delta > 0.015
    {
        return PromotionVerdict::RejectedNumericalOrQuality {
            max_abs_output_error: candidate.max_abs_output_error,
            perplexity_regression_delta: candidate.perplexity_regression_delta,
        };
    }

    let micro_speedup = baseline.isolated_microbench_us / candidate.isolated_microbench_us.max(1e-9);
    let e2e_speedup = baseline.agent_20turn_wall_ms / candidate.agent_20turn_wall_ms.max(1e-9);

    if e2e_speedup <= 1.01 || candidate.p95_itl_ms > baseline.p95_itl_ms {
        if micro_speedup > 1.02 {
            return PromotionVerdict::RejectedMicrobenchOnlyTrap {
                microbench_speedup: round4(micro_speedup),
                end_to_end_ratio: round4(e2e_speedup),
                reason: format!(
                    "Won isolated GEMM ({:.2}x) but regressed/stalled 20-turn end-to-end loop ({:.2}x)",
                    micro_speedup, e2e_speedup
                ),
            };
        }
        return PromotionVerdict::RejectedEndToEndRegression {
            end_to_end_ratio: round4(e2e_speedup),
        };
    }

    let energy_red =
        ((baseline.joules_per_token - candidate.joules_per_token) / baseline.joules_per_token) * 100.0;
    PromotionVerdict::Promoted {
        microbench_speedup: round4(micro_speedup),
        end_to_end_speedup: round4(e2e_speedup),
        energy_reduction_pct: round4(energy_red),
    }
}

/// Exported winner entry for a single `(arch, model_tier, quant_format)` cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TunedWinnerEntry {
    pub backend: AcceleratorBackend,
    pub arch_name: String,
    pub model_tier: String,
    pub quant_format: QuantizationFormat,
    pub winning_schedule: KernelScheduleCandidate,
    pub baseline_scorecard: KernelRunScorecard,
    pub winner_scorecard: KernelRunScorecard,
    pub verdict: PromotionVerdict,
    pub candidates_evaluated: usize,
    pub pruned_infeasible: usize,
    pub rejected_microbench_traps: usize,
}

/// Versioned, content-addressed backend tuning manifest ready for `hanzo-kernel` / `hanzo-nn` dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportedBackendPlan {
    pub schema_version: u32,
    pub git_commit: String,
    pub entries: Vec<TunedWinnerEntry>,
    pub plan_sha256: String,
}

impl ExportedBackendPlan {
    /// Generates a deterministic static Rust dispatch snippet mapping `(arch, model_tier, m_tokens)`
    /// to the verified shape-specialised tile and fusion configuration without any runtime search dependency.
    pub fn render_rust_dispatch_table(&self) -> String {
        let mut lines = Vec::new();
        lines.push("// Auto-generated reproducible shape-specialised dispatch table (HIP-1345)".to_string());
        lines.push(format!("// Plan SHA-256: {}", self.plan_sha256));
        lines.push("pub fn select_tuned_tile(arch: &str, model_tier: &str, m_tokens: u32) -> (u32, u32, u32, u32, &'static str) {".to_string());
        lines.push("    match (arch, model_tier) {".to_string());
        for entry in &self.entries {
            let s = &entry.winning_schedule;
            let fu_name = match s.fusion {
                FusionChoice::Unfused => "unfused",
                FusionChoice::DequantGemmResidual => "dequant_gemm_residual",
                FusionChoice::FusedSwigluRmsNorm => "fused_swiglu_rmsnorm",
            };
            lines.push(format!(
                "        ({:?}, {:?}) => if m_tokens <= {} {{ ({}, {}, {}, {}, {:?}) }} else {{ (64, 128, {}, {}, {:?}) }},",
                entry.arch_name,
                entry.model_tier,
                s.decode_dispatch_max_m,
                s.tile_m,
                s.tile_n,
                s.tile_k,
                s.waves_per_block,
                fu_name,
                s.tile_k,
                s.waves_per_block,
                fu_name
            ));
        }
        lines.push("        _ => (64, 64, 32, 4, \"unfused\"),".to_string());
        lines.push("    }".to_string());
        lines.push("}".to_string());
        lines.join("\n")
    }
}

/// Runs the full autotuning search across hardware targets and the shared `4B`/`9B`/`27B` matrix,
/// pruning hardware-infeasible candidates, rejecting microbenchmark-only traps, and exporting a
/// SHA-256 content-addressed [`ExportedBackendPlan`].
pub fn run_autotune_laboratory(
    targets: &[HardwareTargetProfile],
    models: &[ZenModelScale],
    quant: QuantizationFormat,
    git_commit: &str,
) -> ExportedBackendPlan {
    let candidates = generate_candidate_search_space(quant);
    let baseline_sched = KernelScheduleCandidate::generic_baseline(quant);
    let mut entries = Vec::new();

    for hw in targets {
        for model in models {
            let baseline_sc = simulate_schedule_scorecard(&baseline_sched, hw, model, quant, git_commit);
            let mut best_sched = baseline_sched.clone();
            let mut best_sc = baseline_sc.clone();
            let mut best_verdict = PromotionVerdict::RejectedEndToEndRegression {
                end_to_end_ratio: 1.0,
            };

            let mut evaluated = 0_usize;
            let mut pruned = 0_usize;
            let mut micro_traps = 0_usize;

            for cand in &candidates {
                if check_hardware_feasibility(cand, hw).is_err() {
                    pruned += 1;
                    continue;
                }
                evaluated += 1;
                let sc = simulate_schedule_scorecard(cand, hw, model, quant, git_commit);
                let verdict = evaluate_end_to_end_gate(&baseline_sc, &sc);
                match &verdict {
                    PromotionVerdict::Promoted { .. } => {
                        if sc.agent_20turn_wall_ms < best_sc.agent_20turn_wall_ms {
                            best_sched = cand.clone();
                            best_sc = sc;
                            best_verdict = verdict;
                        }
                    }
                    PromotionVerdict::RejectedMicrobenchOnlyTrap { .. } => {
                        micro_traps += 1;
                    }
                    _ => {}
                }
            }

            entries.push(TunedWinnerEntry {
                backend: hw.backend,
                arch_name: hw.arch_name.clone(),
                model_tier: model.model_tier.clone(),
                quant_format: quant,
                winning_schedule: best_sched,
                baseline_scorecard: baseline_sc,
                winner_scorecard: best_sc,
                verdict: best_verdict,
                candidates_evaluated: evaluated,
                pruned_infeasible: pruned,
                rejected_microbench_traps: micro_traps,
            });
        }
    }

    let serialized_body = serde_json::to_vec(&entries).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(git_commit.as_bytes());
    hasher.update(&serialized_body);
    let plan_sha256 = format!("{:x}", hasher.finalize());

    ExportedBackendPlan {
        schema_version: 1,
        git_commit: git_commit.to_string(),
        entries,
        plan_sha256,
    }
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

fn round6(v: f64) -> f64 {
    (v * 1_000_000.0).round() / 1_000_000.0
}
