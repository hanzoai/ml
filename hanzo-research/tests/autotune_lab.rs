// SPDX-License-Identifier: MIT OR Apache-2.0

use hanzo_research::autotune_lab::{
    check_hardware_feasibility, evaluate_end_to_end_gate, run_autotune_laboratory,
    simulate_schedule_scorecard, AcceleratorBackend, FusionChoice, HardwareTargetProfile,
    KernelScheduleCandidate, PromotionVerdict, PruningReason, QuantizationFormat, ZenModelScale,
};

#[test]
fn prunes_oversized_shared_memory_on_metal_and_hip_while_allowing_cuda_hopper() {
    let big_tile = KernelScheduleCandidate {
        schedule_id: "big_smem_128x128x64_s3".to_string(),
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        waves_per_block: 8,
        software_stages: 3,
        split_k: 1,
        quant_group_size: 128,
        fusion: FusionChoice::FusedSwigluRmsNorm,
        decode_dispatch_max_m: 8,
    };
    // (128*64*2 + 128*64*2) * 3 = 98,304 bytes -> exceeds Metal (32 KB) and HIP RDNA3.5 (64 KB), fits SM90 (160 KB)
    assert_eq!(big_tile.required_shared_mem_bytes(), 98_304);

    let metal = HardwareTargetProfile::metal_apple_gpu_reference();
    let hip = HardwareTargetProfile::hip_gfx1151_reference();
    let cuda = HardwareTargetProfile::cuda_sm90_reference();

    assert!(matches!(
        check_hardware_feasibility(&big_tile, &metal),
        Err(PruningReason::SharedMemoryExceeded { .. })
    ));
    assert!(matches!(
        check_hardware_feasibility(&big_tile, &hip),
        Err(PruningReason::SharedMemoryExceeded { .. })
    ));
    assert!(check_hardware_feasibility(&big_tile, &cuda).is_ok());
}

#[test]
fn explicitly_rejects_splitk_microbenchmark_only_trap_that_regresses_20_turn_agent_loop() {
    let hw = HardwareTargetProfile::hip_gfx1151_reference();
    let model = ZenModelScale::zen_9b();
    let quant = QuantizationFormat::Int4Group128;

    let baseline = simulate_schedule_scorecard(
        &KernelScheduleCandidate::generic_baseline(quant),
        &hw,
        &model,
        quant,
        "commit-ref-01",
    );

    let splitk_trap = KernelScheduleCandidate {
        schedule_id: "trap_splitk8_64x64x32".to_string(),
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        waves_per_block: 4,
        software_stages: 2,
        split_k: 8,
        quant_group_size: 128,
        fusion: FusionChoice::Unfused,
        decode_dispatch_max_m: 1,
    };
    let trap_sc = simulate_schedule_scorecard(&splitk_trap, &hw, &model, quant, "commit-ref-01");

    // Verify isolated GEMM wins while full 20-turn coding-agent loop regresses due to L2 thrash + extra barriers
    assert!(trap_sc.isolated_microbench_us < baseline.isolated_microbench_us);
    assert!(trap_sc.agent_20turn_wall_ms > baseline.agent_20turn_wall_ms);

    let verdict = evaluate_end_to_end_gate(&baseline, &trap_sc);
    assert!(matches!(
        verdict,
        PromotionVerdict::RejectedMicrobenchOnlyTrap { .. }
    ));
}

#[test]
fn tunes_cuda_hip_and_metal_across_4b_9b_27b_and_exports_deterministic_plan() {
    let targets = vec![
        HardwareTargetProfile::cuda_sm90_reference(),
        HardwareTargetProfile::hip_gfx1151_reference(),
        HardwareTargetProfile::metal_apple_gpu_reference(),
    ];
    let models = ZenModelScale::shared_matrix();

    let plan_a = run_autotune_laboratory(&targets, &models, QuantizationFormat::Int4Group128, "sha-0x108");
    let plan_b = run_autotune_laboratory(&targets, &models, QuantizationFormat::Int4Group128, "sha-0x108");

    assert_eq!(plan_a.entries.len(), 9);
    assert_eq!(plan_a.plan_sha256, plan_b.plan_sha256);

    for entry in &plan_a.entries {
        assert!(entry.rejected_microbench_traps > 0);
        assert!(entry.winner_scorecard.output_correct);
        assert!(entry.winner_scorecard.agent_20turn_wall_ms < entry.baseline_scorecard.agent_20turn_wall_ms);
        assert!(entry.winner_scorecard.p50_itl_ms < entry.baseline_scorecard.p50_itl_ms);
        assert!(entry.winner_scorecard.p95_itl_ms < entry.baseline_scorecard.p95_itl_ms);
        assert!(entry.winner_scorecard.joules_per_token < entry.baseline_scorecard.joules_per_token);
        match entry.verdict {
            PromotionVerdict::Promoted {
                end_to_end_speedup,
                energy_reduction_pct,
                ..
            } => {
                assert!(end_to_end_speedup > 1.25);
                assert!(energy_reduction_pct > 20.0);
            }
            ref other => panic!("expected Promoted winner for {} {}, got {:?}", entry.arch_name, entry.model_tier, other),
        }
    }

    let dispatch_rs = plan_a.render_rust_dispatch_table();
    assert!(dispatch_rs.contains("pub fn select_tuned_tile"));
    assert!(dispatch_rs.contains("sm_90a"));
    assert!(dispatch_rs.contains("gfx1151"));
    assert!(dispatch_rs.contains("apple_m4_max_gpu40"));
    assert!(dispatch_rs.contains(&plan_a.plan_sha256));
    assert_eq!(plan_a.entries[0].backend, AcceleratorBackend::Cuda);
}
