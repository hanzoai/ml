//! Integration tests and 20-turn coding-agent scorecard verification for
//! `hanzo_nn::ternary_lut` (#40, HIP-1345).

use hanzo_ml::{DType, Device, Module, Result, Tensor};
use hanzo_nn::ternary_lut::{
    build_radix81_lut, build_symmetric_lut41, check_end_to_end_acceptance_gate,
    fp32_reference_matvec, lookup_symmetric_lut41, pack_trit_group4, quantize_absmean_ternary,
    quantize_absmean_ternary_grouped, radix81_to_trits, run_tmac_scorecard, ternary_lut_matvec_f32,
    ternary_lut_matvec_f32_parallel, ternary_lut_matvec_i32, trits_to_radix81, unpack_trit_group4,
    weight_unpack_deq_matvec_f32, ActivationLut, HardwareMetadata, ModelScaleProfile, OperatorKind,
    OperatorScorecardRow, PackedTernaryMatrix, SimdLutGroup, TernaryTransformerModel,
    TmacKernelBackend, TmacLinear, RADIX3_GROUP4_STATES,
};
use std::sync::Arc;

fn deterministic_trits(len: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            match (s >> 40) % 3 {
                0 => -1,
                1 => 0,
                _ => 1,
            }
        })
        .collect()
}

fn deterministic_i8_acts(len: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let raw = ((s >> 40) % 255) as i16 - 127;
            raw as i8
        })
        .collect()
}

fn reference_i32_matvec(trits: &[i8], rows: usize, cols: usize, qx: &[i8]) -> Vec<i32> {
    let mut out = vec![0i32; rows];
    for r in 0..rows {
        let mut acc = 0i32;
        for c in 0..cols {
            acc += i32::from(trits[r * cols + c]) * i32::from(qx[c]);
        }
        out[r] = acc;
    }
    out
}

#[test]
fn trit_group4_all_81_states_and_symmetric41_mirror() {
    for q in [
        [127i8, 127, 127, 127],
        [-127, 127, -127, 127],
        [0, -1, 1, -127],
        [43, -89, 12, 105],
    ] {
        let sym41 = build_symmetric_lut41(q);
        let lut81 = build_radix81_lut(q);
        let simd_grp = SimdLutGroup::from_group4(q);
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
            assert_eq!(i32::from(lookup_symmetric_lut41(&sym41, u)), exact);
            assert_eq!(i32::from(lut81[u]), exact);
            assert_eq!(simd_grp.lookup_byte_i32(packed), exact);
        }
    }
}

#[test]
fn packed_matrix_memory_compression_4x_int8_16x_fp32() -> Result<()> {
    for (rows, cols) in [(256, 256), (1024, 512), (1536, 576)] {
        let trits = deterministic_trits(rows * cols, 99);
        let scales = vec![0.125f32; rows];
        let packed = PackedTernaryMatrix::from_trits(&trits, rows, cols, scales, None)?;
        assert_eq!(packed.packed_weight_bytes() * 4, rows * cols);
        assert!((packed.compression_vs_int8() - 4.0).abs() < 1e-12);
        assert!((packed.compression_vs_fp32() - 16.0).abs() < 1e-12);
        for r in [0, rows / 2, rows - 1] {
            assert_eq!(packed.unpack_row(r), trits[r * cols..(r + 1) * cols]);
        }
    }
    Ok(())
}

#[test]
fn absmean_quantization_determinism_and_zero_row_handling() -> Result<()> {
    let w = vec![
        1.2f32, -1.1, 0.02, 0.95, // row 0: non-zero
        0.0, 0.0, 0.0, 0.0, // row 1: exact zero
    ];
    let (trits_a, scales_a) = quantize_absmean_ternary(&w, 2, 4)?;
    let (trits_b, scales_b) = quantize_absmean_ternary(&w, 2, 4)?;
    assert_eq!(trits_a, trits_b);
    assert_eq!(scales_a, scales_b);
    assert_eq!(&trits_a[0..4], &[1, -1, 0, 1]);
    assert_eq!(&trits_a[4..8], &[0, 0, 0, 0]);
    assert_eq!(scales_a[1], 0.0);

    let (g_trits, g_scales) = quantize_absmean_ternary_grouped(&w, 2, 4, 2)?;
    assert_eq!(g_scales.len(), 4);
    assert_eq!(&g_trits[4..8], &[0, 0, 0, 0]);
    Ok(())
}

#[test]
fn exact_i32_dot_product_parity_aligned_and_large_k() -> Result<()> {
    // K = 516 crosses 4 chunks of I16_ACCUM_GROUP_CHUNK (128 channels each) and exceeds i16::MAX
    // when all weights are +1 and all activations are +127 (516 * 127 = 65_532 > 32_767).
    let rows = 160usize;
    let cols = 516usize;
    let mut trits = deterministic_trits(rows * cols, 2026);
    // Force row 0 to all +1 and row 1 to all -1 to stress extremal i16 chunk widening
    trits[0..cols].fill(1);
    trits[cols..2 * cols].fill(-1);
    let mut qx = deterministic_i8_acts(cols, 777);
    qx.fill(127);

    let scales = vec![1.0f32; rows];
    let packed = PackedTernaryMatrix::from_trits(&trits, rows, cols, scales, None)?;
    let lut = ActivationLut::from_quantized_i8(qx.clone(), 1.0);
    let expected = reference_i32_matvec(&trits, rows, cols, &qx);
    assert_eq!(expected[0], 516 * 127);
    assert_eq!(expected[1], -516 * 127);

    for backend in [
        TmacKernelBackend::ScalarTiledLut,
        TmacKernelBackend::Avx2Lut,
        TmacKernelBackend::Avx512WideLut,
        TmacKernelBackend::NeonLut,
        TmacKernelBackend::Auto,
    ] {
        let mut got = vec![0i32; rows];
        ternary_lut_matvec_i32(&packed, &lut, backend, &mut got)?;
        assert_eq!(got, expected, "extremal K=516 mismatch on {backend:?}");
    }
    Ok(())
}

#[test]
fn exact_i32_dot_product_parity_odd_and_unaligned_shapes() -> Result<()> {
    let shapes = [
        (1usize, 1usize),
        (1, 3),
        (3, 5),
        (7, 17),
        (19, 67),
        (31, 127),
        (33, 131),
        (65, 259),
        (129, 389),
    ];
    for (idx, &(rows, cols)) in shapes.iter().enumerate() {
        let trits = deterministic_trits(rows * cols, 100 + idx as u64);
        let qx = deterministic_i8_acts(cols, 500 + idx as u64);
        let scales = vec![0.5f32; rows];
        let packed = PackedTernaryMatrix::from_trits(&trits, rows, cols, scales, None)?;
        let lut = ActivationLut::from_quantized_i8(qx.clone(), 0.25);
        let expected = reference_i32_matvec(&trits, rows, cols, &qx);

        for backend in [
            TmacKernelBackend::ScalarTiledLut,
            TmacKernelBackend::Avx2Lut,
            TmacKernelBackend::Avx512WideLut,
            TmacKernelBackend::NeonLut,
            TmacKernelBackend::Auto,
        ] {
            let mut got = vec![0i32; rows];
            ternary_lut_matvec_i32(&packed, &lut, backend, &mut got)?;
            assert_eq!(
                got, expected,
                "integer dot mismatch at shape ({rows}, {cols}) on {backend:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn f32_matvec_parity_across_tmac_unpack_and_fp32_reference() -> Result<()> {
    let rows = 96usize;
    let cols = 196usize;
    let raw_w: Vec<f32> = (0..rows * cols)
        .map(|i| (((i * 37 + 11) % 200) as f32 - 100.0) * 0.01)
        .collect();
    let bias: Vec<f32> = (0..rows).map(|r| (r as f32) * 0.005 - 0.2).collect();
    let x: Vec<f32> = (0..cols)
        .map(|c| (((c * 19 + 5) % 160) as f32 - 80.0) * 0.02)
        .collect();

    let packed = PackedTernaryMatrix::from_f32_absmean(&raw_w, rows, cols, Some(bias.clone()))?;
    let dense = packed.dequantize_f32();

    let mut y_fp32 = vec![0.0f32; rows];
    let mut y_unpack = vec![0.0f32; rows];
    let mut y_tmac_auto = vec![0.0f32; rows];
    let mut y_tmac_scalar = vec![0.0f32; rows];

    fp32_reference_matvec(&dense, Some(&bias), rows, cols, &x, &mut y_fp32)?;
    weight_unpack_deq_matvec_f32(&packed, &x, &mut y_unpack)?;
    ternary_lut_matvec_f32(&packed, &x, TmacKernelBackend::Auto, &mut y_tmac_auto)?;
    ternary_lut_matvec_f32(
        &packed,
        &x,
        TmacKernelBackend::ScalarTiledLut,
        &mut y_tmac_scalar,
    )?;

    assert_eq!(y_tmac_auto, y_tmac_scalar);
    assert_eq!(y_tmac_auto, y_unpack);
    for r in 0..rows {
        let diff = (y_tmac_auto[r] - y_fp32[r]).abs();
        assert!(diff < 1e-5, "row {r} diff {diff} >= 1e-5");
    }
    Ok(())
}

#[test]
fn tmac_linear_module_1d_2d_3d_tensor_shapes_and_dtypes() -> Result<()> {
    let dev = Device::Cpu;
    let w_data: Vec<f32> = (0..48 * 64)
        .map(|i| (((i * 29 + 3) % 101) as f32 - 50.0) * 0.03)
        .collect();
    let w = Tensor::from_vec(w_data, (48, 64), &dev)?;
    let b = Tensor::from_vec(vec![0.1f32; 48], 48, &dev)?;
    let layer = TmacLinear::from_tensor(&w, Some(&b), TmacKernelBackend::Auto)?;

    let x1 = Tensor::from_vec(vec![0.25f32; 64], 64, &dev)?;
    assert_eq!(layer.forward(&x1)?.dims(), &[48]);

    let x2 = Tensor::from_vec(vec![0.25f32; 5 * 64], (5, 64), &dev)?;
    assert_eq!(layer.forward(&x2)?.dims(), &[5, 48]);

    let x3 =
        Tensor::from_vec(vec![0.25f32; 2 * 3 * 64], (2, 3, 64), &dev)?.to_dtype(DType::BF16)?;
    let y3 = layer.forward(&x3)?;
    assert_eq!(y3.dtype(), DType::BF16);
    assert_eq!(y3.dims(), &[2, 3, 48]);
    Ok(())
}

#[test]
fn rejects_invalid_trits_and_mismatched_dimensions() {
    assert!(PackedTernaryMatrix::from_trits(&[], 0, 4, vec![], None).is_err());
    assert!(PackedTernaryMatrix::from_trits(&[0, 2, -1, 1], 1, 4, vec![1.0], None).is_err());
    assert!(PackedTernaryMatrix::from_trits(&[0, 1, -1, 1], 1, 4, vec![1.0, 2.0], None).is_err());

    let valid = PackedTernaryMatrix::from_trits(&[1, -1, 0, 1], 1, 4, vec![1.0], None).unwrap();
    let mut out = [0.0f32; 1];
    let bad = ternary_lut_matvec_f32(&valid, &[1.0, 2.0], TmacKernelBackend::Auto, &mut out);
    assert!(bad.is_err());
}

#[test]
fn multi_layer_transformer_prefill_and_kv_decode_equivalence() -> Result<()> {
    let profile = ModelScaleProfile {
        name: "tiny-verify".to_string(),
        canonical_params_b: 1.0,
        hidden_dim: 64,
        ffn_dim: 192,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 2,
        vocab_size: 128,
    };
    let model = TernaryTransformerModel::from_profile(profile, 4242, true)?;
    let prompt = [3usize, 17, 42, 9, 61, 12];

    let mut kv_fp32 = model.new_kv_cache(16);
    let mut kv_unp = model.new_kv_cache(16);
    let mut kv_tmac = model.new_kv_cache(16);

    let mut l_fp32 = model.prefill(
        &prompt,
        &mut kv_fp32,
        OperatorKind::Fp32Reference,
        TmacKernelBackend::Auto,
    )?;
    let mut l_unp = model.prefill(
        &prompt,
        &mut kv_unp,
        OperatorKind::WeightUnpackDeqReference,
        TmacKernelBackend::Auto,
    )?;
    let mut l_tmac = model.prefill(
        &prompt,
        &mut kv_tmac,
        OperatorKind::TmacLutOperator,
        TmacKernelBackend::Auto,
    )?;

    for step in 0..6usize {
        let tok = (step * 11 + 5) % 128;
        l_fp32 = model.forward_token(
            tok,
            &mut kv_fp32,
            OperatorKind::Fp32Reference,
            TmacKernelBackend::Auto,
        )?;
        l_unp = model.forward_token(
            tok,
            &mut kv_unp,
            OperatorKind::WeightUnpackDeqReference,
            TmacKernelBackend::Auto,
        )?;
        l_tmac = model.forward_token(
            tok,
            &mut kv_tmac,
            OperatorKind::TmacLutOperator,
            TmacKernelBackend::Auto,
        )?;
    }

    assert_eq!(kv_tmac.seq_len(), 12);
    assert_eq!(l_tmac, l_unp);
    let max_diff = l_tmac
        .iter()
        .zip(l_fp32.iter())
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_diff < 1e-5,
        "end-to-end max logit diff {max_diff} >= 1e-5"
    );
    Ok(())
}

#[test]
fn kv_cache_clear_and_deterministic_replay() -> Result<()> {
    let profile = ModelScaleProfile {
        name: "replay".to_string(),
        canonical_params_b: 1.0,
        hidden_dim: 64,
        ffn_dim: 128,
        num_heads: 4,
        num_kv_heads: 2,
        num_layers: 1,
        vocab_size: 64,
    };
    let model = TernaryTransformerModel::from_profile(profile, 999, false)?;
    let mut kv = model.new_kv_cache(4);
    let first = model.prefill(
        &[1, 2, 3, 4],
        &mut kv,
        OperatorKind::TmacLutOperator,
        TmacKernelBackend::Auto,
    )?;
    assert!(model
        .forward_token(
            5,
            &mut kv,
            OperatorKind::TmacLutOperator,
            TmacKernelBackend::Auto
        )
        .is_err());

    kv.clear();
    assert_eq!(kv.seq_len(), 0);
    let replay = model.prefill(
        &[1, 2, 3, 4],
        &mut kv,
        OperatorKind::TmacLutOperator,
        TmacKernelBackend::Auto,
    )?;
    assert_eq!(first, replay);
    Ok(())
}

#[test]
fn concurrent_multithreaded_tmac_inference_bit_identical() -> Result<()> {
    let dev = Device::Cpu;
    let w_vec: Vec<f32> = (0..64 * 128)
        .map(|i| (((i * 17 + 9) % 97) as f32 - 48.0) * 0.02)
        .collect();
    let w = Tensor::from_vec(w_vec, (64, 128), &dev)?;
    let layer = Arc::new(TmacLinear::from_tensor(&w, None, TmacKernelBackend::Auto)?);
    let x = Tensor::from_vec(
        (0..4 * 128)
            .map(|i| ((i % 31) as f32 - 15.0) * 0.05)
            .collect(),
        (4, 128),
        &dev,
    )?;
    let baseline = layer.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..4 {
            let l_clone = Arc::clone(&layer);
            let x_clone = x.clone();
            handles.push(scope.spawn(move || {
                l_clone
                    .forward(&x_clone)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            }));
        }
        for h in handles {
            assert_eq!(h.join().unwrap(), baseline);
        }
    });
    Ok(())
}

#[test]
fn end_to_end_acceptance_gate_rejects_microbenchmark_regressions() {
    let reference = OperatorScorecardRow {
        profile_name: "4B-scaled".to_string(),
        canonical_params_b: 4.0,
        operator: OperatorKind::Fp32Reference,
        active_kernel: "fp32-dense-cpu".to_string(),
        cold_ttft_ms: 4.0,
        warm_ttft_ms: 3.5,
        itl_p50_ms: 0.60,
        itl_p95_ms: 0.75,
        agent_20turn_wall_ms: 110.0,
        sustained_tokens_per_sec: 1650.0,
        effective_gops: 12.5,
        measured_joules_per_token: None,
        weight_memory_bytes: 10_000_000,
        peak_memory_bytes: 11_000_000,
        max_logit_abs_error_vs_fp32: 0.0,
        max_logit_abs_error_vs_unpack: 0.0,
        token_agreement_vs_fp32: 1.0,
        quality_regression: false,
    };

    let mut regressed_wall = reference.clone();
    regressed_wall.operator = OperatorKind::TmacLutOperator;
    regressed_wall.sustained_tokens_per_sec = 2200.0;
    // Candidate faster per decode step, yet slower end-to-end across 20 turns: must reject.
    regressed_wall.agent_20turn_wall_ms = 125.0;
    assert!(!check_end_to_end_acceptance_gate(&regressed_wall, &reference).accepted);

    let mut regressed_quality = reference.clone();
    regressed_quality.operator = OperatorKind::TmacLutOperator;
    regressed_quality.sustained_tokens_per_sec = 3200.0;
    regressed_quality.agent_20turn_wall_ms = 55.0;
    regressed_quality.quality_regression = true;
    assert!(!check_end_to_end_acceptance_gate(&regressed_quality, &reference).accepted);

    // When real hardware energy telemetry is available for both runs, energy regression must reject.
    let mut ref_with_energy = reference.clone();
    ref_with_energy.measured_joules_per_token = Some(0.0110);
    let mut regressed_energy = reference.clone();
    regressed_energy.operator = OperatorKind::TmacLutOperator;
    regressed_energy.sustained_tokens_per_sec = 3200.0;
    regressed_energy.agent_20turn_wall_ms = 55.0;
    regressed_energy.measured_joules_per_token = Some(0.0125);
    assert!(!check_end_to_end_acceptance_gate(&regressed_energy, &ref_with_energy).accepted);
}

#[test]
fn hardware_metadata_populates_required_provenance_fields() -> Result<()> {
    let meta = HardwareMetadata::detect();
    assert!(!meta.cpu_arch.is_empty());
    assert!(!meta.os.is_empty());
    assert!(!meta.cpu_model.is_empty());
    assert!(!meta.detected_isa.is_empty());
    assert!(meta.quantization_format.contains("W1.58A8"));
    assert!(meta.energy_telemetry.is_none());
    let json_txt = serde_json::to_string(&meta).unwrap();
    assert!(json_txt.contains(&meta.detected_isa));
    assert!(json_txt.contains("\"energy_telemetry\":null"));
    Ok(())
}

#[test]
fn canonical_pq2_ptq1_bitcos_parity_on_identical_ternary_weights() -> Result<()> {
    use half::f16;
    use hanzo_ml::quantized::bitcos::{pack_trits_to_pq2_0, pack_trits_to_ptq1_0, BitcosMatrix};
    use hanzo_ml::quantized::ggml_file::qtensor_from_ggml;
    use hanzo_ml::quantized::iq_quants::{BlockPQ2_0, BlockPTQ1_0, QK_PQ2_0};
    use hanzo_ml::quantized::k_quants::{BlockQ8_0, GgmlType, QK8_0};
    use hanzo_ml::quantized::{GgmlDType, QMatMul};

    let rows = 256usize;
    let cols = 512usize;
    let blocks_per_row = cols / QK_PQ2_0;

    let x: Vec<f32> = (0..cols)
        .map(|c| (((c * 31 + 7) % 200) as f32 - 100.0) * 0.015)
        .collect();
    let x_tensor = Tensor::from_slice(&x, (1, cols), &Device::Cpu)?;
    let mut x_q8 = vec![BlockQ8_0::zeros(); cols / QK8_0];
    BlockQ8_0::from_float(&x, &mut x_q8);

    for &(z_label, zero_threshold_pct) in &[("z=0.33", 33u64), ("z=0.65", 65u64)] {
        let mut s = 0x517cc1b727220a95u64 ^ zero_threshold_pct;
        let trits: Vec<i8> = (0..rows * cols)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bucket = (s >> 40) % 100;
                if bucket < zero_threshold_pct {
                    0
                } else if ((s >> 32) & 1) == 0 {
                    1
                } else {
                    -1
                }
            })
            .collect();

        // Identical per-row f16 scale across all formats (broadcast to 128-elem blocks for
        // PQ2_0 / PTQ1_0 / Bitcos so all four representations encode the exact same W matrix).
        let row_scales_f16: Vec<f16> = (0..rows)
            .map(|r| f16::from_f32(0.05 + ((r % 17) as f32) * 0.0075))
            .collect();
        let row_scales_f32: Vec<f32> = row_scales_f16.iter().map(|d| d.to_f32()).collect();
        let mut block_scales_f16 = Vec::with_capacity(rows * blocks_per_row);
        for &d in &row_scales_f16 {
            for _ in 0..blocks_per_row {
                block_scales_f16.push(d);
            }
        }

        // Exact FP32 ground truth: y_ref[r] = scale[r] * sum_c trits[r, c] * x[c]
        let mut y_exact = vec![0.0f32; rows];
        for r in 0..rows {
            let mut sum = 0.0f64;
            for c in 0..cols {
                sum += f64::from(trits[r * cols + c]) * f64::from(x[c]);
            }
            y_exact[r] = (sum * f64::from(row_scales_f32[r])) as f32;
        }

        // 1. T-MAC LUT
        let tmac = PackedTernaryMatrix::from_trits(&trits, rows, cols, row_scales_f32, None)?;
        let mut y_tmac = vec![0.0f32; rows];
        ternary_lut_matvec_f32(&tmac, &x, TmacKernelBackend::Auto, &mut y_tmac)?;

        // 2. Canonical PQ2_0 (1T vec_dot + QMatMul)
        let pq2_blocks = pack_trits_to_pq2_0(&trits, &block_scales_f16)?;
        let mut y_pq2_1t = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &pq2_blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
            y_pq2_1t[r] = BlockPQ2_0::vec_dot(cols, row, &x_q8);
        }
        let pq2_raw = unsafe {
            std::slice::from_raw_parts(
                pq2_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(pq2_blocks.as_slice()),
            )
        };
        let pq2_qmm = QMatMul::from_qtensor(qtensor_from_ggml(
            GgmlDType::PQ2_0,
            pq2_raw,
            vec![rows, cols],
            &Device::Cpu,
        )?)?;
        let y_pq2_qmm = pq2_qmm
            .forward(&x_tensor)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // 3. Canonical PTQ1_0 (1T vec_dot + QMatMul)
        let ptq1_blocks = pack_trits_to_ptq1_0(&trits, &block_scales_f16)?;
        let mut y_ptq1_1t = vec![0.0f32; rows];
        for r in 0..rows {
            let row = &ptq1_blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
            y_ptq1_1t[r] = BlockPTQ1_0::vec_dot(cols, row, &x_q8);
        }
        let ptq1_raw = unsafe {
            std::slice::from_raw_parts(
                ptq1_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(ptq1_blocks.as_slice()),
            )
        };
        let ptq1_qmm = QMatMul::from_qtensor(qtensor_from_ggml(
            GgmlDType::PTQ1_0,
            ptq1_raw,
            vec![rows, cols],
            &Device::Cpu,
        )?)?;
        let y_ptq1_qmm = ptq1_qmm
            .forward(&x_tensor)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // 4. Canonical BitcosMatrix
        let bitcos = BitcosMatrix::encode_from_pq2_0(&pq2_blocks, rows, cols)?;
        let mut y_bitcos_q8 = vec![0.0f32; rows];
        bitcos.matvec_q8_0(&x, &mut y_bitcos_q8)?;
        let mut y_bitcos_f32 = vec![0.0f32; rows];
        bitcos.matvec_f32(&x, &mut y_bitcos_f32)?;

        let cosine = |a: &[f32], b: &[f32]| -> f64 {
            let mut dot = 0.0f64;
            let mut na = 0.0f64;
            let mut nb = 0.0f64;
            for (&u, &v) in a.iter().zip(b.iter()) {
                dot += f64::from(u) * f64::from(v);
                na += f64::from(u) * f64::from(u);
                nb += f64::from(v) * f64::from(v);
            }
            dot / (na.sqrt() * nb.sqrt())
        };
        let max_abs = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b.iter())
                .fold(0.0f32, |m, (u, v)| m.max((u - v).abs()))
        };

        assert!(
            cosine(&y_tmac, &y_exact) > 0.9995,
            "{z_label}: TmacLut cosine vs exact f32 too low: {}",
            cosine(&y_tmac, &y_exact)
        );
        assert!(
            cosine(&y_pq2_1t, &y_exact) > 0.9995,
            "{z_label}: PQ2_0 cosine vs exact f32 too low"
        );
        assert!(
            cosine(&y_ptq1_1t, &y_exact) > 0.9995,
            "{z_label}: PTQ1_0 cosine vs exact f32 too low"
        );
        assert!(
            cosine(&y_bitcos_q8, &y_exact) > 0.9995,
            "{z_label}: Bitcos q8 cosine vs exact f32 too low"
        );
        // PQ2_0, PTQ1_0, and Bitcos share identical Q8_0 activation quantization & f16 scales.
        assert!(max_abs(&y_pq2_1t, &y_ptq1_1t) < 1e-4);
        assert!(max_abs(&y_pq2_1t, &y_pq2_qmm) < 1e-4);
        assert!(max_abs(&y_ptq1_1t, &y_ptq1_qmm) < 1e-4);
        assert!(max_abs(&y_pq2_1t, &y_bitcos_q8) < 1e-4);
        assert!(max_abs(&y_bitcos_f32, &y_exact) < 1e-4);

        // Byte-size & bpw accounting sanity checks across all 4 formats.
        let elems = (rows * cols) as f64;
        let tmac_bpw = ((tmac.packed_weight_bytes() + rows * 4) as f64 * 8.0) / elems;
        let pq2_bpw = (pq2_raw.len() as f64 * 8.0) / elems;
        let ptq1_bpw = (ptq1_raw.len() as f64 * 8.0) / elems;
        let bitcos_bpw = bitcos.effective_bpw();
        assert!((pq2_bpw - 2.125).abs() < 1e-9);
        assert!((ptq1_bpw - 1.75).abs() < 1e-9);
        assert!(tmac_bpw < pq2_bpw);
        if zero_threshold_pct == 65 {
            assert!(
                bitcos_bpw < ptq1_bpw,
                "sparse Bitcos bpw {bitcos_bpw} should beat PTQ1_0"
            );
        }
    }
    Ok(())
}

#[test]
fn parallel_tmac_matvec_bit_identical_to_single_thread() -> Result<()> {
    let shapes = [(64usize, 128usize), (129, 389), (256, 512), (513, 516)];
    for (idx, &(rows, cols)) in shapes.iter().enumerate() {
        let trits = deterministic_trits(rows * cols, 0xA11CE + idx as u64);
        let scales: Vec<f32> = (0..rows).map(|r| 0.05 + ((r % 13) as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..rows).map(|r| ((r % 19) as f32 - 9.0) * 0.02).collect();
        let x: Vec<f32> = (0..cols)
            .map(|c| (((c * 23 + 7) % 180) as f32 - 90.0) * 0.015)
            .collect();

        for with_bias in [false, true] {
            let packed = PackedTernaryMatrix::from_trits(
                &trits,
                rows,
                cols,
                scales.clone(),
                if with_bias { Some(bias.clone()) } else { None },
            )?;
            let mut y_1t = vec![0.0f32; rows];
            let mut y_mt = vec![0.0f32; rows];
            ternary_lut_matvec_f32(&packed, &x, TmacKernelBackend::Auto, &mut y_1t)?;
            ternary_lut_matvec_f32_parallel(&packed, &x, TmacKernelBackend::Auto, &mut y_mt)?;
            assert_eq!(
                y_1t, y_mt,
                "parallel vs 1T mismatch at shape ({rows}, {cols}) bias={with_bias}"
            );
        }
    }
    Ok(())
}

#[test]
fn full_20turn_coding_agent_scorecard_across_4b_9b_27b_matrix() -> Result<()> {
    let profiles = ModelScaleProfile::ci_matrix();
    let report = run_tmac_scorecard(&profiles, TmacKernelBackend::Auto)?;
    assert_eq!(report.turns, 20);
    assert_eq!(report.rows.len(), 9);

    println!(
        "\n[T-MAC Scorecard] host={} isa={} fmt={} energy={:?}",
        report.metadata.cpu_model,
        report.metadata.detected_isa,
        report.metadata.quantization_format,
        report.metadata.energy_telemetry,
    );
    for i in (0..report.rows.len()).step_by(3) {
        let chunk = &report.rows[i..i + 3];
        let fp32_row = &chunk[0];
        let unpack_row = &chunk[1];
        let tmac_row = &chunk[2];
        for r in chunk {
            let energy_str = match r.measured_joules_per_token {
                Some(j) => format!("{j:.5} J/tok"),
                None => "null".to_string(),
            };
            println!(
                "  {} | {:<24} | {:<31} | ttft={:.2}/{:.2}ms p50/95={:.2}/{:.2}ms",
                r.profile_name,
                r.operator.as_str(),
                r.active_kernel,
                r.cold_ttft_ms,
                r.warm_ttft_ms,
                r.itl_p50_ms,
                r.itl_p95_ms,
            );
            println!(
                "    wall20={:.2}ms tok/s={:.1} gops={:.2} energy={} mem={:.2}MB err={:.2e} agree={:.3}",
                r.agent_20turn_wall_ms,
                r.sustained_tokens_per_sec,
                r.effective_gops,
                energy_str,
                (r.peak_memory_bytes as f64) / (1024.0 * 1024.0),
                r.max_logit_abs_error_vs_fp32,
                r.token_agreement_vs_fp32,
            );
            assert!(r.effective_gops > 0.0);
            assert!(r.measured_joules_per_token.is_none());
        }
        assert!(tmac_row.max_logit_abs_error_vs_fp32 < 1e-5);
        assert!(tmac_row.max_logit_abs_error_vs_unpack < 1e-6);
        assert!((tmac_row.token_agreement_vs_fp32 - 1.0).abs() < 1e-12);
        assert!(!tmac_row.quality_regression);
        assert!(tmac_row.weight_memory_bytes * 3 < fp32_row.weight_memory_bytes);
        assert!(tmac_row.peak_memory_bytes < fp32_row.peak_memory_bytes);

        let gate_vs_unpack = check_end_to_end_acceptance_gate(tmac_row, unpack_row);
        assert!(
            gate_vs_unpack.accepted,
            "{} failed gate vs WeightUnpackDeqReference: {}",
            tmac_row.profile_name, gate_vs_unpack.reason
        );
    }
    Ok(())
}
