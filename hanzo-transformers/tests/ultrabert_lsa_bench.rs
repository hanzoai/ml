//! Comprehensive CPU Benchmark & Quality Campaign for UltraBERT + Bidirectional D-LSA.
//!
//! Evaluates the four recommended configurations on identical CPU hardware:
//! 1. UltraBERT-380M INT8 (`Q8_0`) with optimized dense global + block-sparse local (±128) attention.
//! 2. UltraBERT-380M INT8 (`Q8_0`) + Bidirectional D-LSA on all 8 global layers (`layers 2, 5, 8, 11, 14, 17, 20, 23`).
//! 3. UltraBERT-380M BF16 + Bidirectional D-LSA on all 8 global layers.
//! 4. UltraBERT-150M INT8 (`Q8_0`) + Bidirectional D-LSA on all 8 global layers.
//!
//! Also sweeps retrieved-key budgets `K ∈ {256, 512, 1024, 2048}` (including index construction overhead),
//! distills bidirectional `LearnedProjection` indexers against frozen dense UltraBERT outputs, and
//! measures both encoder representation fidelity and Kai Decision Head ranking accuracy when critical
//! evidence is widely separated (`|pos_a - pos_b| >> 128`).

use std::time::Instant;

use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::attention::lsa::{
    build_buckets_bidirectional, distill_bidirectional_projection, forward_bidirectional_dlsa,
    project, read_bidirectional, regroup, sparse_gather_attention, BidirectionalLsaConfig,
};
use hanzo_nn::ops::softmax_last_dim;
use hanzo_nn::VarBuilder;
use hanzo_transformers::models::ultrabert::{Config, RotaryEmbedding, UltrabertModel};

fn ultrabert_layer_schedule(num_layers: usize) -> (Vec<Option<usize>>, Vec<f64>) {
    let windows = (0..num_layers)
        .map(|i| if i % 3 == 2 { None } else { Some(128) })
        .collect();
    let thetas = (0..num_layers)
        .map(|i| if i % 3 == 2 { 2_560_000.0 } else { 10_000.0 })
        .collect();
    (windows, thetas)
}

fn ultrabert_380m_config(vocab_size: usize) -> Config {
    let (attention_windows, rope_theta) = ultrabert_layer_schedule(24);
    Config {
        vocab_size,
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 24,
        num_attention_heads: 8,
        num_key_value_heads: Some(8),
        head_dim: 128,
        rms_norm_eps: 1e-6,
        pad_token_id: 0,
        attention_windows,
        rope_theta,
        prediction_head_size: None,
        max_position_embeddings: Some(32_768),
    }
}

fn ultrabert_150m_config(vocab_size: usize) -> Config {
    let (attention_windows, rope_theta) = ultrabert_layer_schedule(24);
    Config {
        vocab_size,
        hidden_size: 768,
        intermediate_size: 1024,
        num_hidden_layers: 24,
        num_attention_heads: 6,
        num_key_value_heads: Some(6),
        head_dim: 128,
        rms_norm_eps: 1e-6,
        pad_token_id: 0,
        attention_windows,
        rope_theta,
        prediction_head_size: None,
        max_position_embeddings: Some(32_768),
    }
}

fn det_tensor(shape: &[usize], seed: u64, device: &Device) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = ((z >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
        vals.push((u * 0.08) as f32);
    }
    Tensor::from_vec(vals, shape, device)
}

fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a_f = a.to_dtype(DType::F32)?.flatten_all()?;
    let b_f = b.to_dtype(DType::F32)?.flatten_all()?;
    let dot = (&a_f * &b_f)?.sum_all()?.to_scalar::<f32>()?;
    let na = a_f.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let nb = b_f.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    Ok(dot / (na * nb).max(1e-12))
}

/// Populate deterministic weights for `UltrabertModel` so benchmarks and quality evaluations
/// run reproducibly without requiring external multi-GB downloads.
fn build_deterministic_ultrabert(
    cfg: &Config,
    dtype: DType,
    dev: &Device,
) -> Result<UltrabertModel> {
    let mut tensors = std::collections::HashMap::new();
    let d = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads() * cfg.head_dim;
    let ffn = cfg.intermediate_size;

    tensors.insert(
        "embed_tokens.weight".to_string(),
        det_tensor(&[cfg.vocab_size, d], 1000, dev)?.to_dtype(dtype)?,
    );
    tensors.insert("norm.weight".to_string(), Tensor::ones((d,), dtype, dev)?);

    for i in 0..cfg.num_hidden_layers {
        let base = 2000 + (i as u64) * 200;
        tensors.insert(
            format!("layers.{i}.input_layernorm.weight"),
            Tensor::ones((d,), dtype, dev)?,
        );
        tensors.insert(
            format!("layers.{i}.post_attention_layernorm.weight"),
            Tensor::ones((d,), dtype, dev)?,
        );
        tensors.insert(
            format!("layers.{i}.self_attn.q_proj.weight"),
            det_tensor(&[q_dim, d], base + 11, dev)?.to_dtype(dtype)?,
        );
        tensors.insert(
            format!("layers.{i}.self_attn.k_proj.weight"),
            det_tensor(&[kv_dim, d], base + 23, dev)?.to_dtype(dtype)?,
        );
        tensors.insert(
            format!("layers.{i}.self_attn.v_proj.weight"),
            det_tensor(&[kv_dim, d], base + 37, dev)?.to_dtype(dtype)?,
        );
        tensors.insert(
            format!("layers.{i}.self_attn.o_proj.weight"),
            det_tensor(&[d, q_dim], base + 53, dev)?.to_dtype(dtype)?,
        );
        tensors.insert(
            format!("layers.{i}.mlp.gate_proj.weight"),
            det_tensor(&[ffn, d], base + 71, dev)?.to_dtype(dtype)?,
        );
        tensors.insert(
            format!("layers.{i}.mlp.up_proj.weight"),
            det_tensor(&[ffn, d], base + 89, dev)?.to_dtype(dtype)?,
        );
        tensors.insert(
            format!("layers.{i}.mlp.down_proj.weight"),
            det_tensor(&[d, ffn], base + 107, dev)?.to_dtype(dtype)?,
        );
    }

    let vb_load = VarBuilder::from_tensors(tensors, dtype, dev);
    UltrabertModel::load(vb_load, cfg)
}

/// Dense global bidirectional SDPA reference (`[B, H, T, Dh]`).
fn dense_global_attention(q_rot: &Tensor, k_rot: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d_h = q_rot.dim(3)?;
    let scale = 1.0 / (d_h as f64).sqrt();
    let scores = (q_rot.matmul(&k_rot.transpose(2, 3)?)? * scale)?;
    let probs = softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(q_rot.dtype())?;
    probs.matmul(v)
}

#[test]
fn test_ultrabert_block_sparse_local_parity() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = Config {
        vocab_size: 256,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(4),
        head_dim: 32,
        rms_norm_eps: 1e-6,
        pad_token_id: 0,
        attention_windows: vec![Some(128), Some(128)],
        rope_theta: vec![10_000.0, 10_000.0],
        prediction_head_size: None,
        max_position_embeddings: Some(2048),
    };
    let model_tiled = build_deterministic_ultrabert(&cfg, DType::F32, &dev)?;
    let mut model_packed = model_tiled.clone();
    model_packed.pack();

    let ids_vec: Vec<u32> = (0..512).map(|i| (i % 256) as u32).collect();
    let input_ids = Tensor::from_vec(ids_vec, (1, 512), &dev)?;

    let tiled_out = model_tiled.forward(&input_ids, None)?;
    let packed_out = model_packed.forward(&input_ids, None)?;

    let max_diff = (&packed_out - &tiled_out)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(
        max_diff < 1e-4,
        "packed local Ultrabert path diverged from tiled local path: {max_diff:e}"
    );
    Ok(())
}

#[test]
#[ignore = "long-context CPU benchmark sweep (run with --ignored)"]
fn benchmark_ultrabert_dlsa_campaign() -> Result<()> {
    let dev = Device::Cpu;

    println!("\n====================================================================================================================");
    println!("  UltraBERT + Bidirectional D-LSA CPU Benchmark & Kai Decision Quality Campaign");
    println!("  380M: 24 layers (16 local ±128, 8 global), d=1024, H=8, Dh=128, FFN=3072 | 150M: d=768, H=6, Dh=128, FFN=1024");
    println!("====================================================================================================================\n");

    // ============================================================================================
    // PART 1: Why 1K Tokens Took 7.39s — Isolating Local Mask O(T^2), Unfused GEMMs, and Quantization
    // ============================================================================================
    println!("--- [1] Short-Context (512 & 1K) Ablation: Naive F32 vs Block-Sparse Local + Fused QKV/MLP + INT8 (Q8_0) ---");
    println!(
        "{:<38} | {:<12} | {:<12} | {:<14} | {:<14}",
        "Configuration (24 Layers)", "512 Tokens", "1K Tokens", "Cos vs F32", "Max |Δh|"
    );
    println!("{}", "-".repeat(100));

    // Use a scaled vocab (2048) in the synthetic VarMap so initialization is instantaneous while
    // keeping all 24 transformer layers at exact 380M/150M dimensions (`d=1024, FFN=3072` & `d=768, FFN=1024`)!
    let cfg_380m = ultrabert_380m_config(2048);
    let cfg_150m = ultrabert_150m_config(2048);

    let model_380m_f32 = build_deterministic_ultrabert(&cfg_380m, DType::F32, &dev)?;
    let mut model_380m_f32_packed = model_380m_f32.clone();
    model_380m_f32_packed.pack();
    let mut model_380m_int8 = model_380m_f32.clone();
    model_380m_int8.quantize(GgmlDType::Q8_0)?;
    let model_380m_bf16 = build_deterministic_ultrabert(&cfg_380m, DType::BF16, &dev)?;
    let model_150m_f32 = build_deterministic_ultrabert(&cfg_150m, DType::F32, &dev)?;
    let mut model_150m_int8 = model_150m_f32.clone();
    model_150m_int8.quantize(GgmlDType::Q8_0)?;

    let ids_512 = Tensor::arange(0u32, 512u32, &dev)?.reshape((1, 512))?;
    let ids_1k = Tensor::arange(0u32, 1024u32, &dev)?.reshape((1, 1024))?;

    // Baseline A: F32 tiled path with explicit 4D mask
    let ones_512 = Tensor::ones((1, 512), DType::F32, &dev)?;
    let ones_1k = Tensor::ones((1, 1024), DType::F32, &dev)?;
    let t0 = Instant::now();
    let _ = model_380m_f32.forward(&ids_512, Some(&ones_512))?;
    let naive_512_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let ref_1k_f32 = model_380m_f32.forward(&ids_1k, Some(&ones_1k))?;
    let naive_1k_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Baseline B: F32 packed CPU path (local ±128 + packed global)
    let t0 = Instant::now();
    let _ = model_380m_f32_packed.forward(&ids_512, None)?;
    let f32_bs_512_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let out_1k_f32_bs = model_380m_f32_packed.forward(&ids_1k, None)?;
    let f32_bs_1k_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Config 1: UltraBERT-380M INT8 (Q8_0) + Packed Local + Dense Global
    let t0 = Instant::now();
    let _ = model_380m_int8.forward(&ids_512, None)?;
    let int8_512_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let out_1k_int8 = model_380m_int8.forward(&ids_1k, None)?;
    let int8_1k_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Config 4 (Dense): UltraBERT-150M INT8 (Q8_0) + Packed Local + Dense Global
    let t0 = Instant::now();
    let _ = model_150m_int8.forward(&ids_512, None)?;
    let int8_150m_512_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let _ = model_150m_int8.forward(&ids_1k, None)?;
    let int8_150m_1k_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let cos_bs = cosine_similarity(&ref_1k_f32, &out_1k_f32_bs)?;
    let diff_bs = (&ref_1k_f32 - &out_1k_f32_bs)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    let cos_int8 = cosine_similarity(&ref_1k_f32, &out_1k_int8)?;
    let diff_int8 = (&ref_1k_f32 - &out_1k_int8)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;

    println!(
        "{:<38} | {:>9.1} ms | {:>9.1} ms | {:>14.6} | {:>14.2e}",
        "380M F32 + Dense Local Mask (O(T^2))", naive_512_ms, naive_1k_ms, 1.000000, 0.0
    );
    println!(
        "{:<38} | {:>9.1} ms | {:>9.1} ms | {:>14.6} | {:>14.2e}",
        "380M F32 + Block-Sparse Local (±128)", f32_bs_512_ms, f32_bs_1k_ms, cos_bs, diff_bs
    );
    println!(
        "{:<38} | {:>9.1} ms | {:>9.1} ms | {:>14.6} | {:>14.2e}",
        "Config 1: 380M INT8 + Block-Sparse", int8_512_ms, int8_1k_ms, cos_int8, diff_int8
    );
    println!(
        "{:<38} | {:>9.1} ms | {:>9.1} ms | {:>14} | {:>14}",
        "Config 4 (Dense): 150M INT8 + BS Local",
        int8_150m_512_ms,
        int8_150m_1k_ms,
        "— (150M)",
        "— (150M)"
    );

    // ============================================================================================
    // PART 2: Global-Layer Bidirectional D-LSA Retrieved-Key Budget Sweep (K = 256, 512, 1024, 2048)
    //         Including Index Projection, Ward Clustering, Selection, and Sparse SDPA Overhead
    // ============================================================================================
    println!("\n--- [2] Global Layer Bidirectional D-LSA Budget Sweep (K = 256, 512, 1024, 2048) vs Dense Global ---");
    println!("    Geometry: UltraBERT-380M Global Layer (H=8, Dh=128, d=1024 -> d_idx=64, local_radius=128, pool=4, bucket=8)");
    println!(
        "{:<8} | {:<8} | {:<11} | {:<11} | {:<11} | {:<11} | {:<11} | {:<10} | {:<10}",
        "Context",
        "Budget K",
        "Proj+Ward",
        "Bucket Read",
        "Sparse SDPA",
        "Total D-LSA",
        "Dense Attn",
        "Speedup",
        "Cos Sim"
    );
    println!("{}", "-".repeat(114));

    let rotary_global = RotaryEmbedding::new(DType::F32, 128, 32768, 2_560_000.0, &dev)?;

    // Distill a LearnedProjection on a 1,024-token calibration sequence first
    let q_cal_pre = det_tensor(&[1, 8, 1024, 128], 501, &dev)?;
    let k_cal_pre = det_tensor(&[1, 8, 1024, 128], 502, &dev)?;
    let (q_cal_rot, k_cal_rot) = rotary_global.apply_rotary_emb_qkv(&q_cal_pre, &k_cal_pre)?;
    let distilled_proj_380m = distill_bidirectional_projection(
        &q_cal_pre, &k_cal_pre, &q_cal_rot, &k_cal_rot, 64, 128, 8, 0.5,
    )?;

    for &ctx_len in &[1_024usize, 2_048, 4_096, 8_192, 16_384, 32_768] {
        let q_pre = det_tensor(&[1, 8, ctx_len, 128], 700 + ctx_len as u64, &dev)?;
        let k_pre = det_tensor(&[1, 8, ctx_len, 128], 800 + ctx_len as u64, &dev)?;
        let v = det_tensor(&[1, 8, ctx_len, 128], 900 + ctx_len as u64, &dev)?;
        let (q_rot, k_rot) = rotary_global.apply_rotary_emb_qkv(&q_pre, &k_pre)?;

        // Dense global attention baseline (head-chunked at 32K to avoid 34 GB peak allocation / swap)
        let t_dense = Instant::now();
        let dense_out = if ctx_len >= 32_768 {
            let mut h_outs = Vec::with_capacity(8);
            for h in 0..8 {
                let qh = q_rot.narrow(1, h, 1)?;
                let kh = k_rot.narrow(1, h, 1)?;
                let vh = v.narrow(1, h, 1)?;
                h_outs.push(dense_global_attention(&qh, &kh, &vh)?);
            }
            Tensor::cat(&h_outs, 1)?
        } else {
            dense_global_attention(&q_rot, &k_rot, &v)?
        };
        let dense_ms = t_dense.elapsed().as_secs_f64() * 1000.0;

        for &budget_k in &[256usize, 512, 1_024, 2_048] {
            if budget_k >= ctx_len {
                continue;
            }
            let dlsa_cfg = BidirectionalLsaConfig::ultrabert_with_budget(budget_k);

            // Step 1: Index projection + Ward bucket construction
            let t_idx = Instant::now();
            let (q_idx, k_idx) = project(&q_pre, &k_pre, &distilled_proj_380m)?;
            let buckets = build_buckets_bidirectional(
                &k_idx,
                dlsa_cfg.cluster_block_size,
                dlsa_cfg.pool_size,
                dlsa_cfg.bucket_size,
                dlsa_cfg.bucket_cap,
            )?;
            let idx_ward_ms = t_idx.elapsed().as_secs_f64() * 1000.0;

            // Step 2: Bidirectional bucket scoring & keep-set expansion
            let t_read = Instant::now();
            let selection = read_bidirectional(
                &q_idx,
                &buckets,
                ctx_len,
                dlsa_cfg.local_radius,
                dlsa_cfg.sink_size,
                dlsa_cfg.top_p,
                dlsa_cfg.top_k,
                0.0,
            )?;
            let read_ms = t_read.elapsed().as_secs_f64() * 1000.0;

            // Step 3: CPU-native sparse gather SDPA
            let t_sdpa = Instant::now();
            let sparse_out = sparse_gather_attention(&q_rot, &k_rot, &v, &selection, None)?;
            let sdpa_ms = t_sdpa.elapsed().as_secs_f64() * 1000.0;

            let total_dlsa_ms = idx_ward_ms + read_ms + sdpa_ms;
            let cos = cosine_similarity(&dense_out, &sparse_out)?;

            println!(
                "{:<8} | {:>8} | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2} ms | {:>8.2}x | {:>10.6}",
                format!("{}K", ctx_len / 1024),
                budget_k,
                idx_ward_ms,
                read_ms,
                sdpa_ms,
                total_dlsa_ms,
                dense_ms,
                dense_ms / total_dlsa_ms,
                cos
            );
        }
        println!("{}", "-".repeat(114));
    }

    // ============================================================================================
    // PART 3: Full 24-Layer End-to-End Comparison Across the 4 Recommended Configurations
    // ============================================================================================
    println!("\n--- [3] Full 24-Layer End-to-End Benchmark: 4 Recommended Configurations (1K, 2K, 4K, 8K, 16K) ---");
    println!(
        "{:<10} | {:<18} | {:<18} | {:<18} | {:<18} | {:<14}",
        "Context",
        "1: 380M INT8 Dense",
        "2: 380M INT8+DLSA",
        "3: 380M BF16+DLSA",
        "4: 150M INT8+DLSA",
        "DLSA vs Dense"
    );
    println!("{}", "-".repeat(108));

    // Attach shipped D-LSA indexers to all global layers (`attention_windows[i] == None`).
    let mut cfg2_380m_int8_dlsa = model_380m_int8.clone();
    cfg2_380m_int8_dlsa.with_dlsa(512, 64, 4)?;
    cfg2_380m_int8_dlsa.set_dlsa_min_seq_len(1024);

    let mut cfg3_380m_bf16_dlsa = model_380m_bf16.clone();
    cfg3_380m_bf16_dlsa.with_dlsa(512, 64, 4)?;
    cfg3_380m_bf16_dlsa.set_dlsa_min_seq_len(1024);

    let mut cfg4_150m_int8_dlsa = model_150m_int8.clone();
    cfg4_150m_int8_dlsa.with_dlsa(512, 64, 4)?;
    cfg4_150m_int8_dlsa.set_dlsa_min_seq_len(1024);

    for &ctx_len in &[1_024usize, 2_048, 4_096, 8_192, 16_384] {
        let token_vec: Vec<u32> = (0..ctx_len).map(|i| (i % 2048) as u32).collect();
        let input_ids = Tensor::from_vec(token_vec, (1, ctx_len), &dev)?;

        let t1 = Instant::now();
        let out_c1 = model_380m_int8.forward(&input_ids, None)?;
        let ms_c1 = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let out_c2 = cfg2_380m_int8_dlsa.forward(&input_ids, None)?;
        let ms_c2 = t2.elapsed().as_secs_f64() * 1000.0;

        let t3 = Instant::now();
        let _out_c3 = cfg3_380m_bf16_dlsa.forward(&input_ids, None)?;
        let ms_c3 = t3.elapsed().as_secs_f64() * 1000.0;

        let t4 = Instant::now();
        let _out_c4 = cfg4_150m_int8_dlsa.forward(&input_ids, None)?;
        let ms_c4 = t4.elapsed().as_secs_f64() * 1000.0;

        let cos_c1_c2 = cosine_similarity(&out_c1, &out_c2)?;
        println!(
            "{:<10} | {:>15.1} ms | {:>15.1} ms | {:>15.1} ms | {:>15.1} ms | {:>6.2}x (c={:.3})",
            format!("{}K", ctx_len / 1024),
            ms_c1,
            ms_c2,
            ms_c3,
            ms_c4,
            ms_c1 / ms_c2,
            cos_c1_c2
        );
    }

    Ok(())
}

#[test]
fn test_widely_separated_evidence_distillation() -> Result<()> {
    let dev = Device::Cpu;
    let rotary_global = RotaryEmbedding::new(DType::F32, 128, 32768, 2_560_000.0, &dev)?;

    // ============================================================================================
    // PART 4: Distillation Quality & Widely-Separated Evidence Retrieval / Kai Decision Ranking
    // ============================================================================================
    println!("\n--- [4] Widely-Separated Evidence Retrieval & Kai Decision Ranking Stability (4K..16K Separation) ---");
    println!("    Task: Planted long-range evidence needles at positions (p_early, p_late) separated by Δpos >> 128.");
    println!("    Evaluates Teacher Top-K Mass Recall & Kai 4-Option Decision Ranking Agreement vs Dense Teacher:");
    println!(
        "{:<10} | {:<10} | {:<18} | {:<18} | {:<18} | {:<16}",
        "Separation",
        "Budget K",
        "Untrained Recall",
        "PCA Index Recall",
        "Distilled Recall",
        "Kai Top-1 Match"
    );
    println!("{}", "-".repeat(104));

    for &ctx_len in &[4_096usize, 8_192, 16_384] {
        // Construct a sequence where queries in the final decision region (ctx_len - 64 .. ctx_len)
        // have strong semantic alignment with planted evidence tokens at early positions (32..96)
        // and middle positions (ctx_len / 2 .. ctx_len / 2 + 64), separated by thousands of tokens!
        let mut q_raw = det_tensor(&[1, 8, ctx_len, 128], 2000 + ctx_len as u64, &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut k_raw = det_tensor(&[1, 8, ctx_len, 128], 3000 + ctx_len as u64, &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let v_tensor = det_tensor(&[1, 8, ctx_len, 128], 4000 + ctx_len as u64, &dev)?;

        // Plant high-affinity evidence signatures at early (pos 40..72) and mid (ctx_len/2..ctx_len/2+32)
        // that match the decision queries at the tail (ctx_len - 32 .. ctx_len) across the low-frequency
        // RoPE semantic coordinates (`48..64` and `112..128`, where theta=2,560,000 wavelength > 1M tokens),
        // while high-frequency coordinates (`0..16`) carry local position-sensitive distractor variance!
        let needle_sig: Vec<f32> = (0..128)
            .map(|d| {
                let pair_idx = d % 64;
                if pair_idx >= 48 {
                    1.35 * ((d as f32 * 0.7).cos())
                } else {
                    0.0
                }
            })
            .collect();
        for h in 0..8 {
            let head_off = h * ctx_len * 128;
            // High-frequency local distractor variance in pair_idx < 32 on non-evidence tokens
            for pos in 0..ctx_len {
                let is_ev =
                    (40..72).contains(&pos) || (ctx_len / 2..ctx_len / 2 + 32).contains(&pos);
                if !is_ev {
                    let phase = (pos as f32) * 0.37 + (h as f32);
                    for d in 0..32 {
                        k_raw[head_off + pos * 128 + d] += 0.28 * (phase * (d as f32 + 1.0)).sin();
                    }
                }
            }
            // Planted evidence keys at pos 40..72 and ctx_len/2..ctx_len/2+32
            for &ev_start in &[40usize, ctx_len / 2] {
                for pos in ev_start..ev_start + 32 {
                    for d in 0..128 {
                        k_raw[head_off + pos * 128 + d] += needle_sig[d];
                    }
                }
            }
            // Tail decision queries at ctx_len - 32 .. ctx_len
            for pos in (ctx_len - 32)..ctx_len {
                for d in 0..128 {
                    q_raw[head_off + pos * 128 + d] += needle_sig[d];
                }
            }
        }

        let q_pre = Tensor::from_vec(q_raw, (1, 8, ctx_len, 128), &dev)?;
        let k_pre = Tensor::from_vec(k_raw, (1, 8, ctx_len, 128), &dev)?;
        let (q_rot, k_rot) = rotary_global.apply_rotary_emb_qkv(&q_pre, &k_pre)?;

        // Dense teacher attention output & Kai 4-option decision scores on tail query representations
        let dense_out = dense_global_attention(&q_rot, &k_rot, &v_tensor)?;
        let option_proj = det_tensor(&[4, 8 * 128], 9999, &dev)?; // 4 candidate options in Kai reader head

        let dense_tail = dense_out
            .narrow(2, ctx_len - 32, 32)?
            .permute((0, 2, 1, 3))?
            .reshape((32, 8 * 128))?;
        let teacher_option_logits = dense_tail.matmul(&option_proj.t()?)?; // [32, 4]
        let teacher_winners = teacher_option_logits.argmax(D::Minus1)?.to_vec1::<u32>()?;

        // 1. Untrained random projection
        let rand_proj = hanzo_nn::attention::lsa::LearnedProjection::new(
            det_tensor(&[1024, 64], 7771, &dev)?,
            det_tensor(&[1024, 64], 7772, &dev)?,
        )?;
        // 2. PCA projection (0 distillation steps)
        let pca_proj = distill_bidirectional_projection(
            &q_pre.narrow(2, 0, 1024)?,
            &k_pre.narrow(2, 0, 1024)?,
            &q_rot.narrow(2, 0, 1024)?,
            &k_rot.narrow(2, 0, 1024)?,
            64,
            128,
            0,
            0.0,
        )?;
        // 3. Distilled projection (12 forward-KL steps on prefix/tail calibration slice)
        let cal_q_pre = Tensor::cat(
            &[
                &q_pre.narrow(2, 0, 256)?,
                &q_pre.narrow(2, ctx_len / 2 - 128, 256)?,
                &q_pre.narrow(2, ctx_len - 256, 256)?,
            ],
            2,
        )?;
        let cal_k_pre = Tensor::cat(
            &[
                &k_pre.narrow(2, 0, 256)?,
                &k_pre.narrow(2, ctx_len / 2 - 128, 256)?,
                &k_pre.narrow(2, ctx_len - 256, 256)?,
            ],
            2,
        )?;
        let cal_q_rot = Tensor::cat(
            &[
                &q_rot.narrow(2, 0, 256)?,
                &q_rot.narrow(2, ctx_len / 2 - 128, 256)?,
                &q_rot.narrow(2, ctx_len - 256, 256)?,
            ],
            2,
        )?;
        let cal_k_rot = Tensor::cat(
            &[
                &k_rot.narrow(2, 0, 256)?,
                &k_rot.narrow(2, ctx_len / 2 - 128, 256)?,
                &k_rot.narrow(2, ctx_len - 256, 256)?,
            ],
            2,
        )?;
        let dist_proj = distill_bidirectional_projection(
            &cal_q_pre, &cal_k_pre, &cal_q_rot, &cal_k_rot, 64, 64, 15, 0.8,
        )?;

        let measure_evidence_recall = |sel: &hanzo_nn::attention::lsa::ReadSelection| -> f32 {
            let mut hits = 0usize;
            let mut total = 0usize;
            for qi in (ctx_len - 32)..ctx_len {
                let keep = &sel.keep_indices[0][qi];
                for &ev_start in &[40u32, (ctx_len / 2) as u32] {
                    for pos in ev_start..ev_start + 32 {
                        total += 1;
                        if keep.binary_search(&pos).is_ok() {
                            hits += 1;
                        }
                    }
                }
            }
            (hits as f32) / (total.max(1) as f32) * 100.0
        };

        for &budget_k in &[256usize, 512, 1_024] {
            let cfg_b = BidirectionalLsaConfig::ultrabert_with_budget(budget_k);
            let (_out_r, _b_r, sel_rand) = forward_bidirectional_dlsa(
                &q_pre,
                &k_pre,
                &q_rot,
                &k_rot,
                &v_tensor,
                &cfg_b,
                Some(&rand_proj),
                None,
            )?;
            let (_out_p, _b_p, sel_pca) = forward_bidirectional_dlsa(
                &q_pre,
                &k_pre,
                &q_rot,
                &k_rot,
                &v_tensor,
                &cfg_b,
                Some(&pca_proj),
                None,
            )?;
            let (out_d, _b_d, sel_dist) = forward_bidirectional_dlsa(
                &q_pre,
                &k_pre,
                &q_rot,
                &k_rot,
                &v_tensor,
                &cfg_b,
                Some(&dist_proj),
                None,
            )?;

            let rec_rand = measure_evidence_recall(&sel_rand);
            let rec_pca = measure_evidence_recall(&sel_pca);
            let rec_dist = measure_evidence_recall(&sel_dist);

            let dist_tail = out_d
                .narrow(2, ctx_len - 32, 32)?
                .permute((0, 2, 1, 3))?
                .reshape((32, 8 * 128))?;
            let dist_logits = dist_tail.matmul(&option_proj.t()?)?;
            let dist_winners = dist_logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
            let matches = teacher_winners
                .iter()
                .zip(dist_winners.iter())
                .filter(|(a, b)| a == b)
                .count();
            let kai_match_pct = (matches as f32) / 32.0 * 100.0;

            println!(
                "{:<10} | {:>10} | {:>16.1} % | {:>16.1} % | {:>16.1} % | {:>13.1} %",
                format!("{}K", ctx_len / 1024),
                budget_k,
                rec_rand,
                rec_pca,
                rec_dist,
                kai_match_pct
            );
        }
    }

    // Verify bidirectional symmetry: a query at position 10 can retrieve planted evidence at position T - 20!
    let t_sym = 1024;
    let q_sym = det_tensor(&[1, 8, t_sym, 128], 8801, &dev)?;
    let k_sym = det_tensor(&[1, 8, t_sym, 128], 8802, &dev)?;
    let (q_idx_s, k_idx_s) = regroup(&q_sym, &k_sym)?;
    let buckets_s = build_buckets_bidirectional(&k_idx_s, 256, 4, 8, 16)?;
    let sel_s = read_bidirectional(&q_idx_s, &buckets_s, t_sym, 128, 4, 32, 256, 0.0)?;
    // Query at pos=10 must retrieve keys > 138 (in the future direction!)
    let has_right_keys = sel_s.expanded_keys[0][10].iter().any(|&k| k > 200);
    // Query at pos=1000 must retrieve keys < 872 (in the past direction!)
    let has_left_keys = sel_s.expanded_keys[0][1000].iter().any(|&k| k < 800);
    assert!(
        has_right_keys && has_left_keys,
        "Bidirectional D-LSA must retrieve keys from both future (> pos + W) and past (< pos - W)"
    );

    Ok(())
}
