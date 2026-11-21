//! End-to-end benchmarks and accuracy evaluation for Latent Sparse Attention (LSA)
//! in `hanzo-nn` using the Qwen3.8-27B / Hanzo Zen-27B GQA geometry:
//! `Hq = 24`, `Hkv = 4`, `head_dim = 256`, `rotary_dim = 64` (Prefix), `d_free = 192`,
//! `input_dim = 4 * 192 = 768`, `index_dim = 64`, `window_size = 512`, `sink_size = 4`,
//! `pool_size = 4`, `bucket_size = 8` (64 buckets/block), `bucket_cap = 16`.

use std::time::Instant;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::attention::lsa::{
    build_buckets, build_hierarchy, project_partial, read, read_hierarchy, regroup_partial,
    shape_buckets, sparse_gather_attention, LearnedProjection, RotaryLayout,
};
use hanzo_nn::ops::softmax_last_dim;

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
        vals.push(u as f32);
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

/// Full GQA SDPA baseline (`q_len` queries attending to `kv_len` keys).
fn full_gqa_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    let (b, h_q, t_q, d_h) = q.dims4()?;
    let (_, h_kv, t_k, _) = k.dims4()?;
    let d_v = v.dim(3)?;
    let group = h_q / h_kv;
    let scale = 1.0 / (d_h as f64).sqrt();

    let q_g = q.reshape((b * h_kv, group * t_q, d_h))?;
    let k_t = k
        .reshape((b * h_kv, t_k, d_h))?
        .transpose(1, 2)?
        .contiguous()?;
    let v_g = v.reshape((b * h_kv, t_k, d_v))?;

    let mut scores = (q_g.matmul(&k_t)? * scale)?.reshape((b, h_q, t_q, t_k))?;
    if causal && t_q > 1 {
        let offset = t_k.saturating_sub(t_q);
        let mut mask = vec![0.0f32; t_q * t_k];
        for i in 0..t_q {
            let pos = offset + i;
            for j in (pos + 1)..t_k {
                mask[i * t_k + j] = f32::NEG_INFINITY;
            }
        }
        let mask_t = Tensor::from_vec(mask, (1, 1, t_q, t_k), q.device())?
            .broadcast_as((b, h_q, t_q, t_k))?;
        scores = scores.add(&mask_t)?;
    }
    let probs = softmax_last_dim(&scores)?.reshape((b * h_kv, group * t_q, t_k))?;
    probs.matmul(&v_g)?.reshape((b, h_q, t_q, d_v))
}

#[test]
fn benchmark_lsa_suite() -> Result<()> {
    let device = Device::Cpu;
    // Exact Qwen3.8-27B / Hanzo Zen-27B single-layer geometry:
    // 24 query heads, 4 KV heads (G = 6), head_dim = 256, rotary_dim = 64 (Prefix),
    // d_free = 192 -> input_dim = 4 * 192 = 768 -> projected to index_dim = 64.
    let b = 1;
    let h_q = 24;
    let h_kv = 4;
    let d_h = 256;
    let rotary_dim = 64;
    let d_free = d_h - rotary_dim; // 192
    let d_in = h_kv * d_free; // 768
    let d_idx = 64;

    let window_size = 512;
    let sink_size = 4;
    let pool_size = 4;
    let bucket_size = 8; // 64 buckets per 512-token block
    let bucket_cap = 16;

    println!("\n==========================================================================================================");
    println!("  Latent Sparse Attention (LSA) Benchmark — Qwen3.8-27B / Zen-27B Geometry");
    println!("  Hq={h_q}, Hkv={h_kv} (G=6), Dh={d_h}, rotary_dim={rotary_dim} (Prefix), d_in={d_in} -> d_idx={d_idx}");
    println!("  window_size={window_size}, sink_size={sink_size}, pool_size={pool_size}, bucket_size={bucket_size}, bucket_cap={bucket_cap}");
    println!("==========================================================================================================\n");

    // ------------------------------------------------------------------------
    // Part 1: Decode Step Latency & Routing Throughput across Context Lengths (4K -> 128K)
    // ------------------------------------------------------------------------
    let top_k = 512;
    let top_p = 64;
    println!("--- [1] Single-Layer Decode Latency (4K to 128K Context, top_k={top_k}, top_p={top_p}, window={window_size}) ---");
    println!(
        "{:<10} | {:<13} | {:<14} | {:<14} | {:<14} | {:<12} | {:<12}",
        "Context",
        "Ward Build",
        "Full Decode",
        "D-LSA Flat",
        "D-LSA Tree",
        "Flat Speedup",
        "Tree Speedup"
    );
    println!("{}", "-".repeat(104));

    let w_q = det_tensor(&[d_in, d_idx], 10, &device)?;
    let w_k = det_tensor(&[d_in, d_idx], 20, &device)?;
    let proj = LearnedProjection::new(w_q, w_k)?;

    for &ctx_len in &[4_096usize, 8_192, 16_384, 32_768, 65_536, 131_072] {
        let q = det_tensor(&[b, h_q, 1, d_h], 100 + ctx_len as u64, &device)?;
        let k = det_tensor(&[b, h_kv, ctx_len, d_h], 200 + ctx_len as u64, &device)?;
        let v = det_tensor(&[b, h_kv, ctx_len, d_h], 300 + ctx_len as u64, &device)?;

        let (q_idx, k_idx) = project_partial(&q, &k, rotary_dim, RotaryLayout::Prefix, &proj)?;

        // Measure Ward bucket construction on the full context
        let t0 = Instant::now();
        let buckets = build_buckets(&k_idx, window_size, pool_size, bucket_size, bucket_cap)?;
        let ward_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Build 2-level hierarchy for Tree LSA (node_bucket_size=4, node_cluster_block_size=256)
        let c0 = buckets.centroids.narrow(0, 0, 1)?.squeeze(0)?;
        let hier = build_hierarchy(
            &c0,
            &buckets.bucket_sizes[0],
            buckets.buckets_per_block,
            None,
            2,
            4,
            256,
        )?;
        let hierarchies = vec![hier];

        // Warmup
        let _ = full_gqa_attention(&q, &k, &v, false)?;
        let sel_f = read(
            &q_idx,
            &buckets,
            window_size,
            sink_size,
            top_p,
            top_k,
            0.0,
            None,
            ctx_len - 1,
        )?;
        let _ = sparse_gather_attention(&q, &k, &v, &sel_f, None)?;

        let iters = if ctx_len >= 65_536 { 5 } else { 10 };

        // 1. Full Attention Decode
        let t_full_start = Instant::now();
        for _ in 0..iters {
            let _ = full_gqa_attention(&q, &k, &v, false)?;
        }
        let full_ms = (t_full_start.elapsed().as_secs_f64() * 1000.0) / (iters as f64);

        // 2. D-LSA Flat Decode (selection + sparse gather SDPA)
        let t_flat_start = Instant::now();
        for _ in 0..iters {
            let sel = read(
                &q_idx,
                &buckets,
                window_size,
                sink_size,
                top_p,
                top_k,
                0.0,
                None,
                ctx_len - 1,
            )?;
            let _ = sparse_gather_attention(&q, &k, &v, &sel, None)?;
        }
        let flat_ms = (t_flat_start.elapsed().as_secs_f64() * 1000.0) / (iters as f64);

        // 3. D-LSA Tree Decode (hierarchical beam search + sparse gather SDPA)
        let t_tree_start = Instant::now();
        for _ in 0..iters {
            let sel = read_hierarchy(
                &q_idx,
                &buckets,
                &hierarchies,
                window_size,
                sink_size,
                top_p,
                top_k,
                64,
                256,
                None,
                ctx_len - 1,
            )?;
            let _ = sparse_gather_attention(&q, &k, &v, &sel, None)?;
        }
        let tree_ms = (t_tree_start.elapsed().as_secs_f64() * 1000.0) / (iters as f64);

        println!(
            "{:<10} | {:>10.2} ms | {:>11.3} ms | {:>11.3} ms | {:>11.3} ms | {:>10.2}x | {:>10.2}x",
            format!("{}K", ctx_len / 1024),
            ward_ms,
            full_ms,
            flat_ms,
            tree_ms,
            full_ms / flat_ms,
            full_ms / tree_ms,
        );
    }

    // ------------------------------------------------------------------------
    // Part 2: Chunked Prefill Latency (q_len = 128 attending to 4K -> 64K prefix)
    // ------------------------------------------------------------------------
    let q_chunk_len = 128;
    println!(
        "\n--- [2] Chunked Prefill Latency (q_len={q_chunk_len} attending to 4K..64K prefix, top_k={top_k}) ---"
    );
    println!(
        "{:<10} | {:<15} | {:<15} | {:<15} | {:<12} | {:<12}",
        "Prefix", "Full Prefill", "D-LSA Flat", "D-LSA Tree", "Flat Speedup", "Tree Speedup"
    );
    println!("{}", "-".repeat(88));

    for &ctx_len in &[4_096usize, 8_192, 16_384, 32_768, 65_536] {
        let q = det_tensor(&[b, h_q, q_chunk_len, d_h], 400 + ctx_len as u64, &device)?;
        let k = det_tensor(&[b, h_kv, ctx_len, d_h], 500 + ctx_len as u64, &device)?;
        let v = det_tensor(&[b, h_kv, ctx_len, d_h], 600 + ctx_len as u64, &device)?;

        let (q_idx, k_idx) = project_partial(&q, &k, rotary_dim, RotaryLayout::Prefix, &proj)?;
        let buckets = build_buckets(&k_idx, window_size, pool_size, bucket_size, bucket_cap)?;
        let c0 = buckets.centroids.narrow(0, 0, 1)?.squeeze(0)?;
        let hier = build_hierarchy(
            &c0,
            &buckets.bucket_sizes[0],
            buckets.buckets_per_block,
            None,
            2,
            4,
            256,
        )?;
        let hierarchies = vec![hier];
        let query_offset = ctx_len - q_chunk_len;

        let t0 = Instant::now();
        let _ = full_gqa_attention(&q, &k, &v, true)?;
        let full_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let sel_flat = read(
            &q_idx,
            &buckets,
            window_size,
            sink_size,
            top_p,
            top_k,
            0.0,
            None,
            query_offset,
        )?;
        let _ = sparse_gather_attention(&q, &k, &v, &sel_flat, None)?;
        let flat_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let sel_tree = read_hierarchy(
            &q_idx,
            &buckets,
            &hierarchies,
            window_size,
            sink_size,
            top_p,
            top_k,
            64,
            256,
            None,
            query_offset,
        )?;
        let _ = sparse_gather_attention(&q, &k, &v, &sel_tree, None)?;
        let tree_ms = t2.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{:<10} | {:>12.2} ms | {:>12.2} ms | {:>12.2} ms | {:>10.2}x | {:>10.2}x",
            format!("{}K", ctx_len / 1024),
            full_ms,
            flat_ms,
            tree_ms,
            full_ms / flat_ms,
            full_ms / tree_ms,
        );
    }

    // ------------------------------------------------------------------------
    // Part 3: Output Cosine Similarity & Retrieval Accuracy vs Sparsity Budget
    // ------------------------------------------------------------------------
    println!("\n--- [3] Output Accuracy (Cosine Similarity vs Full Attention at 8K Context) ---");
    let ctx_acc = 8_192;
    let q_acc = det_tensor(&[b, h_q, 16, d_h], 777, &device)?;
    let k_base = det_tensor(&[b, h_kv, ctx_acc, d_h], 888, &device)?;
    let v_acc = det_tensor(&[b, h_kv, ctx_acc, d_h], 999, &device)?;
    let q_grouped = q_acc
        .reshape((b, h_kv, h_q / h_kv, 16, d_h))?
        .mean(2)?
        .mean(2)?
        .unsqueeze(2)?;
    let mut k_slices = Vec::new();
    for blk in 0..(ctx_acc / 256) {
        let slice = k_base.narrow(2, blk * 256, 256)?;
        if blk % 8 == 2 {
            k_slices.push((slice + q_grouped.broadcast_as((b, h_kv, 256, d_h))? * 2.5)?);
        } else {
            k_slices.push(slice);
        }
    }
    let k_acc = Tensor::cat(&k_slices, 2)?;

    let full_out = full_gqa_attention(&q_acc, &k_acc, &v_acc, true)?;
    let (q_idx_reg, k_idx_reg) = regroup_partial(&q_acc, &k_acc, rotary_dim, RotaryLayout::Prefix)?;
    let mut buckets_acc =
        build_buckets(&k_idx_reg, window_size, pool_size, bucket_size, bucket_cap)?;
    shape_buckets(&mut buckets_acc, &k_idx_reg, 2)?;

    println!(
        "{:<18} | {:<12} | {:<16} | {:<18} | {:<18}",
        "Budget (top_k)", "Attended %", "Flat Cosine Sim", "Shaped Cosine Sim", "Max Abs Diff"
    );
    println!("{}", "-".repeat(90));

    for &(p_bud, k_bud) in &[
        (16usize, 128usize),
        (32, 256),
        (64, 512),
        (128, 1024),
        (256, 2048),
        (1024, 8192),
    ] {
        let sel_flat = read(
            &q_idx_reg,
            &buckets_acc,
            window_size,
            sink_size,
            p_bud,
            k_bud,
            0.0,
            None,
            ctx_acc - 16,
        )?;
        let out_flat = sparse_gather_attention(&q_acc, &k_acc, &v_acc, &sel_flat, None)?;
        let cos_flat = cosine_similarity(&full_out, &out_flat)?;

        let sel_shaped = read(
            &q_idx_reg,
            &buckets_acc,
            window_size,
            sink_size,
            p_bud,
            k_bud,
            2.0,
            None,
            ctx_acc - 16,
        )?;
        let out_shaped = sparse_gather_attention(&q_acc, &k_acc, &v_acc, &sel_shaped, None)?;
        let cos_shaped = cosine_similarity(&full_out, &out_shaped)?;
        let max_diff = (&full_out - &out_shaped)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;

        let attended = (sink_size + window_size + k_bud).min(ctx_acc);
        let pct = (attended as f64 / ctx_acc as f64) * 100.0;
        println!(
            "top_k={:<12} | {:>10.1}% | {:>16.6} | {:>18.6} | {:>18.6}",
            k_bud, pct, cos_flat, cos_shaped, max_diff
        );
    }

    // ------------------------------------------------------------------------
    // Part 4: Qwen3.8-27B (16 Full-Attention Layers) VRAM & Bandwidth Accounting
    // ------------------------------------------------------------------------
    println!("\n--- [4] Qwen3.8-27B / Zen-27B (16 Full-Attn Layers) Memory & Bandwidth Footprint (32K -> 10M) ---");
    println!(
        "{:<10} | {:<14} | {:<16} | {:<16} | {:<16} | {:<14}",
        "Context",
        "BF16 KV Cache",
        "LSA Bucket Idx",
        "Full Read/Step",
        "LSA Read/Step",
        "BW Reduction"
    );
    println!("{}", "-".repeat(98));

    let num_full_layers = 16usize;
    let kv_bytes_per_token_per_layer = 2 * h_kv * d_h * 2; // K + V in BF16 = 2 * 4 * 256 * 2 = 4096 B/layer -> 64 KiB/token total
    let lsa_bytes_per_bucket = d_idx * 2 + bucket_cap * 4 + 4; // BF16 centroid (128B) + i32 members (64B) + size (4B) = 196 B per bucket (8 tokens)

    for &ctx_len in &[32_768usize, 131_072, 262_144, 1_048_576, 10_485_760] {
        let kv_total_gib = (ctx_len * kv_bytes_per_token_per_layer * num_full_layers) as f64
            / (1024.0 * 1024.0 * 1024.0);
        let num_buckets = ctx_len / bucket_size;
        let idx_total_mib =
            (num_buckets * lsa_bytes_per_bucket * num_full_layers) as f64 / (1024.0 * 1024.0);

        // Full attention reads entire KV cache every decode token; LSA reads (sink + window + top_k = 4612 tokens) + bucket centroids
        let lsa_attended_tokens = (sink_size + window_size + 4096).min(ctx_len);
        let lsa_kv_read_mib = (lsa_attended_tokens * kv_bytes_per_token_per_layer * num_full_layers)
            as f64
            / (1024.0 * 1024.0);
        let lsa_centroid_scan_mib =
            (num_buckets * d_idx * 2 * num_full_layers) as f64 / (1024.0 * 1024.0);
        let lsa_total_read_mib = lsa_kv_read_mib + lsa_centroid_scan_mib;
        let full_read_mib = kv_total_gib * 1024.0;
        let bw_reduction = full_read_mib / lsa_total_read_mib;

        let label = if ctx_len >= 1_048_576 {
            format!("{}M", ctx_len / 1_048_576)
        } else {
            format!("{}K", ctx_len / 1024)
        };
        println!(
            "{:<10} | {:>10.2} GiB | {:>12.1} MiB | {:>12.1} MiB | {:>12.1} MiB | {:>11.1}x",
            label, kv_total_gib, idx_total_mib, full_read_mib, lsa_total_read_mib, bw_reduction
        );
    }

    Ok(())
}
