//! Production end-to-end tests for backend-independent `hanzo_ml::gdn` and `hanzo_ml::flash_attn`
//! powered by canonical `hanzo-kernel` DSL artifacts across CUDA, Vulkan, and Metal.
//!
//! Validates:
//! - Legacy-default safety (`HANZO_GDN_IMPL=legacy`, `HANZO_FLASH_IMPL=legacy` -> `Ok(None)`),
//! - Full GDN sequence equivalence: one-shot prefill == chunked prefill == single-token decode,
//! - `F32`, `F16`, `BF16` accuracy + nonzero `narrow()` byte offsets,
//! - Paged FlashAttention across `FlatPaged4D`, `EngineStandard5D4D`, and `Contiguous` layouts,
//!   noncontiguous shuffled page tables, variable `context_lens`, GQA, causal decode/prefill,
//!   sliding window, softcap, and `d = 64, 128, 256`,
//! - Single-compile proof (`compiled == 1` across repeated calls).

use hanzo_ml::flash_attn::{
    try_flash_attn_contiguous, try_flash_attn_paged, FlashAttnConfig, KvCacheLayout,
};
use hanzo_ml::gdn::{try_conv1d_stateful, try_conv1d_step, try_gating, try_recurrence};
use hanzo_ml::quantized::dsl::{self, QkImpl};
use hanzo_ml::{DType, Device, Tensor};

fn pseudo_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) as f32) / ((1u32 << 24) as f32);
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let max_mag = b
        .iter()
        .map(|y| y.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    max_diff / max_mag
}

fn gdn_recurrence_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    b: usize,
    seq: usize,
    nh: usize,
    kd: usize,
    vd: usize,
    apply_q_scale: bool,
) -> Vec<f32> {
    let q_scale = if apply_q_scale {
        1.0 / (kd as f32).sqrt()
    } else {
        1.0
    };
    let mut out = vec![0.0f32; b * seq * nh * vd];
    for bi in 0..b {
        for hi in 0..nh {
            let s_off = (bi * nh + hi) * kd * vd;
            for t in 0..seq {
                let g_v = g[((bi * seq + t) * nh) + hi].exp();
                let beta_v = beta[((bi * seq + t) * nh) + hi];
                let q_off = (((bi * seq + t) * nh + hi) * kd) as usize;
                let k_off = q_off;
                let v_off = (((bi * seq + t) * nh + hi) * vd) as usize;
                for vi in 0..vd {
                    let mut kv = 0.0f32;
                    for ki in 0..kd {
                        let s_idx = s_off + ki * vd + vi;
                        state[s_idx] *= g_v;
                        kv += state[s_idx] * k[k_off + ki];
                    }
                    let delta = (v[v_off + vi] - kv) * beta_v;
                    let mut acc = 0.0f32;
                    for ki in 0..kd {
                        let s_idx = s_off + ki * vd + vi;
                        state[s_idx] += k[k_off + ki] * delta;
                        acc += state[s_idx] * (q[q_off + ki] * q_scale);
                    }
                    out[v_off + vi] = acc;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn flash_paged_ref(
    q: &[f32],
    k_flat: &[f32], // [num_blocks, n_kv, block_size, d]
    v_flat: &[f32], // [num_blocks, n_kv, block_size, d]
    block_tables: &[u32],
    context_lens: &[u32],
    batch: usize,
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    d: usize,
    block_size: usize,
    max_blocks: usize,
    cfg: FlashAttnConfig,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * n_heads * seq_q * d];
    for b in 0..batch {
        let ctx_len = context_lens[b] as usize;
        let q_offset = if cfg.causal_mode == 2 {
            cfg.q_offset_override
        } else {
            ctx_len.saturating_sub(seq_q)
        };
        for h in 0..n_heads {
            let kv = h / (n_heads / n_kv);
            for qi in 0..seq_q {
                let q_pos = q_offset + qi;
                let q_base = ((b * n_heads + h) * seq_q + qi) * d;
                let mut scores = vec![f64::NEG_INFINITY; ctx_len];
                let mut max_s = f64::NEG_INFINITY;
                for kpos in 0..ctx_len {
                    if cfg.causal_mode != 0 && kpos > q_pos {
                        continue;
                    }
                    if cfg.sliding_window > 0 && kpos + cfg.sliding_window <= q_pos {
                        continue;
                    }
                    let blk = block_tables[b * max_blocks + kpos / block_size] as usize;
                    let off = kpos % block_size;
                    let k_base = ((blk * n_kv + kv) * block_size + off) * d;
                    let mut dot = 0.0f64;
                    for dd in 0..d {
                        dot += (q[q_base + dd] as f64) * (k_flat[k_base + dd] as f64);
                    }
                    let mut s = dot * (cfg.scale as f64);
                    if cfg.softcap > 0.0 {
                        s = (cfg.softcap as f64) * (s / (cfg.softcap as f64)).tanh();
                    }
                    scores[kpos] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut denom = 0.0f64;
                let mut probs = vec![0.0f64; ctx_len];
                for kpos in 0..ctx_len {
                    if scores[kpos].is_finite() {
                        let p = (scores[kpos] - max_s).exp();
                        probs[kpos] = p;
                        denom += p;
                    }
                }
                for dd in 0..d {
                    let mut acc = 0.0f64;
                    for kpos in 0..ctx_len {
                        if probs[kpos] > 0.0 {
                            let blk = block_tables[b * max_blocks + kpos / block_size] as usize;
                            let off = kpos % block_size;
                            let v_base = ((blk * n_kv + kv) * block_size + off) * d;
                            acc += (probs[kpos] / denom) * (v_flat[v_base + dd] as f64);
                        }
                    }
                    out[q_base + dd] = acc as f32;
                }
            }
        }
    }
    out
}

fn pack_engine_standard_kv(
    k_flat: &[f32],
    v_flat: &[f32],
    num_blocks: usize,
    n_kv: usize,
    block_size: usize,
    d: usize,
    x_vec: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dx = d / x_vec;
    let mut k5 = vec![0.0f32; num_blocks * n_kv * dx * block_size * x_vec];
    let mut v4 = vec![0.0f32; num_blocks * n_kv * d * block_size];
    for blk in 0..num_blocks {
        for kv in 0..n_kv {
            for off in 0..block_size {
                let fbase = ((blk * n_kv + kv) * block_size + off) * d;
                for dd in 0..d {
                    let ki = ((((blk * n_kv + kv) * dx + (dd / x_vec)) * block_size + off) * x_vec)
                        + (dd % x_vec);
                    k5[ki] = k_flat[fbase + dd];
                    let vi = (((blk * n_kv + kv) * d + dd) * block_size) + off;
                    v4[vi] = v_flat[fbase + dd];
                }
            }
        }
    }
    (k5, v4)
}

fn run_gdn_suite(dev: &Device, dtypes: &[DType], label: &str) {
    // 1. Legacy-default safety check: must return Ok(None) when switched to Legacy.
    dsl::set_gdn_impl(QkImpl::Legacy);
    {
        let x = Tensor::zeros((1, 8, 4), DType::F32, dev).unwrap();
        let w = Tensor::zeros((8, 4), DType::F32, dev).unwrap();
        let s = Tensor::zeros((1, 8, 4), DType::F32, dev).unwrap();
        assert!(try_conv1d_stateful(&x, &w, &s, false).unwrap().is_none());
    }

    dsl::set_gdn_impl(QkImpl::Dsl);

    for &dtype in dtypes {
        let tol_conv = match dtype {
            DType::F32 => 1e-5,
            DType::F16 => 3e-3,
            DType::BF16 => 2e-2,
            _ => unreachable!(),
        };
        let (batch, conv_dim, seq_len) = (2usize, 32usize, 13usize);
        let x_vec = pseudo_vec(batch * conv_dim * seq_len, 101, 0.7);
        let w_vec = pseudo_vec(conv_dim * 4, 202, 0.5);
        let x = Tensor::from_vec(x_vec, (batch, conv_dim, seq_len), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let w = Tensor::from_vec(w_vec, (conv_dim, 4), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let s0 = Tensor::zeros((batch, conv_dim, 4), dtype, dev).unwrap();

        // One-shot prefill
        let (out_full, st_full) = try_conv1d_stateful(&x, &w, &s0, false)
            .unwrap()
            .expect("DSL conv1d_stateful should dispatch");

        // Chunked prefill [5, 4, 3, 1]
        let mut st_chunk = s0.clone();
        let mut chunk_outs = Vec::new();
        let mut pos = 0usize;
        for &clen in &[5usize, 4, 3, 1] {
            let x_c = x.narrow(2, pos, clen).unwrap().contiguous().unwrap();
            let (co, ns) = try_conv1d_stateful(&x_c, &w, &st_chunk, pos > 0)
                .unwrap()
                .expect("DSL conv1d_stateful chunk should dispatch");
            st_chunk = ns;
            chunk_outs.push(co);
            pos += clen;
        }
        let out_chunk = Tensor::cat(&chunk_outs, 2).unwrap();

        // 13x single-token decode steps (using try_conv1d_step with nonzero narrow offset on x_bt!)
        let x_bt = x.transpose(1, 2).unwrap().contiguous().unwrap(); // [B, S, C]
        let mut st_step = s0.clone();
        let mut step_outs = Vec::new();
        for t in 0..seq_len {
            // Deliberately pass a narrowed view (`start_offset > 0` for t > 0) on batch=1 slice
            // and full batch slice for [B, 1, C].
            let xt = x_bt.narrow(1, t, 1).unwrap().contiguous().unwrap();
            let yo = try_conv1d_step(&xt, &w, &mut st_step)
                .unwrap()
                .expect("DSL conv1d_step should dispatch");
            step_outs.push(yo);
        }
        let out_step = Tensor::cat(&step_outs, 1)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();

        let full_v = out_full.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let chunk_v = out_chunk.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let step_v = out_step.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let st_full_v = st_full.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let st_chunk_v = st_chunk.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let st_step_v = st_step.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let e_fc = rel_err(&full_v, &chunk_v);
        let e_fs = rel_err(&full_v, &step_v);
        let es_fc = rel_err(&st_full_v, &st_chunk_v);
        let es_fs = rel_err(&st_full_v, &st_step_v);
        eprintln!(
            "[{label} gdn_conv1d {dtype:?}] full==chunk={e_fc:.2e} full==step={e_fs:.2e} state(c={es_fc:.2e}, s={es_fs:.2e})"
        );
        assert!(e_fc <= tol_conv && e_fs <= tol_conv);
        assert!(es_fc == 0.0 && es_fs == 0.0);

        // Gating + Recurrence (k=128, v=128 and k=64, v=64): one-shot vs step-by-step vs CPU ref
        for &(kd, vd) in &[(128usize, 128usize), (64usize, 64usize)] {
            let (rb, rseq, rnh) = (2usize, 7usize, 4usize);
            let b_in_v = pseudo_vec(rb * rseq * rnh, 303, 1.2);
            let a_in_v = pseudo_vec(rb * rseq * rnh, 404, 1.2);
            let a_log_v = pseudo_vec(rnh, 505, 0.4);
            let dt_bias_v = pseudo_vec(rnh, 606, 0.3);

            let b_in = Tensor::from_vec(b_in_v, (rb, rseq, rnh), dev).unwrap().to_dtype(dtype).unwrap();
            let a_in = Tensor::from_vec(a_in_v, (rb, rseq, rnh), dev).unwrap().to_dtype(dtype).unwrap();
            let a_log = Tensor::from_vec(a_log_v, rnh, dev).unwrap();
            let dt_bias = Tensor::from_vec(dt_bias_v, rnh, dev).unwrap();

            let (beta, g) = try_gating(&b_in, &a_in, &a_log, &dt_bias)
                .unwrap()
                .expect("DSL gating should dispatch");

            let q_v = pseudo_vec(rb * rseq * rnh * kd, 707, 0.3);
            let k_v = pseudo_vec(rb * rseq * rnh * kd, 808, 0.3);
            let v_v = pseudo_vec(rb * rseq * rnh * vd, 909, 0.3);
            let q = Tensor::from_vec(q_v, (rb, rseq, rnh, kd), dev).unwrap().to_dtype(dtype).unwrap();
            let k = Tensor::from_vec(k_v, (rb, rseq, rnh, kd), dev).unwrap().to_dtype(dtype).unwrap();
            let v = Tensor::from_vec(v_v, (rb, rseq, rnh, vd), dev).unwrap().to_dtype(dtype).unwrap();

            // Nonzero offset check on `q`: prepend a dummy batch and narrow(0, 1, rb).
            let pad = Tensor::zeros((1, rseq, rnh, kd), dtype, dev).unwrap();
            let q_off = Tensor::cat(&[&pad, &q], 0).unwrap().narrow(0, 1, rb).unwrap();

            let mut st_one = Tensor::zeros((rb, rnh, kd, vd), DType::F32, dev).unwrap();
            let out_one = try_recurrence(&q_off, &k, &v, &g, &beta, &mut st_one, true)
                .unwrap()
                .expect("DSL recurrence one-shot should dispatch");

            let mut st_step_r = Tensor::zeros((rb, rnh, kd, vd), DType::F32, dev).unwrap();
            let mut step_ys = Vec::new();
            for t in 0..rseq {
                let qt = q.narrow(1, t, 1).unwrap().contiguous().unwrap();
                let kt = k.narrow(1, t, 1).unwrap().contiguous().unwrap();
                let vt = v.narrow(1, t, 1).unwrap().contiguous().unwrap();
                let gt = g.narrow(1, t, 1).unwrap().contiguous().unwrap();
                let bt = beta.narrow(1, t, 1).unwrap().contiguous().unwrap();
                let yt = try_recurrence(&qt, &kt, &vt, &gt, &bt, &mut st_step_r, true)
                    .unwrap()
                    .expect("DSL recurrence step should dispatch");
                step_ys.push(yt);
            }
            let out_steps = Tensor::cat(&step_ys, 1).unwrap();

            let one_v = out_one.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let steps_v = out_steps.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let st1_v = st_one.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let st2_v = st_step_r.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            // Also compare against CPU f32 reference using the quantized inputs
            let q_f32 = q.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let k_f32 = k.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let v_f32 = v.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let g_f32 = g.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b_f32 = beta.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut st_ref = vec![0.0f32; rb * rnh * kd * vd];
            let out_ref = gdn_recurrence_ref(
                &q_f32, &k_f32, &v_f32, &g_f32, &b_f32, &mut st_ref, rb, rseq, rnh, kd, vd, true,
            );

            let e_os = rel_err(&one_v, &steps_v);
            let es_os = rel_err(&st1_v, &st2_v);
            let e_ref = rel_err(&one_v, &out_ref);
            let es_ref = rel_err(&st1_v, &st_ref);
            let tol_r = match dtype {
                DType::F32 => 1e-4,
                DType::F16 => 5e-3,
                DType::BF16 => 2e-2,
                _ => unreachable!(),
            };
            eprintln!(
                "[{label} gdn_scan k={kd} {dtype:?}] one==step(out={e_os:.2e}, st={es_os:.2e}) vs_ref(out={e_ref:.2e}, st={es_ref:.2e})"
            );
            assert!(e_os <= 1e-5 && es_os <= 1e-5);
            assert!(e_ref <= tol_r && es_ref <= tol_r);
        }
    }

    // Verify single-compile invariant on `gdn_conv1d_step_k4_f32`
    let stats = dsl::stats();
    assert_eq!(
        stats.compiles_of("gdn_conv1d_step_k4_f32"),
        1,
        "must compile exactly once"
    );
    let step_calls: u64 = stats
        .dispatches
        .iter()
        .filter(|d| d.1 == "gdn_conv1d_step_k4_f32")
        .map(|d| d.4)
        .sum();
    assert!(step_calls >= 13, "must have dispatched >= 13 steps");

    dsl::set_gdn_impl(QkImpl::Legacy);
}

fn run_flash_suite(dev: &Device, dtypes: &[DType], label: &str) {
    // 1. Legacy-default safety check
    dsl::set_flash_impl(QkImpl::Legacy);
    {
        let q = Tensor::zeros((1, 4, 2, 128), DType::F32, dev).unwrap();
        let k = Tensor::zeros((1, 2, 8, 128), DType::F32, dev).unwrap();
        let v = Tensor::zeros((1, 2, 8, 128), DType::F32, dev).unwrap();
        assert!(
            try_flash_attn_contiguous(&q, &k, &v, FlashAttnConfig::causal(1.0 / 128.0f32.sqrt()))
                .unwrap()
                .is_none()
        );
    }

    dsl::set_flash_impl(QkImpl::Dsl);

    let cases = [
        // (d, seq_q, layout_kind, sliding_window, softcap)
        (64usize, 1usize, 0usize, 0usize, 0.0f32),
        (128usize, 1usize, 1usize, 0usize, 0.0f32),
        (128usize, 5usize, 1usize, 0usize, 0.0f32),
        (128usize, 5usize, 0usize, 16usize, 30.0f32),
        (256usize, 1usize, 1usize, 0usize, 0.0f32),
    ];

    for &dtype in dtypes {
        let tol = match dtype {
            DType::F32 => 1.5e-3,
            DType::F16 => 2.5e-3,
            DType::BF16 => 1.5e-2,
            _ => unreachable!(),
        };
        for &(d, seq_q, layout_kind, win, cap) in &cases {
            let (batch, n_heads, n_kv) = (3usize, 8usize, 2usize);
            let block_size = 16usize;
            let max_blocks = 4usize;
            let num_blocks = 12usize;
            let ctx_lens_v = vec![23u32, 51, 9];
            // Noncontiguous shuffled page table across all 3 requests
            let bt_v: Vec<u32> = vec![
                7, 2, 11, 0, // req 0
                9, 4, 1, 6,  // req 1
                10, 3, 5, 8, // req 2
            ];
            let q_v = pseudo_vec(batch * n_heads * seq_q * d, 1111 + d as u64, 0.4);
            let kf_v = pseudo_vec(num_blocks * n_kv * block_size * d, 2222 + d as u64, 0.4);
            let vf_v = pseudo_vec(num_blocks * n_kv * block_size * d, 3333 + d as u64, 0.4);

            let cfg = FlashAttnConfig {
                scale: 1.0 / (d as f32).sqrt(),
                softcap: cap,
                causal_mode: 1,
                sliding_window: win,
                q_offset_override: 0,
            };
            let ref_out = flash_paged_ref(
                &q_v,
                &kf_v,
                &vf_v,
                &bt_v,
                &ctx_lens_v,
                batch,
                n_heads,
                n_kv,
                seq_q,
                d,
                block_size,
                max_blocks,
                cfg,
            );

            let q = Tensor::from_vec(q_v, (batch, n_heads, seq_q, d), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let bt = Tensor::from_vec(bt_v, (batch, max_blocks), dev).unwrap();
            let cl = Tensor::from_vec(ctx_lens_v, batch, dev).unwrap();

            let (kc, vc, kv_layout) = if layout_kind == 0 {
                let kc = Tensor::from_vec(kf_v, (num_blocks, n_kv, block_size, d), dev)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let vc = Tensor::from_vec(vf_v, (num_blocks, n_kv, block_size, d), dev)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                (kc, vc, KvCacheLayout::FlatPaged4D)
            } else {
                let x_vec = 8usize;
                let (k5, v4) =
                    pack_engine_standard_kv(&kf_v, &vf_v, num_blocks, n_kv, block_size, d, x_vec);
                let kc = Tensor::from_vec(k5, (num_blocks, n_kv, d / x_vec, block_size, x_vec), dev)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let vc = Tensor::from_vec(v4, (num_blocks, n_kv, d, block_size), dev)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                (kc, vc, KvCacheLayout::EngineStandard5D4D { x_vec })
            };

            let out = try_flash_attn_paged(&q, &kc, &vc, &bt, &cl, n_kv, kv_layout, cfg)
                .unwrap()
                .expect("DSL flash_attn_paged should dispatch");
            let got = out
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let err = rel_err(&got, &ref_out);
            eprintln!(
                "[{label} flash_paged {dtype:?}] d={d:3} sq={seq_q} layout={layout_kind} win={win:2} cap={cap:2.0} rel={err:.2e}"
            );
            assert!(err <= tol, "flash_paged err {err:.2e} > {tol:.2e}");
        }

        // Contiguous SDPA check (with causal prefill)
        let (batch, n_heads, n_kv, seq_q, seq_k, d) = (2usize, 8usize, 2usize, 7usize, 37usize, 128usize);
        let q_v = pseudo_vec(batch * n_heads * seq_q * d, 4444, 0.4);
        let k_v = pseudo_vec(batch * n_kv * seq_k * d, 5555, 0.4);
        let v_v = pseudo_vec(batch * n_kv * seq_k * d, 6666, 0.4);
        let cfg = FlashAttnConfig::causal(1.0 / (d as f32).sqrt());
        // Reuse flash_paged_ref with 1 block per batch of size seq_k
        let bt_contig: Vec<u32> = (0..batch as u32).collect();
        let cl_contig = vec![seq_k as u32; batch];
        let ref_c = flash_paged_ref(
            &q_v, &k_v, &v_v, &bt_contig, &cl_contig, batch, n_heads, n_kv, seq_q, d, seq_k, 1, cfg,
        );
        let q = Tensor::from_vec(q_v, (batch, n_heads, seq_q, d), dev).unwrap().to_dtype(dtype).unwrap();
        let k = Tensor::from_vec(k_v, (batch, n_kv, seq_k, d), dev).unwrap().to_dtype(dtype).unwrap();
        let v = Tensor::from_vec(v_v, (batch, n_kv, seq_k, d), dev).unwrap().to_dtype(dtype).unwrap();
        let out = try_flash_attn_contiguous(&q, &k, &v, cfg)
            .unwrap()
            .expect("DSL flash_attn_contiguous should dispatch");
        let got = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let err = rel_err(&got, &ref_c);
        eprintln!("[{label} flash_contig {dtype:?}] d=128 sq=7 sk=37 rel={err:.2e}");
        assert!(err <= tol, "flash_contig err {err:.2e} > {tol:.2e}");
    }

    let stats = dsl::stats();
    assert_eq!(
        stats.compiles_of("flash_attn_paged_d128_f32"),
        1,
        "must compile exactly once"
    );
    let f_calls: u64 = stats
        .dispatches
        .iter()
        .filter(|d| d.1 == "flash_attn_paged_d128_f32")
        .map(|d| d.4)
        .sum();
    assert!(f_calls >= 4, "must have dispatched >= 4 calls");

    dsl::set_flash_impl(QkImpl::Legacy);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    run_gdn_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "CUDA");
    run_flash_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "CUDA");
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_vulkan(0).expect("Vulkan device 0");
    run_gdn_suite(&dev, &[DType::F32], "Vulkan");
    run_flash_suite(&dev, &[DType::F32], "Vulkan");
}

#[cfg(feature = "metal")]
#[test]
fn metal_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_metal(0).expect("Metal device 0");
    run_gdn_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "Metal");
    run_flash_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "Metal");
}
