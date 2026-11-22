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

#![cfg(any(
    feature = "cuda",
    feature = "vulkan",
    feature = "metal",
    feature = "rocm"
))]

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
    let max_mag = b.iter().map(|y| y.abs()).fold(0.0f32, f32::max).max(1e-6);
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
                let q_off = ((bi * seq + t) * nh + hi) * kd;
                let k_off = q_off;
                let v_off = ((bi * seq + t) * nh + hi) * vd;
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

        let full_v = out_full
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let chunk_v = out_chunk
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let step_v = out_step
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let st_full_v = st_full
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let st_chunk_v = st_chunk
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let st_step_v = st_step
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

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

            let b_in = Tensor::from_vec(b_in_v, (rb, rseq, rnh), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let a_in = Tensor::from_vec(a_in_v, (rb, rseq, rnh), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let a_log = Tensor::from_vec(a_log_v, rnh, dev).unwrap();
            let dt_bias = Tensor::from_vec(dt_bias_v, rnh, dev).unwrap();

            let (beta, g) = try_gating(&b_in, &a_in, &a_log, &dt_bias)
                .unwrap()
                .expect("DSL gating should dispatch");

            let q_v = pseudo_vec(rb * rseq * rnh * kd, 707, 0.3);
            let k_v = pseudo_vec(rb * rseq * rnh * kd, 808, 0.3);
            let v_v = pseudo_vec(rb * rseq * rnh * vd, 909, 0.3);
            let q = Tensor::from_vec(q_v, (rb, rseq, rnh, kd), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let k = Tensor::from_vec(k_v, (rb, rseq, rnh, kd), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let v = Tensor::from_vec(v_v, (rb, rseq, rnh, vd), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();

            // Nonzero offset check on `q`: prepend a dummy batch and narrow(0, 1, rb).
            let pad = Tensor::zeros((1, rseq, rnh, kd), dtype, dev).unwrap();
            let q_off = Tensor::cat(&[&pad, &q], 0)
                .unwrap()
                .narrow(0, 1, rb)
                .unwrap();

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

            let one_v = out_one
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let steps_v = out_steps
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let st1_v = st_one.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let st2_v = st_step_r.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            // Also compare against CPU f32 reference using the quantized inputs
            let q_f32 = q
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let k_f32 = k
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let v_f32 = v
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let g_f32 = g
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b_f32 = beta
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let mut st_ref = vec![0.0f32; rb * rnh * kd * vd];
            let out_ref = gdn_recurrence_ref(
                &q_f32,
                &k_f32,
                &v_f32,
                &g_f32,
                &b_f32,
                &mut st_ref,
                rb,
                rseq,
                rnh,
                kd,
                vd,
                true,
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

    // Verify single-compile invariant on `gdn_conv1d_step_k4_f32` (or `_p64` on Wave64 Vulkan)
    let stats = dsl::stats();
    let step_compiles = stats.compiles_of("gdn_conv1d_step_k4_f32")
        + stats.compiles_of("gdn_conv1d_step_k4_f32_p64");
    assert_eq!(step_compiles, 1, "must compile exactly once");
    let step_calls: u64 = stats
        .dispatches
        .iter()
        .filter(|d| d.1 == "gdn_conv1d_step_k4_f32" || d.1 == "gdn_conv1d_step_k4_f32_p64")
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
        assert!(try_flash_attn_contiguous(
            &q,
            &k,
            &v,
            FlashAttnConfig::causal(1.0 / 128.0f32.sqrt())
        )
        .unwrap()
        .is_none());
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
                9, 4, 1, 6, // req 1
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
                let kc =
                    Tensor::from_vec(k5, (num_blocks, n_kv, d / x_vec, block_size, x_vec), dev)
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
        let (batch, n_heads, n_kv, seq_q, seq_k, d) =
            (2usize, 8usize, 2usize, 7usize, 37usize, 128usize);
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
        let q = Tensor::from_vec(q_v, (batch, n_heads, seq_q, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let k = Tensor::from_vec(k_v, (batch, n_kv, seq_k, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let v = Tensor::from_vec(v_v, (batch, n_kv, seq_k, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let out = try_flash_attn_contiguous(&q, &k, &v, cfg)
            .unwrap()
            .expect("DSL flash_attn_contiguous should dispatch");
        let got = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let err = rel_err(&got, &ref_c);
        eprintln!("[{label} flash_contig {dtype:?}] d=128 sq=7 sk=37 rel={err:.2e}");
        assert!(err <= tol, "flash_contig err {err:.2e} > {tol:.2e}");

        // Long-context Split-KV decode & speculative-verify (seq_q in {1, 3}, seq_k = 512 -> num_splits = 8)
        for &(d, seq_q, win, cap) in &[
            (64usize, 1usize, 0usize, 0.0f32),
            (128usize, 1usize, 0usize, 0.0f32),
            (128usize, 3usize, 128usize, 30.0f32),
            (256usize, 1usize, 0usize, 0.0f32),
        ] {
            let (batch, n_heads, n_kv, seq_k) = (1usize, 8usize, 2usize, 512usize);
            let q_v = pseudo_vec(batch * n_heads * seq_q * d, (7771 + d + seq_q) as u64, 0.4);
            let k_v = pseudo_vec(batch * n_kv * seq_k * d, (7772 + d) as u64, 0.4);
            let v_v = pseudo_vec(batch * n_kv * seq_k * d, (7773 + d) as u64, 0.4);
            let cfg = FlashAttnConfig {
                scale: 1.0 / (d as f32).sqrt(),
                softcap: cap,
                causal_mode: 1,
                sliding_window: win,
                q_offset_override: 0,
            };
            let q = Tensor::from_vec(q_v, (batch, n_heads, seq_q, d), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let k = Tensor::from_vec(k_v, (batch, n_kv, seq_k, d), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let v = Tensor::from_vec(v_v, (batch, n_kv, seq_k, d), dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let q_rt = q.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let k_rt = k.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let v_rt = v.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bt_c = vec![0u32];
            let cl_c = vec![seq_k as u32];
            let ref_skv = flash_paged_ref(
                &q_rt, &k_rt, &v_rt, &bt_c, &cl_c, batch, n_heads, n_kv, seq_q, d, seq_k, 1, cfg,
            );
            let out_skv = try_flash_attn_contiguous(&q, &k, &v, cfg)
                .unwrap()
                .expect("DSL Split-KV flash_attn_contiguous should dispatch");
            let got_skv = out_skv
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let err_skv = rel_err(&got_skv, &ref_skv);
            eprintln!(
                "[{label} flash_split_kv {dtype:?}] d={d:3} sq={seq_q} sk={seq_k} win={win:3} cap={cap:2.0} rel={err_skv:.2e}"
            );
            if err_skv > tol {
                for qh in 0..(batch * n_heads * seq_q) {
                    let s = qh * d;
                    let e = rel_err(&got_skv[s..s + d], &ref_skv[s..s + d]);
                    eprintln!("  qh={qh:2} rel={e:.2e} got0..4={:?} ref0..4={:?}", &got_skv[s..s+4], &ref_skv[s..s+4]);
                }
            }
            assert!(err_skv <= tol, "flash_split_kv err {err_skv:.2e} > {tol:.2e}");

            // Zero-copy strided KV cache view (narrowed dim=2 from capacity=1024 to seq_k=512)
            let cap_seq = 1024usize;
            let k_cap = Tensor::from_vec(
                pseudo_vec(batch * n_kv * cap_seq * d, (7872 + d) as u64, 0.4),
                (batch, n_kv, cap_seq, d),
                dev,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let v_cap = Tensor::from_vec(
                pseudo_vec(batch * n_kv * cap_seq * d, (7873 + d) as u64, 0.4),
                (batch, n_kv, cap_seq, d),
                dev,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let k_zc = k_cap.narrow(2, 0, seq_k).unwrap();
            let v_zc = v_cap.narrow(2, 0, seq_k).unwrap();
            assert!(!k_zc.is_contiguous(), "narrowed GQA KV cache view must be strided");
            let k_zc_contig = k_zc.contiguous().unwrap();
            let v_zc_contig = v_zc.contiguous().unwrap();
            let out_zc = try_flash_attn_contiguous(&q, &k_zc, &v_zc, cfg)
                .unwrap()
                .expect("zero-copy strided Split-KV must dispatch");
            let out_zc_ref = try_flash_attn_contiguous(&q, &k_zc_contig, &v_zc_contig, cfg)
                .unwrap()
                .unwrap();
            let got_zc = out_zc.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let got_zc_ref = out_zc_ref.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let err_zc = rel_err(&got_zc, &got_zc_ref);
            if err_zc > 1e-6 {
                for qh in 0..(batch * n_heads * seq_q) {
                    let s = qh * d;
                    let e = rel_err(&got_zc[s..s + d], &got_zc_ref[s..s + d]);
                    eprintln!("  ZC qh={qh:2} rel={e:.2e} got0..4={:?} ref0..4={:?}", &got_zc[s..s+4], &got_zc_ref[s..s+4]);
                }
            }
            assert!(err_zc <= 1e-6, "zero-copy strided Split-KV mismatch {err_zc:.2e}");
        }
    }

    let stats = dsl::stats();
    let f_compiles = stats.compiles_of("flash_attn_paged_d128_f32")
        + stats.compiles_of("flash_attn_paged_d128_f32_p64");
    assert_eq!(f_compiles, 1, "flash_attn_paged_d128_f32 must compile exactly once");
    let skv_compiles = stats.compiles_of("flash_attn_paged_split_kv_d128_f32")
        + stats.compiles_of("flash_attn_paged_split_kv_d128_f32_p64");
    assert_eq!(skv_compiles, 1, "flash_attn_paged_split_kv_d128_f32 must compile exactly once");
    let red_compiles = stats.compiles_of("flash_attn_split_kv_reduce_d128_f32")
        + stats.compiles_of("flash_attn_split_kv_reduce_d128_f32_p64");
    assert_eq!(red_compiles, 1, "flash_attn_split_kv_reduce_d128_f32 must compile exactly once");
    let f_calls: u64 = stats
        .dispatches
        .iter()
        .filter(|d| {
            d.1 == "flash_attn_paged_d128_f32"
                || d.1 == "flash_attn_paged_d128_f32_p64"
                || d.1 == "flash_attn_paged_split_kv_d128_f32"
                || d.1 == "flash_attn_paged_split_kv_d128_f32_p64"
        })
        .map(|d| d.4)
        .sum();
    assert!(f_calls >= 6, "must have dispatched >= 6 d128 flash calls");

    dsl::set_flash_impl(QkImpl::Legacy);
}

fn fwht_ref_f64(x: &[f32], signs: Option<&[f32]>, b: usize, inverse: bool) -> Vec<f32> {
    let n_blocks = x.len() / b;
    let scale = 1.0f64 / (b as f64).sqrt();
    let mut out = vec![0.0f32; x.len()];
    let mut buf = vec![0.0f64; b];
    for blk in 0..n_blocks {
        let base = blk * b;
        for i in 0..b {
            let mut v = x[base + i] as f64;
            if let Some(s) = signs {
                if !inverse {
                    v *= s[(base + i) % s.len()] as f64;
                }
            }
            buf[i] = v;
        }
        let mut half = 1usize;
        while half < b {
            let step = half << 1;
            let mut start = 0usize;
            while start < b {
                for j in 0..half {
                    let a = buf[start + j];
                    let bv = buf[start + j + half];
                    buf[start + j] = a + bv;
                    buf[start + j + half] = a - bv;
                }
                start += step;
            }
            half = step;
        }
        for i in 0..b {
            let mut v = buf[i] * scale;
            if let Some(s) = signs {
                if inverse {
                    v *= s[(base + i) % s.len()] as f64;
                }
            }
            out[base + i] = v as f32;
        }
    }
    out
}

fn run_fwht_suite(dev: &Device, dtypes: &[DType], label: &str) {
    // 1. Default is DSL; explicit legacy returns Ok(None).
    assert_eq!(dsl::fwht_impl(), QkImpl::Dsl, "FWHT default must be dsl");
    dsl::set_fwht_impl(QkImpl::Legacy);
    let x_probe = Tensor::zeros((2, 1024), DType::F32, dev).unwrap();
    assert!(
        hanzo_ml::fwht::try_hadamard_fwht(&x_probe, None, 1024, false)
            .unwrap()
            .is_none(),
        "HANZO_FWHT_IMPL=legacy must return Ok(None)"
    );
    dsl::set_fwht_impl(QkImpl::Dsl);
    dsl::reset_stats();

    for &dt in dtypes {
        let tol = match dt {
            DType::F32 => 1e-4,
            DType::F16 => 5e-3,
            DType::BF16 => 4e-2,
            _ => unreachable!(),
        };
        for &b in &[64usize, 128, 256, 512, 1024, 2048] {
            let width = if b == 1024 { 5120 } else { b * 3 };
            let rows = 3usize;
            let x_data = pseudo_vec(rows * width, (b as u64) * 31 + 1, 1.0);
            let s_data: Vec<f32> = pseudo_vec(width, (b as u64) * 97 + 7, 1.0)
                .into_iter()
                .map(|v| if v >= 0.0 { 1.0 } else { -1.0 })
                .collect();
            let x = Tensor::from_vec(x_data.clone(), (1, rows, width), dev)
                .unwrap()
                .to_dtype(dt)
                .unwrap();
            let x_round: Vec<f32> = x
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let signs = Tensor::from_vec(s_data.clone(), width, dev).unwrap();

            for &inv in &[false, true] {
                let got_t = hanzo_ml::fwht::try_hadamard_fwht(&x, Some(&signs), b, inv)
                    .unwrap()
                    .expect("fwht DSL must handle supported block size");
                assert_eq!(got_t.dims(), x.dims());
                assert_eq!(got_t.dtype(), dt);
                let got: Vec<f32> = got_t
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                let want = fwht_ref_f64(&x_round, Some(&s_data), b, inv);
                let err = rel_err(&got, &want);
                assert!(
                    err <= tol,
                    "[{label}] fwht b={b} dt={dt:?} inv={inv} err={err:.3e} > tol={tol:.1e}"
                );
            }

            // Round-trip: inverse(forward(x)) == x and signless check
            let fwd = hanzo_ml::fwht::try_hadamard_fwht(&x, Some(&signs), b, false)
                .unwrap()
                .unwrap();
            let rt = hanzo_ml::fwht::try_hadamard_fwht(&fwd, Some(&signs), b, true)
                .unwrap()
                .unwrap();
            let rt_v: Vec<f32> = rt
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let rt_err = rel_err(&rt_v, &x_round);
            assert!(
                rt_err <= tol * 2.5,
                "[{label}] fwht roundtrip b={b} dt={dt:?} err={rt_err:.3e}"
            );

            let nosign = hanzo_ml::fwht::try_hadamard_fwht(&x, None, b, false)
                .unwrap()
                .unwrap();
            let ns_v: Vec<f32> = nosign
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let ns_want = fwht_ref_f64(&x_round, None, b, false);
            let ns_err = rel_err(&ns_v, &ns_want);
            assert!(
                ns_err <= tol,
                "[{label}] fwht signless b={b} dt={dt:?} err={ns_err:.3e}"
            );
        }
        eprintln!("[{label}] FWHT B=64..2048 dt={dt:?} fwd/inv/roundtrip/signless OK");
    }

    let stats = dsl::stats();
    let fwht_compiles = stats.compiles_of("fwht_hadamard_b1024_f32")
        + stats.compiles_of("fwht_hadamard_b1024_f32_p64");
    assert_eq!(
        fwht_compiles, 1,
        "fwht_hadamard_b1024_f32[_p64] must compile exactly once"
    );
    let calls: u64 = stats
        .dispatches
        .iter()
        .filter(|d| d.1 == "fwht_hadamard_b1024_f32" || d.1 == "fwht_hadamard_b1024_f32_p64")
        .map(|d| d.4)
        .sum();
    assert!(calls >= 5, "must have dispatched >= 5 FWHT b=1024 calls");
}

/// Regression guard for:
/// 1) Narrowed contiguous slices with non-zero `start_offset()` passed to DSL GDN / FWHT / Flash
///    (must materialize at offset 0 on Vulkan instead of silently reading element 0).
/// 2) In-place read-then-write (WAR) sequences within one unflushed GPU command batch.
fn run_offset_and_war_regression_suite(dev: &Device, label: &str) {
    dsl::set_gdn_impl(QkImpl::Dsl);
    dsl::set_fwht_impl(QkImpl::Dsl);
    dsl::set_flash_impl(QkImpl::Dsl);

    // 1. Narrowed non-zero start_offset on FWHT
    let x_big = Tensor::from_vec(pseudo_vec(3 * 128, 701, 0.8), (3, 128), dev).unwrap();
    let x_slice = x_big.narrow(0, 2, 1).unwrap();
    assert!(x_slice.is_contiguous() && x_slice.layout().start_offset() > 0);
    let x_packed = x_slice.force_contiguous().unwrap();
    let y_slice = hanzo_ml::fwht::try_hadamard_fwht(&x_slice, None, 128, false)
        .unwrap()
        .expect("FWHT DSL active")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let y_packed = hanzo_ml::fwht::try_hadamard_fwht(&x_packed, None, 128, false)
        .unwrap()
        .expect("FWHT DSL active")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let fwht_err = rel_err(&y_slice, &y_packed);
    assert!(
        fwht_err <= 1e-5,
        "[{label}] FWHT non-zero start_offset mismatch: {fwht_err:.2e}"
    );

    // 1b. Narrowed non-zero start_offset on FlashAttention contiguous SDPA
    let q_big = Tensor::from_vec(pseudo_vec(3 * 2 * 4 * 64, 702, 0.4), (3, 2, 4, 64), dev).unwrap();
    let k_big = Tensor::from_vec(pseudo_vec(3 * 2 * 16 * 64, 703, 0.4), (3, 2, 16, 64), dev).unwrap();
    let v_big = Tensor::from_vec(pseudo_vec(3 * 2 * 16 * 64, 704, 0.4), (3, 2, 16, 64), dev).unwrap();
    let q_sl = q_big.narrow(0, 1, 2).unwrap();
    let k_sl = k_big.narrow(0, 1, 2).unwrap();
    let v_sl = v_big.narrow(0, 1, 2).unwrap();
    assert!(q_sl.is_contiguous() && q_sl.layout().start_offset() > 0);
    let flash_cfg = FlashAttnConfig::causal(1.0 / 8.0);
    let flash_sl = try_flash_attn_contiguous(&q_sl, &k_sl, &v_sl, flash_cfg)
        .unwrap()
        .expect("Flash DSL active")
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let flash_pk = try_flash_attn_contiguous(
        &q_sl.force_contiguous().unwrap(),
        &k_sl.force_contiguous().unwrap(),
        &v_sl.force_contiguous().unwrap(),
        flash_cfg,
    )
    .unwrap()
    .expect("Flash DSL active")
    .flatten_all()
    .unwrap()
    .to_vec1::<f32>()
    .unwrap();
    let flash_off_err = rel_err(&flash_sl, &flash_pk);
    assert!(
        flash_off_err <= 1e-5,
        "[{label}] FlashAttention non-zero start_offset mismatch: {flash_off_err:.2e}"
    );

    // 2. Narrowed non-zero start_offset on GDN recurrence + WAR in-place state snapshot
    let (b, total_seq, nh, kd, vd) = (1usize, 5usize, 2usize, 64usize, 64usize);
    let q_all = Tensor::from_vec(pseudo_vec(b * total_seq * nh * kd, 711, 0.3), (b, total_seq, nh, kd), dev).unwrap();
    let k_all = Tensor::from_vec(pseudo_vec(b * total_seq * nh * kd, 712, 0.3), (b, total_seq, nh, kd), dev).unwrap();
    let v_all = Tensor::from_vec(pseudo_vec(b * total_seq * nh * vd, 713, 0.3), (b, total_seq, nh, vd), dev).unwrap();
    let g_all = Tensor::from_vec(
        pseudo_vec(b * total_seq * nh, 714, 0.25).into_iter().map(|x| x - 0.35).collect(),
        (b, total_seq, nh),
        dev,
    )
    .unwrap();
    let beta_all = Tensor::from_vec(
        pseudo_vec(b * total_seq * nh, 715, 0.4).into_iter().map(|x| (x + 1.0) * 0.5).collect(),
        (b, total_seq, nh),
        dev,
    )
    .unwrap();

    // Run full 5 steps in one shot
    let mut st_one = Tensor::zeros((b, nh, kd, vd), DType::F32, dev).unwrap();
    let y_one = try_recurrence(&q_all, &k_all, &v_all, &g_all, &beta_all, &mut st_one, true)
        .unwrap()
        .expect("GDN DSL active");

    // Run step-by-step using `.narrow(1, t, 1)` (non-zero start_offset for t >= 1!)
    // AND snapshot `st_step` via on-GPU `force_contiguous()` (no host readback flush!) before next in-place step (WAR test).
    let mut st_step = Tensor::zeros((b, nh, kd, vd), DType::F32, dev).unwrap();
    let mut y_steps = Vec::new();
    let mut snapshots = Vec::new();
    for t in 0..total_seq {
        let qt = q_all.narrow(1, t, 1).unwrap();
        let kt = k_all.narrow(1, t, 1).unwrap();
        let vt = v_all.narrow(1, t, 1).unwrap();
        let gt = g_all.narrow(1, t, 1).unwrap();
        let bt = beta_all.narrow(1, t, 1).unwrap();
        let yt = try_recurrence(&qt, &kt, &vt, &gt, &bt, &mut st_step, true)
            .unwrap()
            .expect("GDN DSL active");
        snapshots.push(st_step.force_contiguous().unwrap());
        y_steps.push(yt);
    }
    let y_cat = Tensor::cat(&y_steps, 1).unwrap();
    let y_err = rel_err(
        &y_one.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &y_cat.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
    let st_err = rel_err(
        &st_one.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &st_step.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
    // Snapshot taken at t=2 must match an independent 3-step run (proves no WAR overwrite by t=3,4!)
    let mut st_3 = Tensor::zeros((b, nh, kd, vd), DType::F32, dev).unwrap();
    let _ = try_recurrence(
        &q_all.narrow(1, 0, 3).unwrap().force_contiguous().unwrap(),
        &k_all.narrow(1, 0, 3).unwrap().force_contiguous().unwrap(),
        &v_all.narrow(1, 0, 3).unwrap().force_contiguous().unwrap(),
        &g_all.narrow(1, 0, 3).unwrap().force_contiguous().unwrap(),
        &beta_all.narrow(1, 0, 3).unwrap().force_contiguous().unwrap(),
        &mut st_3,
        true,
    )
    .unwrap()
    .expect("GDN DSL active");
    let war_err = rel_err(
        &snapshots[2].flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &st_3.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );

    // 3. Prefix-fork & speculative-rollback from snapshot[2]:
    // Fork A replays tokens 3..5 from snapshots[2] (must match 5-step st_one).
    // Fork B branches onto an alternate 2-token speculative continuation from snapshots[2]
    // (must match an independent 3+2 reference run, and leave snapshots[2] bit-identical).
    let mut fork_a = snapshots[2].force_contiguous().unwrap();
    let y_fork_a = try_recurrence(
        &q_all.narrow(1, 3, 2).unwrap(),
        &k_all.narrow(1, 3, 2).unwrap(),
        &v_all.narrow(1, 3, 2).unwrap(),
        &g_all.narrow(1, 3, 2).unwrap(),
        &beta_all.narrow(1, 3, 2).unwrap(),
        &mut fork_a,
        true,
    )
    .unwrap()
    .expect("GDN DSL active");
    let fork_a_st_err = rel_err(
        &fork_a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &st_one.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
    let fork_a_y_err = rel_err(
        &y_fork_a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &y_one.narrow(1, 3, 2).unwrap().force_contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );

    let q_alt = Tensor::from_vec(pseudo_vec(b * 2 * nh * kd, 811, 0.3), (b, 2, nh, kd), dev).unwrap();
    let k_alt = Tensor::from_vec(pseudo_vec(b * 2 * nh * kd, 812, 0.3), (b, 2, nh, kd), dev).unwrap();
    let v_alt = Tensor::from_vec(pseudo_vec(b * 2 * nh * vd, 813, 0.3), (b, 2, nh, vd), dev).unwrap();
    let g_alt = Tensor::from_vec(
        pseudo_vec(b * 2 * nh, 814, 0.25).into_iter().map(|x| x - 0.35).collect(),
        (b, 2, nh),
        dev,
    )
    .unwrap();
    let beta_alt = Tensor::from_vec(
        pseudo_vec(b * 2 * nh, 815, 0.4).into_iter().map(|x| (x + 1.0) * 0.5).collect(),
        (b, 2, nh),
        dev,
    )
    .unwrap();

    let mut fork_b = snapshots[2].force_contiguous().unwrap();
    let y_fork_b = try_recurrence(&q_alt, &k_alt, &v_alt, &g_alt, &beta_alt, &mut fork_b, true)
        .unwrap()
        .expect("GDN DSL active");
    let mut st_ref_b = st_3.force_contiguous().unwrap();
    let y_ref_b = try_recurrence(&q_alt, &k_alt, &v_alt, &g_alt, &beta_alt, &mut st_ref_b, true)
        .unwrap()
        .expect("GDN DSL active");
    let fork_b_st_err = rel_err(
        &fork_b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &st_ref_b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );
    let fork_b_y_err = rel_err(
        &y_fork_b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        &y_ref_b.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
    );

    // 4. Vulkan command-graph capture + repeated replay WAR test:
    // Record a graph that reads `st_graph` (via affine scale=1, bias=0 into `snap_before`)
    // and then mutates `st_graph` in-place via `try_recurrence`. Replay 3 times and verify
    // `snap_before` always reflects the pre-step state while `st_graph` advances to step r.
    #[cfg(feature = "vulkan")]
    if let Device::Vulkan(vk_dev) = dev {
        let q1 = q_all.narrow(1, 0, 1).unwrap().force_contiguous().unwrap();
        let k1 = k_all.narrow(1, 0, 1).unwrap().force_contiguous().unwrap();
        let v1 = v_all.narrow(1, 0, 1).unwrap().force_contiguous().unwrap();
        let g1 = g_all.narrow(1, 0, 1).unwrap().force_contiguous().unwrap();
        let b1 = beta_all.narrow(1, 0, 1).unwrap().force_contiguous().unwrap();

        let mut st_graph = st_3.force_contiguous().unwrap();
        if vk_dev.begin_graph_capture().is_ok() {
            let snap_before = st_graph.affine(1.0, 0.0).unwrap();
            let _y_g = try_recurrence(&q1, &k1, &v1, &g1, &b1, &mut st_graph, true)
                .unwrap()
                .expect("GDN DSL active in Vulkan graph capture");
            let graph = vk_dev.end_graph_capture().expect("end_graph_capture");

            let mut st_expected = st_3.force_contiguous().unwrap();
            for replay_idx in 1..=3 {
                let expected_before = st_expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let _ = try_recurrence(&q1, &k1, &v1, &g1, &b1, &mut st_expected, true)
                    .unwrap()
                    .expect("GDN DSL reference step");
                let expected_after = st_expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();

                graph.replay().expect("VkGraph::replay");
                let got_before = snap_before.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let got_after = st_graph.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let err_before = rel_err(&got_before, &expected_before);
                let err_after = rel_err(&got_after, &expected_after);
                assert!(
                    err_before <= 1e-4 && err_after <= 1e-4,
                    "[{label}] VkGraph replay #{replay_idx} WAR mismatch: before={err_before:.2e} after={err_after:.2e}"
                );
            }
        }
    }

    eprintln!(
        "[{label} offset+WAR regression] fwht={fwht_err:.2e} gdn_y={y_err:.2e} gdn_st={st_err:.2e} war_snap={war_err:.2e} fork_a=({fork_a_y_err:.2e},{fork_a_st_err:.2e}) fork_b=({fork_b_y_err:.2e},{fork_b_st_err:.2e})"
    );
    assert!(y_err <= 1e-4 && st_err <= 1e-4, "[{label}] GDN narrowed slice mismatch: y={y_err:.2e} st={st_err:.2e}");
    assert!(war_err <= 1e-4, "[{label}] WAR snapshot corrupted by subsequent in-place GDN step: {war_err:.2e}");
    assert!(
        fork_a_st_err <= 1e-4 && fork_a_y_err <= 1e-4 && fork_b_st_err <= 1e-4 && fork_b_y_err <= 1e-4,
        "[{label}] prefix-fork / speculative-rollback mismatch: fork_a=({fork_a_y_err:.2e},{fork_a_st_err:.2e}) fork_b=({fork_b_y_err:.2e},{fork_b_st_err:.2e})"
    );

    dsl::set_gdn_impl(QkImpl::Legacy);
    dsl::set_flash_impl(QkImpl::Legacy);
}

/// Verifies:
/// 1. 32K-context (`seq_k = 32768`, Zen6-27B GQA shape `n_heads = 24, n_kv = 2, d = 128`) Split-KV
///    numerical parity across `num_splits in [16, 32, 64, 128, 256]` vs `num_splits = 1`.
/// 2. Phase-aware and policy-based FlashAttention rollback on live GPU tensors.
fn run_split_kv_32k_and_policy_rollback_suite(dev: &Device, dtype: DType, label: &str) {
    use hanzo_ml::flash_attn::set_flash_split_kv_override;

    dsl::clear_op_overrides();
    dsl::set_dsl_policy_auto(false);
    dsl::set_flash_impl(QkImpl::Dsl);

    let (batch, n_heads, n_kv, seq_q, seq_k, d) =
        (1usize, 24usize, 2usize, 1usize, 32_768usize, 128usize);
    let q = Tensor::from_vec(pseudo_vec(batch * n_heads * seq_q * d, 32001, 0.35), (batch, n_heads, seq_q, d), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let k = Tensor::from_vec(pseudo_vec(batch * n_kv * seq_k * d, 32002, 0.35), (batch, n_kv, seq_k, d), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let v = Tensor::from_vec(pseudo_vec(batch * n_kv * seq_k * d, 32003, 0.35), (batch, n_kv, seq_k, d), dev)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let cfg = FlashAttnConfig::causal(1.0 / (d as f32).sqrt());

    set_flash_split_kv_override(Some(1));
    let ref_1 = try_flash_attn_contiguous(&q, &k, &v, cfg)
        .unwrap()
        .expect("32K single-pass reference must dispatch")
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let tol = match dtype {
        DType::F32 => 5e-4,
        DType::F16 => 5e-3,
        DType::BF16 => 2e-2,
        _ => unreachable!(),
    };
    let mut errs = Vec::new();
    for &ns in &[16usize, 32, 64, 128, 256] {
        set_flash_split_kv_override(Some(ns));
        let got = try_flash_attn_contiguous(&q, &k, &v, cfg)
            .unwrap()
            .expect("32K Split-KV must dispatch")
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let e = rel_err(&got, &ref_1);
        assert!(
            e <= tol,
            "[{label} 32K Split-KV {dtype:?}] num_splits={ns} rel_err={e:.3e} > tol={tol:.1e}"
        );
        errs.push(format!("{ns}:{e:.2e}"));
    }
    set_flash_split_kv_override(None);
    eprintln!(
        "[{label} 32K Split-KV parity {dtype:?}] d={d} sk={seq_k} rel_vs_s1=[{}]",
        errs.join(", ")
    );

    // Phase-aware rollback: decode forced to legacy while prefill remains DSL
    dsl::set_flash_impl(QkImpl::Dsl);
    dsl::set_flash_decode_impl(Some(QkImpl::Legacy));
    dsl::set_flash_prefill_impl(Some(QkImpl::Dsl));
    let q_dec = Tensor::zeros((1, 4, 1, 128), dtype, dev).unwrap();
    let q_pre = Tensor::zeros((1, 4, 8, 128), dtype, dev).unwrap();
    let k_s = Tensor::zeros((1, 2, 64, 128), dtype, dev).unwrap();
    let v_s = Tensor::zeros((1, 2, 64, 128), dtype, dev).unwrap();
    let k_l = Tensor::zeros((1, 2, 512, 128), dtype, dev).unwrap();
    let v_l = Tensor::zeros((1, 2, 512, 128), dtype, dev).unwrap();
    assert!(
        try_flash_attn_contiguous(&q_dec, &k_s, &v_s, cfg)
            .unwrap()
            .is_none(),
        "[{label}] flash decode rollback to legacy failed"
    );
    assert!(
        try_flash_attn_contiguous(&q_pre, &k_s, &v_s, cfg)
            .unwrap()
            .is_some(),
        "[{label}] flash prefill should remain DSL while decode rolled back"
    );

    // Policy auto mode: Metal enables short decode (seq_k <= 128) and rolls back at seq_k = 512;
    // non-Metal backends stay conservatively on Legacy until end-to-end qualified.
    dsl::clear_op_overrides();
    dsl::set_dsl_policy_auto(true);
    let short_auto = try_flash_attn_contiguous(&q_dec, &k_s, &v_s, cfg).unwrap();
    let long_auto = try_flash_attn_contiguous(&q_dec, &k_l, &v_l, cfg).unwrap();
    if dev.is_metal() {
        assert!(
            short_auto.is_some(),
            "[{label}] auto policy must enable short-context Metal DSL decode"
        );
        assert!(
            long_auto.is_none(),
            "[{label}] auto policy must preserve legacy Metal attention at seq_k=512"
        );
    } else {
        assert!(
            short_auto.is_none() && long_auto.is_none(),
            "[{label}] auto policy must keep non-Metal FlashAttention conservatively on Legacy"
        );
    }

    // Restore clean conservative state
    dsl::clear_op_overrides();
    dsl::set_dsl_policy_auto(false);
    set_flash_split_kv_override(None);
}

fn run_split_kv_partition_sweep(dev: &Device, dtype: DType, label: &str) {
    use hanzo_ml::flash_attn::{choose_flash_split_kv_for, set_flash_split_kv_override};
    use std::time::Instant;

    dsl::clear_op_overrides();
    dsl::set_flash_impl(QkImpl::Dsl);

    let backend = dsl::QkBackend::from_device(dev);
    let elem_bytes = dtype.size_in_bytes();
    let (batch, n_heads, n_kv, seq_q) = (1usize, 24usize, 2usize, 1usize);
    let splits_grid = [1usize, 8, 16, 32, 64, 128, 256];
    let tol = match dtype {
        DType::F32 => 5e-4,
        DType::F16 => 5e-3,
        DType::BF16 => 2e-2,
        _ => unreachable!(),
    };

    eprintln!(
        "\n=== [{label} Split-KV Partition Sweep | {dtype:?} | B={batch} H={n_heads} H_kv={n_kv} seq_q={seq_q}] ==="
    );
    eprintln!(
        "| d | seq_k | auto_s | s=1 (us) | s=8 (us) | s=16 (us) | s=32 (us) | s=64 (us) | s=128 (us) | s=256 (us) | best_s | best_us | best_GB/s | max_rel |"
    );
    eprintln!(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );

    for &d in &[64usize, 128, 256] {
        let q = Tensor::from_vec(
            pseudo_vec(batch * n_heads * seq_q * d, (9100 + d) as u64, 0.35),
            (batch, n_heads, seq_q, d),
            dev,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let cfg = FlashAttnConfig::causal(1.0 / (d as f32).sqrt());

        for &seq_k in &[512usize, 4096, 8192, 16384, 32768] {
            set_flash_split_kv_override(None);
            let auto_s = choose_flash_split_kv_for(backend, batch, n_heads, seq_q, d, seq_k, 0);

            let k = Tensor::from_vec(
                pseudo_vec(batch * n_kv * seq_k * d, (9200 + d + seq_k) as u64, 0.35),
                (batch, n_kv, seq_k, d),
                dev,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let v = Tensor::from_vec(
                pseudo_vec(batch * n_kv * seq_k * d, (9300 + d + seq_k) as u64, 0.35),
                (batch, n_kv, seq_k, d),
                dev,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

            // Reference output with num_splits = 1
            set_flash_split_kv_override(Some(1));
            let ref_out = try_flash_attn_contiguous(&q, &k, &v, cfg)
                .unwrap()
                .expect("Split-KV sweep dispatch")
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let mut med_us = Vec::with_capacity(splits_grid.len());
            let mut max_rel = 0.0f32;
            let mut best_s = 1usize;
            let mut best_us = f64::INFINITY;

            for &ns in &splits_grid {
                set_flash_split_kv_override(Some(ns));
                // Correctness check vs s=1
                let check_out = try_flash_attn_contiguous(&q, &k, &v, cfg)
                    .unwrap()
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let err = rel_err(&check_out, &ref_out);
                assert!(
                    err <= tol,
                    "[{label} sweep] d={d} sk={seq_k} s={ns} rel={err:.3e} > tol={tol:.1e}"
                );
                if err > max_rel {
                    max_rel = err;
                }

                // Warmup + amortized multi-dispatch samples (avoids per-call sync/OS jitter
                // and mirrors multi-layer transformer decode where layers queue back-to-back).
                let warmup = if ns == 1 && seq_k >= 16384 { 1 } else { 4 };
                let (batches, reps) = if ns == 1 && seq_k >= 16384 {
                    (3usize, 1usize)
                } else if ns == 1 && seq_k >= 4096 {
                    (5usize, 2usize)
                } else if seq_k >= 16384 {
                    (5usize, 8usize)
                } else {
                    (5usize, 16usize)
                };
                for _ in 0..warmup {
                    let _ = try_flash_attn_contiguous(&q, &k, &v, cfg).unwrap().unwrap();
                }
                dev.synchronize().unwrap();

                let mut samples = Vec::with_capacity(batches);
                for _ in 0..batches {
                    let t0 = Instant::now();
                    for _ in 0..reps {
                        let _ = try_flash_attn_contiguous(&q, &k, &v, cfg).unwrap().unwrap();
                    }
                    dev.synchronize().unwrap();
                    samples.push(t0.elapsed().as_secs_f64() * 1e6 / (reps as f64));
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = samples[samples.len() / 2];
                med_us.push(med);
                if med < best_us {
                    best_us = med;
                    best_s = ns;
                }
            }

            let kv_bytes = (2 * batch * n_kv * seq_k * d * elem_bytes) as f64;
            let best_gb_s = kv_bytes / (best_us * 1e3);
            eprintln!(
                "| {d} | {seq_k} | {auto_s} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {best_s} | {:.1} | {:.1} | {:.2e} |",
                med_us[0], med_us[1], med_us[2], med_us[3], med_us[4], med_us[5], med_us[6],
                best_us, best_gb_s, max_rel
            );
        }
    }

    set_flash_split_kv_override(None);
    dsl::set_flash_impl(QkImpl::Legacy);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    run_gdn_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "CUDA");
    run_flash_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "CUDA");
    run_fwht_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "CUDA");
    run_offset_and_war_regression_suite(&dev, "CUDA");
    run_split_kv_32k_and_policy_rollback_suite(&dev, DType::BF16, "CUDA");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_split_kv_partition_sweep() {
    let dev = Device::new_cuda(0).expect("CUDA device 0");
    run_split_kv_partition_sweep(&dev, DType::BF16, "CUDA");
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_vulkan(0).expect("Vulkan device 0");
    run_gdn_suite(&dev, &[DType::F32], "Vulkan");
    run_flash_suite(&dev, &[DType::F32], "Vulkan");
    run_fwht_suite(&dev, &[DType::F32], "Vulkan");
    run_offset_and_war_regression_suite(&dev, "Vulkan");
    run_split_kv_32k_and_policy_rollback_suite(&dev, DType::F32, "Vulkan");
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore]
fn vulkan_split_kv_partition_sweep() {
    let dev = Device::new_vulkan(0).expect("Vulkan device 0");
    run_split_kv_partition_sweep(&dev, DType::F32, "Vulkan");
}

#[cfg(feature = "metal")]
#[test]
fn metal_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_metal(0).expect("Metal device 0");
    run_gdn_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "Metal");
    run_flash_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "Metal");
    run_fwht_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "Metal");
    run_offset_and_war_regression_suite(&dev, "Metal");
    run_split_kv_32k_and_policy_rollback_suite(&dev, DType::BF16, "Metal");
}

#[cfg(feature = "metal")]
#[test]
#[ignore]
fn metal_split_kv_partition_sweep() {
    let dev = Device::new_metal(0).expect("Metal device 0");
    run_split_kv_partition_sweep(&dev, DType::BF16, "Metal");
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_gdn_and_flash_paged_prod_dsl() {
    let dev = Device::new_rocm(0).expect("ROCm device 0");
    run_gdn_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "ROCm");
    run_flash_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "ROCm");
    run_fwht_suite(&dev, &[DType::F32, DType::F16, DType::BF16], "ROCm");
    run_offset_and_war_regression_suite(&dev, "ROCm");
    run_split_kv_32k_and_policy_rollback_suite(&dev, DType::BF16, "ROCm");
}

#[cfg(feature = "rocm")]
#[test]
#[ignore]
fn rocm_split_kv_partition_sweep() {
    let dev = Device::new_rocm(0).expect("ROCm device 0");
    run_split_kv_partition_sweep(&dev, DType::BF16, "ROCm");
}
