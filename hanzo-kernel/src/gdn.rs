//! Gated-DeltaNet (GDN) linear-attention ops in the DSL: one Rust source -> every backend.
//!
//! Qwen3.5 / Qwen3.6 / AgentWorld (`qwen3_5_moe` arch) are HYBRID models: most layers are GDN
//! linear-attention, not softmax attention. The GDN block is three ops -- a causal depthwise conv1d,
//! a fused gating, and a recurrent gated delta-rule scan. On Vulkan these are FUSED (three `.comp`
//! shaders); on CUDA/ROCm they are ops-composed (many tiny tensor launches), which is a big part of
//! hanzo's prefill gap + launch-bound decode versus llama.cpp on these models. Writing the three ops
//! ONCE in the DSL lowers them to CUDA/ROCm/Vulkan/Metal from one source, so the same fused kernel
//! serves every backend and the "same op, N impls, N numeric behaviors" fork cannot occur.
//!
//! Reference impls (bit-exact target): hanzo-ml `gdn_conv_full.comp`, `gdn_gating.comp`,
//! `gdn_chunked.comp`, and the engine's portable f32 scan `recurrence_portable` (models/gdn.rs).
//! Every op is f32 end-to-end (GDN runs in f32 to avoid bf16/f32 boundary drift), matching the shaders.
//!
//! Design (decomplected): the recurrence is a SEQUENTIAL scan over the sequence but FULLY PARALLEL
//! over `(batch*head, v_column)` -- each output column owns an independent state column `s[k_dim]` and
//! scans the whole sequence with no cross-thread communication (see the engine's `gdn_step_scalar`).
//! `gdn_scan` is exactly that: one thread per `(bh, v)` column, private state array, sequential over
//! `seq`. It fuses the ENTIRE recurrence into ONE launch -- the actual cure for the ops-composed
//! CUDA/ROCm launch storm -- and, needing no shared memory or barriers, it lowers to EVERY backend
//! including CPU (so it is CPU-bit-exact testable, the whole point). The Vulkan `gdn_chunked.comp`
//! chunk-tiling (shared-mem K reuse + prefix-sum + forward-substitution) is a GPU-only *performance*
//! reorganization of the identical math; it needs cooperative thread blocks, which cubecl-cpu does not
//! host, so it cannot be CPU-gated -- it stays a peak-path specialization, and `gdn_scan` is the
//! portable, correctness-proven primitive that already closes the launch gap.

use crate::prelude::*;

// ============================== 1. Causal depthwise conv1d (+ SiLU) ==============================

/// Causal depthwise conv1d over a full sequence, then SiLU -- the GDN qkv mixer. Mirrors
/// `gdn_conv_full.comp` / cuda `causal_conv1d_full_kernel`. One thread per output element
/// `(b, ch, pos)`: a left-padded causal window of `k` taps ending at `pos`, dotted with the channel's
/// weight row, then `silu(acc) = acc / (1 + exp(-acc))`. Positions before the start zero-pad.
///
/// `x`, `out`: `[batch, conv_dim, seq_len]` row-major; `w`: `[conv_dim, k]`. `conv_dim`/`seq_len` are
/// RUNTIME (`dims`), so one compiled kernel serves every model's conv width and any prompt length;
/// only `k` (the kernel size, model-fixed at 4) is comptime so the tap loop unrolls.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn gdn_conv1d<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    out: &mut Array<F>,
    dims: &Array<u32>, // [conv_dim, seq_len]
    #[comptime] k: usize,
) {
    let gid = ABSOLUTE_POS;
    if gid < out.len() {
        let conv_dim = dims[0] as usize;
        let seq_len = dims[1] as usize;
        let pos = gid % seq_len;
        let row = gid / seq_len; // b * conv_dim + ch
        let ch = row % conv_dim;
        let x_base = row * seq_len;
        let w_base = ch * k;

        let mut acc = F::new(0.0);
        for i in 0..k {
            // src_pos = pos - (k-1) + i; contribute only when it is in-range (causal zero-pad).
            if pos + i >= k - 1 {
                let src = pos + i - (k - 1);
                acc += x[x_base + src] * w[w_base + i];
            }
        }
        // SiLU (matches the shader's acc * sigmoid(acc), sigmoid = 1/(1+exp(-acc))).
        out[x_base + pos] = acc / (F::new(1.0) + (-acc).exp());
    }
}

// ================================= 2. Fused GDN gating ==========================================

/// Fused GDN gating -- mirrors `gdn_gating.comp` / cuda `fused_gdn_gating_kernel`. Per element:
///   `beta = sigmoid(b)`,  `g = -exp(a_log) * softplus(a + dt_bias)`,  `softplus(x) = log(1 + exp(x))`.
/// `a_log` and `dt_bias` are per-head (indexed `idx % num_heads`), broadcast over the leading dims.
///
/// `a_log` here is the RAW `A_log` (the shader convention). The GGUF `ssm_a` instead stores the
/// precomputed `-exp(A_log)`; a caller holding that passes it as `a_log` with the sign folded, i.e.
/// use `gdn_gating_ref`'s `neg_exp_a_log=true` path -- one op, both conventions, no second kernel.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn gdn_gating<F: Float>(
    b_in: &Array<F>,
    a_in: &Array<F>,
    a_log: &Array<F>,
    dt_bias: &Array<F>,
    beta_out: &mut Array<F>,
    g_out: &mut Array<F>,
    dims: &Array<u32>, // [num_heads]
) {
    let idx = ABSOLUTE_POS;
    if idx < beta_out.len() {
        let num_heads = dims[0] as usize;
        let head = idx % num_heads;

        let beta = F::new(1.0) / (F::new(1.0) + (-b_in[idx]).exp());
        let softplus = (F::new(1.0) + (a_in[idx] + dt_bias[head]).exp()).ln();
        let g = -(a_log[head].exp()) * softplus;

        beta_out[idx] = beta;
        g_out[idx] = g;
    }
}

// ========================= 3. Recurrent gated delta-rule scan (the recurrence) ==================

/// Recurrent gated delta-rule scan -- the linear-attention core, portable formulation of
/// `gdn_chunked.comp` / the engine's `recurrence_portable`. One thread per `(bh, v_column)`: it holds
/// its state column `s[k_dim]` in a private register array and scans the sequence in order, with NO
/// shared memory and NO cross-thread carry (each column is independent). Per step `t`:
///   `s *= exp(g_t)`;  `kv = Σ_j s[j]*k_t[j]`;  `δ = (v_t - kv)*β_t`;
///   `s[j] += k_t[j]*δ`;  `y_t = Σ_j s[j]*q_t[j]`.
/// This is the entire recurrence in ONE launch (the fix for the ops-composed launch storm) and is
/// bit-identical to the engine's portable scan / `gdn_step_scalar`.
///
/// Layout matches the fused-kernel contract (`recurrence_flatten`): `q`,`k`: `[bh, seq, k_dim]`;
/// `v`,`out`: `[bh, seq, v_dim]`; `g`,`beta`: `[bh, seq]`; `state`: `[bh, k_dim, v_dim]`, updated
/// IN PLACE. `q` arrives PRE-SCALED by `1/sqrt(k_dim)` (the caller applies it, exactly as the CUDA /
/// Metal fused kernels expect) -- the scale is a projection-pipeline concern, kept out of the kernel.
/// `bh`/`seq` are runtime (`dims`); `k_dim`/`v_dim` are comptime (per-head, model-fixed) so the state
/// array sizes and the inner reductions lower cleanly.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn gdn_scan<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    g: &Array<F>,
    beta: &Array<F>,
    state: &mut Array<F>,
    out: &mut Array<F>,
    dims: &Array<u32>, // [bh, seq]
    #[comptime] k_dim: usize,
    #[comptime] v_dim: usize,
) {
    let tid = ABSOLUTE_POS; // over bh * v_dim
    let bh = dims[0] as usize;
    let seq = dims[1] as usize;
    if tid < bh * v_dim {
        let b = tid / v_dim;
        let vi = tid % v_dim;
        let qk_bh = b * seq * k_dim;
        let v_bh = b * seq * v_dim;
        let gb_bh = b * seq;
        let state_base = b * k_dim * v_dim;

        // Private state column s[k_dim] (registers), loaded from the recurrent state.
        let mut s = Array::<F>::new(k_dim);
        for j in 0..k_dim {
            s[j] = state[state_base + j * v_dim + vi];
        }

        for t in 0..seq {
            let decay = g[gb_bh + t].exp();
            let beta_t = beta[gb_bh + t];
            let v_t = v[v_bh + t * v_dim + vi];

            // Decay the state, then read the pre-update memory kv = Σ_j (s[j]*decay) * k_t[j].
            let mut kv_mem = F::new(0.0);
            for j in 0..k_dim {
                let sj = s[j] * decay;
                s[j] = sj;
                kv_mem += sj * k[qk_bh + t * k_dim + j];
            }
            let delta = (v_t - kv_mem) * beta_t;

            // Rank-1 update s[j] += k_t[j]*δ, then read out y_t = Σ_j s[j]*q_t[j] (post-update).
            let mut y_t = F::new(0.0);
            for j in 0..k_dim {
                let sj = s[j] + k[qk_bh + t * k_dim + j] * delta;
                s[j] = sj;
                y_t += sj * q[qk_bh + t * k_dim + j];
            }
            out[v_bh + t * v_dim + vi] = y_t;
        }

        // Carry the advanced state back for the next chunk / decode step.
        for j in 0..k_dim {
            state[state_base + j * v_dim + vi] = s[j];
        }
    }
}

// ==================================== Host launchers ============================================

/// Host launch for `gdn_conv1d`. `x`: `[batch, conv_dim, seq_len]`, `w`: `[conv_dim, k]`.
pub fn gdn_conv1d_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    batch: usize,
    conv_dim: usize,
    seq_len: usize,
    k: usize,
) -> Vec<f32> {
    let total = batch * conv_dim * seq_len;
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let dh = client.create_from_slice(u32::as_bytes(&[conv_dim as u32, seq_len as u32]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; total]));
    let block = 64u32;
    let grid = (total as u32).div_ceil(block);
    unsafe {
        gdn_conv1d::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(oh.clone(), total),
            ArrayArg::from_raw_parts(dh.clone(), 2),
            k,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Host launch for `gdn_gating`. `b`/`a`: `[total]`; `a_log`/`dt_bias`: `[num_heads]`. Returns `(beta, g)`.
pub fn gdn_gating_run<R: Runtime>(
    client: &ComputeClient<R>,
    b: &[f32],
    a: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    num_heads: usize,
) -> (Vec<f32>, Vec<f32>) {
    let total = b.len();
    let bh = client.create_from_slice(f32::as_bytes(b));
    let ah = client.create_from_slice(f32::as_bytes(a));
    let alh = client.create_from_slice(f32::as_bytes(a_log));
    let dth = client.create_from_slice(f32::as_bytes(dt_bias));
    let betah = client.create_from_slice(f32::as_bytes(&vec![0.0f32; total]));
    let gh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; total]));
    let dh = client.create_from_slice(u32::as_bytes(&[num_heads as u32]));
    let block = 64u32;
    let grid = (total as u32).div_ceil(block);
    unsafe {
        gdn_gating::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(bh.clone(), b.len()),
            ArrayArg::from_raw_parts(ah.clone(), a.len()),
            ArrayArg::from_raw_parts(alh.clone(), a_log.len()),
            ArrayArg::from_raw_parts(dth.clone(), dt_bias.len()),
            ArrayArg::from_raw_parts(betah.clone(), total),
            ArrayArg::from_raw_parts(gh.clone(), total),
            ArrayArg::from_raw_parts(dh.clone(), 1),
        );
    }
    let beta = f32::from_bytes(&client.read_one_unchecked(betah)).to_vec();
    let g = f32::from_bytes(&client.read_one_unchecked(gh)).to_vec();
    (beta, g)
}

/// Host launch for `gdn_scan`. `q`/`k`: `[bh, seq, k_dim]` (`q` pre-scaled by `1/sqrt(k_dim)`);
/// `v`: `[bh, seq, v_dim]`; `g`/`beta`: `[bh, seq]`; `state`: `[bh, k_dim, v_dim]`. Returns
/// `(out [bh, seq, v_dim], new_state [bh, k_dim, v_dim])`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_run<R: Runtime>(
    client: &ComputeClient<R>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &[f32],
    bh: usize,
    seq: usize,
    k_dim: usize,
    v_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let out_len = bh * seq * v_dim;
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k));
    let vh = client.create_from_slice(f32::as_bytes(v));
    let gh = client.create_from_slice(f32::as_bytes(g));
    let betah = client.create_from_slice(f32::as_bytes(beta));
    let sh = client.create_from_slice(f32::as_bytes(state));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; out_len]));
    let dh = client.create_from_slice(u32::as_bytes(&[bh as u32, seq as u32]));
    let block = 64u32;
    let grid = ((bh * v_dim) as u32).div_ceil(block);
    unsafe {
        gdn_scan::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), q.len()),
            ArrayArg::from_raw_parts(kh.clone(), k.len()),
            ArrayArg::from_raw_parts(vh.clone(), v.len()),
            ArrayArg::from_raw_parts(gh.clone(), g.len()),
            ArrayArg::from_raw_parts(betah.clone(), beta.len()),
            ArrayArg::from_raw_parts(sh.clone(), state.len()),
            ArrayArg::from_raw_parts(oh.clone(), out_len),
            ArrayArg::from_raw_parts(dh.clone(), 2),
            k_dim,
            v_dim,
        );
    }
    let out = f32::from_bytes(&client.read_one_unchecked(oh)).to_vec();
    let new_state = f32::from_bytes(&client.read_one_unchecked(sh)).to_vec();
    (out, new_state)
}

// ================================== CPU oracles (references) ====================================

/// CPU oracle for `gdn_conv1d`: causal depthwise conv1d + SiLU, the trusted reference the DSL kernel
/// is gated against. Mirrors `gdn_conv_full.comp` element-for-element.
pub fn gdn_conv1d_ref(
    x: &[f32],
    w: &[f32],
    batch: usize,
    conv_dim: usize,
    seq_len: usize,
    k: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * conv_dim * seq_len];
    for b in 0..batch {
        for ch in 0..conv_dim {
            let x_base = (b * conv_dim + ch) * seq_len;
            let w_base = ch * k;
            for pos in 0..seq_len {
                let mut acc = 0.0f32;
                for i in 0..k {
                    let src = pos as isize - (k as isize - 1) + i as isize;
                    if src >= 0 {
                        acc += x[x_base + src as usize] * w[w_base + i];
                    }
                }
                out[x_base + pos] = acc / (1.0 + (-acc).exp()); // silu
            }
        }
    }
    out
}

/// CPU oracle for `gdn_gating`. `neg_exp_a_log=false` matches the shader (`a_log` is raw `A_log`, so
/// `g = -exp(a_log)*softplus`); `true` matches the GGUF `ssm_a` convention (`a_log` already holds
/// `-exp(A_log)`, so `g = a_log*softplus`). Returns `(beta, g)`.
pub fn gdn_gating_ref(
    b: &[f32],
    a: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    num_heads: usize,
    neg_exp_a_log: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n = b.len();
    let mut beta = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    for idx in 0..n {
        let head = idx % num_heads;
        beta[idx] = 1.0 / (1.0 + (-b[idx]).exp());
        let softplus = (1.0 + (a[idx] + dt_bias[head]).exp()).ln();
        let coeff = if neg_exp_a_log { a_log[head] } else { -(a_log[head].exp()) };
        g[idx] = coeff * softplus;
    }
    (beta, g)
}

/// CPU oracle for `gdn_scan` -- the engine's portable gated delta-rule scan (`recurrence_portable` /
/// `gdn_step_scalar`), per `(bh, v)` column, sequential over `seq`. `state` is updated in place;
/// returns `out [bh, seq, v_dim]`. `q` is pre-scaled by the caller (same contract as the kernel).
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    bh: usize,
    seq: usize,
    k_dim: usize,
    v_dim: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; bh * seq * v_dim];
    for b in 0..bh {
        let qk_bh = b * seq * k_dim;
        let v_bh = b * seq * v_dim;
        let gb_bh = b * seq;
        let state_base = b * k_dim * v_dim;
        for vi in 0..v_dim {
            let mut s = vec![0.0f32; k_dim];
            for j in 0..k_dim {
                s[j] = state[state_base + j * v_dim + vi];
            }
            for t in 0..seq {
                let decay = g[gb_bh + t].exp();
                let beta_t = beta[gb_bh + t];
                let v_t = v[v_bh + t * v_dim + vi];
                let mut kv_mem = 0.0f32;
                for j in 0..k_dim {
                    let sj = s[j] * decay;
                    s[j] = sj;
                    kv_mem += sj * k[qk_bh + t * k_dim + j];
                }
                let delta = (v_t - kv_mem) * beta_t;
                let mut y_t = 0.0f32;
                for j in 0..k_dim {
                    let sj = s[j] + k[qk_bh + t * k_dim + j] * delta;
                    s[j] = sj;
                    y_t += sj * q[qk_bh + t * k_dim + j];
                }
                out[v_bh + t * v_dim + vi] = y_t;
            }
            for j in 0..k_dim {
                state[state_base + j * v_dim + vi] = s[j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
            .fold(0.0, f32::max)
    }

    // ---- causal conv1d (k=4), realistic GDN conv widths ----
    #[cfg(feature = "cpu")]
    #[test]
    fn gdn_conv1d_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        // (batch, conv_dim, seq_len). conv_dim = key_dim*2 + value_dim for the model; k=4 always.
        // Includes seq_len < k (pure left-pad) and seq_len >> k (steady state).
        for &(batch, conv_dim, seq_len) in &[(1usize, 512usize, 48usize), (1, 64, 3), (2, 128, 32)] {
            let x = rnd(batch * conv_dim * seq_len, 0x1111 + conv_dim as u64);
            let w = rnd(conv_dim * 4, 0x2222 + seq_len as u64);
            let got = gdn_conv1d_run::<CpuRuntime>(&c, &x, &w, batch, conv_dim, seq_len, 4);
            let want = gdn_conv1d_ref(&x, &w, batch, conv_dim, seq_len, 4);
            let rel = max_rel(&want, &got);
            eprintln!("[gdn_conv1d CPU] b{batch} c{conv_dim} s{seq_len} k4 max_rel={rel:.2e}");
            assert!(rel < 2e-3, "gdn_conv1d b{batch} c{conv_dim} s{seq_len} max_rel {rel}");
        }
    }

    // ---- fused gating: beta = sigmoid(b), g = -exp(a_log)*softplus(a+dt_bias) ----
    #[cfg(feature = "cpu")]
    #[test]
    fn gdn_gating_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        // total = batch*seq*num_heads; a_log/dt_bias per head. Shader convention (raw A_log).
        for &(num_heads, tokens) in &[(32usize, 40usize), (16, 1), (8, 128)] {
            let total = tokens * num_heads;
            let b = rnd(total, 0x3333 + num_heads as u64);
            let a = rnd(total, 0x4444 + tokens as u64);
            let a_log = rnd(num_heads, 0x5555); // raw A_log
            let dt_bias = rnd(num_heads, 0x6666);
            let (gb, gg) = gdn_gating_run::<CpuRuntime>(&c, &b, &a, &a_log, &dt_bias, num_heads);
            let (wb, wg) = gdn_gating_ref(&b, &a, &a_log, &dt_bias, num_heads, false);
            let (rb, rg) = (max_rel(&wb, &gb), max_rel(&wg, &gg));
            eprintln!("[gdn_gating CPU] h{num_heads} tok{tokens}  beta_rel={rb:.2e} g_rel={rg:.2e}");
            assert!(rb < 2e-3 && rg < 2e-3, "gdn_gating h{num_heads} beta={rb} g={rg}");
        }
    }

    // ---- recurrent gated delta-rule scan (the hard one) ----
    #[cfg(feature = "cpu")]
    #[test]
    fn gdn_scan_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        // (bh, seq, k_dim, v_dim). bh = batch*num_v_heads; k_dim=head_k_dim; v_dim=head_v_dim.
        // Covers decode (seq=1), a real prefill chunk (seq=64), the shipping head dim (128), and odd
        // non-square dims.
        for &(bh, seq, k_dim, v_dim) in &[
            (4usize, 32usize, 64usize, 64usize),
            (2, 64, 128, 128),
            (8, 1, 32, 48),
            (3, 40, 48, 32),
        ] {
            let scale = 1.0f32 / (k_dim as f32).sqrt();
            // q pre-scaled by 1/sqrt(k_dim), matching the kernel/engine contract.
            let q: Vec<f32> = rnd(bh * seq * k_dim, 0x7001 + k_dim as u64)
                .iter()
                .map(|x| x * scale)
                .collect();
            let k = rnd(bh * seq * k_dim, 0x7002 + seq as u64);
            let v = rnd(bh * seq * v_dim, 0x7003 + v_dim as u64);
            // g < 0 (decay in (0,1]); beta in [0,1] -- the physical GDN gate ranges.
            let g: Vec<f32> = rnd(bh * seq, 0x7004).iter().map(|x| x * 0.5 - 0.5).collect();
            let beta: Vec<f32> = rnd(bh * seq, 0x7005).iter().map(|x| (x + 1.0) * 0.5).collect();
            let state0 = rnd(bh * k_dim * v_dim, 0x7006);

            let (got_out, got_state) =
                gdn_scan_run::<CpuRuntime>(&c, &q, &k, &v, &g, &beta, &state0, bh, seq, k_dim, v_dim);
            let mut ref_state = state0.clone();
            let want_out =
                gdn_scan_ref(&q, &k, &v, &g, &beta, &mut ref_state, bh, seq, k_dim, v_dim);
            let ro = max_rel(&want_out, &got_out);
            let rs = max_rel(&ref_state, &got_state);
            eprintln!(
                "[gdn_scan CPU] bh{bh} s{seq} k{k_dim} v{v_dim}  out_rel={ro:.2e} state_rel={rs:.2e}"
            );
            assert!(ro < 2e-3 && rs < 2e-3, "gdn_scan bh{bh} s{seq} out={ro} state={rs}");
        }
    }
}
