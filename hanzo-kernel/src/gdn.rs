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
use crate::tune::{Config, Evaluator, Space, Verdict};

// ============================== 1. Causal depthwise conv1d (+ SiLU) ==============================

/// Causal depthwise conv1d over a full sequence, then SiLU -- the GDN qkv mixer. Mirrors
/// `gdn_conv_full.comp` / cuda `causal_conv1d_full_kernel`. One thread per output element
/// `(b, ch, pos)`: a left-padded causal window of `k` taps ending at `pos`, dotted with the channel's
/// weight row, then `silu(acc) = acc / (1 + exp(-acc))`. Positions before the start zero-pad.
///
/// `x`, `out`: `[batch, conv_dim, seq_len]` row-major; `w`: `[conv_dim, k]`. `conv_dim`/`seq_len` are
/// RUNTIME (`dims`), so one compiled kernel serves every model's conv width and any prompt length;
/// only `k` (the kernel size, model-fixed at 4) is comptime so the tap loop unrolls.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu, rocm), unchecked)]
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

        let mut acc = 0.0f32;
        for i in 0..k {
            // src_pos = pos - (k-1) + i; contribute only when it is in-range (causal zero-pad).
            if pos + i >= k - 1 {
                let src = pos + i - (k - 1);
                acc += f32::cast_from(x[x_base + src]) * f32::cast_from(w[w_base + i]);
            }
        }
        // SiLU (matches the shader's acc * sigmoid(acc), sigmoid = 1/(1+exp(-acc))).
        out[x_base + pos] = F::cast_from(acc / (1.0f32 + (-acc).exp()));
    }
}

/// Single-step causal depthwise conv1d + SiLU (decode, seq_len == 1).
///
/// Shifts `conv_state [batch, conv_dim, k]` left by 1, inserts `x [batch, conv_dim]` at index
/// `k - 1`, dots against `w [conv_dim, k]` in f32, and writes SiLU output to `out [batch, conv_dim]`.
/// `dims = [batch, conv_dim]`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu, rocm), unchecked)]
pub fn gdn_conv1d_step<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    conv_state: &mut Array<F>,
    out: &mut Array<F>,
    dims: &Array<u32>, // [batch, conv_dim]
    #[comptime] k: usize,
) {
    let gid = ABSOLUTE_POS;
    let batch = dims[0] as usize;
    let conv_dim = dims[1] as usize;
    if gid < batch * conv_dim {
        let ch = gid % conv_dim;
        let cs_base = gid * k;
        let w_base = ch * k;

        let mut acc = 0.0f32;
        for i in 0..k - 1 {
            let v = conv_state[cs_base + i + 1];
            conv_state[cs_base + i] = v;
            acc += f32::cast_from(v) * f32::cast_from(w[w_base + i]);
        }
        let xn = x[gid];
        conv_state[cs_base + k - 1] = xn;
        acc += f32::cast_from(xn) * f32::cast_from(w[w_base + k - 1]);
        out[gid] = F::cast_from(acc / (1.0f32 + (-acc).exp()));
    }
}

// ================================= 2. Fused GDN gating ==========================================

/// Fused GDN gating -- mirrors `gdn_gating.comp` / cuda `fused_gdn_gating_kernel`. Per element:
///   `beta = sigmoid(b)`,  `g = -exp(a_log) * softplus(a + dt_bias)`,  `softplus(x) = log(1 + exp(x))`.
/// `a_log` and `dt_bias` are per-head f32 tensors (`idx % num_heads`), broadcast over leading dims.
///
/// `a_log` here is the RAW `A_log` (the shader convention). The GGUF `ssm_a` instead stores the
/// precomputed `-exp(A_log)`; a caller holding that passes it as `a_log` with the sign folded, i.e.
/// use `gdn_gating_ref`'s `neg_exp_a_log=true` path -- one op, both conventions, no second kernel.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu, rocm), unchecked)]
pub fn gdn_gating<F: Float>(
    b_in: &Array<F>,
    a_in: &Array<F>,
    a_log: &Array<f32>,
    dt_bias: &Array<f32>,
    beta_out: &mut Array<F>,
    g_out: &mut Array<F>,
    dims: &Array<u32>, // [num_heads, precomputed_neg_exp_a]
) {
    let idx = ABSOLUTE_POS;
    if idx < beta_out.len() {
        let num_heads = dims[0] as usize;
        let precomputed_neg_exp = dims[1];
        let head = idx % num_heads;

        let bf = f32::cast_from(b_in[idx]);
        let af = f32::cast_from(a_in[idx]);
        let beta = 1.0f32 / (1.0f32 + (-bf).exp());
        let x = af + dt_bias[head];
        let mut softplus = x;
        if x <= 20.0f32 {
            softplus = (1.0f32 + x.exp()).ln();
        }
        let mut neg_exp_a = a_log[head];
        if precomputed_neg_exp == 0u32 {
            neg_exp_a = -(neg_exp_a.exp());
        }
        let g = neg_exp_a * softplus;

        beta_out[idx] = F::cast_from(beta);
        g_out[idx] = F::cast_from(g);
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
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu, rocm), unchecked)]
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
            let mut kv_mem = F::new(0.0f32);
            for j in 0..k_dim {
                let sj = s[j] * decay;
                s[j] = sj;
                kv_mem += sj * k[qk_bh + t * k_dim + j];
            }
            let delta = (v_t - kv_mem) * beta_t;

            // Rank-1 update s[j] += k_t[j]*δ, then read out y_t = Σ_j s[j]*q_t[j] (post-update).
            let mut y_t = F::new(0.0f32);
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
    gdn_conv1d_run_as::<f32, R>(client, x, w, batch, conv_dim, seq_len, k)
}

/// Typed host launch for [`gdn_conv1d`] supporting `F32`, `F16`, and `BF16`.
pub fn gdn_conv1d_run_as<F: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    x: &[F],
    w: &[F],
    batch: usize,
    conv_dim: usize,
    seq_len: usize,
    k: usize,
) -> Vec<F> {
    let total = batch * conv_dim * seq_len;
    let xh = client.create_from_slice(F::as_bytes(x));
    let wh = client.create_from_slice(F::as_bytes(w));
    let dh = client.create_from_slice(u32::as_bytes(&[conv_dim as u32, seq_len as u32]));
    let oh = client.create_from_slice(F::as_bytes(&vec![F::new(0.0f32); total]));
    let block = 64u32;
    let grid = (total as u32).div_ceil(block);
    unsafe {
        gdn_conv1d::launch_unchecked::<F, R>(
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
    F::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Host launch for `gdn_conv1d_step`. `x`: `[batch, conv_dim]`, `w`: `[conv_dim, k]`,
/// `conv_state`: `[batch, conv_dim, k]`. Returns `(out, new_conv_state)`.
pub fn gdn_conv1d_step_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    conv_state: &[f32],
    batch: usize,
    conv_dim: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    gdn_conv1d_step_run_as::<f32, R>(client, x, w, conv_state, batch, conv_dim, k)
}

/// Typed host launch for [`gdn_conv1d_step`] supporting `F32`, `F16`, and `BF16`.
pub fn gdn_conv1d_step_run_as<F: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    x: &[F],
    w: &[F],
    conv_state: &[F],
    batch: usize,
    conv_dim: usize,
    k: usize,
) -> (Vec<F>, Vec<F>) {
    let total = batch * conv_dim;
    let xh = client.create_from_slice(F::as_bytes(x));
    let wh = client.create_from_slice(F::as_bytes(w));
    let ch = client.create_from_slice(F::as_bytes(conv_state));
    let oh = client.create_from_slice(F::as_bytes(&vec![F::new(0.0f32); total]));
    let dh = client.create_from_slice(u32::as_bytes(&[batch as u32, conv_dim as u32]));
    let block = 64u32;
    let grid = (total as u32).div_ceil(block);
    unsafe {
        gdn_conv1d_step::launch_unchecked::<F, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(ch.clone(), conv_state.len()),
            ArrayArg::from_raw_parts(oh.clone(), total),
            ArrayArg::from_raw_parts(dh.clone(), 2),
            k,
        );
    }
    let out = F::from_bytes(&client.read_one_unchecked(oh)).to_vec();
    let new_cs = F::from_bytes(&client.read_one_unchecked(ch)).to_vec();
    (out, new_cs)
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
    gdn_gating_run_as::<f32, R>(client, b, a, a_log, dt_bias, num_heads)
}

/// Typed host launch for [`gdn_gating`] (`b`/`a`/`beta`/`g` in `F`, `a_log`/`dt_bias` in `f32`).
pub fn gdn_gating_run_as<F: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    b: &[F],
    a: &[F],
    a_log: &[f32],
    dt_bias: &[f32],
    num_heads: usize,
) -> (Vec<F>, Vec<F>) {
    let total = b.len();
    let bh = client.create_from_slice(F::as_bytes(b));
    let ah = client.create_from_slice(F::as_bytes(a));
    let alh = client.create_from_slice(f32::as_bytes(a_log));
    let dth = client.create_from_slice(f32::as_bytes(dt_bias));
    let betah = client.create_from_slice(F::as_bytes(&vec![F::new(0.0f32); total]));
    let gh = client.create_from_slice(F::as_bytes(&vec![F::new(0.0f32); total]));
    let dh = client.create_from_slice(u32::as_bytes(&[num_heads as u32, 0u32]));
    let block = 64u32;
    let grid = (total as u32).div_ceil(block);
    unsafe {
        gdn_gating::launch_unchecked::<F, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(bh.clone(), b.len()),
            ArrayArg::from_raw_parts(ah.clone(), a.len()),
            ArrayArg::from_raw_parts(alh.clone(), a_log.len()),
            ArrayArg::from_raw_parts(dth.clone(), dt_bias.len()),
            ArrayArg::from_raw_parts(betah.clone(), total),
            ArrayArg::from_raw_parts(gh.clone(), total),
            ArrayArg::from_raw_parts(dh.clone(), 2),
        );
    }
    let beta = F::from_bytes(&client.read_one_unchecked(betah)).to_vec();
    let g = F::from_bytes(&client.read_one_unchecked(gh)).to_vec();
    (beta, g)
}

#[allow(clippy::too_many_arguments)]
fn with_scan_buffers_as<F: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    q: &[F],
    k: &[F],
    v: &[F],
    g: &[F],
    beta: &[F],
    state: &[f32],
    bh: usize,
    seq: usize,
    v_dim: usize,
    launch: impl FnOnce(
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        &cubecl::server::Handle,
        usize,
    ),
) -> (Vec<F>, Vec<f32>) {
    let out_len = bh * seq * v_dim;
    let qh = client.create_from_slice(F::as_bytes(q));
    let kh = client.create_from_slice(F::as_bytes(k));
    let vh = client.create_from_slice(F::as_bytes(v));
    let gh = client.create_from_slice(F::as_bytes(g));
    let betah = client.create_from_slice(F::as_bytes(beta));
    let sh = client.create_from_slice(f32::as_bytes(state));
    let oh = client.create_from_slice(F::as_bytes(&vec![F::new(0.0f32); out_len]));
    let dh = client.create_from_slice(u32::as_bytes(&[bh as u32, seq as u32]));
    launch(&qh, &kh, &vh, &gh, &betah, &sh, &oh, &dh, out_len);
    let out = F::from_bytes(&client.read_one_unchecked(oh)).to_vec();
    let new_state = f32::from_bytes(&client.read_one_unchecked(sh)).to_vec();
    (out, new_state)
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
    with_scan_buffers_as::<f32, R>(client, q, k, v, g, beta, state, bh, seq, v_dim, |qh, kh, vh, gh, betah, sh, oh, dh, out_len| {
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
    })
}

/// Host launch for [`gdn_scan_lanes`]. Same tensor layout as [`gdn_scan_run`].
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_lanes_run<R: Runtime>(
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
    lanes: usize,
    block: usize,
    vk_layout: bool,
) -> (Vec<f32>, Vec<f32>) {
    gdn_scan_lanes_run_as::<f32, R>(
        client, q, k, v, g, beta, state, bh, seq, k_dim, v_dim, lanes, block, vk_layout,
    )
}

/// Typed host launch for [`gdn_scan_lanes`] (`q`/`k`/`v`/`g`/`beta`/`out` in `F`, `state` always `f32`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_lanes_run_as<F: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    q: &[F],
    k: &[F],
    v: &[F],
    g: &[F],
    beta: &[F],
    state: &[f32],
    bh: usize,
    seq: usize,
    k_dim: usize,
    v_dim: usize,
    lanes: usize,
    block: usize,
    vk_layout: bool,
) -> (Vec<F>, Vec<f32>) {
    with_scan_buffers_as::<F, R>(client, q, k, v, g, beta, state, bh, seq, v_dim, |qh, kh, vh, gh, betah, sh, oh, dh, out_len| {
        let grid = ((bh * v_dim * lanes) as u32).div_ceil(block as u32);
        unsafe {
            gdn_scan_lanes::launch_unchecked::<F, R>(
                client,
                Grid::Static(grid, 1, 1),
                Block::new_1d(block as u32),
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
                lanes,
                vk_layout,
            );
        }
    })
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
        let coeff = if neg_exp_a_log {
            a_log[head]
        } else {
            -(a_log[head].exp())
        };
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

// ============ Lane-split scan, its schedule space and its cold-memory evaluator =================

/// The gated delta-rule scan with each `(bh, v_row)` state split over `lanes` adjacent lanes of a
/// plane. The recurrence is sequential in `t`, so the way to fill a GPU is to split the state, not
/// the sequence: a lane keeps `k_dim / lanes` entries of its row in registers, forms its part of
/// `S^T k`, and a `plane_shuffle_xor` butterfly sums the parts; the row is corrected and read back
/// the same way with `q`. With `lanes == 1` this is `gdn_scan`. No shared memory and no barriers,
/// so it lowers to every backend that has planes.
///
/// `vk_layout` picks how the `[k_dim, v_dim]` state sits in memory: `false` is `state[j * v_dim + vi]`
/// (V contiguous, what `gdn_scan` and the engine allocate), `true` is `state[vi * k_dim + j]` (K
/// contiguous, so a lane's slice is one coalesced run). Which wins is a per-device fact the tuner
/// measures; the kernel does not guess.
///
/// Per-lane split and shuffle reduction follow ggml-metal's `kernel_gated_delta_net` (ggml-org/llama.cpp,
/// MIT; as carried in PrismML-Eng/llama.cpp), expressed once here instead of per backend.
///
/// Layout and contracts are `gdn_scan`'s: `q`,`k` `[bh, seq, k_dim]` (q pre-scaled), `v`,`out`
/// `[bh, seq, v_dim]`, `g`,`beta` `[bh, seq]`, `state` updated in place. `lanes` divides `k_dim`, is a
/// power of two, and is at most the plane size; the block is a multiple of `lanes`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn gdn_scan_lanes<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    g: &Array<F>,
    beta: &Array<F>,
    state: &mut Array<f32>,
    out: &mut Array<F>,
    dims: &Array<u32>, // [bh, seq]
    #[comptime] k_dim: usize,
    #[comptime] v_dim: usize,
    #[comptime] lanes: usize,
    #[comptime] vk_layout: bool,
) {
    let bh = dims[0] as usize;
    let seq = dims[1] as usize;
    let seg = comptime!(k_dim / lanes);
    let lanes_u = comptime!(lanes as u32);
    let total = bh * v_dim;
    let group_raw = ABSOLUTE_POS / lanes;
    let lane = ABSOLUTE_POS % lanes;
    // Every lane of a plane runs the body (the shuffles need all partners), so out-of-range groups
    // are clamped to a real one and only their writes are suppressed.
    let valid = group_raw < total;
    let mut group = group_raw;
    if !valid {
        group = 0usize;
    }
    let b = group / v_dim;
    let vi = group % v_dim;
    let qk_bh = b * seq * k_dim + lane * seg;
    let v_bh = b * seq * v_dim + vi;
    let gb_bh = b * seq;
    let state_base = b * k_dim * v_dim;

    let mut s = Array::<f32>::new(seg);
    for j in 0..seg {
        let kj = lane * seg + j;
        if vk_layout {
            s[j] = state[state_base + vi * k_dim + kj];
        } else {
            s[j] = state[state_base + kj * v_dim + vi];
        }
    }

    for t in 0..seq {
        let decay = f32::cast_from(g[gb_bh + t]).exp();
        let beta_t = f32::cast_from(beta[gb_bh + t]);
        let v_t = f32::cast_from(v[v_bh + t * v_dim]);

        let mut kv_mem = 0.0f32;
        for j in 0..seg {
            let sj = s[j] * decay;
            s[j] = sj;
            kv_mem += sj * f32::cast_from(k[qk_bh + t * k_dim + j]);
        }
        let mut off = 1u32;
        while off < lanes_u {
            kv_mem += plane_shuffle_xor(kv_mem, off);
            off *= 2;
        }
        let delta = (v_t - kv_mem) * beta_t;

        let mut y_t = 0.0f32;
        for j in 0..seg {
            let kj_t = f32::cast_from(k[qk_bh + t * k_dim + j]);
            let sj = s[j] + kj_t * delta;
            s[j] = sj;
            y_t += sj * f32::cast_from(q[qk_bh + t * k_dim + j]);
        }
        off = 1u32;
        while off < lanes_u {
            y_t += plane_shuffle_xor(y_t, off);
            off *= 2;
        }
        if valid && lane == 0 {
            out[v_bh + t * v_dim] = F::cast_from(y_t);
        }
    }

    if valid {
        for j in 0..seg {
            let kj = lane * seg + j;
            if vk_layout {
                state[state_base + vi * k_dim + kj] = s[j];
            } else {
                state[state_base + kj * v_dim + vi] = s[j];
            }
        }
    }
}

/// Deterministic inputs shaped like a Qwen3.5 GDN layer: unit-norm `k`, `q` pre-scaled by
/// `1/sqrt(k_dim)`, log-decay `g` in `(-0.51, -0.01)`, write strength `beta` in `(0.1, 0.9)`.
/// Returns `[q, k, v, g, beta, state]`.
pub fn gdn_synth(bh: usize, seq: usize, k_dim: usize, v_dim: usize, seed: u64) -> [Vec<f32>; 6] {
    let mut x = seed | 1;
    let mut unit = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        ((x >> 40) as f32) / (1u32 << 24) as f32
    };
    let mut k: Vec<f32> = (0..bh * seq * k_dim).map(|_| unit() - 0.5).collect();
    for row in k.chunks_mut(k_dim) {
        let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        row.iter_mut().for_each(|v| *v /= n);
    }
    let scale = 1.0 / (k_dim as f32).sqrt();
    let q = (0..bh * seq * k_dim)
        .map(|_| (unit() - 0.5) * scale)
        .collect();
    let v = (0..bh * seq * v_dim).map(|_| unit() * 2.0 - 1.0).collect();
    let g = (0..bh * seq).map(|_| -(0.01 + 0.5 * unit())).collect();
    let beta = (0..bh * seq).map(|_| 0.1 + 0.8 * unit()).collect();
    let state = (0..bh * k_dim * v_dim)
        .map(|_| (unit() - 0.5) * 0.1)
        .collect();
    [q, k, v, g, beta, state]
}

/// The schedule space of [`gdn_scan_lanes`]: lanes per state row, threads per block and state layout.
/// `lanes` must divide `k_dim` and the block; the device's plane width is checked by the evaluator.
pub fn gdn_scan_lanes_space(k_dim: usize) -> Space {
    Space::new()
        .param("LANES", [1, 2, 4, 8, 16, 32])
        .param("BLOCK", [32, 64, 128, 256])
        .param("LAYOUT", [0, 1])
        .constraint(move |c, s| {
            let l = c.get(s, "LANES") as usize;
            let b = c.get(s, "BLOCK") as usize;
            k_dim % l == 0 && b % l == 0
        })
}

struct GdnBank {
    q: cubecl::server::Handle,
    k: cubecl::server::Handle,
    v: cubecl::server::Handle,
    g: cubecl::server::Handle,
    beta: cubecl::server::Handle,
    out: cubecl::server::Handle,
    st: [cubecl::server::Handle; 2], // the same initial state in each layout
}

/// Cold-memory, oracle-gated fitness for [`gdn_scan_lanes_space`]. Holds `nbanks` resident copies of a
/// layer's inputs (distinct buffers, so a rotation reads cold the way a 48-layer forward does) and the
/// state in both layouts. The first dispatch of a config runs on a fresh state and must match the CPU
/// oracle (output and final state); only then is it timed, rotating banks.
pub struct GdnLanesEval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<GdnBank>,
    st_host: [Vec<f32>; 2],
    dims: cubecl::server::Handle,
    shape: (usize, usize, usize, usize), // bh, seq, k_dim, v_dim
    oracle_out: Vec<f32>,
    oracle_state: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> GdnLanesEval<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        inputs: &[Vec<f32>; 6],
        shape: (usize, usize, usize, usize),
        nbanks: usize,
        repeats: usize,
    ) -> Self {
        let (bh, seq, k_dim, v_dim) = shape;
        let [q, k, v, g, beta, state] = inputs;
        let mut vk = vec![0.0f32; state.len()];
        for b in 0..bh {
            for j in 0..k_dim {
                for vi in 0..v_dim {
                    vk[b * k_dim * v_dim + vi * k_dim + j] =
                        state[b * k_dim * v_dim + j * v_dim + vi];
                }
            }
        }
        let up = |x: &[f32]| client.create_from_slice(f32::as_bytes(x));
        let banks = (0..nbanks.max(1))
            .map(|_| GdnBank {
                q: up(q),
                k: up(k),
                v: up(v),
                g: up(g),
                beta: up(beta),
                out: up(&vec![0.0f32; bh * seq * v_dim]),
                st: [up(state), up(&vk)],
            })
            .collect();
        let mut oracle_state = state.clone();
        let oracle_out = gdn_scan_ref(q, k, v, g, beta, &mut oracle_state, bh, seq, k_dim, v_dim);
        let maxref = oracle_out
            .iter()
            .fold(0f32, |a, &x| a.max(x.abs()))
            .max(1e-30);
        Self {
            client,
            space,
            banks,
            st_host: [state.clone(), vk],
            dims: client.create_from_slice(u32::as_bytes(&[bh as u32, seq as u32])),
            shape,
            oracle_out,
            oracle_state,
            maxref,
            repeats,
            worst_rel: std::cell::Cell::new(0.0),
        }
    }

    /// Worst scale-relative divergence any measured config showed (the correctness witness).
    pub fn worst_rel(&self) -> f32 {
        self.worst_rel.get()
    }

    pub fn nbanks(&self) -> usize {
        self.banks.len()
    }

    fn cfg(&self, c: &Config) -> (usize, usize, bool) {
        (
            c.get(self.space, "LANES") as usize,
            c.get(self.space, "BLOCK") as usize,
            c.get(self.space, "LAYOUT") == 1,
        )
    }

    fn dispatch(
        &self,
        bank: &GdnBank,
        st: &cubecl::server::Handle,
        lanes: usize,
        block: usize,
        vk: bool,
    ) {
        let (bh, seq, k_dim, v_dim) = self.shape;
        let grid = ((bh * v_dim * lanes) as u32).div_ceil(block as u32);
        unsafe {
            gdn_scan_lanes::launch_unchecked::<f32, R>(
                self.client,
                Grid::Static(grid, 1, 1),
                Block::new_1d(block as u32),
                ArrayArg::from_raw_parts(bank.q.clone(), bh * seq * k_dim),
                ArrayArg::from_raw_parts(bank.k.clone(), bh * seq * k_dim),
                ArrayArg::from_raw_parts(bank.v.clone(), bh * seq * v_dim),
                ArrayArg::from_raw_parts(bank.g.clone(), bh * seq),
                ArrayArg::from_raw_parts(bank.beta.clone(), bh * seq),
                ArrayArg::from_raw_parts(st.clone(), bh * k_dim * v_dim),
                ArrayArg::from_raw_parts(bank.out.clone(), bh * seq * v_dim),
                ArrayArg::from_raw_parts(self.dims.clone(), 2),
                k_dim,
                v_dim,
                lanes,
                vk,
            );
        }
    }

    fn read(&self, h: &cubecl::server::Handle) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(h.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for GdnLanesEval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (lanes, block, _) = self.cfg(cfg);
        let hw = &self.client.properties().hardware;
        if lanes as u32 > hw.plane_size_min {
            return Verdict::Reject(format!(
                "{lanes} lanes exceed the device's {}-wide plane",
                hw.plane_size_min
            ));
        }
        if block as u32 > hw.max_units_per_cube {
            return Verdict::Reject(format!(
                "block {block} exceeds {} units per cube",
                hw.max_units_per_cube
            ));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (lanes, block, vk) = self.cfg(cfg);
        let (_, _, k_dim, v_dim) = self.shape;
        let layout = vk as usize;

        // Correctness: a fresh state, one dispatch, output and final state against the oracle.
        let fresh = self
            .client
            .create_from_slice(f32::as_bytes(&self.st_host[layout]));
        self.dispatch(&self.banks[0], &fresh, lanes, block, vk);
        let out = self.read(&self.banks[0].out);
        let mut st = self.read(&fresh);
        if vk {
            let mut kv = vec![0.0f32; st.len()];
            for b in 0..st.len() / (k_dim * v_dim) {
                for j in 0..k_dim {
                    for vi in 0..v_dim {
                        kv[b * k_dim * v_dim + j * v_dim + vi] =
                            st[b * k_dim * v_dim + vi * k_dim + j];
                    }
                }
            }
            st = kv;
        }
        let rel_out = out
            .iter()
            .zip(&self.oracle_out)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / self.maxref;
        let rel_st = st
            .iter()
            .zip(&self.oracle_state)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rel = rel_out.max(rel_st);
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if !(rel <= 1e-3) {
            return f64::INFINITY;
        }

        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(
                &self.banks[i % nb],
                &self.banks[i % nb].st[layout],
                lanes,
                block,
                vk,
            );
        }
        let _ = self.read(&self.banks[0].out);
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(
                        &self.banks[i % nb],
                        &self.banks[i % nb].st[layout],
                        lanes,
                        block,
                        vk,
                    );
                }
                let _ = self.read(&self.banks[0].out);
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
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
        for &(batch, conv_dim, seq_len) in &[(1usize, 512usize, 48usize), (1, 64, 3), (2, 128, 32)]
        {
            let x = rnd(batch * conv_dim * seq_len, 0x1111 + conv_dim as u64);
            let w = rnd(conv_dim * 4, 0x2222 + seq_len as u64);
            let got = gdn_conv1d_run::<CpuRuntime>(&c, &x, &w, batch, conv_dim, seq_len, 4);
            let want = gdn_conv1d_ref(&x, &w, batch, conv_dim, seq_len, 4);
            let rel = max_rel(&want, &got);
            eprintln!("[gdn_conv1d CPU] b{batch} c{conv_dim} s{seq_len} k4 max_rel={rel:.2e}");
            assert!(
                rel < 2e-3,
                "gdn_conv1d b{batch} c{conv_dim} s{seq_len} max_rel {rel}"
            );
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn gdn_conv1d_step_cpu_matches_full() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        for &(batch, conv_dim, seq_len) in &[(1usize, 128usize, 11usize), (2, 64, 7)] {
            let x_full = rnd(batch * conv_dim * seq_len, 0x1357 + conv_dim as u64);
            let w = rnd(conv_dim * 4, 0x2468 + seq_len as u64);
            let want_full = gdn_conv1d_ref(&x_full, &w, batch, conv_dim, seq_len, 4);

            let mut conv_state = vec![0.0f32; batch * conv_dim * 4];
            let mut got_full = vec![0.0f32; batch * conv_dim * seq_len];
            for t in 0..seq_len {
                let mut xt = Vec::with_capacity(batch * conv_dim);
                for b in 0..batch {
                    for ch in 0..conv_dim {
                        xt.push(x_full[(b * conv_dim + ch) * seq_len + t]);
                    }
                }
                let (yt, new_cs) =
                    gdn_conv1d_step_run::<CpuRuntime>(&c, &xt, &w, &conv_state, batch, conv_dim, 4);
                conv_state = new_cs;
                for b in 0..batch {
                    for ch in 0..conv_dim {
                        got_full[(b * conv_dim + ch) * seq_len + t] = yt[b * conv_dim + ch];
                    }
                }
            }
            let rel = max_rel(&want_full, &got_full);
            eprintln!("[gdn_conv1d_step CPU] b{batch} c{conv_dim} s{seq_len} max_rel={rel:.2e}");
            assert!(rel < 1e-5, "gdn_conv1d_step rel {rel}");
            for b in 0..batch {
                for ch in 0..conv_dim {
                    for i in 0..4 {
                        let got_s = conv_state[(b * conv_dim + ch) * 4 + i];
                        let pos = seq_len as isize - 4 + i as isize;
                        let want_s = if pos >= 0 {
                            x_full[(b * conv_dim + ch) * seq_len + pos as usize]
                        } else {
                            0.0
                        };
                        assert!((got_s - want_s).abs() < 1e-6);
                    }
                }
            }
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
            eprintln!(
                "[gdn_gating CPU] h{num_heads} tok{tokens}  beta_rel={rb:.2e} g_rel={rg:.2e}"
            );
            assert!(
                rb < 2e-3 && rg < 2e-3,
                "gdn_gating h{num_heads} beta={rb} g={rg}"
            );
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
            let g: Vec<f32> = rnd(bh * seq, 0x7004)
                .iter()
                .map(|x| x * 0.5 - 0.5)
                .collect();
            let beta: Vec<f32> = rnd(bh * seq, 0x7005)
                .iter()
                .map(|x| (x + 1.0) * 0.5)
                .collect();
            let state0 = rnd(bh * k_dim * v_dim, 0x7006);

            let (got_out, got_state) = gdn_scan_run::<CpuRuntime>(
                &c, &q, &k, &v, &g, &beta, &state0, bh, seq, k_dim, v_dim,
            );
            let mut ref_state = state0.clone();
            let want_out =
                gdn_scan_ref(&q, &k, &v, &g, &beta, &mut ref_state, bh, seq, k_dim, v_dim);
            let ro = max_rel(&want_out, &got_out);
            let rs = max_rel(&ref_state, &got_state);
            eprintln!(
                "[gdn_scan CPU] bh{bh} s{seq} k{k_dim} v{v_dim}  out_rel={ro:.2e} state_rel={rs:.2e}"
            );
            assert!(
                ro < 2e-3 && rs < 2e-3,
                "gdn_scan bh{bh} s{seq} out={ro} state={rs}"
            );
        }
    }
}
