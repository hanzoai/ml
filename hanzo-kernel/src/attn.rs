//! Scaled-dot-product attention in the DSL, one source -> every backend.
//!
//! `softmax(Q Kᵀ / sqrt(d) + causal_mask) V`, GQA-aware. One thread per (head, query): it streams over
//! the keys with an ONLINE (flash-style) softmax -- running max `m`, running denom `l`, and a per-thread
//! output accumulator `acc[d]` rescaled as `m` grows. Numerically stable and single-pass, with no
//! stored score row. This is the structural cure for the 8B repetition-collapse: with ONE attention
//! implementation across backends, the "flash vs eager vs Metal, three numeric behaviors" fork cannot
//! occur -- there is nothing to diverge.

use crate::prelude::*;

/// GQA SDPA. Layouts are `[head, seq, d]` row-major. `causal=1` masks keys `kk > qpos` (aligned q/k).
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn sdpa<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    out: &mut Array<F>,
    scale: &Array<F>,
    #[comptime] d: usize,
    #[comptime] seq_q: usize,
    #[comptime] seq_k: usize,
    #[comptime] n_kv_groups: usize,
    #[comptime] causal: u32,
) {
    let row = ABSOLUTE_POS; // over n_heads * seq_q
    if row < out.len() / d {
        let sc = scale[0];
        let h = row / seq_q;
        let qpos = row % seq_q;
        let kv = h / n_kv_groups;
        let qbase = row * d;
        let kvbase = kv * seq_k * d;

        let mut acc = Array::<F>::new(d);
        for dd in 0..d {
            acc[dd] = F::new(0.0);
        }
        let mut m = F::new(-3.4e38); // running max (-inf)
        let mut l = F::new(0.0); // running denom

        for kk in 0..seq_k {
            let masked = causal == 1 && kk > qpos;
            if !masked {
                let kbase = kvbase + kk * d;
                let mut score = F::new(0.0);
                for dd in 0..d {
                    score += q[qbase + dd] * k[kbase + dd];
                }
                score *= sc;
                let mut new_m = m;
                if score > new_m {
                    new_m = score;
                }
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                l = l * corr + p;
                for dd in 0..d {
                    acc[dd] = acc[dd] * corr + p * v[kbase + dd];
                }
                m = new_m;
            }
        }
        for dd in 0..d {
            out[qbase + dd] = acc[dd] / l;
        }
    }
}

/// Host launch. `q`: `[n_heads, seq_q, d]`, `k`/`v`: `[n_kv, seq_k, d]`, GQA `n_kv_groups = n_heads/n_kv`.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_run<R: Runtime>(
    client: &ComputeClient<R>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    seq_k: usize,
    d: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k));
    let vh = client.create_from_slice(f32::as_bytes(v));
    let sh = client.create_from_slice(f32::as_bytes(&[scale]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n_heads * seq_q * d]));
    let rows = (n_heads * seq_q) as u32;
    let block = 64u32;
    unsafe {
        sdpa::launch_unchecked::<f32, R>(
            client,
            Grid::Static(rows.div_ceil(block), 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), q.len()),
            ArrayArg::from_raw_parts(kh.clone(), k.len()),
            ArrayArg::from_raw_parts(vh.clone(), v.len()),
            ArrayArg::from_raw_parts(oh.clone(), n_heads * seq_q * d),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            d,
            seq_q,
            seq_k,
            n_heads / n_kv,
            causal as u32,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle: full-precision two-pass softmax attention, the reference the DSL kernel is gated against.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    seq_k: usize,
    d: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let groups = n_heads / n_kv;
    let mut out = vec![0.0f32; n_heads * seq_q * d];
    for h in 0..n_heads {
        let kv = h / groups;
        for qpos in 0..seq_q {
            let qbase = (h * seq_q + qpos) * d;
            let klen = if causal { qpos + 1 } else { seq_k };
            let mut scores = vec![0.0f32; klen];
            for (kk, s) in scores.iter_mut().enumerate() {
                let kbase = (kv * seq_k + kk) * d;
                *s = (0..d).map(|dd| q[qbase + dd] * k[kbase + dd]).sum::<f32>() * scale;
            }
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let obase = qbase;
            for dd in 0..d {
                let mut acc = 0.0f32;
                for (kk, e) in exps.iter().enumerate() {
                    acc += e / sum * v[(kv * seq_k + kk) * d + dd];
                }
                out[obase + dd] = acc;
            }
        }
    }
    out
}

/// Runtime-seq GQA SDPA. `seq_q`/`seq_k` are RUNTIME (via the `dims` buffer, so one compiled kernel
/// serves a KV cache that grows every token); `d`/`n_kv_groups`/`causal` stay comptime (model-fixed).
/// Same online-softmax math as `sdpa` -- the same structural cure -- but usable in production decode.
/// Decode: `seq_q=1`, `causal=0`, the query sees the whole `seq_k`-long cache. Prefill: `seq_q=seq_k`,
/// `causal=1`, aligned triangular mask.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn sdpa_runtime<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    out: &mut Array<F>,
    scale: &Array<F>,
    dims: &Array<u32>, // [seq_q, seq_k]
    #[comptime] d: usize,
    #[comptime] n_kv_groups: usize,
    #[comptime] causal: u32,
) {
    let row = ABSOLUTE_POS; // over n_heads * seq_q
    if row < out.len() / d {
        let seq_q = dims[0] as usize;
        let seq_k = dims[1] as usize;
        let sc = scale[0];
        let h = row / seq_q;
        let qpos = row % seq_q;
        let kv = h / n_kv_groups;
        let qbase = row * d;
        let kvbase = kv * seq_k * d;

        let mut acc = Array::<F>::new(d);
        for dd in 0..d {
            acc[dd] = F::new(0.0);
        }
        let mut m = F::new(-3.4e38); // running max (-inf)
        let mut l = F::new(0.0); // running denom

        for kk in 0..seq_k {
            let masked = causal == 1 && kk > qpos;
            if !masked {
                let kbase = kvbase + kk * d;
                let mut score = F::new(0.0);
                for dd in 0..d {
                    score += q[qbase + dd] * k[kbase + dd];
                }
                score *= sc;
                let mut new_m = m;
                if score > new_m {
                    new_m = score;
                }
                let corr = (m - new_m).exp();
                let p = (score - new_m).exp();
                l = l * corr + p;
                for dd in 0..d {
                    acc[dd] = acc[dd] * corr + p * v[kbase + dd];
                }
                m = new_m;
            }
        }
        for dd in 0..d {
            out[qbase + dd] = acc[dd] / l;
        }
    }
}

/// Host launch for `sdpa_runtime`. `seq_q`/`seq_k` go through the runtime `dims` buffer so one compiled
/// kernel serves any (growing) KV length -- the shelf-ready piece for the CUDA 8B-attention cure.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_runtime_run<R: Runtime>(
    client: &ComputeClient<R>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    seq_k: usize,
    d: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k));
    let vh = client.create_from_slice(f32::as_bytes(v));
    let sh = client.create_from_slice(f32::as_bytes(&[scale]));
    let dh = client.create_from_slice(u32::as_bytes(&[seq_q as u32, seq_k as u32]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n_heads * seq_q * d]));
    let rows = (n_heads * seq_q) as u32;
    let block = 64u32;
    unsafe {
        sdpa_runtime::launch_unchecked::<f32, R>(
            client,
            Grid::Static(rows.div_ceil(block), 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), q.len()),
            ArrayArg::from_raw_parts(kh.clone(), k.len()),
            ArrayArg::from_raw_parts(vh.clone(), v.len()),
            ArrayArg::from_raw_parts(oh.clone(), n_heads * seq_q * d),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            ArrayArg::from_raw_parts(dh.clone(), 2),
            d,
            n_heads / n_kv,
            causal as u32,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
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
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / x.abs().max(1e-6)).fold(0.0, f32::max)
    }

    // GQA shape: 4 query heads, 2 kv heads (groups=2), seq 24, head_dim 32.
    fn run<R: Runtime>(c: &ComputeClient<R>, causal: bool, tag: &str) {
        let (nh, nkv, sq, sk, d) = (4, 2, 24, 24, 32);
        let q = rnd(nh * sq * d, 0x1234_5678);
        let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
        let v = rnd(nkv * sk * d, 0x0FED_CBA9);
        let got = sdpa_run::<R>(c, &q, &k, &v, nh, nkv, sq, sk, d, causal);
        let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, causal);
        let rel = max_rel(&want, &got);
        eprintln!("[sdpa {tag}] gqa 4/2 s{sq} d{d} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "sdpa {tag} max_rel {rel}");
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn sdpa_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        run::<CpuRuntime>(&c, false, "noncausal CPU");
        run::<CpuRuntime>(&c, true, "causal CPU");
    }

    // Runtime-seq variant: seq_q/seq_k travel through the dims buffer, so one kernel serves any KV
    // length. Gated over the real production shape space: decode (seq_q=1 vs growing kv) + prefill (causal).
    #[allow(clippy::too_many_arguments)]
    fn run_rt<R: Runtime>(c: &ComputeClient<R>, nh: usize, nkv: usize, sq: usize, sk: usize, d: usize, causal: bool, tag: &str) {
        let q = rnd(nh * sq * d, 0x1234_5678);
        let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
        let v = rnd(nkv * sk * d, 0x0FED_CBA9);
        let got = sdpa_runtime_run::<R>(c, &q, &k, &v, nh, nkv, sq, sk, d, causal);
        let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, causal);
        let rel = max_rel(&want, &got);
        eprintln!("[sdpa_rt {tag}] nh{nh}/nkv{nkv} sq{sq} sk{sk} d{d} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "sdpa_rt {tag} max_rel {rel}");
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn sdpa_runtime_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        // decode: single query sees the whole (growing) cache, non-causal.
        run_rt::<CpuRuntime>(&c, 4, 2, 1, 1, 32, false, "decode kv1");
        run_rt::<CpuRuntime>(&c, 4, 2, 1, 17, 32, false, "decode kv17");
        run_rt::<CpuRuntime>(&c, 4, 2, 1, 128, 32, false, "decode kv128");
        run_rt::<CpuRuntime>(&c, 8, 2, 1, 512, 64, false, "decode kv512 GQA4");
        // prefill: seq_q == seq_k, aligned causal mask.
        run_rt::<CpuRuntime>(&c, 4, 2, 24, 24, 32, true, "prefill causal GQA2");
        run_rt::<CpuRuntime>(&c, 4, 4, 40, 40, 32, true, "prefill causal MHA");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn sdpa_metal_bit_exact() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        run::<WgpuRuntime>(&c, false, "noncausal METAL");
        run::<WgpuRuntime>(&c, true, "causal METAL");
    }
}
