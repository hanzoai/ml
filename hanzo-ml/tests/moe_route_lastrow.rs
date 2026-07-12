// Regression guard for the MoE-router "last-row collapse" class of bug.
//
// A now-fixed CUDA path (engine ops.rs `cuda_topk` via the dead `final_logits_row` helper) narrowed a
// [tokens, experts] router-logit tensor to its LAST row before top-k, so every prefill token was routed
// by the LAST token's experts -- silent routing corruption for all `.topk()` MoE models. The ROCm and
// CUDA fused `moe_route` kernels never had this bug (one block per token, `tok = blockIdx.x`), and the
// GGUF Qwen3-30B-A3B path routes through `quantized::moe_route` directly. This test pins the property
// the bug violated on the backend-agnostic public API: routing is PER-TOKEN, never last-token-broadcast.
//
// The on-device HIP/CUDA kernels are gated bit-faithfully against the same per-token semantics by the
// `moe_route_numeric` oracle (engine hanzo-cli, features rocm/cuda, ntok up to 1024 at 128 experts/topk=8).
// This CPU test needs no GPU, so it runs anywhere and can never wedge a kfd driver.

use hanzo_ml::{Device, Tensor};

// The buggy behavior the CUDA path used to have: route EVERY token by the last token's experts.
fn simulate_last_row_collapse(
    logits: &[f32],
    ntok: usize,
    n_experts: usize,
    topk: usize,
) -> Vec<u32> {
    let last = &logits[(ntok - 1) * n_experts..ntok * n_experts];
    let last_ids = per_token_topk_ids(last, n_experts, topk);
    // Every token gets the last token's experts.
    (0..ntok).flat_map(|_| last_ids.clone()).collect()
}

// Reference: descending-logit top-k, lowest index wins a tie (matches the fused kernels + the CPU fallback).
fn per_token_topk_ids(row: &[f32], n_experts: usize, topk: usize) -> Vec<u32> {
    let mut used = vec![false; n_experts];
    let mut ids = Vec::with_capacity(topk);
    for _ in 0..topk {
        let mut best = usize::MAX;
        let mut bestv = f32::NEG_INFINITY;
        for e in 0..n_experts {
            if !used[e] && row[e] > bestv {
                bestv = row[e];
                best = e;
            }
        }
        used[best] = true;
        ids.push(best as u32);
    }
    ids
}

fn route_ids(logits: &[f32], ntok: usize, n_experts: usize, topk: usize) -> Vec<u32> {
    let dev = Device::Cpu;
    let lt = Tensor::from_vec(logits.to_vec(), (ntok, n_experts), &dev).expect("logits");
    let (ids_t, _w) = hanzo_ml::quantized::moe_route(&lt, topk, true).expect("moe_route");
    assert_eq!(ids_t.dims(), &[ntok, topk], "ids shape");
    ids_t
        .flatten_all()
        .unwrap()
        .to_vec1::<u32>()
        .expect("ids vec")
}

// Distinct, well-separated experts per token so routing is unambiguous (no ties): token t peaks at
// experts [t, t+1, ...]. The last token's experts are DIFFERENT from every earlier token's, so a
// last-row-collapse is observable as token 0 wearing the last token's ids.
fn staggered_logits(ntok: usize, n_experts: usize, topk: usize) -> Vec<f32> {
    let mut v = vec![0f32; ntok * n_experts];
    for t in 0..ntok {
        for e in 0..n_experts {
            // Peak of magnitude `topk-rank` at experts t..t+topk (wrapped), flat 0 elsewhere.
            let mut val = 0f32;
            for r in 0..topk {
                if e == (t + r) % n_experts {
                    val = (topk - r) as f32 * 10.0;
                }
            }
            v[t * n_experts + e] = val;
        }
    }
    v
}

#[test]
fn moe_route_is_per_token_not_last_row() {
    // Shape mirrors a real prefill (many tokens, model-sized experts/topk).
    for &(ntok, n_experts, topk) in &[(4usize, 8usize, 2usize), (128, 128, 8), (1024, 128, 8)] {
        let logits = staggered_logits(ntok, n_experts, topk);
        let got = route_ids(&logits, ntok, n_experts, topk);

        // 1) Every token matches its OWN per-token top-k.
        for t in 0..ntok {
            let want = per_token_topk_ids(
                &logits[t * n_experts..(t + 1) * n_experts],
                n_experts,
                topk,
            );
            let slice = &got[t * topk..(t + 1) * topk];
            assert_eq!(slice, &want[..], "token {t} routed wrong (ntok={ntok})");
        }

        // 2) The buggy last-row-collapse would give a DIFFERENT result -- prove moe_route avoids it.
        let buggy = simulate_last_row_collapse(&logits, ntok, n_experts, topk);
        assert_ne!(
            got, buggy,
            "moe_route must NOT reproduce the last-row-collapse (ntok={ntok})"
        );
        // Concretely: token 0 keeps its own experts, not the last token's.
        assert_ne!(
            &got[0..topk],
            &buggy[0..topk],
            "token 0 wears the last token's experts -- last-row collapse (ntok={ntok})"
        );
    }
}

// Near-tie determinism: the "borderline prompt" failure class is a near-tie between two candidates. A
// near-tie is NOT a tie -- the strictly-greater logit wins deterministically on every backend -- so
// routing is stable and reproducible, and cannot be the source of a first-token argmax flip.
#[test]
fn moe_route_near_tie_is_deterministic() {
    let (ntok, n_experts, topk) = (1usize, 8usize, 2usize);
    // Experts 3 and 4 are a near-tie (1e-4 apart), 3 is strictly larger -> 3 then 4, every run.
    let mut logits = vec![0f32; n_experts];
    logits[3] = 1.0;
    logits[4] = 1.0 - 1e-4;
    for _ in 0..16 {
        let got = route_ids(&logits, ntok, n_experts, topk);
        assert_eq!(got, vec![3u32, 4u32], "near-tie routing must be deterministic");
    }
}
