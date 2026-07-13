//! Semantics lock for the ROCm top-k selection.
//!
//! The `topk_last_dim_*` HIP kernel (hanzo-rocm-kernels/src/kernels/sort.hip) and its host launcher
//! `RocmStorage::topk_last_dim` replace the router hot path's `asort(desc) + narrow(k)` full bitonic
//! sort with `k` block-wide argmax passes. This test proves -- in pure CPU Rust, so it runs with the
//! HIP kernel gated behind `--features rocm` -- that the k-pass argmax selection makes the IDENTICAL
//! choice as a stable descending sort narrowed to `k`, on random logits INCLUDING exact ties. Both
//! use the canonical tie-break value DESC, index ASC (first-index-wins on equal scores), the same
//! rule as the fused `moe_route` kernel and the engine's `deterministic_topk_indices`; routing stays
//! bit-identical to that canonical selection.

/// Reference "old" selection: full descending sort, canonical (value DESC, index ASC) tie-break,
/// take the first `k`. Mirrors `asort(desc) + narrow(k)` under the canonical tie-break.
fn sort_then_narrow(scores: &[f32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..scores.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        scores[b as usize]
            .total_cmp(&scores[a as usize])
            .then(a.cmp(&b))
    });
    idx[..k].to_vec()
}

/// "New" selection: exact CPU mirror of the `k_topk` HIP kernel -- `k` argmax passes with the same
/// (value DESC, index ASC) tie-break, clobbering the selected column to -inf between passes.
fn kpass_argmax(scores: &[f32], k: usize) -> Vec<u32> {
    let mut s = scores.to_vec();
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        let mut bv = f32::NEG_INFINITY;
        let mut bi = s.len();
        for (i, &v) in s.iter().enumerate() {
            if v > bv || (v == bv && i < bi) {
                bv = v;
                bi = i;
            }
        }
        out.push(bi as u32);
        s[bi] = f32::NEG_INFINITY;
    }
    out
}

/// Small deterministic xorshift so the test needs no rng dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A logit quantized to a small ladder of values so exact ties occur frequently.
    fn tied_logit(&mut self, levels: u32) -> f32 {
        (self.next() % levels as u64) as f32
    }
    fn logit(&mut self) -> f32 {
        // finite, no -0.0 / NaN: total_cmp and IEEE `>`/`==` agree on this range.
        (self.next() % 100_000) as f32 / 1000.0
    }
}

#[test]
fn topk_matches_sort_narrow_random_and_ties() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    // Cover the router regime (n_experts, top_k) plus edge k == n and k == 1.
    let cases = [
        (256usize, 8usize),
        (128, 8),
        (256, 1),
        (64, 16),
        (32, 32),
        (1024, 8),
        (7, 7),
        (1, 1),
    ];
    for &(n, k) in &cases {
        for trial in 0..200 {
            // Mix of well-separated logits and heavily-tied ladders to exercise the tie-break.
            let row: Vec<f32> = if trial % 3 == 0 {
                // heavy ties: only a few distinct levels across n columns
                (0..n).map(|_| rng.tied_logit(4)).collect()
            } else if trial % 3 == 1 {
                (0..n).map(|_| rng.tied_logit(16)).collect()
            } else {
                (0..n).map(|_| rng.logit()).collect()
            };
            let old = sort_then_narrow(&row, k);
            let new = kpass_argmax(&row, k);
            assert_eq!(
                new, old,
                "selection diverged for n={n} k={k} trial={trial}\nrow={row:?}"
            );
        }
    }
}

#[test]
fn topk_all_equal_is_first_k_indices() {
    // Degenerate all-ties row: first-index-wins must yield exactly 0..k.
    for &(n, k) in &[(256usize, 8usize), (128, 8), (10, 10)] {
        let row = vec![1.0f32; n];
        let expect: Vec<u32> = (0..k as u32).collect();
        assert_eq!(kpass_argmax(&row, k), expect);
        assert_eq!(sort_then_narrow(&row, k), expect);
    }
}
