pub mod attention_residuals;
pub mod cpu_flash;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod lsa;
pub mod rocm;
pub mod varlen;

use hanzo_ml::{DType, Result, Tensor};

pub use attention_residuals::AttentionResiduals;
pub use cpu_flash::flash_attn;
pub use cpu_flash::packed::packed_dlsa;
pub use cpu_flash::varlen::flash_attn_varlen_cpu;
pub use rocm::{rocm_flash_attn, rocm_flash_attn_decode};
pub use varlen::flash_attn_varlen_unfused;

#[derive(Debug, Clone, Default)]
pub enum AttnMask {
    #[default]
    None,
    Causal {
        kv_offset: usize,
    },
    Mask(Tensor),
}

impl AttnMask {
    #[inline]
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

    #[inline]
    pub fn causal_with_offset(kv_offset: usize) -> Self {
        AttnMask::Causal { kv_offset }
    }

    #[inline]
    pub fn is_causal(&self) -> bool {
        matches!(self, AttnMask::Causal { .. })
    }

    #[inline]
    pub fn kv_offset(&self) -> usize {
        match self {
            AttnMask::Causal { kv_offset } => *kv_offset,
            _ => 0,
        }
    }
}

/// Whether [`packed`] runs on `dev` in `dtype` with heads of `head` dimensions.
pub fn packs(dev: &hanzo_ml::Device, dtype: hanzo_ml::DType, head: usize) -> bool {
    match dev {
        hanzo_ml::Device::Cpu => dtype == hanzo_ml::DType::F32,
        hanzo_ml::Device::Cuda(_) => {
            cfg!(feature = "cuda") && dtype == hanzo_ml::DType::BF16 && head == 64
        }
        _ => false,
    }
}

/// [`cpu_flash::packed::packed`] on the device of `qkv`; on CUDA it carries a gradient.
#[allow(clippy::too_many_arguments)]
pub fn packed(
    qkv: &Tensor,
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    match qkv.device() {
        hanzo_ml::Device::Cpu => {
            cpu_flash::packed::packed(qkv, lens, heads, window, cos, sin, scale)
        }
        #[cfg(feature = "cuda")]
        hanzo_ml::Device::Cuda(_) => cuda::packed(qkv, lens, heads, window, cos, sin, scale),
        d => hanzo_ml::bail!("packed attention: no kernel on {d:?}"),
    }
}

/// Batched block-bucket Bidirectional D-LSA on GPU (ROCm, CUDA, Metal) or CPU tensor fallback.
///
/// - `q, k, v`: `[B, H, L, D]`, with `q` already scaled by `1 / sqrt(D)`.
/// - `q_idx, k_idx`: `[B, L, d_idx]` low-rank index projections.
/// - Pools queries into tiles of `B_Q` (512 or 64) and keys into buckets of `B_K = 64` on the
///   device, selects the top `k_budget / 64` buckets per query tile (always retaining prefix
///   `sink_tokens` and bilateral `local_window`), and executes strided-batched GEMMs over the
///   gathered `[B·H·chunk_qb, k_eff, D]` keys and values.
#[allow(clippy::too_many_arguments)]
pub fn dlsa_batched_gpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_idx: &Tensor,
    k_idx: &Tensor,
    k_budget: usize,
    sink_tokens: usize,
    local_window: usize,
) -> Result<Tensor> {
    let (b, h, l, d) = q.dims4()?;
    let b_k = 64usize;
    if b != 1 || l <= k_budget || l % b_k != 0 {
        let kt = k.t()?.contiguous()?;
        let v = v.contiguous()?;
        let step = ((1usize << 26) / (b * h * l)).clamp(1, l);
        let mut out = Vec::with_capacity(l.div_ceil(step));
        for at in (0..l).step_by(step) {
            let t = step.min(l - at);
            let scores = q.narrow(2, at, t)?.contiguous()?.matmul(&kt)?;
            let p = crate::ops::softmax_last_dim(&scores)?;
            out.push(p.matmul(&v)?);
        }
        return Tensor::cat(&out, 2);
    }

    let b_q = if l % 512 == 0 { 512 } else { 64 };
    let n_qb = l / b_q;
    let n_kb = l / b_k;
    let target_nb = k_budget.div_ceil(b_k).clamp(1, n_kb);
    let k_eff = target_nb * b_k;
    let d_idx = q_idx.dim(2)?;

    // Pool q_idx [1, n_qb, b_q, d_idx] and k_idx [1, n_kb, b_k, d_idx] on device
    let qb_t = q_idx
        .to_dtype(DType::F32)?
        .reshape((1, n_qb, b_q, d_idx))?
        .mean(2)?;
    let kb_t = k_idx
        .to_dtype(DType::F32)?
        .reshape((1, n_kb, b_k, d_idx))?
        .mean(2)?;
    let scores_2d = qb_t.matmul(&kb_t.t()?)?.squeeze(0)?.to_vec2::<f32>()?;

    let sink_nb = sink_tokens.div_ceil(b_k).min(n_kb);
    let mut forced = vec![false; n_kb];
    let mut cands: Vec<(f32, usize)> = Vec::with_capacity(n_kb);
    let mut sel: Vec<usize> = Vec::with_capacity(target_nb + 4);
    let mut all_indices: Vec<u32> = Vec::with_capacity(n_qb * k_eff);

    for (qi, row) in scores_2d.iter().enumerate() {
        forced.fill(false);
        sel.clear();
        for bj in 0..sink_nb {
            forced[bj] = true;
            sel.push(bj);
        }
        let q_start = qi * b_q;
        let q_end = (qi + 1) * b_q;
        let lo_tok = q_start.saturating_sub(local_window);
        let hi_tok = (q_end + local_window).min(l);
        let lo_b = lo_tok / b_k;
        let hi_b = hi_tok.div_ceil(b_k).min(n_kb);
        for bj in lo_b..hi_b {
            if !forced[bj] && sel.len() < target_nb {
                forced[bj] = true;
                sel.push(bj);
            }
        }
        if sel.len() < target_nb {
            let rem = target_nb - sel.len();
            cands.clear();
            for (bj, &sc) in row.iter().enumerate() {
                if !forced[bj] {
                    cands.push((sc, bj));
                }
            }
            if cands.len() > rem {
                cands.select_nth_unstable_by(rem - 1, |a, b| {
                    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                cands.truncate(rem);
            }
            for &(_, bj) in &cands {
                sel.push(bj);
            }
        }
        sel.truncate(target_nb);
        sel.sort_unstable();
        for &bj in &sel {
            let base = (bj * b_k) as u32;
            for off in 0..(b_k as u32) {
                all_indices.push(base + off);
            }
        }
    }

    let q_c = q.contiguous()?;
    let k_c = k.contiguous()?;
    let v_c = v.contiguous()?;

    // Chunk query blocks so score tensor [h * cur_qb, b_q, k_eff] stays within 64M elements
    let max_qb_per_chunk = ((1usize << 26) / (h * b_q * k_eff)).clamp(1, n_qb);
    let mut chunks_out = Vec::with_capacity(n_qb.div_ceil(max_qb_per_chunk));

    for qb0 in (0..n_qb).step_by(max_qb_per_chunk) {
        let cur_qb = max_qb_per_chunk.min(n_qb - qb0);
        let idx_slice = &all_indices[qb0 * k_eff..(qb0 + cur_qb) * k_eff];
        let idx_gpu = Tensor::from_slice(idx_slice, cur_qb * k_eff, q.device())?;

        // Gather keys & values: [1, h, cur_qb * k_eff, d] -> [h * cur_qb, k_eff, d]
        let k_sel = k_c
            .index_select(&idx_gpu, 2)?
            .reshape((h * cur_qb, k_eff, d))?;
        let v_sel = v_c
            .index_select(&idx_gpu, 2)?
            .reshape((h * cur_qb, k_eff, d))?;
        let q_blk = q_c
            .narrow(2, qb0 * b_q, cur_qb * b_q)?
            .contiguous()?
            .reshape((h * cur_qb, b_q, d))?;

        let att = q_blk.matmul(&k_sel.t()?)?;
        let p = crate::ops::softmax_last_dim(&att)?;
        let out_blk = p.matmul(&v_sel)?.reshape((1, h, cur_qb * b_q, d))?;
        chunks_out.push(out_blk);
    }

    Tensor::cat(&chunks_out, 2)
}
