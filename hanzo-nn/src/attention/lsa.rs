//! Latent Sparse Attention (LSA) — Native & Distilled Sparse Attention.
//!
//! Reference: *Latent Sparse Attention* (Prefrontal AI, arXiv:2610.42147).
//!
//! Implements the complete LSA stack:
//! 1. **Index spaces**:
//!    - Position-free GQA head regrouping (`position_free`, `regroup`, `regroup_partial`)
//!    - MLA Cholesky isometric latent metric (`LatentMetric`, `regroup_latent`)
//!    - Learned / Distilled and training-free PCA projections (`LearnedProjection`, `pca_projections`)
//! 2. **Pooling & Mutual-Cheapest Ward Clustering**:
//!    - Consecutive pool sums (`pool_keys`)
//!    - Size-capped mutual-cheapest Ward merge rounds (`ward_rounds`, `ward_merge`, `build_buckets`)
//! 3. **1-bit Bucket Shapes**:
//!    - Within-bucket key covariance, 1-bit sign directions, directional variances, and trace floor
//!      (`Shapes`, `build_shapes`, `spread`, `shaped_scores`, `pack_shape_basis`, `unpack_shape_basis`)
//! 4. **Flat & Hierarchical (Tree) Bucket Routing**:
//!    - Multi-level Ward node pairing (`Hierarchy`, `build_hierarchy`, `growth_at`, `read_hierarchy`)
//!    - Early budget scheduling (`EarlyBudget`, `budget_at`)
//!    - Bucket expansion with sink & sliding-window deduplication (`expand_buckets`, `read`, `keep_set`)
//! 5. **Subquadratic Sparse Read & N-LSA Two-Path Differentiable Summaries**:
//!    - Gather-based sparse SDPA (`sparse_gather_attention`) and reference masked SDPA (`topk_attention`)
//!    - Second-order isotropic Gaussian log-mass summaries for unselected keys (`lsa_summary_terms`,
//!      `lsa_two_path_attention`)
//!    - Stateful incremental layer index (`LsaConfig`, `LsaLayerIndex`) for prefill & token-by-token decode.

use hanzo_ml::{DType, Device, Error, Result, Tensor, D};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const NORM_EPS: f32 = 1e-12;
const CENTROID_EPS: f32 = 1e-30;
const MERGE_COST_EPS: f32 = 1e-12;

// ============================================================================
// 1. Index Space (Position-Free Regrouping, MLA Latent Metric, Projections)
// ============================================================================

/// Where rotary position embedding sits inside the head dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RotaryLayout {
    /// First `rotary_dim` coordinates carry RoPE; trailing `head_dim - rotary_dim` are position-free.
    #[serde(alias = "Prefix", alias = "PREFIX")]
    Prefix,
    /// Last `rotary_dim` coordinates carry RoPE; leading `head_dim - rotary_dim` are position-free.
    #[serde(alias = "Suffix", alias = "SUFFIX")]
    Suffix,
    /// Whole head dimension is rotated; pass pre-RoPE queries and keys into `regroup`.
    #[serde(alias = "None", alias = "NONE")]
    None,
}

/// Normalize each row (last axis) to unit Euclidean length; zero rows stay zero.
pub fn rownorm(x: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sq = x_f32.sqr()?;
    let norm = sq
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .clamp(NORM_EPS as f64, f64::MAX)?;
    x_f32.broadcast_div(&norm)?.to_dtype(dtype)
}

/// Keep the unrotated slice of a `[..., head_dim]` tensor.
pub fn position_free(x: &Tensor, rotary_dim: usize, layout: RotaryLayout) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    if rotary_dim >= d {
        hanzo_ml::bail!(
            "rotary_dim ({rotary_dim}) must be smaller than head_dim ({d}) to leave a position-free slice"
        );
    }
    match layout {
        RotaryLayout::Prefix => x.narrow(D::Minus1, rotary_dim, d - rotary_dim),
        RotaryLayout::Suffix => x.narrow(D::Minus1, 0, d - rotary_dim),
        RotaryLayout::None => hanzo_ml::bail!(
            "RotaryLayout::None means the whole head is rotated; pass pre-RoPE tensors to regroup()"
        ),
    }
}

/// Regroup GQA heads into a single index vector per position and unit-normalize.
///
/// Queries are summed within each KV group and concatenated across KV heads;
/// keys are concatenated across KV heads. Before normalization:
/// `q_regrouped_i . k_regrouped_j == sum_h (q_{i,h} . k_{j, h / G})`.
///
/// - `queries`: `[B, Hq, Tq, D]`
/// - `keys`: `[B, Hkv, Tk, D]`
/// - Returns `(index_queries, index_keys)` of shapes `[B, Tq, Hkv * D]` and `[B, Tk, Hkv * D]`.
pub fn regroup(queries: &Tensor, keys: &Tensor) -> Result<(Tensor, Tensor)> {
    let (b, h_q, t_q, d) = queries.dims4()?;
    let (b_k, h_kv, t_k, d_k) = keys.dims4()?;
    if b != b_k || d != d_k {
        hanzo_ml::bail!(
            "regroup shape mismatch: queries {:?} vs keys {:?}",
            queries.dims(),
            keys.dims()
        );
    }
    if h_kv == 0 || h_q % h_kv != 0 {
        hanzo_ml::bail!("H_q ({h_q}) must be a positive multiple of H_kv ({h_kv})");
    }
    let group_size = h_q / h_kv;

    // [B, Hkv, G, Tq, D] -> sum(dim=2) -> [B, Hkv, Tq, D] -> transpose(1, 2) -> [B, Tq, Hkv * D]
    let grouped_q = queries
        .reshape((b, h_kv, group_size, t_q, d))?
        .sum(2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, t_q, h_kv * d))?;

    let flat_k = keys
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, t_k, h_kv * d))?;

    Ok((rownorm(&grouped_q)?, rownorm(&flat_k)?))
}

/// Strip rotated coordinates under partial RoPE and regroup GQA heads into unit index vectors.
pub fn regroup_partial(
    queries: &Tensor,
    keys: &Tensor,
    rotary_dim: usize,
    layout: RotaryLayout,
) -> Result<(Tensor, Tensor)> {
    let q_free = position_free(queries, rotary_dim, layout)?;
    let k_free = position_free(keys, rotary_dim, layout)?;
    regroup(&q_free, &k_free)
}

/// Isometric latent metric for Multi-Head Latent Attention (MLA) where `k_h = W_h c_KV`.
///
/// Precomputes the Cholesky factor `L` of `M = sum_h W_h^T W_h = L L^T` and `W_sum L^{-T}`.
#[derive(Debug, Clone)]
pub struct LatentMetric {
    /// Lower-triangular Cholesky factor `L` of shape `[r, r]`.
    pub chol: Tensor,
    /// `W_sum @ L^{-T}` of shape `[H, d_free, r]`.
    pub query_map: Tensor,
}

impl LatentMetric {
    /// Build the MLA latent metric from key up-projection `[H, d_free, r]`.
    pub fn from_key_up_projection(key_up_proj: &Tensor) -> Result<Self> {
        let (h, d_free, r) = key_up_proj.dims3()?;
        let device = key_up_proj.device();
        let dtype = key_up_proj.dtype();
        let w = key_up_proj.to_dtype(DType::F32)?;
        let w_vec = w.flatten_all()?.to_vec1::<f32>()?;

        // Compute M = sum_h W_h^T W_h in f64 for numerical stability: [r, r]
        let mut m = vec![0.0f64; r * r];
        for head in 0..h {
            let base = head * d_free * r;
            for d in 0..d_free {
                let row = &w_vec[base + d * r..base + (d + 1) * r];
                for i in 0..r {
                    let wi = row[i] as f64;
                    for j in 0..=i {
                        m[i * r + j] += wi * (row[j] as f64);
                    }
                }
            }
        }
        for i in 0..r {
            for j in (i + 1)..r {
                m[i * r + j] = m[j * r + i];
            }
        }

        // Cholesky factorization M = L L^T (with tiny diagonal jitter if needed)
        let mut l = vec![0.0f64; r * r];
        for i in 0..r {
            for j in 0..=i {
                let mut sum = m[i * r + j];
                for k in 0..j {
                    sum -= l[i * r + k] * l[j * r + k];
                }
                if i == j {
                    let diag = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
                    l[i * r + i] = diag;
                } else {
                    l[i * r + j] = sum / l[j * r + j];
                }
            }
        }

        // Invert L (lower triangular) -> L^{-1}
        let mut l_inv = vec![0.0f64; r * r];
        for i in 0..r {
            l_inv[i * r + i] = 1.0 / l[i * r + i];
            for j in 0..i {
                let mut sum = 0.0f64;
                for k in j..i {
                    sum += l[i * r + k] * l_inv[k * r + j];
                }
                l_inv[i * r + j] = -sum / l[i * r + i];
            }
        }

        // Compute query_map[h, d, j] = (W_h @ L^{-T})[d, j] = sum_k W_h[d, k] * L^{-1}[j, k]
        let mut q_map = vec![0.0f32; h * d_free * r];
        for head in 0..h {
            let base = head * d_free * r;
            for d in 0..d_free {
                let w_row = &w_vec[base + d * r..base + (d + 1) * r];
                for j in 0..r {
                    let mut acc = 0.0f64;
                    for k in 0..=j {
                        acc += (w_row[k] as f64) * l_inv[j * r + k];
                    }
                    q_map[base + d * r + j] = acc as f32;
                }
            }
        }

        let chol_f32: Vec<f32> = l.into_iter().map(|v| v as f32).collect();
        let chol = Tensor::from_vec(chol_f32, (r, r), device)?.to_dtype(dtype)?;
        let query_map = Tensor::from_vec(q_map, (h, d_free, r), device)?.to_dtype(dtype)?;
        Ok(Self { chol, query_map })
    }
}

/// Express unrotated MLA queries `[B, H, Tq, d_free]` and KV latents `[B, Tk, r]` in the latent space.
pub fn regroup_latent(
    queries_free: &Tensor,
    kv_latent: &Tensor,
    metric: &LatentMetric,
) -> Result<(Tensor, Tensor)> {
    let (b, h, t_q, d_free) = queries_free.dims4()?;
    let (_, _, r) = metric.query_map.dims3()?;
    // queries_free: [B, H, Tq, d_free] -> transpose(1, 2) -> [B, Tq, H * d_free]
    let q_flat = queries_free
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t_q, h * d_free))?;
    let map_flat = metric.query_map.reshape((h * d_free, r))?;
    let q_lat = q_flat.matmul(&map_flat)?.reshape((b, t_q, r))?;

    let (b_k, t_k, r_k) = kv_latent.dims3()?;
    let k_lat = kv_latent
        .reshape((b_k * t_k, r_k))?
        .matmul(&metric.chol)?
        .reshape((b_k, t_k, r))?;

    Ok((rownorm(&q_lat)?, rownorm(&k_lat)?))
}

/// Linear projections `W_Q^{(idx)}` and `W_K^{(idx)}` from `[input_dim]` to `[index_dim]`.
#[derive(Debug, Clone)]
pub struct LearnedProjection {
    pub query_proj: Tensor,
    pub key_proj: Tensor,
}

impl LearnedProjection {
    pub fn new(query_proj: Tensor, key_proj: Tensor) -> Result<Self> {
        if query_proj.dims() != key_proj.dims() || query_proj.rank() != 2 {
            hanzo_ml::bail!(
                "LearnedProjection requires matching 2D [input_dim, index_dim] tensors, got {:?} and {:?}",
                query_proj.dims(),
                key_proj.dims()
            );
        }
        Ok(Self {
            query_proj,
            key_proj,
        })
    }

    pub fn input_dim(&self) -> Result<usize> {
        self.query_proj.dim(0)
    }

    pub fn index_dim(&self) -> Result<usize> {
        self.query_proj.dim(1)
    }
}

/// Apply `LearnedProjection` to regrouped unnormalized queries and keys and L2 row-normalize.
pub fn project(
    queries: &Tensor,
    keys: &Tensor,
    projection: &LearnedProjection,
) -> Result<(Tensor, Tensor)> {
    let (b, h_q, t_q, d) = queries.dims4()?;
    let (_, h_kv, t_k, _) = keys.dims4()?;
    let group_size = h_q / h_kv;

    let q_in = queries
        .reshape((b, h_kv, group_size, t_q, d))?
        .sum(2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t_q, h_kv * d))?;
    let k_in = keys
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t_k, h_kv * d))?;

    let d_idx = projection.index_dim()?;
    let q_idx = q_in
        .matmul(&projection.query_proj)?
        .reshape((b, t_q, d_idx))?;
    let k_idx = k_in
        .matmul(&projection.key_proj)?
        .reshape((b, t_k, d_idx))?;

    Ok((rownorm(&q_idx)?, rownorm(&k_idx)?))
}

/// Strip rotated coordinates and project into `index_dim` via `LearnedProjection`.
pub fn project_partial(
    queries: &Tensor,
    keys: &Tensor,
    rotary_dim: usize,
    layout: RotaryLayout,
    projection: &LearnedProjection,
) -> Result<(Tensor, Tensor)> {
    let q_free = position_free(queries, rotary_dim, layout)?;
    let k_free = position_free(keys, rotary_dim, layout)?;
    project(&q_free, &k_free, projection)
}

/// Training-free PCA initialization of `LearnedProjection`: sets `W_Q^{(idx)} = W_K^{(idx)}`
/// to the top `index_dim` principal directions of the flattened unrotated keys.
pub fn pca_projections(keys_in: &Tensor, index_dim: usize) -> Result<LearnedProjection> {
    let dims = keys_in.dims();
    let d_in = *dims.last().ok_or_else(|| Error::msg("empty keys_in"))?;
    let n_rows = keys_in.elem_count() / d_in;
    if index_dim > d_in {
        hanzo_ml::bail!("index_dim ({index_dim}) cannot exceed input_dim ({d_in})");
    }
    let flat = keys_in
        .to_dtype(DType::F32)?
        .reshape((n_rows, d_in))?
        .to_vec2::<f32>()?;

    // Compute empirical covariance C = (1/N) X^T X of shape [d_in, d_in]
    let mut cov = vec![0.0f32; d_in * d_in];
    let inv_n = 1.0 / (n_rows.max(1) as f32);
    for row in &flat {
        for i in 0..d_in {
            let ri = row[i] * inv_n;
            let base = i * d_in;
            for j in i..d_in {
                cov[base + j] += ri * row[j];
            }
        }
    }
    for i in 0..d_in {
        for j in 0..i {
            cov[i * d_in + j] = cov[j * d_in + i];
        }
    }

    let (eigvecs, _eigvals) = top_k_symmetric_eigen(&cov, d_in, index_dim);
    // eigvecs is [index_dim, d_in]; transpose to [d_in, index_dim]
    let mut w = vec![0.0f32; d_in * index_dim];
    for r in 0..index_dim {
        for i in 0..d_in {
            w[i * index_dim + r] = eigvecs[r * d_in + i];
        }
    }
    let proj =
        Tensor::from_vec(w, (d_in, index_dim), keys_in.device())?.to_dtype(keys_in.dtype())?;
    LearnedProjection::new(proj.clone(), proj)
}

// ============================================================================
// 2. Pooling & Mutual-Cheapest Ward Hierarchical Clustering
// ============================================================================

/// Sum consecutive runs of `pool_size` index keys within each cluster block.
///
/// - `index_keys`: `[N, P * pool_size, d_idx]`
/// - Returns `(pool_sums, pool_sizes)` of shapes `[N, P, d_idx]` and `[N, P]`.
pub fn pool_keys(index_keys: &Tensor, pool_size: usize) -> Result<(Tensor, Vec<Vec<i32>>)> {
    let (n, n_cluster, d) = index_keys.dims3()?;
    if pool_size == 0 || n_cluster % pool_size != 0 {
        hanzo_ml::bail!("n_cluster ({n_cluster}) must be divisible by pool_size ({pool_size})");
    }
    let p = n_cluster / pool_size;
    let pool_sums = index_keys.reshape((n, p, pool_size, d))?.sum(2)?;
    let pool_sizes = vec![vec![pool_size as i32; p]; n];
    Ok((pool_sums, pool_sizes))
}

/// Core mutual-cheapest Ward merge algorithm on a single block of `P` pools/nodes.
///
/// - `sums`: `[P * d]` flattened key sums (modified in place as buckets merge)
/// - `sizes`: `[P]` initial key counts per pool
/// - `p`: number of initial pools
/// - `d`: index dimension
/// - `num_buckets`: target surviving bucket count `m`
/// - `bucket_cap`: maximum keys per bucket (`>= 2 * (n_cluster / num_buckets)`)
///
/// Returns `(surviving_sums [m * d], surviving_sizes [m], labels [P])`, where `labels[pool]`
/// is the surviving bucket index `0..m` in order of each bucket's earliest pool.
pub fn ward_rounds_block(
    mut sums: Vec<f32>,
    mut sizes: Vec<i32>,
    p: usize,
    d: usize,
    num_buckets: usize,
    bucket_cap: i32,
) -> Result<(Vec<f32>, Vec<i32>, Vec<usize>)> {
    if num_buckets == 0 || num_buckets > p {
        hanzo_ml::bail!("num_buckets ({num_buckets}) must be in 1..={p}");
    }
    let mut alive = vec![true; p];
    let mut alive_count = p;
    let mut owner: Vec<usize> = (0..p).collect();

    // Maintain norms ||s_b||_2 and dot-product table G[b, b'] = s_b . s_{b'}
    let mut norms = vec![0.0f32; p];
    for b in 0..p {
        let row = &sums[b * d..(b + 1) * d];
        norms[b] = row.iter().map(|x| x * x).sum::<f32>().sqrt();
    }

    let mut gram = vec![0.0f32; p * p];
    for b in 0..p {
        let rb = &sums[b * d..(b + 1) * d];
        for bp in b..p {
            let rbp = &sums[bp * d..(bp + 1) * d];
            let dot: f32 = rb.iter().zip(rbp.iter()).map(|(x, y)| x * y).sum();
            gram[b * p + bp] = dot;
            gram[bp * p + b] = dot;
        }
    }

    let mut best_partner = vec![usize::MAX; p];
    let mut best_cost = vec![f32::INFINITY; p];

    while alive_count > num_buckets {
        let needed = alive_count - num_buckets;

        for b in 0..p {
            best_partner[b] = usize::MAX;
            best_cost[b] = f32::INFINITY;
            if !alive[b] {
                continue;
            }
            let nb = norms[b];
            let sb = sizes[b];
            for bp in 0..p {
                if bp == b || !alive[bp] || sb + sizes[bp] > bucket_cap {
                    continue;
                }
                let nbp = norms[bp];
                let sq = (nb * nb + nbp * nbp + 2.0 * gram[b * p + bp]).max(MERGE_COST_EPS);
                let cost = nb + nbp - sq.sqrt();
                if cost < best_cost[b] {
                    best_cost[b] = cost;
                    best_partner[b] = bp;
                }
            }
        }

        // Collect mutual cheapest pairs (b < bp)
        let mut mutual: Vec<(f32, usize, usize)> = Vec::new();
        for b in 0..p {
            if !alive[b] {
                continue;
            }
            let bp = best_partner[b];
            if bp != usize::MAX && b < bp && alive[bp] && best_partner[bp] == b {
                mutual.push((best_cost[b], b, bp));
            }
        }

        if mutual.is_empty() {
            hanzo_ml::bail!(
                "ward_rounds_block: no allowed mutual pair while alive={alive_count} > target={num_buckets} (bucket_cap={bucket_cap})"
            );
        }

        // Sort by merge cost ascending, ties broken by lower bucket index
        mutual.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let merges = mutual.len().min(needed);
        for &(_cost, b, bp) in &mutual[..merges] {
            // Merge bp into b
            for k in 0..d {
                sums[b * d + k] += sums[bp * d + k];
                sums[bp * d + k] = 0.0;
            }
            sizes[b] += sizes[bp];
            sizes[bp] = 0;
            alive[bp] = false;
            alive_count -= 1;

            // Update gram row/col for b and norm[b]
            for other in 0..p {
                if other != b && alive[other] {
                    let g = gram[b * p + other] + gram[bp * p + other];
                    gram[b * p + other] = g;
                    gram[other * p + b] = g;
                }
            }
            let rb = &sums[b * d..(b + 1) * d];
            let self_dot: f32 = rb.iter().map(|x| x * x).sum();
            gram[b * p + b] = self_dot;
            norms[b] = self_dot.sqrt();

            for pool_owner in &mut owner {
                if *pool_owner == bp {
                    *pool_owner = b;
                }
            }
        }
    }

    // Number surviving buckets 0..num_buckets in order of their first pool (which is ascending `b`)
    let mut survivor_rank = vec![usize::MAX; p];
    let mut surviving_sums = Vec::with_capacity(num_buckets * d);
    let mut surviving_sizes = Vec::with_capacity(num_buckets);
    let mut rank = 0usize;
    for b in 0..p {
        if alive[b] {
            survivor_rank[b] = rank;
            surviving_sums.extend_from_slice(&sums[b * d..(b + 1) * d]);
            surviving_sizes.push(sizes[b]);
            rank += 1;
        }
    }

    let labels: Vec<usize> = owner.into_iter().map(|o| survivor_rank[o]).collect();
    Ok((surviving_sums, surviving_sizes, labels))
}

/// Batched Ward clustering Output on `N` blocks.
#[derive(Debug, Clone)]
pub struct WardMergeOutput {
    /// Unit-length bucket centroids `[N, m, d_idx]`.
    pub centroids: Tensor,
    /// Number of keys in each bucket `[N][m]`.
    pub bucket_sizes: Vec<Vec<i32>>,
    /// Padded global key indices `[N][m][bucket_cap]` (`-1` where unused).
    pub bucket_members: Vec<Vec<Vec<i32>>>,
    /// Pool-to-bucket assignment `[N][P]`.
    pub labels: Vec<Vec<usize>>,
}

/// Turn pool-to-bucket labels `[P]` into padded global key indices `[m][bucket_cap]`.
pub fn members_from_labels_block(
    labels: &[usize],
    num_buckets: usize,
    pool_size: usize,
    bucket_cap: usize,
    block_offset: i32,
) -> Vec<Vec<i32>> {
    let mut buckets = vec![Vec::with_capacity(bucket_cap); num_buckets];
    for (pool_idx, &bucket_id) in labels.iter().enumerate() {
        let base = block_offset + (pool_idx * pool_size) as i32;
        for offset in 0..(pool_size as i32) {
            if buckets[bucket_id].len() < bucket_cap {
                buckets[bucket_id].push(base + offset);
            }
        }
    }
    for member_list in &mut buckets {
        member_list.resize(bucket_cap, -1);
    }
    buckets
}

/// Pool and Ward-cluster `N` closed blocks of `n_cluster` index keys into `m = n_cluster / bucket_size` buckets.
///
/// - `index_keys`: `[N, n_cluster, d_idx]`
/// - `block_offsets`: optional starting global key position for each of the `N` blocks (defaults to `block * n_cluster`).
pub fn ward_merge(
    index_keys: &Tensor,
    pool_size: usize,
    bucket_size: usize,
    bucket_cap: usize,
    block_offsets: Option<&[i32]>,
) -> Result<WardMergeOutput> {
    let (n, n_cluster, d) = index_keys.dims3()?;
    if pool_size == 0 || n_cluster % pool_size != 0 {
        hanzo_ml::bail!("n_cluster ({n_cluster}) must be divisible by pool_size ({pool_size})");
    }
    if bucket_size == 0 || n_cluster % bucket_size != 0 {
        hanzo_ml::bail!("n_cluster ({n_cluster}) must be divisible by bucket_size ({bucket_size})");
    }
    if bucket_cap < 2 * bucket_size {
        hanzo_ml::bail!(
            "bucket_cap ({bucket_cap}) must be at least 2 * bucket_size ({}) so a valid merge pair always exists",
            2 * bucket_size
        );
    }

    let p = n_cluster / pool_size;
    let m = n_cluster / bucket_size;
    let (pool_sums_tensor, _) = pool_keys(index_keys, pool_size)?;
    let pool_sums_flat = pool_sums_tensor
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let default_offsets: Vec<i32>;
    let offsets: &[i32] = match block_offsets {
        Some(o) => {
            if o.len() != n {
                hanzo_ml::bail!("block_offsets len {} != N ({n})", o.len());
            }
            o
        }
        None => {
            default_offsets = (0..n).map(|blk| (blk * n_cluster) as i32).collect();
            &default_offsets
        }
    };

    // Run mutual-cheapest Ward rounds in parallel across the N blocks
    type WardBlockResult = (Vec<f32>, Vec<i32>, Vec<Vec<i32>>, Vec<usize>);
    let per_block: Result<Vec<WardBlockResult>> = (0..n)
        .into_par_iter()
        .map(|blk| {
            let start = blk * p * d;
            let block_sums = pool_sums_flat[start..start + p * d].to_vec();
            let block_sizes = vec![pool_size as i32; p];
            let (mut surv_sums, surv_sizes, labels) =
                ward_rounds_block(block_sums, block_sizes, p, d, m, bucket_cap as i32)?;
            // Normalize surviving sums into unit centroids
            for b in 0..m {
                let slice = &mut surv_sums[b * d..(b + 1) * d];
                let norm = slice
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(CENTROID_EPS);
                for val in slice.iter_mut() {
                    *val /= norm;
                }
            }
            let members =
                members_from_labels_block(&labels, m, pool_size, bucket_cap, offsets[blk]);
            Ok((surv_sums, surv_sizes, members, labels))
        })
        .collect();

    let per_block = per_block?;
    let mut all_centroids = Vec::with_capacity(n * m * d);
    let mut bucket_sizes = Vec::with_capacity(n);
    let mut bucket_members = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for (c, sz, mem, lbl) in per_block {
        all_centroids.extend_from_slice(&c);
        bucket_sizes.push(sz);
        bucket_members.push(mem);
        labels.push(lbl);
    }

    let centroids = Tensor::from_vec(all_centroids, (n, m, d), index_keys.device())?
        .to_dtype(index_keys.dtype())?;

    Ok(WardMergeOutput {
        centroids,
        bucket_sizes,
        bucket_members,
        labels,
    })
}

/// Clustered buckets produced by `build_buckets` across a batch `[B, T, d_idx]`.
#[derive(Debug, Clone)]
pub struct Buckets {
    /// Unit-length bucket centroids `[B, B_total, d_idx]`.
    pub centroids: Tensor,
    /// Global key indices in each bucket `[B][B_total][bucket_cap]` (`-1` where padded).
    pub bucket_members: Vec<Vec<Vec<i32>>>,
    /// Number of keys in each bucket `[B][B_total]`.
    pub bucket_sizes: Vec<Vec<i32>>,
    /// Number of buckets produced per closed cluster block (`n_cluster / bucket_size`).
    pub buckets_per_block: usize,
    /// Number of closed cluster blocks (`T / n_cluster`).
    pub num_closed_blocks: usize,
    /// Optional 1-bit covariance shape descriptors (`[B]`).
    pub shapes: Option<Vec<Shapes>>,
}

/// Partition closed cluster blocks of `index_keys` `[B, T, d_idx]` and Ward-cluster each block into buckets.
pub fn build_buckets(
    index_keys: &Tensor,
    window_size: usize,
    pool_size: usize,
    bucket_size: usize,
    bucket_cap: usize,
) -> Result<Buckets> {
    let (b_sz, seq_len, d_idx) = index_keys.dims3()?;
    if window_size == 0
        || !window_size.is_multiple_of(pool_size)
        || !window_size.is_multiple_of(bucket_size)
    {
        hanzo_ml::bail!(
            "window_size ({window_size}) must be a positive multiple of pool_size ({pool_size}) and bucket_size ({bucket_size})"
        );
    }
    let num_closed_blocks = seq_len / window_size;
    let buckets_per_block = window_size / bucket_size;
    let total_buckets = num_closed_blocks * buckets_per_block;

    if num_closed_blocks == 0 {
        let centroids = Tensor::zeros((b_sz, 0, d_idx), index_keys.dtype(), index_keys.device())?;
        return Ok(Buckets {
            centroids,
            bucket_members: vec![Vec::new(); b_sz],
            bucket_sizes: vec![Vec::new(); b_sz],
            buckets_per_block,
            num_closed_blocks: 0,
            shapes: None,
        });
    }

    let closed_len = num_closed_blocks * window_size;
    let closed_keys = index_keys
        .narrow(1, 0, closed_len)?
        .contiguous()?
        .reshape((b_sz * num_closed_blocks, window_size, d_idx))?;

    let mut offsets = Vec::with_capacity(b_sz * num_closed_blocks);
    for _ in 0..b_sz {
        for blk in 0..num_closed_blocks {
            offsets.push((blk * window_size) as i32);
        }
    }

    let merged = ward_merge(
        &closed_keys,
        pool_size,
        bucket_size,
        bucket_cap,
        Some(&offsets),
    )?;
    let centroids = merged.centroids.reshape((b_sz, total_buckets, d_idx))?;

    let mut bucket_members = vec![Vec::with_capacity(total_buckets); b_sz];
    let mut bucket_sizes = vec![Vec::with_capacity(total_buckets); b_sz];
    for batch_idx in 0..b_sz {
        for blk in 0..num_closed_blocks {
            let flat_idx = batch_idx * num_closed_blocks + blk;
            bucket_members[batch_idx].extend_from_slice(&merged.bucket_members[flat_idx]);
            bucket_sizes[batch_idx].extend_from_slice(&merged.bucket_sizes[flat_idx]);
        }
    }

    Ok(Buckets {
        centroids,
        bucket_members,
        bucket_sizes,
        buckets_per_block,
        num_closed_blocks,
        shapes: None,
    })
}

// ============================================================================
// 3. Bucket Shapes (Covariance, 1-Bit Sign Basis, Directional Spread)
// ============================================================================

/// Low-rank 1-bit sign basis + residual floor for within-bucket key spread.
#[derive(Debug, Clone)]
pub struct Shapes {
    /// Sign-quantized leading directions `[B_total, rank, d_idx]` with entries `+/- 1 / sqrt(d_idx)`.
    pub directions: Tensor,
    /// Variance of keys along each leading direction `[B_total, rank]`.
    pub variances: Tensor,
    /// Isotropic residual variance across the remaining `d_idx - rank` dimensions `[B_total]`.
    pub floor: Tensor,
}

/// Top-`rank` eigenvalues and eigenvectors of a symmetric positive semi-definite matrix `a` (`[d, d]`).
/// Uses power iteration with Hotelling deflation and deterministic sign convention
/// (the coordinate of largest magnitude in each eigenvector is made positive).
fn top_k_symmetric_eigen(a: &[f32], d: usize, rank: usize) -> (Vec<f32>, Vec<f32>) {
    let mut mat = a.to_vec();
    let mut vecs = vec![0.0f32; rank * d];
    let mut vals = vec![0.0f32; rank];

    for r in 0..rank {
        // Initialize deterministic unit vector orthogonalized against previous eigenvectors
        let mut v = vec![0.0f32; d];
        for i in 0..d {
            // Choose starting seed direction along highest remaining diagonal + slight break
            v[i] = ((i + r * 7 + 1) as f32).sin() + if i == r % d { 2.0 } else { 0.0 };
        }
        // Orthogonalize against previous eigenvectors
        for prev in 0..r {
            let pv = &vecs[prev * d..(prev + 1) * d];
            let dot: f32 = v.iter().zip(pv.iter()).map(|(x, y)| x * y).sum();
            for i in 0..d {
                v[i] -= dot * pv[i];
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v {
            *x /= norm;
        }

        let mut eigval = 0.0f32;
        for _iter in 0..32 {
            let mut w = vec![0.0f32; d];
            for i in 0..d {
                let row = &mat[i * d..(i + 1) * d];
                w[i] = row.iter().zip(v.iter()).map(|(x, y)| x * y).sum();
            }
            for prev in 0..r {
                let pv = &vecs[prev * d..(prev + 1) * d];
                let dot: f32 = w.iter().zip(pv.iter()).map(|(x, y)| x * y).sum();
                for i in 0..d {
                    w[i] -= dot * pv[i];
                }
            }
            let w_norm = w.iter().map(|x| x * x).sum::<f32>().sqrt();
            if w_norm < 1e-12 {
                eigval = 0.0;
                break;
            }
            eigval = w_norm;
            let mut diff = 0.0f32;
            for i in 0..d {
                let next_vi = w[i] / w_norm;
                diff += (next_vi - v[i]).abs();
                v[i] = next_vi;
            }
            if diff < 1e-6 {
                break;
            }
        }

        // Deterministic sign: make the entry of largest magnitude positive
        let mut max_idx = 0usize;
        let mut max_abs = -1.0f32;
        for (i, &val) in v.iter().enumerate() {
            if val.abs() > max_abs {
                max_abs = val.abs();
                max_idx = i;
            }
        }
        if v[max_idx] < 0.0 {
            for val in &mut v {
                *val = -*val;
            }
        }

        // Deflate mat -= eigval * v * v^T
        for i in 0..d {
            for j in 0..d {
                mat[i * d + j] -= eigval * v[i] * v[j];
            }
        }

        vecs[r * d..(r + 1) * d].copy_from_slice(&v);
        vals[r] = eigval.max(0.0);
    }

    (vecs, vals)
}

/// Build 1-bit sign-quantized covariance shapes for a single batch element's buckets.
///
/// - `index_keys`: `[T, d_idx]`
/// - `centroids`: `[B_total, d_idx]`
/// - `bucket_members`: `[B_total][bucket_cap]`
/// - `bucket_sizes`: `[B_total]`
pub fn build_shapes(
    index_keys: &Tensor,
    centroids: &Tensor,
    bucket_members: &[Vec<i32>],
    bucket_sizes: &[i32],
    rank: usize,
) -> Result<Shapes> {
    let (_t, d) = index_keys.dims2()?;
    let (b_total, d_c) = centroids.dims2()?;
    if d != d_c || rank == 0 || rank >= d {
        hanzo_ml::bail!("build_shapes requires 0 < rank ({rank}) < d_idx ({d}) and matching dims");
    }

    let keys_flat = index_keys
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let centroids_flat = centroids
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scale = 1.0f32 / (d as f32).sqrt();

    let per_bucket: Vec<(Vec<f32>, Vec<f32>, f32)> = (0..b_total)
        .into_par_iter()
        .map(|b| {
            let count = bucket_sizes[b].max(1) as f32;
            let c = &centroids_flat[b * d..(b + 1) * d];
            let members = &bucket_members[b];

            let mut cov = vec![0.0f32; d * d];
            for &idx in members {
                if idx < 0 {
                    continue;
                }
                let k = &keys_flat[(idx as usize) * d..(idx as usize + 1) * d];
                for i in 0..d {
                    let di = k[i] - c[i];
                    let row_off = i * d;
                    for j in i..d {
                        cov[row_off + j] += di * (k[j] - c[j]);
                    }
                }
            }
            let inv_count = 1.0 / count;
            let mut trace = 0.0f32;
            for i in 0..d {
                cov[i * d + i] *= inv_count;
                trace += cov[i * d + i];
                for j in (i + 1)..d {
                    let v = cov[i * d + j] * inv_count;
                    cov[i * d + j] = v;
                    cov[j * d + i] = v;
                }
            }

            let (eigvecs, eigvals) = top_k_symmetric_eigen(&cov, d, rank);
            let mut dirs = vec![0.0f32; rank * d];
            for r in 0..rank {
                for i in 0..d {
                    let v = eigvecs[r * d + i];
                    dirs[r * d + i] = if v >= 0.0 { scale } else { -scale };
                }
            }
            let top_sum: f32 = eigvals.iter().sum();
            let floor = ((trace - top_sum) / ((d - rank) as f32)).max(0.0);
            (dirs, eigvals, floor)
        })
        .collect();

    let mut all_dirs = Vec::with_capacity(b_total * rank * d);
    let mut all_vars = Vec::with_capacity(b_total * rank);
    let mut all_floors = Vec::with_capacity(b_total);
    for (dirs, vars, fl) in per_bucket {
        all_dirs.extend_from_slice(&dirs);
        all_vars.extend_from_slice(&vars);
        all_floors.push(fl);
    }

    let device = index_keys.device();
    let dtype = index_keys.dtype();
    Ok(Shapes {
        directions: Tensor::from_vec(all_dirs, (b_total, rank, d), device)?.to_dtype(dtype)?,
        variances: Tensor::from_vec(all_vars, (b_total, rank), device)?.to_dtype(dtype)?,
        floor: Tensor::from_vec(all_floors, (b_total,), device)?.to_dtype(dtype)?,
    })
}

/// Attach `Shapes` to every batch element of `Buckets`.
pub fn shape_buckets(buckets: &mut Buckets, index_keys: &Tensor, rank: usize) -> Result<()> {
    let b_sz = index_keys.dim(0)?;
    let mut shapes_vec = Vec::with_capacity(b_sz);
    for b in 0..b_sz {
        let k_b = index_keys.narrow(0, b, 1)?.squeeze(0)?;
        let c_b = buckets.centroids.narrow(0, b, 1)?.squeeze(0)?;
        let s = build_shapes(
            &k_b,
            &c_b,
            &buckets.bucket_members[b],
            &buckets.bucket_sizes[b],
            rank,
        )?;
        shapes_vec.push(s);
    }
    buckets.shapes = Some(shapes_vec);
    Ok(())
}

/// Pack 1-bit sign directions `[B_total, rank, d_idx]` (where `d_idx % 8 == 0`) into `[B_total, rank, d_idx / 8]` `u8` bits.
pub fn pack_shape_basis(directions: &Tensor) -> Result<Vec<u8>> {
    let (b_total, rank, d) = directions.dims3()?;
    if d % 8 != 0 {
        hanzo_ml::bail!("pack_shape_basis requires d_idx ({d}) divisible by 8");
    }
    let flat = directions
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let packed_bytes = b_total * rank * (d / 8);
    let mut packed = vec![0u8; packed_bytes];
    for idx in 0..packed_bytes {
        let base = idx * 8;
        let mut byte = 0u8;
        for bit in 0..8 {
            if flat[base + bit] >= 0.0 {
                byte |= 1u8 << bit;
            }
        }
        packed[idx] = byte;
    }
    Ok(packed)
}

/// Unpack `[B_total, rank, d_idx / 8]` `u8` sign bits back into `[B_total, rank, d_idx]` with `+/- 1 / sqrt(d_idx)`.
pub fn unpack_shape_basis(
    packed: &[u8],
    b_total: usize,
    rank: usize,
    d_idx: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    if !d_idx.is_multiple_of(8) || packed.len() != b_total * rank * (d_idx / 8) {
        hanzo_ml::bail!("unpack_shape_basis length mismatch");
    }
    let scale = 1.0f32 / (d_idx as f32).sqrt();
    let mut flat = Vec::with_capacity(b_total * rank * d_idx);
    for &byte in packed {
        for bit in 0..8 {
            if (byte >> bit) & 1 == 1 {
                flat.push(scale);
            } else {
                flat.push(-scale);
            }
        }
    }
    Tensor::from_vec(flat, (b_total, rank, d_idx), device)?.to_dtype(dtype)
}

/// Directional key standard deviation along `index_queries` `[Tq, d_idx]` for each bucket in `shapes`:
/// `sqrt(sum_j lambda_{b,j} (q_i . u_{b,j})^2 + floor_b * (1 - sum_j (q_i . u_{b,j})^2))`.
pub fn spread(index_queries: &Tensor, shapes: &Shapes) -> Result<Tensor> {
    let (t_q, d) = index_queries.dims2()?;
    let (b_total, rank, _) = shapes.directions.dims3()?;
    if b_total == 0 {
        return Tensor::zeros((t_q, 0), DType::F32, index_queries.device());
    }
    let q_f32 = index_queries.to_dtype(DType::F32)?;
    let dirs_flat = shapes
        .directions
        .to_dtype(DType::F32)?
        .reshape((b_total * rank, d))?;
    // proj: [Tq, B_total * rank] -> [Tq, B_total, rank]
    let proj_sq = q_f32
        .matmul(&dirs_flat.t()?)?
        .reshape((t_q, b_total, rank))?
        .sqr()?;

    let vars = shapes
        .variances
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .broadcast_as((t_q, b_total, rank))?;
    let captured = proj_sq.mul(&vars)?.sum(D::Minus1)?; // [Tq, B_total]
    let proj_norm_sq = proj_sq.sum(D::Minus1)?; // [Tq, B_total]
    let ones = Tensor::ones((t_q, b_total), DType::F32, index_queries.device())?;
    let residual_weight = ones.sub(&proj_norm_sq)?.clamp(0.0, 1.0)?;
    let floor = shapes
        .floor
        .to_dtype(DType::F32)?
        .unsqueeze(0)?
        .broadcast_as((t_q, b_total))?;
    let total_var = captured
        .add(&residual_weight.mul(&floor)?)?
        .clamp(0.0, f64::MAX)?;
    total_var.sqrt()
}

/// Bucket score combining centroid alignment and directional spread: `q . c_b + deviations * spread(q, b)`.
pub fn shaped_scores(
    index_queries: &Tensor,
    centroids: &Tensor,
    shapes: &Shapes,
    deviations: f32,
) -> Result<Tensor> {
    let q_f32 = index_queries.to_dtype(DType::F32)?;
    let c_f32 = centroids.to_dtype(DType::F32)?;
    let base = q_f32.matmul(&c_f32.t()?)?;
    if deviations == 0.0 {
        return Ok(base);
    }
    let sp = spread(&q_f32, shapes)?;
    base.add(&(sp * (deviations as f64))?)
}

// ============================================================================
// 4. Tree / Hierarchical Bucket Routing & Early Budget
// ============================================================================

/// Smaller `top_p` / `top_k` budget applied to early sequence positions `< position`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EarlyBudget {
    pub position: usize,
    pub top_p: usize,
    pub top_k: usize,
}

/// Return `(effective_top_p, effective_top_k)` at sequence position `pos`.
#[inline]
pub fn budget_at(
    pos: usize,
    top_p: usize,
    top_k: usize,
    early_budget: Option<EarlyBudget>,
) -> (usize, usize) {
    if let Some(eb) = early_budget {
        if pos < eb.position {
            return (eb.top_p, eb.top_k);
        }
    }
    (top_p, top_k)
}

/// Multi-level tree of node buckets built above level-0 buckets (for a single sequence).
#[derive(Debug, Clone)]
pub struct Hierarchy {
    /// `centroids[l]` has shape `[N_l, d_idx]` for level `l = 0..=top_level`.
    pub centroids: Vec<Tensor>,
    /// `children[l]` for `l >= 1` has shape `[N_l][node_bucket_size]` indexing into level `l - 1`.
    pub children: Vec<Vec<Vec<usize>>>,
    /// Number of level-0 buckets per closed cluster block (`m`).
    pub buckets_per_block: usize,
    /// How many level `l - 1` buckets must accumulate before a node block closes (`node_cluster_block_size`).
    pub node_cluster_block_size: usize,
    /// Exact number of children per node bucket (`node_bucket_size`, a power of two).
    pub node_bucket_size: usize,
    /// Optional level-0 shapes.
    pub shapes: Option<Shapes>,
}

impl Hierarchy {
    pub fn top_level(&self) -> usize {
        self.centroids.len().saturating_sub(1)
    }
}

/// How much of each tree level exists when a query's cluster block opens.
#[derive(Debug, Clone)]
pub struct Growth {
    /// `available_at_level[l]` is the number of level-`l` buckets that exist at that moment.
    pub available_at_level: Vec<usize>,
    /// `covered_by_parent[l]` is how many of those level-`l` buckets are already grouped into level `l + 1`.
    pub covered_by_parent: Vec<usize>,
    /// Highest level `l >= 1` with at least one node bucket (or `0` if none).
    pub highest_grown_level: usize,
}

/// Pair `n_blocks` node blocks of `node_cluster_block_size` child sums into `node_bucket_size`-child node buckets
/// via `log2(node_bucket_size)` rounds of Ward pairing (`bucket_cap = 2` at each halving step).
pub fn pair_into_node_buckets(
    child_sums: &[f32],
    n_blocks: usize,
    node_cluster_block_size: usize,
    d: usize,
    node_bucket_size: usize,
) -> Result<(Vec<f32>, Vec<Vec<usize>>)> {
    if !node_bucket_size.is_power_of_two() || node_bucket_size < 2 {
        hanzo_ml::bail!("node_bucket_size ({node_bucket_size}) must be a power of two >= 2");
    }
    if !node_cluster_block_size.is_multiple_of(node_bucket_size) {
        hanzo_ml::bail!(
            "node_cluster_block_size ({node_cluster_block_size}) must be divisible by node_bucket_size ({node_bucket_size})"
        );
    }
    let halvings = node_bucket_size.trailing_zeros() as usize;
    let out_per_block = node_cluster_block_size / node_bucket_size;

    let mut all_parent_sums = Vec::with_capacity(n_blocks * out_per_block * d);
    let mut all_children = Vec::with_capacity(n_blocks * out_per_block);

    for blk in 0..n_blocks {
        let base_child = blk * node_cluster_block_size;
        let mut cur_p = node_cluster_block_size;
        let mut cur_sums =
            child_sums[base_child * d..(base_child + node_cluster_block_size) * d].to_vec();
        let mut members: Vec<Vec<usize>> = (0..node_cluster_block_size)
            .map(|i| vec![base_child + i])
            .collect();

        for _ in 0..halvings {
            let next_p = cur_p / 2;
            let sizes = vec![1i32; cur_p];
            let (surv_sums, _surv_sizes, labels) =
                ward_rounds_block(cur_sums, sizes, cur_p, d, next_p, 2)?;
            let mut next_members = vec![Vec::new(); next_p];
            for (idx, &lbl) in labels.iter().enumerate() {
                next_members[lbl].extend_from_slice(&members[idx]);
            }
            cur_sums = surv_sums;
            members = next_members;
            cur_p = next_p;
        }

        all_parent_sums.extend_from_slice(&cur_sums);
        all_children.extend(members);
    }

    Ok((all_parent_sums, all_children))
}

/// Build a multi-level `Hierarchy` above level-0 buckets for a single sequence.
///
/// - `buckets`: single-batch (`B = 1`) or batch slice of `Buckets`.
pub fn build_hierarchy(
    centroids_l0: &Tensor,
    bucket_sizes_l0: &[i32],
    buckets_per_block: usize,
    shapes_l0: Option<Shapes>,
    top_level: usize,
    node_bucket_size: usize,
    node_cluster_block_size: usize,
) -> Result<Hierarchy> {
    let (b_total, d) = centroids_l0.dims2()?;
    let device = centroids_l0.device();
    let dtype = centroids_l0.dtype();

    let c0_vec = centroids_l0
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Unnormalized key sum approximations: centroid * bucket_size (since keys are unit length,
    // weighting by bucket_size gives the exact sum direction when built from centroids).
    let mut cur_sums = vec![0.0f32; b_total * d];
    for b in 0..b_total {
        let w = (bucket_sizes_l0[b].max(1)) as f32;
        for k in 0..d {
            cur_sums[b * d + k] = c0_vec[b * d + k] * w;
        }
    }
    let mut cur_count = b_total;

    let mut centroids = Vec::with_capacity(top_level + 1);
    let mut children = Vec::with_capacity(top_level + 1);
    centroids.push(centroids_l0.clone());
    children.push(Vec::new()); // level 0 has no child node buckets (its members are keys)

    for _level in 1..=top_level {
        let n_blocks = cur_count / node_cluster_block_size;
        if n_blocks == 0 {
            centroids.push(Tensor::zeros((0, d), dtype, device)?);
            children.push(Vec::new());
            cur_sums.clear();
            cur_count = 0;
            continue;
        }
        let (parent_sums, parent_children) = pair_into_node_buckets(
            &cur_sums,
            n_blocks,
            node_cluster_block_size,
            d,
            node_bucket_size,
        )?;
        let n_parents = parent_children.len();
        let mut parent_centroids = parent_sums.clone();
        for p in 0..n_parents {
            let slice = &mut parent_centroids[p * d..(p + 1) * d];
            let norm = slice
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(CENTROID_EPS);
            for v in slice.iter_mut() {
                *v /= norm;
            }
        }
        centroids
            .push(Tensor::from_vec(parent_centroids, (n_parents, d), device)?.to_dtype(dtype)?);
        children.push(parent_children);
        cur_sums = parent_sums;
        cur_count = n_parents;
    }

    Ok(Hierarchy {
        centroids,
        children,
        buckets_per_block,
        node_cluster_block_size,
        node_bucket_size,
        shapes: shapes_l0,
    })
}

/// Compute `Growth` at sequence position `query_pos`.
pub fn growth_at(hierarchy: &Hierarchy, query_pos: usize, window_size: usize) -> Growth {
    let top_level = hierarchy.top_level();
    let mut available_at_level = vec![0usize; top_level + 1];
    let mut covered_by_parent = vec![0usize; top_level + 1];

    let closed_blocks = query_pos / window_size;
    available_at_level[0] = closed_blocks * hierarchy.buckets_per_block;

    let mut highest_grown_level = 0usize;
    for l in 1..=top_level {
        let child_avail = available_at_level[l - 1];
        let closed_node_blocks = child_avail / hierarchy.node_cluster_block_size;
        covered_by_parent[l - 1] = closed_node_blocks * hierarchy.node_cluster_block_size;
        available_at_level[l] =
            closed_node_blocks * (hierarchy.node_cluster_block_size / hierarchy.node_bucket_size);
        if available_at_level[l] > 0 {
            highest_grown_level = l;
        }
    }

    Growth {
        available_at_level,
        covered_by_parent,
        highest_grown_level,
    }
}

/// Top-down beam search over `Hierarchy` for a single query vector `q` (`[d_idx]`),
/// returning the selected level-0 bucket indices sorted by descending score.
pub fn beam_search_single_query(
    q: &[f32],
    l0_centroids: &[f32],
    level_centroids: &[Vec<f32>],
    hierarchy: &Hierarchy,
    growth: &Growth,
    top_p: usize,
    beam_width: usize,
    start_nodes: usize,
) -> Vec<usize> {
    let d = q.len();
    let avail_l0 = growth.available_at_level[0];
    if avail_l0 == 0 || top_p == 0 {
        return Vec::new();
    }

    // Determine start level: lowest level >= 1 with available <= start_nodes, else highest_grown_level
    let mut start_level = 0usize;
    if growth.highest_grown_level >= 1 {
        start_level = growth.highest_grown_level;
        for l in 1..=growth.highest_grown_level {
            if growth.available_at_level[l] > 0 && growth.available_at_level[l] <= start_nodes {
                start_level = l;
                break;
            }
        }
    }

    let dot = |centroids_flat: &[f32], idx: usize| -> f32 {
        let row = &centroids_flat[idx * d..(idx + 1) * d];
        q.iter().zip(row.iter()).map(|(a, b)| a * b).sum()
    };

    if start_level == 0 {
        // Flat score over all available level-0 buckets
        let mut scored: Vec<(f32, usize)> =
            (0..avail_l0).map(|b| (dot(l0_centroids, b), b)).collect();
        let keep = top_p.min(scored.len());
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        return scored[..keep].iter().map(|&(_, b)| b).collect();
    }

    // Score all available node buckets at `start_level`
    let avail_start = growth.available_at_level[start_level];
    let mut beam: Vec<usize> = {
        let c_flat = &level_centroids[start_level];
        let mut scored: Vec<(f32, usize)> = (0..avail_start)
            .map(|idx| (dot(c_flat, idx), idx))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let keep = beam_width.min(scored.len());
        scored[..keep].iter().map(|&(_, idx)| idx).collect()
    };

    // Descend through intermediate levels start_level - 1 down to 1
    for level in (1..start_level).rev() {
        let mut candidates: Vec<usize> = Vec::with_capacity(
            beam.len() * hierarchy.node_bucket_size
                + (growth.available_at_level[level] - growth.covered_by_parent[level]),
        );
        for &parent_idx in &beam {
            candidates.extend_from_slice(&hierarchy.children[level + 1][parent_idx]);
        }
        for open_idx in growth.covered_by_parent[level]..growth.available_at_level[level] {
            candidates.push(open_idx);
        }
        let c_flat = &level_centroids[level];
        let mut scored: Vec<(f32, usize)> = candidates
            .into_iter()
            .map(|idx| (dot(c_flat, idx), idx))
            .collect();
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
        let keep = beam_width.min(scored.len());
        beam = scored[..keep].iter().map(|&(_, idx)| idx).collect();
    }

    // Level 0: score children of level-1 beam alongside open uncovered level-0 buckets
    let mut l0_candidates: Vec<usize> = Vec::with_capacity(
        beam.len() * hierarchy.node_bucket_size
            + (growth.available_at_level[0] - growth.covered_by_parent[0]),
    );
    for &parent_idx in &beam {
        l0_candidates.extend_from_slice(&hierarchy.children[1][parent_idx]);
    }
    for open_idx in growth.covered_by_parent[0]..growth.available_at_level[0] {
        l0_candidates.push(open_idx);
    }

    let mut scored_l0: Vec<(f32, usize)> = l0_candidates
        .into_iter()
        .map(|b| (dot(l0_centroids, b), b))
        .collect();
    scored_l0.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let keep = top_p.min(scored_l0.len());
    scored_l0[..keep].iter().map(|&(_, b)| b).collect()
}

// ============================================================================
// 5. Bucket Expansion, Flat & Hierarchical Read, and Keep Set
// ============================================================================

/// Expand selected bucket IDs into up to `top_k` long-range key indices, filtering out `-1` padding,
/// sink positions (`< sink_size`), and sliding-window positions (`>= window_start`).
pub fn expand_buckets_single_query(
    selected_buckets: &[usize],
    bucket_members: &[Vec<i32>],
    query_pos: usize,
    window_size: usize,
    sink_size: usize,
    top_k: usize,
) -> Vec<i32> {
    if top_k == 0 || selected_buckets.is_empty() {
        return Vec::new();
    }
    let window_start = (query_pos + 1).saturating_sub(window_size) as i32;
    let sink_limit = sink_size as i32;
    let mut expanded = Vec::with_capacity(top_k);
    for &b in selected_buckets {
        if b >= bucket_members.len() {
            continue;
        }
        for &key_pos in &bucket_members[b] {
            if key_pos < 0 {
                continue;
            }
            if key_pos >= sink_limit && key_pos < window_start {
                expanded.push(key_pos);
                if expanded.len() >= top_k {
                    return expanded;
                }
            }
        }
    }
    expanded
}

/// Complete per-query index selection result across `[B, Tq]`.
#[derive(Debug, Clone)]
pub struct ReadSelection {
    /// Long-range keys unpacked from selected buckets `[B][Tq]` (each of length `<= top_k`, padded with `-1` when viewed as rectangular).
    pub expanded_keys: Vec<Vec<Vec<i32>>>,
    /// Selected level-0 bucket IDs per query `[B][Tq]` (for N-LSA summary path & diagnostics).
    pub selected_buckets: Vec<Vec<Vec<usize>>>,
    /// Full deduplicated causal key keep-set `sinks U expanded U window` per query `[B][Tq]`.
    pub keep_indices: Vec<Vec<Vec<u32>>>,
    ///Query starting global position offset (`kv_len - q_len`).
    pub query_offset: usize,
    pub window_size: usize,
    pub sink_size: usize,
    pub top_k: usize,
}

/// Build the full sorted causal keep-set `sinks U expanded U window` for a query at global position `query_pos`.
pub fn build_query_keep_set(
    expanded: &[i32],
    query_pos: usize,
    window_size: usize,
    sink_size: usize,
) -> Vec<u32> {
    let win_start = (query_pos + 1).saturating_sub(window_size);
    let sink_end = sink_size.min(win_start).min(query_pos + 1);
    let mut keep = Vec::with_capacity(sink_end + expanded.len() + (query_pos + 1 - win_start));
    for s in 0..sink_end {
        keep.push(s as u32);
    }
    for &k in expanded {
        if k >= 0 && (k as usize) <= query_pos {
            keep.push(k as u32);
        }
    }
    for w in win_start..=query_pos {
        keep.push(w as u32);
    }
    keep
}

/// Flat (or Shaped) LSA bucket read across batch `[B, Tq, d_idx]`.
///
/// Each query at global position `pos = query_offset + i` sees only buckets from cluster blocks
/// that closed before `pos`'s own block: `visible_buckets = (pos / window_size) * buckets_per_block`.
#[allow(clippy::too_many_arguments)]
pub fn read(
    index_queries: &Tensor,
    buckets: &Buckets,
    window_size: usize,
    sink_size: usize,
    top_p: usize,
    top_k: usize,
    deviations: f32,
    early_budget: Option<EarlyBudget>,
    query_offset: usize,
) -> Result<ReadSelection> {
    let (b_sz, t_q, d_idx) = index_queries.dims3()?;
    let b_total = buckets.centroids.dim(1)?;

    let mut all_expanded = Vec::with_capacity(b_sz);
    let mut all_selected_buckets = Vec::with_capacity(b_sz);
    let mut all_keep = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let q_b = index_queries
            .narrow(0, b, 1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let c_b = buckets
            .centroids
            .narrow(0, b, 1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;

        let scores_mat = if b_total == 0 {
            Vec::new()
        } else {
            let scores_t = if let Some(ref shapes_vec) = buckets.shapes {
                shaped_scores(&q_b, &c_b, &shapes_vec[b], deviations)?
            } else {
                q_b.matmul(&c_b.t()?)?
            };
            scores_t.flatten_all()?.to_vec1::<f32>()?
        };

        let members_b = &buckets.bucket_members[b];
        let b_per_blk = buckets.buckets_per_block;

        let per_q: Vec<(Vec<i32>, Vec<usize>, Vec<u32>)> = (0..t_q)
            .into_par_iter()
            .map(|i| {
                let pos = query_offset + i;
                let visible = ((pos / window_size) * b_per_blk).min(b_total);
                let (eff_p, eff_k) = budget_at(pos, top_p, top_k, early_budget);

                let selected: Vec<usize> = if visible == 0 || eff_p == 0 {
                    Vec::new()
                } else {
                    let row = &scores_mat[i * b_total..i * b_total + visible];
                    let mut pairs: Vec<(f32, usize)> = row
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(idx, s)| (s, idx))
                        .collect();
                    pairs.sort_by(|a, b| {
                        b.0.partial_cmp(&a.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    let keep_p = eff_p.min(visible);
                    pairs[..keep_p].iter().map(|&(_, idx)| idx).collect()
                };

                let expanded = expand_buckets_single_query(
                    &selected,
                    members_b,
                    pos,
                    window_size,
                    sink_size,
                    eff_k,
                );
                let keep = build_query_keep_set(&expanded, pos, window_size, sink_size);
                let _ = d_idx;
                (expanded, selected, keep)
            })
            .collect();

        let mut exp_b = Vec::with_capacity(t_q);
        let mut sel_b = Vec::with_capacity(t_q);
        let mut keep_b = Vec::with_capacity(t_q);
        for (e, s, k) in per_q {
            exp_b.push(e);
            sel_b.push(s);
            keep_b.push(k);
        }
        all_expanded.push(exp_b);
        all_selected_buckets.push(sel_b);
        all_keep.push(keep_b);
    }

    Ok(ReadSelection {
        expanded_keys: all_expanded,
        selected_buckets: all_selected_buckets,
        keep_indices: all_keep,
        query_offset,
        window_size,
        sink_size,
        top_k,
    })
}

/// Hierarchical (Tree) LSA bucket read across batch `[B, Tq, d_idx]`.
#[allow(clippy::too_many_arguments)]
pub fn read_hierarchy(
    index_queries: &Tensor,
    buckets: &Buckets,
    hierarchies: &[Hierarchy],
    window_size: usize,
    sink_size: usize,
    top_p: usize,
    top_k: usize,
    beam_width: usize,
    start_nodes: usize,
    early_budget: Option<EarlyBudget>,
    query_offset: usize,
) -> Result<ReadSelection> {
    let (b_sz, t_q, d_idx) = index_queries.dims3()?;
    if hierarchies.len() != b_sz {
        hanzo_ml::bail!(
            "hierarchies len ({}) must match batch size ({b_sz})",
            hierarchies.len()
        );
    }

    let mut all_expanded = Vec::with_capacity(b_sz);
    let mut all_selected_buckets = Vec::with_capacity(b_sz);
    let mut all_keep = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let q_flat = index_queries
            .narrow(0, b, 1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let hier = &hierarchies[b];
        let mut level_vecs = Vec::with_capacity(hier.centroids.len());
        for c in &hier.centroids {
            if c.elem_count() == 0 {
                level_vecs.push(Vec::new());
            } else {
                level_vecs.push(c.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?);
            }
        }
        let l0_flat = &level_vecs[0];
        let members_b = &buckets.bucket_members[b];

        let per_q: Vec<(Vec<i32>, Vec<usize>, Vec<u32>)> = (0..t_q)
            .into_par_iter()
            .map(|i| {
                let pos = query_offset + i;
                let (eff_p, eff_k) = budget_at(pos, top_p, top_k, early_budget);
                let growth = growth_at(hier, pos, window_size);
                let q_row = &q_flat[i * d_idx..(i + 1) * d_idx];
                let selected = beam_search_single_query(
                    q_row,
                    l0_flat,
                    &level_vecs,
                    hier,
                    &growth,
                    eff_p,
                    beam_width,
                    start_nodes,
                );
                let expanded = expand_buckets_single_query(
                    &selected,
                    members_b,
                    pos,
                    window_size,
                    sink_size,
                    eff_k,
                );
                let keep = build_query_keep_set(&expanded, pos, window_size, sink_size);
                (expanded, selected, keep)
            })
            .collect();

        let mut exp_b = Vec::with_capacity(t_q);
        let mut sel_b = Vec::with_capacity(t_q);
        let mut keep_b = Vec::with_capacity(t_q);
        for (e, s, k) in per_q {
            exp_b.push(e);
            sel_b.push(s);
            keep_b.push(k);
        }
        all_expanded.push(exp_b);
        all_selected_buckets.push(sel_b);
        all_keep.push(keep_b);
    }

    Ok(ReadSelection {
        expanded_keys: all_expanded,
        selected_buckets: all_selected_buckets,
        keep_indices: all_keep,
        query_offset,
        window_size,
        sink_size,
        top_k,
    })
}

// ============================================================================
// 6. Sparse Attention Read (Subquadratic Gather & Reference Mask) + N-LSA Summaries
// ============================================================================

/// Gather `idx_tensor` (`[M]`) along the sequence axis (dim 2) of `kv` (`[B, Hkv, Tk, D]`)
/// for batch `b`, returning `[Hkv, M, D]`.
/// Avoids copying the full KV cache even when `kv` is a `.narrow(2, 0, cur_len)` view.
fn gather_kv_positions(kv: &Tensor, b: usize, idx_tensor: &Tensor) -> Result<Tensor> {
    let kv_b = kv.narrow(0, b, 1)?.squeeze(0)?; // [Hkv, Tk, D]
    if kv_b.is_contiguous() {
        return kv_b.index_select(idx_tensor, 1);
    }
    let h_kv = kv_b.dim(0)?;
    let mut heads = Vec::with_capacity(h_kv);
    for h in 0..h_kv {
        let kv_bh = kv_b.narrow(0, h, 1)?; // [1, Tk, D]
        let kv_bh = if kv_bh.is_contiguous() {
            kv_bh
        } else {
            kv_bh.contiguous()?
        };
        heads.push(kv_bh.index_select(idx_tensor, 1)?);
    }
    Tensor::cat(&heads, 0)
}

/// Subquadratic sparse scaled dot-product attention over `ReadSelection`.
///
/// Instead of materializing a dense `[B, H, Tq, Tk]` score matrix, gathers only the selected
/// `<= sink_size + top_k + window_size` keys per query (or deduplicated union per query micro-batch),
/// reducing attention complexity from `O(Tq * Tk)` to `O(Tq * (sink_size + top_k + window_size))`.
///
/// - `queries`: `[B, Hq, Tq, Dh]`
/// - `keys`: `[B, Hkv, Tk, Dh]`
/// - `values`: `[B, Hkv, Tk, Dv]`
pub fn sparse_gather_attention(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    selection: &ReadSelection,
    softmax_scale: Option<f32>,
) -> Result<Tensor> {
    let (b_sz, h_q, t_q, d_h) = queries.dims4()?;
    let (_, h_kv, _t_k, _) = keys.dims4()?;
    let d_v = values.dim(3)?;
    if h_kv == 0 || h_q % h_kv != 0 {
        hanzo_ml::bail!("H_q ({h_q}) must be divisible by H_kv ({h_kv})");
    }
    let group_size = h_q / h_kv;
    let scale = softmax_scale.unwrap_or_else(|| 1.0 / (d_h as f32).sqrt()) as f64;
    let dtype = queries.dtype();
    let device = queries.device();

    let max_keep_len = selection
        .keep_indices
        .iter()
        .flat_map(|b| b.iter().map(|v| v.len()))
        .max()
        .unwrap_or(0);

    if max_keep_len == 0 {
        return Tensor::zeros((b_sz, h_q, t_q, d_v), dtype, device);
    }

    // Fast path for single-query decode (`t_q == 1`, `b_sz == 1`)
    if t_q == 1 && b_sz == 1 {
        let indices = &selection.keep_indices[0][0];
        let m = indices.len().max(1);
        let idx_tensor = Tensor::from_slice(indices, (m,), device)?;
        let k_sel = gather_kv_positions(keys, 0, &idx_tensor)?; // [Hkv, m, Dh]
        let v_sel = gather_kv_positions(values, 0, &idx_tensor)?; // [Hkv, m, Dv]
        let q_g = queries.reshape((h_kv, group_size, d_h))?;
        let scores = (q_g.matmul(&k_sel.t()?)? * scale)?; // [Hkv, group_size, m]
        let probs = crate::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        return probs.matmul(&v_sel)?.reshape((1, h_q, 1, d_v));
    }

    // Fast CPU-native parallel kernel for multi-query prefill/encoder (`t_q > 1` on `Device::Cpu`):
    // Evaluates each `(batch, q_head, query_i)` directly against its exact `keep_indices[b][qi]`
    // in thread-local scratch without unioning keys across 64 queries or allocating per-chunk Tensors.
    if device.is_cpu() {
        let q_f32 = queries.to_dtype(DType::F32)?.contiguous()?;
        let k_f32 = keys.to_dtype(DType::F32)?.contiguous()?;
        let v_f32 = values.to_dtype(DType::F32)?.contiguous()?;

        let q_vec = q_f32.flatten_all()?.to_vec1::<f32>()?;
        let k_vec = k_f32.flatten_all()?.to_vec1::<f32>()?;
        let v_vec = v_f32.flatten_all()?.to_vec1::<f32>()?;
        let t_k = keys.dim(2)?;
        let scale_f32 = scale as f32;

        const Q_TILE: usize = 64;
        let num_q_tiles = t_q.div_ceil(Q_TILE);
        let total_tasks = b_sz * h_q * num_q_tiles;

        let tile_chunks: Vec<Vec<f32>> = (0..total_tasks)
            .into_par_iter()
            .map(|task| {
                let bh = task / num_q_tiles;
                let q_tile_idx = task % num_q_tiles;
                let b = bh / h_q;
                let h = bh % h_q;
                let h_kv_idx = h / group_size;

                let q_start = q_tile_idx * Q_TILE;
                let q_end = (q_start + Q_TILE).min(t_q);
                let q_len = q_end - q_start;

                let q_head_base = (b * h_q + h) * t_q * d_h;
                let k_head_base = (b * h_kv + h_kv_idx) * t_k * d_h;
                let v_head_base = (b * h_kv + h_kv_idx) * t_k * d_v;
                let q_head = &q_vec[q_head_base..q_head_base + t_q * d_h];
                let k_head = &k_vec[k_head_base..k_head_base + t_k * d_h];
                let v_head = &v_vec[v_head_base..v_head_base + t_k * d_v];
                let keep_b = &selection.keep_indices[b];

                let mut out_tile = vec![0.0f32; q_len * d_v];
                let mut scores = Vec::with_capacity(max_keep_len);

                for (local_i, qi) in (q_start..q_end).enumerate() {
                    let keep_q = &keep_b[qi];
                    if keep_q.is_empty() {
                        continue;
                    }
                    let q_row = &q_head[qi * d_h..(qi + 1) * d_h];
                    scores.clear();
                    let mut max_s = f32::NEG_INFINITY;
                    for &k_idx_u32 in keep_q {
                        let k_idx = k_idx_u32 as usize;
                        let k_row = &k_head[k_idx * d_h..(k_idx + 1) * d_h];
                        let mut dot = 0.0f32;
                        for d in 0..d_h {
                            dot += q_row[d] * k_row[d];
                        }
                        let s = dot * scale_f32;
                        if s > max_s {
                            max_s = s;
                        }
                        scores.push(s);
                    }

                    let mut sum_exp = 0.0f32;
                    for s in &mut scores {
                        let e = (*s - max_s).exp();
                        *s = e;
                        sum_exp += e;
                    }
                    let inv_sum = 1.0f32 / sum_exp.max(1e-30);

                    let out_row = &mut out_tile[local_i * d_v..(local_i + 1) * d_v];
                    for (prob_unnorm, &k_idx_u32) in scores.iter().zip(keep_q.iter()) {
                        let w = *prob_unnorm * inv_sum;
                        let k_idx = k_idx_u32 as usize;
                        let v_row = &v_head[k_idx * d_v..(k_idx + 1) * d_v];
                        for d in 0..d_v {
                            out_row[d] += w * v_row[d];
                        }
                    }
                }
                out_tile
            })
            .collect();

        let mut all_out = Vec::with_capacity(b_sz * h_q * t_q * d_v);
        for chunk in tile_chunks {
            all_out.extend_from_slice(&chunk);
        }
        return Tensor::from_vec(all_out, (b_sz, h_q, t_q, d_v), device)?.to_dtype(dtype);
    }

    // Device fallback for multi-query prefill (`t_q > 1` on non-CPU devices):
    const Q_CHUNK: usize = 64;
    let mut batch_outputs = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let q_b = queries.narrow(0, b, 1)?.squeeze(0)?; // [Hq, Tq, Dh]

        let keep_b = &selection.keep_indices[b];
        let mut q_chunks_out = Vec::with_capacity(t_q.div_ceil(Q_CHUNK));

        for q_start in (0..t_q).step_by(Q_CHUNK) {
            let q_len = Q_CHUNK.min(t_q - q_start);

            let mut union_keys: Vec<u32> = Vec::new();
            for qi in 0..q_len {
                union_keys.extend_from_slice(&keep_b[q_start + qi]);
            }
            union_keys.sort_unstable();
            union_keys.dedup();
            if union_keys.is_empty() {
                union_keys.push(0);
            }
            let u_len = union_keys.len();

            let mut mask_vals = vec![f32::NEG_INFINITY; q_len * u_len];
            for qi in 0..q_len {
                let row_off = qi * u_len;
                for &k_idx in &keep_b[q_start + qi] {
                    if let Ok(pos) = union_keys.binary_search(&k_idx) {
                        mask_vals[row_off + pos] = 0.0;
                    }
                }
            }

            let idx_tensor = Tensor::from_vec(union_keys, (u_len,), device)?;
            let k_sel = gather_kv_positions(keys, b, &idx_tensor)?; // [Hkv, u_len, Dh]
            let v_sel = gather_kv_positions(values, b, &idx_tensor)?; // [Hkv, u_len, Dv]

            let q_chunk = q_b.narrow(1, q_start, q_len)?.contiguous()?.reshape((
                h_kv,
                group_size * q_len,
                d_h,
            ))?;

            let scores = (q_chunk.matmul(&k_sel.t()?)? * scale)?
                .reshape((h_kv, group_size, q_len, u_len))?;
            let mask_tensor = Tensor::from_vec(mask_vals, (1, 1, q_len, u_len), device)?
                .broadcast_as((h_kv, group_size, q_len, u_len))?;
            let scores_f32 = scores.to_dtype(DType::F32)?.add(&mask_tensor)?;
            let probs = crate::ops::softmax_last_dim(&scores_f32)?
                .to_dtype(dtype)?
                .reshape((h_kv, group_size * q_len, u_len))?;

            let out_chunk = probs.matmul(&v_sel)?.reshape((h_q, q_len, d_v))?;
            q_chunks_out.push(out_chunk);
        }

        let b_out = Tensor::cat(&q_chunks_out, 1)?.unsqueeze(0)?; // [1, Hq, Tq, Dv]
        batch_outputs.push(b_out);
    }

    Tensor::cat(&batch_outputs, 0)
}

/// Reference masked `topk_attention` (identical arithmetic to `lsa.core.attention.topk_attention`).
pub fn topk_attention(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    selection: &ReadSelection,
    softmax_scale: Option<f32>,
) -> Result<Tensor> {
    sparse_gather_attention(queries, keys, values, selection, softmax_scale)
}

/// N-LSA Two-Path Differentiable Summary Attention (`lsa.training.native.lsa_summaries`).
///
/// Combines exact sparse attention over `sinks U selected_buckets U window` with a second-order
/// isotropic Gaussian log-mass summary over unselected closed buckets:
/// `log_mass(U) = log(|U|) + scale * (q . mu_U) + 0.5 * scale^2 * ||q||^2 * sigma_U^2`,
/// where `sigma_U^2 = max(0, (mean_t ||k_t||^2 - ||mu_U||^2) / d_h)`.
/// Because unselected bucket centroids and value means are differentiable functions of all keys and
/// values in those buckets, gradients flow to every token even when `top_k << T`.
pub fn lsa_two_path_attention(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    buckets: &Buckets,
    selection: &ReadSelection,
    softmax_scale: Option<f32>,
) -> Result<Tensor> {
    let (b_sz, h_q, t_q, d_h) = queries.dims4()?;
    let (_, h_kv, t_k, _) = keys.dims4()?;
    let d_v = values.dim(3)?;
    let group_size = h_q / h_kv;
    let scale = softmax_scale.unwrap_or_else(|| 1.0 / (d_h as f32).sqrt()) as f64;
    let dtype = queries.dtype();
    let device = queries.device();

    let b_total = buckets.centroids.dim(1)?;
    if b_total == 0 {
        return sparse_gather_attention(queries, keys, values, selection, softmax_scale);
    }

    let mut batch_outputs = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let q_b = queries.narrow(0, b, 1)?.squeeze(0)?.to_dtype(DType::F32)?; // [Hq, Tq, Dh]
        let k_b = keys.narrow(0, b, 1)?.squeeze(0)?.to_dtype(DType::F32)?; // [Hkv, Tk, Dh]
        let v_b = values.narrow(0, b, 1)?.squeeze(0)?.to_dtype(DType::F32)?; // [Hkv, Tk, Dv]

        // Repeat KV heads if GQA so all ops are [Hq, ...]
        let (k_rep, v_rep) = if group_size > 1 {
            let kr = k_b
                .unsqueeze(1)?
                .broadcast_as((h_kv, group_size, t_k, d_h))?
                .reshape((h_q, t_k, d_h))?;
            let vr = v_b
                .unsqueeze(1)?
                .broadcast_as((h_kv, group_size, t_k, d_v))?
                .reshape((h_q, t_k, d_v))?;
            (kr, vr)
        } else {
            (k_b, v_b)
        };

        let k_time_first = k_rep.transpose(0, 1)?.contiguous()?; // [Tk, Hq, Dh]
        let v_time_first = v_rep.transpose(0, 1)?.contiguous()?; // [Tk, Hq, Dv]

        // Differentiably gather bucket means mu_b [B_total, Hq, Dh], v_bar_b [B_total, Hq, Dv], and sigma_b^2 [B_total, Hq]
        let members_b = &buckets.bucket_members[b];
        let sizes_b = &buckets.bucket_sizes[b];
        let mut mu_list = Vec::with_capacity(b_total);
        let mut vbar_list = Vec::with_capacity(b_total);
        let mut var_list = Vec::with_capacity(b_total);
        let mut log_count_vec = Vec::with_capacity(b_total);

        for bucket_id in 0..b_total {
            let valid: Vec<u32> = members_b[bucket_id]
                .iter()
                .copied()
                .filter(|&idx| idx >= 0)
                .map(|idx| idx as u32)
                .collect();
            let count = valid.len().max(1);
            log_count_vec.push((sizes_b[bucket_id].max(1) as f32).ln());
            let idx_t = Tensor::from_vec(valid, (count,), device)?;
            let k_mem = k_time_first.index_select(&idx_t, 0)?; // [count, Hq, Dh]
            let v_mem = v_time_first.index_select(&idx_t, 0)?; // [count, Hq, Dv]

            let inv_c = 1.0 / (count as f64);
            let mu = (k_mem.sum(0)? * inv_c)?; // [Hq, Dh]
            let vbar = (v_mem.sum(0)? * inv_c)?; // [Hq, Dv]
            let mean_k_norm_sq = (k_mem.sqr()?.sum(D::Minus1)?.sum(0)? * inv_c)?; // [Hq]
            let mu_norm_sq = mu.sqr()?.sum(D::Minus1)?; // [Hq]
            let sigma_sq =
                (mean_k_norm_sq.sub(&mu_norm_sq)? * (1.0 / (d_h as f64)))?.clamp(0.0, f64::MAX)?; // [Hq]

            mu_list.push(mu.unsqueeze(1)?); // [Hq, 1, Dh]
            vbar_list.push(vbar.unsqueeze(1)?); // [Hq, 1, Dv]
            var_list.push(sigma_sq.unsqueeze(1)?); // [Hq, 1]
        }

        let mu_all = Tensor::cat(&mu_list, 1)?; // [Hq, B_total, Dh]
        let vbar_all = Tensor::cat(&vbar_list, 1)?; // [Hq, B_total, Dv]
        let sigma_sq_all = Tensor::cat(&var_list, 1)?; // [Hq, B_total]

        // Summary logits for all (query, bucket):
        // scale * (q . mu_b) + 0.5 * scale^2 * ||q||^2 * sigma_b^2 + log(|b|)
        let q_dot_mu = (q_b.matmul(&mu_all.transpose(1, 2)?)? * scale)?; // [Hq, Tq, B_total]
        let q_norm_sq = q_b.sqr()?.sum_keepdim(D::Minus1)?; // [Hq, Tq, 1]
        let second_order =
            q_norm_sq.broadcast_mul(&sigma_sq_all.unsqueeze(1)?)? * (0.5 * scale * scale);
        let log_counts = Tensor::from_vec(log_count_vec, (1, 1, b_total), device)?
            .broadcast_as((h_q, t_q, b_total))?;
        let summary_logits = q_dot_mu.add(&second_order?)?.add(&log_counts)?; // [Hq, Tq, B_total]

        // Build per-(query, bucket) mask: 0.0 for visible unselected buckets, -inf for selected or invisible buckets
        let mut summary_mask = vec![f32::NEG_INFINITY; t_q * b_total];
        for qi in 0..t_q {
            let pos = selection.query_offset + qi;
            let visible = ((pos / selection.window_size) * buckets.buckets_per_block).min(b_total);
            for bucket_id in 0..visible {
                summary_mask[qi * b_total + bucket_id] = 0.0;
            }
            for &sel_id in &selection.selected_buckets[b][qi] {
                if sel_id < b_total {
                    summary_mask[qi * b_total + sel_id] = f32::NEG_INFINITY;
                }
            }
        }
        let summary_mask_t = Tensor::from_vec(summary_mask, (1, t_q, b_total), device)?
            .broadcast_as((h_q, t_q, b_total))?;
        let masked_summary_logits = summary_logits.add(&summary_mask_t)?;

        // Exact logits over keep_indices per query + combine with summary logits in a single softmax!
        let keep_b = &selection.keep_indices[b];
        let max_keep = keep_b.iter().map(|v| v.len()).max().unwrap_or(1);
        let mut flat_indices = Vec::with_capacity(t_q * max_keep);
        let mut exact_mask = Vec::with_capacity(t_q * max_keep);
        for qi in 0..t_q {
            let indices = &keep_b[qi];
            for col in 0..max_keep {
                if col < indices.len() {
                    flat_indices.push(indices[col]);
                    exact_mask.push(0.0f32);
                } else {
                    flat_indices.push(0u32);
                    exact_mask.push(f32::NEG_INFINITY);
                }
            }
        }
        let idx_t = Tensor::from_vec(flat_indices, (t_q * max_keep,), device)?;
        let k_sel = k_time_first
            .index_select(&idx_t, 0)?
            .reshape((t_q, max_keep, h_q, d_h))?
            .permute((2, 0, 1, 3))?
            .contiguous()?; // [Hq, Tq, max_keep, Dh]
        let v_sel = v_time_first
            .index_select(&idx_t, 0)?
            .reshape((t_q, max_keep, h_q, d_v))?
            .permute((2, 0, 1, 3))?
            .contiguous()?; // [Hq, Tq, max_keep, Dv]

        let q_exp = q_b.unsqueeze(2)?; // [Hq, Tq, 1, Dh]
        let exact_logits = (q_exp.matmul(&k_sel.transpose(2, 3)?)?.squeeze(2)? * scale)?; // [Hq, Tq, max_keep]
        let exact_mask_t = Tensor::from_vec(exact_mask, (1, t_q, max_keep), device)?
            .broadcast_as((h_q, t_q, max_keep))?;
        let masked_exact_logits = exact_logits.add(&exact_mask_t)?;

        // Concatenate exact and summary logits along last dim: [Hq, Tq, max_keep + B_total]
        let combined_logits =
            Tensor::cat(&[&masked_exact_logits, &masked_summary_logits], D::Minus1)?;
        let combined_probs = crate::ops::softmax_last_dim(&combined_logits)?;
        let probs_exact = combined_probs.narrow(D::Minus1, 0, max_keep)?; // [Hq, Tq, max_keep]
        let probs_summary = combined_probs.narrow(D::Minus1, max_keep, b_total)?; // [Hq, Tq, B_total]

        let out_exact = probs_exact.unsqueeze(2)?.matmul(&v_sel)?.squeeze(2)?; // [Hq, Tq, Dv]
        let out_summary = probs_summary.matmul(&vbar_all)?; // [Hq, Tq, Dv]
        let out_b = out_exact.add(&out_summary)?.to_dtype(dtype)?.unsqueeze(0)?;
        batch_outputs.push(out_b);
    }

    Tensor::cat(&batch_outputs, 0)
}

// ============================================================================
// 7. Stateful Layer Index (`LsaConfig` & `LsaLayerIndex`) for Prefill + Decode
// ============================================================================

/// Configuration for Latent Sparse Attention (D-LSA & N-LSA, Flat & Tree).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LsaConfig {
    /// Sliding window & cluster block size `n_w = n_cluster` (default 512).
    pub window_size: usize,
    /// Leading attention sink tokens `n_sink` (default 4).
    pub sink_size: usize,
    /// Maximum long-range keys unpacked per query `k` (default 4096).
    pub top_k: usize,
    /// Number of buckets selected per query `p` (default 1024).
    pub top_p: usize,
    /// Pool size `n_pool` before Ward clustering (default 4).
    pub pool_size: usize,
    /// Mean keys per bucket `\bar{n}_b` (default 8 -> 64 buckets per 512-key block).
    pub bucket_size: usize,
    /// Hard maximum keys per bucket `n_b^{max}` (default 16).
    pub bucket_cap: usize,
    /// Unrotated RoPE layout (`Prefix`, `Suffix`, or `None`).
    pub rotary_layout: RotaryLayout,
    /// Number of rotated dimensions in each head (`0` if pre-RoPE or `RotaryLayout::None`).
    pub rotary_dim: usize,
    /// Optional 1-bit covariance shape rank (`None` or `Some((rank, deviations))`).
    pub bucket_shape: Option<(usize, f32)>,
    /// Tree depth (`0` for Flat design, `>= 1` for Tree/Hierarchical design).
    pub top_level: usize,
    /// Children per node bucket in Tree design (default 4).
    pub node_bucket_size: usize,
    /// Buckets per node cluster block in Tree design (default 4096).
    pub node_cluster_block_size: usize,
    /// Beam width in Tree design (default 256).
    pub beam_width: usize,
    /// Start node threshold in Tree design (default 1024).
    pub start_nodes: usize,
    /// Optional early budget for shorter prefix positions.
    pub early_budget: Option<EarlyBudget>,
}

impl Default for LsaConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            sink_size: 4,
            top_k: 4096,
            top_p: 1024,
            pool_size: 4,
            bucket_size: 8,
            bucket_cap: 16,
            rotary_layout: RotaryLayout::Prefix,
            rotary_dim: 0,
            bucket_shape: None,
            top_level: 0,
            node_bucket_size: 4,
            node_cluster_block_size: 4096,
            beam_width: 256,
            start_nodes: 1024,
            early_budget: None,
        }
    }
}

impl LsaConfig {
    /// Preset for Qwen3.8-27B / Hanzo Zen-27B (`rotary_dim = 64`, `RotaryLayout::Prefix`,
    /// `EarlyBudget { position: 262_144, top_k: 1024, top_p: 256 }`).
    pub fn zen_27b() -> Self {
        Self {
            rotary_layout: RotaryLayout::Prefix,
            rotary_dim: 64,
            early_budget: Some(EarlyBudget {
                position: 262_144,
                top_k: 1024,
                top_p: 256,
            }),
            ..Self::default()
        }
    }

    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    pub fn with_sink_size(mut self, sink_size: usize) -> Self {
        self.sink_size = sink_size;
        self
    }

    pub fn with_budget(mut self, top_k: usize, top_p: usize) -> Self {
        self.top_k = top_k;
        self.top_p = top_p;
        self
    }

    pub fn with_early_budget(mut self, early_budget: Option<EarlyBudget>) -> Self {
        self.early_budget = early_budget;
        self
    }

    pub fn with_rotary(mut self, layout: RotaryLayout, rotary_dim: usize) -> Self {
        self.rotary_layout = layout;
        self.rotary_dim = rotary_dim;
        self
    }

    pub fn with_tree(mut self, top_level: usize, beam_width: usize, start_nodes: usize) -> Self {
        self.top_level = top_level;
        self.beam_width = beam_width;
        self.start_nodes = start_nodes;
        self
    }

    pub fn with_bucket_shape(mut self, shape: Option<(usize, f32)>) -> Self {
        self.bucket_shape = shape;
        self
    }

    pub fn with_ward_clustering(
        mut self,
        pool_size: usize,
        bucket_size: usize,
        bucket_cap: usize,
    ) -> Self {
        self.pool_size = pool_size;
        self.bucket_size = bucket_size;
        self.bucket_cap = bucket_cap;
        self
    }
}

/// Incremental per-layer LSA bucket cache for streaming prefill and decode.
///
/// Buffers index keys as tokens arrive, Ward-clusters each `window_size` block as soon as it
/// closes, and reuses the cached bucket centroids across all future decode steps.
#[derive(Debug, Clone)]
pub struct LsaLayerIndex {
    pub config: LsaConfig,
    pub projection: Option<LearnedProjection>,
    /// Accumulated index keys `[B, T_cached, d_idx]`.
    index_keys: Option<Tensor>,
    /// Number of cluster blocks (`window_size` keys each) already clustered into `buckets`.
    clustered_blocks: usize,
    /// Cached closed-block buckets.
    buckets: Option<Buckets>,
}

impl LsaLayerIndex {
    pub fn new(config: LsaConfig, projection: Option<LearnedProjection>) -> Self {
        Self {
            config,
            projection,
            index_keys: None,
            clustered_blocks: 0,
            buckets: None,
        }
    }

    pub fn reset(&mut self) {
        self.index_keys = None;
        self.clustered_blocks = 0;
        self.buckets = None;
    }

    pub fn clustered_blocks(&self) -> usize {
        self.clustered_blocks
    }

    pub fn buckets(&self) -> Option<&Buckets> {
        self.buckets.as_ref()
    }

    /// Compute `(index_queries, index_keys)` for the current step's `q` (`[B, Hq, Tq, Dh]`) and `new_k` (`[B, Hkv, Tk_new, Dh]`).
    pub fn compute_index_vectors(&self, q: &Tensor, new_k: &Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.projection, self.config.rotary_dim > 0) {
            (Some(proj), true) => project_partial(
                q,
                new_k,
                self.config.rotary_dim,
                self.config.rotary_layout,
                proj,
            ),
            (Some(proj), false) => project(q, new_k, proj),
            (None, true) => {
                regroup_partial(q, new_k, self.config.rotary_dim, self.config.rotary_layout)
            }
            (None, false) => regroup(q, new_k),
        }
    }

    /// Append newly arrived `new_index_keys` (`[B, Tk_new, d_idx]`) and incrementally cluster any newly closed blocks.
    pub fn append_index_keys(&mut self, new_index_keys: &Tensor) -> Result<()> {
        let all_keys = match self.index_keys.take() {
            Some(prev) => Tensor::cat(&[&prev, new_index_keys], 1)?,
            None => new_index_keys.clone(),
        };
        let (b_sz, total_len, d_idx) = all_keys.dims3()?;
        let total_closed = total_len / self.config.window_size;
        let new_blocks = total_closed.saturating_sub(self.clustered_blocks);

        if new_blocks > 0 {
            let start_pos = self.clustered_blocks * self.config.window_size;
            let len_to_cluster = new_blocks * self.config.window_size;
            let slice_keys = all_keys
                .narrow(1, start_pos, len_to_cluster)?
                .contiguous()?
                .reshape((b_sz * new_blocks, self.config.window_size, d_idx))?;

            let mut offsets = Vec::with_capacity(b_sz * new_blocks);
            for _ in 0..b_sz {
                for blk in 0..new_blocks {
                    offsets.push(((self.clustered_blocks + blk) * self.config.window_size) as i32);
                }
            }

            let merged = ward_merge(
                &slice_keys,
                self.config.pool_size,
                self.config.bucket_size,
                self.config.bucket_cap,
                Some(&offsets),
            )?;
            let b_per_blk = self.config.window_size / self.config.bucket_size;
            let new_b_total = new_blocks * b_per_blk;
            let new_centroids = merged.centroids.reshape((b_sz, new_b_total, d_idx))?;

            let mut new_members = vec![Vec::with_capacity(new_b_total); b_sz];
            let mut new_sizes = vec![Vec::with_capacity(new_b_total); b_sz];
            for b in 0..b_sz {
                for blk in 0..new_blocks {
                    let flat_idx = b * new_blocks + blk;
                    new_members[b].extend_from_slice(&merged.bucket_members[flat_idx]);
                    new_sizes[b].extend_from_slice(&merged.bucket_sizes[flat_idx]);
                }
            }

            match self.buckets.take() {
                Some(mut existing) => {
                    existing.centroids = Tensor::cat(&[&existing.centroids, &new_centroids], 1)?;
                    for b in 0..b_sz {
                        existing.bucket_members[b].extend_from_slice(&new_members[b]);
                        existing.bucket_sizes[b].extend_from_slice(&new_sizes[b]);
                    }
                    existing.num_closed_blocks = total_closed;
                    if let Some((rank, _)) = self.config.bucket_shape {
                        shape_buckets(&mut existing, &all_keys, rank)?;
                    }
                    self.buckets = Some(existing);
                }
                None => {
                    let mut created = Buckets {
                        centroids: new_centroids,
                        bucket_members: new_members,
                        bucket_sizes: new_sizes,
                        buckets_per_block: b_per_blk,
                        num_closed_blocks: total_closed,
                        shapes: None,
                    };
                    if let Some((rank, _)) = self.config.bucket_shape {
                        shape_buckets(&mut created, &all_keys, rank)?;
                    }
                    self.buckets = Some(created);
                }
            }
            self.clustered_blocks = total_closed;
        } else if self.buckets.is_none() {
            let b_per_blk = self.config.window_size / self.config.bucket_size;
            self.buckets = Some(Buckets {
                centroids: Tensor::zeros((b_sz, 0, d_idx), all_keys.dtype(), all_keys.device())?,
                bucket_members: vec![Vec::new(); b_sz],
                bucket_sizes: vec![Vec::new(); b_sz],
                buckets_per_block: b_per_blk,
                num_closed_blocks: 0,
                shapes: None,
            });
        }

        self.index_keys = Some(all_keys);
        Ok(())
    }

    /// Run a full LSA forward step (prefill or decode) given `q` (`[B, Hq, Tq, Dh]`),
    /// full cached `k` (`[B, Hkv, Tk, Dh]`) and `v` (`[B, Hkv, Tk, Dv]`), and the newly appended `new_k`.
    pub fn forward(
        &mut self,
        q: &Tensor,
        k_full: &Tensor,
        v_full: &Tensor,
        new_k: &Tensor,
        softmax_scale: Option<f32>,
    ) -> Result<Tensor> {
        let t_q = q.dim(2)?;
        let t_k = k_full.dim(2)?;
        let query_offset = t_k.saturating_sub(t_q);

        let (q_idx, k_idx_new) = self.compute_index_vectors(q, new_k)?;
        self.append_index_keys(&k_idx_new)?;

        let buckets = self.buckets.as_ref().unwrap();
        let deviations = self.config.bucket_shape.map(|(_, z)| z).unwrap_or(0.0);

        let selection = if self.config.top_level == 0 {
            read(
                &q_idx,
                buckets,
                self.config.window_size,
                self.config.sink_size,
                self.config.top_p,
                self.config.top_k,
                deviations,
                self.config.early_budget,
                query_offset,
            )?
        } else {
            let b_sz = q.dim(0)?;
            let mut hierarchies = Vec::with_capacity(b_sz);
            for b in 0..b_sz {
                let c0 = buckets.centroids.narrow(0, b, 1)?.squeeze(0)?;
                let s0 = buckets.shapes.as_ref().map(|sv| sv[b].clone());
                let h = build_hierarchy(
                    &c0,
                    &buckets.bucket_sizes[b],
                    buckets.buckets_per_block,
                    s0,
                    self.config.top_level,
                    self.config.node_bucket_size,
                    self.config.node_cluster_block_size,
                )?;
                hierarchies.push(h);
            }
            read_hierarchy(
                &q_idx,
                buckets,
                &hierarchies,
                self.config.window_size,
                self.config.sink_size,
                self.config.top_p,
                self.config.top_k,
                self.config.beam_width,
                self.config.start_nodes,
                self.config.early_budget,
                query_offset,
            )?
        };

        sparse_gather_attention(q, k_full, v_full, &selection, softmax_scale)
    }
}

// ============================================================================
// 8. Bidirectional D-LSA for Non-Causal Encoders (UltraBERT / ModernBERT)
// ============================================================================

/// Configuration for Bidirectional Distilled Latent Sparse Attention (D-LSA)
/// on global encoder layers (e.g., UltraBERT-380M / UltraBERT-150M).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BidirectionalLsaConfig {
    /// Bilateral local sliding-window radius `W` around each query `[pos - W, pos + W]` (128 for UltraBERT).
    pub local_radius: usize,
    /// Global prefix/CLS sink tokens `[0..sink_size]` always attended to (default 1 for `[CLS]`, or 4).
    pub sink_size: usize,
    /// Cluster block size `n_cluster` for Ward bucket construction across the sequence (default 256).
    pub cluster_block_size: usize,
    /// Consecutive pool size `n_pool` before Ward clustering (default 4).
    pub pool_size: usize,
    /// Mean keys per bucket `\bar{n}_b` (default 8 -> 32 buckets per 256-key block).
    pub bucket_size: usize,
    /// Hard maximum keys per bucket `n_b^{max}` (default 16).
    pub bucket_cap: usize,
    /// Retrieved long-range key budget `K` outside the bilateral window (`256`, `512`, `1024`, or `2048`).
    pub top_k: usize,
    /// Number of candidate buckets ranked and unpacked per query (`p`, default `top_k / bucket_size * 2`).
    pub top_p: usize,
    /// Index space dimension `d_idx` (default 64).
    pub index_dim: usize,
    /// Optional 1-bit covariance shape `(rank, deviations)` for directional spread scoring.
    pub bucket_shape: Option<(usize, f32)>,
}

impl Default for BidirectionalLsaConfig {
    fn default() -> Self {
        Self::ultrabert_with_budget(512)
    }
}

impl BidirectionalLsaConfig {
    /// Preset for UltraBERT global layers (`local_radius = 128`, `sink_size = 4`,
    /// `cluster_block_size = 256`, `pool_size = 4`, `bucket_size = 8`, `bucket_cap = 16`, `index_dim = 64`)
    /// at a given retrieved-key budget `top_k ∈ {256, 512, 1024, 2048}`.
    pub fn ultrabert_with_budget(top_k: usize) -> Self {
        let bucket_size = 8;
        // Over-select buckets by 1.5x so filtering out local-window keys still fills `top_k`
        let top_p = ((top_k / bucket_size) * 3 / 2).max(top_k.div_ceil(bucket_size));
        Self {
            local_radius: 128,
            sink_size: 4,
            cluster_block_size: 256,
            pool_size: 4,
            bucket_size,
            bucket_cap: 16,
            top_k,
            top_p,
            index_dim: 64,
            bucket_shape: None,
        }
    }
}

/// Build Ward-clustered buckets across the **entire bidirectional sequence** `[0, T)`.
///
/// Unlike causal `build_buckets` (which only clusters closed prefix blocks `< pos / window_size`),
/// a bidirectional encoder observes all `T` tokens simultaneously, so every block across `[0, T)`
/// (including any partial tail block rounded down to a multiple of `bucket_size`) is clustered
/// upfront and visible to queries from both directions.
pub fn build_buckets_bidirectional(
    index_keys: &Tensor,
    cluster_block_size: usize,
    pool_size: usize,
    bucket_size: usize,
    bucket_cap: usize,
) -> Result<Buckets> {
    let (b_sz, seq_len, d_idx) = index_keys.dims3()?;
    if cluster_block_size == 0
        || !cluster_block_size.is_multiple_of(pool_size)
        || !cluster_block_size.is_multiple_of(bucket_size)
    {
        hanzo_ml::bail!(
            "cluster_block_size ({cluster_block_size}) must be a positive multiple of pool_size ({pool_size}) and bucket_size ({bucket_size})"
        );
    }

    let num_full_blocks = seq_len / cluster_block_size;
    let buckets_per_block = cluster_block_size / bucket_size;
    let rem = seq_len % cluster_block_size;
    let tail_buckets = rem / bucket_size;
    let tail_len = tail_buckets * bucket_size;

    let mut buckets = if num_full_blocks > 0 {
        build_buckets(
            index_keys,
            cluster_block_size,
            pool_size,
            bucket_size,
            bucket_cap,
        )?
    } else {
        Buckets {
            centroids: Tensor::zeros((b_sz, 0, d_idx), index_keys.dtype(), index_keys.device())?,
            bucket_members: vec![Vec::new(); b_sz],
            bucket_sizes: vec![Vec::new(); b_sz],
            buckets_per_block,
            num_closed_blocks: 0,
            shapes: None,
        }
    };

    // If there is a partial tail block with at least `bucket_size` keys (and divisible by `pool_size`),
    // cluster the tail block as well so right-end tokens are also indexed in buckets!
    if tail_len >= bucket_size && tail_len.is_multiple_of(pool_size) {
        let tail_start = num_full_blocks * cluster_block_size;
        let tail_keys = index_keys.narrow(1, tail_start, tail_len)?.contiguous()?;
        let tail_offsets: Vec<i32> = vec![tail_start as i32; b_sz];
        let tail_merged = ward_merge(
            &tail_keys,
            pool_size,
            bucket_size,
            bucket_cap,
            Some(&tail_offsets),
        )?;
        buckets.centroids = if buckets.centroids.dim(1)? == 0 {
            tail_merged.centroids
        } else {
            Tensor::cat(&[&buckets.centroids, &tail_merged.centroids], 1)?
        };
        for b in 0..b_sz {
            buckets.bucket_members[b].extend_from_slice(&tail_merged.bucket_members[b]);
            buckets.bucket_sizes[b].extend_from_slice(&tail_merged.bucket_sizes[b]);
        }
    }

    Ok(buckets)
}

/// Expand selected bucket IDs into up to `top_k` bidirectional long-range key indices,
/// retrieving keys from **both** `< query_pos - local_radius` and `> query_pos + local_radius`
/// while excluding `-1` padding, global sink positions `< sink_size`, and the bilateral local window
/// `[query_pos - local_radius ..= query_pos + local_radius]`.
pub fn expand_buckets_bidirectional_single_query(
    selected_buckets: &[usize],
    bucket_members: &[Vec<i32>],
    query_pos: usize,
    seq_len: usize,
    local_radius: usize,
    sink_size: usize,
    top_k: usize,
) -> Vec<i32> {
    if top_k == 0 || selected_buckets.is_empty() {
        return Vec::new();
    }
    let local_start = query_pos.saturating_sub(local_radius) as i32;
    let local_end = (query_pos + local_radius).min(seq_len.saturating_sub(1)) as i32;
    let sink_limit = sink_size as i32;
    let seq_limit = seq_len as i32;

    let mut expanded = Vec::with_capacity(top_k);
    for &b in selected_buckets {
        if b >= bucket_members.len() {
            continue;
        }
        for &key_pos in &bucket_members[b] {
            if key_pos < 0 || key_pos >= seq_limit {
                continue;
            }
            // Exclude sink tokens and bilateral local window [local_start ..= local_end]
            if key_pos >= sink_limit && (key_pos < local_start || key_pos > local_end) {
                expanded.push(key_pos);
                if expanded.len() >= top_k {
                    return expanded;
                }
            }
        }
    }
    expanded
}

/// Build the full sorted deduplicated bidirectional keep-set:
/// `sinks [0..sink_size] U expanded_bidirectional_keys U local_window [pos - W ..= pos + W]`.
pub fn build_bidirectional_keep_set(
    expanded: &[i32],
    query_pos: usize,
    seq_len: usize,
    local_radius: usize,
    sink_size: usize,
) -> Vec<u32> {
    if seq_len == 0 {
        return Vec::new();
    }
    let local_start = query_pos.saturating_sub(local_radius);
    let local_end = (query_pos + local_radius).min(seq_len - 1);
    let sink_end = sink_size.min(seq_len);

    let mut keep = Vec::with_capacity(sink_end + expanded.len() + (local_end - local_start + 1));
    for s in 0..sink_end {
        if s < local_start || s > local_end {
            keep.push(s as u32);
        }
    }
    for &k in expanded {
        if k >= 0 && (k as usize) < seq_len {
            let ku = k as usize;
            if ku >= sink_end && (ku < local_start || ku > local_end) {
                keep.push(ku as u32);
            }
        }
    }
    for w in local_start..=local_end {
        keep.push(w as u32);
    }
    keep.sort_unstable();
    keep.dedup();
    keep
}

/// Bidirectional D-LSA bucket read across batch `[B, T, d_idx]`.
///
/// Every query `pos ∈ 0..T` scores **all** `B_total` buckets across both directions of the sequence,
/// selects the top `top_p` buckets outside its bilateral local window `[pos - local_radius, pos + local_radius]`,
/// unpacks up to `top_k` long-range keys from either side, and merges them with `[0..sink_size]` and
/// `[pos - local_radius ..= pos + local_radius]`.
pub fn read_bidirectional(
    index_queries: &Tensor,
    buckets: &Buckets,
    seq_len: usize,
    local_radius: usize,
    sink_size: usize,
    top_p: usize,
    top_k: usize,
    deviations: f32,
) -> Result<ReadSelection> {
    let (b_sz, t_q, _d_idx) = index_queries.dims3()?;
    let b_total = buckets.centroids.dim(1)?;

    let mut all_expanded = Vec::with_capacity(b_sz);
    let mut all_selected_buckets = Vec::with_capacity(b_sz);
    let mut all_keep = Vec::with_capacity(b_sz);

    for b in 0..b_sz {
        let q_b = index_queries
            .narrow(0, b, 1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let c_b = buckets
            .centroids
            .narrow(0, b, 1)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;

        let scores_mat = if b_total == 0 {
            Vec::new()
        } else {
            let scores_t = if let Some(ref shapes_vec) = buckets.shapes {
                shaped_scores(&q_b, &c_b, &shapes_vec[b], deviations)?
            } else {
                q_b.matmul(&c_b.t()?)?
            };
            scores_t.flatten_all()?.to_vec1::<f32>()?
        };

        let members_b = &buckets.bucket_members[b];

        // Precompute min/max token position per bucket so buckets completely inside
        // a query's bilateral local window [pos - W, pos + W] can be skipped in O(1).
        let bucket_spans: Vec<(i32, i32)> = members_b
            .iter()
            .map(|mem| {
                let mut lo = i32::MAX;
                let mut hi = i32::MIN;
                for &idx in mem {
                    if idx >= 0 {
                        if idx < lo {
                            lo = idx;
                        }
                        if idx > hi {
                            hi = idx;
                        }
                    }
                }
                (lo, hi)
            })
            .collect();

        let per_q: Vec<(Vec<i32>, Vec<usize>, Vec<u32>)> = (0..t_q)
            .into_par_iter()
            .map(|pos| {
                let local_start = pos.saturating_sub(local_radius) as i32;
                let local_end = (pos + local_radius).min(seq_len.saturating_sub(1)) as i32;

                let selected: Vec<usize> = if b_total == 0 || top_p == 0 || top_k == 0 {
                    Vec::new()
                } else {
                    let row = &scores_mat[pos * b_total..(pos + 1) * b_total];
                    let mut pairs: Vec<(f32, usize)> = Vec::with_capacity(b_total);
                    for (idx, &s) in row.iter().enumerate() {
                        let (b_lo, b_hi) = bucket_spans[idx];
                        // Only rank buckets that contain at least one key outside the local window
                        if b_lo < local_start || b_hi > local_end {
                            pairs.push((s, idx));
                        }
                    }
                    let keep_p = top_p.min(pairs.len());
                    if keep_p < pairs.len() {
                        pairs.select_nth_unstable_by(keep_p, |a, b| {
                            b.0.partial_cmp(&a.0)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| a.1.cmp(&b.1))
                        });
                        pairs.truncate(keep_p);
                    }
                    pairs.sort_unstable_by(|a, b| {
                        b.0.partial_cmp(&a.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.1.cmp(&b.1))
                    });
                    pairs.into_iter().map(|(_, idx)| idx).collect()
                };

                let expanded = expand_buckets_bidirectional_single_query(
                    &selected,
                    members_b,
                    pos,
                    seq_len,
                    local_radius,
                    sink_size,
                    top_k,
                );
                let keep =
                    build_bidirectional_keep_set(&expanded, pos, seq_len, local_radius, sink_size);
                (expanded, selected, keep)
            })
            .collect();

        let mut exp_b = Vec::with_capacity(t_q);
        let mut sel_b = Vec::with_capacity(t_q);
        let mut keep_b = Vec::with_capacity(t_q);
        for (e, s, k) in per_q {
            exp_b.push(e);
            sel_b.push(s);
            keep_b.push(k);
        }
        all_expanded.push(exp_b);
        all_selected_buckets.push(sel_b);
        all_keep.push(keep_b);
    }

    Ok(ReadSelection {
        expanded_keys: all_expanded,
        selected_buckets: all_selected_buckets,
        keep_indices: all_keep,
        query_offset: 0,
        window_size: local_radius,
        sink_size,
        top_k,
    })
}

/// Complete single-layer Bidirectional D-LSA forward pass (with timing breakdown optional):
///
/// 1. Projects **pre-RoPE** `q_pre_rope` and `k_pre_rope` (`[B, H, T, Dh]`) into `[B, T, d_idx]`
///    unit index vectors via `projection` (or head regrouping if `None`).
/// 2. Ward-clusters all `T` index keys into `B_total` buckets across both directions.
/// 3. Selects top buckets per query outside `[pos - local_radius, pos + local_radius]` up to `top_k` keys.
/// 4. Runs CPU-native sparse gather SDPA over **post-RoPE** `q_rot`, `k_rot`, and `v`!
pub fn forward_bidirectional_dlsa(
    q_pre_rope: &Tensor,
    k_pre_rope: &Tensor,
    q_rot: &Tensor,
    k_rot: &Tensor,
    v: &Tensor,
    config: &BidirectionalLsaConfig,
    projection: Option<&LearnedProjection>,
    softmax_scale: Option<f32>,
) -> Result<(Tensor, Buckets, ReadSelection)> {
    let seq_len = q_rot.dim(2)?;

    // For short contexts where `seq_len <= 2 * local_radius + 1 + top_k`, every token is covered
    // by the local window + budget anyway, but we still run the full pipeline when requested.
    let (q_idx, k_idx) = match projection {
        Some(proj) => project(q_pre_rope, k_pre_rope, proj)?,
        None => regroup(q_pre_rope, k_pre_rope)?,
    };

    let mut buckets = build_buckets_bidirectional(
        &k_idx,
        config.cluster_block_size,
        config.pool_size,
        config.bucket_size,
        config.bucket_cap,
    )?;
    if let Some((rank, _)) = config.bucket_shape {
        shape_buckets(&mut buckets, &k_idx, rank)?;
    }
    let deviations = config.bucket_shape.map(|(_, z)| z).unwrap_or(0.0);

    let selection = read_bidirectional(
        &q_idx,
        &buckets,
        seq_len,
        config.local_radius,
        config.sink_size,
        config.top_p,
        config.top_k,
        deviations,
    )?;

    let out = sparse_gather_attention(q_rot, k_rot, v, &selection, softmax_scale)?;
    Ok((out, buckets, selection))
}

/// Distill a bidirectional `LearnedProjection` (`W_Q^{(idx)}, W_K^{(idx)} ∈ R^{d_{in} × d_{idx}}`)
/// for an encoder global layer (such as UltraBERT's 8 global layers) against the frozen dense
/// global attention teacher distribution over non-local keys `|j - i| > local_radius`.
///
/// Initializes `W_Q^{(idx)} = W_K^{(idx)}` from the top `index_dim` PCA eigenvectors of the
/// regrouped pre-RoPE keys, calibrates temperature `tau`, and performs `steps` of forward-KL
/// gradient descent on the non-local teacher attention distribution.
#[allow(clippy::too_many_arguments)]
pub fn distill_bidirectional_projection(
    q_pre_rope: &Tensor,
    k_pre_rope: &Tensor,
    q_rot: &Tensor,
    k_rot: &Tensor,
    index_dim: usize,
    local_radius: usize,
    steps: usize,
    lr: f32,
) -> Result<LearnedProjection> {
    let (b, h_q, t, d_h) = q_pre_rope.dims4()?;
    let (_, h_kv, _, _) = k_pre_rope.dims4()?;
    let group_size = h_q / h_kv;
    let d_in = h_kv * d_h;

    let q_in = q_pre_rope
        .to_dtype(DType::F32)?
        .reshape((b, h_kv, group_size, t, d_h))?
        .sum(2)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t, d_in))?;
    let k_in = k_pre_rope
        .to_dtype(DType::F32)?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b * t, d_in))?;

    // Initialize from leading PCA directions of the unrotated keys
    let init_proj = pca_projections(&k_in, index_dim)?;
    if steps == 0 || t <= 2 * local_radius + 2 {
        return Ok(init_proj);
    }

    // Compute teacher long-range attention probability distribution P_teacher[i, j]
    // averaged across heads on batch element 0 for `|j - i| > local_radius`.
    let q_r0 = q_rot.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::F32)?; // [Hq, T, Dh]
    let k_r0 = k_rot.narrow(0, 0, 1)?.squeeze(0)?.to_dtype(DType::F32)?; // [Hkv, T, Dh]
    let k_rep = if group_size > 1 {
        k_r0.unsqueeze(1)?
            .broadcast_as((h_kv, group_size, t, d_h))?
            .reshape((h_q, t, d_h))?
    } else {
        k_r0
    };
    let scale = 1.0 / (d_h as f64).sqrt();
    let teacher_logits = (q_r0.matmul(&k_rep.transpose(1, 2)?)? * scale)?; // [Hq, T, T]
    let teacher_probs_mean = crate::ops::softmax_last_dim(&teacher_logits)?
        .mean(0)? // [T, T]
        .flatten_all()?
        .to_vec1::<f32>()?;

    // Extract normalized non-local target mass per query i and weight each query by its
    // non-local KL concentration `D_KL(P_i || Uniform)^2` so uniform noise queries do not corrupt W_Q, W_K.
    let mut target_mass = vec![0.0f32; t * t];
    let mut query_weights = vec![0.0f32; t];
    for i in 0..t {
        let lo = i.saturating_sub(local_radius);
        let hi = (i + local_radius).min(t - 1);
        let mut row_sum = 0.0f32;
        let mut n_nonlocal = 0usize;
        for j in 0..t {
            if j < lo || j > hi {
                let p = teacher_probs_mean[i * t + j];
                target_mass[i * t + j] = p;
                row_sum += p;
                n_nonlocal += 1;
            }
        }
        if row_sum > 1e-12 && n_nonlocal > 1 {
            let inv = 1.0 / row_sum;
            let n_f = n_nonlocal as f32;
            let mut kl_unif = 0.0f32;
            for j in 0..t {
                if j < lo || j > hi {
                    let p_norm = target_mass[i * t + j] * inv;
                    target_mass[i * t + j] = p_norm;
                    if p_norm > 1e-12 {
                        kl_unif += p_norm * (p_norm * n_f).ln();
                    }
                }
            }
            let w = kl_unif.max(0.0);
            query_weights[i] = w * w;
        }
    }
    let w_sum: f32 = query_weights.iter().sum::<f32>().max(1e-12);
    for w in &mut query_weights {
        *w /= w_sum;
    }

    let q_in_0 = q_in.narrow(0, 0, t)?.flatten_all()?.to_vec1::<f32>()?;
    let k_in_0 = k_in.narrow(0, 0, t)?.flatten_all()?.to_vec1::<f32>()?;
    let mut w_q = init_proj
        .query_proj
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut w_k = init_proj
        .key_proj
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let tau = 0.1f32;
    let inv_tau = 1.0f32 / tau;

    let mut best_wq = w_q.clone();
    let mut best_wk = w_k.clone();
    let mut best_kl = f32::INFINITY;
    let mut cur_lr = lr;

    for _step in 0..steps {
        // Forward: u_q = q_in @ w_q, q_hat = u_q / ||u_q||; u_k = k_in @ w_k, k_hat = u_k / ||u_k||
        let mut q_hat = vec![0.0f32; t * index_dim];
        let mut q_inv_norm = vec![0.0f32; t];
        let mut k_hat = vec![0.0f32; t * index_dim];
        let mut k_inv_norm = vec![0.0f32; t];

        for i in 0..t {
            let q_row = &q_in_0[i * d_in..(i + 1) * d_in];
            let k_row = &k_in_0[i * d_in..(i + 1) * d_in];
            let mut uq_sq = 0.0f32;
            let mut uk_sq = 0.0f32;
            for r in 0..index_dim {
                let mut acc_q = 0.0f32;
                let mut acc_k = 0.0f32;
                for d in 0..d_in {
                    acc_q += q_row[d] * w_q[d * index_dim + r];
                    acc_k += k_row[d] * w_k[d * index_dim + r];
                }
                q_hat[i * index_dim + r] = acc_q;
                k_hat[i * index_dim + r] = acc_k;
                uq_sq += acc_q * acc_q;
                uk_sq += acc_k * acc_k;
            }
            let iq = 1.0 / uq_sq.sqrt().max(1e-6);
            let ik = 1.0 / uk_sq.sqrt().max(1e-6);
            q_inv_norm[i] = iq;
            k_inv_norm[i] = ik;
            for r in 0..index_dim {
                q_hat[i * index_dim + r] *= iq;
                k_hat[i * index_dim + r] *= ik;
            }
        }

        // Compute dL/dq_hat [T, index_dim] and dL/dk_hat [T, index_dim] under concentration-weighted Forward KL
        let mut grad_q_hat = vec![0.0f32; t * index_dim];
        let mut grad_k_hat = vec![0.0f32; t * index_dim];
        let mut step_ce_loss = 0.0f32;

        for i in 0..t {
            let qw = query_weights[i];
            if qw < 1e-7 {
                continue;
            }
            let lo = i.saturating_sub(local_radius);
            let hi = (i + local_radius).min(t - 1);
            let qi_hat = &q_hat[i * index_dim..(i + 1) * index_dim];

            let mut scores = vec![f32::NEG_INFINITY; t];
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..t {
                if j < lo || j > hi {
                    let kj_hat = &k_hat[j * index_dim..(j + 1) * index_dim];
                    let mut dot = 0.0f32;
                    for r in 0..index_dim {
                        dot += qi_hat[r] * kj_hat[r];
                    }
                    let s = dot * inv_tau;
                    if s > max_s {
                        max_s = s;
                    }
                    scores[j] = s;
                }
            }

            let mut sum_exp = 0.0f32;
            for j in 0..t {
                if j < lo || j > hi {
                    let e = (scores[j] - max_s).exp();
                    scores[j] = e;
                    sum_exp += e;
                } else {
                    scores[j] = 0.0;
                }
            }
            let inv_sum = 1.0 / sum_exp.max(1e-30);
            let scale_grad = qw * inv_tau;

            for j in 0..t {
                if j < lo || j > hi {
                    let student_p = (scores[j] * inv_sum).max(1e-12);
                    let target_p = target_mass[i * t + j];
                    if target_p > 1e-12 {
                        step_ce_loss -= qw * target_p * student_p.ln();
                    }
                    let dz = (student_p - target_p) * scale_grad;
                    let kj_hat = &k_hat[j * index_dim..(j + 1) * index_dim];
                    for r in 0..index_dim {
                        grad_q_hat[i * index_dim + r] += dz * kj_hat[r];
                        grad_k_hat[j * index_dim + r] += dz * qi_hat[r];
                    }
                }
            }
        }

        if step_ce_loss <= best_kl {
            best_kl = step_ce_loss;
            best_wq.copy_from_slice(&w_q);
            best_wk.copy_from_slice(&w_k);
        } else {
            // Step overshot the geodesic valley: restore best weights and halve step size
            w_q.copy_from_slice(&best_wq);
            w_k.copy_from_slice(&best_wk);
            cur_lr *= 0.5;
            continue;
        }

        // Backprop through unit normalization: dL/du = inv_norm * (dL/dhat - (dL/dhat . hat) * hat)
        let mut grad_wq = vec![0.0f32; d_in * index_dim];
        let mut grad_wk = vec![0.0f32; d_in * index_dim];
        for i in 0..t {
            let q_row = &q_in_0[i * d_in..(i + 1) * d_in];
            let k_row = &k_in_0[i * d_in..(i + 1) * d_in];
            let qi_hat = &q_hat[i * index_dim..(i + 1) * index_dim];
            let ki_hat = &k_hat[i * index_dim..(i + 1) * index_dim];
            let gq = &grad_q_hat[i * index_dim..(i + 1) * index_dim];
            let gk = &grad_k_hat[i * index_dim..(i + 1) * index_dim];

            let mut dot_q = 0.0f32;
            let mut dot_k = 0.0f32;
            for r in 0..index_dim {
                dot_q += gq[r] * qi_hat[r];
                dot_k += gk[r] * ki_hat[r];
            }

            let iq = q_inv_norm[i];
            let ik = k_inv_norm[i];
            for r in 0..index_dim {
                let du_q = iq * (gq[r] - dot_q * qi_hat[r]);
                let du_k = ik * (gk[r] - dot_k * ki_hat[r]);
                for d in 0..d_in {
                    grad_wq[d * index_dim + r] += q_row[d] * du_q;
                    grad_wk[d * index_dim + r] += k_row[d] * du_k;
                }
            }
        }

        let n_params = (d_in * index_dim).max(1) as f32;
        let rms_gq = (grad_wq.iter().map(|g| g * g).sum::<f32>() / n_params)
            .sqrt()
            .max(1e-8);
        let rms_gk = (grad_wk.iter().map(|g| g * g).sum::<f32>() / n_params)
            .sqrt()
            .max(1e-8);
        let step_scale = (cur_lr * 0.025) / (d_in as f32).sqrt();
        for idx in 0..(d_in * index_dim) {
            w_q[idx] -= step_scale * (grad_wq[idx] / rms_gq);
            w_k[idx] -= step_scale * (grad_wk[idx] / rms_gk);
        }
    }

    let device = q_pre_rope.device();
    let dtype = q_pre_rope.dtype();
    let query_proj = Tensor::from_vec(best_wq, (d_in, index_dim), device)?.to_dtype(dtype)?;
    let key_proj = Tensor::from_vec(best_wk, (d_in, index_dim), device)?.to_dtype(dtype)?;
    LearnedProjection::new(query_proj, key_proj)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det_tensor(shape: &[usize], seed: u64) -> Result<Tensor> {
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
        Tensor::from_vec(vals, shape, &Device::Cpu)
    }

    #[test]
    fn test_regroup_preserves_exact_logit_sum() -> Result<()> {
        let (b, h_q, h_kv, t_q, t_k, d) = (1, 8, 2, 6, 10, 16);
        let q = det_tensor(&[b, h_q, t_q, d], 101)?;
        let k = det_tensor(&[b, h_kv, t_k, d], 202)?;

        // Head-summed logits directly: sum_h q_{i,h} . k_{j, h / G}
        let group = h_q / h_kv;
        let k_rep = k
            .unsqueeze(2)?
            .broadcast_as((b, h_kv, group, t_k, d))?
            .reshape((b, h_q, t_k, d))?;
        let direct_sum = q.matmul(&k_rep.transpose(2, 3)?)?.sum(1)?; // [B, Tq, Tk]

        // Via regroup (unnormalized dot product equals direct_sum)
        let grouped_q = q
            .reshape((b, h_kv, group, t_q, d))?
            .sum(2)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t_q, h_kv * d))?;
        let flat_k = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t_k, h_kv * d))?;
        let regrouped_dot = grouped_q.matmul(&flat_k.transpose(1, 2)?)?;

        let diff = (direct_sum - regrouped_dot)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-5, "regroup dot diff {diff} >= 1e-5");

        // Also check normalized regroup produces unit rows
        let (q_idx, k_idx) = regroup(&q, &k)?;
        let q_norms = q_idx.sqr()?.sum(D::Minus1)?.sqrt()?.to_vec2::<f32>()?;
        for row in q_norms {
            for n in row {
                assert!((n - 1.0).abs() < 1e-5);
            }
        }
        let _ = k_idx;
        Ok(())
    }

    #[test]
    fn test_mla_latent_metric_isometry() -> Result<()> {
        let (b, h, t_q, t_k, d_free, r) = (1, 4, 5, 8, 12, 6);
        let q_free = det_tensor(&[b, h, t_q, d_free], 303)?;
        let kv_latent = det_tensor(&[b, t_k, r], 404)?;
        let w_up = det_tensor(&[h, d_free, r], 505)?;

        let metric = LatentMetric::from_key_up_projection(&w_up)?;
        let (q_idx, k_idx) = regroup_latent(&q_free, &kv_latent, &metric)?;
        assert_eq!(q_idx.dims(), &[b, t_q, r]);
        assert_eq!(k_idx.dims(), &[b, t_k, r]);

        // Expand keys explicitly: k_h = kv_latent @ W_h^T -> [B, H, Tk, d_free]
        let mut expanded_heads = Vec::with_capacity(h);
        for head in 0..h {
            let wh = w_up.narrow(0, head, 1)?.squeeze(0)?; // [d_free, r]
            let kh = kv_latent
                .squeeze(0)?
                .matmul(&wh.t()?)?
                .unsqueeze(0)?
                .unsqueeze(1)?; // [1, 1, Tk, d_free]
            expanded_heads.push(kh);
        }
        let k_expanded = Tensor::cat(&expanded_heads, 1)?;
        let (q_reg, k_reg) = regroup(&q_free, &k_expanded)?;

        // 1. Key-key cosine similarities in latent space must be an exact isometry of expanded keys
        let kk_latent = k_idx.matmul(&k_idx.transpose(1, 2)?)?;
        let kk_expanded = k_reg.matmul(&k_reg.transpose(1, 2)?)?;
        let kk_err = (kk_latent - kk_expanded)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(kk_err < 1e-5, "MLA key-key isometry error: {kk_err}");

        // 2. Query-key scores match up to a positive per-query scale factor (since wide queries have
        // components orthogonal to the r-dim key subspace that only inflate ||q_reg||)
        let narrow = q_idx.matmul(&k_idx.transpose(1, 2)?)?;
        let wide = q_reg.matmul(&k_reg.transpose(1, 2)?)?;
        let num = (&narrow * &wide)?.sum_keepdim(D::Minus1)?;
        let den = (&wide * &wide)?.sum_keepdim(D::Minus1)?;
        let scale = num.broadcast_div(&den)?;
        let min_scale = scale.min_all()?.to_scalar::<f32>()?;
        assert!(
            min_scale > 0.0,
            "per-query scale must be positive, got {min_scale}"
        );
        let max_err = (&narrow - wide.broadcast_mul(&scale)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            max_err < 1e-5,
            "MLA query-key scaled score error: {max_err}"
        );
        Ok(())
    }

    #[test]
    fn test_ward_clustering_invariants_and_shapes() -> Result<()> {
        let (b, t, d_idx) = (1, 128, 16);
        let window_size = 64;
        let pool_size = 4;
        let bucket_size = 8;
        let bucket_cap = 16;

        let keys = rownorm(&det_tensor(&[b, t, d_idx], 606)?)?;
        let mut buckets = build_buckets(&keys, window_size, pool_size, bucket_size, bucket_cap)?;

        // 2 closed blocks of 64 keys -> 8 buckets per block -> 16 buckets total
        assert_eq!(buckets.num_closed_blocks, 2);
        assert_eq!(buckets.buckets_per_block, 8);
        assert_eq!(buckets.centroids.dims(), &[1, 16, d_idx]);

        // Every bucket size must be in [pool_size..=bucket_cap] and sum to 128
        let total_keys: i32 = buckets.bucket_sizes[0].iter().sum();
        assert_eq!(total_keys, 128);
        for &sz in &buckets.bucket_sizes[0] {
            assert!(sz >= pool_size as i32 && sz <= bucket_cap as i32);
        }

        // Shape buckets and test bit-packing roundtrip
        shape_buckets(&mut buckets, &keys, 2)?;
        let shapes = &buckets.shapes.as_ref().unwrap()[0];
        let packed = pack_shape_basis(&shapes.directions)?;
        let unpacked = unpack_shape_basis(&packed, 16, 2, d_idx, &Device::Cpu, DType::F32)?;
        let diff = (&shapes.directions - &unpacked)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-6, "pack/unpack basis diff: {diff}");
        Ok(())
    }

    #[test]
    fn test_flat_tree_and_dense_attention_parity() -> Result<()> {
        let (b, h_q, h_kv, t, d_h) = (1, 4, 2, 160, 16);
        let q = det_tensor(&[b, h_q, t, d_h], 707)?;
        let k = det_tensor(&[b, h_kv, t, d_h], 808)?;
        let v = det_tensor(&[b, h_kv, t, d_h], 909)?;

        let window_size = 32;
        let sink_size = 4;
        let pool_size = 2;
        let bucket_size = 4;
        let bucket_cap = 8;

        let (q_idx, k_idx) = regroup(&q, &k)?;
        let buckets = build_buckets(&k_idx, window_size, pool_size, bucket_size, bucket_cap)?;

        // When top_p and top_k >= T, LSA selects every key and must match dense causal SDPA to ~1e-5!
        let sel_full = read(
            &q_idx,
            &buckets,
            window_size,
            sink_size,
            1024,
            1024,
            0.0,
            None,
            0,
        )?;
        let out_lsa_full = topk_attention(&q, &k, &v, &sel_full, None)?;

        // Compute dense causal attention reference
        let group = h_q / h_kv;
        let k_rep = k
            .unsqueeze(2)?
            .broadcast_as((b, h_kv, group, t, d_h))?
            .reshape((b, h_q, t, d_h))?;
        let v_rep = v
            .unsqueeze(2)?
            .broadcast_as((b, h_kv, group, t, d_h))?
            .reshape((b, h_q, t, d_h))?;
        let scale = 1.0 / (d_h as f64).sqrt();
        let mut mask = vec![0.0f32; t * t];
        for i in 0..t {
            for j in (i + 1)..t {
                mask[i * t + j] = f32::NEG_INFINITY;
            }
        }
        let mask_t =
            Tensor::from_vec(mask, (1, 1, t, t), &Device::Cpu)?.broadcast_as((b, h_q, t, t))?;
        let dense_scores = (q.matmul(&k_rep.transpose(2, 3)?)? * scale)?.add(&mask_t)?;
        let dense_out = crate::ops::softmax_last_dim(&dense_scores)?.matmul(&v_rep)?;

        let parity_diff = (&out_lsa_full - &dense_out)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            parity_diff < 1e-5,
            "LSA full-budget output diverged from dense causal attention: {parity_diff}"
        );

        // Also test Tree / Hierarchy search and incremental LsaLayerIndex
        let c0 = buckets.centroids.narrow(0, 0, 1)?.squeeze(0)?;
        let hier = build_hierarchy(
            &c0,
            &buckets.bucket_sizes[0],
            buckets.buckets_per_block,
            None,
            2,
            2,
            8,
        )?;
        let sel_tree = read_hierarchy(
            &q_idx,
            &buckets,
            &[hier],
            window_size,
            sink_size,
            4,
            16,
            4,
            8,
            Some(EarlyBudget {
                position: 64,
                top_p: 2,
                top_k: 8,
            }),
            0,
        )?;
        let out_tree = sparse_gather_attention(&q, &k, &v, &sel_tree, None)?;
        assert_eq!(out_tree.dims(), &[b, h_q, t, d_h]);

        // Test N-LSA two-path summary attention
        let out_two_path = lsa_two_path_attention(&q, &k, &v, &buckets, &sel_tree, None)?;
        assert_eq!(out_two_path.dims(), &[b, h_q, t, d_h]);

        Ok(())
    }
}
