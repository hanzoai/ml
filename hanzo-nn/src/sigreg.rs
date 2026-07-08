//! SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA).
//!
//! LeJEPA (Balestriero & LeCun, *Provable and Scalable Self-Supervised Learning
//! Without the Heuristics*, [arXiv:2511.08544](https://arxiv.org/abs/2511.08544))
//! replaces the stop-gradient / EMA / teacher-student heuristics of prior JEPAs
//! with a single, theory-backed regularizer: constrain the embedding distribution
//! toward an isotropic Gaussian. SIGReg enforces that by a statistical goodness-of-fit
//! test applied to many random 1-D projections ("sketches") of the embeddings — if
//! every 1-D marginal is standard normal, the joint is isotropic Gaussian.
//!
//! The default univariate test is **Epps–Pulley**, an empirical-characteristic-function
//! normality test. For samples \\(x_1,\dots,x_N\\) of one projection, with empirical
//! characteristic function \\(\hat\varphi(t)=\frac1N\sum_j e^{i t x_j}\\) and the
//! standard-normal characteristic function \\(\varphi(t)=e^{-t^2/2}\\), the statistic is
//!
//! \\[ T = N \int_{-\infty}^{\infty} \bigl|\hat\varphi(t)-\varphi(t)\bigr|^2\, e^{-t^2/2}\,dt. \\]
//!
//! Splitting into real/imaginary parts and exploiting the even symmetry of the
//! integrand (integrate on \\([0,t_{\max}]\\) and double), the trapezoidal quadrature is
//!
//! \\[ T = N \sum_k w_k\Bigl[\bigl(\tfrac1N\textstyle\sum_j\cos(t_k x_j)-\varphi(t_k)\bigr)^2
//!        + \bigl(\tfrac1N\textstyle\sum_j\sin(t_k x_j)\bigr)^2\Bigr]. \\]
//!
//! This is a first-order objective (no stop-gradient); the SIGReg loss is the mean of
//! \\(T\\) over the random projections. In LeJEPA it is paired with a prediction /
//! invariance loss over multi-crop views and traded off by a single \\(\lambda\\):
//! `loss = sigreg * lambda + inv * (1 - lambda)`. This module implements the SIGReg
//! loss itself; the view generation and \\(\lambda\\) schedule belong to the training
//! pipeline (see the demo in the tests).

use hanzo_ml::{DType, Device, Result, Tensor, D};

/// SIGReg objective configured by its Epps–Pulley quadrature.
///
/// [`SigReg::default`] reproduces the LeJEPA paper's settings (`t_max = 3`, 17 knots).
#[derive(Clone, Copy, Debug)]
pub struct SigReg {
    /// Upper bound of the characteristic-function integration domain \\([0, t_{\max}]\\).
    pub t_max: f64,
    /// Number of trapezoidal quadrature knots on \\([0, t_{\max}]\\) (symmetry-doubled).
    pub n_knots: usize,
}

impl Default for SigReg {
    fn default() -> Self {
        Self {
            t_max: 3.0,
            n_knots: 17,
        }
    }
}

/// Precomputed trapezoidal quadrature: knot positions `t`, target CF `phi = e^{-t^2/2}`,
/// and integration weights already folded with the Gaussian window `w_k * phi_k`.
struct Quadrature {
    t: Tensor,
    phi: Tensor,
    weights: Tensor,
}

impl SigReg {
    fn quadrature(&self, dtype: DType, device: &Device) -> Result<Quadrature> {
        let k = self.n_knots;
        if k < 2 {
            hanzo_ml::bail!("SIGReg needs at least 2 quadrature knots, got {k}");
        }
        let dt = self.t_max / (k - 1) as f64;
        let mut t = Vec::with_capacity(k);
        let mut phi = Vec::with_capacity(k);
        let mut weights = Vec::with_capacity(k);
        for i in 0..k {
            let ti = i as f64 * dt;
            // Trapezoidal weights doubled for the [-t_max, t_max] -> [0, t_max] fold,
            // with half weight at the two endpoints; window function e^{-t^2/2} folded in.
            let wi = if i == 0 || i == k - 1 { dt } else { 2.0 * dt };
            let pi = (-0.5 * ti * ti).exp();
            t.push(ti);
            phi.push(pi);
            weights.push(wi * pi);
        }
        Ok(Quadrature {
            t: Tensor::from_vec(t, k, device)?.to_dtype(dtype)?,
            phi: Tensor::from_vec(phi, k, device)?.to_dtype(dtype)?,
            weights: Tensor::from_vec(weights, k, device)?.to_dtype(dtype)?,
        })
    }

    /// Per-projection Epps–Pulley statistic for already-projected data.
    ///
    /// * `proj`: `(n, s)` — `n` samples of `s` univariate projections.
    ///
    /// Returns `(s,)`: the statistic \\(T\\) for each projection. Larger values mean the
    /// projection's empirical distribution is farther from standard normal.
    pub fn statistic(&self, proj: &Tensor) -> Result<Tensor> {
        let (n, _s) = proj.dims2()?;
        let q = self.quadrature(proj.dtype(), proj.device())?;
        // x_t[j, s, k] = proj[j, s] * t[k]  ->  (n, s, k)
        let x_t = proj.unsqueeze(D::Minus1)?.broadcast_mul(&q.t)?;
        // Empirical characteristic function, averaged over the n samples -> (s, k).
        let cos_mean = x_t.cos()?.mean(0)?;
        let sin_mean = x_t.sin()?.mean(0)?;
        // |phi_hat - phi|^2 = (Re - phi)^2 + Im^2
        let real = cos_mean.broadcast_sub(&q.phi)?.sqr()?;
        let imag = sin_mean.sqr()?;
        let err = (real + imag)?;
        // Weighted integration over knots, scaled by n -> (s,).
        let stat = err.broadcast_mul(&q.weights)?.sum(D::Minus1)?;
        stat * n as f64
    }

    /// SIGReg loss for `embeddings` projected onto explicit unit-norm `directions`.
    ///
    /// * `embeddings`: `(n, d)`.
    /// * `directions`: `(d, s)` — each column a unit-norm projection direction.
    ///
    /// Returns the scalar mean of [`SigReg::statistic`] over the `s` slices.
    pub fn loss_with_directions(&self, embeddings: &Tensor, directions: &Tensor) -> Result<Tensor> {
        let (_n, d) = embeddings.dims2()?;
        let (dd, _s) = directions.dims2()?;
        if d != dd {
            hanzo_ml::bail!("embeddings dim ({d}) and directions dim ({dd}) mismatch");
        }
        let proj = embeddings.matmul(directions)?;
        self.statistic(&proj)?.mean_all()
    }

    /// SIGReg loss with `n_slices` random unit projection directions.
    ///
    /// Directions are freshly sampled from the standard normal on the embeddings'
    /// device and L2-normalized per column, matching the paper's per-step resampling.
    /// Seed via [`hanzo_ml::Device::set_seed`] for deterministic runs.
    pub fn loss(&self, embeddings: &Tensor, n_slices: usize) -> Result<Tensor> {
        let (_n, d) = embeddings.dims2()?;
        let directions = random_directions(d, n_slices, embeddings.dtype(), embeddings.device())?;
        self.loss_with_directions(embeddings, &directions)
    }
}

/// Sample `n_slices` unit-norm projection directions in \\(\mathbb{R}^d\\).
///
/// Columns are drawn i.i.d. standard normal and L2-normalized, giving uniform
/// directions on the unit sphere. Result shape is `(d, n_slices)`.
pub fn random_directions(
    d: usize,
    n_slices: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let a = Tensor::randn(0f32, 1f32, (d, n_slices), device)?.to_dtype(dtype)?;
    let norm = a.sqr()?.sum_keepdim(0)?.sqrt()?;
    a.broadcast_div(&norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    // Reference fixture generated from the official rbalestr-lab/lejepa EppsPulley.
    include!("sigreg_fixture.rs");

    fn max_abs_diff(a: &Tensor, b: &[f64]) -> f64 {
        let a = a.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
        let a = a.to_vec1::<f64>().unwrap();
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    /// The trapezoidal quadrature buffers must match the official EppsPulley buffers.
    #[test]
    fn quadrature_matches_official() {
        let dev = Device::Cpu;
        let cfg = SigReg {
            t_max: T_MAX,
            n_knots: N_KNOTS,
        };
        let q = cfg.quadrature(DType::F64, &dev).unwrap();
        assert!(max_abs_diff(&q.t, REF_T) < 1e-12, "t buffer mismatch");
        assert!(max_abs_diff(&q.phi, REF_PHI) < 1e-12, "phi buffer mismatch");
        assert!(
            max_abs_diff(&q.weights, REF_WEIGHTS) < 1e-12,
            "weights buffer mismatch"
        );
    }

    /// FALSIFIABLE GATE: the per-slice statistic and the mean SIGReg loss must match
    /// the official implementation to 1e-5 on fixed inputs and fixed directions.
    #[test]
    fn statistic_matches_official_f64() {
        let dev = Device::Cpu;
        let cfg = SigReg {
            t_max: T_MAX,
            n_knots: N_KNOTS,
        };
        let proj = Tensor::from_slice(PROJ, (N, S), &dev).unwrap();
        let stat = cfg.statistic(&proj).unwrap();
        let d = max_abs_diff(&stat, REF_PER_SLICE);
        assert!(d < 1e-5, "per-slice statistic diff {d:e} exceeds 1e-5");

        // Full chain from embeddings + directions through the mean loss.
        let x = Tensor::from_slice(X, (N, D), &dev).unwrap();
        let a = Tensor::from_slice(A, (D, S), &dev).unwrap();
        let loss = cfg.loss_with_directions(&x, &a).unwrap();
        let dl = (loss.to_scalar::<f64>().unwrap() - REF_SIGREG).abs();
        assert!(dl < 1e-5, "sigreg loss diff {dl:e} exceeds 1e-5");
    }

    /// The shipped f32 path also matches the official stock-f32 statistic (per-slice
    /// values reach ~40, so the tolerance is relative to that scale, well under 1e-4).
    #[test]
    fn statistic_matches_official_f32() {
        let dev = Device::Cpu;
        let cfg = SigReg {
            t_max: T_MAX,
            n_knots: N_KNOTS,
        };
        let proj = Tensor::from_slice(PROJ, (N, S), &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let stat = cfg.statistic(&proj).unwrap();
        let d = max_abs_diff(&stat, REF_PER_SLICE_F32);
        assert!(d < 1e-3, "f32 per-slice statistic diff {d:e} exceeds 1e-3");

        let x = Tensor::from_slice(X, (N, D), &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let a = Tensor::from_slice(A, (D, S), &dev)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let loss = cfg.loss_with_directions(&x, &a).unwrap();
        let dl = (loss.to_scalar::<f32>().unwrap() as f64 - REF_SIGREG_F32).abs();
        assert!(dl < 1e-3, "f32 sigreg loss diff {dl:e} exceeds 1e-3");
    }

    /// `random_directions` produces unit-norm columns.
    #[test]
    fn random_directions_are_unit_norm() {
        let dev = Device::Cpu;
        let dirs = random_directions(16, 64, DType::F32, &dev).unwrap();
        let norms = dirs
            .sqr()
            .unwrap()
            .sum_keepdim(0)
            .unwrap()
            .sqrt()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for nrm in norms {
            assert!((nrm - 1.0).abs() < 1e-5, "column norm {nrm} != 1");
        }
    }

    // ---- Trainable demo: a 2-layer MLP encoder driven toward isotropy by SIGReg ----

    /// Variance of the covariance eigenvalues WITHOUT an eigensolver: for a symmetric
    /// covariance `C`, the Frobenius norm is rotation-invariant, so
    /// `||C - (tr(C)/D) I||_F^2 = sum_i (lambda_i - mean_lambda)^2`. And since
    /// `C = Zc^T Zc / N`, `tr(C) = ||Zc||_F^2 / N`. Shrinking this = the embedding
    /// covariance moving toward a scaled identity (isotropy).
    fn eigenvalue_spread(z: &Tensor) -> f64 {
        let (n, d) = z.dims2().unwrap();
        let mean = z.mean_keepdim(0).unwrap();
        let zc = z.broadcast_sub(&mean).unwrap();
        let cov = (zc.t().unwrap().matmul(&zc).unwrap() / n as f64).unwrap();
        let fro_sq = cov
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap();
        let trace = (zc
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f64>()
            .unwrap())
            / n as f64;
        fro_sq - trace * trace / d as f64
    }

    #[test]
    fn demo_training_drives_embeddings_toward_isotropy() {
        use crate::optim::{AdamW, Optimizer, ParamsAdamW};
        use hanzo_ml::Var;
        use rand::{rngs::StdRng, SeedableRng};
        use rand_distr::{Distribution, StandardNormal};

        // The CPU backend rng is not seedable, so draw all randomness from a seeded
        // StdRng into explicit tensors -> the demo is fully deterministic.
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(42);
        let mut randn = |rows: usize, cols: usize| {
            let v: Vec<f32> = (0..rows * cols)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            Tensor::from_vec(v, (rows, cols), &dev).unwrap()
        };

        let (n, in_dim, hidden, embed) = (64usize, 6usize, 16usize, 8usize);
        let sigreg = SigReg::default();
        let lambda = 0.7; // weight on SIGReg vs the invariance/prediction loss

        // Anisotropic synthetic inputs: widely varying per-dimension scales so the
        // initial embedding covariance is far from isotropic.
        let scale = Tensor::from_vec(
            (0..in_dim)
                .map(|i| 0.3 + 0.5 * i as f32)
                .collect::<Vec<_>>(),
            (1, in_dim),
            &dev,
        )
        .unwrap();
        let x_base = randn(n, in_dim).broadcast_mul(&scale).unwrap();

        // Two views = base +/- small augmentation noise (the JEPA prediction target).
        let noise = (randn(n, in_dim) * 0.1).unwrap();
        let v1 = (&x_base + &noise).unwrap();
        let v2 = (&x_base - &noise).unwrap();

        // 2-layer MLP encoder parameters (Kaiming-ish scaled init). Vars own the
        // storage; forward reads the live tensors so optimizer updates are visible.
        let w1 =
            Var::from_tensor(&(randn(in_dim, hidden) * (1.0 / (in_dim as f64).sqrt())).unwrap())
                .unwrap();
        let b1 = Var::from_tensor(&Tensor::zeros((1, hidden), DType::F32, &dev).unwrap()).unwrap();
        let w2 =
            Var::from_tensor(&(randn(hidden, embed) * (1.0 / (hidden as f64).sqrt())).unwrap())
                .unwrap();
        let b2 = Var::from_tensor(&Tensor::zeros((1, embed), DType::F32, &dev).unwrap()).unwrap();
        let params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];

        // A fixed bank of 128 unit-norm projection directions (deterministic). The
        // paper resamples directions each step; the LOSS module supports both (see
        // `SigReg::loss`), and this demo uses a fixed bank for test determinism.
        let dirs = {
            let a = randn(embed, 48);
            let norm = a.sqr().unwrap().sum_keepdim(0).unwrap().sqrt().unwrap();
            a.broadcast_div(&norm).unwrap()
        };

        let forward = |x: &Tensor| -> Tensor {
            x.matmul(w1.as_tensor())
                .unwrap()
                .broadcast_add(b1.as_tensor())
                .unwrap()
                .relu()
                .unwrap()
                .matmul(w2.as_tensor())
                .unwrap()
                .broadcast_add(b2.as_tensor())
                .unwrap()
        };
        let embeddings = || {
            let e1 = forward(&v1);
            let e2 = forward(&v2);
            Tensor::cat(&[&e1, &e2], 0)
                .unwrap()
                .to_dtype(DType::F64)
                .unwrap()
        };
        // One optimization step's loss: LeJEPA = sigreg*lambda + invariance*(1-lambda).
        let step_loss = || -> (Tensor, f64) {
            let e1 = forward(&v1);
            let e2 = forward(&v2);
            let inv = (&e1 - &e2).unwrap().sqr().unwrap().mean_all().unwrap();
            let emb = Tensor::cat(&[&e1, &e2], 0).unwrap();
            let sig = sigreg.loss_with_directions(&emb, &dirs).unwrap();
            let loss = ((&sig * lambda).unwrap() + (inv * (1.0 - lambda)).unwrap()).unwrap();
            let sig_v = sig.to_scalar::<f32>().unwrap() as f64;
            (loss, sig_v)
        };

        let mut opt = AdamW::new(
            params,
            ParamsAdamW {
                lr: 1e-2,
                ..Default::default()
            },
        )
        .unwrap();

        let (loss0_t, sig0) = step_loss();
        let loss0 = loss0_t.to_scalar::<f32>().unwrap();
        let spread0 = eigenvalue_spread(&embeddings());

        let mut last = loss0;
        for _ in 0..150 {
            let (loss, _) = step_loss();
            last = loss.to_scalar::<f32>().unwrap();
            opt.backward_step(&loss).unwrap();
        }

        let (_, sig1) = step_loss();
        let spread1 = eigenvalue_spread(&embeddings());

        eprintln!(
            "loss {loss0:.4} -> {last:.4}   sigreg {sig0:.4} -> {sig1:.4}   eig-spread {spread0:.4} -> {spread1:.4}"
        );

        assert!(
            last < loss0,
            "total loss did not decrease: {loss0} -> {last}"
        );
        assert!(
            sig1 < sig0,
            "sigreg statistic did not decrease: {sig0} -> {sig1}"
        );
        assert!(
            spread1 < 0.7 * spread0,
            "eigenvalue spread did not measurably shrink: {spread0} -> {spread1}"
        );
    }
}
