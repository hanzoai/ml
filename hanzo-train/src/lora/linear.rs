//! `TrainableLoraLinear` — wraps a frozen base [`candle_nn::Linear`] with
//! trainable `lora_A` and `lora_B` projections that are exposed as
//! [`candle::Var`] for the optimizer.
//!
//! Forward computation matches PEFT's `LoraLayer.forward`:
//!
//! ```text
//! y = base(x) + scale * B(A(dropout(x)))
//! scale = alpha / rank
//! ```
//!
//! Storage layout matches the PEFT wire format that
//! `mistralrs-core::lora::make_adapter` expects:
//!
//! * `lora_A.weight` of shape `(rank, in_features)`
//! * `lora_B.weight` of shape `(out_features, rank)`
//!
//! Initialisation matches the PEFT default: Kaiming-uniform on A,
//! zeros on B (so the initial delta is zero and the wrapped model
//! is functionally identical to the base at step 0).

use candle::{DType, Device, Module, Result, Tensor, Var};
use candle_nn::{init, ops::Dropout, Linear};

use super::LoraConfig;

/// A trainable LoRA adapter sitting on top of a frozen base [`Linear`].
///
/// The base layer's weight (and bias) are held by-value but are **not**
/// wrapped as a `Var`, so they will not appear in the optimizer's
/// trainable-parameter list — only `lora_a` and `lora_b` do.
#[derive(Debug)]
pub struct TrainableLoraLinear {
    base: Linear,
    /// `(rank, in_features)`
    lora_a: Var,
    /// `(out_features, rank)`
    lora_b: Var,
    scale: f64,
    dropout: Dropout,
    in_features: usize,
    out_features: usize,
    rank: usize,
}

impl TrainableLoraLinear {
    /// Wrap a frozen `base` Linear and create freshly-initialised LoRA
    /// matrices on the same device + dtype.
    pub fn new(base: Linear, cfg: &LoraConfig) -> Result<Self> {
        let (out_features, in_features) = {
            let dims = base.weight().dims();
            if dims.len() != 2 {
                candle::bail!(
                    "TrainableLoraLinear: base weight must be 2-D, got {:?}",
                    dims
                );
            }
            (dims[0], dims[1])
        };
        let device = base.weight().device().clone();
        let dtype = base.weight().dtype();
        Self::with_shapes(base, in_features, out_features, &device, dtype, cfg)
    }

    /// Build with explicit shapes — used when the base layer's weight is
    /// quantised (so `base.weight()` does not return a usable shape).
    pub fn with_shapes(
        base: Linear,
        in_features: usize,
        out_features: usize,
        device: &Device,
        dtype: DType,
        cfg: &LoraConfig,
    ) -> Result<Self> {
        if cfg.rank == 0 {
            candle::bail!("LoRA rank must be > 0");
        }
        // PEFT default: Kaiming uniform for A, zeros for B.
        let a_init = init::DEFAULT_KAIMING_UNIFORM;
        let a = a_init.var((cfg.rank, in_features), dtype, device)?;
        let b = Tensor::zeros((out_features, cfg.rank), dtype, device)?;
        let b = Var::from_tensor(&b)?;

        Ok(Self {
            base,
            lora_a: a,
            lora_b: b,
            scale: cfg.scale(),
            dropout: Dropout::new(cfg.dropout),
            in_features,
            out_features,
            rank: cfg.rank,
        })
    }

    /// Trainable params: `[lora_a, lora_b]`.
    pub fn trainable_vars(&self) -> Vec<Var> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    /// Reference to the frozen base layer.
    pub fn base(&self) -> &Linear {
        &self.base
    }

    pub fn lora_a(&self) -> &Var {
        &self.lora_a
    }

    pub fn lora_b(&self) -> &Var {
        &self.lora_b
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// `y = base(x) + scale * B(A(dropout(x)))`.
    ///
    /// `train` toggles dropout. Pass `false` for eval / inference.
    pub fn forward_with_training(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        if self.rank == 0 {
            return Ok(base_out);
        }
        let x_d = self.dropout.forward(x, train)?;

        // A: (r, in)  ->  A x^T has shape (r, ...) when we use matmul with x.t().
        // Implementation: a_linear = Linear::new(a_w, None); b_linear similarly.
        // We do not allocate Linear because we want the raw Var to track grads.
        let a_w = self.lora_a.as_tensor();
        let b_w = self.lora_b.as_tensor();

        // x_d shape: (..., in). We compute x_d @ A^T -> (..., r), then @ B^T -> (..., out).
        let xa = matmul_last_transposed(&x_d, a_w)?;
        let xab = matmul_last_transposed(&xa, b_w)?;
        let scaled = (xab * self.scale)?;
        base_out + scaled
    }

    /// Convenience inference forward (no dropout).
    pub fn forward_eval(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_training(x, false)
    }
}

impl Module for TrainableLoraLinear {
    /// Defaults to eval (no dropout). Use [`forward_with_training`] inside
    /// training loops.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_training(x, false)
    }
}

/// Compute `x @ w^T` for arbitrary leading batch dims on `x` and a 2-D `w`.
///
/// `x.shape = (..., k)`, `w.shape = (m, k)`  ->  result `(..., m)`.
fn matmul_last_transposed(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let k = *dims.last().unwrap();
    if w.dims().len() != 2 || w.dims()[1] != k {
        candle::bail!(
            "matmul_last_transposed: shape mismatch x={:?} w={:?}",
            dims,
            w.dims()
        );
    }
    let m = w.dims()[0];
    let lead: usize = dims.iter().take(dims.len() - 1).product();
    let x2 = x.reshape((lead, k))?;
    let wt = w.t()?;
    let out = x2.matmul(&wt)?;
    let mut out_shape: Vec<usize> = dims.iter().take(dims.len() - 1).copied().collect();
    out_shape.push(m);
    out.reshape(out_shape)
}
