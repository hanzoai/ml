//! High-level training loop.
//!
//! ```text
//! Trainer::new(model_forward, optimizer, data_iter)
//!     .step()         -> StepStats   // one optimizer step on next batch
//!     .save_adapter() -> ()          // emit PEFT adapter dir
//! ```
//!
//! `model_forward` is a closure: `&Tensor input_ids -> Tensor logits`.
//! That keeps the trainer architecture-agnostic — Qwen3, DeepSeek and
//! GLM4 all expose a `forward(input_ids, offset) -> logits` and the
//! caller decides what to pass.
//!
//! The loss is causal-LM cross-entropy with `IGNORE_INDEX = -100` per
//! HuggingFace convention. Logits are shifted left by 1 and labels are
//! consumed from index 1, so the standard
//! `logits[..-1]` ↔ `labels[1..]` alignment holds.

use candle::{DType, Tensor};
use candle_nn::loss::cross_entropy;

use crate::data::pack::{Batch, IGNORE_INDEX};
use crate::lora::TrainableLoraLinear;
use crate::optim::adamw::TrainableAdamW;

/// One step's accounting.
#[derive(Clone, Copy, Debug)]
pub struct StepStats {
    pub step: usize,
    pub loss: f64,
    pub lr: f64,
}

/// Trainer configuration.
#[derive(Clone, Debug)]
pub struct TrainerConfig {
    /// Save the adapter every `save_every` steps. `None` to disable.
    pub save_every: Option<usize>,
    /// Per-step gradient accumulation factor. We do not currently use
    /// gradient accumulation (one step = one batch) — this field is here
    /// so future expansion does not require API changes. Set to `1`.
    pub grad_accum: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            save_every: None,
            grad_accum: 1,
        }
    }
}

/// Trainer glues a forward closure to an optimizer and a batch iterator.
///
/// The lifetime parameter lets the closure capture references to the
/// wrapped model fields (`layers: &mut [DecoderLayer]`, etc.) without
/// any heap allocation per step.
pub struct Trainer<'a, F, I>
where
    F: FnMut(&Tensor) -> candle::Result<Tensor> + 'a,
    I: Iterator<Item = crate::Result<Batch>> + 'a,
{
    forward: F,
    optim: TrainableAdamW,
    data: I,
    cfg: TrainerConfig,
    _life: std::marker::PhantomData<&'a ()>,
}

impl<'a, F, I> Trainer<'a, F, I>
where
    F: FnMut(&Tensor) -> candle::Result<Tensor> + 'a,
    I: Iterator<Item = crate::Result<Batch>> + 'a,
{
    pub fn new(
        forward: F,
        optim: TrainableAdamW,
        data: I,
        cfg: TrainerConfig,
    ) -> Self {
        Self {
            forward,
            optim,
            data,
            cfg,
            _life: std::marker::PhantomData,
        }
    }

    /// Pull the next batch and run one optimizer step. Returns `None`
    /// once the data iterator is exhausted.
    pub fn step(&mut self) -> Option<crate::Result<StepStats>> {
        let batch = match self.data.next()? {
            Ok(b) => b,
            Err(e) => return Some(Err(e)),
        };
        Some(self.run_step(batch))
    }

    fn run_step(&mut self, batch: Batch) -> crate::Result<StepStats> {
        let logits = (self.forward)(&batch.input_ids)?;
        let loss = causal_lm_loss(&logits, &batch.labels)?;
        let loss_value = loss.to_dtype(DType::F32)?.to_scalar::<f32>()? as f64;
        let lr = self.optim.backward_step(&loss)?;
        Ok(StepStats {
            step: self.optim.step_count(),
            loss: loss_value,
            lr,
        })
    }

    /// Save the current adapter weights as a PEFT directory. The
    /// `layers` argument is the same `Vec<(String, TrainableLoraLinear)>`
    /// returned by [`crate::lora::attach::attach_lora`] — pass it back
    /// in so the trainer can persist their `Var`s without owning them.
    pub fn save_adapter<P: AsRef<std::path::Path>>(
        &self,
        out_dir: P,
        layers: &[(String, TrainableLoraLinear)],
        cfg: &crate::lora::LoraConfig,
        base_model: Option<String>,
    ) -> crate::Result<()> {
        crate::lora::save_peft_adapter(out_dir, layers, cfg, base_model)
    }

    pub fn step_count(&self) -> usize {
        self.optim.step_count()
    }

    pub fn config(&self) -> &TrainerConfig {
        &self.cfg
    }
}

/// Causal-LM cross-entropy with `-100` ignore mask.
///
/// `logits` shape `[B, T, V]`. `labels` shape `[B, T]`.
/// We shift: predict `labels[..., 1:]` from `logits[..., :-1, :]`.
pub fn causal_lm_loss(logits: &Tensor, labels: &Tensor) -> candle::Result<Tensor> {
    let (b, t, v) = logits.dims3()?;
    if t < 2 {
        candle::bail!("causal_lm_loss: seq_len must be >= 2, got {t}");
    }
    let shift_logits = logits.narrow(1, 0, t - 1)?;
    let shift_labels = labels.narrow(1, 1, t - 1)?;
    // Flatten to (B*(T-1), V) and (B*(T-1),).
    let flat_logits = shift_logits.reshape((b * (t - 1), v))?;
    let flat_labels = shift_labels.reshape((b * (t - 1),))?;

    // Mask out IGNORE_INDEX rows. We do this by:
    //   1. Compute per-token CE on a safe-label tensor (clamp -100 -> 0).
    //   2. Multiply by a 0/1 mask derived from the original labels.
    //   3. Divide by mask sum.
    //
    // candle_nn::loss::cross_entropy returns the mean; for masked loss
    // we need element-wise NLL.
    let is_keep = flat_labels.ne(IGNORE_INDEX)?.to_dtype(flat_logits.dtype())?;
    let zero = Tensor::zeros_like(&flat_labels)?;
    let safe_labels = flat_labels.where_cond(&flat_labels.ne(IGNORE_INDEX)?, &zero)?;
    // Per-row CE: -log_softmax(logits)[label]
    let log_probs = candle_nn::ops::log_softmax(&flat_logits, candle::D::Minus1)?;
    let gathered = log_probs.gather(&safe_labels.unsqueeze(1)?, 1)?.squeeze(1)?;
    let nll = gathered.neg()?;
    let masked = nll.mul(&is_keep)?;
    let n = is_keep.sum_all()?;
    masked.sum_all()?.broadcast_div(&n.maximum(1f32)?)
}

// Unused import guard
#[allow(dead_code)]
const _: fn() = || {
    let _ = cross_entropy;
};
