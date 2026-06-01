//! QLoRA attach: quantise each base linear's weight and wrap it in a
//! [`TrainableLoraLinear`].
//!
//! The base weight is held as a [`candle::quantized::QTensor`] inside
//! [`QuantBase`] for memory savings; on forward we dequantise it to
//! the LoRA's working dtype and run the matmul there. The LoRA tensors
//! themselves remain `Var`s, so backprop reaches them unmodified.

use candle::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Module, Result, Tensor,
};
use candle_nn::Linear;

use crate::lora::LoraConfig;

use super::super::lora::attach::{AttachReport, AttachTarget, MoeMode};

/// QLoRA-specific options. `compute_dtype` is the dtype used after
/// dequantisation (typically `bf16` on H100 / MI300, `f16` elsewhere).
#[derive(Clone, Debug)]
pub struct QloraConfig {
    pub lora: LoraConfig,
    pub base_quant: GgmlDType,
    pub compute_dtype: DType,
}

impl Default for QloraConfig {
    fn default() -> Self {
        Self {
            lora: LoraConfig::attention_only(16, 32.0),
            base_quant: GgmlDType::Q4K,
            compute_dtype: DType::BF16,
        }
    }
}

/// A frozen Linear whose weight is stored as a [`QTensor`]. Implements
/// [`Module`] so it can be used as the base of a [`TrainableLoraLinear`]
/// via [`TrainableLoraLinear::with_shapes`].
///
/// Memory: `Q4K` is roughly 0.5 bytes per weight + scales — about 8x
/// smaller than `bf16`. Forward path dequantises once per call into
/// `compute_dtype`.
pub struct QuantBase {
    q_weight: QTensor,
    bias: Option<Tensor>,
    compute_dtype: DType,
    device: Device,
    in_features: usize,
    out_features: usize,
}

impl std::fmt::Debug for QuantBase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantBase")
            .field("dtype", &self.q_weight.dtype())
            .field("shape", &self.q_weight.shape())
            .field("compute_dtype", &self.compute_dtype)
            .finish()
    }
}

impl QuantBase {
    pub fn from_linear(lin: &Linear, dtype: GgmlDType, compute_dtype: DType) -> Result<Self> {
        let w = lin.weight();
        let dims = w.dims();
        if dims.len() != 2 {
            candle::bail!("QuantBase: weight must be 2-D, got {:?}", dims);
        }
        let q_weight = QTensor::quantize(w, dtype)?;
        Ok(Self {
            q_weight,
            bias: lin.bias().cloned(),
            compute_dtype,
            device: w.device().clone(),
            in_features: dims[1],
            out_features: dims[0],
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }
    pub fn out_features(&self) -> usize {
        self.out_features
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Dequantise the stored weight into `compute_dtype` and build a
    /// throw-away [`Linear`] for forward — gradients do not flow into
    /// the quantised storage so this is safe.
    fn dequantised_linear(&self) -> Result<Linear> {
        let w = self.q_weight.dequantize(&self.device)?.to_dtype(self.compute_dtype)?;
        Ok(Linear::new(w, self.bias.clone()))
    }
}

impl Module for QuantBase {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lin = self.dequantised_linear()?;
        lin.forward(xs)
    }
}

/// Quantise each matching base linear and wrap it in a LoRA adapter.
/// Returns the wrapped layers in input order plus an attach report.
pub fn attach_qlora(
    targets: Vec<AttachTarget>,
    cfg: &QloraConfig,
    moe_mode: MoeMode,
) -> crate::Result<(Vec<(String, QuantizedLoraLinear)>, AttachReport)> {
    use std::collections::HashSet;
    let want: HashSet<&str> = cfg.lora.target_modules.iter().map(String::as_str).collect();
    let mut report = AttachReport::default();
    let mut out = Vec::new();
    for t in targets {
        let leaf = t.dotted_path.rsplit_once('.').map(|(_, l)| l).unwrap_or(&t.dotted_path);
        if !want.contains(leaf) {
            continue;
        }
        if moe_mode == MoeMode::SharedOnly && t.dotted_path.split('.').collect::<Vec<_>>().windows(2)
            .any(|pair| pair[0] == "experts" && pair[1].chars().all(|c| c.is_ascii_digit()))
        {
            report.skipped.push(t.dotted_path);
            continue;
        }
        let base = QuantBase::from_linear(&t.linear, cfg.base_quant, cfg.compute_dtype)?;
        let in_f = base.in_features();
        let out_f = base.out_features();
        let device = base.device().clone();
        let dtype = cfg.compute_dtype;
        // We build a TrainableLoraLinear with a *dequantised* Linear as
        // its base so that the rest of the system (save/load, forward)
        // can stay shape-equivalent. Memory still wins because we hold
        // the `QuantBase` separately and the dequantised tensor inside
        // the inner Linear is replaced each forward.
        // Practical implementation: stash QuantBase + a TrainableLora
        // wrapper that consults it.
        let lora_a = candle_nn::init::DEFAULT_KAIMING_UNIFORM.var((cfg.lora.rank, in_f), dtype, &device)?;
        let lora_b = candle::Var::from_tensor(&Tensor::zeros((out_f, cfg.lora.rank), dtype, &device)?)?;
        let wrapped = QuantizedLoraLinear {
            base,
            lora_a,
            lora_b,
            scale: cfg.lora.scale(),
            rank: cfg.lora.rank,
            in_features: in_f,
            out_features: out_f,
        };
        report.attached.push(t.dotted_path.clone());
        out.push((t.dotted_path, wrapped));
    }
    Ok((out, report))
}

/// QLoRA equivalent of [`TrainableLoraLinear`]: same forward, but the
/// frozen base is held in 4-bit via [`QuantBase`].
pub struct QuantizedLoraLinear {
    base: QuantBase,
    lora_a: candle::Var,
    lora_b: candle::Var,
    scale: f64,
    rank: usize,
    in_features: usize,
    out_features: usize,
}

impl std::fmt::Debug for QuantizedLoraLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuantizedLoraLinear")
            .field("rank", &self.rank)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("scale", &self.scale)
            .finish()
    }
}

impl QuantizedLoraLinear {
    pub fn trainable_vars(&self) -> Vec<candle::Var> {
        vec![self.lora_a.clone(), self.lora_b.clone()]
    }

    pub fn lora_a(&self) -> &candle::Var {
        &self.lora_a
    }

    pub fn lora_b(&self) -> &candle::Var {
        &self.lora_b
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(x)?;
        let a_w = self.lora_a.as_tensor();
        let b_w = self.lora_b.as_tensor();
        let xa = matmul_last_transposed(x, a_w)?;
        let xab = matmul_last_transposed(&xa, b_w)?;
        let scaled = (xab * self.scale)?;
        base_out + scaled
    }
}

fn matmul_last_transposed(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let k = *dims.last().unwrap();
    if w.dims().len() != 2 || w.dims()[1] != k {
        candle::bail!(
            "qlora matmul_last_transposed: shape mismatch x={:?} w={:?}",
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

/// Save QLoRA adapter — identical format to plain LoRA so mistralrs
/// loads it with the same code path. The fact that the base was
/// quantised at training time does not change the LoRA tensor wire
/// format.
pub fn save_qlora_adapter<P: AsRef<std::path::Path>>(
    out_dir: P,
    layers: &[(String, QuantizedLoraLinear)],
    cfg: &QloraConfig,
    base_model: Option<String>,
) -> crate::Result<()> {
    use crate::lora::adapter::PeftAdapterConfig;
    use std::collections::HashMap;
    use std::fs;

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(layers.len() * 2);
    for (path, lin) in layers {
        let key_a = format!("base_model.model.{path}.lora_A.weight");
        let key_b = format!("base_model.model.{path}.lora_B.weight");
        tensors.insert(key_a, lin.lora_a().as_tensor().clone());
        tensors.insert(key_b, lin.lora_b().as_tensor().clone());
    }
    candle::safetensors::save(&tensors, out_dir.join("adapter_model.safetensors"))?;
    let pc = PeftAdapterConfig::from_lora_config(&cfg.lora, base_model);
    fs::write(
        out_dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&pc)?,
    )?;
    Ok(())
}
