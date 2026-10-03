//! ModernBERT
//!
//! ModernBERT is a modernized bidirectional encoder-only Transformer model.
//! - [Arxiv](https://arxiv.org/abs/2412.13663) "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference"
//! - Upstream [GitHub repo](https://github.com/AnswerDotAI/ModernBERT).
//! - See modernbert in [hanzo-ml-examples](https://github.com/hanzoai/ml/tree/main/hanzo-ml-examples/) for runnable code
//!

use hanzo_ml::quantized::{repack::tiled, GgmlDType, QMatMul, QTensor};
use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_nn::{
    embedding, layer_norm_no_bias, linear, linear_no_bias,
    ops::{softmax, softmax_last_dim},
    Embedding, LayerNorm, Linear, Module, VarBuilder,
};
use serde::Deserialize;

use core::f32;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    /// YaRN's stretch `s` (Peng et al. 2023, arXiv:2309.00071): a sequence longer than
    /// `max_position_embeddings` reads its global layers' rotary tables interpolated `s` times
    /// and its local layers' extended, to [`Config::positions`]; a sequence within
    /// `max_position_embeddings` reads the trained tables, unchanged.
    #[serde(default)]
    pub yarn: Option<f64>,
    #[serde(default)]
    #[serde(flatten)]
    pub classifier_config: Option<ClassifierConfig>,
}

impl Config {
    /// The longest sequence the backbone reads: its trained positions, times `yarn`.
    pub fn positions(&self) -> usize {
        self.yarn.map_or(self.max_position_embeddings, |s| {
            (self.max_position_embeddings as f64 * s).round() as usize
        })
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Copy, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierPooling {
    #[default]
    CLS,
    MEAN,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClassifierConfig {
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, String>,
    pub classifier_pooling: ClassifierPooling,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// YaRN's tables over positions `0..n` for a stretch `s` of `original` trained positions
    /// (Peng et al. 2023, §3.2-3.4): a frequency whose wavelength fits in `original` 32 times or
    /// more is kept, one whose wavelength exceeds `original` is divided by `s`, the rest blended
    /// linearly between (NTK-by-parts, β = 32, α = 1, the band's ends rounded outward); `cos` and
    /// `sin` carry the attention factor `0.1 ln s + 1`. Angles in f64, stored in `dtype`. `s = 1`
    /// is the trained frequencies over more positions.
    fn yarn(
        dtype: DType,
        config: &Config,
        rope_theta: f64,
        s: f64,
        n: usize,
        dev: &Device,
    ) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let half = dim / 2;
        let original = config.max_position_embeddings as f64;
        // the frequency index whose wavelength fits in `original` `turns` times
        let index = |turns: f64| {
            dim as f64 * (original / (turns * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * rope_theta.ln())
        };
        let low = index(32.0).floor().max(0.0);
        let high = index(1.0).ceil().min(dim as f64 - 1.0);
        let high = if high == low { high + 0.001 } else { high };
        let freq: Vec<f64> = (0..half)
            .map(|i| {
                let f = 1.0 / rope_theta.powf((2 * i) as f64 / dim as f64);
                let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
                f / s * ramp + f * (1.0 - ramp)
            })
            .collect();
        let mul = 0.1 * s.ln() + 1.0;
        let mut cos = Vec::with_capacity(n * half);
        let mut sin = Vec::with_capacity(n * half);
        for p in 0..n {
            for f in &freq {
                let (sn, cs) = (p as f64 * f).sin_cos();
                cos.push((cs * mul) as f32);
                sin.push((sn * mul) as f32);
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (n, half), dev)?.to_dtype(dtype)?,
            sin: Tensor::from_vec(sin, (n, half), dev)?.to_dtype(dtype)?,
        })
    }

    /// The trained tables followed by `long`'s: position `p` of `long` is row
    /// `trained + p`.
    fn join(&self, long: &RotaryEmbedding) -> Result<Self> {
        Ok(Self {
            cos: Tensor::cat(&[&self.cos, &long.cos], 0)?,
            sin: Tensor::cat(&[&self.sin, &long.sin], 0)?,
        })
    }

    /// The tables at positions `pos [B, L]` (u32): `[B, L, d/2]` each.
    fn at(&self, pos: &Tensor) -> Result<Self> {
        let (b, l) = pos.dims2()?;
        let flat = pos.flatten_all()?;
        let take =
            |t: &Tensor| -> Result<Tensor> { t.index_select(&flat, 0)?.reshape((b, l, t.dim(1)?)) };
        Ok(Self {
            sin: take(&self.sin)?,
            cos: take(&self.cos)?,
        })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_embed = hanzo_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = hanzo_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Where a padded pass's tokens sit for the rotary embedding.
#[derive(Clone, Copy)]
enum Place<'a> {
    /// Token `i` at position `i` of the trained tables.
    Index,
    /// Each token at `pos [B, L]` of the trained tables.
    At(&'a Tensor),
    /// Each token at `pos [B, L]` of the trained tables followed by the stretched ones
    /// ([`RotaryEmbedding::join`]).
    Long(&'a Tensor),
}

/// A projection's weights: dense, as trained, or quantized for inference on the CPU.
#[derive(Clone)]
enum Proj {
    Dense(Linear),
    Quant(QMatMul),
}

impl Proj {
    /// The weights as `dtype` blocks when this CPU tiles their matmul; weights already
    /// quantized, or it does not, stay as they are.
    fn quantize(&self, dtype: GgmlDType) -> Result<Proj> {
        match self {
            Proj::Dense(l) if !tiled(dtype, l.weight().dim(0)?, l.weight().dim(1)?) => {
                Ok(self.clone())
            }
            Proj::Dense(l) if l.bias().is_none() => Ok(Proj::Quant(QMatMul::from_qtensor(
                QTensor::quantize(&l.weight().contiguous()?, dtype)?,
            )?)),
            Proj::Dense(_) => hanzo_ml::bail!("a projection with a bias is not quantized"),
            Proj::Quant(_) => Ok(self.clone()),
        }
    }
}

impl Module for Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Dense(l) => l.forward(xs),
            Proj::Quant(q) => q.forward(xs),
        }
    }
}

#[derive(Clone)]
struct ModernBertAttention {
    qkv: Proj,
    proj: Proj,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    /// The trained tables followed by the stretched ones, with [`Config::yarn`].
    long: Option<Arc<RotaryEmbedding>>,
}

impl ModernBertAttention {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        long: Option<Arc<RotaryEmbedding>>,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = Proj::Dense(linear_no_bias(
            config.hidden_size,
            config.hidden_size * 3,
            vb.pp("Wqkv"),
        )?);
        let proj = Proj::Dense(linear_no_bias(
            config.hidden_size,
            config.hidden_size,
            vb.pp("Wo"),
        )?);

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
            long,
        })
    }

    /// The stretched tables: rows `trained..` of [`ModernBertAttention::long`].
    fn stretched(&self) -> Result<Option<(Tensor, Tensor)>> {
        let Some(long) = &self.long else {
            return Ok(None);
        };
        let trained = self.rotary_emb.cos.dim(0)?;
        let n = long.cos.dim(0)? - trained;
        Ok(Some((
            long.cos.narrow(0, trained, n)?,
            long.sin.narrow(0, trained, n)?,
        )))
    }

    /// Packed attention over `qkv [T, 3·H·D]` (see `hanzo_nn::attention::packed`): a sequence
    /// within the trained positions reads the trained tables, a longer one the stretched tables,
    /// each run of consecutive sequences of one kind in one call.
    fn packed(&self, qkv: &Tensor, lens: &[usize], window: Option<usize>) -> Result<Tensor> {
        let (heads, scale) = (
            self.num_attention_heads,
            (self.attention_head_size as f64).powf(-0.5) as f32,
        );
        let trained = self.rotary_emb.cos.dim(0)?;
        let stretched = self.stretched()?;
        let tables = |long: bool| match (long, &stretched) {
            (true, Some((c, s))) => (c, s),
            _ => (&self.rotary_emb.cos, &self.rotary_emb.sin),
        };
        let runs: Vec<&[usize]> = lens
            .chunk_by(|a, b| (*a > trained) == (*b > trained))
            .collect();
        if runs.len() <= 1 {
            let (c, s) = tables(lens.iter().any(|&n| n > trained));
            return hanzo_nn::attention::packed(qkv, lens, heads, window, c, s, scale);
        }
        let mut out = Vec::with_capacity(runs.len());
        let mut at = 0;
        for run in runs {
            let t: usize = run.iter().sum();
            let (c, s) = tables(run[0] > trained);
            let part = qkv.narrow(0, at, t)?;
            out.push(hanzo_nn::attention::packed(
                &part, run, heads, window, c, s, scale,
            )?);
            at += t;
        }
        Tensor::cat(&out, 0)
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        place: Place,
    ) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let (q, k) = match place {
            Place::Index => self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?,
            Place::At(p) => self.rotary_emb.at(p)?.apply_rotary_emb_qkv(&q, &k)?,
            Place::Long(p) => match &self.long {
                Some(long) => long.at(p)?.apply_rotary_emb_qkv(&q, &k)?,
                None => hanzo_ml::bail!("a long place without stretched tables"),
            },
        };

        let scale = (self.attention_head_size as f64).powf(-0.5);
        let q = (q * scale)?;
        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let att = att.broadcast_add(attention_mask)?;
        let xs = softmax_last_dim(&att)?.matmul(&v)?;

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }
}

/// GeGLU: `Wo(gelu(x Wgᵀ) ⊙ x Wuᵀ)`. The checkpoint packs `Wg` and `Wu` as the two row halves of
/// one `Wi`; they are applied as two GEMMs so gelu and the product run on contiguous
/// activations rather than on strided halves of one wide one.
#[derive(Clone)]
pub struct ModernBertMLP {
    gate: Proj,
    up: Proj,
    wo: Proj,
}

impl ModernBertMLP {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let (i, d) = (config.intermediate_size, config.hidden_size);
        let wi = vb.pp("Wi").get_with_hints(
            (2 * i, d),
            "weight",
            hanzo_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self {
            gate: Proj::Dense(Linear::new(wi.narrow(0, 0, i)?, None)),
            up: Proj::Dense(Linear::new(wi.narrow(0, i, i)?, None)),
            wo: Proj::Dense(wo),
        })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        hanzo_nn::ops::geglu(&xs.apply(&self.gate)?, &xs.apply(&self.up)?)?.apply(&self.wo)
    }
}

#[derive(Clone)]
pub struct ModernBertLayer {
    attn: ModernBertAttention,
    mlp: ModernBertMLP,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl ModernBertLayer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        (rotary_emb, long): (Arc<RotaryEmbedding>, Option<Arc<RotaryEmbedding>>),
        uses_local_attention: bool,
    ) -> Result<Self> {
        let attn = ModernBertAttention::load(vb.pp("attn"), config, rotary_emb, long)?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;
        let attn_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();
        let mlp_norm =
            layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    /// A packed pass through the layer: `xs [T, d]` holds sequences of `lens` tokens; a local
    /// layer's queries read the keys within `window` of them.
    fn packed(&self, xs: &Tensor, lens: &[usize], window: usize) -> Result<Tensor> {
        let h = match &self.attn_norm {
            Some(norm) => xs.apply(norm)?,
            None => xs.clone(),
        };
        let a = &self.attn;
        let att = a.packed(
            &h.apply(&a.qkv)?,
            lens,
            self.uses_local_attention.then_some(window),
        )?;
        let xs = (att.apply(&a.proj)? + xs)?;
        let mlp = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        xs + mlp
    }

    /// `local_attention_mask` is the global mask with the sliding window already added; `place`
    /// places the tokens for the rotary embedding.
    fn forward(
        &self,
        xs: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
        place: Place,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let attention_mask = if self.uses_local_attention {
            local_attention_mask
        } else {
            global_attention_mask
        };
        let xs = self.attn.forward(&xs, attention_mask, place)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertHead {
    dense: Linear,
    norm: LayerNorm,
}

impl ModernBertHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let norm = layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;
        Ok(Self { dense, norm })
    }
}

impl Module for ModernBertHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.dense)?.gelu_erf()?.apply(&self.norm)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertDecoder {
    decoder: Linear,
}

impl ModernBertDecoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The decoder weights are tied with the embeddings layer weights
        let decoder_weights = vb.get(
            (config.vocab_size, config.hidden_size),
            "model.embeddings.tok_embeddings.weight",
        )?;
        let decoder_bias = vb.get(config.vocab_size, "decoder.bias")?;
        let decoder = Linear::new(decoder_weights, Some(decoder_bias));
        Ok(Self { decoder })
    }
}

impl Module for ModernBertDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.decoder)?;
        Ok(xs)
    }
}

/// Additive score for a masked key: the most negative finite value of `dtype`.
///
/// Finite, not `-inf`: a padding query whose whole sliding window is padding then attends
/// uniformly instead of producing NaN, and NaN at a padding position still reaches every
/// real position through a later attention's `0 * NaN`. `f32::MIN` does not fit in `F16`
/// (it becomes `-inf`), so each dtype takes its own minimum.
pub fn masked_value(dtype: DType) -> f64 {
    match dtype {
        // `f16::MIN` and `bf16::MIN`: -(2 - 2^-10) * 2^15 and -(2 - 2^-7) * 2^127.
        DType::F16 => -65504.0,
        DType::BF16 => -3.3895313892515355e38,
        _ => f32::MIN as f64,
    }
}

// Global attention mask calculated from padded token inputs
fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?
        .to_dtype(DType::F32)?;

    let inverted_mask = (1.0 - expanded_mask)?;

    (inverted_mask * masked_value(dtype))?.to_dtype(dtype)
}

// Attention mask caused by the sliding window
fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (j as i32 - i as i32).abs() > max_distance as i32 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)?.to_dtype(dtype)
}

// ModernBERT backbone
#[derive(Clone)]
pub struct ModernBert {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
    /// Positions the trained tables hold, `max_position_embeddings`.
    trained: usize,
    /// Whether a longer sequence reads stretched tables ([`Config::yarn`]).
    long: bool,
    /// Projections quantized ([`ModernBert::quantize`]): the CPU inference pass.
    quantized: bool,
}

impl ModernBert {
    /// Load from a `ModernBertFor*` checkpoint, where the backbone sits under `model.`.
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        Self::new(vb.pp("model"), config)
    }

    /// Load a bare backbone whose weights sit at the root of `vb`: `embeddings.*`,
    /// `layers.*`, `final_norm.*`, as a `ModernBertModel` checkpoint stores them.
    pub fn new(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("embeddings.norm"),
        )?;
        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);
        // past the trained positions the global layers read YaRN's tables; a local layer's
        // window spans no more than it was trained on, so it reads its own frequencies, extended
        let long = |trained: &RotaryEmbedding, theta: f64, s: f64| -> Result<_> {
            let n = config.positions();
            let t = RotaryEmbedding::yarn(vb.dtype(), config, theta, s, n, vb.device())?;
            Ok(Arc::new(trained.join(&t)?))
        };
        let (global_long, local_long) = match config.yarn {
            Some(s) => (
                Some(long(&global_rotary_emb, config.global_rope_theta, s)?),
                Some(long(&local_rotary_emb, config.local_rope_theta, 1.0)?),
            ),
            None => (None, None),
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let layer_uses_local_attention = layer_id % config.global_attn_every_n_layers != 0;
            layers.push(ModernBertLayer::load(
                vb.pp(format!("layers.{layer_id}")),
                config,
                if layer_uses_local_attention {
                    (local_rotary_emb.clone(), local_long.clone())
                } else {
                    (global_rotary_emb.clone(), global_long.clone())
                },
                layer_uses_local_attention,
            )?);
        }

        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_norm"),
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
            trained: config.max_position_embeddings,
            long: config.yarn.is_some(),
            quantized: false,
        })
    }

    /// Every layer's projections as `dtype` blocks where this CPU tiles their matmul
    /// ([`tiled`]), for inference on the CPU, which then runs the packed pass: each sequence's
    /// real tokens only, attention per sequence straight from the QKV projection. The rest stays
    /// in the weights' dtype. Training reads dense weights, so a quantized backbone has no
    /// gradient through its projections.
    pub fn quantize(&mut self, dtype: GgmlDType) -> Result<()> {
        for l in &mut self.layers {
            l.attn.qkv = l.attn.qkv.quantize(dtype)?;
            l.attn.proj = l.attn.proj.quantize(dtype)?;
            l.mlp.gate = l.mlp.gate.quantize(dtype)?;
            l.mlp.up = l.mlp.up.quantize(dtype)?;
            l.mlp.wo = l.mlp.wo.quantize(dtype)?;
        }
        self.quantized = true;
        Ok(())
    }

    /// Whether [`ModernBert::forward`] runs the packed pass.
    pub fn packs(&self) -> bool {
        let w = self.word_embeddings.embeddings();
        let head = self
            .layers
            .first()
            .map_or(0, |l| l.attn.attention_head_size);
        match w.device() {
            Device::Cpu => self.quantized,
            dev => hanzo_nn::attention::packs(dev, w.dtype(), head),
        }
    }

    /// Token states `[B, L, d]` of right-padded `xs [B, L]`; the packed pass leaves pads zero.
    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        if self.packs() {
            let rows = mask.to_dtype(DType::U32)?.to_vec2::<u32>()?;
            let lens: Vec<usize> = rows
                .iter()
                .map(|r| r.iter().take_while(|&&m| m != 0).count())
                .collect();
            let right = rows
                .iter()
                .zip(&lens)
                .all(|(r, &n)| r[n..].iter().all(|&m| m == 0));
            if right {
                return self.packed(xs, &lens);
            }
        }
        self.padded(xs, mask)
    }

    /// Token states `[B, L, d]` of `xs [B, L]` at positions `pos [B, L]` (u32), query `i` reading
    /// key `j` where `allow [B, L, L]` is nonzero; a local layer reads, of those, the keys whose
    /// position is within its window of the query's. The padded pass: tokens at the same
    /// positions that read the same keys come out alike wherever they sit in the sequence.
    /// `allow` lets no query read a padding key.
    pub fn forward_masked(&self, xs: &Tensor, pos: &Tensor, allow: &Tensor) -> Result<Tensor> {
        let dtype = self.word_embeddings.embeddings().dtype();
        let allow = allow.to_dtype(DType::F32)?;
        let p = pos.to_dtype(DType::F32)?;
        let near = p
            .unsqueeze(2)?
            .broadcast_sub(&p.unsqueeze(1)?)?
            .abs()?
            .le((self.local_attention_size / 2) as f64)?
            .to_dtype(DType::F32)?;
        let additive = |a: &Tensor| -> Result<Tensor> {
            ((1.0 - a)? * masked_value(dtype))?
                .to_dtype(dtype)?
                .unsqueeze(1)
        };
        let global = additive(&allow)?;
        let local = additive(&(&allow * near)?)?;
        // a row placing a key some query reads past the trained positions reads the stretched
        // tables
        let long = match self.long {
            true => {
                let read = allow.max(1)?;
                let last = (p * read)?.max(1)?.to_vec1::<f32>()?;
                self.shifted(pos, last.iter().map(|&m| m as usize >= self.trained))?
            }
            false => None,
        };
        let place = match &long {
            Some(p) => Place::Long(p),
            None => Place::At(pos),
        };
        let mut h = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in &self.layers {
            h = layer.forward(&h, &global, &local, place)?;
        }
        h.apply(&self.final_norm)
    }

    /// `pos [B, L]` into the trained tables followed by the stretched ones
    /// ([`RotaryEmbedding::join`]): each long row's positions past the trained ones. `None` when
    /// no row is long.
    fn shifted(&self, pos: &Tensor, long: impl Iterator<Item = bool>) -> Result<Option<Tensor>> {
        let shift: Vec<u32> = long
            .map(|l| if l { self.trained as u32 } else { 0 })
            .collect();
        if shift.iter().all(|&s| s == 0) {
            return Ok(None);
        }
        let n = shift.len();
        let shift = Tensor::from_vec(shift, (n, 1), pos.device())?;
        Ok(Some(pos.broadcast_add(&shift)?.contiguous()?))
    }

    /// Every sequence's first `lens` tokens packed into one: embeddings, projections and norms
    /// over the real tokens only, attention per sequence.
    fn packed(&self, xs: &Tensor, lens: &[usize]) -> Result<Tensor> {
        let (b, l) = xs.dims2()?;
        let ids = xs.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let flat: Vec<u32> = ids
            .iter()
            .zip(lens)
            .flat_map(|(r, &n)| r[..n].iter().copied())
            .collect();
        let t = flat.len();
        let mut h = Tensor::from_vec(flat, t, xs.device())?
            .apply(&self.word_embeddings)?
            .apply(&self.norm)?;
        let window = self.local_attention_size / 2;
        for layer in &self.layers {
            h = layer.packed(&h, lens, window)?;
        }
        let h = h.apply(&self.final_norm)?;
        let d = h.dim(1)?;
        let h = Tensor::cat(&[&h, &Tensor::zeros((1, d), h.dtype(), h.device())?], 0)?;
        let mut at = Vec::with_capacity(b * l);
        let mut back = Vec::with_capacity(t + 1);
        let mut next = 0u32;
        for (row, &n) in lens.iter().enumerate() {
            at.extend(next..next + n as u32);
            at.extend(std::iter::repeat_n(t as u32, l - n));
            back.extend((row * l) as u32..(row * l + n) as u32);
            next += n as u32;
        }
        back.push((b * l) as u32);
        let at = Tensor::from_vec(at, b * l, h.device())?;
        let back = Tensor::from_vec(back, t + 1, h.device())?;
        hanzo_nn::ops::select(&h, &at, &back)?.reshape((b, l, d))
    }

    fn padded(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let seq_len = xs.shape().dims()[1];
        let dtype = self.word_embeddings.embeddings().dtype();
        let global_attention_mask =
            prepare_4d_attention_mask(mask, dtype, None)?.to_device(xs.device())?;
        let local_attention_mask =
            get_local_attention_mask(seq_len, self.local_attention_size / 2, dtype, xs.device())?
                .broadcast_add(&global_attention_mask)?;
        // a row longer than the trained positions reads the stretched tables
        let long = match self.long && seq_len > self.trained {
            true => {
                let lens = mask.to_dtype(DType::U32)?.sum(1)?.to_vec1::<u32>()?;
                let pos = Tensor::arange(0u32, seq_len as u32, xs.device())?
                    .unsqueeze(0)?
                    .broadcast_as((lens.len(), seq_len))?;
                self.shifted(&pos, lens.iter().map(|&n| n as usize > self.trained))?
            }
            false => None,
        };
        let place = match &long {
            Some(p) => Place::Long(p),
            None => Place::Index,
        };
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &global_attention_mask, &local_attention_mask, place)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }
}

// ModernBERT for the fill-mask task
#[derive(Clone)]
pub struct ModernBertForMaskedLM {
    model: ModernBert,
    decoder: ModernBertDecoder,
    head: ModernBertHead,
}

impl ModernBertForMaskedLM {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        let decoder = ModernBertDecoder::load(vb.clone(), config)?;
        let head = ModernBertHead::load(vb.pp("head"), config)?;
        Ok(Self {
            model,
            decoder,
            head,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let xs = self
            .model
            .forward(xs, mask)?
            .apply(&self.head)?
            .apply(&self.decoder)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertClassifier {
    classifier: Linear,
}

impl ModernBertClassifier {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The decoder weights are tied with the embeddings layer weights
        let classifier = linear(
            config.hidden_size,
            config
                .classifier_config
                .as_ref()
                .map(|cc| cc.id2label.len())
                .unwrap_or_default(),
            vb.pp("classifier"),
        )?;
        Ok(Self { classifier })
    }
}

impl Module for ModernBertClassifier {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.classifier)?;
        softmax(&xs, D::Minus1)
    }
}

#[derive(Clone)]
pub struct ModernBertForSequenceClassification {
    model: ModernBert,
    head: ModernBertHead,
    classifier: ModernBertClassifier,
    classifier_pooling: ClassifierPooling,
}

impl ModernBertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        let classifier = ModernBertClassifier::load(vb.clone(), config)?;
        let head = ModernBertHead::load(vb.pp("head"), config)?;
        Ok(Self {
            model,
            head,
            classifier,
            classifier_pooling: config
                .classifier_config
                .as_ref()
                .map(|cc| cc.classifier_pooling)
                .unwrap_or_default(),
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let output = self.model.forward(xs, mask)?;
        let last_hidden_state = match self.classifier_pooling {
            ClassifierPooling::CLS => output.i((.., 0, ..))?.contiguous()?,
            ClassifierPooling::MEAN => {
                let unsqueezed_mask = &mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
                let sum_output = output.broadcast_mul(unsqueezed_mask)?.sum(1)?;
                sum_output.broadcast_div(&mask.sum_keepdim(1)?.to_dtype(DType::F32)?)?
            }
        };
        let xs = self
            .head
            .forward(&last_hidden_state)?
            .apply(&self.classifier)?;
        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// mmBERT's attention geometry: 64-wide heads, θ 160000, 8,192 trained positions.
    fn mmbert(yarn: Option<f64>) -> Config {
        Config {
            vocab_size: 8,
            hidden_size: 768,
            num_hidden_layers: 1,
            num_attention_heads: 12,
            intermediate_size: 8,
            max_position_embeddings: 8192,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160000.,
            local_attention: 128,
            local_rope_theta: 160000.,
            yarn,
            classifier_config: None,
        }
    }

    /// Each frequency of the tables at position 1, the attention factor taken out.
    fn freqs(t: &RotaryEmbedding, mul: f64) -> Vec<f64> {
        let sin = t.sin.get(1).unwrap().to_vec1::<f32>().unwrap();
        sin.iter().map(|&s| (s as f64 / mul).asin()).collect()
    }

    #[test]
    fn positions_are_the_trained_ones_stretched() {
        assert_eq!(mmbert(None).positions(), 8192);
        assert_eq!(mmbert(Some(4.0)).positions(), 32768);
    }

    /// For mmBERT at s = 4 the band runs from index 9 (wavelength 8192 / 32) to 20 (8192): below
    /// it a frequency is kept, above it divided by 4; cos at position 0 is the attention factor.
    #[test]
    fn yarn_keeps_the_fast_frequencies_and_divides_the_slow() {
        let cfg = mmbert(Some(4.0));
        let mul = 0.1 * 4f64.ln() + 1.0;
        let t = RotaryEmbedding::yarn(DType::F32, &cfg, 160000., 4.0, 32768, &Device::Cpu).unwrap();
        assert_eq!(t.cos.dims(), &[32768, 32]);
        let c0 = t.cos.get(0).unwrap().to_vec1::<f32>().unwrap();
        assert!(c0.iter().all(|&c| (c as f64 - mul).abs() < 1e-6), "{c0:?}");
        let got = freqs(&t, mul);
        for (i, g) in got.iter().enumerate() {
            let f = 1.0 / 160000f64.powf(2.0 * i as f64 / 64.0);
            let r = g / f;
            match i {
                0..=9 => assert!((r - 1.0).abs() < 1e-4, "{i}: {r}"),
                20.. => assert!((r - 0.25).abs() < 1e-4, "{i}: {r}"),
                _ => assert!(r < 1.0 && r > 0.25, "{i}: {r}"),
            }
        }
        // linear across the band: index 10 is 1/11 of the way from 1 to 1/4
        let r: Vec<f64> = (0..32)
            .map(|i| got[i] * 160000f64.powf(2.0 * i as f64 / 64.0))
            .collect();
        for (i, x) in r.iter().enumerate().take(20).skip(10) {
            let want = 1.0 - 0.75 * (i as f64 - 9.0) / 11.0;
            assert!((x - want).abs() < 1e-6, "{i}: {x} against {want}");
        }
    }

    /// At s = 1 the tables are the trained ones over more positions: unscaled, every frequency
    /// kept, equal to the trained tables where both have a row.
    #[test]
    fn yarn_at_one_is_the_trained_tables_extended() {
        let cfg = mmbert(Some(4.0));
        let dev = Device::Cpu;
        let t = RotaryEmbedding::yarn(DType::F32, &cfg, 160000., 1.0, 32768, &dev).unwrap();
        let trained = RotaryEmbedding::new(DType::F32, &cfg, 160000., &dev).unwrap();
        let (a, b) = (
            t.cos.narrow(0, 0, 1024).unwrap(),
            trained.cos.narrow(0, 0, 1024).unwrap(),
        );
        let gap = (a - b).unwrap().abs().unwrap().max_all().unwrap();
        assert!(gap.to_scalar::<f32>().unwrap() < 1e-4);
        let got = freqs(&t, 1.0);
        for (i, g) in got.iter().enumerate() {
            let f = 1.0 / 160000f64.powf(2.0 * i as f64 / 64.0);
            assert!((g / f - 1.0).abs() < 1e-4, "{i}");
        }
    }
}
