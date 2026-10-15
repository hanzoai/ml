//! UltraBERT
//!
//! UltraBERT is a modernized bidirectional encoder-only Transformer model.
//! - [Arxiv](https://arxiv.org/abs/2610.15028) "UltraBERT: Beyond the Limit of BERT Pre-Training"
//! - Upstream [HuggingFace repo](https://huggingface.co/dmis-lab/UltraBERT-380M).
//!

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{
    embedding, linear_no_bias, ops::softmax, rms_norm, Embedding, Linear, Module, RmsNorm,
    VarBuilder,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

pub const DEFAULT_MAX_SEQ_LEN: usize = 32768;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub pad_token_id: u32,
    pub attention_windows: Vec<Option<usize>>,
    pub rope_theta: Vec<f64>,
    #[serde(default)]
    pub prediction_head_size: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings.unwrap_or(DEFAULT_MAX_SEQ_LEN)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            cos: freqs.cos()?,
            sin: freqs.sin()?,
        })
    }

    pub fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = hanzo_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = hanzo_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
            .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

#[derive(Clone)]
pub struct UltrabertAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl UltrabertAttention {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads();
        let head_dim = config.head_dim;

        let q_proj = linear_no_bias(
            config.hidden_size,
            num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            config.hidden_size,
            num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary_emb,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;
        let q = hidden_states.apply(&self.q_proj)?;
        let k = hidden_states.apply(&self.k_proj)?;
        let v = hidden_states.apply(&self.v_proj)?;

        let q = q
            .reshape((b, seq_len, self.num_attention_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        let k = if self.num_key_value_heads < self.num_attention_heads {
            let n_rep = self.num_attention_heads / self.num_key_value_heads;
            repeat_kv(k, n_rep)?
        } else {
            k
        };
        let v = if self.num_key_value_heads < self.num_attention_heads {
            let n_rep = self.num_attention_heads / self.num_key_value_heads;
            repeat_kv(v, n_rep)?
        } else {
            v
        };

        let scale = (self.head_dim as f64).powf(-0.5);
        let q = (q * scale)?;

        let mut att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        if let Some(mask) = attention_mask {
            att = att.broadcast_add(mask)?;
        }
        let att = softmax(&att, D::Minus1)?;
        let xs = att.matmul(&v)?;

        let xs = xs
            .permute((0, 2, 1, 3))?
            .reshape((b, seq_len, self.num_attention_heads * self.head_dim))?;
        xs.apply(&self.o_proj)
    }
}

#[derive(Clone)]
pub struct UltrabertMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl UltrabertMLP {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for UltrabertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = hanzo_nn::ops::silu(&xs.apply(&self.gate_proj)?)?;
        let up = xs.apply(&self.up_proj)?;
        let intermediate = (gate * up)?;
        intermediate.apply(&self.down_proj)
    }
}

#[derive(Clone)]
pub struct UltrabertLayer {
    self_attn: UltrabertAttention,
    mlp: UltrabertMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pub uses_local_attention: bool,
}

impl UltrabertLayer {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
    ) -> Result<Self> {
        let self_attn = UltrabertAttention::load(vb.pp("self_attn"), config, rotary_emb)?;
        let mlp = UltrabertMLP::load(vb.pp("mlp"), config)?;
        let input_layernorm =
            rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            uses_local_attention,
        })
    }

    pub fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let normed = xs.apply(&self.input_layernorm)?;
        let attn_out = self.self_attn.forward(&normed, attention_mask)?;
        let xs = (residual + attn_out)?;

        let residual = xs.clone();
        let normed = xs.apply(&self.post_attention_layernorm)?;
        let mlp_out = self.mlp.forward(&normed)?;
        let xs = (residual + mlp_out)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct UltrabertModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<UltrabertLayer>,
    pub norm: RmsNorm,
    pub config: Config,
}

impl UltrabertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embed_tokens = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let max_seq_len = config.max_position_embeddings();
        let mut rotary_embeddings: HashMap<u64, Arc<RotaryEmbedding>> = HashMap::new();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("layers");

        for i in 0..config.num_hidden_layers {
            let theta = config
                .rope_theta
                .get(i)
                .copied()
                .unwrap_or(10000.0);
            let theta_key = (theta * 1000.0) as u64;

            let rotary_emb = if let Some(emb) = rotary_embeddings.get(&theta_key) {
                emb.clone()
            } else {
                let emb = Arc::new(RotaryEmbedding::new(
                    vb.dtype(),
                    config.head_dim,
                    max_seq_len,
                    theta,
                    vb.device(),
                )?);
                rotary_embeddings.insert(theta_key, emb.clone());
                emb
            };

            let uses_local = config
                .attention_windows
                .get(i)
                .map(|w| w.is_some())
                .unwrap_or(false);

            let layer = UltrabertLayer::load(
                vb_layers.pp(i),
                config,
                rotary_emb,
                uses_local,
            )?;
            layers.push(layer);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let global_mask = match attention_mask {
            Some(mask) => Some(prepare_4d_attention_mask(mask, xs.dtype(), None)?),
            None => None,
        };

        // Precompute local sliding-window attention mask if any local layer exists
        let has_local_layers = self.layers.iter().any(|l| l.uses_local_attention);
        let local_mask = if has_local_layers {
            Some(get_local_attention_mask(seq_len, 128, xs.device())?)
        } else {
            None
        };

        for layer in &self.layers {
            let mask = if layer.uses_local_attention {
                match (&global_mask, &local_mask) {
                    (Some(g), Some(l)) => Some(g.broadcast_add(l)?),
                    (None, Some(l)) => Some(l.clone()),
                    (Some(g), None) => Some(g.clone()),
                    (None, None) => None,
                }
            } else {
                global_mask.clone()
            };

            xs = layer.forward(&xs, mask.as_ref())?;
        }

        xs.apply(&self.norm)
    }
}

#[derive(Clone)]
pub struct UltrabertPredictionHead {
    linear1: Linear,
    linear2: Linear,
}

impl UltrabertPredictionHead {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Option<Self>> {
        match config.prediction_head_size {
            Some(dim) => {
                let linear1 = linear_no_bias(config.hidden_size, dim, vb.pp("linear1"))?;
                let linear2 = linear_no_bias(dim, config.hidden_size, vb.pp("linear2"))?;
                Ok(Some(Self { linear1, linear2 }))
            }
            None => Ok(None),
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.linear1)?.gelu()?;
        let xs = xs.apply(&self.linear2)?.gelu()?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct UltrabertForMaskedLM {
    pub ultrabert: UltrabertModel,
    pub head: Option<UltrabertPredictionHead>,
    pub cls: Linear,
}

impl UltrabertForMaskedLM {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let ultrabert = UltrabertModel::load(vb.pp("ultrabert"), config)?;
        let head = UltrabertPredictionHead::load(vb.pp("head"), config)?;
        let cls = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("cls"))?;
        Ok(Self {
            ultrabert,
            head,
            cls,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let hidden_states = self.ultrabert.forward(input_ids, attention_mask)?;
        let hidden_states = match &self.head {
            Some(head) => head.forward(&hidden_states)?,
            None => hidden_states,
        };
        hidden_states.apply(&self.cls)
    }
}

pub fn prepare_4d_attention_mask(
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
        .to_dtype(dtype)?;

    let inverted_mask = (1.0 - expanded_mask)?;
    (inverted_mask * (f32::NEG_INFINITY as f64))?.to_dtype(dtype)
}

pub fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
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
    Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)
}
