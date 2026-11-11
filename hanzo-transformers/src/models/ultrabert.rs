//! UltraBERT
//!
//! UltraBERT is a modernized bidirectional encoder-only Transformer model.
//! - [Arxiv](https://arxiv.org/abs/2610.15028) "UltraBERT: Beyond the Limit of BERT Pre-Training"
//! - Upstream [HuggingFace repo](https://huggingface.co/dmis-lab/UltraBERT-380M).
//!

use hanzo_ml::quantized::{repack::tiled, GgmlDType, QMatMul, QTensor};
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{
    embedding, linear_no_bias, ops::softmax, ops::softmax_last_dim, rms_norm, Embedding, Linear,
    Module, RmsNorm, VarBuilder,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

pub const DEFAULT_MAX_SEQ_LEN: usize = 32768;
/// Qualified minimum sequence length before Bidirectional D-LSA engages (`< 4096` stays dense).
pub const DEFAULT_DLSA_MIN_SEQ_LEN: usize = 4096;
/// Qualified Bidirectional D-LSA key budget on long-context UltraBERT (`>= 4096` tokens).
pub const DEFAULT_DLSA_K_BUDGET: usize = 512;

/// Maximum attention logits per tile on padded/GPU global pass (`8M` elements = `32 MiB` in F32).
const LOGITS: usize = 1 << 23;
/// Queries per local sliding-window tile on padded/GPU pass.
const LOCAL_TILE: usize = 512;

#[derive(Clone, Debug)]
enum Proj {
    Dense(Linear),
    Quant(QMatMul),
}

impl Proj {
    fn quantize(&self, dtype: GgmlDType) -> Result<Self> {
        match self {
            Self::Dense(l) if !tiled(dtype, l.weight().dim(0)?, l.weight().dim(1)?) => {
                Ok(self.clone())
            }
            Self::Dense(l) if l.bias().is_none() => Ok(Self::Quant(QMatMul::from_qtensor(
                QTensor::quantize(&l.weight().contiguous()?, dtype)?,
            )?)),
            other => Ok(other.clone()),
        }
    }

    fn device(&self) -> Device {
        match self {
            Self::Dense(l) => l.weight().device().clone(),
            Self::Quant(_) => Device::Cpu,
        }
    }

    fn dtype(&self) -> DType {
        match self {
            Self::Dense(l) => l.weight().dtype(),
            Self::Quant(_) => DType::F32,
        }
    }
}

impl Module for Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(l) => l.forward(xs),
            Self::Quant(q) => q.forward(xs),
        }
    }
}

#[derive(Clone, Debug)]
pub struct UltrabertDlsaIndexer {
    w_q_idx: Proj,
    w_k_idx: Proj,
    pub d_idx: usize,
    pub k_budget: usize,
    pub min_seq_len: usize,
    pub sink_tokens: usize,
    pub local_window: usize,
}

impl UltrabertDlsaIndexer {
    pub fn new_deterministic(
        hidden_size: usize,
        d_idx: usize,
        k_budget: usize,
        sink_tokens: usize,
        local_window: usize,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let mut wq = vec![0f32; d_idx * hidden_size];
        let mut wk = vec![0f32; d_idx * hidden_size];
        let scale = (1.0 / (hidden_size as f32).sqrt()) * 0.5;
        for r in 0..d_idx {
            for c in 0..hidden_size {
                let idx = r * hidden_size + c;
                let s1 = ((idx as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407)
                    >> 33) as f32
                    / (u32::MAX as f32);
                let s2 = ((idx as u64)
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(3037000493)
                    >> 33) as f32
                    / (u32::MAX as f32);
                wq[idx] = (s1 * 2.0 - 1.0) * scale;
                wk[idx] = (s2 * 2.0 - 1.0) * scale;
            }
        }
        let w_q_t = Tensor::from_vec(wq, (d_idx, hidden_size), dev)?.to_dtype(dtype)?;
        let w_k_t = Tensor::from_vec(wk, (d_idx, hidden_size), dev)?.to_dtype(dtype)?;
        Ok(Self {
            w_q_idx: Proj::Dense(Linear::new(w_q_t, None)),
            w_k_idx: Proj::Dense(Linear::new(w_k_t, None)),
            d_idx,
            k_budget,
            min_seq_len: DEFAULT_DLSA_MIN_SEQ_LEN,
            sink_tokens,
            local_window,
        })
    }

    /// Returns `true` only when `seq_len` meets the long-context threshold (`min_seq_len`)
    /// and exceeds the sparse key budget (`k_budget`). Short contexts run exact dense attention.
    #[inline]
    pub fn should_apply(&self, seq_len: usize) -> bool {
        seq_len >= self.min_seq_len && seq_len > self.k_budget
    }
}

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
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let angles: Vec<f32> = (0..max_seq_len)
            .flat_map(|p| inv_freq.iter().map(move |f| p as f32 * f))
            .collect();
        let freqs = Tensor::from_vec(angles, (max_seq_len, inv_freq.len()), dev)?;
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
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

    pub fn tables(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        Ok((
            self.cos.narrow(0, 0, seq_len)?,
            self.sin.narrow(0, 0, seq_len)?,
        ))
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

/// Precomputed local/global tile plan for padded/GPU execution of UltraBERT.
#[derive(Clone)]
struct UltrabertTiles {
    keys: Option<Tensor>,
    global_step: usize,
    local: Vec<UltrabertLocalTile>,
}

#[derive(Clone)]
struct UltrabertLocalTile {
    at: usize,
    t: usize,
    lo: usize,
    n: usize,
    window: Tensor,
}

impl UltrabertTiles {
    fn new(
        b: usize,
        l: usize,
        heads: usize,
        window: usize,
        mask_4d: Option<&Tensor>,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self> {
        let local_step = LOCAL_TILE.min((LOGITS / (b * heads * (LOCAL_TILE + 2 * window))).max(1));
        let mut tiles: Vec<UltrabertLocalTile> = Vec::new();
        for at in (0..l).step_by(local_step) {
            let t = local_step.min(l - at);
            let lo = at.saturating_sub(window);
            let n = (at + t + window).min(l) - lo;
            let alike = tiles
                .iter()
                .find(|x| (x.at - x.lo, x.t, x.n) == (at - lo, t, n));
            let win = match alike {
                Some(x) => x.window.clone(),
                None => {
                    let far: Vec<f32> = (0..t)
                        .flat_map(|i| {
                            (0..n).map(move |j| {
                                if (lo + j).abs_diff(at + i) > window {
                                    f32::NEG_INFINITY
                                } else {
                                    0.0
                                }
                            })
                        })
                        .collect();
                    Tensor::from_vec(far, (1, 1, t, n), dev)?.to_dtype(dtype)?
                }
            };
            tiles.push(UltrabertLocalTile {
                at,
                t,
                lo,
                n,
                window: win,
            });
        }
        if let Some(m) = mask_4d {
            for x in &mut tiles {
                let k_slice = m.narrow(3, x.lo, x.n)?;
                x.window = x.window.broadcast_add(&k_slice)?;
            }
        }
        Ok(Self {
            keys: mask_4d.cloned(),
            global_step: (LOGITS / (b * heads * l)).clamp(1, l),
            local: tiles,
        })
    }

    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, local: bool) -> Result<Tensor> {
        let out = if local {
            self.local
                .iter()
                .map(|x| {
                    let q_t = q.narrow(2, x.at, x.t)?;
                    let k_t = k.narrow(2, x.lo, x.n)?.t()?.contiguous()?;
                    let v_t = v.narrow(2, x.lo, x.n)?.contiguous()?;
                    let scores = q_t.matmul(&k_t)?.broadcast_add(&x.window)?;
                    let p = softmax_last_dim(&scores)?;
                    p.matmul(&v_t)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            let (l, kt, vc) = (q.dim(2)?, k.t()?.contiguous()?, v.contiguous()?);
            (0..l)
                .step_by(self.global_step)
                .map(|at| {
                    let t = self.global_step.min(l - at);
                    let mut scores = q.narrow(2, at, t)?.matmul(&kt)?;
                    if let Some(keys) = &self.keys {
                        scores = scores.broadcast_add(keys)?;
                    }
                    let p = softmax_last_dim(&scores)?;
                    p.matmul(&vc)
                })
                .collect::<Result<Vec<_>>>()?
        };
        Tensor::cat(&out, 2)
    }
}

#[derive(Clone)]
pub struct UltrabertAttention {
    wqkv: Proj,
    o_proj: Proj,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    pub dlsa: Option<UltrabertDlsaIndexer>,
}

impl UltrabertAttention {
    pub fn load(vb: VarBuilder, config: &Config, rotary_emb: Arc<RotaryEmbedding>) -> Result<Self> {
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

        let wqkv_weight = Tensor::cat(&[q_proj.weight(), k_proj.weight(), v_proj.weight()], 0)?;
        let wqkv = Proj::Dense(Linear::new(wqkv_weight, None));

        Ok(Self {
            wqkv,
            o_proj: Proj::Dense(o_proj),
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_size: config.hidden_size,
            rotary_emb,
            dlsa: None,
        })
    }

    pub fn set_dlsa(
        &mut self,
        k_budget: usize,
        d_idx: usize,
        sink_tokens: usize,
        local_window: usize,
    ) -> Result<()> {
        let dev = self.wqkv.device();
        let dtype = self.wqkv.dtype();
        self.dlsa = Some(UltrabertDlsaIndexer::new_deterministic(
            self.hidden_size,
            d_idx,
            k_budget,
            sink_tokens,
            local_window,
            dtype,
            &dev,
        )?);
        Ok(())
    }

    /// Packed CPU execution over `[T, hidden_size]` unpadded sequences of lengths `lens`.
    fn packed(
        &self,
        h_in: &Tensor,
        qkv: &Tensor,
        lens: &[usize],
        window: Option<usize>,
    ) -> Result<Tensor> {
        let max_l = lens.iter().copied().max().unwrap_or(0);
        let (cos, sin) = self.rotary_emb.tables(max_l)?;
        let scale = (self.head_dim as f64).powf(-0.5) as f32;
        if window.is_none() {
            if let Some(dlsa) = &self.dlsa {
                if dlsa.should_apply(max_l) {
                    let q_idx = h_in.apply(&dlsa.w_q_idx)?;
                    let k_idx = h_in.apply(&dlsa.w_k_idx)?;
                    return hanzo_nn::attention::packed_dlsa(
                        qkv,
                        &q_idx,
                        &k_idx,
                        lens,
                        self.num_attention_heads,
                        &cos,
                        &sin,
                        scale,
                        dlsa.k_budget,
                        dlsa.sink_tokens,
                        dlsa.local_window,
                    );
                }
            }
        }
        hanzo_nn::attention::packed(
            qkv,
            lens,
            self.num_attention_heads,
            window,
            &cos,
            &sin,
            scale,
        )
    }

    fn forward_tiled(
        &self,
        hidden_states: &Tensor,
        tiles: &UltrabertTiles,
        local: bool,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;
        let q_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.num_key_value_heads * self.head_dim;

        let qkv = hidden_states.apply(&self.wqkv)?;
        let q = qkv
            .narrow(2, 0, q_dim)?
            .reshape((b, seq_len, self.num_attention_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = qkv
            .narrow(2, q_dim, kv_dim)?
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = qkv
            .narrow(2, q_dim + kv_dim, kv_dim)?
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

        let xs = if !local {
            if let Some(dlsa) = &self.dlsa {
                if dlsa.should_apply(seq_len) {
                    let q_idx = hidden_states.apply(&dlsa.w_q_idx)?;
                    let k_idx = hidden_states.apply(&dlsa.w_k_idx)?;
                    let attn_out = hanzo_nn::attention::dlsa_batched_gpu(
                        &q,
                        &k,
                        &v,
                        &q_idx,
                        &k_idx,
                        dlsa.k_budget,
                        dlsa.sink_tokens,
                        dlsa.local_window,
                    )?;
                    return attn_out
                        .permute((0, 2, 1, 3))?
                        .reshape((b, seq_len, q_dim))?
                        .apply(&self.o_proj);
                }
            }
            tiles.attend(&q, &k, &v, false)?
        } else {
            tiles.attend(&q, &k, &v, true)?
        };

        let xs = xs.permute((0, 2, 1, 3))?.reshape((b, seq_len, q_dim))?;
        xs.apply(&self.o_proj)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;
        let q_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.num_key_value_heads * self.head_dim;

        let qkv = hidden_states.apply(&self.wqkv)?;
        let q = qkv
            .narrow(2, 0, q_dim)?
            .reshape((b, seq_len, self.num_attention_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = qkv
            .narrow(2, q_dim, kv_dim)?
            .reshape((b, seq_len, self.num_key_value_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = qkv
            .narrow(2, q_dim + kv_dim, kv_dim)?
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

        let xs = xs.permute((0, 2, 1, 3))?.reshape((
            b,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ))?;
        xs.apply(&self.o_proj)
    }
}

#[derive(Clone)]
pub struct UltrabertMLP {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
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
            gate_proj: Proj::Dense(gate_proj),
            up_proj: Proj::Dense(up_proj),
            down_proj: Proj::Dense(down_proj),
        })
    }
}

impl Module for UltrabertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        hanzo_nn::ops::silu_mul(&xs.apply(&self.gate_proj)?, &xs.apply(&self.up_proj)?)?
            .apply(&self.down_proj)
    }
}

#[derive(Clone)]
pub struct UltrabertLayer {
    pub self_attn: UltrabertAttention,
    mlp: UltrabertMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pub uses_local_attention: bool,
    pub window: Option<usize>,
}

impl UltrabertLayer {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        window: Option<usize>,
    ) -> Result<Self> {
        let self_attn = UltrabertAttention::load(vb.pp("self_attn"), config, rotary_emb)?;
        let mlp = UltrabertMLP::load(vb.pp("mlp"), config)?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
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
            uses_local_attention: window.is_some(),
            window,
        })
    }

    fn packed(&self, xs: &Tensor, lens: &[usize]) -> Result<Tensor> {
        let h = xs.apply(&self.input_layernorm)?;
        let a = &self.self_attn;
        let att = a.packed(&h, &h.apply(&a.wqkv)?, lens, self.window)?;
        let xs = (att.apply(&a.o_proj)? + xs)?;
        let mlp = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        xs + mlp
    }

    fn forward_tiled(&self, xs: &Tensor, tiles: &UltrabertTiles) -> Result<Tensor> {
        let residual = xs.clone();
        let normed = xs.apply(&self.input_layernorm)?;
        let attn_out = self
            .self_attn
            .forward_tiled(&normed, tiles, self.uses_local_attention)?;
        let xs = (residual + attn_out)?;

        let residual = xs.clone();
        let normed = xs.apply(&self.post_attention_layernorm)?;
        let mlp_out = self.mlp.forward(&normed)?;
        residual + mlp_out
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
    pub packed_cpu: bool,
}

impl UltrabertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let max_seq_len = config.max_position_embeddings();
        let mut rotary_embeddings: HashMap<u64, Arc<RotaryEmbedding>> = HashMap::new();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let vb_layers = vb.pp("layers");

        for i in 0..config.num_hidden_layers {
            let theta = config.rope_theta.get(i).copied().unwrap_or(10000.0);
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

            let window = config.attention_windows.get(i).copied().flatten();

            let layer = UltrabertLayer::load(vb_layers.pp(i), config, rotary_emb, window)?;
            layers.push(layer);
        }

        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            config: config.clone(),
            packed_cpu: false,
        })
    }

    /// Enable the packed CPU execution pass (`[T, d]` unpadded sequences + `gemm::gemm` block-packed
    /// local sliding-window attention and D-LSA) without quantizing linear projections.
    pub fn pack(&mut self) {
        self.packed_cpu = true;
    }

    /// Quantize all linear projections (`wqkv`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
    /// to `dtype` (e.g., `GgmlDType::Q8_0`) and enable the packed CPU execution pass.
    pub fn quantize(&mut self, dtype: GgmlDType) -> Result<()> {
        for l in &mut self.layers {
            l.self_attn.wqkv = l.self_attn.wqkv.quantize(dtype)?;
            l.self_attn.o_proj = l.self_attn.o_proj.quantize(dtype)?;
            l.mlp.gate_proj = l.mlp.gate_proj.quantize(dtype)?;
            l.mlp.up_proj = l.mlp.up_proj.quantize(dtype)?;
            l.mlp.down_proj = l.mlp.down_proj.quantize(dtype)?;
        }
        self.packed_cpu = true;
        Ok(())
    }

    /// Attach Bidirectional D-LSA indexers to the global attention layers of UltraBERT.
    pub fn with_dlsa(&mut self, k_budget: usize, d_idx: usize, sink_tokens: usize) -> Result<()> {
        let default_window = self.layers.iter().find_map(|l| l.window).unwrap_or(128);
        for l in &mut self.layers {
            if !l.uses_local_attention {
                l.self_attn
                    .set_dlsa(k_budget, d_idx, sink_tokens, default_window)?;
            }
        }
        Ok(())
    }

    /// Dynamically adjust or disable (`None`) the D-LSA `k_budget` on global layers.
    pub fn set_dlsa_k_budget(&mut self, k_budget: Option<usize>) {
        for l in &mut self.layers {
            if let Some(dlsa) = &mut l.self_attn.dlsa {
                dlsa.k_budget = k_budget.unwrap_or(usize::MAX);
            }
        }
    }

    /// Set the minimum sequence length threshold before D-LSA activates (`< min_seq_len` stays dense).
    pub fn set_dlsa_min_seq_len(&mut self, min_seq_len: usize) {
        for l in &mut self.layers {
            if let Some(dlsa) = &mut l.self_attn.dlsa {
                dlsa.min_seq_len = min_seq_len;
            }
        }
    }

    /// Whether [`UltrabertModel::forward`] runs the packed CPU pass.
    pub fn packs(&self) -> bool {
        let w = self.embed_tokens.embeddings();
        self.packed_cpu
            && w.device().is_cpu()
            && w.dtype() == DType::F32
            && self.config.num_key_value_heads() == self.config.num_attention_heads
    }

    /// The residual stream entering layer `at` of the pass [`UltrabertModel::forward`] runs.
    pub fn stream(
        &self,
        input_ids: &Tensor,
        mask: &Tensor,
        at: usize,
    ) -> Result<super::modernbert::Stream> {
        if at > self.layers.len() {
            hanzo_ml::bail!("layer {at} of {}", self.layers.len());
        }
        let (b, seq_len) = input_ids.dims2()?;
        let dev = input_ids.device();
        let lens: Vec<usize> = mask
            .to_dtype(DType::U32)?
            .sum(1)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|x| x as usize)
            .collect();
        if self.packs() {
            let t_total: usize = lens.iter().sum();
            let mut ids_flat = Vec::with_capacity(t_total);
            let ids_2d = input_ids.to_vec2::<u32>()?;
            for (row, &n) in ids_2d.iter().zip(&lens) {
                ids_flat.extend_from_slice(&row[..n]);
            }
            let flat_ids = Tensor::from_vec(ids_flat, (t_total,), dev)?;
            let mut h = self.embed_tokens.forward(&flat_ids)?;
            for layer in &self.layers[..at] {
                h = layer.packed(&h, &lens)?;
            }
            let mut rows = Vec::with_capacity(b);
            let mut pos = 0;
            for &n in &lens {
                rows.push(h.narrow(0, pos, n)?);
                pos += n;
            }
            return super::modernbert::Stream::packed(&rows, at);
        }
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let global_mask = prepare_4d_attention_mask(mask, xs.dtype(), Some(1))?;
        let window = self.layers.iter().find_map(|l| l.window).unwrap_or(128);
        let tiles = UltrabertTiles::new(
            b,
            seq_len,
            self.config.num_attention_heads,
            window,
            Some(&global_mask),
            xs.dtype(),
            dev,
        )?;
        for layer in &self.layers[..at] {
            xs = layer.forward_tiled(&xs, &tiles)?;
        }
        let rows: Vec<Tensor> = lens
            .iter()
            .enumerate()
            .map(|(j, &n)| xs.narrow(0, j, 1)?.squeeze(0)?.narrow(0, 0, n))
            .collect::<Result<_>>()?;
        super::modernbert::Stream::packed(&rows, at)
    }

    /// Run `s` through layers `s.at()..` and the final norm.
    pub fn finish(&self, s: super::modernbert::Stream) -> Result<Tensor> {
        let at = s.at();
        if at > self.layers.len() {
            hanzo_ml::bail!("layer {at} of {}", self.layers.len());
        }
        let rows = s.rows()?;
        let b = rows.len();
        let lens: Vec<usize> = rows.iter().map(|r| r.dim(0)).collect::<Result<_>>()?;
        let seq_len = lens.iter().copied().max().unwrap_or(0);
        let d = self.config.hidden_size;
        let dev = self.embed_tokens.embeddings().device();
        if self.packs() {
            let mut h = Tensor::cat(&rows, 0)?;
            for layer in &self.layers[at..] {
                h = layer.packed(&h, &lens)?;
            }
            let h = h.apply(&self.norm)?;
            if b == 1 && lens[0] == seq_len {
                return h.reshape((1, seq_len, d));
            }
            let mut out = Vec::with_capacity(b);
            let mut pos = 0;
            for &n in &lens {
                let r = h.narrow(0, pos, n)?;
                pos += n;
                if n < seq_len {
                    let pad = Tensor::zeros((seq_len - n, d), h.dtype(), dev)?;
                    out.push(Tensor::cat(&[&r, &pad], 0)?.unsqueeze(0)?);
                } else {
                    out.push(r.unsqueeze(0)?);
                }
            }
            return Tensor::cat(&out, 0);
        }
        let dtype = self.embed_tokens.embeddings().dtype();
        let mut padded_rows = Vec::with_capacity(b);
        let mut mask_vec = vec![0u32; b * seq_len];
        for (j, (r, &n)) in rows.iter().zip(&lens).enumerate() {
            mask_vec[j * seq_len..j * seq_len + n].fill(1);
            if n < seq_len {
                let pad = Tensor::zeros((seq_len - n, d), dtype, dev)?;
                padded_rows.push(Tensor::cat(&[r, &pad], 0)?.unsqueeze(0)?);
            } else {
                padded_rows.push(r.unsqueeze(0)?);
            }
        }
        let mut xs = Tensor::cat(&padded_rows, 0)?;
        let mask = Tensor::from_vec(mask_vec, (b, seq_len), dev)?;
        let global_mask = prepare_4d_attention_mask(&mask, xs.dtype(), Some(1))?;
        let window = self.layers.iter().find_map(|l| l.window).unwrap_or(128);
        let tiles = UltrabertTiles::new(
            b,
            seq_len,
            self.config.num_attention_heads,
            window,
            Some(&global_mask),
            xs.dtype(),
            dev,
        )?;
        for layer in &self.layers[at..] {
            xs = layer.forward_tiled(&xs, &tiles)?;
        }
        xs.apply(&self.norm)
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len) = input_ids.dims2()?;
        let dev = input_ids.device();

        // Fast packed CPU path when enabled and MHA (`num_key_value_heads == num_attention_heads`
        // and `embed_tokens` is F32 on CPU).
        if self.packs() {
            let lens: Vec<usize> = match attention_mask {
                Some(m) => m
                    .to_dtype(DType::U32)?
                    .sum(1)?
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|x| x as usize)
                    .collect(),
                None => vec![seq_len; b],
            };
            let t_total: usize = lens.iter().sum();
            let mut ids_flat = Vec::with_capacity(t_total);
            let ids_2d = input_ids.to_vec2::<u32>()?;
            for (row, &n) in ids_2d.iter().zip(&lens) {
                ids_flat.extend_from_slice(&row[..n]);
            }
            let flat_ids = Tensor::from_vec(ids_flat, (t_total,), dev)?;
            let mut h = self.embed_tokens.forward(&flat_ids)?;
            for layer in &self.layers {
                h = layer.packed(&h, &lens)?;
            }
            let h = h.apply(&self.norm)?;
            if b == 1 && lens[0] == seq_len {
                return h.reshape((1, seq_len, self.config.hidden_size));
            }
            let d = self.config.hidden_size;
            let mut rows = Vec::with_capacity(b);
            let mut at = 0;
            for &n in &lens {
                let r = h.narrow(0, at, n)?;
                at += n;
                if n < seq_len {
                    let pad = Tensor::zeros((seq_len - n, d), h.dtype(), dev)?;
                    rows.push(Tensor::cat(&[&r, &pad], 0)?.unsqueeze(0)?);
                } else {
                    rows.push(r.unsqueeze(0)?);
                }
            }
            return Tensor::cat(&rows, 0);
        }

        let mut xs = self.embed_tokens.forward(input_ids)?;
        let global_mask = match attention_mask {
            Some(mask) => Some(prepare_4d_attention_mask(mask, xs.dtype(), Some(1))?),
            None => None,
        };

        let window = self.layers.iter().find_map(|l| l.window).unwrap_or(128);
        let tiles = UltrabertTiles::new(
            b,
            seq_len,
            self.config.num_attention_heads,
            window,
            global_mask.as_ref(),
            xs.dtype(),
            dev,
        )?;

        for layer in &self.layers {
            xs = layer.forward_tiled(&xs, &tiles)?;
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

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
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
        .to_dtype(DType::F32)?;

    let inverted_mask = (1.0 - expanded_mask)?;
    (inverted_mask * super::modernbert::masked_value(dtype))?.to_dtype(dtype)
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

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_nn::VarMap;

    #[test]
    fn short_context_stays_dense_until_min_seq_len() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: Some(2),
            head_dim: 16,
            rms_norm_eps: 1e-5,
            pad_token_id: 0,
            attention_windows: vec![Some(8), None],
            rope_theta: vec![10_000.0, 160_000.0],
            prediction_head_size: None,
            max_position_embeddings: Some(512),
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        // Initialize deterministic non-zero weights so attention scores vary across positions.
        let _ = UltrabertModel::load(vb.clone(), &cfg)?;
        for (i, (_, var)) in varmap.data().lock().unwrap().iter_mut().enumerate() {
            let elem_count = var.elem_count();
            let vals: Vec<f32> = (0..elem_count)
                .map(|j| (((i * 131 + j * 17 + 3) % 97) as f32 - 48.0) * 0.01)
                .collect();
            let t = Tensor::from_vec(vals, var.shape(), &dev)?;
            var.set(&t)?;
        }
        let model_dense = UltrabertModel::load(vb, &cfg)?;

        let mut model_gated = model_dense.clone();
        // Attach D-LSA with small k_budget=16, default min_seq_len=4096.
        model_gated.with_dlsa(16, 8, 2)?;
        assert_eq!(
            model_gated.layers[1]
                .self_attn
                .dlsa
                .as_ref()
                .unwrap()
                .min_seq_len,
            DEFAULT_DLSA_MIN_SEQ_LEN
        );

        // Sequence length 64 > k_budget(16), but 64 < min_seq_len(4096): MUST match dense bit-for-bit!
        let ids: Vec<u32> = (0..64).map(|i| ((i * 7 + 1) % 63 + 1) as u32).collect();
        let input_ids = Tensor::from_vec(ids, (1, 64), &dev)?;
        let mask = Tensor::ones((1, 64), DType::U32, &dev)?;

        let out_dense = model_dense.forward(&input_ids, Some(&mask))?;
        let out_short = model_gated.forward(&input_ids, Some(&mask))?;
        let max_diff_short = (&out_dense - &out_short)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert_eq!(
            max_diff_short, 0.0,
            "short context (< min_seq_len) must run exact dense attention"
        );

        // Packed CPU path must also stay exact dense below min_seq_len, and sparse above it.
        let mut packed_dense = model_dense.clone();
        packed_dense.pack();
        let mut packed_gated = model_gated.clone();
        packed_gated.pack();
        let packed_out_dense = packed_dense.forward(&input_ids, Some(&mask))?;
        let packed_out_short = packed_gated.forward(&input_ids, Some(&mask))?;
        let packed_diff_short = (&packed_out_dense - &packed_out_short)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert_eq!(
            packed_diff_short, 0.0,
            "packed short context must run exact dense attention"
        );

        // Lower min_seq_len to 32 so seq_len=64 qualifies as long-context: D-LSA now activates.
        packed_gated.set_dlsa_min_seq_len(32);
        let packed_out_long = packed_gated.forward(&input_ids, Some(&mask))?;
        let packed_diff_long = (&packed_out_dense - &packed_out_long)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            packed_diff_long > 0.0,
            "long context (>= min_seq_len) must engage sparse D-LSA"
        );
        Ok(())
    }
}
