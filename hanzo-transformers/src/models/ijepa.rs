//! I-JEPA vision encoder (Image-based Joint-Embedding Predictive Architecture).
//!
//! I-JEPA is a self-supervised vision transformer from Meta AI. This module implements
//! the encoder (context/target ViT) as exposed by Hugging Face `transformers.IJepaModel`.
//! It is a standard ViT with two differences from [`super::vit`]:
//!   - there is **no** `[CLS]` token — position embeddings cover exactly the patch grid
//!     and the model emits one hidden vector **per patch**;
//!   - the model output is the per-patch `last_hidden_state` after a final LayerNorm,
//!     not a pooled classification logit.
//!
//! The module tree mirrors the published `facebook/ijepa_vith14_1k` checkpoint layout
//! (`embeddings.*`, `encoder.layer.N.*`, `layernorm.*`), so those weights load directly
//! with no key remapping.
//!
//! - [Paper](https://arxiv.org/abs/2301.08243): "Self-Supervised Learning from Images
//!   with a Joint-Embedding Predictive Architecture"
//! - [Model card](https://huggingface.co/facebook/ijepa_vith14_1k)
//!
//! Note: the released checkpoint ships the **encoder only**. The latent `predictor`
//! (used for masked-target rollout during pre-training) is not part of `IJepaModel`
//! and is out of scope here.

use crate::models::with_tracing::{conv2d, layer_norm, linear_b, Conv2d, LayerNorm, Linear};
use hanzo_ml::{Module, Result, Tensor};
use hanzo_nn::VarBuilder;

// https://huggingface.co/facebook/ijepa_vith14_1k/blob/main/config.json
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: hanzo_nn::Activation,
    pub layer_norm_eps: f64,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub qkv_bias: bool,
}

impl Config {
    /// ViT-Huge/14 — the `facebook/ijepa_vith14_1k` reference checkpoint.
    pub fn vit_huge_patch14_224() -> Self {
        Self {
            hidden_size: 1280,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            intermediate_size: 5120,
            hidden_act: hanzo_nn::Activation::Gelu,
            layer_norm_eps: 1e-6,
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            qkv_bias: true,
        }
    }

    fn num_patches(&self) -> usize {
        let grid = self.image_size / self.patch_size;
        grid * grid
    }
}

#[derive(Debug, Clone)]
struct PatchEmbeddings {
    projection: Conv2d,
    num_patches: usize,
}

impl PatchEmbeddings {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = hanzo_nn::Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let projection = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("projection"),
        )?;
        Ok(Self {
            projection,
            num_patches: cfg.num_patches(),
        })
    }
}

impl Module for PatchEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // (b, c, h, w) -> (b, hidden, gh, gw) -> (b, num_patches, hidden)
        self.projection
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct Embeddings {
    patch_embeddings: PatchEmbeddings,
    position_embeddings: Tensor,
}

impl Embeddings {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let patch_embeddings = PatchEmbeddings::new(cfg, vb.pp("patch_embeddings"))?;
        let position_embeddings = vb.get(
            (1, patch_embeddings.num_patches, cfg.hidden_size),
            "position_embeddings",
        )?;
        Ok(Self {
            patch_embeddings,
            position_embeddings,
        })
    }
}

impl Module for Embeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let embeddings = self.patch_embeddings.forward(pixel_values)?;
        let n = embeddings.dim(1)?;
        let expected = self.position_embeddings.dim(1)?;
        if n != expected {
            hanzo_ml::bail!(
                "I-JEPA patch count {n} does not match the learned position grid {expected}; \
                 pass an image sized to the model's `image_size` (no pos-embed interpolation)"
            )
        }
        embeddings.broadcast_add(&self.position_embeddings)
    }
}

#[derive(Debug, Clone)]
struct SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let all = cfg.num_attention_heads * head_dim;
        let query = linear_b(cfg.hidden_size, all, cfg.qkv_bias, vb.pp("query"))?;
        let key = linear_b(cfg.hidden_size, all, cfg.qkv_bias, vb.pp("key"))?;
        let value = linear_b(cfg.hidden_size, all, cfg.qkv_bias, vb.pp("value"))?;
        Ok(Self {
            query,
            key,
            value,
            num_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn shape_heads(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        xs.reshape((b, n, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }
}

impl Module for SelfAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let q = self.shape_heads(&self.query.forward(xs)?)?;
        let k = self.shape_heads(&self.key.forward(xs)?)?;
        let v = self.shape_heads(&self.value.forward(xs)?)?;
        let scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let probs = hanzo_nn::ops::softmax_last_dim(&scores)?;
        probs.matmul(&v)?.permute((0, 2, 1, 3))?.reshape((b, n, c))
    }
}

#[derive(Debug, Clone)]
struct Attention {
    attention: SelfAttention,
    output: Linear,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = SelfAttention::new(cfg, vb.pp("attention"))?;
        let output = linear_b(
            cfg.hidden_size,
            cfg.hidden_size,
            true,
            vb.pp("output").pp("dense"),
        )?;
        Ok(Self { attention, output })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.output.forward(&self.attention.forward(xs)?)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    intermediate: Linear,
    output: Linear,
    act: hanzo_nn::Activation,
}

impl Mlp {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let intermediate = linear_b(
            cfg.hidden_size,
            cfg.intermediate_size,
            true,
            vb.pp("intermediate").pp("dense"),
        )?;
        let output = linear_b(
            cfg.intermediate_size,
            cfg.hidden_size,
            true,
            vb.pp("output").pp("dense"),
        )?;
        Ok(Self {
            intermediate,
            output,
            act: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.output
            .forward(&self.intermediate.forward(xs)?.apply(&self.act)?)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attention: Attention,
    mlp: Mlp,
    layernorm_before: LayerNorm,
    layernorm_after: LayerNorm,
}

impl Layer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // In the published checkpoint `intermediate.dense` and `output.dense` are direct
        // children of the layer (no `mlp.` prefix), so the MLP consumes the layer vb.
        let attention = Attention::new(cfg, vb.pp("attention"))?;
        let layernorm_before = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("layernorm_before"),
        )?;
        let layernorm_after = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("layernorm_after"),
        )?;
        let mlp = Mlp::new(cfg, vb)?;
        Ok(Self {
            attention,
            mlp,
            layernorm_before,
            layernorm_after,
        })
    }
}

impl Module for Layer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Pre-norm attention with residual, then pre-norm MLP with residual.
        let xs = (self
            .attention
            .forward(&self.layernorm_before.forward(xs)?)?
            + xs)?;
        let ys = self.mlp.forward(&self.layernorm_after.forward(&xs)?)?;
        ys + xs
    }
}

#[derive(Debug, Clone)]
pub struct IJepaModel {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    layernorm: LayerNorm,
}

impl IJepaModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embeddings = Embeddings::new(cfg, vb.pp("embeddings"))?;
        let vb_l = vb.pp("encoder").pp("layer");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| Layer::new(cfg, vb_l.pp(i)))
            .collect::<Result<Vec<_>>>()?;
        let layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layernorm"))?;
        Ok(Self {
            embeddings,
            layers,
            layernorm,
        })
    }

    /// Per-patch `last_hidden_state`, shape `(batch, num_patches, hidden_size)`.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        self.layernorm.forward(&xs)
    }

    /// Mean-pooled image embedding, shape `(batch, hidden_size)`.
    pub fn forward_pooled(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.forward(pixel_values)?.mean(1)
    }
}

impl Module for IJepaModel {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        IJepaModel::forward(self, pixel_values)
    }
}
