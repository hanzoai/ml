//! Trainable Qwen3 **DSpark** draft model (CPU / f32 MVP).
//!
//! Mirrors the DeepSpec reference (`deepspec/modeling/dspark/{common,markov_head,qwen3/modeling}.py`)
//! and the engine inference port (`hanzo-engine/src/models/qwen3_dspark.rs`) so the checkpoint this
//! writes loads byte-for-byte into the engine. Trainable weights live in a [`VarMap`]; the
//! `embed_tokens` / `lm_head` are loaded FROZEN from a real init checkpoint (never optimized).

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use hanzo_ml::{DType, Device, Result, Shape, Tensor, Var, D};
use hanzo_nn::{ops, Linear, Module, VarBuilder, VarMap};

fn io<E: std::fmt::Display>(e: E) -> hanzo_ml::Error {
    hanzo_ml::Error::msg(e.to_string())
}

/// Streams FROZEN `embed_tokens.weight` rows from the init safetensors on demand instead of holding
/// the whole `[vocab, hidden]` f16 table (~1GB) resident. DSpark only embeds ≤`block` tokens per
/// anchor (the anchor token + a constant MASK row), so per-row reads keep the trainer's resident
/// set small enough to coexist with a memory-heavy co-resident job.
pub struct EmbedReader {
    path: PathBuf,
    data_start: u64,
    vocab: usize,
    hidden: usize,
}

impl EmbedReader {
    pub fn open(path: &Path, vocab: usize, hidden: usize) -> Result<Self> {
        let mut f = File::open(path).map_err(io)?;
        let mut lenb = [0u8; 8];
        f.read_exact(&mut lenb).map_err(io)?;
        let hlen = u64::from_le_bytes(lenb) as usize;
        let mut hbuf = vec![0u8; hlen];
        f.read_exact(&mut hbuf).map_err(io)?;
        let hdr: serde_json::Value = serde_json::from_slice(&hbuf).map_err(io)?;
        let meta = &hdr["embed_tokens.weight"];
        let dtype = meta["dtype"].as_str().unwrap_or_default();
        if dtype != "F16" {
            hanzo_ml::bail!("embed_tokens.weight must be F16, got {dtype}");
        }
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_default();
        if shape != [vocab, hidden] {
            hanzo_ml::bail!("embed_tokens.weight shape {shape:?} != [{vocab}, {hidden}]");
        }
        let s = meta["data_offsets"][0].as_u64().unwrap_or(0);
        Ok(Self {
            path: path.to_path_buf(),
            data_start: 8 + hlen as u64 + s,
            vocab,
            hidden,
        })
    }

    fn read_f16_at(&self, byte_off: u64, count: usize) -> Result<Vec<half::f16>> {
        let mut f = File::open(&self.path).map_err(io)?;
        f.seek(SeekFrom::Start(byte_off)).map_err(io)?;
        let mut buf = vec![0u8; count * 2];
        f.read_exact(&mut buf).map_err(io)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    /// One embedding row as f32.
    pub fn row_f32(&self, token: usize) -> Result<Vec<f32>> {
        if token >= self.vocab {
            hanzo_ml::bail!("embed row {token} out of vocab {}", self.vocab);
        }
        let off = self.data_start + (token * self.hidden * 2) as u64;
        Ok(self
            .read_f16_at(off, self.hidden)?
            .into_iter()
            .map(|h| h.to_f32())
            .collect())
    }

    /// The full `[vocab, hidden]` f16 table — materialized only at checkpoint-save time.
    pub fn full_f16(&self, dev: &Device) -> Result<Tensor> {
        let vals = self.read_f16_at(self.data_start, self.vocab * self.hidden)?;
        Tensor::from_vec(vals, (self.vocab, self.hidden), dev)
    }
}

/// DSpark draft config. MVP knobs; the hidden width is fixed at 4096 by the cache.
#[derive(Clone, Debug)]
pub struct DsparkCfg {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f64,
    pub rope_theta: f64,
    pub max_pos: usize,
    pub block: usize,
    pub mask_token_id: u32,
    pub markov_rank: usize,
    pub target_layer_ids: Vec<i64>,
    /// Init constant for the output-side `norm` (RMSNorm before `lm_head`). The frozen head rows
    /// have L2 norm ~11.3, so a unit-scale output makes initial logits std ~11.3 (CE ~63). Starting
    /// the output norm small (e.g. 0.1) puts the initial loss near `ln(vocab)` and well-conditions
    /// training: AdamW grows the scale as the draft aligns with the correct head rows. `1.0` = faithful.
    pub final_norm_init: f64,
}

impl DsparkCfg {
    pub fn n_fused(&self) -> usize {
        self.target_layer_ids.len()
    }
}

struct Layer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Tensor, // [head_dim]
    k_norm: Tensor, // [head_dim]
    input_ln: Tensor,
    post_ln: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
}

pub struct Dspark {
    cfg: DsparkCfg,
    dev: Device,
    varmap: VarMap,
    fc: Linear,
    hidden_norm: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    markov_w1: Tensor,    // [vocab, rank] (nn.Embedding weight)
    markov_w2: Tensor,    // [vocab, rank] (nn.Linear rank->vocab weight)
    embed: EmbedReader,   // FROZEN [vocab, hidden], streamed on demand
    mask_embed: Vec<f32>, // FROZEN embedding row for `mask_token_id` (block slots 1..)
    lm_head: Linear,      // FROZEN weight [vocab, hidden] (F16)
    conf_w: Tensor,       // zero-init [1, hidden] (load-compat, untrained)
    conf_b: Tensor,       // zero-init [1]
    cos: Tensor,          // [max_pos, head_dim]
    sin: Tensor,          // [max_pos, head_dim]
    train_markov: bool, // when false, the Markov head is frozen (excluded from the optimizer + detached)
}

fn tvar(
    vm: &VarMap,
    dev: &Device,
    shape: impl Into<Shape>,
    name: &str,
    init: hanzo_nn::Init,
) -> Result<Tensor> {
    vm.get(shape, name, init, DType::F32, dev)
}

impl Dspark {
    /// Build the model: trainable weights randomly initialized, `embed_tokens`/`lm_head` loaded
    /// FROZEN from `init_path` (keys `embed_tokens.weight`, `lm_head.weight`). `train_markov=false`
    /// freezes the vocab×rank Markov head (excludes it from the optimizer + detaches its bias),
    /// which removes its AdamW state — a memory lever for constrained hosts.
    pub fn new(cfg: DsparkCfg, init_path: &Path, dev: &Device, train_markov: bool) -> Result<Self> {
        use hanzo_nn::Init;
        let normal = Init::Randn {
            mean: 0.0,
            stdev: 0.02,
        };
        let ones = Init::Const(1.0);
        let vm = VarMap::new();

        let h = cfg.hidden;
        let hd = cfg.head_dim;
        let qd = cfg.heads * hd;
        let kvd = cfg.kv_heads * hd;
        let fin = cfg.n_fused() * h;

        let fc = Linear::new(tvar(&vm, dev, (h, fin), "fc.weight", normal)?, None);
        let hidden_norm = tvar(&vm, dev, h, "hidden_norm.weight", ones)?;

        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(Layer {
                q_proj: Linear::new(
                    tvar(&vm, dev, (qd, h), &p("self_attn.q_proj.weight"), normal)?,
                    None,
                ),
                k_proj: Linear::new(
                    tvar(&vm, dev, (kvd, h), &p("self_attn.k_proj.weight"), normal)?,
                    None,
                ),
                v_proj: Linear::new(
                    tvar(&vm, dev, (kvd, h), &p("self_attn.v_proj.weight"), normal)?,
                    None,
                ),
                o_proj: Linear::new(
                    tvar(&vm, dev, (h, qd), &p("self_attn.o_proj.weight"), normal)?,
                    None,
                ),
                q_norm: tvar(&vm, dev, hd, &p("self_attn.q_norm.weight"), ones)?,
                k_norm: tvar(&vm, dev, hd, &p("self_attn.k_norm.weight"), ones)?,
                input_ln: tvar(&vm, dev, h, &p("input_layernorm.weight"), ones)?,
                post_ln: tvar(&vm, dev, h, &p("post_attention_layernorm.weight"), ones)?,
                gate: Linear::new(
                    tvar(
                        &vm,
                        dev,
                        (cfg.intermediate, h),
                        &p("mlp.gate_proj.weight"),
                        normal,
                    )?,
                    None,
                ),
                up: Linear::new(
                    tvar(
                        &vm,
                        dev,
                        (cfg.intermediate, h),
                        &p("mlp.up_proj.weight"),
                        normal,
                    )?,
                    None,
                ),
                down: Linear::new(
                    tvar(
                        &vm,
                        dev,
                        (h, cfg.intermediate),
                        &p("mlp.down_proj.weight"),
                        normal,
                    )?,
                    None,
                ),
            });
        }

        let norm = tvar(&vm, dev, h, "norm.weight", Init::Const(cfg.final_norm_init))?;
        let markov_w1 = tvar(
            &vm,
            dev,
            (cfg.vocab, cfg.markov_rank),
            "markov_head.markov_w1.weight",
            normal,
        )?;
        let markov_w2 = tvar(
            &vm,
            dev,
            (cfg.vocab, cfg.markov_rank),
            "markov_head.markov_w2.weight",
            normal,
        )?;

        // FROZEN embed streamed on demand (not resident); FROZEN lm_head kept in F16 (1.05GB, vs
        // 2.1GB f32). The CPU backend supports F16 matmul (not BF16). `lm_head`'s [hidden, vocab]
        // gradient is never formed (see `frozen_head_ce`).
        let embed = EmbedReader::open(init_path, cfg.vocab, h)?;
        let mask_embed = embed.row_f32(cfg.mask_token_id as usize)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[init_path], DType::F16, dev)? };
        let lm_head = Linear::new(vb.get((cfg.vocab, h), "lm_head.weight")?, None);

        let conf_w = Tensor::zeros((1, h), DType::F32, dev)?;
        let conf_b = Tensor::zeros((1,), DType::F32, dev)?;

        let (cos, sin) = rope_tables(cfg.max_pos, hd, cfg.rope_theta, dev)?;

        Ok(Self {
            cfg,
            dev: dev.clone(),
            varmap: vm,
            fc,
            hidden_norm,
            layers,
            norm,
            markov_w1,
            markov_w2,
            embed,
            mask_embed,
            lm_head,
            conf_w,
            conf_b,
            cos,
            sin,
            train_markov,
        })
    }

    pub fn cfg(&self) -> &DsparkCfg {
        &self.cfg
    }

    /// The trainable variables for the optimizer (everything except the frozen head/embed/confidence,
    /// and — when `train_markov` is false — the Markov head).
    pub fn trainable_vars(&self) -> Vec<Var> {
        let all = self.varmap.all_vars();
        if self.train_markov {
            return all;
        }
        let skip = [self.markov_w1.id(), self.markov_w2.id()];
        all.into_iter()
            .filter(|v| !skip.contains(&v.id()))
            .collect()
    }

    /// Fuse the target hidden states into the DSpark context memory, over the whole sequence at once
    /// (`hidden_norm(fc(target_hidden))`). Anchors slice prefixes of this — identical to the
    /// per-anchor computation since both ops are row-wise, and far cheaper than recomputing per anchor.
    ///
    /// `target_hidden`: `[seq, n_fused*hidden]` f32.  Returns `[seq, hidden]`.
    pub fn fuse(&self, target_hidden: &Tensor) -> Result<Tensor> {
        let x = self.fc.forward(target_hidden)?;
        ops::rms_norm(&x.contiguous()?, &self.hidden_norm, self.cfg.rms_eps as f32)
    }

    /// One DSpark draft-block forward for a single anchor `a` (context = fused rows `0..a`, strictly
    /// before the anchor). Returns `(block_out[n_valid, hidden], markov_bias[n_valid, vocab],
    /// targets[n_valid])` where the supervised slots are the leading contiguous run whose target
    /// token is in-range and `loss_mask`-enabled (matching `build_eval_mask`'s cumprod). `Ok(None)`
    /// if no slot is supervised. The frozen `lm_head` is deliberately NOT applied here — it is
    /// applied once over the concatenated batch in [`frozen_head_ce`].
    pub fn draft_anchor(
        &self,
        fused: &Tensor,
        input_ids: &[i32],
        loss_mask: &[u8],
        a: usize,
    ) -> Result<Option<(Tensor, Tensor, Vec<u32>)>> {
        let seq = fused.dim(0)?;
        let bs = self.cfg.block;
        let vocab = self.cfg.vocab;
        let eps = self.cfg.rms_eps as f32;

        // Validate the block ids we will index the embedding / markov tables with.
        let tok = |i: usize| -> Option<u32> {
            let t = input_ids[i];
            if t >= 0 && (t as usize) < vocab {
                Some(t as u32)
            } else {
                None
            }
        };

        let h = self.cfg.hidden;
        let ctx = fused.narrow(0, 0, a)?; // [a, hidden]

        // create_noise_embed: slot 0 = anchor token, slots 1.. = mask_token_id. Streamed from the
        // FROZEN embedding; the block state is a plain leaf (never optimized, no grad to the table).
        let anchor_tok =
            tok(a).ok_or_else(|| hanzo_ml::Error::msg("anchor token out of vocab"))? as usize;
        let mut hv = Vec::with_capacity(bs * h);
        hv.extend_from_slice(&self.embed.row_f32(anchor_tok)?);
        for _ in 1..bs {
            hv.extend_from_slice(&self.mask_embed);
        }
        let mut hstate = Tensor::from_vec(hv, (bs, h), &self.dev)?; // [bs, hidden]

        // RoPE: block queries at positions a..a+bs, context+block keys at 0..a+bs.
        let cos_full = self.cos.narrow(0, 0, a + bs)?;
        let sin_full = self.sin.narrow(0, 0, a + bs)?;
        let cos_draft = self.cos.narrow(0, a, bs)?;
        let sin_draft = self.sin.narrow(0, a, bs)?;
        let mask = dspark_mask(a, bs, a, &self.dev)?; // [bs, a+bs]

        for layer in &self.layers {
            let normed = ops::rms_norm(&hstate.contiguous()?, &layer.input_ln, eps)?;
            let attn = self.attention(
                layer, &normed, &ctx, &cos_draft, &sin_draft, &cos_full, &sin_full, &mask,
            )?;
            hstate = hstate.add(&attn)?;
            let normed2 = ops::rms_norm(&hstate.contiguous()?, &layer.post_ln, eps)?;
            let gate = ops::silu(&layer.gate.forward(&normed2)?)?;
            let up = layer.up.forward(&normed2)?;
            let mlp = layer.down.forward(&gate.mul(&up)?)?;
            hstate = hstate.add(&mlp)?;
        }
        let block_out = ops::rms_norm(&hstate.contiguous()?, &self.norm, eps)?; // [bs, hidden]

        // Vanilla Markov teacher-forced bias: prev[k] = ids[a+k]. (The frozen lm_head is applied
        // later, once, over the whole batch — see `frozen_head_ce`.)
        let mut prevs = Vec::with_capacity(bs);
        for k in 0..bs {
            prevs.push(
                tok(a + k).ok_or_else(|| hanzo_ml::Error::msg("markov prev token out of vocab"))?,
            );
        }
        let prev_t = Tensor::from_vec(prevs, (bs,), &self.dev)?;
        // Frozen Markov ⇒ detach so no vocab×rank grad is formed and it stays at init.
        let bias = if self.train_markov {
            markov_bias(&self.markov_w1, &self.markov_w2, &prev_t)?
        } else {
            markov_bias(&self.markov_w1.detach(), &self.markov_w2.detach(), &prev_t)?
        }; // [bs, vocab]

        // Supervised slots: leading contiguous run of in-range, loss-enabled targets a+1+k.
        let mut targets = Vec::with_capacity(bs);
        for k in 0..bs {
            let tp = a + 1 + k;
            if tp >= seq || loss_mask[tp] == 0 {
                break;
            }
            match tok(tp) {
                Some(t) => targets.push(t),
                None => break,
            }
        }
        if targets.is_empty() {
            return Ok(None);
        }
        let nv = targets.len();
        Ok(Some((
            block_out.narrow(0, 0, nv)?,
            bias.narrow(0, 0, nv)?,
            targets,
        )))
    }

    /// Cross-entropy through the FROZEN `lm_head` for a whole batch, returning
    /// `(loss_value, surrogate)`. See [`frozen_head_ce`]; this just supplies the head weight.
    pub fn head_ce(
        &self,
        block_cat: &Tensor,
        bias_cat: &Tensor,
        targets: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        frozen_head_ce(block_cat, self.lm_head.weight(), bias_cat, targets)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &Layer,
        x: &Tensor,   // [bs, hidden] block
        ctx: &Tensor, // [a, hidden] fused context
        cos_draft: &Tensor,
        sin_draft: &Tensor,
        cos_full: &Tensor,
        sin_full: &Tensor,
        mask: &Tensor, // [bs, a+bs]
    ) -> Result<Tensor> {
        let bs = x.dim(0)?;
        let a = ctx.dim(0)?;
        let kl = a + bs;
        let hd = self.cfg.head_dim;
        let nh = self.cfg.heads;
        let nkv = self.cfg.kv_heads;
        let nrep = nh / nkv;
        let eps = self.cfg.rms_eps as f32;

        // Query: block only.
        let q = layer.q_proj.forward(x)?.reshape((bs, nh, hd))?;
        let q = ops::rms_norm(&q.contiguous()?, &layer.q_norm, eps)?;
        let q = q.transpose(0, 1)?.contiguous()?; // [nh, bs, hd]
        let q = apply_rope(&q, cos_draft, sin_draft)?;

        // Keys/values: [fused context ‖ block].
        let k = Tensor::cat(&[&layer.k_proj.forward(ctx)?, &layer.k_proj.forward(x)?], 0)?
            .reshape((kl, nkv, hd))?;
        let k = ops::rms_norm(&k.contiguous()?, &layer.k_norm, eps)?;
        let k = k.transpose(0, 1)?.contiguous()?; // [nkv, kl, hd]
        let k = apply_rope(&k, cos_full, sin_full)?;
        let k = repeat_kv(&k, nrep)?; // [nh, kl, hd]

        let v = Tensor::cat(&[&layer.v_proj.forward(ctx)?, &layer.v_proj.forward(x)?], 0)?
            .reshape((kl, nkv, hd))?
            .transpose(0, 1)?
            .contiguous()?; // [nkv, kl, hd]
        let v = repeat_kv(&v, nrep)?; // [nh, kl, hd]

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)? // [nh, bs, kl]
            .affine(scale, 0.0)?
            .broadcast_add(&mask.unsqueeze(0)?)?;
        let probs = ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v.contiguous()?)?; // [nh, bs, hd]
        let out = out.transpose(0, 1)?.contiguous()?.reshape((bs, nh * hd))?;
        layer.o_proj.forward(&out)
    }

    /// Write the checkpoint (`model.safetensors` + `config.json`) with the EXACT engine key names.
    pub fn save(&self, out_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(out_dir).map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        let mut map: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in self.varmap.data().lock().unwrap().iter() {
            map.insert(k.clone(), v.as_tensor().clone());
        }
        map.insert(
            "embed_tokens.weight".into(),
            self.embed.full_f16(&self.dev)?,
        );
        map.insert("lm_head.weight".into(), self.lm_head.weight().clone());
        map.insert("confidence_head.proj.weight".into(), self.conf_w.clone());
        map.insert("confidence_head.proj.bias".into(), self.conf_b.clone());
        hanzo_ml::safetensors::save(&map, out_dir.join("model.safetensors"))?;
        std::fs::write(out_dir.join("config.json"), self.config_json())
            .map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        Ok(())
    }

    fn config_json(&self) -> String {
        let c = &self.cfg;
        let v = serde_json::json!({
            "architectures": ["Qwen3DSparkModel"],
            "model_type": "qwen3",
            "attention_bias": false,
            "attention_dropout": 0.0,
            "vocab_size": c.vocab,
            "hidden_size": c.hidden,
            "intermediate_size": c.intermediate,
            "num_hidden_layers": c.layers,
            "num_attention_heads": c.heads,
            "num_key_value_heads": c.kv_heads,
            "head_dim": c.head_dim,
            "hidden_act": "silu",
            "rms_norm_eps": c.rms_eps,
            // Advertised inference context — independent of the (possibly smaller) training RoPE
            // table `max_pos`. The engine rebuilds its RoPE table from `rope_theta` up to this value.
            "max_position_embeddings": 40960,
            "rope_parameters": { "rope_theta": c.rope_theta, "rope_type": "default" },
            "block_size": c.block,
            "mask_token_id": c.mask_token_id,
            "markov_rank": c.markov_rank,
            "markov_head_type": "vanilla",
            "target_layer_ids": c.target_layer_ids,
            "num_target_layers": 48,
            "enable_confidence_head": false,
            "confidence_head_with_markov": false,
            "tie_word_embeddings": false
        });
        serde_json::to_string_pretty(&v).unwrap()
    }
}

/// neox `rotate_half`: split the last dim in half → `[-x2, x1]`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply neox RoPE. `x`: `[heads, L, head_dim]`; `cos`/`sin`: `[L, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?;
    let rot = rotate_half(x)?;
    x.broadcast_mul(&cos)?.add(&rot.broadcast_mul(&sin)?)
}

/// GQA key/value expansion (repeat_interleave): `[nkv, L, d]` → `[nkv*nrep, L, d]`.
fn repeat_kv(x: &Tensor, nrep: usize) -> Result<Tensor> {
    if nrep == 1 {
        return Ok(x.clone());
    }
    let (nkv, l, d) = x.dims3()?;
    x.unsqueeze(1)?
        .broadcast_as((nkv, nrep, l, d))?
        .contiguous()?
        .reshape((nkv * nrep, l, d))
}

/// Precompute neox RoPE cos/sin tables `[max_pos, head_dim]` (each half duplicated).
fn rope_tables(
    max_pos: usize,
    head_dim: usize,
    theta: f64,
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_pos * head_dim];
    let mut sin = vec![0f32; max_pos * head_dim];
    for p in 0..max_pos {
        for j in 0..half {
            let inv = (theta as f32).powf(-(2.0 * j as f32) / head_dim as f32);
            let (s, c) = (p as f32 * inv).sin_cos();
            cos[p * head_dim + j] = c;
            cos[p * head_dim + half + j] = c;
            sin[p * head_dim + j] = s;
            sin[p * head_dim + half + j] = s;
        }
    }
    Ok((
        Tensor::from_vec(cos, (max_pos, head_dim), dev)?,
        Tensor::from_vec(sin, (max_pos, head_dim), dev)?,
    ))
}

/// Additive DSpark attention mask `[block, ctx_len+block]`: `0` where attended, `-inf` where masked.
/// A context column `j` is attended iff `j < anchor_pos`; every block column is attended
/// (bidirectional block self-attention). Mirrors `create_dspark_attention_mask`.
pub fn dspark_mask(
    ctx_len: usize,
    block: usize,
    anchor_pos: usize,
    dev: &Device,
) -> Result<Tensor> {
    let kl = ctx_len + block;
    let neg = f32::NEG_INFINITY;
    let mut row = vec![0f32; kl];
    for (j, cell) in row.iter_mut().enumerate().take(ctx_len) {
        if j >= anchor_pos {
            *cell = neg;
        }
    }
    let mut data = Vec::with_capacity(block * kl);
    for _ in 0..block {
        data.extend_from_slice(&row);
    }
    Tensor::from_vec(data, (block, kl), dev)
}

/// Vanilla Markov bias `W1[prev] · W2^T → [n, vocab]`. `w1`/`w2` are both `[vocab, rank]`
/// (`nn.Embedding` and `nn.Linear(rank->vocab)` weights). Reference for the model AND the test.
pub fn markov_bias(w1: &Tensor, w2: &Tensor, prev: &Tensor) -> Result<Tensor> {
    let latent = w1.index_select(prev, 0)?; // [n, rank]
    latent.matmul(&w2.t()?) // [n, vocab]
}

/// Cross-entropy through a **frozen** linear head `lm_w [vocab, hidden]`, returning
/// `(loss_value, surrogate)`.
///
/// `loss_value` is the detached CE scalar (for logging). `surrogate` is a scalar whose gradient
/// w.r.t. `block_cat` and `bias_cat` EQUALS the true CE gradient, yet it never materializes the
/// head's `[hidden, vocab]` gradient — hanzo-ml's backprop otherwise computes that (2.1GB f32)
/// unconditionally for the head matmul's constant operand. Skipping it is the key memory/compute
/// saving that lets the frozen head train a draft on a shared, memory-constrained host.
///
/// Identity: for `z = block_cat·lm_wᵀ + bias_cat`, `CE = mean_n(logsumexp(z_n) - z_n[t_n])`, so
/// `dL/dz = (softmax(z) - onehot)/N`, `dL/d block_cat = (dL/dz)·lm_w`, `dL/d bias_cat = dL/dz`.
/// The surrogate `⟨block_cat, (dL/dz)·lm_w⟩ + ⟨bias_cat, dL/dz⟩` reproduces exactly those grads
/// because the head/`resid` factors are detached constants.
///
/// `block_cat [N, hidden]` (trainable), `bias_cat [N, vocab]` (trainable), `targets [N] u32`.
pub fn frozen_head_ce(
    block_cat: &Tensor,
    lm_w: &Tensor,
    bias_cat: &Tensor,
    targets: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let n = block_cat.dim(0)?;
    let vocab = lm_w.dim(0)?;
    let dev = block_cat.device();
    let hd = lm_w.dtype(); // frozen head dtype (F16 in the trainer)

    // Forward logits, fully DETACHED (constants) — used only to form p and the loss value.
    let z = block_cat
        .detach()
        .to_dtype(hd)?
        .matmul(&lm_w.t()?)? // [N, vocab]
        .to_dtype(DType::F32)?
        .add(&bias_cat.detach())?;
    let loss_value = hanzo_nn::loss::cross_entropy(&z, targets)?; // detached scalar

    let p = ops::softmax(&z, 1)?; // [N, vocab]
    let onehot = Tensor::zeros((n, vocab), DType::F32, dev)?.scatter_add(
        &targets.unsqueeze(1)?,
        &Tensor::ones((n, 1), DType::F32, dev)?,
        1,
    )?;
    let resid = p.sub(&onehot)?.affine(1.0 / n as f64, 0.0)?; // (softmax - onehot)/N, [N, vocab]

    // dL/d block_cat = resid · lm_w  →  [N, hidden] (constant; no head grad is formed).
    let gblock = resid.to_dtype(hd)?.matmul(lm_w)?.to_dtype(DType::F32)?;

    // Surrogate whose gradient equals the true CE gradient on the two trainable inputs.
    let surrogate = block_cat
        .mul(&gblock)?
        .sum_all()?
        .add(&bias_cat.mul(&resid)?.sum_all()?)?;
    Ok((loss_value, surrogate))
}

/// Reopen a saved checkpoint and assert every expected engine key is present with the right shape.
/// This is the MVP loop-closure gate (the engine-side load test is a follow-on).
pub fn verify_checkpoint(path: &Path, cfg: &DsparkCfg) -> Result<()> {
    let map = hanzo_ml::safetensors::load(path, &Device::Cpu)?;
    let h = cfg.hidden;
    let hd = cfg.head_dim;
    let qd = cfg.heads * hd;
    let kvd = cfg.kv_heads * hd;

    let mut expected: Vec<(String, Vec<usize>)> = vec![
        ("embed_tokens.weight".into(), vec![cfg.vocab, h]),
        ("fc.weight".into(), vec![h, cfg.n_fused() * h]),
        ("hidden_norm.weight".into(), vec![h]),
        ("norm.weight".into(), vec![h]),
        ("lm_head.weight".into(), vec![cfg.vocab, h]),
        (
            "markov_head.markov_w1.weight".into(),
            vec![cfg.vocab, cfg.markov_rank],
        ),
        (
            "markov_head.markov_w2.weight".into(),
            vec![cfg.vocab, cfg.markov_rank],
        ),
        ("confidence_head.proj.weight".into(), vec![1, h]),
        ("confidence_head.proj.bias".into(), vec![1]),
    ];
    for i in 0..cfg.layers {
        let p = |s: &str| format!("layers.{i}.{s}");
        expected.extend([
            (p("self_attn.q_proj.weight"), vec![qd, h]),
            (p("self_attn.k_proj.weight"), vec![kvd, h]),
            (p("self_attn.v_proj.weight"), vec![kvd, h]),
            (p("self_attn.o_proj.weight"), vec![h, qd]),
            (p("self_attn.q_norm.weight"), vec![hd]),
            (p("self_attn.k_norm.weight"), vec![hd]),
            (p("input_layernorm.weight"), vec![h]),
            (p("post_attention_layernorm.weight"), vec![h]),
            (p("mlp.gate_proj.weight"), vec![cfg.intermediate, h]),
            (p("mlp.up_proj.weight"), vec![cfg.intermediate, h]),
            (p("mlp.down_proj.weight"), vec![h, cfg.intermediate]),
        ]);
    }

    for (k, shape) in &expected {
        let t = map
            .get(k)
            .ok_or_else(|| hanzo_ml::Error::msg(format!("missing key {k}")))?;
        if t.dims() != shape.as_slice() {
            hanzo_ml::bail!("shape mismatch for {k}: got {:?}, want {shape:?}", t.dims());
        }
    }
    if map.len() != expected.len() {
        hanzo_ml::bail!(
            "checkpoint has {} tensors, expected {}",
            map.len(),
            expected.len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markov_bias_hand_case() -> Result<()> {
        let dev = Device::Cpu;
        // vocab=2, rank=2. w1 rows = prev-token latents; w2 rows = per-vocab projection.
        let w1 = Tensor::from_vec(vec![1f32, 0., 0., 1.], (2, 2), &dev)?; // [[1,0],[0,1]]
        let w2 = Tensor::from_vec(vec![1f32, 2., 3., 4.], (2, 2), &dev)?; // [[1,2],[3,4]]
                                                                          // prev token 0 -> latent [1,0]; bias = latent · w2^T = [1*1+0*2, 1*3+0*4] = [1,3].
        let prev = Tensor::from_vec(vec![0u32], (1,), &dev)?;
        let bias = markov_bias(&w1, &w2, &prev)?;
        assert_eq!(bias.to_vec2::<f32>()?, vec![vec![1.0, 3.0]]);
        // prev token 1 -> latent [0,1]; bias = [2,4].
        let prev1 = Tensor::from_vec(vec![1u32], (1,), &dev)?;
        assert_eq!(
            markov_bias(&w1, &w2, &prev1)?.to_vec2::<f32>()?,
            vec![vec![2.0, 4.0]]
        );
        Ok(())
    }

    #[test]
    fn frozen_head_ce_matches_autograd() -> Result<()> {
        use hanzo_ml::Var;
        let dev = Device::Cpu;
        let (n, h, v) = (3usize, 4usize, 5usize);
        let block = Var::from_tensor(&Tensor::from_vec(
            (0..n * h).map(|i| i as f32 * 0.1 - 0.5).collect::<Vec<_>>(),
            (n, h),
            &dev,
        )?)?;
        let lm_w = Tensor::from_vec(
            (0..v * h)
                .map(|i| (i % 7) as f32 * 0.13 - 0.4)
                .collect::<Vec<_>>(),
            (v, h),
            &dev,
        )?;
        let bias = Var::from_tensor(&Tensor::from_vec(
            (0..n * v)
                .map(|i| (i % 5) as f32 * 0.05)
                .collect::<Vec<_>>(),
            (n, v),
            &dev,
        )?)?;
        let targets = Tensor::from_vec(vec![1u32, 4, 2], (n,), &dev)?;

        // Reference: full autograd through the head (lm_w kept f32 so no rounding).
        let z_ref = block
            .as_tensor()
            .matmul(&lm_w.t()?)?
            .add(bias.as_tensor())?;
        let loss_ref = hanzo_nn::loss::cross_entropy(&z_ref, &targets)?;
        let g_ref = loss_ref.backward()?;
        let gb_ref = g_ref.get(&block).unwrap().flatten_all()?.to_vec1::<f32>()?;
        let gbias_ref = g_ref.get(&bias).unwrap().flatten_all()?.to_vec1::<f32>()?;

        // Surrogate: same grads, no [hidden, vocab] head grad formed.
        let (loss_val, surrogate) =
            frozen_head_ce(block.as_tensor(), &lm_w, bias.as_tensor(), &targets)?;
        let g = surrogate.backward()?;
        let gb = g.get(&block).unwrap().flatten_all()?.to_vec1::<f32>()?;
        let gbias = g.get(&bias).unwrap().flatten_all()?.to_vec1::<f32>()?;

        assert!((loss_val.to_scalar::<f32>()? - loss_ref.to_scalar::<f32>()?).abs() < 1e-5);
        for (a, b) in gb.iter().zip(gb_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "block grad mismatch {a} vs {b}");
        }
        for (a, b) in gbias.iter().zip(gbias_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "bias grad mismatch {a} vs {b}");
        }
        Ok(())
    }

    /// End-to-end proof that the training mechanism (frozen-head CE surrogate + real autograd +
    /// hanzo-nn AdamW) actually DECREASES loss — the same code path the trainer uses, but tiny and
    /// memory-free so it runs anywhere. A trainable projection `theta` feeds a FROZEN head; CE must
    /// fall substantially from its ~ln(vocab) start.
    #[test]
    fn training_loop_reduces_ce_through_frozen_head() -> Result<()> {
        use hanzo_ml::Var;
        use hanzo_nn::optim::{AdamW, Optimizer, ParamsAdamW};
        let dev = Device::Cpu;
        let (n, d, h, v) = (8usize, 6usize, 5usize, 11usize);
        let x = Tensor::from_vec(
            (0..n * d)
                .map(|i| ((i * 7) % 13) as f32 * 0.1 - 0.6)
                .collect::<Vec<_>>(),
            (n, d),
            &dev,
        )?;
        let w = Tensor::from_vec(
            (0..v * h)
                .map(|i| ((i * 5) % 17) as f32 * 0.1 - 0.8)
                .collect::<Vec<_>>(),
            (v, h),
            &dev,
        )?;
        let bias = Tensor::zeros((n, v), DType::F32, &dev)?;
        let targets = Tensor::from_vec(
            (0..n as u32)
                .map(|i| (i * 3) % v as u32)
                .collect::<Vec<_>>(),
            (n,),
            &dev,
        )?;
        // theta = 0 ⇒ block = 0 ⇒ uniform logits ⇒ CE ≈ ln(v) at step 0.
        let theta = Var::from_tensor(&Tensor::zeros((d, h), DType::F32, &dev)?)?;

        let mut opt = AdamW::from_slice(
            &[&theta],
            ParamsAdamW {
                lr: 0.2,
                ..Default::default()
            },
        )?;
        let mut first = 0f32;
        let mut last = 0f32;
        for step in 0..60 {
            let block = x.matmul(theta.as_tensor())?; // [n, h], trainable via theta
            let (loss, surrogate) = frozen_head_ce(&block, &w, &bias, &targets)?;
            opt.step(&surrogate.backward()?)?;
            let l = loss.to_scalar::<f32>()?;
            if step == 0 {
                first = l;
            }
            last = l;
            if step % 10 == 0 || step == 59 {
                eprintln!("  [proof] step {step:>2} CE {l:.4}");
            }
        }
        let ln_v = (v as f32).ln();
        assert!(
            (first - ln_v).abs() < 0.2,
            "start CE {first} should be ~ln(v) {ln_v}"
        );
        assert!(
            last < first * 0.6,
            "CE should drop substantially: {first} -> {last}"
        );
        Ok(())
    }

    #[test]
    fn dspark_mask_shape_and_values() -> Result<()> {
        let dev = Device::Cpu;
        // ctx_len=3, block=2, anchor_pos=3: all 3 context cols < anchor => visible; block cols visible.
        let m = dspark_mask(3, 2, 3, &dev)?;
        assert_eq!(m.dims(), &[2, 5]);
        let rows = m.to_vec2::<f32>()?;
        assert_eq!(rows[0], vec![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(rows[1], rows[0]);
        // anchor_pos=1 with ctx_len=3: cols 1,2 are >= anchor => masked (-inf).
        let m2 = dspark_mask(3, 1, 1, &dev)?;
        let r = m2.to_vec2::<f32>()?;
        assert_eq!(r[0][0], 0.0);
        assert!(r[0][1].is_infinite() && r[0][1] < 0.0);
        assert!(r[0][2].is_infinite() && r[0][2] < 0.0);
        assert_eq!(r[0][3], 0.0); // block column always visible
        Ok(())
    }
}
