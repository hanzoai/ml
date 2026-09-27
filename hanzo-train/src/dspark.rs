//! DSpark on the cluster runtime ([`crate::cluster`]): a draft trains on the DeepSpec target
//! cache on one machine or many, and its checkpoint is written in the exact layout the engine's
//! `qwen3_dspark.rs` loader expects.
//!
//! - [`Spec`]: the run, as the coordinator sends it to every worker: the model's shape, the
//!   plan (every step's samples, drawn up front from `seed`) and the optimizer (hanzo-nn's AdamW
//!   defaults, unclipped, at `lr`; `warmup` and `cool` shape the schedule).
//! - [`Draft`]: a worker's model. A step draws each sample's anchors from a stream fixed by the
//!   seed, the step and the sample, so every worker (and a replay) draws alike.
//! - [`Engine`]: the coordinator's side: each worker's loss per round, and checkpoints.
//! - [`run`], [`load`]: what `hanzo-train fit` serves and trains in, and what `hanzo-train join`
//!   loads, its cache checked against the coordinator's by digest.

use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use clap::Args;
use hanzo_ml::{Device, Tensor};
use hanzo_nn::VarMap;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::Cache;
use crate::cluster::{
    span, Adam, Batch, Closed, Hyper, Keep, Mark, Model, Params, Plan, Round, Run, Setup, Start,
    Step,
};
use crate::model::{verify_checkpoint, Dspark, DsparkCfg};
/// This machine's copies of the data the run reads.
#[derive(Args, Clone, Debug)]
pub struct Local {
    /// DeepSpec target-cache v2 directory (manifest.json + samples.idx + shard-*.bin).
    #[arg(long)]
    pub cache_dir: PathBuf,
    /// Frozen embed_tokens + lm_head init safetensors (keys embed_tokens.weight, lm_head.weight).
    #[arg(
        long,
        default_value = "/home/z/work/zen/hf/v4-dspark-init/embed_head.safetensors"
    )]
    pub init: PathBuf,
}

/// The run: the model's shape, its batches and its optimizer, as the coordinator sends it.
#[derive(Args, Clone, Debug, Serialize, Deserialize)]
pub struct Spec {
    /// Optimizer steps: batches in the plan.
    #[arg(long, default_value_t = 100)]
    pub steps: usize,
    /// Decoder layers (MVP default 2; full model uses 5).
    #[arg(long, default_value_t = 2)]
    pub layers: usize,
    #[arg(long, default_value_t = 4096)]
    pub intermediate: usize,
    #[arg(long, default_value_t = 32)]
    pub heads: usize,
    #[arg(long, default_value_t = 8)]
    pub kv_heads: usize,
    #[arg(long, default_value_t = 128)]
    pub head_dim: usize,
    #[arg(long, default_value_t = 129280)]
    pub vocab: usize,
    #[arg(long, default_value_t = 256)]
    pub markov_rank: usize,
    #[arg(long, default_value_t = 7)]
    pub block: usize,
    #[arg(long, default_value_t = 129279)]
    pub mask_token_id: u32,
    /// Anchors sampled per training sample.
    #[arg(long, default_value_t = 4)]
    pub num_anchors: usize,
    /// Samples per optimizer step.
    #[arg(long, default_value_t = 4)]
    pub micro_batch: usize,
    #[arg(long, default_value_t = 6e-4)]
    pub lr: f64,
    /// Share of steps warming up; with `cool` 0 as well, every step is at `lr`.
    #[arg(long, default_value_t = 0.0)]
    pub warmup: f64,
    /// Share of steps in the final decay to a tenth of `lr`.
    #[arg(long, default_value_t = 0.0)]
    pub cool: f64,
    /// Init scale for the output-side RMSNorm (see DsparkCfg::final_norm_init). `1.0` = faithful;
    /// a small value (e.g. 0.1) starts the loss near ln(vocab) and well-conditions training.
    #[arg(long, default_value_t = 0.1)]
    pub final_norm_init: f64,
    /// Only train on samples with seq_len <= this (0 = no cap). Bounds per-step cost/memory and
    /// keeps sequence lengths uniform for a smoother curve. The `fc` fuse dominates cost on long seqs.
    #[arg(long, default_value_t = 0)]
    pub max_seq: usize,
    /// Draws the plan, each sample's anchors and the initial weights.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Freeze the vocab×rank Markov head (exclude it from AdamW + detach its bias). Drops ~0.8GB of
    /// optimizer/grad memory; use on memory-constrained hosts. The head is still saved at its init.
    #[arg(long, default_value_t = false)]
    pub freeze_markov: bool,
    /// Overfit a single fixed batch (same samples+anchors every step). Removes sample-to-sample
    /// variance for a clean monotonic curve — the standard proof that the training loop learns.
    #[arg(long, default_value_t = false)]
    pub fixed_batch: bool,
    /// Seconds of local training per round; 0 is one step a round.
    #[arg(long, default_value_t = 0.0)]
    pub round: f64,
    /// Seconds past a round's deadline a worker's report is waited for; then the worker is
    /// dropped, its steps requeued.
    #[arg(long, default_value_t = 60.0)]
    pub grace: f64,
    /// The outer step's rate: at 1 without momentum, one step a round is plain training up to
    /// the bf16 exchange.
    #[arg(long, default_value_t = 1.0)]
    pub outer: f64,
    /// Outer Nesterov momentum.
    #[arg(long, default_value_t = 0.0)]
    pub momentum: f64,
    /// A checkpoint every this many steps, and at the end; 0 only at the end.
    #[arg(long, default_value_t = 0)]
    pub every: usize,
}

impl Spec {
    /// The model this spec builds over `cache`.
    pub fn cfg(&self, cache: &Cache) -> DsparkCfg {
        // RoPE table must cover the longest sequence (+ block) in the cache.
        let longest = cache
            .records
            .iter()
            .map(|r| r.seq_len as usize)
            .max()
            .unwrap_or(0);
        DsparkCfg {
            vocab: self.vocab,
            hidden: cache.manifest.hidden_size,
            intermediate: self.intermediate,
            layers: self.layers,
            heads: self.heads,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            rms_eps: 1e-6,
            rope_theta: 1e6,
            max_pos: longest + self.block + 1,
            block: self.block,
            mask_token_id: self.mask_token_id,
            markov_rank: self.markov_rank,
            target_layer_ids: cache.manifest.target_layer_ids.clone(),
            final_norm_init: self.final_norm_init,
        }
    }

    /// hanzo-nn's AdamW defaults, unclipped.
    pub fn adam(&self) -> Hyper {
        Hyper {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            decay: 0.01,
            clip: f64::MAX,
        }
    }

    /// Every step's samples: `micro_batch` drawn from the samples long enough to host a full
    /// block (and within `max_seq`), or under `fixed_batch` the first draw every step.
    pub fn plan(&self, cache: &Cache) -> Result<Plan> {
        let eligible: Vec<usize> = (0..cache.len())
            .filter(|&i| {
                let s = cache.records[i].seq_len as usize;
                s >= self.block + 2 && (self.max_seq == 0 || s <= self.max_seq)
            })
            .collect();
        ensure!(
            !eligible.is_empty(),
            "no eligible samples (seq_len in [{}, {}])",
            self.block + 2,
            self.max_seq
        );
        println!(
            "eligible samples: {}/{} (max_seq={})",
            eligible.len(),
            cache.len(),
            self.max_seq
        );
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut draw = || -> Vec<usize> {
            (0..self.micro_batch)
                .map(|_| eligible[rng.random_range(0..eligible.len())])
                .collect()
        };
        let batches = if self.fixed_batch {
            vec![draw(); self.steps]
        } else {
            (0..self.steps).map(|_| draw()).collect()
        };
        let (warm, cool) = span(self.steps, self.warmup, self.cool);
        Ok(Plan {
            batches,
            ends: vec![self.steps],
            warm,
            cool,
        })
    }
}

/// SHA-256 of the cache's manifest and index: what a worker's copy must hash to.
pub fn digest(dir: &Path) -> Result<String> {
    let mut h = Sha256::new();
    for f in ["manifest.json", "samples.idx"] {
        h.update(std::fs::read(dir.join(f)).with_context(|| format!("{}", dir.join(f).display()))?);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn pick_anchors(
    loss_mask: &[u8],
    seq: usize,
    block: usize,
    n: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    // Candidate anchors: loss_mask[a] && loss_mask[a+1], and a+block <= seq-1 so all block targets fit.
    if seq < block + 2 {
        return Vec::new();
    }
    let mut cands: Vec<usize> = (1..=seq - block - 1)
        .filter(|&a| loss_mask[a] != 0 && loss_mask[a + 1] != 0)
        .collect();
    let take = n.min(cands.len());
    // Partial Fisher-Yates for `take` uniform picks without replacement.
    for i in 0..take {
        let j = i + rng.random_range(0..(cands.len() - i));
        cands.swap(i, j);
    }
    cands.truncate(take);
    cands
}

/// A DSpark draft in a run: the model, the cache it reads, the run's spec.
pub struct Draft {
    pub model: Dspark,
    pub cache: Cache,
    pub spec: Spec,
}

impl Draft {
    pub fn new(spec: &Spec, local: &Local) -> Result<Draft> {
        let cache = Cache::open(&local.cache_dir)?;
        let cfg = spec.cfg(&cache);
        ensure!(
            cfg.heads * cfg.head_dim == cfg.hidden,
            "heads*head_dim must equal hidden"
        );
        let model = Dspark::new(
            cfg,
            &local.init,
            &Device::Cpu,
            !spec.freeze_markov,
            spec.seed,
        )?;
        Ok(Draft {
            model,
            cache,
            spec: spec.clone(),
        })
    }
}

impl Model for Draft {
    fn vars(&self) -> &VarMap {
        self.model.vars()
    }

    fn rate(&self, name: &str) -> Option<f64> {
        (self.model.trains_markov() || !name.starts_with("markov_head.")).then_some(self.spec.lr)
    }

    /// Every sample's anchors, drawn from a stream fixed by the seed, the step (the first under
    /// `fixed_batch`) and the sample; cross-entropy through the frozen head over all of them.
    fn step(&mut self, adam: &mut Adam, batch: &Batch, _: &Round) -> Result<Step> {
        let (cfg, dev) = (self.model.cfg(), Device::Cpu);
        let fused_width = self.cache.n_fused * cfg.hidden;
        let key = if self.spec.fixed_batch {
            0
        } else {
            batch.id as u64 + 1
        };
        let (mut blocks, mut biases, mut targets) = (Vec::new(), Vec::new(), Vec::new());
        for &si in &batch.rows {
            let mut s = self.cache.read_sample(si)?;
            let mut rng = StdRng::seed_from_u64(
                self.spec.seed ^ key.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ ((si as u64) << 32),
            );
            let anchors = pick_anchors(
                &s.loss_mask,
                s.seq_len,
                cfg.block,
                self.spec.num_anchors,
                &mut rng,
            );
            if anchors.is_empty() {
                continue;
            }
            let th = Tensor::from_vec(
                std::mem::take(&mut s.target_hidden),
                (s.seq_len, fused_width),
                &dev,
            )?;
            let fused = self.model.fuse(&th)?;
            for a in anchors {
                if let Some((block, bias, t)) =
                    self.model
                        .draft_anchor(&fused, &s.input_ids, &s.loss_mask, a)?
                {
                    targets.extend_from_slice(&t);
                    blocks.push(block);
                    biases.push(bias);
                }
            }
        }
        if targets.is_empty() {
            return Ok(Step {
                tokens: 0,
                sums: vec![0.0],
            });
        }
        let block = Tensor::cat(&blocks.iter().collect::<Vec<_>>(), 0)?; // [N, hidden]
        let bias = Tensor::cat(&biases.iter().collect::<Vec<_>>(), 0)?; // [N, vocab]
        let n = targets.len();
        let tgt = Tensor::from_vec(targets, n, &dev)?;
        // CE through the FROZEN head via surrogate backward (no [hidden, vocab] head grad formed).
        let (loss, surrogate) = self.model.head_ce(&block, &bias, &tgt)?;
        adam.step(&surrogate.backward()?)?;
        Ok(Step {
            tokens: n as u64,
            sums: vec![loss.to_scalar::<f32>()? as f64],
        })
    }

    fn line(&self, sums: &[f64], steps: u64, _: f64) -> String {
        format!(" loss {:.4}", sums[0] / steps.max(1) as f64)
    }
}

/// The coordinator's side: each worker's loss, and checkpoints in the engine's layout.
pub struct Engine {
    pub spec: Spec,
    pub local: Local,
}

impl Keep for Engine {
    fn round(&mut self, closed: &Closed, _: &Round) -> Result<(String, serde_json::Value)> {
        let mut line = String::new();
        let mut workers = Vec::new();
        for w in closed.works {
            let loss = w.sums.first().copied().unwrap_or(0.0) / w.steps.max(1) as f64;
            let tps = w.tokens as f64 / w.secs.max(1e-9);
            line += &format!(
                " | {} {} steps {tps:.0} tok/s loss {loss:.4}",
                w.name, w.steps
            );
            workers.push(serde_json::json!({"worker": w.name, "steps": w.steps,
                "tokens": w.tokens, "tokens_per_s": tps, "loss": loss}));
        }
        Ok((line, serde_json::json!({ "workers": workers })))
    }

    /// θ_global into a model built as every worker's is, written in the engine's layout and
    /// read back against it.
    fn save(&mut self, dir: &Path, _: &Mark, at: &Round, _: &serde_json::Value) -> Result<()> {
        let d = Draft::new(&self.spec, &self.local)?;
        at.params.write(d.model.vars(), at.theta)?;
        d.model.save(dir)?;
        let ckpt = dir.join("model.safetensors");
        verify_checkpoint(&ckpt, d.model.cfg())?;
        println!(
            "saved checkpoint -> {}; every engine key present with matching shape",
            ckpt.display()
        );
        Ok(())
    }
}

/// The build a coordinator and its workers must share.
pub const CODE: &str = env!("CARGO_PKG_VERSION");

/// The run of `spec` over this machine's `local` data as its coordinator serves it, checkpoints
/// into `out`, from a fresh draft; and that draft, this machine's worker.
pub fn run(spec: &Spec, local: &Local, out: Option<PathBuf>) -> Result<(Run, Start, Draft)> {
    let draft = Draft::new(spec, local)?;
    let params = Params::of(draft.vars(), |n| draft.rate(n).is_some());
    let plan = spec.plan(&draft.cache)?;
    let begin = Start::new(params.read(draft.vars())?, plan.total());
    let run = Run {
        code: CODE.into(),
        body: serde_json::json!({"spec": spec, "cache": digest(&local.cache_dir)?}),
        limit: plan.total(),
        adam: spec.adam(),
        outer: spec.outer,
        momentum: spec.momentum,
        round: spec.round,
        grace: spec.grace,
        every: match spec.every {
            0 => usize::MAX,
            n => n,
        },
        out,
        params,
        plan,
    };
    Ok((run, begin, draft))
}

/// The draft a coordinator's `setup` describes, over this machine's `local` data, once its cache
/// hashes as the coordinator's.
pub fn load(setup: &Setup, local: &Local) -> Result<Draft> {
    let spec: Spec = serde_json::from_value(setup.body["spec"].clone())?;
    let cache = digest(&local.cache_dir)?;
    if setup.body["cache"] != cache.as_str() {
        bail!("cache {cache}, the coordinator's {}", setup.body["cache"]);
    }
    Draft::new(&spec, local)
}
