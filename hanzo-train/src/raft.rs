//! Retrieval-Aware Fine-Tuning (`RAFT`, arXiv:2403.10131, HIP-1346 / HIP-1347) for Zen on
//! `hanzo-train`.
//!
//! Trains a small Apache-2.0 / MIT eligible Zen checkpoint to:
//! 1. Exploit oracle retrieved passages (`D*`) and emit verifiable passage citations when `D*` is
//!    present among `k` shuffled candidate passages (fraction `P` of training instances), and
//! 2. Reject hard distractors (selecting the explicit `ABSTAIN` citation slot and relying on
//!    parametric domain weights rather than hallucinating citations) when `D*` is absent
//!    (fraction `1 - P` of training instances).
//!
//! Enforces Apache-2.0 / MIT base-model license verification before checkpoint export and strict
//! pre-batch multi-tenant access-control checks (`tenant_id` isolation is never relaxed).

use crate::adam::{Adam, Hyper};
use anyhow::{bail, ensure, Context, Result};
use hanzo_ml::{DType, Device, Tensor, D};
use hanzo_nn::{embedding, linear, loss, ops, Embedding, Linear, Module, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Instant;

/// SPDX license identifiers approved for production Hanzo Zen RAFT checkpoints.
pub const APPROVED_LICENSES: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "MIT OR Apache-2.0",
    "Apache-2.0 OR MIT",
];

/// Verifies that a base model or dataset carries an approved permissive license before training
/// or publishing a checkpoint.
pub fn verify_base_model_license(spdx_id: &str) -> Result<()> {
    let trimmed = spdx_id.trim();
    ensure!(!trimmed.is_empty(), "base model license must not be empty");
    if APPROVED_LICENSES
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(trimmed))
    {
        Ok(())
    } else {
        bail!(
            "license '{}' is not cleared for Zen production checkpoints; require Apache-2.0 or MIT",
            trimmed
        )
    }
}

/// One candidate passage returned by `hanzoai/index` retrieval.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Passage {
    pub doc_id: String,
    pub tenant_id: String,
    pub tokens: Vec<u32>,
    pub is_oracle: bool,
}

/// Raw domain QA training or evaluation record before RAFT `P` / `1 - P` mixing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainQaRecord {
    pub tenant_id: String,
    pub query_tokens: Vec<u32>,
    pub oracle_passage: Passage,
    pub distractor_passages: Vec<Passage>,
    pub answer_label: u32,
}

/// Enforces pre-retrieval / pre-batch tenant isolation: neither oracle nor distractor passages
/// may cross tenant boundaries.
pub fn ensure_tenant_isolation(query_tenant: &str, passages: &[Passage]) -> Result<()> {
    ensure!(
        !query_tenant.trim().is_empty(),
        "query tenant_id must not be empty"
    );
    for p in passages {
        if p.tenant_id != query_tenant {
            bail!(
                "tenant isolation violation: query tenant '{}' cannot access passage '{}' owned by '{}'",
                query_tenant,
                p.doc_id,
                p.tenant_id
            );
        }
    }
    Ok(())
}

/// One prepared RAFT training or evaluation sample with fixed `num_passages` (`k`) and `seq_len`.
///
/// `citation_target` lies in `0..=num_passages`:
/// - `0..num_passages`: index of the shuffled oracle passage `D*`,
/// - `num_passages`: explicit `ABSTAIN` (no supporting oracle passage present; all `k` are distractors).
#[derive(Clone, Debug)]
pub struct RaftSample {
    pub query_tokens: Vec<u32>,
    pub passage_tokens: Vec<Vec<u32>>,
    pub answer_label: u32,
    pub citation_target: u32,
    pub has_oracle: bool,
}

/// Hyperparameters and architectural dimensions for a bounded Zen RAFT model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaftConfig {
    pub base_model_license: String,
    pub vocab_size: usize,
    pub num_answers: usize,
    pub hidden_dim: usize,
    pub num_passages: usize,
    pub seq_len: usize,
    /// Fraction `P` in `[0.0, 1.0]` of training samples that retain the golden oracle passage `D*`.
    pub oracle_keep_prob: f64,
    /// Weight `lambda` on the passage-citation / distractor-rejection cross-entropy loss.
    pub citation_loss_weight: f64,
    pub learning_rate: f64,
    pub steps: usize,
    pub batch_size: usize,
}

impl RaftConfig {
    pub fn validate(&self) -> Result<()> {
        verify_base_model_license(&self.base_model_license)?;
        ensure!(self.vocab_size >= 8, "vocab_size must be >= 8");
        ensure!(self.num_answers >= 2, "num_answers must be >= 2");
        ensure!(self.hidden_dim >= 8, "hidden_dim must be >= 8");
        ensure!(self.num_passages >= 2, "num_passages (k) must be >= 2");
        ensure!(self.seq_len >= 2, "seq_len must be >= 2");
        ensure!(
            (0.0..=1.0).contains(&self.oracle_keep_prob),
            "oracle_keep_prob P must be in [0.0, 1.0], got {}",
            self.oracle_keep_prob
        );
        ensure!(
            self.citation_loss_weight >= 0.0,
            "citation_loss_weight must be >= 0.0"
        );
        ensure!(self.learning_rate > 0.0, "learning_rate must be > 0.0");
        ensure!(self.batch_size >= 1, "batch_size must be >= 1");
        Ok(())
    }
}

fn pad_or_truncate(tokens: &[u32], seq_len: usize, vocab_size: usize) -> Result<Vec<u32>> {
    let mut out = vec![0u32; seq_len];
    for (i, &tok) in tokens.iter().take(seq_len).enumerate() {
        ensure!(
            (tok as usize) < vocab_size,
            "token id {} exceeds vocab_size {}",
            tok,
            vocab_size
        );
        out[i] = tok;
    }
    Ok(out)
}

/// Builds a deterministic RAFT sample set applying the `P` oracle-present / `1 - P` distractor-only
/// mixture from Zhang et al. (arXiv:2403.10131).
pub fn prepare_raft_samples(
    records: &[DomainQaRecord],
    cfg: &RaftConfig,
    oracle_keep_prob: f64,
    seed: u64,
) -> Result<Vec<RaftSample>> {
    ensure!(
        (0.0..=1.0).contains(&oracle_keep_prob),
        "oracle_keep_prob must lie in [0.0, 1.0]"
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let k = cfg.num_passages;
    let mut samples = Vec::with_capacity(records.len());

    for rec in records {
        ensure!(
            (rec.answer_label as usize) < cfg.num_answers,
            "answer_label {} exceeds num_answers {}",
            rec.answer_label,
            cfg.num_answers
        );
        ensure!(
            rec.distractor_passages.len() >= k,
            "record needs at least {} distractor passages for distractor-only mode, got {}",
            k,
            rec.distractor_passages.len()
        );

        let mut all_passages = Vec::with_capacity(1 + rec.distractor_passages.len());
        all_passages.push(rec.oracle_passage.clone());
        all_passages.extend(rec.distractor_passages.iter().cloned());
        ensure_tenant_isolation(&rec.tenant_id, &all_passages)?;

        let q = pad_or_truncate(&rec.query_tokens, cfg.seq_len, cfg.vocab_size)?;
        let keep_oracle = rng.random::<f64>() < oracle_keep_prob;

        let mut chosen: Vec<(bool, Vec<u32>)> = Vec::with_capacity(k);
        let mut distractor_indices: Vec<usize> = (0..rec.distractor_passages.len()).collect();
        distractor_indices.shuffle(&mut rng);

        if keep_oracle {
            chosen.push((
                true,
                pad_or_truncate(&rec.oracle_passage.tokens, cfg.seq_len, cfg.vocab_size)?,
            ));
            for &di in distractor_indices.iter().take(k - 1) {
                chosen.push((
                    false,
                    pad_or_truncate(
                        &rec.distractor_passages[di].tokens,
                        cfg.seq_len,
                        cfg.vocab_size,
                    )?,
                ));
            }
            chosen.shuffle(&mut rng);
            let oracle_pos = chosen
                .iter()
                .position(|(is_orc, _)| *is_orc)
                .expect("oracle inserted") as u32;
            samples.push(RaftSample {
                query_tokens: q,
                passage_tokens: chosen.into_iter().map(|(_, t)| t).collect(),
                answer_label: rec.answer_label,
                citation_target: oracle_pos,
                has_oracle: true,
            });
        } else {
            for &di in distractor_indices.iter().take(k) {
                chosen.push((
                    false,
                    pad_or_truncate(
                        &rec.distractor_passages[di].tokens,
                        cfg.seq_len,
                        cfg.vocab_size,
                    )?,
                ));
            }
            samples.push(RaftSample {
                query_tokens: q,
                passage_tokens: chosen.into_iter().map(|(_, t)| t).collect(),
                answer_label: rec.answer_label,
                citation_target: k as u32,
                has_oracle: false,
            });
        }
    }

    Ok(samples)
}

/// Differentiable retrieval-aware Zen micro-architecture on `hanzo-ml` / `hanzo-nn`.
pub struct ZenRaftModel {
    embed: Embedding,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    abstain_proj: Linear,
    q_ans: Linear,
    ctx_ans: Linear,
    ans_head: Linear,
    hidden_dim: usize,
}

impl ZenRaftModel {
    pub fn new(vb: VarBuilder, cfg: &RaftConfig) -> hanzo_ml::Result<Self> {
        let h = cfg.hidden_dim;
        let embed = embedding(cfg.vocab_size, h, vb.pp("embed"))?;
        let q_proj = linear(h, h, vb.pp("q_proj"))?;
        let k_proj = linear(h, h, vb.pp("k_proj"))?;
        let v_proj = linear(h, h, vb.pp("v_proj"))?;
        let abstain_proj = linear(h, 1, vb.pp("abstain_proj"))?;
        let q_ans = linear(h, h, vb.pp("q_ans"))?;
        let ctx_ans = linear(h, h, vb.pp("ctx_ans"))?;
        let ans_head = linear(h, cfg.num_answers, vb.pp("ans_head"))?;
        Ok(Self {
            embed,
            q_proj,
            k_proj,
            v_proj,
            abstain_proj,
            q_ans,
            ctx_ans,
            ans_head,
            hidden_dim: h,
        })
    }

    /// Forward pass returning `(answer_logits [B, A], citation_logits [B, K + 1])`.
    pub fn forward(
        &self,
        queries: &Tensor,
        passages: &Tensor,
        use_context: bool,
    ) -> hanzo_ml::Result<(Tensor, Tensor)> {
        let (b, _seq_len) = queries.dims2()?;
        let (_, k, p_seq) = passages.dims3()?;
        let scale = 1.0 / (self.hidden_dim as f64).sqrt();

        let q_emb = self.embed.forward(queries)?.mean(1)?;
        let q_key = self.q_proj.forward(&q_emb)?;

        let flat_p = passages.reshape((b * k, p_seq))?;
        let p_emb_flat = self.embed.forward(&flat_p)?.mean(1)?;
        let p_keys = self
            .k_proj
            .forward(&p_emb_flat)?
            .reshape((b, k, self.hidden_dim))?;
        let p_vals = self
            .v_proj
            .forward(&p_emb_flat)?
            .reshape((b, k, self.hidden_dim))?;

        let q_exp = q_key.unsqueeze(1)?.broadcast_as((b, k, self.hidden_dim))?;
        let passage_scores = ((&p_keys * &q_exp)?.sum(2)? * scale)?;

        let max_p_score = passage_scores.max_keepdim(1)?;
        let base_abstain = self.abstain_proj.forward(&q_emb)?;
        let abstain_logit = (&base_abstain - &max_p_score)?;

        let citation_logits = Tensor::cat(&[&passage_scores, &abstain_logit], 1)?;

        let hidden = if use_context {
            let cite_probs = ops::softmax(&citation_logits, D::Minus1)?;
            let doc_probs = cite_probs.narrow(1, 0, k)?.unsqueeze(2)?;
            let doc_probs_b = doc_probs.broadcast_as((b, k, self.hidden_dim))?;
            let attended_ctx = (&p_vals * &doc_probs_b)?.sum(1)?;
            (self.q_ans.forward(&q_emb)? + self.ctx_ans.forward(&attended_ctx)?)?.silu()?
        } else {
            self.q_ans.forward(&q_emb)?.silu()?
        };

        let answer_logits = self.ans_head.forward(&hidden)?;
        Ok((answer_logits, citation_logits))
    }
}

/// Summary metrics for training cost, domain QA accuracy, citation support, and distractor rejection.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RaftEvalReport {
    pub overall_qa_accuracy: f64,
    pub oracle_present_qa_accuracy: f64,
    pub distractor_only_qa_accuracy: f64,
    pub citation_support_accuracy: f64,
    pub distractor_rejection_rate: f64,
    pub false_citation_rate: f64,
    pub param_count: usize,
    pub train_steps: usize,
    pub train_time_ms: f64,
    pub ms_per_step: f64,
    pub checkpoint_bytes: u64,
}

/// Rectangular device-resident tensor batch for `ZenRaftModel` training and evaluation.
///
/// Includes explicit `attention_mask` (`1` for non-pad tokens, `0` for `0` pad tokens) and
/// `prompt_loss_mask` (`0.0` across prompt/context input positions `[B, (1 + K) * S]` and `1.0` on
/// supervised citation + answer output positions `[B, 2]`) so autoregressive or encoder-decoder
/// Zen heads never backpropagate prompt-token reconstruction loss.
#[derive(Clone, Debug)]
pub struct PreparedRaftBatch {
    pub query_ids: Tensor,
    pub passage_ids: Tensor,
    pub attention_mask: Tensor,
    pub prompt_loss_mask: Tensor,
    pub answer_targets: Tensor,
    pub citation_targets: Tensor,
}

/// Collates a slice of [`RaftSample`] records into rectangular device-resident tensors for Zen
/// RAFT training and inference without stubs.
pub fn prepare_batch(
    samples: &[RaftSample],
    cfg: &RaftConfig,
    dev: &Device,
) -> Result<PreparedRaftBatch> {
    ensure!(
        !samples.is_empty(),
        "samples must not be empty in prepare_batch"
    );
    let indices: Vec<usize> = (0..samples.len()).collect();
    prepare_indexed_batch(samples, &indices, cfg, dev)
}

fn prepare_indexed_batch(
    samples: &[RaftSample],
    indices: &[usize],
    cfg: &RaftConfig,
    dev: &Device,
) -> Result<PreparedRaftBatch> {
    ensure!(!indices.is_empty(), "batch indices must not be empty");
    let b = indices.len();
    let k = cfg.num_passages;
    let s = cfg.seq_len;
    let ctx_tokens = (1 + k) * s;
    let total_seq = ctx_tokens + 2;

    let mut q_flat = Vec::with_capacity(b * s);
    let mut p_flat = Vec::with_capacity(b * k * s);
    let mut attn_flat = Vec::with_capacity(b * ctx_tokens);
    let mut loss_mask_flat = Vec::with_capacity(b * total_seq);
    let mut ans = Vec::with_capacity(b);
    let mut cite = Vec::with_capacity(b);

    for &idx in indices {
        ensure!(
            idx < samples.len(),
            "sample index {} out of bounds {}",
            idx,
            samples.len()
        );
        let item = &samples[idx];
        ensure!(
            item.query_tokens.len() == s,
            "query_tokens len {} != seq_len {}",
            item.query_tokens.len(),
            s
        );
        ensure!(
            item.passage_tokens.len() == k,
            "passage_tokens len {} != num_passages {}",
            item.passage_tokens.len(),
            k
        );
        q_flat.extend_from_slice(&item.query_tokens);
        for &tok in &item.query_tokens {
            attn_flat.push(u32::from(tok != 0));
        }
        for p in &item.passage_tokens {
            ensure!(p.len() == s, "passage len {} != seq_len {}", p.len(), s);
            p_flat.extend_from_slice(p);
            for &tok in p {
                attn_flat.push(u32::from(tok != 0));
            }
        }
        loss_mask_flat.extend(std::iter::repeat_n(0.0f32, ctx_tokens));
        loss_mask_flat.push(1.0f32);
        loss_mask_flat.push(1.0f32);
        ans.push(item.answer_label);
        cite.push(item.citation_target);
    }

    Ok(PreparedRaftBatch {
        query_ids: Tensor::from_vec(q_flat, (b, s), dev)?,
        passage_ids: Tensor::from_vec(p_flat, (b, k, s), dev)?,
        attention_mask: Tensor::from_vec(attn_flat, (b, ctx_tokens), dev)?,
        prompt_loss_mask: Tensor::from_vec(loss_mask_flat, (b, total_seq), dev)?,
        answer_targets: Tensor::from_vec(ans, b, dev)?,
        citation_targets: Tensor::from_vec(cite, b, dev)?,
    })
}

fn batch_tensors(
    samples: &[RaftSample],
    indices: &[usize],
    cfg: &RaftConfig,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let batch = prepare_indexed_batch(samples, indices, cfg, dev)?;
    Ok((
        batch.query_ids,
        batch.passage_ids,
        batch.answer_targets,
        batch.citation_targets,
    ))
}

/// Counts scalar parameters stored in `varmap`.
pub fn count_parameters(varmap: &VarMap) -> usize {
    let data = varmap.data().lock().expect("varmap lock");
    data.values().map(|v| v.as_tensor().elem_count()).sum()
}

/// Trains a `ZenRaftModel` via `hanzo_train::adam::Adam` and optionally saves a `.safetensors`
/// checkpoint after passing the license check.
pub fn train_zen_raft(
    train_samples: &[RaftSample],
    cfg: &RaftConfig,
    use_context: bool,
    checkpoint_out: Option<&Path>,
    seed: u64,
) -> Result<(VarMap, ZenRaftModel, f64, u64)> {
    cfg.validate()?;
    ensure!(!train_samples.is_empty(), "train_samples must not be empty");

    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ZenRaftModel::new(vb, cfg)?;

    let hyper = Hyper {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        decay: 0.01,
        clip: 1.0,
    };
    let mut opt = Adam::new(&varmap, |_| Some(cfg.learning_rate), hyper)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut order: Vec<usize> = (0..train_samples.len()).collect();
    let start = Instant::now();

    for step in 0..cfg.steps {
        if step.is_multiple_of(train_samples.len().div_ceil(cfg.batch_size).max(1)) {
            order.shuffle(&mut rng);
        }
        let batch_len = cfg.batch_size.min(train_samples.len());
        let offset = (step * batch_len) % train_samples.len();
        let mut batch_idx = Vec::with_capacity(batch_len);
        for i in 0..batch_len {
            batch_idx.push(order[(offset + i) % train_samples.len()]);
        }

        let (q_t, p_t, ans_t, cite_t) = batch_tensors(train_samples, &batch_idx, cfg, &dev)?;
        let (ans_logits, cite_logits) = model.forward(&q_t, &p_t, use_context)?;
        let ans_loss = loss::cross_entropy(&ans_logits, &ans_t)?;
        let total_loss = if use_context && cfg.citation_loss_weight > 0.0 {
            let cite_loss = loss::cross_entropy(&cite_logits, &cite_t)?;
            (ans_loss + (cite_loss * cfg.citation_loss_weight)?)?
        } else {
            ans_loss
        };

        let grads = total_loss.backward()?;
        opt.step(&grads)?;
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut ckpt_bytes = 0u64;
    if let Some(path) = checkpoint_out {
        verify_base_model_license(&cfg.base_model_license)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create checkpoint parent dir")?;
        }
        varmap.save(path).context("save safetensors checkpoint")?;
        ckpt_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }

    Ok((varmap, model, elapsed_ms, ckpt_bytes))
}

/// Evaluates domain QA accuracy, oracle citation accuracy, and distractor rejection rate.
pub fn evaluate_zen_raft(
    model: &ZenRaftModel,
    varmap: &VarMap,
    eval_samples: &[RaftSample],
    cfg: &RaftConfig,
    use_context: bool,
    train_time_ms: f64,
    checkpoint_bytes: u64,
) -> Result<RaftEvalReport> {
    ensure!(!eval_samples.is_empty(), "eval_samples must not be empty");
    let dev = Device::Cpu;
    let indices: Vec<usize> = (0..eval_samples.len()).collect();
    let (q_t, p_t, _, _) = batch_tensors(eval_samples, &indices, cfg, &dev)?;
    let (ans_logits, cite_logits) = model.forward(&q_t, &p_t, use_context)?;

    let pred_ans = ans_logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let pred_cite = cite_logits.argmax(D::Minus1)?.to_vec1::<u32>()?;

    let mut total_ok = 0usize;
    let mut oracle_n = 0usize;
    let mut oracle_qa_ok = 0usize;
    let mut oracle_cite_ok = 0usize;
    let mut distractor_n = 0usize;
    let mut distractor_qa_ok = 0usize;
    let mut distractor_reject_ok = 0usize;
    let mut false_citations = 0usize;

    let abstain_id = cfg.num_passages as u32;

    for (i, sample) in eval_samples.iter().enumerate() {
        let qa_hit = pred_ans[i] == sample.answer_label;
        if qa_hit {
            total_ok += 1;
        }
        if sample.has_oracle {
            oracle_n += 1;
            if qa_hit {
                oracle_qa_ok += 1;
            }
            if use_context && pred_cite[i] == sample.citation_target {
                oracle_cite_ok += 1;
            }
        } else {
            distractor_n += 1;
            if qa_hit {
                distractor_qa_ok += 1;
            }
            if !use_context || pred_cite[i] == abstain_id {
                distractor_reject_ok += 1;
            } else {
                false_citations += 1;
            }
        }
    }

    let n = eval_samples.len() as f64;
    let steps = cfg.steps.max(1);
    Ok(RaftEvalReport {
        overall_qa_accuracy: total_ok as f64 / n,
        oracle_present_qa_accuracy: if oracle_n > 0 {
            oracle_qa_ok as f64 / oracle_n as f64
        } else {
            0.0
        },
        distractor_only_qa_accuracy: if distractor_n > 0 {
            distractor_qa_ok as f64 / distractor_n as f64
        } else {
            0.0
        },
        citation_support_accuracy: if oracle_n > 0 && use_context {
            oracle_cite_ok as f64 / oracle_n as f64
        } else {
            0.0
        },
        distractor_rejection_rate: if distractor_n > 0 {
            distractor_reject_ok as f64 / distractor_n as f64
        } else {
            0.0
        },
        false_citation_rate: if distractor_n > 0 {
            false_citations as f64 / distractor_n as f64
        } else {
            0.0
        },
        param_count: count_parameters(varmap),
        train_steps: cfg.steps,
        train_time_ms,
        ms_per_step: train_time_ms / steps as f64,
        checkpoint_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn license_gate_accepts_apache2_and_mit_only() {
        assert!(verify_base_model_license("Apache-2.0").is_ok());
        assert!(verify_base_model_license("MIT").is_ok());
        assert!(verify_base_model_license("MIT OR Apache-2.0").is_ok());
        assert!(verify_base_model_license("CC-BY-NC-4.0").is_err());
        assert!(verify_base_model_license("GPL-3.0-only").is_err());
        assert!(verify_base_model_license("").is_err());
    }

    #[test]
    fn tenant_isolation_rejects_cross_tenant_passage() {
        let passages = vec![
            Passage {
                doc_id: "doc-1".into(),
                tenant_id: "org-a".into(),
                tokens: vec![1, 2, 3, 4],
                is_oracle: true,
            },
            Passage {
                doc_id: "doc-2".into(),
                tenant_id: "org-b".into(),
                tokens: vec![5, 6, 7, 8],
                is_oracle: false,
            },
        ];
        let err = ensure_tenant_isolation("org-a", &passages).unwrap_err();
        assert!(err.to_string().contains("tenant isolation violation"));
    }
}
