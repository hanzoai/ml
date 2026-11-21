//! RAFT: Retrieval-Aware Zen Fine-Tuning Dataset Builder (`HIP-1346`).
//!
//! Clean-room implementation of Retrieval-Augmented Fine-Tuning (RAFT,
//! Zhang et al., 2024, <https://arxiv.org/abs/2403.10131>) for training
//! small eligible Zen checkpoints to cite supporting oracle evidence while
//! rejecting hard-negative distractors:
//!
//! 1. **Zen License & Eligibility Gate ([`ZenEligibility`])**:
//!    Validates that both the target Zen base model and the domain QA corpus
//!    carry permissive licenses (e.g. `Apache-2.0`, `MIT`, `BSD-3-Clause`,
//!    `CC-BY-4.0`) before generating fine-tuning artifacts.
//! 2. **BM25 Hard-Negative Retriever ([`PassageCorpus`])**:
//!    Indexes domain passages and selects top lexical hard-negative
//!    distractors for each question while filtering out all oracle IDs and
//!    exact/normalized content duplicates so oracle text never leaks into
//!    distractor slots.
//! 3. **Configurable Oracle-Dropout `P` ([`RaftConfig::oracle_dropout_prob`])**:
//!    With probability `P`, all oracle passages are dropped and the context
//!    contains only `k` hard-negative distractors (teaching closed-book domain
//!    recall or calibrated abstention without hallucinated citations). With
//!    probability `1 - P`, oracle passages are mixed with hard negatives,
//!    shuffled deterministically, and paired with verbatim
//!    `##begin_quote## ... ##end_quote##` `[Doc i]` citations.

use rand::seq::SliceRandom;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use thiserror::Error;

/// Errors returned during RAFT dataset construction or eligibility checks.
#[derive(Debug, Error)]
pub enum RaftError {
    #[error("invalid RAFT configuration: {0}")]
    InvalidConfig(String),
    #[error("ineligible license for Zen fine-tuning: {0}")]
    IneligibleLicense(String),
    #[error("invalid QA sample '{sample_id}': {reason}")]
    InvalidSample { sample_id: String, reason: String },
    #[error("duplicate passage id '{0}'")]
    DuplicatePassage(String),
}

pub type Result<T> = std::result::Result<T, RaftError>;

/// License classification for base checkpoints and domain QA corpora.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LicenseKind {
    Apache2,
    Mit,
    Bsd2Clause,
    Bsd3Clause,
    CcBy4,
    Cc0,
    PermissiveCustom(String),
    CcByNc,
    Gpl,
    Agpl,
    Elv2,
    ProprietaryRestricted,
    Unknown,
}

impl LicenseKind {
    /// Returns `true` only when the license allows commercial/fine-tuning Zen artifact release.
    pub fn is_zen_eligible(&self) -> bool {
        matches!(
            self,
            Self::Apache2
                | Self::Mit
                | Self::Bsd2Clause
                | Self::Bsd3Clause
                | Self::CcBy4
                | Self::Cc0
                | Self::PermissiveCustom(_)
        )
    }

    /// Canonical SPDX-style label.
    pub fn label(&self) -> &str {
        match self {
            Self::Apache2 => "Apache-2.0",
            Self::Mit => "MIT",
            Self::Bsd2Clause => "BSD-2-Clause",
            Self::Bsd3Clause => "BSD-3-Clause",
            Self::CcBy4 => "CC-BY-4.0",
            Self::Cc0 => "CC0-1.0",
            Self::PermissiveCustom(name) => name.as_str(),
            Self::CcByNc => "CC-BY-NC",
            Self::Gpl => "GPL",
            Self::Agpl => "AGPL",
            Self::Elv2 => "ELv2",
            Self::ProprietaryRestricted => "Proprietary-Restricted",
            Self::Unknown => "Unknown",
        }
    }
}

/// Provenance and license verification policy for Zen fine-tuning runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZenEligibility {
    pub base_model_id: String,
    pub base_model_license: LicenseKind,
    pub dataset_name: String,
    pub dataset_license: LicenseKind,
}

impl ZenEligibility {
    /// Verify both base model and corpus carry permissive Zen-eligible licenses.
    pub fn verify(&self) -> Result<()> {
        if self.base_model_id.trim().is_empty() {
            return Err(RaftError::InvalidConfig(
                "base_model_id must not be empty".to_string(),
            ));
        }
        if self.dataset_name.trim().is_empty() {
            return Err(RaftError::InvalidConfig(
                "dataset_name must not be empty".to_string(),
            ));
        }
        if !self.base_model_license.is_zen_eligible() {
            return Err(RaftError::IneligibleLicense(format!(
                "base model '{}' has non-permissive license '{}'",
                self.base_model_id,
                self.base_model_license.label()
            )));
        }
        if !self.dataset_license.is_zen_eligible() {
            return Err(RaftError::IneligibleLicense(format!(
                "dataset '{}' has non-permissive license '{}'",
                self.dataset_name,
                self.dataset_license.label()
            )));
        }
        Ok(())
    }
}

/// A corpus passage eligible to act as an oracle document or distractor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Passage {
    pub id: String,
    pub title: Option<String>,
    pub text: String,
    pub source_doc: Option<String>,
}

/// A distractor scored against a question via BM25 lexical similarity.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredDistractor {
    pub passage_index: usize,
    pub passage_id: String,
    pub bm25_score: f64,
}

/// BM25-indexed passage corpus for hard-negative distractor mining.
#[derive(Debug, Clone)]
pub struct PassageCorpus {
    passages: Vec<Passage>,
    id_to_index: HashMap<String, usize>,
    fingerprints: Vec<u64>,
    doc_lens: Vec<usize>,
    avg_doc_len: f64,
    term_freqs: Vec<HashMap<String, usize>>,
    doc_freq: HashMap<String, usize>,
}

impl PassageCorpus {
    /// Build a BM25 index over `passages`.
    pub fn new(passages: Vec<Passage>) -> Result<Self> {
        let mut id_to_index = HashMap::with_capacity(passages.len());
        let mut fingerprints = Vec::with_capacity(passages.len());
        let mut doc_lens = Vec::with_capacity(passages.len());
        let mut term_freqs = Vec::with_capacity(passages.len());
        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;

        for (idx, p) in passages.iter().enumerate() {
            if p.id.is_empty() {
                return Err(RaftError::InvalidConfig(format!(
                    "passage at index {idx} has empty id"
                )));
            }
            if id_to_index.insert(p.id.clone(), idx).is_some() {
                return Err(RaftError::DuplicatePassage(p.id.clone()));
            }
            fingerprints.push(normalized_fingerprint(&p.text));
            let tokens = tokenize(&p.text);
            total_len += tokens.len();
            doc_lens.push(tokens.len());

            let mut tf: HashMap<String, usize> = HashMap::new();
            for tok in tokens {
                *tf.entry(tok).or_insert(0) += 1;
            }
            for term in tf.keys() {
                *doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
            term_freqs.push(tf);
        }

        let avg_doc_len = if passages.is_empty() {
            1.0
        } else {
            (total_len as f64 / passages.len() as f64).max(1.0)
        };

        Ok(Self {
            passages,
            id_to_index,
            fingerprints,
            doc_lens,
            avg_doc_len,
            term_freqs,
            doc_freq,
        })
    }

    /// Number of indexed passages.
    pub fn len(&self) -> usize {
        self.passages.len()
    }

    /// Whether the corpus is empty.
    pub fn is_empty(&self) -> bool {
        self.passages.is_empty()
    }

    /// Look up a passage by ID.
    pub fn get_by_id(&self, id: &str) -> Option<(usize, &Passage)> {
        self.id_to_index
            .get(id)
            .map(|&idx| (idx, &self.passages[idx]))
    }

    /// Return up to `limit` hard-negative distractors ranked by descending BM25 score
    /// against `query`, excluding all `oracle_ids` and any passage whose normalized text
    /// matches an oracle passage.
    pub fn hard_negatives(
        &self,
        query: &str,
        oracle_ids: &[String],
        limit: usize,
    ) -> Vec<ScoredDistractor> {
        if limit == 0 || self.passages.is_empty() {
            return Vec::new();
        }
        let oracle_set: HashSet<&str> = oracle_ids.iter().map(String::as_str).collect();
        let oracle_fps: HashSet<u64> = oracle_ids
            .iter()
            .filter_map(|id| self.id_to_index.get(id))
            .map(|&idx| self.fingerprints[idx])
            .collect();

        let q_terms = tokenize(query);
        let n_docs = self.passages.len() as f64;
        const K1: f64 = 1.5;
        const B: f64 = 0.75;

        let mut scored = Vec::new();
        for (idx, p) in self.passages.iter().enumerate() {
            if oracle_set.contains(p.id.as_str()) || oracle_fps.contains(&self.fingerprints[idx]) {
                continue;
            }
            let dl = self.doc_lens[idx] as f64;
            let tf_map = &self.term_freqs[idx];
            let mut score = 0.0;
            for term in &q_terms {
                let Some(&tf) = tf_map.get(term) else {
                    continue;
                };
                let df = self.doc_freq.get(term).copied().unwrap_or(0) as f64;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln();
                let tf_f = tf as f64;
                let norm =
                    tf_f * (K1 + 1.0) / (tf_f + K1 * (1.0 - B + B * (dl / self.avg_doc_len)));
                score += idf * norm;
            }
            scored.push(ScoredDistractor {
                passage_index: idx,
                passage_id: p.id.clone(),
                bm25_score: score,
            });
        }

        scored.sort_by(|a, b| {
            b.bm25_score
                .total_cmp(&a.bm25_score)
                .then_with(|| a.passage_id.cmp(&b.passage_id))
        });
        scored.truncate(limit);
        scored
    }
}

/// Assistant target behavior when oracle passages are dropped (`oracle_kept == false`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropOracleMode {
    /// Model answers from parametric domain knowledge with reasoning and no fake `[Doc i]` citations.
    ClosedBookReasoned,
    /// Model explicitly abstains when the provided passages do not support the answer.
    AbstainWhenUnsupported,
}

/// Configuration for the RAFT distractor-aware example builder.
#[derive(Debug, Clone, PartialEq)]
pub struct RaftConfig {
    /// Probability `P` in `[0.0, 1.0]` of dropping ALL oracle passages from an example's
    /// context (leaving only hard-negative distractors). `keep_prob = 1.0 - oracle_dropout_prob`.
    pub oracle_dropout_prob: f64,
    /// Total passages `k` placed into each example's context window.
    pub num_context_docs: usize,
    /// Number of top BM25 hard-negative candidates to sample distractors from (`>= num_context_docs`).
    pub hard_negative_pool: usize,
    /// Deterministic PRNG seed for dropout decisions, distractor draws, and passage shuffling.
    pub seed: u64,
    /// Target behavior when oracle passages are dropped.
    pub drop_oracle_mode: DropOracleMode,
    /// Optional custom system prompt for Zen chat fine-tuning records.
    pub system_prompt: String,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            oracle_dropout_prob: 0.2,
            num_context_docs: 5,
            hard_negative_pool: 16,
            seed: 42,
            drop_oracle_mode: DropOracleMode::ClosedBookReasoned,
            system_prompt: "You are a retrieval-grounded Zen assistant. Cite verbatim quotes \
                            using ##begin_quote## ... ##end_quote## and [Doc i] only when \
                            directly supported by the provided passages."
                .to_string(),
        }
    }
}

impl RaftConfig {
    /// Convenience constructor expressed in terms of the RAFT paper's oracle retention
    /// probability `keep_prob` (`oracle_dropout_prob = 1.0 - keep_prob`).
    pub fn with_oracle_keep_prob(keep_prob: f64) -> Result<Self> {
        if !keep_prob.is_finite() || !(0.0..=1.0).contains(&keep_prob) {
            return Err(RaftError::InvalidConfig(format!(
                "keep_prob must be in [0.0, 1.0], got {keep_prob}"
            )));
        }
        Ok(Self {
            oracle_dropout_prob: 1.0 - keep_prob,
            ..Self::default()
        })
    }

    /// Validate all configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if !self.oracle_dropout_prob.is_finite() || !(0.0..=1.0).contains(&self.oracle_dropout_prob)
        {
            return Err(RaftError::InvalidConfig(format!(
                "oracle_dropout_prob P must be in [0.0, 1.0], got {}",
                self.oracle_dropout_prob
            )));
        }
        if self.num_context_docs == 0 {
            return Err(RaftError::InvalidConfig(
                "num_context_docs must be > 0".to_string(),
            ));
        }
        if self.hard_negative_pool < self.num_context_docs {
            return Err(RaftError::InvalidConfig(format!(
                "hard_negative_pool ({}) must be >= num_context_docs ({})",
                self.hard_negative_pool, self.num_context_docs
            )));
        }
        Ok(())
    }
}

/// Input QA annotation referencing one or more oracle passage IDs in [`PassageCorpus`].
#[derive(Debug, Clone)]
pub struct QaSample {
    pub id: String,
    pub question: String,
    pub answer: String,
    /// Optional chain-of-thought rationale explaining how evidence yields `answer`.
    pub reasoning: Option<String>,
    /// Passage IDs in [`PassageCorpus`] that support `answer`.
    pub oracle_ids: Vec<String>,
    /// Optional verbatim evidence quotes (`quote` must appear inside an oracle passage).
    /// When empty, a representative verbatim span is extracted from the first oracle passage.
    pub oracle_quotes: Vec<String>,
}

/// One passage as placed in a generated RAFT prompt after shuffling.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextPassage {
    /// 1-based position in the prompt (`[Doc 1]`, `[Doc 2]`, ...).
    pub rank_in_prompt: usize,
    pub passage_id: String,
    pub title: Option<String>,
    pub text: String,
    pub is_oracle: bool,
    pub bm25_score: f64,
}

/// Verified verbatim quote citation linking back to a shuffled prompt passage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    pub doc_rank: usize,
    pub passage_id: String,
    pub quote: String,
}

/// One generated distractor-aware RAFT fine-tuning example.
#[derive(Debug, Clone, PartialEq)]
pub struct RaftExample {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub oracle_kept: bool,
    pub oracle_positions: Vec<usize>,
    pub distractor_count: usize,
    pub context_passages: Vec<ContextPassage>,
    pub citations: Vec<Citation>,
    pub system_prompt: String,
    pub user_prompt: String,
    pub assistant_completion: String,
}

impl RaftExample {
    /// Serialize this training example to one Zen chat-SFT JSONL line.
    pub fn to_jsonl_line(&self, base_model_id: &str) -> String {
        let mut out = String::with_capacity(512);
        out.push('{');
        push_kv_str(&mut out, "id", &self.id, true);
        push_kv_str(&mut out, "base_model", base_model_id, false);
        let _ = write!(out, ",\"oracle_kept\":{}", self.oracle_kept);
        let _ = write!(out, ",\"distractor_count\":{}", self.distractor_count);
        let _ = write!(out, ",\"citation_count\":{}", self.citations.len());
        out.push_str(",\"messages\":[");
        push_message(&mut out, "system", &self.system_prompt, true);
        push_message(&mut out, "user", &self.user_prompt, false);
        push_message(&mut out, "assistant", &self.assistant_completion, false);
        out.push_str("]}");
        out
    }
}

/// Aggregate quality, dropout, distractor-hardness, and citation metrics for a built dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct RaftDatasetStats {
    pub total_examples: usize,
    pub oracle_kept_count: usize,
    pub oracle_dropped_count: usize,
    pub configured_dropout_prob: f64,
    pub empirical_dropout_rate: f64,
    pub mean_context_docs: f64,
    pub mean_distractors_per_example: f64,
    pub mean_hard_negative_bm25: f64,
    pub citation_support_rate: f64,
    pub oracle_leak_count: usize,
}

/// Built RAFT fine-tuning dataset with verified eligibility and dataset-level metrics.
#[derive(Debug, Clone)]
pub struct RaftDataset {
    pub eligibility: ZenEligibility,
    pub config: RaftConfig,
    pub examples: Vec<RaftExample>,
    pub stats: RaftDatasetStats,
}

impl RaftDataset {
    /// Export every example as newline-delimited JSONL for Zen fine-tuning.
    pub fn to_jsonl(&self) -> String {
        let mut buf = String::new();
        for ex in &self.examples {
            buf.push_str(&ex.to_jsonl_line(&self.eligibility.base_model_id));
            buf.push('\n');
        }
        buf
    }
}

/// Builder constructing distractor-aware RAFT training examples under configurable
/// oracle-dropout probability `P`.
pub struct RaftDatasetBuilder {
    eligibility: ZenEligibility,
    config: RaftConfig,
}

impl RaftDatasetBuilder {
    /// Create a validated RAFT dataset builder after enforcing Zen license eligibility.
    pub fn new(eligibility: ZenEligibility, config: RaftConfig) -> Result<Self> {
        eligibility.verify()?;
        config.validate()?;
        Ok(Self {
            eligibility,
            config,
        })
    }

    /// Build a complete [`RaftDataset`] from `corpus` and `samples`.
    pub fn build(&self, corpus: &PassageCorpus, samples: &[QaSample]) -> Result<RaftDataset> {
        if samples.is_empty() {
            return Err(RaftError::InvalidConfig(
                "samples must not be empty".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut examples = Vec::with_capacity(samples.len());

        let mut oracle_kept_count = 0usize;
        let mut oracle_dropped_count = 0usize;
        let mut total_ctx_docs = 0usize;
        let mut total_distractors = 0usize;
        let mut distractor_bm25_sum = 0.0f64;
        let mut supported_citations = 0usize;
        let mut oracle_leak_count = 0usize;

        for sample in samples {
            if sample.id.trim().is_empty() || sample.question.trim().is_empty() {
                return Err(RaftError::InvalidSample {
                    sample_id: sample.id.clone(),
                    reason: "id and question must not be empty".to_string(),
                });
            }
            if sample.oracle_ids.is_empty() {
                return Err(RaftError::InvalidSample {
                    sample_id: sample.id.clone(),
                    reason: "oracle_ids must contain at least one passage id".to_string(),
                });
            }

            // Resolve all oracle passages up front and verify they exist in the corpus.
            let mut oracles: Vec<(&Passage, f64)> = Vec::with_capacity(sample.oracle_ids.len());
            let mut seen_oracle_ids = HashSet::new();
            for oid in &sample.oracle_ids {
                if !seen_oracle_ids.insert(oid.as_str()) {
                    continue;
                }
                let Some((_, p)) = corpus.get_by_id(oid) else {
                    return Err(RaftError::InvalidSample {
                        sample_id: sample.id.clone(),
                        reason: format!("oracle passage '{oid}' not found in corpus"),
                    });
                };
                oracles.push((p, 0.0));
            }

            // Decide whether oracle passages are dropped under probability P.
            let drop_draw: f64 = rng.random();
            let drop_oracle = drop_draw < self.config.oracle_dropout_prob;
            let oracle_kept = !drop_oracle;

            let mut selected: Vec<(&Passage, bool, f64)> =
                Vec::with_capacity(self.config.num_context_docs.max(oracles.len()));

            let distractors_needed = if oracle_kept {
                for (p, s) in &oracles {
                    selected.push((*p, true, *s));
                }
                self.config.num_context_docs.saturating_sub(oracles.len())
            } else {
                self.config.num_context_docs
            };

            if distractors_needed > 0 {
                let mut pool = corpus.hard_negatives(
                    &sample.question,
                    &sample.oracle_ids,
                    self.config.hard_negative_pool,
                );
                // Always keep top hard negatives weighted near the front, while shuffling
                // within the top hard-negative pool so multiple QA pairs don't reuse the
                // exact identical distractor list verbatim.
                let take = distractors_needed.min(pool.len());
                if pool.len() > take {
                    // Keep the single hardest negative guaranteed, sample the rest from pool[1..].
                    let top_hard = pool.remove(0);
                    pool.shuffle(&mut rng);
                    let mut chosen = Vec::with_capacity(take);
                    chosen.push(top_hard);
                    chosen.extend(pool.into_iter().take(take - 1));
                    pool = chosen;
                } else {
                    pool.truncate(take);
                }

                for d in pool {
                    let (_, dp) = corpus
                        .get_by_id(&d.passage_id)
                        .expect("indexed distractor id exists");
                    selected.push((dp, false, d.bm25_score));
                    distractor_bm25_sum += d.bm25_score;
                    total_distractors += 1;
                }
            }

            // Deterministically shuffle context passages to eliminate positional bias.
            selected.shuffle(&mut rng);

            let mut context_passages = Vec::with_capacity(selected.len());
            let mut oracle_positions = Vec::new();
            for (idx, (p, is_oracle, score)) in selected.into_iter().enumerate() {
                let rank = idx + 1;
                if is_oracle {
                    oracle_positions.push(rank);
                    if !oracle_kept {
                        oracle_leak_count += 1;
                    }
                }
                context_passages.push(ContextPassage {
                    rank_in_prompt: rank,
                    passage_id: p.id.clone(),
                    title: p.title.clone(),
                    text: p.text.clone(),
                    is_oracle,
                    bm25_score: score,
                });
            }

            let distractor_count = context_passages.iter().filter(|c| !c.is_oracle).count();
            total_ctx_docs += context_passages.len();

            // Build verbatim citations when oracle is kept; zero citations when dropped.
            let citations = if oracle_kept {
                let cits = build_verbatim_citations(sample, &context_passages)?;
                if !cits.is_empty()
                    && cits.iter().all(|c| {
                        context_passages
                            .get(c.doc_rank - 1)
                            .is_some_and(|p| p.is_oracle && p.text.contains(&c.quote))
                    })
                {
                    supported_citations += 1;
                }
                oracle_kept_count += 1;
                cits
            } else {
                oracle_dropped_count += 1;
                Vec::new()
            };

            let user_prompt = format_user_prompt(&sample.question, &context_passages);
            let assistant_completion = format_assistant_completion(
                sample,
                oracle_kept,
                &citations,
                self.config.drop_oracle_mode,
            );

            examples.push(RaftExample {
                id: sample.id.clone(),
                question: sample.question.clone(),
                answer: sample.answer.clone(),
                oracle_kept,
                oracle_positions,
                distractor_count,
                context_passages,
                citations,
                system_prompt: self.config.system_prompt.clone(),
                user_prompt,
                assistant_completion,
            });
        }

        let n = examples.len() as f64;
        let stats = RaftDatasetStats {
            total_examples: examples.len(),
            oracle_kept_count,
            oracle_dropped_count,
            configured_dropout_prob: self.config.oracle_dropout_prob,
            empirical_dropout_rate: oracle_dropped_count as f64 / n,
            mean_context_docs: total_ctx_docs as f64 / n,
            mean_distractors_per_example: total_distractors as f64 / n,
            mean_hard_negative_bm25: if total_distractors == 0 {
                0.0
            } else {
                distractor_bm25_sum / total_distractors as f64
            },
            citation_support_rate: if oracle_kept_count == 0 {
                1.0
            } else {
                supported_citations as f64 / oracle_kept_count as f64
            },
            oracle_leak_count,
        };

        Ok(RaftDataset {
            eligibility: self.eligibility.clone(),
            config: self.config.clone(),
            examples,
            stats,
        })
    }
}

fn build_verbatim_citations(
    sample: &QaSample,
    context: &[ContextPassage],
) -> Result<Vec<Citation>> {
    let mut citations = Vec::new();
    let oracle_docs: Vec<&ContextPassage> = context.iter().filter(|p| p.is_oracle).collect();

    if sample.oracle_quotes.is_empty() {
        if let Some(first) = oracle_docs.first() {
            let quote = extract_representative_quote(&first.text);
            if !quote.is_empty() {
                citations.push(Citation {
                    doc_rank: first.rank_in_prompt,
                    passage_id: first.passage_id.clone(),
                    quote,
                });
            }
        }
        return Ok(citations);
    }

    for q in &sample.oracle_quotes {
        let Some(owner) = oracle_docs.iter().find(|p| p.text.contains(q.as_str())) else {
            return Err(RaftError::InvalidSample {
                sample_id: sample.id.clone(),
                reason: format!(
                    "oracle_quote '{q}' is not a verbatim substring of any oracle passage"
                ),
            });
        };
        citations.push(Citation {
            doc_rank: owner.rank_in_prompt,
            passage_id: owner.passage_id.clone(),
            quote: q.clone(),
        });
    }
    Ok(citations)
}

fn extract_representative_quote(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(end) = trimmed.find(['.', '!', '?']) {
        let candidate = trimmed[..=end].trim();
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    trimmed.chars().take(160).collect()
}

fn format_user_prompt(question: &str, context: &[ContextPassage]) -> String {
    let mut out = String::new();
    out.push_str("Passages:\n");
    for p in context {
        match &p.title {
            Some(t) if !t.trim().is_empty() => {
                let _ = writeln!(out, "[Doc {}] ({}) {}", p.rank_in_prompt, t.trim(), p.text);
            }
            _ => {
                let _ = writeln!(out, "[Doc {}] {}", p.rank_in_prompt, p.text);
            }
        }
    }
    let _ = write!(out, "\nQuestion: {question}");
    out
}

fn format_assistant_completion(
    sample: &QaSample,
    oracle_kept: bool,
    citations: &[Citation],
    drop_mode: DropOracleMode,
) -> String {
    let mut out = String::new();
    if oracle_kept {
        out.push_str("##Reason: Based on the provided passages, ");
        for (i, c) in citations.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let _ = write!(
                out,
                "[Doc {}] states ##begin_quote##{}##end_quote##.",
                c.doc_rank, c.quote
            );
        }
        if let Some(r) = &sample.reasoning {
            if !r.trim().is_empty() {
                let _ = write!(out, " {}", r.trim());
            }
        }
        let _ = write!(out, "\n##Answer: {}", sample.answer);
    } else {
        match drop_mode {
            DropOracleMode::ClosedBookReasoned => {
                let rationale = sample
                    .reasoning
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .unwrap_or("The provided passages do not contain the supporting evidence, so I answer from verified domain knowledge.");
                let _ = write!(out, "##Reason: {rationale}\n##Answer: {}", sample.answer);
            }
            DropOracleMode::AbstainWhenUnsupported => {
                out.push_str(
                    "##Reason: None of the provided passages contain verifiable evidence for this question.\n##Answer: Insufficient evidence in provided passages.",
                );
            }
        }
    }
    out
}

fn is_stopword(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "been"
            | "being"
            | "but"
            | "by"
            | "can"
            | "could"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "how"
            | "if"
            | "in"
            | "into"
            | "is"
            | "it"
            | "its"
            | "may"
            | "more"
            | "must"
            | "no"
            | "not"
            | "of"
            | "on"
            | "or"
            | "our"
            | "should"
            | "so"
            | "such"
            | "than"
            | "that"
            | "the"
            | "their"
            | "them"
            | "then"
            | "there"
            | "these"
            | "they"
            | "this"
            | "to"
            | "use"
            | "used"
            | "uses"
            | "using"
            | "was"
            | "we"
            | "were"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "will"
            | "with"
            | "would"
    )
}

fn tokenize(text: &str) -> Vec<String> {
    let mut raw = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lc in ch.to_lowercase() {
                cur.push(lc);
            }
        } else if !cur.is_empty() {
            raw.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        raw.push(cur);
    }
    let filtered: Vec<String> = raw
        .iter()
        .filter(|t| !is_stopword(t.as_str()))
        .cloned()
        .collect();
    if filtered.is_empty() {
        raw
    } else {
        filtered
    }
}

fn normalized_fingerprint(text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut prev_space = true;
    for ch in text.trim().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                hash ^= b' ' as u64;
                hash = hash.wrapping_mul(0x1000_0000_01b3);
                prev_space = true;
            }
        } else {
            for lc in ch.to_lowercase() {
                let mut buf = [0u8; 4];
                for &b in lc.encode_utf8(&mut buf).as_bytes() {
                    hash ^= b as u64;
                    hash = hash.wrapping_mul(0x1000_0000_01b3);
                }
            }
            prev_space = false;
        }
    }
    hash
}

fn escape_json_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_kv_str(out: &mut String, key: &str, val: &str, first: bool) {
    if !first {
        out.push(',');
    }
    escape_json_str(key, out);
    out.push(':');
    escape_json_str(val, out);
}

fn push_message(out: &mut String, role: &str, content: &str, first: bool) {
    if !first {
        out.push(',');
    }
    out.push('{');
    push_kv_str(out, "role", role, true);
    push_kv_str(out, "content", content, false);
    out.push('}');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eligible_zen() -> ZenEligibility {
        ZenEligibility {
            base_model_id: "zen-eco-4b".to_string(),
            base_model_license: LicenseKind::Apache2,
            dataset_name: "zen-domain-qa-v1".to_string(),
            dataset_license: LicenseKind::Mit,
        }
    }

    fn build_domain_corpus() -> PassageCorpus {
        let passages = vec![
            Passage {
                id: "p_oracle_1".to_string(),
                title: Some("Hanzo VFS Storage".to_string()),
                text: "Hanzo VFS persists immutable checkpoint chunks to S3 with SHA-256 content digests and 64 MiB multipart boundaries.".to_string(),
                source_doc: Some("vfs.md".to_string()),
            },
            Passage {
                id: "p_oracle_1_dup".to_string(),
                title: Some("Mirror".to_string()),
                text: "  hanzo vfs persists immutable checkpoint chunks to s3 with sha-256 content digests and 64 mib multipart boundaries. ".to_string(),
                source_doc: Some("mirror.md".to_string()),
            },
            Passage {
                id: "p_hard_neg_1".to_string(),
                title: Some("Hanzo S3 Multipart Spec".to_string()),
                text: "Hanzo S3 gateway configures multipart upload retries and SHA-256 ETag verification for inference weight tarballs, not checkpoint chunk manifests.".to_string(),
                source_doc: Some("s3.md".to_string()),
            },
            Passage {
                id: "p_hard_neg_2".to_string(),
                title: Some("VFS Cache Eviction".to_string()),
                text: "Hanzo VFS local NVMe read cache evicts cold checkpoint blobs under FIFO-LRU pressure and logs S3 miss latency histograms.".to_string(),
                source_doc: Some("cache.md".to_string()),
            },
            Passage {
                id: "p_hard_neg_3".to_string(),
                title: Some("DiLoCo Checkpoint Layout".to_string()),
                text: "DiLoCo coordinator checkpoints store momentum and carry safetensors files locally before VFS replication uploads shards.".to_string(),
                source_doc: Some("diloco.md".to_string()),
            },
            Passage {
                id: "p_hard_neg_4".to_string(),
                title: Some("Visor Scheduling".to_string()),
                text: "Visor orchestrates S3 artifact staging for cluster workers and verifies SHA-256 manifest signatures at startup.".to_string(),
                source_doc: Some("visor.md".to_string()),
            },
            Passage {
                id: "p_bg_1".to_string(),
                title: Some("Gardening Notes".to_string()),
                text: "Tomatoes thrive in loamy soil with steady morning sunlight and regular pruning of lateral suckers.".to_string(),
                source_doc: Some("garden.md".to_string()),
            },
            Passage {
                id: "p_bg_2".to_string(),
                title: Some("Baking Bread".to_string()),
                text: "Sourdough fermentation relies on wild yeast cultures and long cold retardation to develop blistered crusts.".to_string(),
                source_doc: Some("baking.md".to_string()),
            },
        ];
        PassageCorpus::new(passages).unwrap()
    }

    fn sample_qa(n: usize) -> Vec<QaSample> {
        (0..n)
            .map(|i| QaSample {
                id: format!("qa-{i:03}"),
                question: "How does Hanzo VFS persist checkpoint chunks to S3 and what multipart boundary size does it use?".to_string(),
                answer: "SHA-256 content digests with 64 MiB multipart boundaries.".to_string(),
                reasoning: Some("The VFS storage specification defines content-addressed S3 chunking at 64 MiB boundaries.".to_string()),
                oracle_ids: vec!["p_oracle_1".to_string()],
                oracle_quotes: vec![
                    "SHA-256 content digests and 64 MiB multipart boundaries".to_string(),
                ],
            })
            .collect()
    }

    #[test]
    fn zen_license_eligibility_gate_enforces_permissive_sources() {
        assert!(eligible_zen().verify().is_ok());

        for bad_lic in [
            LicenseKind::CcByNc,
            LicenseKind::Gpl,
            LicenseKind::Agpl,
            LicenseKind::Elv2,
            LicenseKind::ProprietaryRestricted,
            LicenseKind::Unknown,
        ] {
            let mut bad_ds = eligible_zen();
            bad_ds.dataset_license = bad_lic.clone();
            let err = RaftDatasetBuilder::new(bad_ds, RaftConfig::default())
                .err()
                .expect("non-permissive dataset license must be rejected");
            assert!(matches!(err, RaftError::IneligibleLicense(_)));

            let mut bad_model = eligible_zen();
            bad_model.base_model_license = bad_lic;
            let err = RaftDatasetBuilder::new(bad_model, RaftConfig::default())
                .err()
                .expect("non-permissive base model license must be rejected");
            assert!(matches!(err, RaftError::IneligibleLicense(_)));
        }
    }

    #[test]
    fn oracle_dropout_probability_p_controls_oracle_presence_across_zero_fractional_and_one(
    ) -> Result<()> {
        let corpus = build_domain_corpus();
        let qa = sample_qa(200);

        // P = 0.0 -> 0% oracle dropped (100% oracle kept)
        let cfg_keep_all = RaftConfig {
            oracle_dropout_prob: 0.0,
            num_context_docs: 4,
            hard_negative_pool: 5,
            seed: 7,
            ..RaftConfig::default()
        };
        let ds_all = RaftDatasetBuilder::new(eligible_zen(), cfg_keep_all)?.build(&corpus, &qa)?;
        assert_eq!(ds_all.stats.oracle_kept_count, 200);
        assert_eq!(ds_all.stats.oracle_dropped_count, 0);
        assert_eq!(ds_all.stats.oracle_leak_count, 0);
        assert!((ds_all.stats.citation_support_rate - 1.0).abs() < 1e-12);

        // P = 1.0 -> 100% oracle dropped (distractors only, 0 leaks, 0 fake citations)
        let cfg_drop_all = RaftConfig {
            oracle_dropout_prob: 1.0,
            num_context_docs: 4,
            hard_negative_pool: 5,
            seed: 7,
            drop_oracle_mode: DropOracleMode::AbstainWhenUnsupported,
            ..RaftConfig::default()
        };
        let ds_none = RaftDatasetBuilder::new(eligible_zen(), cfg_drop_all)?.build(&corpus, &qa)?;
        assert_eq!(ds_none.stats.oracle_kept_count, 0);
        assert_eq!(ds_none.stats.oracle_dropped_count, 200);
        assert_eq!(ds_none.stats.oracle_leak_count, 0);
        for ex in &ds_none.examples {
            assert!(!ex.oracle_kept);
            assert!(ex.oracle_positions.is_empty());
            assert!(ex.citations.is_empty());
            assert!(!ex.assistant_completion.contains("##begin_quote##"));
            assert!(!ex.assistant_completion.contains("[Doc "));
            assert!(ex
                .assistant_completion
                .contains("Insufficient evidence in provided passages"));
            // Also verify normalized duplicate of oracle was excluded from distractors!
            assert!(ex
                .context_passages
                .iter()
                .all(|p| p.passage_id != "p_oracle_1" && p.passage_id != "p_oracle_1_dup"));
        }

        // P = 0.25 (equivalent to with_oracle_keep_prob(0.75))
        let mut cfg_p25 = RaftConfig::with_oracle_keep_prob(0.75)?;
        cfg_p25.num_context_docs = 4;
        cfg_p25.hard_negative_pool = 5;
        cfg_p25.seed = 99;
        let ds_p25 = RaftDatasetBuilder::new(eligible_zen(), cfg_p25)?.build(&corpus, &qa)?;
        println!(
            "[raft P=0.25] kept={} dropped={} emp_dropout={:.3} mean_bm25={:.3} citation_support={:.3}",
            ds_p25.stats.oracle_kept_count,
            ds_p25.stats.oracle_dropped_count,
            ds_p25.stats.empirical_dropout_rate,
            ds_p25.stats.mean_hard_negative_bm25,
            ds_p25.stats.citation_support_rate
        );
        assert!((ds_p25.stats.empirical_dropout_rate - 0.25).abs() < 0.08);
        assert_eq!(ds_p25.stats.oracle_leak_count, 0);
        assert!((ds_p25.stats.citation_support_rate - 1.0).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn hard_negative_mining_prefers_relevant_distractors_and_shuffles_oracle_positions(
    ) -> Result<()> {
        let corpus = build_domain_corpus();
        let qa = sample_qa(40);
        let cfg = RaftConfig {
            oracle_dropout_prob: 0.0,
            num_context_docs: 4,
            hard_negative_pool: 4,
            seed: 123,
            ..RaftConfig::default()
        };
        let ds = RaftDatasetBuilder::new(eligible_zen(), cfg)?.build(&corpus, &qa)?;

        // Hard negatives must never select irrelevant gardening/baking passages when 4
        // domain-relevant hard negatives exist in the corpus.
        let mut seen_ranks = HashSet::new();
        for ex in &ds.examples {
            assert_eq!(ex.context_passages.len(), 4);
            assert_eq!(ex.distractor_count, 3);
            assert_eq!(ex.oracle_positions.len(), 1);
            let oracle_rank = ex.oracle_positions[0];
            seen_ranks.insert(oracle_rank);
            assert_eq!(ex.citations[0].doc_rank, oracle_rank);
            assert!(ex
                .assistant_completion
                .contains(&format!("[Doc {oracle_rank}]")));
            assert!(ex.assistant_completion.contains("##begin_quote##"));
            for p in &ex.context_passages {
                assert!(!p.passage_id.starts_with("p_bg_"));
                assert_ne!(p.passage_id, "p_oracle_1_dup");
            }
        }
        // Shuffling distributes oracle across multiple prompt positions (no fixed position bias).
        assert!(seen_ranks.len() >= 3, "saw oracle positions {seen_ranks:?}");
        Ok(())
    }

    #[test]
    fn jsonl_export_is_deterministic_and_properly_escaped() -> Result<()> {
        let corpus = build_domain_corpus();
        let mut qa = sample_qa(3);
        qa[0].question = "Why does \"Hanzo VFS\"\nuse 64 MiB\\chunks?".to_string();
        let cfg = RaftConfig {
            oracle_dropout_prob: 0.3,
            num_context_docs: 3,
            hard_negative_pool: 4,
            seed: 2026,
            ..RaftConfig::default()
        };
        let b1 = RaftDatasetBuilder::new(eligible_zen(), cfg.clone())?;
        let b2 = RaftDatasetBuilder::new(eligible_zen(), cfg)?;
        let d1 = b1.build(&corpus, &qa)?;
        let d2 = b2.build(&corpus, &qa)?;
        let jsonl1 = d1.to_jsonl();
        let jsonl2 = d2.to_jsonl();
        assert_eq!(jsonl1, jsonl2);
        let lines: Vec<&str> = jsonl1.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("\\\"Hanzo VFS\\\"\\nuse 64 MiB\\\\chunks?"));
        assert!(lines[0].contains("\"base_model\":\"zen-eco-4b\""));
        Ok(())
    }

    #[test]
    fn fabricated_oracle_quote_is_rejected_and_closed_book_drop_avoids_doc_citations() -> Result<()>
    {
        let corpus = build_domain_corpus();
        let mut bad_qa = sample_qa(1);
        bad_qa[0].oracle_quotes = vec!["nonexistent fabricated quote".to_string()];

        let cfg_keep = RaftConfig {
            oracle_dropout_prob: 0.0,
            num_context_docs: 3,
            hard_negative_pool: 4,
            seed: 5,
            ..RaftConfig::default()
        };
        let err = RaftDatasetBuilder::new(eligible_zen(), cfg_keep)?
            .build(&corpus, &bad_qa)
            .expect_err("ungrounded quote must be refused");
        assert!(matches!(err, RaftError::InvalidSample { .. }));

        let good_qa = sample_qa(5);
        let cfg_closed_book = RaftConfig {
            oracle_dropout_prob: 1.0,
            num_context_docs: 3,
            hard_negative_pool: 4,
            seed: 5,
            drop_oracle_mode: DropOracleMode::ClosedBookReasoned,
            ..RaftConfig::default()
        };
        let ds =
            RaftDatasetBuilder::new(eligible_zen(), cfg_closed_book)?.build(&corpus, &good_qa)?;
        for ex in &ds.examples {
            assert!(!ex.oracle_kept);
            assert!(!ex.assistant_completion.contains("##begin_quote##"));
            assert!(!ex.assistant_completion.contains("[Doc "));
            assert!(ex.assistant_completion.contains("##Answer:"));
        }
        Ok(())
    }
}
