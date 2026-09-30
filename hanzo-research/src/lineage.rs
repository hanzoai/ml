//! Lineage, contamination guards, purpose propagation, and evidence types.
//!
//! Enforces the Permanent Research Substrate invariants:
//! 1. Transitive lineage-chain purpose propagation (effective purpose cannot be less restrictive than any ancestor).
//! 2. `derivation_allowed` is true only if true for all ancestors.
//! 3. Fail-closed typed contamination error (`STAGE_REJECTED_CONTAMINATION`).
//! 4. Integer nanodollars (`cost_nanodollars: u64`) for all monetary fields.
//! 5. Content-addressed immutable gate specs and claims.
//! 6. WAL-ordered research events with sequence and hash chains.

use crate::canonical::rfc8785_digest;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Epistemic purpose of a dataset item or research artifact.
///
/// Ord ordering is defined such that:
/// Train (0) < TeacherMining (1) < Dev (2) < EvalOnly (3) < SealedEval (4)
/// Thus, `max(a, b)` selects the more restrictive purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataPurpose {
    /// Permitted for direct model optimization and pretraining/finetuning.
    Train = 0,
    /// Permitted for synthetic teacher distillation and preference pair mining.
    TeacherMining = 1,
    /// Development split for parameter selection. Not for final train or promotion claims.
    Dev = 2,
    /// Frozen evaluation split. Strictly forbidden from entering training or teacher derivation.
    EvalOnly = 3,
    /// Frozen sealed benchmark gate. Strictly forbidden from entering training or teacher derivation forever.
    SealedEval = 4,
}

impl DataPurpose {
    /// Return the most restrictive of two purposes.
    pub fn most_restrictive(a: Self, b: Self) -> Self {
        std::cmp::max(a, b)
    }

    /// Whether this purpose permits direct optimizer training.
    pub fn is_training_allowed(&self) -> bool {
        matches!(self, DataPurpose::Train | DataPurpose::TeacherMining)
    }

    /// Whether this purpose is an evaluation benchmark.
    pub fn is_eval(&self) -> bool {
        matches!(self, DataPurpose::Dev | DataPurpose::EvalOnly | DataPurpose::SealedEval)
    }
}

/// Lineage policy governing transitive derivation and purpose restriction across the DAG.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LineagePolicy {
    pub purpose: DataPurpose,
    pub derivation_allowed: bool,
}

impl LineagePolicy {
    /// Create a lineage policy for a root/source artifact.
    pub fn root(purpose: DataPurpose, derivation_allowed: bool) -> Self {
        LineagePolicy {
            purpose,
            // EvalOnly and SealedEval can never allow derivation into training
            derivation_allowed: if matches!(purpose, DataPurpose::EvalOnly | DataPurpose::SealedEval) {
                false
            } else {
                derivation_allowed
            },
        }
    }

    /// Derive policy for a child artifact from its parents and declared intent.
    ///
    /// Invariant: A child can ONLY make lineage MORE restrictive, never less.
    /// effective_purpose = max(declared_child_purpose, max(parent.effective_purpose...))
    /// effective_derivation_allowed = declared_child_derivation_allowed && all(parent.effective_derivation_allowed)
    pub fn derive(
        declared_child_purpose: DataPurpose,
        declared_child_derivation_allowed: bool,
        parents: &[&LineagePolicy],
    ) -> Self {
        let max_parent_purpose = parents
            .iter()
            .map(|p| p.purpose)
            .max()
            .unwrap_or(DataPurpose::Train);
        let effective_purpose = std::cmp::max(declared_child_purpose, max_parent_purpose);
        let effective_derivation_allowed = declared_child_derivation_allowed
            && parents.iter().all(|p| p.derivation_allowed)
            && !matches!(effective_purpose, DataPurpose::EvalOnly | DataPurpose::SealedEval);

        LineagePolicy {
            purpose: effective_purpose,
            derivation_allowed: effective_derivation_allowed,
        }
    }

    /// Assert that this lineage policy permits training. Returns a typed error if contaminated.
    pub fn assert_trainable(&self, context: &str) -> Result<(), ContaminationError> {
        if !self.purpose.is_training_allowed() {
            return Err(ContaminationError::ContaminatedPurpose {
                artifact_id: context.to_string(),
                purpose: self.purpose,
            });
        }
        if !self.derivation_allowed {
            return Err(ContaminationError::DerivationDisallowed {
                artifact_id: context.to_string(),
            });
        }
        Ok(())
    }
}

/// Typed error returned during preflight contamination guards.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum ContaminationError {
    #[error("STAGE_REJECTED_CONTAMINATION: artifact {artifact_id} has purpose {purpose:?} (forbidden in training)")]
    ContaminatedPurpose {
        artifact_id: String,
        purpose: DataPurpose,
    },
    #[error("STAGE_REJECTED_CONTAMINATION: artifact {artifact_id} has derivation_allowed = false")]
    DerivationDisallowed {
        artifact_id: String,
    },
    #[error("STAGE_REJECTED_CONTAMINATION: contamination detected in stage: {message}")]
    StageRejectedContamination {
        message: String,
    },
}

fn default_media_type() -> String {
    "application/octet-stream".to_string()
}

fn default_canonicalization() -> String {
    "raw".to_string()
}

fn default_compression() -> String {
    "none".to_string()
}

fn default_worker_id() -> String {
    "worker-0".to_string()
}

/// An immutable content-addressed reference to an artifact in `/v1/research/artifacts/:sha256`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    #[serde(default)]
    pub artifact_id: String,
    pub sha256: String,
    #[serde(default)]
    pub content_sha256: String,
    #[serde(default = "default_media_type")]
    pub media_type: String,
    #[serde(default = "default_canonicalization")]
    pub canonicalization: String,
    #[serde(default = "default_compression")]
    pub compression: String,
    #[serde(default)]
    pub uncompressed_size: u64,
    pub ref_uri: String, // e.g. "sha256:<hash>"
    pub lineage: LineagePolicy,
}

impl ArtifactRef {
    pub fn new(sha256: String, lineage: LineagePolicy) -> Self {
        let ref_uri = format!("sha256:{sha256}");
        ArtifactRef {
            artifact_id: format!("artifact:{sha256}"),
            sha256: sha256.clone(),
            content_sha256: sha256,
            media_type: default_media_type(),
            canonicalization: default_canonicalization(),
            compression: default_compression(),
            uncompressed_size: 0,
            ref_uri,
            lineage,
        }
    }

    pub fn with_metadata(
        artifact_id: impl Into<String>,
        content_sha256: impl Into<String>,
        media_type: impl Into<String>,
        canonicalization: impl Into<String>,
        compression: impl Into<String>,
        uncompressed_size: u64,
        lineage: LineagePolicy,
    ) -> Self {
        let content_sha256 = content_sha256.into();
        let ref_uri = format!("sha256:{content_sha256}");
        ArtifactRef {
            artifact_id: artifact_id.into(),
            sha256: content_sha256.clone(),
            content_sha256,
            media_type: media_type.into(),
            canonicalization: canonicalization.into(),
            compression: compression.into(),
            uncompressed_size,
            ref_uri,
            lineage,
        }
    }
}

/// Environment fingerprint capturing the exact reproducibility state of the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentV1 {
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub host: String,
    pub os: String,
    pub kernel_version: String,
    pub cargo_lock_hash: String,
    pub image_digest: String,
    pub precision_config: String,
    pub determinism_config: String,
    pub compiler_version: String,
    pub lib_versions: BTreeMap<String, String>,
}

impl EnvironmentV1 {
    /// Compute the RFC 8785 canonical hash of this environment.
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        rfc8785_digest(self)
    }
}

/// Result of a stepped determinism probe on GPU or cluster.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeterminismReport {
    pub steps_compared: Vec<u64>,
    pub loss_exact_match: bool,
    pub gradient_norm_exact_match: bool,
    pub max_weight_divergence: f64,
    pub passed: bool,
    pub probe_duration_secs: f64,
    pub notes: String,
}

impl DeterminismReport {
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        rfc8785_digest(self)
    }
}

/// An observation or inference call harvested from a teacher model (e.g. Jev).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TeacherInteraction {
    pub model: String,
    pub input_prompt_hash: String,
    pub output_text: String,
    pub tokens_prompt: u64,
    pub tokens_completion: u64,
    /// Monetary cost strictly stored in integer nanodollars ($1 USD = 1,000,000,000 nano-dollars).
    pub cost_nanodollars: u64,
    pub latency_ms: u64,
    pub lineage: LineagePolicy,
    pub when: String,
}

/// Bradley-Terry pairwise preference evidence derived from teacher responses or humans.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairEvidence {
    pub prompt_hash: String,
    pub chosen_hash: String,
    pub rejected_hash: String,
    pub margin: f64,
    pub cost_nanodollars: u64,
    pub lineage: LineagePolicy,
}

/// Preregistered, immutable gate specification for model promotion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateSpec {
    pub gate_id: String,
    pub candidate_discovery_recall_floor: f64,
    pub reranking_top1_min: f64,
    pub reranking_mrr_min: f64,
    pub conditional_reranking_top1_min: f64,
    pub allow_loss_regression_pct: f64,
    pub notes: String,
}

impl GateSpec {
    pub fn digest(&self) -> Result<String, serde_json::Error> {
        rfc8785_digest(self)
    }
}

/// Formal claim regarding benchmark performance or promotion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    pub id: String,
    pub run_id: String,
    pub benchmark_id: String,
    pub metric: String,
    pub claimed_value: f64,
    pub verified: bool,
    pub gate_sha256: String,
    pub lineage: LineagePolicy,
    pub created_at: String,
}

/// Authoritative derivation of lineage policy from parent artifacts.
///
/// Invariant:
/// - effective_purpose = max(declared_child_purpose, max(parents.effective_purpose...))
/// - derivation_allowed = declared_child_derivation_allowed && all(parents.derivation_allowed)
///   && !matches!(effective_purpose, DataPurpose::EvalOnly | DataPurpose::SealedEval)
pub fn derive_lineage_policy(
    declared_child_purpose: DataPurpose,
    declared_child_derivation_allowed: bool,
    parents: &[&LineagePolicy],
) -> LineagePolicy {
    LineagePolicy::derive(declared_child_purpose, declared_child_derivation_allowed, parents)
}

/// Validate whether an artifact or checkpoint is legally eligible for model promotion.
///
/// Acceptance rule:
/// Given any served checkpoint, reconstruct the full lineage and verify that
/// EVERY transitive ancestor is derivation-allowed for promotion (derivation_allowed == true),
/// and no ancestor has an evaluation purpose (EvalOnly / SealedEval / Dev).
pub fn verify_promotion_eligibility(
    artifact_id: &str,
    policy: &LineagePolicy,
    transitive_ancestors: &[&LineagePolicy],
) -> Result<(), ContaminationError> {
    if !policy.derivation_allowed {
        return Err(ContaminationError::DerivationDisallowed {
            artifact_id: artifact_id.to_string(),
        });
    }
    if !policy.purpose.is_training_allowed() {
        return Err(ContaminationError::ContaminatedPurpose {
            artifact_id: artifact_id.to_string(),
            purpose: policy.purpose,
        });
    }
    for (i, ancestor) in transitive_ancestors.iter().enumerate() {
        if !ancestor.derivation_allowed {
            return Err(ContaminationError::DerivationDisallowed {
                artifact_id: format!("{artifact_id}->transitive_ancestor[{i}]"),
            });
        }
        if !ancestor.purpose.is_training_allowed() {
            return Err(ContaminationError::ContaminatedPurpose {
                artifact_id: format!("{artifact_id}->transitive_ancestor[{i}]"),
                purpose: ancestor.purpose,
            });
        }
    }
    Ok(())
}

/// Tamper-evident WAL event envelope for ordered research telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResearchEvent {
    pub run_id: String,
    #[serde(default = "default_worker_id")]
    pub worker_id: String,
    pub event_id: String,
    pub sequence: u64,
    pub previous_event_hash: String,
    pub payload_hash: String,
    pub payload: serde_json::Value,
    pub created_at: String,
}

impl ResearchEvent {
    pub fn new(
        run_id: String,
        worker_id: String,
        event_id: String,
        sequence: u64,
        previous_event_hash: String,
        payload: serde_json::Value,
        created_at: String,
    ) -> Result<Self, serde_json::Error> {
        let payload_hash = rfc8785_digest(&payload)?;
        Ok(ResearchEvent {
            run_id,
            worker_id,
            event_id,
            sequence,
            previous_event_hash,
            payload_hash,
            payload,
            created_at,
        })
    }

    pub fn digest(&self) -> Result<String, serde_json::Error> {
        rfc8785_digest(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lineage_policy_propagation() {
        let train_policy = LineagePolicy::root(DataPurpose::Train, true);
        assert!(train_policy.assert_trainable("source").is_ok());

        let eval_policy = LineagePolicy::root(DataPurpose::EvalOnly, false);
        assert!(eval_policy.assert_trainable("eval_src").is_err());

        // Derived child from eval parent MUST be tainted
        let derived = LineagePolicy::derive(DataPurpose::TeacherMining, true, &[&train_policy, &eval_policy]);
        assert_eq!(derived.purpose, DataPurpose::EvalOnly);
        assert!(!derived.derivation_allowed);
        assert!(derived.assert_trainable("derived_from_eval").is_err());

        // Derived child from clean train parent remains TeacherMining
        let clean_child = LineagePolicy::derive(DataPurpose::TeacherMining, true, &[&train_policy]);
        assert_eq!(clean_child.purpose, DataPurpose::TeacherMining);
        assert!(clean_child.derivation_allowed);
        assert!(clean_child.assert_trainable("clean_child").is_ok());
    }

    #[test]
    fn test_promotion_eligibility_acceptance_check() {
        let root_train = LineagePolicy::root(DataPurpose::Train, true);
        let teacher_mining = LineagePolicy::derive(DataPurpose::TeacherMining, true, &[&root_train]);
        let checkpoint_policy = LineagePolicy::derive(DataPurpose::Train, true, &[&root_train, &teacher_mining]);

        // Clean ancestry passes
        assert!(verify_promotion_eligibility(
            "ckpt-a10",
            &checkpoint_policy,
            &[&root_train, &teacher_mining]
        ).is_ok());

        // Tainted ancestor with derivation_allowed = false must fail
        let mut tainted_ancestor = root_train.clone();
        tainted_ancestor.derivation_allowed = false;
        assert!(verify_promotion_eligibility(
            "ckpt-a10",
            &checkpoint_policy,
            &[&tainted_ancestor]
        ).is_err());

        // Tainted ancestor with EvalOnly must fail
        let eval_ancestor = LineagePolicy::root(DataPurpose::EvalOnly, false);
        assert!(verify_promotion_eligibility(
            "ckpt-a10",
            &checkpoint_policy,
            &[&eval_ancestor]
        ).is_err());
    }
}
