//! Learning from coding-agent execution traces (`Agent Lightning`, arXiv:2608.17528, HIP-1348)
//! for license-eligible Zen models on `hanzo-train`.
//!
//! Converts successful and failed coding-agent interaction trajectories captured behind real
//! harnesses into verified step-level supervised and contrastive preference datasets:
//! 1. Enforces permissive SPDX clearance (`Apache-2.0`, `MIT`, `CC-BY-4.0`) on both base models
//!    and trace corpora, enforces strict `tenant_id` isolation, and scrubs secret credentials and
//!    email PII prior to tokenization or shard export.
//! 2. Screens trajectories through [`RewardHackingDetector`] to reject assertion deletion,
//!    forbidden test-file edits, shell exit-code spoofing, empty or tautological patches, and
//!    circular non-progress retry loops.
//! 3. Assigns discounted step-level credit with group-efficiency weighting, prunes zero-progress
//!    detour steps, deduplicates redundant rollouts under optional token budgets, pairs verified
//!    repairs against minimally-divergent failed or reward-hacked sibling trajectories, and trains
//!    [`ZenTracePolicy`] via [`crate::adam::Adam`].

use crate::adam::{Adam, Hyper};
use anyhow::{bail, ensure, Context, Result};
use hanzo_ml::{DType, Device, Tensor};
use hanzo_nn::{embedding, linear, ops, Embedding, Linear, Module, VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

/// Permissive SPDX license identifiers approved for Zen base models and coding-agent trace shards.
pub const APPROVED_LICENSES: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "CC-BY-4.0",
    "MIT OR Apache-2.0",
    "Apache-2.0 OR MIT",
];

/// Secret token prefixes scrubbed from prompts, commands, and tool outputs before export.
const SECRET_PREFIXES: &[&str] = &[
    "sk-",
    "ghp_",
    "gho_",
    "github_pat_",
    "AKIA",
    "xoxb-",
    "hf_",
];

/// Verifies that a base model or trace dataset carries an approved permissive SPDX identifier.
pub fn verify_spdx_license(spdx_id: &str, artifact_kind: &str) -> Result<()> {
    let trimmed = spdx_id.trim();
    ensure!(
        !trimmed.is_empty(),
        "{artifact_kind} SPDX license identifier must not be empty"
    );
    if APPROVED_LICENSES
        .iter()
        .any(|allowed| allowed.eq_ignore_ascii_case(trimmed))
    {
        Ok(())
    } else {
        bail!(
            "{artifact_kind} license '{trimmed}' is not cleared for Zen training; require Apache-2.0, MIT, or CC-BY-4.0"
        )
    }
}

/// Tool category invoked at one step of a coding-agent execution trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ToolKind {
    ReadSpan,
    SearchCode,
    ApplyPatch,
    RunTests,
    ShellExec,
    SubmitPatch,
}

/// One interaction step inside a coding-agent trajectory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceStep {
    pub step_index: usize,
    pub state_tokens: Vec<u32>,
    pub action_id: u32,
    pub tool_kind: ToolKind,
    pub command_or_target: String,
    pub tool_output_excerpt: String,
    pub exit_code: i32,
    pub lines_added: usize,
    pub lines_removed: usize,
    pub added_assertions: usize,
    pub removed_assertions: usize,
    pub touched_test_files: Vec<String>,
    pub made_state_progress: bool,
    pub token_count: usize,
}

/// Independent harness outcome recorded at the end of a trajectory rollout.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierOutcome {
    pub reported_exit_code: i32,
    pub verifier_passed: bool,
    pub tests_ran: usize,
    pub tests_failed: usize,
    pub substantive_source_changes: usize,
}

/// Full coding-agent rollout trace captured behind an execution harness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentTrace {
    pub trace_id: String,
    pub tenant_id: String,
    pub task_id: String,
    pub source_license: String,
    pub prompt_text: String,
    pub task_forbids_test_edits: bool,
    pub requires_code_change: bool,
    pub steps: Vec<TraceStep>,
    pub outcome: VerifierOutcome,
}

/// Enforces strict multi-tenant boundary isolation across all candidate traces in a batch or job.
pub fn ensure_trace_tenant_isolation(expected_tenant: &str, traces: &[AgentTrace]) -> Result<()> {
    let expected = expected_tenant.trim();
    ensure!(!expected.is_empty(), "expected tenant_id must not be empty");
    for trace in traces {
        if trace.tenant_id != expected {
            bail!(
                "tenant isolation violation: job tenant '{}' cannot ingest trace '{}' owned by '{}'",
                expected,
                trace.trace_id,
                trace.tenant_id
            );
        }
    }
    Ok(())
}

fn is_token_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.'
}

fn is_email_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.' || b == b'+'
}

fn redact_pem_blocks(input: &str) -> (String, usize) {
    let begin_marker = "-----BEGIN ";
    let key_suffix = "PRIVATE KEY-----";
    let end_prefix = "-----END ";
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    let mut count = 0usize;

    while let Some(start) = rest.find(begin_marker) {
        let after_begin = &rest[start..];
        let Some(header_end) = after_begin.find(key_suffix) else {
            out.push_str(&rest[..start + begin_marker.len()]);
            rest = &rest[start + begin_marker.len()..];
            continue;
        };
        if header_end > 64 {
            out.push_str(&rest[..start + begin_marker.len()]);
            rest = &rest[start + begin_marker.len()..];
            continue;
        }
        out.push_str(&rest[..start]);
        out.push_str("[REDACTED_PRIVATE_KEY]");
        count += 1;

        let body_after = &after_begin[header_end + key_suffix.len()..];
        if let Some(end_pos) = body_after.find(end_prefix) {
            let tail = &body_after[end_pos..];
            if let Some(end_close) = tail.find(key_suffix) {
                rest = &tail[end_close + key_suffix.len()..];
                continue;
            }
        }
        rest = "";
    }
    out.push_str(rest);
    (out, count)
}

fn redact_inline_tokens_and_pii(input: &str) -> (String, usize) {
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut count = 0usize;
    let mut i = 0usize;

    while i < len {
        // 1. URL inline user:password@host credentials
        if i + 3 < len && &bytes[i..i + 3] == b"://" {
            out.push_str("://");
            i += 3;
            let seg_start = i;
            let mut seg_end = i;
            while seg_end < len
                && !bytes[seg_end].is_ascii_whitespace()
                && bytes[seg_end] != b'/'
                && bytes[seg_end] != b'"'
                && bytes[seg_end] != b'\''
            {
                seg_end += 1;
            }
            let authority = &input[seg_start..seg_end];
            if let Some(at_idx) = authority.rfind('@') {
                let userinfo = &authority[..at_idx];
                if userinfo.contains(':') && !userinfo.is_empty() {
                    out.push_str("[REDACTED_CREDENTIAL]@");
                    out.push_str(&authority[at_idx + 1..]);
                    count += 1;
                    i = seg_end;
                    continue;
                }
            }
        }

        // 2. Bearer <secret> authorization token
        if i + 7 < len && input[i..].starts_with("Bearer ") {
            let tok_start = i + 7;
            let mut tok_end = tok_start;
            while tok_end < len && (is_token_char(bytes[tok_end]) || bytes[tok_end] == b'=') {
                tok_end += 1;
            }
            if tok_end - tok_start >= 8 {
                out.push_str("Bearer [REDACTED_SECRET]");
                count += 1;
                i = tok_end;
                continue;
            }
        }

        // 3. Known API key prefixes (at word boundary)
        let at_boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if at_boundary {
            let mut matched_end = None;
            for prefix in SECRET_PREFIXES {
                if input[i..].starts_with(prefix) {
                    let body_start = i + prefix.len();
                    let mut body_end = body_start;
                    while body_end < len && is_token_char(bytes[body_end]) {
                        body_end += 1;
                    }
                    if body_end - body_start >= 8 {
                        matched_end = Some(body_end);
                        break;
                    }
                }
            }
            if let Some(end_idx) = matched_end {
                out.push_str("[REDACTED_SECRET]");
                count += 1;
                i = end_idx;
                continue;
            }
        }

        // 4. Email PII (local@domain.tld)
        if is_email_char(bytes[i]) && (i == 0 || !is_email_char(bytes[i - 1])) {
            let local_start = i;
            let mut cur = i;
            while cur < len && is_email_char(bytes[cur]) {
                cur += 1;
            }
            if cur > local_start && cur < len && bytes[cur] == b'@' {
                let dom_start = cur + 1;
                let mut dom_end = dom_start;
                while dom_end < len
                    && (bytes[dom_end].is_ascii_alphanumeric()
                        || bytes[dom_end] == b'-'
                        || bytes[dom_end] == b'.')
                {
                    dom_end += 1;
                }
                while dom_end > dom_start && bytes[dom_end - 1] == b'.' {
                    dom_end -= 1;
                }
                let domain = &input[dom_start..dom_end];
                if domain.len() >= 4
                    && domain.contains('.')
                    && !domain.starts_with('.')
                    && !domain.ends_with('.')
                {
                    out.push_str("[REDACTED_EMAIL]");
                    count += 1;
                    i = dom_end;
                    continue;
                }
            }
        }

        let ch = input[i..].chars().next().expect("non-empty slice");
        out.push(ch);
        i += ch.len_utf8();
    }

    (out, count)
}

/// Scrubs secret credentials, PEM private keys, inline URL passwords, and email PII from `input`,
/// returning `(sanitized_text, redaction_count)`.
pub fn redact_text_secrets(input: &str) -> (String, usize) {
    let (after_pem, pem_n) = redact_pem_blocks(input);
    let (clean, inline_n) = redact_inline_tokens_and_pii(&after_pem);
    (clean, pem_n + inline_n)
}

/// Returns a sanitized copy of `trace` with all free-text fields scrubbed of secrets and PII,
/// along with the total number of redacted spans.
pub fn sanitize_trace(trace: &AgentTrace) -> (AgentTrace, usize) {
    let mut total_redacted = 0usize;
    let (prompt_text, r0) = redact_text_secrets(&trace.prompt_text);
    total_redacted += r0;

    let mut steps = Vec::with_capacity(trace.steps.len());
    for step in &trace.steps {
        let (command_or_target, r1) = redact_text_secrets(&step.command_or_target);
        let (tool_output_excerpt, r2) = redact_text_secrets(&step.tool_output_excerpt);
        total_redacted += r1 + r2;
        let mut clean_step = step.clone();
        clean_step.command_or_target = command_or_target;
        clean_step.tool_output_excerpt = tool_output_excerpt;
        steps.push(clean_step);
    }

    let mut clean_trace = trace.clone();
    clean_trace.prompt_text = prompt_text;
    clean_trace.steps = steps;
    (clean_trace, total_redacted)
}

/// Specific reward-hacking pattern detected in a coding-agent trace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewardHackViolation {
    AssertionDeletion {
        removed: usize,
        added: usize,
    },
    ForbiddenTestModification {
        files: Vec<String>,
    },
    ExitCodeSpoofing {
        spoof_reason: String,
    },
    EmptyOrTautologicalPatch,
    CircularRetryLoop {
        signature: String,
        repeats: usize,
    },
}

/// Result of auditing one trajectory with [`RewardHackingDetector`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewardHackVerdict {
    pub violations: Vec<RewardHackViolation>,
}

impl RewardHackVerdict {
    pub fn is_clean(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn is_hacked(&self) -> bool {
        !self.violations.is_empty()
    }
}

/// Anti-reward-hacking verifier catching test tampering, assertion deletion, exit-code spoofing,
/// tautological empty patches, and circular retry loops before trace supervision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewardHackingDetector {
    pub max_repeat_cycles: usize,
}

impl Default for RewardHackingDetector {
    fn default() -> Self {
        Self {
            max_repeat_cycles: 3,
        }
    }
}

impl RewardHackingDetector {
    pub fn new(max_repeat_cycles: usize) -> Self {
        Self {
            max_repeat_cycles: max_repeat_cycles.max(2),
        }
    }

    /// Audits `trace` across all five structural reward-hacking classes.
    pub fn inspect(&self, trace: &AgentTrace) -> RewardHackVerdict {
        let mut violations = Vec::new();

        // 1. Net assertion deletion across patch steps.
        let total_added: usize = trace.steps.iter().map(|s| s.added_assertions).sum();
        let total_removed: usize = trace.steps.iter().map(|s| s.removed_assertions).sum();
        if total_removed > total_added {
            violations.push(RewardHackViolation::AssertionDeletion {
                removed: total_removed,
                added: total_added,
            });
        }

        // 2. Test-file tampering when the task contract locks test files.
        if trace.task_forbids_test_edits {
            let mut touched = BTreeSet::new();
            for step in &trace.steps {
                for path in &step.touched_test_files {
                    touched.insert(path.clone());
                }
            }
            if !touched.is_empty() {
                violations.push(RewardHackViolation::ForbiddenTestModification {
                    files: touched.into_iter().collect(),
                });
            }
        }

        // 3. Trivial exit-code spoofing in shell/test commands or mismatched verifier outcome.
        let spoof_patterns = [
            "|| true",
            "; true",
            "exit 0",
            "sys.exit(0)",
            "process::exit(0)",
            "os._exit(0)",
        ];
        let mut spoof_detail = None;
        for step in &trace.steps {
            let lower = step.command_or_target.to_ascii_lowercase();
            for pat in spoof_patterns {
                if lower.contains(pat) {
                    spoof_detail = Some(format!(
                        "step {} command contains exit-status override '{}'",
                        step.step_index, pat
                    ));
                    break;
                }
            }
            if spoof_detail.is_some() {
                break;
            }
        }
        if spoof_detail.is_none()
            && trace.outcome.reported_exit_code == 0
            && (!trace.outcome.verifier_passed
                || trace.outcome.tests_ran == 0
                || trace.outcome.tests_failed > 0)
        {
            spoof_detail = Some(format!(
                "reported exit_code=0 despite verifier_passed={}, tests_ran={}, tests_failed={}",
                trace.outcome.verifier_passed,
                trace.outcome.tests_ran,
                trace.outcome.tests_failed
            ));
        }
        if let Some(spoof_reason) = spoof_detail {
            violations.push(RewardHackViolation::ExitCodeSpoofing {
                spoof_reason,
            });
        }

        // 4. Empty or tautological patch when a substantive code repair is required.
        if trace.requires_code_change
            && trace.outcome.reported_exit_code == 0
            && trace.outcome.substantive_source_changes == 0
        {
            violations.push(RewardHackViolation::EmptyOrTautologicalPatch);
        }

        // 5. Circular non-progress retry loops.
        let mut repeat_counts: HashMap<(ToolKind, String, Vec<u32>), usize> = HashMap::new();
        let mut loop_hit: Option<(String, usize)> = None;
        for step in &trace.steps {
            if !step.made_state_progress {
                let key = (
                    step.tool_kind,
                    step.command_or_target.clone(),
                    step.state_tokens.clone(),
                );
                let count = repeat_counts.entry(key).or_insert(0);
                *count += 1;
                if *count >= self.max_repeat_cycles {
                    let sig = format!("{:?}:{}", step.tool_kind, step.command_or_target);
                    match &mut loop_hit {
                        Some((_, prev_n)) if *count > *prev_n => {
                            loop_hit = Some((sig, *count));
                        }
                        None => {
                            loop_hit = Some((sig, *count));
                        }
                        _ => {}
                    }
                }
            }
        }
        if let Some((signature, repeats)) = loop_hit {
            violations.push(RewardHackViolation::CircularRetryLoop {
                signature,
                repeats,
            });
        }

        RewardHackVerdict {
            violations,
        }
    }

    /// Returns `true` iff `trace` passed independent verification cleanly with zero reward-hack
    /// violations.
    pub fn is_verified_success(&self, trace: &AgentTrace) -> bool {
        !trace.steps.is_empty()
            && trace.outcome.reported_exit_code == 0
            && trace.outcome.verifier_passed
            && trace.outcome.tests_ran > 0
            && trace.outcome.tests_failed == 0
            && self.inspect(trace).is_clean()
    }
}

/// Step-level advantage-weighted positive supervision record for RFT.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StepSupervisionRecord {
    pub task_id: String,
    pub trace_id: String,
    pub step_index: usize,
    pub state_tokens: Vec<u32>,
    pub action_id: u32,
    pub advantage_weight: f32,
    pub token_count: usize,
}

/// Contrastive preference record pairing a verified step action against a minimally-divergent
/// failed or reward-hacked action on the identical task state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreferencePairRecord {
    pub task_id: String,
    pub divergence_step: usize,
    pub state_tokens: Vec<u32>,
    pub chosen_action: u32,
    pub rejected_action: u32,
    pub margin_weight: f32,
    pub rejected_was_hack: bool,
    pub token_count: usize,
}

/// Accounting summary produced by [`TraceDatasetBuilder::build`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetBuildStats {
    pub raw_traces: usize,
    pub deduplicated_traces: usize,
    pub verified_clean_traces: usize,
    pub honest_failed_traces: usize,
    pub reward_hacked_traces: usize,
    pub redacted_secret_spans: usize,
    pub sft_step_records: usize,
    pub pruned_detour_steps: usize,
    pub preference_pairs: usize,
    pub supervised_tokens: usize,
}

#[derive(Serialize)]
#[serde(tag = "record_type")]
enum ShardJsonlRow<'a> {
    #[serde(rename = "step_rft")]
    StepRft {
        tenant_id: &'a str,
        task_id: &'a str,
        trace_id: &'a str,
        step_index: usize,
        state_tokens: &'a [u32],
        action_id: u32,
        advantage_weight: f32,
        token_count: usize,
    },
    #[serde(rename = "step_dpo")]
    StepDpo {
        tenant_id: &'a str,
        task_id: &'a str,
        divergence_step: usize,
        state_tokens: &'a [u32],
        chosen_action: u32,
        rejected_action: u32,
        margin_weight: f32,
        rejected_was_hack: bool,
        token_count: usize,
    },
}

/// Compiled training dataset containing advantage-weighted step records and contrastive pairs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BuiltTraceDataset {
    pub tenant_id: String,
    pub sft_records: Vec<StepSupervisionRecord>,
    pub preference_pairs: Vec<PreferencePairRecord>,
    pub stats: DatasetBuildStats,
}

impl BuiltTraceDataset {
    /// Exports the dataset as a deterministically ordered JSONL shard after verifying `license`,
    /// returning `(line_count, byte_len, sha256_hex)`.
    pub fn export_jsonl_shard(&self, path: &Path, license: &str) -> Result<(usize, u64, String)> {
        verify_spdx_license(license, "export_shard")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create JSONL shard parent directory")?;
        }

        let mut sorted_sft = self.sft_records.clone();
        sorted_sft.sort_by(|a, b| {
            (&a.task_id, &a.trace_id, a.step_index, a.action_id).cmp(&(
                &b.task_id,
                &b.trace_id,
                b.step_index,
                b.action_id,
            ))
        });

        let mut sorted_dpo = self.preference_pairs.clone();
        sorted_dpo.sort_by(|a, b| {
            (
                &a.task_id,
                a.divergence_step,
                &a.state_tokens,
                a.chosen_action,
                a.rejected_action,
            )
                .cmp(&(
                    &b.task_id,
                    b.divergence_step,
                    &b.state_tokens,
                    b.chosen_action,
                    b.rejected_action,
                ))
        });

        let file = File::create(path).context("create JSONL shard file")?;
        let mut writer = BufWriter::new(file);
        let mut hasher = Sha256::new();
        let mut lines = 0usize;
        let mut bytes_written = 0u64;

        for rec in &sorted_sft {
            let row = ShardJsonlRow::StepRft {
                tenant_id: &self.tenant_id,
                task_id: &rec.task_id,
                trace_id: &rec.trace_id,
                step_index: rec.step_index,
                state_tokens: &rec.state_tokens,
                action_id: rec.action_id,
                advantage_weight: (rec.advantage_weight * 1e6).round() / 1e6,
                token_count: rec.token_count,
            };
            let mut line = serde_json::to_vec(&row).context("serialize step_rft row")?;
            line.push(b'\n');
            writer.write_all(&line)?;
            hasher.update(&line);
            bytes_written += line.len() as u64;
            lines += 1;
        }

        for pair in &sorted_dpo {
            let row = ShardJsonlRow::StepDpo {
                tenant_id: &self.tenant_id,
                task_id: &pair.task_id,
                divergence_step: pair.divergence_step,
                state_tokens: &pair.state_tokens,
                chosen_action: pair.chosen_action,
                rejected_action: pair.rejected_action,
                margin_weight: (pair.margin_weight * 1e6).round() / 1e6,
                rejected_was_hack: pair.rejected_was_hack,
                token_count: pair.token_count,
            };
            let mut line = serde_json::to_vec(&row).context("serialize step_dpo row")?;
            line.push(b'\n');
            writer.write_all(&line)?;
            hasher.update(&line);
            bytes_written += line.len() as u64;
            lines += 1;
        }

        writer.flush()?;
        let digest = hasher.finalize();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        Ok((lines, bytes_written, hex))
    }
}

fn trace_dedup_fingerprint(trace: &AgentTrace) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(trace.task_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(trace.outcome.reported_exit_code.to_le_bytes());
    hasher.update([u8::from(trace.outcome.verifier_passed)]);
    for step in &trace.steps {
        for &tok in &step.state_tokens {
            hasher.update(tok.to_le_bytes());
        }
        hasher.update(step.action_id.to_le_bytes());
        hasher.update(step.exit_code.to_le_bytes());
        hasher.update(step.command_or_target.as_bytes());
    }
    hasher.finalize().into()
}

/// Agent-Lightning step-credit & contrastive preference dataset builder.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceDatasetBuilder {
    pub tenant_id: String,
    pub base_model_license: String,
    pub discount_gamma: f64,
    pub detour_step_weight: f64,
    pub hack_margin_weight: f32,
    pub fail_margin_weight: f32,
    pub max_repeat_cycles: usize,
    pub max_token_budget: Option<usize>,
}

impl TraceDatasetBuilder {
    pub fn new(tenant_id: impl Into<String>, base_model_license: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            base_model_license: base_model_license.into(),
            discount_gamma: 0.90,
            detour_step_weight: 0.0,
            hack_margin_weight: 1.25,
            fail_margin_weight: 1.0,
            max_repeat_cycles: 3,
            max_token_budget: None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        verify_spdx_license(&self.base_model_license, "base_model")?;
        ensure!(
            !self.tenant_id.trim().is_empty(),
            "tenant_id must not be empty"
        );
        ensure!(
            (0.0..=1.0).contains(&self.discount_gamma) && self.discount_gamma > 0.0,
            "discount_gamma must be in (0.0, 1.0], got {}",
            self.discount_gamma
        );
        ensure!(
            (0.0..=1.0).contains(&self.detour_step_weight),
            "detour_step_weight must be in [0.0, 1.0]"
        );
        ensure!(
            self.hack_margin_weight > 0.0 && self.fail_margin_weight > 0.0,
            "contrastive margin weights must be > 0.0"
        );
        Ok(())
    }

    /// Cleans, verifies, deduplicates, credits, and pairs raw coding-agent trajectories into a
    /// [`BuiltTraceDataset`].
    pub fn build(&self, raw_traces: &[AgentTrace]) -> Result<BuiltTraceDataset> {
        self.validate()?;
        ensure_trace_tenant_isolation(&self.tenant_id, raw_traces)?;
        for trace in raw_traces {
            verify_spdx_license(&trace.source_license, "trace_source")?;
        }

        let detector = RewardHackingDetector::new(self.max_repeat_cycles);
        let mut stats = DatasetBuildStats {
            raw_traces: raw_traces.len(),
            ..Default::default()
        };

        // 1. Secret/PII redaction and deterministic SHA-256 trajectory deduplication.
        let mut seen_hashes: BTreeSet<[u8; 32]> = BTreeSet::new();
        let mut unique_traces = Vec::with_capacity(raw_traces.len());
        for raw in raw_traces {
            let (clean_trace, redacted_n) = sanitize_trace(raw);
            stats.redacted_secret_spans += redacted_n;
            let fp = trace_dedup_fingerprint(&clean_trace);
            if seen_hashes.insert(fp) {
                unique_traces.push(clean_trace);
            }
        }
        stats.deduplicated_traces = unique_traces.len();

        // 2. Partition into verified clean vs reward-hacked vs honest failed per task_id.
        let mut verified_by_task: BTreeMap<String, Vec<AgentTrace>> = BTreeMap::new();
        let mut rejected_by_task: BTreeMap<String, Vec<(AgentTrace, bool)>> = BTreeMap::new();

        for trace in unique_traces {
            let verdict = detector.inspect(&trace);
            if verdict.is_hacked() {
                stats.reward_hacked_traces += 1;
                rejected_by_task
                    .entry(trace.task_id.clone())
                    .or_default()
                    .push((trace, true));
            } else if detector.is_verified_success(&trace) {
                stats.verified_clean_traces += 1;
                verified_by_task
                    .entry(trace.task_id.clone())
                    .or_default()
                    .push(trace);
            } else {
                stats.honest_failed_traces += 1;
                rejected_by_task
                    .entry(trace.task_id.clone())
                    .or_default()
                    .push((trace, false));
            }
        }

        // 3. Compute discounted step-level credit with per-task group efficiency scaling.
        let mut candidate_trajectories: Vec<(usize, Vec<StepSupervisionRecord>, usize)> =
            Vec::new();
        for traces in verified_by_task.values() {
            let sum_len: usize = traces.iter().map(|t| t.steps.len()).sum();
            let mean_len = sum_len as f64 / traces.len() as f64;
            for trace in traces {
                let t_len = trace.steps.len();
                let efficiency_scale = (mean_len / t_len.max(1) as f64).clamp(0.65, 1.35);
                let mut trace_records = Vec::new();
                let mut pruned = 0usize;
                let mut kept_tokens = 0usize;

                for (idx, step) in trace.steps.iter().enumerate() {
                    let dist_to_goal = (t_len - 1 - idx) as i32;
                    let discount = self.discount_gamma.powi(dist_to_goal);
                    let base_quality = if step.exit_code != 0 {
                        0.0
                    } else if step.made_state_progress {
                        1.0
                    } else {
                        self.detour_step_weight
                    };
                    let weight = (base_quality * discount * efficiency_scale) as f32;
                    if weight > 1e-6 {
                        kept_tokens += step.token_count;
                        trace_records.push(StepSupervisionRecord {
                            task_id: trace.task_id.clone(),
                            trace_id: trace.trace_id.clone(),
                            step_index: step.step_index,
                            state_tokens: step.state_tokens.clone(),
                            action_id: step.action_id,
                            advantage_weight: weight,
                            token_count: step.token_count,
                        });
                    } else {
                        pruned += 1;
                    }
                }
                candidate_trajectories.push((kept_tokens, trace_records, pruned));
            }
        }

        // When a token budget cap is configured, prioritize concise high-density trajectories.
        if self.max_token_budget.is_some() {
            candidate_trajectories.sort_by_key(|(toks, recs, _)| (*toks / recs.len().max(1), *toks));
        }

        let mut sft_records = Vec::new();
        for (kept_tokens, recs, pruned) in candidate_trajectories {
            if recs.is_empty() {
                stats.pruned_detour_steps += pruned;
                continue;
            }
            if let Some(budget) = self.max_token_budget {
                if stats.supervised_tokens + kept_tokens > budget {
                    continue;
                }
            }
            stats.supervised_tokens += kept_tokens;
            stats.pruned_detour_steps += pruned;
            sft_records.extend(recs);
        }
        stats.sft_step_records = sft_records.len();

        // 4. Extract minimally-divergent contrastive preference pairs on shared task states.
        let mut pair_keys: BTreeSet<(String, Vec<u32>, u32, u32)> = BTreeSet::new();
        let mut preference_pairs = Vec::new();

        for (task_id, wins) in &verified_by_task {
            let Some(losers) = rejected_by_task.get(task_id) else {
                continue;
            };
            for win in wins {
                for (lose, is_hack) in losers {
                    let margin_w = if *is_hack {
                        self.hack_margin_weight
                    } else {
                        self.fail_margin_weight
                    };

                    // First check exact prefix divergence step t*.
                    let common = win.steps.len().min(lose.steps.len());
                    let mut paired_any = false;
                    for idx in 0..common {
                        let ws = &win.steps[idx];
                        let ls = &lose.steps[idx];
                        if ws.state_tokens == ls.state_tokens && ws.action_id != ls.action_id {
                            if ws.made_state_progress && ws.exit_code == 0 {
                                let key = (
                                    task_id.clone(),
                                    ws.state_tokens.clone(),
                                    ws.action_id,
                                    ls.action_id,
                                );
                                if pair_keys.insert(key) {
                                    preference_pairs.push(PreferencePairRecord {
                                        task_id: task_id.clone(),
                                        divergence_step: idx,
                                        state_tokens: ws.state_tokens.clone(),
                                        chosen_action: ws.action_id,
                                        rejected_action: ls.action_id,
                                        margin_weight: margin_w,
                                        rejected_was_hack: *is_hack,
                                        token_count: ws.token_count,
                                    });
                                }
                                paired_any = true;
                            }
                            break;
                        }
                    }

                    // Fallback: match any shared state where `win` progressed and `lose` failed/hacked.
                    if !paired_any {
                        for ws in &win.steps {
                            if !ws.made_state_progress || ws.exit_code != 0 {
                                continue;
                            }
                            for (l_idx, ls) in lose.steps.iter().enumerate() {
                                if ls.state_tokens == ws.state_tokens
                                    && ls.action_id != ws.action_id
                                    && (!ls.made_state_progress || ls.exit_code != 0 || *is_hack)
                                {
                                    let key = (
                                        task_id.clone(),
                                        ws.state_tokens.clone(),
                                        ws.action_id,
                                        ls.action_id,
                                    );
                                    if pair_keys.insert(key) {
                                        preference_pairs.push(PreferencePairRecord {
                                            task_id: task_id.clone(),
                                            divergence_step: l_idx,
                                            state_tokens: ws.state_tokens.clone(),
                                            chosen_action: ws.action_id,
                                            rejected_action: ls.action_id,
                                            margin_weight: margin_w,
                                            rejected_was_hack: *is_hack,
                                            token_count: ws.token_count,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        stats.preference_pairs = preference_pairs.len();

        Ok(BuiltTraceDataset {
            tenant_id: self.tenant_id.clone(),
            sft_records,
            preference_pairs,
            stats,
        })
    }
}

/// Baseline A dataset builder: naive outcome SFT trusting raw `reported_exit_code == 0` without
/// [`RewardHackingDetector`], trajectory deduplication, step-credit pruning, or contrastive DPO.
pub fn build_unfiltered_sft_dataset(
    tenant_id: &str,
    base_model_license: &str,
    raw_traces: &[AgentTrace],
) -> Result<BuiltTraceDataset> {
    verify_spdx_license(base_model_license, "base_model")?;
    ensure_trace_tenant_isolation(tenant_id, raw_traces)?;
    for trace in raw_traces {
        verify_spdx_license(&trace.source_license, "trace_source")?;
    }

    let mut stats = DatasetBuildStats {
        raw_traces: raw_traces.len(),
        deduplicated_traces: raw_traces.len(),
        ..Default::default()
    };
    let mut sft_records = Vec::new();
    for raw in raw_traces {
        let (clean, red_n) = sanitize_trace(raw);
        stats.redacted_secret_spans += red_n;
        if clean.outcome.reported_exit_code == 0 {
            for step in &clean.steps {
                stats.supervised_tokens += step.token_count;
                sft_records.push(StepSupervisionRecord {
                    task_id: clean.task_id.clone(),
                    trace_id: clean.trace_id.clone(),
                    step_index: step.step_index,
                    state_tokens: step.state_tokens.clone(),
                    action_id: step.action_id,
                    advantage_weight: 1.0,
                    token_count: step.token_count,
                });
            }
        }
    }
    stats.sft_step_records = sft_records.len();

    Ok(BuiltTraceDataset {
        tenant_id: tenant_id.to_string(),
        sft_records,
        preference_pairs: Vec::new(),
        stats,
    })
}

/// Baseline B dataset builder: trajectory-level Rejection Fine-Tuning keeping only
/// [`RewardHackingDetector`]-verified clean trajectories with uniform step weight `1.0` across all
/// steps (no step-credit detour pruning and no contrastive preference pairs).
pub fn build_filtered_rft_dataset(
    tenant_id: &str,
    base_model_license: &str,
    raw_traces: &[AgentTrace],
    max_repeat_cycles: usize,
) -> Result<BuiltTraceDataset> {
    verify_spdx_license(base_model_license, "base_model")?;
    ensure_trace_tenant_isolation(tenant_id, raw_traces)?;
    for trace in raw_traces {
        verify_spdx_license(&trace.source_license, "trace_source")?;
    }

    let detector = RewardHackingDetector::new(max_repeat_cycles);
    let mut stats = DatasetBuildStats {
        raw_traces: raw_traces.len(),
        deduplicated_traces: raw_traces.len(),
        ..Default::default()
    };
    let mut sft_records = Vec::new();
    for raw in raw_traces {
        let (clean, red_n) = sanitize_trace(raw);
        stats.redacted_secret_spans += red_n;
        if detector.inspect(&clean).is_hacked() {
            stats.reward_hacked_traces += 1;
        } else if detector.is_verified_success(&clean) {
            stats.verified_clean_traces += 1;
            for step in &clean.steps {
                stats.supervised_tokens += step.token_count;
                sft_records.push(StepSupervisionRecord {
                    task_id: clean.task_id.clone(),
                    trace_id: clean.trace_id.clone(),
                    step_index: step.step_index,
                    state_tokens: step.state_tokens.clone(),
                    action_id: step.action_id,
                    advantage_weight: 1.0,
                    token_count: step.token_count,
                });
            }
        } else {
            stats.honest_failed_traces += 1;
        }
    }
    stats.sft_step_records = sft_records.len();

    Ok(BuiltTraceDataset {
        tenant_id: tenant_id.to_string(),
        sft_records,
        preference_pairs: Vec::new(),
        stats,
    })
}

/// Hyperparameters and architectural dimensions for [`ZenTracePolicy`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentLightningConfig {
    pub base_model_license: String,
    pub vocab_size: usize,
    pub num_actions: usize,
    pub hidden_dim: usize,
    pub seq_len: usize,
    pub dpo_beta: f64,
    pub dpo_loss_weight: f64,
    pub learning_rate: f64,
    pub steps: usize,
    pub batch_size: usize,
}

impl AgentLightningConfig {
    pub fn validate(&self) -> Result<()> {
        verify_spdx_license(&self.base_model_license, "base_model")?;
        ensure!(self.vocab_size >= 16, "vocab_size must be >= 16");
        ensure!(self.num_actions >= 2, "num_actions must be >= 2");
        ensure!(self.hidden_dim >= 8, "hidden_dim must be >= 8");
        ensure!(self.seq_len >= 2, "seq_len must be >= 2");
        ensure!(self.dpo_beta > 0.0, "dpo_beta must be > 0.0");
        ensure!(
            self.dpo_loss_weight >= 0.0,
            "dpo_loss_weight must be >= 0.0"
        );
        ensure!(self.learning_rate > 0.0, "learning_rate must be > 0.0");
        ensure!(self.batch_size >= 1, "batch_size must be >= 1");
        Ok(())
    }
}

/// Position-aware Zen micro-policy on `hanzo-ml` / `hanzo-nn` mapping tokenized trajectory states
/// to tool-action logits.
pub struct ZenTracePolicy {
    tok_embed: Embedding,
    pos_embed: Embedding,
    in_proj: Linear,
    mid_proj: Linear,
    act_head: Linear,
    hidden_dim: usize,
    seq_len: usize,
}

impl ZenTracePolicy {
    pub fn new(vb: VarBuilder, cfg: &AgentLightningConfig) -> hanzo_ml::Result<Self> {
        let h = cfg.hidden_dim;
        let s = cfg.seq_len;
        let tok_embed = embedding(cfg.vocab_size, h, vb.pp("tok_embed"))?;
        let pos_embed = embedding(s, h, vb.pp("pos_embed"))?;
        let in_proj = linear(s * h, h, vb.pp("in_proj"))?;
        let mid_proj = linear(h, h, vb.pp("mid_proj"))?;
        let act_head = linear(h, cfg.num_actions, vb.pp("act_head"))?;
        Ok(Self {
            tok_embed,
            pos_embed,
            in_proj,
            mid_proj,
            act_head,
            hidden_dim: h,
            seq_len: s,
        })
    }

    /// Computes action logits `[B, num_actions]` for batched state token sequences `[B, seq_len]`.
    pub fn forward(&self, states: &Tensor) -> hanzo_ml::Result<Tensor> {
        let (b, s) = states.dims2()?;
        if s != self.seq_len {
            hanzo_ml::bail!(
                "state seq_len mismatch: expected {}, got {}",
                self.seq_len,
                s
            );
        }
        let tok_e = self.tok_embed.forward(states)?;
        let pos_ids = Tensor::arange(0u32, s as u32, states.device())?;
        let pos_e = self.pos_embed.forward(&pos_ids)?.unsqueeze(0)?;
        let encoded = tok_e
            .broadcast_add(&pos_e)?
            .reshape((b, s * self.hidden_dim))?;
        let h1 = self.in_proj.forward(&encoded)?.silu()?;
        let h2 = (&self.mid_proj.forward(&h1)? + &h1)?.silu()?;
        self.act_head.forward(&h2)
    }
}

/// Counts scalar parameters stored in `varmap`.
pub fn count_parameters(varmap: &VarMap) -> usize {
    let data = varmap.data().lock().expect("varmap lock");
    data.values().map(|v| v.as_tensor().elem_count()).sum()
}

fn pad_or_truncate_state(tokens: &[u32], seq_len: usize, vocab_size: usize) -> Result<Vec<u32>> {
    let mut out = vec![0u32; seq_len];
    for (i, &tok) in tokens.iter().take(seq_len).enumerate() {
        ensure!(
            (tok as usize) < vocab_size,
            "state token id {} exceeds vocab_size {}",
            tok,
            vocab_size
        );
        out[i] = tok;
    }
    Ok(out)
}

fn batch_sft_tensors(
    records: &[StepSupervisionRecord],
    indices: &[usize],
    cfg: &AgentLightningConfig,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor, usize)> {
    let b = indices.len();
    let s = cfg.seq_len;
    let mut states_flat = Vec::with_capacity(b * s);
    let mut actions = Vec::with_capacity(b);
    let mut weights = Vec::with_capacity(b);
    let mut tokens_sum = 0usize;

    for &idx in indices {
        let rec = &records[idx];
        ensure!(
            (rec.action_id as usize) < cfg.num_actions,
            "action_id {} exceeds num_actions {}",
            rec.action_id,
            cfg.num_actions
        );
        states_flat.extend(pad_or_truncate_state(
            &rec.state_tokens,
            s,
            cfg.vocab_size,
        )?);
        actions.push(rec.action_id);
        weights.push(rec.advantage_weight);
        tokens_sum += rec.token_count;
    }

    Ok((
        Tensor::from_vec(states_flat, (b, s), dev)?,
        Tensor::from_vec(actions, b, dev)?,
        Tensor::from_vec(weights, b, dev)?,
        tokens_sum,
    ))
}

fn batch_dpo_tensors(
    pairs: &[PreferencePairRecord],
    indices: &[usize],
    cfg: &AgentLightningConfig,
    dev: &Device,
) -> Result<(Tensor, Tensor, Tensor, Tensor, usize)> {
    let b = indices.len();
    let s = cfg.seq_len;
    let mut states_flat = Vec::with_capacity(b * s);
    let mut chosen = Vec::with_capacity(b);
    let mut rejected = Vec::with_capacity(b);
    let mut weights = Vec::with_capacity(b);
    let mut tokens_sum = 0usize;

    for &idx in indices {
        let pair = &pairs[idx];
        ensure!(
            (pair.chosen_action as usize) < cfg.num_actions
                && (pair.rejected_action as usize) < cfg.num_actions,
            "preference pair action exceeds num_actions {}",
            cfg.num_actions
        );
        states_flat.extend(pad_or_truncate_state(
            &pair.state_tokens,
            s,
            cfg.vocab_size,
        )?);
        chosen.push(pair.chosen_action);
        rejected.push(pair.rejected_action);
        weights.push(pair.margin_weight);
        tokens_sum += pair.token_count;
    }

    Ok((
        Tensor::from_vec(states_flat, (b, s), dev)?,
        Tensor::from_vec(chosen, b, dev)?,
        Tensor::from_vec(rejected, b, dev)?,
        Tensor::from_vec(weights, b, dev)?,
        tokens_sum,
    ))
}

/// Summary of optimization cost and checkpoint metadata returned by [`train_zen_trace_policy`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainingSummary {
    pub param_count: usize,
    pub train_steps: usize,
    pub dataset_supervised_tokens: usize,
    pub total_train_tokens_processed: usize,
    pub train_time_ms: f64,
    pub ms_per_step: f64,
    pub checkpoint_bytes: u64,
}

/// Trains a [`ZenTracePolicy`] on `dataset` using [`crate::adam::Adam`], combining step-level
/// advantage-weighted NLL with numerically stable contrastive preference optimization when
/// `dataset.preference_pairs` is non-empty and `cfg.dpo_loss_weight > 0.0`.
pub fn train_zen_trace_policy(
    dataset: &BuiltTraceDataset,
    cfg: &AgentLightningConfig,
    checkpoint_out: Option<&Path>,
    seed: u64,
) -> Result<(VarMap, ZenTracePolicy, TrainingSummary)> {
    cfg.validate()?;
    ensure!(
        !dataset.sft_records.is_empty(),
        "dataset.sft_records must not be empty"
    );

    let dev = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = ZenTracePolicy::new(vb, cfg)?;

    let hyper = Hyper {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        decay: 0.01,
        clip: 1.0,
    };
    let mut opt = Adam::new(&varmap, |_| Some(cfg.learning_rate), hyper)?;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut sft_order: Vec<usize> = (0..dataset.sft_records.len()).collect();
    sft_order.shuffle(&mut rng);
    let mut sft_cursor = 0usize;

    let mut dpo_order: Vec<usize> = (0..dataset.preference_pairs.len()).collect();
    if !dpo_order.is_empty() {
        dpo_order.shuffle(&mut rng);
    }
    let mut dpo_cursor = 0usize;

    let mut total_tokens_processed = 0usize;
    let start = Instant::now();

    for _step in 0..cfg.steps {
        let sft_batch_len = cfg.batch_size.min(sft_order.len());
        if sft_cursor + sft_batch_len > sft_order.len() {
            sft_order.shuffle(&mut rng);
            sft_cursor = 0;
        }
        let sft_idx = &sft_order[sft_cursor..sft_cursor + sft_batch_len];
        sft_cursor += sft_batch_len;

        let (states_t, actions_t, weights_t, sft_toks) =
            batch_sft_tensors(&dataset.sft_records, sft_idx, cfg, &dev)?;
        total_tokens_processed += sft_toks;

        let logits = model.forward(&states_t)?;
        let log_probs = ops::log_softmax(&logits, 1)?;
        let chosen_lp = log_probs.gather(&actions_t.unsqueeze(1)?, 1)?.squeeze(1)?;
        let step_nll = chosen_lp.affine(-1.0, 0.0)?;
        let rft_loss = (&step_nll * &weights_t)?.mean_all()?;

        let total_loss = if !dpo_order.is_empty() && cfg.dpo_loss_weight > 0.0 {
            let dpo_batch_len = (cfg.batch_size / 2).max(1).min(dpo_order.len());
            if dpo_cursor + dpo_batch_len > dpo_order.len() {
                dpo_order.shuffle(&mut rng);
                dpo_cursor = 0;
            }
            let dpo_idx = &dpo_order[dpo_cursor..dpo_cursor + dpo_batch_len];
            dpo_cursor += dpo_batch_len;

            let (p_states, p_chosen, p_rejected, p_weights, dpo_toks) =
                batch_dpo_tensors(&dataset.preference_pairs, dpo_idx, cfg, &dev)?;
            total_tokens_processed += dpo_toks;

            let p_logits = model.forward(&p_states)?;
            let p_lp = ops::log_softmax(&p_logits, 1)?;
            let c_lp = p_lp.gather(&p_chosen.unsqueeze(1)?, 1)?.squeeze(1)?;
            let r_lp = p_lp.gather(&p_rejected.unsqueeze(1)?, 1)?.squeeze(1)?;
            let scaled_margin = ((&c_lp - &r_lp)? * cfg.dpo_beta)?;
            let neg_m = scaled_margin.affine(-1.0, 0.0)?;
            // Numerically stable -log(sigmoid(m)) = relu(-m) + log(1 + exp(-|m|)).
            let softplus_neg = (&neg_m.relu()?
                + &neg_m
                    .abs()?
                    .affine(-1.0, 0.0)?
                    .exp()?
                    .affine(1.0, 1.0)?
                    .log()?)?;
            let dpo_loss = (&softplus_neg * &p_weights)?.mean_all()?;
            (&rft_loss + &(dpo_loss * cfg.dpo_loss_weight)?)?
        } else {
            rft_loss
        };

        let grads = total_loss.backward()?;
        opt.step(&grads)?;
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut ckpt_bytes = 0u64;
    if let Some(path) = checkpoint_out {
        verify_spdx_license(&cfg.base_model_license, "checkpoint_export")?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("create checkpoint parent directory")?;
        }
        varmap.save(path).context("save safetensors checkpoint")?;
        ckpt_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    }

    let steps = cfg.steps.max(1);
    let summary = TrainingSummary {
        param_count: count_parameters(&varmap),
        train_steps: cfg.steps,
        dataset_supervised_tokens: dataset.stats.supervised_tokens,
        total_train_tokens_processed: total_tokens_processed,
        train_time_ms: elapsed_ms,
        ms_per_step: elapsed_ms / steps as f64,
        checkpoint_bytes: ckpt_bytes,
    };
    Ok((varmap, model, summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_clean_trace() -> AgentTrace {
        AgentTrace {
            trace_id: "tr-1".to_string(),
            tenant_id: "org-zen".to_string(),
            task_id: "task-bounds-1".to_string(),
            source_license: "Apache-2.0".to_string(),
            prompt_text: "Fix slice bounds check in parser".to_string(),
            task_forbids_test_edits: true,
            requires_code_change: true,
            steps: vec![
                TraceStep {
                    step_index: 0,
                    state_tokens: vec![4, 8, 48, 56],
                    action_id: 0,
                    tool_kind: ToolKind::ReadSpan,
                    command_or_target: "src/parser.rs:10-40".to_string(),
                    tool_output_excerpt: "found off-by-one index".to_string(),
                    exit_code: 0,
                    lines_added: 0,
                    lines_removed: 0,
                    added_assertions: 0,
                    removed_assertions: 0,
                    touched_test_files: vec![],
                    made_state_progress: true,
                    token_count: 32,
                },
                TraceStep {
                    step_index: 1,
                    state_tokens: vec![4, 8, 49, 57],
                    action_id: 4,
                    tool_kind: ToolKind::ApplyPatch,
                    command_or_target: "src/parser.rs".to_string(),
                    tool_output_excerpt: "patched bounds guard".to_string(),
                    exit_code: 0,
                    lines_added: 3,
                    lines_removed: 1,
                    added_assertions: 0,
                    removed_assertions: 0,
                    touched_test_files: vec![],
                    made_state_progress: true,
                    token_count: 48,
                },
                TraceStep {
                    step_index: 2,
                    state_tokens: vec![4, 8, 50, 60],
                    action_id: 8,
                    tool_kind: ToolKind::RunTests,
                    command_or_target: "cargo test -p parser".to_string(),
                    tool_output_excerpt: "12 passed; 0 failed".to_string(),
                    exit_code: 0,
                    lines_added: 0,
                    lines_removed: 0,
                    added_assertions: 0,
                    removed_assertions: 0,
                    touched_test_files: vec![],
                    made_state_progress: true,
                    token_count: 24,
                },
            ],
            outcome: VerifierOutcome {
                reported_exit_code: 0,
                verifier_passed: true,
                tests_ran: 12,
                tests_failed: 0,
                substantive_source_changes: 4,
            },
        }
    }

    #[test]
    fn spdx_license_gate_enforces_permissive_allowlist() {
        assert!(verify_spdx_license("Apache-2.0", "model").is_ok());
        assert!(verify_spdx_license("MIT", "model").is_ok());
        assert!(verify_spdx_license("CC-BY-4.0", "trace").is_ok());
        assert!(verify_spdx_license("MIT OR Apache-2.0", "model").is_ok());
        assert!(verify_spdx_license("GPL-3.0-only", "model").is_err());
        assert!(verify_spdx_license("CC-BY-NC-4.0", "trace").is_err());
        assert!(verify_spdx_license("", "model").is_err());
    }

    #[test]
    fn tenant_isolation_blocks_cross_tenant_traces() {
        let t1 = sample_clean_trace();
        let mut t2 = sample_clean_trace();
        t2.trace_id = "tr-2".to_string();
        t2.tenant_id = "org-other".to_string();
        let err = ensure_trace_tenant_isolation("org-zen", &[t1, t2]).unwrap_err();
        assert!(err.to_string().contains("tenant isolation violation"));
    }

    #[test]
    fn secret_redactor_scrubs_credentials_and_emails() {
        let synthetic_api = format!("{}{}", "sk-", "synthetic0000test9999");
        let synthetic_bearer = format!("Bearer {}", "tok_synthetic_0000_abcd");
        let pem_header = format!("-----BEGIN {}-----", "TEST PRIVATE KEY");
        let pem_footer = format!("-----END {}-----", "TEST PRIVATE KEY");
        let raw = format!(
            "contact ops.user@example.org with {synthetic_api} or {synthetic_bearer} db postgres://admin:pass123@db.internal:5432/app {pem_header}body{pem_footer}"
        );
        let (clean, count) = redact_text_secrets(&raw);
        assert!(count >= 5, "expected >= 5 redactions, got {count}: {clean}");
        assert!(!clean.contains("synthetic0000"));
        assert!(!clean.contains("tok_synthetic"));
        assert!(!clean.contains("ops.user@example.org"));
        assert!(!clean.contains("pass123"));
        assert!(clean.contains("[REDACTED_EMAIL]"));
        assert!(clean.contains("[REDACTED_SECRET]"));
        assert!(clean.contains("[REDACTED_CREDENTIAL]"));
        assert!(clean.contains("[REDACTED_PRIVATE_KEY]"));
    }

    #[test]
    fn detector_catches_all_five_reward_hacking_modes() {
        let detector = RewardHackingDetector::new(3);
        let clean = sample_clean_trace();
        assert!(detector.is_verified_success(&clean));

        // 1. Assertion deletion
        let mut del = clean.clone();
        del.steps[1].removed_assertions = 2;
        let has_del = detector
            .inspect(&del)
            .violations
            .iter()
            .any(|v| matches!(v, RewardHackViolation::AssertionDeletion { .. }));
        assert!(has_del);

        // 2. Forbidden test-file modification
        let mut edit_test = clean.clone();
        edit_test.steps[1].touched_test_files = vec!["tests/parser_spec.rs".to_string()];
        let has_forbidden = detector
            .inspect(&edit_test)
            .violations
            .iter()
            .any(|v| matches!(v, RewardHackViolation::ForbiddenTestModification { .. }));
        assert!(has_forbidden);

        // 3. Trivial exit-code spoofing
        let mut spoof = clean.clone();
        spoof.steps[2].command_or_target = "cargo test -p parser || true".to_string();
        let has_spoof = detector
            .inspect(&spoof)
            .violations
            .iter()
            .any(|v| matches!(v, RewardHackViolation::ExitCodeSpoofing { .. }));
        assert!(has_spoof);

        // 4. Empty / tautological patch
        let mut empty = clean.clone();
        empty.outcome.substantive_source_changes = 0;
        let has_empty = detector
            .inspect(&empty)
            .violations
            .contains(&RewardHackViolation::EmptyOrTautologicalPatch);
        assert!(has_empty);

        // 5. Circular non-progress retry loop
        let mut looped = clean;
        let retry_step = TraceStep {
            step_index: 3,
            state_tokens: vec![4, 8, 48, 56],
            action_id: 9,
            tool_kind: ToolKind::SearchCode,
            command_or_target: "grep -rn foo src/".to_string(),
            tool_output_excerpt: "no match".to_string(),
            exit_code: 0,
            lines_added: 0,
            lines_removed: 0,
            added_assertions: 0,
            removed_assertions: 0,
            touched_test_files: vec![],
            made_state_progress: false,
            token_count: 20,
        };
        looped.steps = vec![retry_step.clone(), retry_step.clone(), retry_step];
        let has_loop = detector
            .inspect(&looped)
            .violations
            .iter()
            .any(|v| matches!(v, RewardHackViolation::CircularRetryLoop { .. }));
        assert!(has_loop);
    }
}
