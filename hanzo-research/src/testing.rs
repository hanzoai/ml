//! Anytime-valid adaptive testing, paired statistical comparisons, and dependency scoping.
//!
//! Rust peer of `hanzo_research.testing`: implements geometric look schedules, exact
//! paired Student-t one-sided p-values, alpha-spent progressive A/B screening, and
//! conservative dependency-scoped regression routing (`impacted_benchmarks`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::protocol::{
    paired_comparison, CorrectnessChecks, ExperimentRecordV2, LocalResearchLedger,
    StatisticalComparison, SyncReceipt,
};
use crate::{Error, Result};

pub const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

/// Fixed geometric look schedule `initial -> 2*initial -> ... -> n`.
pub fn planned_looks(n: usize, initial: usize) -> Result<Vec<usize>> {
    if n == 0 || initial == 0 {
        return Err(Error::Message("n and initial must be positive".into()));
    }
    let mut sizes = Vec::new();
    let mut size = initial.min(n);
    while size < n {
        sizes.push(size);
        size = (size * 2).min(n);
    }
    sizes.push(n);
    Ok(sizes)
}

/// Deterministic seed-keyed stratified ordering matching Python `stable_order`.
pub fn stable_order(ids: &[String], seed: u64, strata: Option<&[String]>) -> Result<Vec<usize>> {
    let unique: BTreeSet<&str> = ids.iter().map(String::as_str).collect();
    if unique.len() != ids.len() {
        return Err(Error::Message(
            "example IDs must be unique for paired evaluation".into(),
        ));
    }
    if let Some(st) = strata {
        if st.len() != ids.len() {
            return Err(Error::Message("strata must match example IDs".into()));
        }
    }
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for i in 0..ids.len() {
        let key = strata
            .map(|s| s[i].clone())
            .unwrap_or_else(|| "__all__".to_string());
        groups.entry(key).or_default().push(i);
    }
    for group in groups.values_mut() {
        group.sort_by_key(|&i| {
            let input = format!("{seed}\x1f{}", ids[i]);
            format!("{:x}", Sha256::digest(input.as_bytes()))
        });
    }
    let labels: Vec<String> = groups.keys().cloned().collect();
    let mut out = Vec::with_capacity(ids.len());
    while groups.values().any(|g| !g.is_empty()) {
        for label in &labels {
            if let Some(idx) = groups.get_mut(label).and_then(Vec::pop) {
                out.push(idx);
            }
        }
    }
    Ok(out)
}

/// Exact upper-tail P(T_df >= t_stat) for integer df >= 1.
pub fn student_t_one_sided_p(t_stat: f64, df: usize) -> Result<f64> {
    if df == 0 {
        return Err(Error::Message("df must be >= 1".into()));
    }
    if t_stat.is_nan() {
        return Err(Error::Message("t_stat cannot be NaN".into()));
    }
    if t_stat.is_infinite() {
        return Ok(if t_stat > 0.0 { 0.0 } else { 1.0 });
    }
    let z = t_stat.abs() / (df as f64).sqrt();
    let u = 1.0 / (1.0 + z * z);
    let inner = if df == 1 {
        (2.0 / std::f64::consts::PI) * z.atan()
    } else if df % 2 == 1 {
        let m = (df - 1) / 2;
        let mut term = 1.0;
        let mut acc = 1.0;
        for k in 1..m {
            term *= (2.0 * k as f64 / (2.0 * k as f64 + 1.0)) * u;
            acc += term;
        }
        (2.0 / std::f64::consts::PI) * (z.atan() + z * u * acc)
    } else {
        let m = df / 2;
        let mut term = 1.0;
        let mut acc = 1.0;
        for k in 1..m {
            term *= ((2.0 * k as f64 - 1.0) / (2.0 * k as f64)) * u;
            acc += term;
        }
        z * u.sqrt() * acc
    };
    let p_two = (1.0 - inner).clamp(0.0, 1.0);
    Ok(if t_stat >= 0.0 {
        0.5 * p_two
    } else {
        1.0 - 0.5 * p_two
    })
}

fn lgamma_approx(x: f64) -> f64 {
    // Lanczos approximation for log(Gamma(x)), accurate to >14 digits for x >= 1.
    let g = 7.0;
    let p = [
        0.9999999999998099,
        676.5203681218851,
        -1259.1392167224028,
        771.3234287776531,
        -176.6150291621406,
        12.507343278686905,
        -0.13857109526572012,
        9.984369578019572e-6,
        1.5056327351493116e-7,
    ];
    let z = x - 1.0;
    let mut a = p[0];
    for (i, &coef) in p.iter().enumerate().skip(1) {
        a += coef / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + a.ln()
}

fn binom_upper_tail(successes: usize, trials: usize) -> f64 {
    if successes == 0 {
        return 1.0;
    }
    if successes > trials {
        return 0.0;
    }
    let logs: Vec<f64> = (successes..=trials)
        .map(|i| {
            lgamma_approx((trials + 1) as f64)
                - lgamma_approx((i + 1) as f64)
                - lgamma_approx((trials - i + 1) as f64)
                - (trials as f64) * std::f64::consts::LN_2
        })
        .collect();
    let peak = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sum_exp: f64 = logs.iter().map(|&v| (v - peak).exp()).sum();
    (peak.exp() * sum_exp).min(1.0)
}

/// Two-sided Beta-Binomial e-value under H0: p = 0.5 on discordant counts (b, c).
pub fn beta_binomial_e_value(b: usize, c: usize, a0: f64, b0: f64) -> f64 {
    let d = b + c;
    if d == 0 {
        return 1.0;
    }
    let log_num = lgamma_approx(b as f64 + a0) + lgamma_approx(c as f64 + b0)
        - lgamma_approx(d as f64 + a0 + b0);
    let log_den = lgamma_approx(a0) + lgamma_approx(b0)
        - lgamma_approx(a0 + b0)
        - (d as f64) * std::f64::consts::LN_2;
    (log_num - log_den).clamp(-700.0, 700.0).exp()
}

/// Exact nonnegative H0-martingale mixture-SPRT + Beta(0.5, 0.5) e-value for H+: p > 0.5.
pub fn one_sided_mixture_e_value(wins: usize, losses: usize) -> f64 {
    let d = wins + losses;
    if d == 0 {
        return 1.0;
    }
    let grid: [(f64, f64); 4] = [(0.35, 0.56), (0.30, 0.62), (0.20, 0.75), (0.15, 0.97)];
    let ln2 = std::f64::consts::LN_2;
    let logs: [f64; 4] = grid.map(|(w, p)| {
        w.ln() + (wins as f64) * (p.ln() + ln2) + (losses as f64) * ((1.0 - p).ln() + ln2)
    });
    let peak = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e_mix = peak.min(700.0).exp() * logs.iter().map(|&v| (v - peak).exp()).sum::<f64>();
    0.5 * e_mix + 0.5 * beta_binomial_e_value(wins, losses, 0.5, 0.5)
}

const EMPTY_BYTES_SHA256: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

/// Validate an authentic content-addressed dataset manifest SHA-256 digest.
pub fn validate_dataset_manifest_sha256(digest_str: &str) -> Result<String> {
    let trimmed = digest_str.trim().to_ascii_lowercase();
    let raw = trimmed.strip_prefix("sha256:").unwrap_or(&trimmed);
    if raw.len() != 64 || !raw.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Message(
            "dataset_manifest_sha256 must be a valid 64-hex SHA-256 digest".into(),
        ));
    }
    let uniq: BTreeSet<u8> = raw.bytes().collect();
    if raw == EMPTY_BYTES_SHA256 || uniq.len() <= 1 {
        return Err(Error::Message(
            "dataset_manifest_sha256 cannot be empty-bytes or synthetic repeated-char placeholder"
                .into(),
        ));
    }
    Ok(raw.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedTraceCheck {
    pub n: usize,
    pub candidate_bools: Vec<bool>,
    pub baseline_bools: Vec<bool>,
    pub example_ids: Vec<String>,
    pub dataset_manifest_sha256: String,
    pub paired_trace_sha256: String,
}

/// Validate per-example paired binary outcomes and reject aggregate-rate pseudo-arrays.
pub fn validate_paired_binary_trace(
    candidate_outcomes: &[bool],
    baseline_outcomes: &[bool],
    example_ids: Option<&[String]>,
    dataset_manifest_sha256: Option<&str>,
    require_example_ids: bool,
) -> Result<PairedTraceCheck> {
    if candidate_outcomes.is_empty() || candidate_outcomes.len() != baseline_outcomes.len() {
        return Err(Error::Message(
            "paired binary traces must be non-empty and equal length".into(),
        ));
    }
    let n = candidate_outcomes.len();
    let mut cleaned_ids = Vec::new();
    let mut manifest_hex = String::new();

    if require_example_ids {
        let Some(eids) = example_ids.filter(|ids| ids.len() == n) else {
            return Err(Error::Message(
                "require_example_ids=true requires example_ids aligned to every paired outcome"
                    .into(),
            ));
        };
        cleaned_ids = eids.iter().map(|s| s.trim().to_string()).collect();
        let uniq: BTreeSet<&str> = cleaned_ids.iter().map(String::as_str).collect();
        if cleaned_ids.iter().any(|s| s.is_empty()) || uniq.len() != n {
            return Err(Error::Message(
                "example_ids must be non-empty and strictly distinct for paired evaluation".into(),
            ));
        }
        manifest_hex = validate_dataset_manifest_sha256(dataset_manifest_sha256.unwrap_or(""))?;

        let pairs: Vec<(bool, bool)> = candidate_outcomes
            .iter()
            .copied()
            .zip(baseline_outcomes.iter().copied())
            .collect();
        let distinct_states: BTreeSet<(bool, bool)> = pairs.iter().copied().collect();
        let pair_runs = 1 + pairs.windows(2).filter(|w| w[0] != w[1]).count();
        let cand_runs = 1 + candidate_outcomes
            .windows(2)
            .filter(|w| w[0] != w[1])
            .count();
        let base_runs = 1 + baseline_outcomes
            .windows(2)
            .filter(|w| w[0] != w[1])
            .count();
        let cand_ones = candidate_outcomes.iter().filter(|&&x| x).count();
        let base_ones = baseline_outcomes.iter().filter(|&&x| x).count();

        if n >= 16
            && distinct_states.len() >= 2
            && (pair_runs <= distinct_states.len()
                || (cand_ones > 0
                    && cand_ones < n
                    && cand_runs <= 2
                    && base_ones > 0
                    && base_ones < n
                    && base_runs <= 2))
        {
            return Err(Error::Message(
                "reconstructed aggregate-rate pseudo-array rejected: paired binary trace requires genuine per-example ordering".into(),
            ));
        }
    } else {
        if let Some(eids) = example_ids {
            if eids.len() != n {
                return Err(Error::Message(
                    "example_ids length must match paired outcomes".into(),
                ));
            }
            cleaned_ids = eids.iter().map(|s| s.trim().to_string()).collect();
            let uniq: BTreeSet<&str> = cleaned_ids.iter().map(String::as_str).collect();
            if cleaned_ids.iter().any(|s| s.is_empty()) || uniq.len() != n {
                return Err(Error::Message(
                    "example_ids must be non-empty and strictly distinct".into(),
                ));
            }
        }
        if let Some(m) = dataset_manifest_sha256.filter(|s| !s.trim().is_empty()) {
            manifest_hex = validate_dataset_manifest_sha256(m)?;
        }
    }

    let mut trace_buf = String::new();
    for i in 0..n {
        if i > 0 {
            trace_buf.push('\n');
        }
        let id_tok = if cleaned_ids.is_empty() {
            i.to_string()
        } else {
            cleaned_ids[i].clone()
        };
        trace_buf.push_str(&format!(
            "{id_tok}:{}:{}",
            u8::from(candidate_outcomes[i]),
            u8::from(baseline_outcomes[i])
        ));
    }
    let trace_sha = format!("{:x}", Sha256::digest(trace_buf.as_bytes()));
    Ok(PairedTraceCheck {
        n,
        candidate_bools: candidate_outcomes.to_vec(),
        baseline_bools: baseline_outcomes.to_vec(),
        example_ids: cleaned_ids,
        dataset_manifest_sha256: manifest_hex,
        paired_trace_sha256: trace_sha,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McNemarResult {
    pub method: String,
    pub n: usize,
    pub candidate_only: usize,
    pub baseline_only: usize,
    pub discordant: usize,
    pub candidate_accuracy: f64,
    pub baseline_accuracy: f64,
    pub delta: f64,
    pub effect_pct: f64,
    pub one_sided_p_better: f64,
    pub one_sided_p_worse: f64,
    pub two_sided_p: f64,
    pub e_value_better: f64,
    pub e_value_worse: f64,
    pub verdict: String,
    pub canonical_qualification: bool,
    pub dataset_manifest_sha256: String,
    pub paired_trace_sha256: String,
}

/// Exact conditional paired McNemar test with per-example trace provenance guard.
#[allow(clippy::too_many_arguments)]
pub fn paired_mcnemar(
    candidate_correct: &[bool],
    baseline_correct: &[bool],
    example_ids: Option<&[String]>,
    dataset_manifest_sha256: Option<&str>,
    require_example_ids: bool,
    alpha: f64,
    min_effect_pct: f64,
    promising_effect_pct: Option<f64>,
    futility_alpha: f64,
    canonical_qualification: bool,
) -> Result<McNemarResult> {
    if !(0.0 < alpha && alpha < 1.0) || !(0.0 < futility_alpha && futility_alpha < 1.0) {
        return Err(Error::Message(
            "alpha and futility_alpha must be in (0, 1)".into(),
        ));
    }
    let checked = validate_paired_binary_trace(
        candidate_correct,
        baseline_correct,
        example_ids,
        dataset_manifest_sha256,
        require_example_ids,
    )?;
    let n = checked.n;
    let mut only_cand = 0usize;
    let mut only_base = 0usize;
    for (&c, &b) in checked
        .candidate_bools
        .iter()
        .zip(checked.baseline_bools.iter())
    {
        if c && !b {
            only_cand += 1;
        } else if b && !c {
            only_base += 1;
        }
    }
    let discordant = only_cand + only_base;
    let delta = (only_cand as f64 - only_base as f64) / n as f64;
    let effect_pct = delta * 100.0;
    let p_better = binom_upper_tail(only_cand, discordant);
    let p_worse = binom_upper_tail(only_base, discordant);
    let p_two = (2.0 * p_better.min(p_worse)).min(1.0);
    let e_better = one_sided_mixture_e_value(only_cand, only_base);
    let e_worse = one_sided_mixture_e_value(only_base, only_cand);
    let eff_promising = promising_effect_pct.unwrap_or(0.5 * min_effect_pct);

    let verdict = if effect_pct >= min_effect_pct && p_better <= alpha {
        if canonical_qualification {
            "proven"
        } else {
            "promising_inconclusive"
        }
    } else if (effect_pct <= -min_effect_pct && p_worse <= alpha)
        || (delta < 0.0 && p_worse <= futility_alpha)
    {
        "refuted"
    } else if effect_pct >= eff_promising && p_worse > futility_alpha {
        "promising_inconclusive"
    } else {
        "inconclusive"
    };

    Ok(McNemarResult {
        method: "paired_exact_mcnemar".into(),
        n,
        candidate_only: only_cand,
        baseline_only: only_base,
        discordant,
        candidate_accuracy: checked.candidate_bools.iter().filter(|&&x| x).count() as f64
            / n as f64,
        baseline_accuracy: checked.baseline_bools.iter().filter(|&&x| x).count() as f64 / n as f64,
        delta,
        effect_pct,
        one_sided_p_better: p_better,
        one_sided_p_worse: p_worse,
        two_sided_p: p_two,
        e_value_better: e_better,
        e_value_worse: e_worse,
        verdict: verdict.into(),
        canonical_qualification,
        dataset_manifest_sha256: checked.dataset_manifest_sha256,
        paired_trace_sha256: checked.paired_trace_sha256,
    })
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AccuracyLook {
    pub decision: String,
    pub verdict: String,
    pub n: usize,
    pub look_index: usize,
    pub candidate_only: usize,
    pub jev_only: usize,
    pub discordant: usize,
    pub delta: f64,
    pub effect_pct: f64,
    pub e_value_beta_binomial: f64,
    pub e_value_better: f64,
    pub e_value_worse: f64,
    pub e_threshold: f64,
    pub one_sided_p_better: f64,
    pub one_sided_p_worse: f64,
    pub spent_alpha_each_direction: f64,
    pub canonical_qualification: bool,
}

pub fn paired_accuracy_look(
    candidate_correct: &[bool],
    baseline_correct: &[bool],
    look_index: usize,
    alpha: f64,
    min_samples: usize,
    min_effect: f64,
    task_arm_budget: usize,
) -> Result<AccuracyLook> {
    paired_accuracy_look_checked(
        candidate_correct,
        baseline_correct,
        look_index,
        alpha,
        min_samples,
        min_effect,
        task_arm_budget,
        None,
        None,
        None,
        false,
        None,
        None,
        0.10,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn paired_accuracy_look_checked(
    candidate_correct: &[bool],
    baseline_correct: &[bool],
    look_index: usize,
    alpha: f64,
    min_samples: usize,
    min_effect: f64,
    task_arm_budget: usize,
    population_n: Option<usize>,
    example_ids: Option<&[String]>,
    dataset_manifest_sha256: Option<&str>,
    require_example_ids: bool,
    min_effect_pct: Option<f64>,
    promising_effect_pct: Option<f64>,
    futility_alpha: f64,
) -> Result<AccuracyLook> {
    if !(0.0 < alpha && alpha < 1.0) || look_index == 0 || task_arm_budget == 0 {
        return Err(Error::Message(
            "invalid alpha, look_index, or task_arm_budget".into(),
        ));
    }
    if candidate_correct.is_empty() || candidate_correct.len() != baseline_correct.len() {
        return Err(Error::Message(
            "paired predictions must be nonempty and aligned".into(),
        ));
    }
    if require_example_ids || example_ids.is_some() || dataset_manifest_sha256.is_some() {
        validate_paired_binary_trace(
            candidate_correct,
            baseline_correct,
            example_ids,
            dataset_manifest_sha256,
            require_example_ids,
        )?;
    }
    let n = candidate_correct.len();
    let mut only_cand = 0usize;
    let mut only_base = 0usize;
    for (&c, &b) in candidate_correct.iter().zip(baseline_correct.iter()) {
        if c && !b {
            only_cand += 1;
        } else if b && !c {
            only_base += 1;
        }
    }
    let discordant = only_cand + only_base;
    let delta = (only_cand as f64 - only_base as f64) / n as f64;
    let effect_pct = delta * 100.0;
    let p_better = binom_upper_tail(only_cand, discordant);
    let p_worse = binom_upper_tail(only_base, discordant);
    let e_bb = beta_binomial_e_value(only_cand, only_base, 0.5, 0.5);
    let e_better = one_sided_mixture_e_value(only_cand, only_base);
    let e_worse = one_sided_mixture_e_value(only_base, only_cand);
    let (e_threshold, alpha_final) = if task_arm_budget > 16 {
        (task_arm_budget as f64 / alpha, 0.0)
    } else {
        (
            (1.0 / (0.25 * alpha)).max(task_arm_budget as f64 / (2.0 * alpha)),
            0.75 * alpha,
        )
    };
    let mut spent_equiv = 1.0 / e_threshold;
    let is_terminal = population_n.is_some_and(|pop| n >= pop) && alpha_final > 0.0;
    if is_terminal {
        spent_equiv = spent_equiv.max(alpha_final);
    }

    let eff_min_pct = min_effect_pct.unwrap_or(if min_effect > 0.0 {
        min_effect * 100.0
    } else {
        2.0
    });
    let eff_promising_pct = promising_effect_pct.unwrap_or(0.5 * eff_min_pct);

    let mut decision = "continue";
    if n >= min_samples {
        if delta > min_effect
            && (e_better >= e_threshold || (is_terminal && p_better <= alpha_final))
        {
            decision = "promising";
        } else if delta < -min_effect
            && (e_worse >= e_threshold || (is_terminal && p_worse <= alpha_final))
        {
            decision = "futility";
        }
    }
    let verdict = if decision == "promising" {
        "promising"
    } else if decision == "futility"
        || (n >= min_samples && delta < 0.0 && p_worse <= futility_alpha)
    {
        "refuted"
    } else if n >= min_samples && effect_pct >= eff_promising_pct && p_worse > futility_alpha {
        "promising_inconclusive"
    } else {
        "inconclusive"
    };

    Ok(AccuracyLook {
        decision: decision.to_string(),
        verdict: verdict.to_string(),
        n,
        look_index,
        candidate_only: only_cand,
        jev_only: only_base,
        discordant,
        delta,
        effect_pct,
        e_value_beta_binomial: e_bb,
        e_value_better: e_better,
        e_value_worse: e_worse,
        e_threshold,
        one_sided_p_better: p_better,
        one_sided_p_worse: p_worse,
        spent_alpha_each_direction: spent_equiv,
        canonical_qualification: false,
    })
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ThroughputLook {
    pub decision: String,
    pub stage_name: String,
    pub n: usize,
    pub look_index: usize,
    pub signed_delta: f64,
    pub relative_gain: f64,
    pub effect_pct: f64,
    pub t_stat: f64,
    pub one_sided_p_better: f64,
    pub one_sided_p_worse: f64,
    pub spent_alpha_each_direction: f64,
    pub comparison: StatisticalComparison,
}

pub fn paired_throughput_anytime_look(
    metric: &str,
    unit: &str,
    baseline_samples: &[f64],
    candidate_samples: &[f64],
    look_index: usize,
    higher_is_better: bool,
    alpha: f64,
    family_budget: usize,
    min_relative_gain: f64,
) -> Result<ThroughputLook> {
    if !(0.0 < alpha && alpha < 1.0) || look_index == 0 || family_budget == 0 {
        return Err(Error::Message(
            "invalid alpha, look_index, or family_budget".into(),
        ));
    }
    let n = baseline_samples.len();
    if n < 2 || n != candidate_samples.len() {
        return Err(Error::Message(
            "paired throughput look requires >= 2 aligned baseline and candidate samples".into(),
        ));
    }
    let comp = paired_comparison(
        metric,
        unit,
        baseline_samples,
        candidate_samples,
        higher_is_better,
    )?;
    let diffs: Vec<f64> = baseline_samples
        .iter()
        .zip(candidate_samples.iter())
        .map(|(&b, &c)| if higher_is_better { c - b } else { b - c })
        .collect();
    let d_mean = diffs.iter().sum::<f64>() / n as f64;
    let d_var = diffs.iter().map(|&x| (x - d_mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let d_std = d_var.sqrt();
    let t_stat = if d_std == 0.0 {
        if d_mean > 0.0 {
            f64::INFINITY
        } else if d_mean < 0.0 {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    } else {
        d_mean / (d_std / (n as f64).sqrt())
    };
    let p_better = student_t_one_sided_p(t_stat, n - 1)?;
    let p_worse = student_t_one_sided_p(-t_stat, n - 1)?;
    let threshold =
        alpha / (2.0 * family_budget as f64 * look_index as f64 * (look_index + 1) as f64);
    let base_abs = comp.baseline_mean.abs();
    let rel_gain = if base_abs > 1e-12 {
        d_mean / base_abs
    } else {
        0.0
    };

    let decision = if (rel_gain < -min_relative_gain && p_worse <= threshold)
        || comp.verdict == "loss"
    {
        "futility"
    } else if rel_gain >= min_relative_gain && (p_better <= threshold || comp.verdict == "win") {
        "promising"
    } else {
        "continue"
    };

    Ok(ThroughputLook {
        decision: decision.to_string(),
        stage_name: String::new(),
        n,
        look_index,
        signed_delta: d_mean,
        relative_gain: rel_gain,
        effect_pct: rel_gain * 100.0,
        t_stat: if t_stat.is_finite() { t_stat } else { 0.0 },
        one_sided_p_better: p_better,
        one_sided_p_worse: p_worse,
        spent_alpha_each_direction: threshold,
        comparison: comp,
    })
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LineageRule {
    pub paths: Vec<String>,
    pub artifacts: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImpactReport {
    pub tasks: Vec<String>,
    pub safety: bool,
    pub full_release_suite: bool,
    pub reason: BTreeMap<String, Vec<String>>,
    pub unknown_shared_changes: Vec<String>,
    pub method: String,
}

fn wildcard_match(text: &[u8], pat: &[u8]) -> bool {
    let (mut ti, mut pi) = (0usize, 0usize);
    let (mut star_pi, mut star_ti) = (None, 0usize);
    while ti < text.len() {
        if pi < pat.len() && (pat[pi] == b'?' || pat[pi] == text[ti]) {
            ti += 1;
            pi += 1;
        } else if pi < pat.len() && pat[pi] == b'*' {
            star_pi = Some(pi);
            pi += 1;
            star_ti = ti;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < pat.len() && pat[pi] == b'*' {
        pi += 1;
    }
    pi == pat.len()
}

fn glob_match(path: &str, pattern: &str) -> bool {
    if wildcard_match(path.as_bytes(), pattern.as_bytes()) {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix("/**") {
        if path == prefix || path.starts_with(&format!("{prefix}/")) {
            return true;
        }
    }
    false
}

fn normalize_posix_path(raw: &str) -> Result<String> {
    let replaced = raw.replace('\\', "/");
    let trimmed = replaced.strip_prefix("./").unwrap_or(&replaced).trim();
    if trimmed.is_empty() {
        return Err(Error::Message("invalid changed path".into()));
    }
    for seg in trimmed.split('/') {
        if seg == ".." {
            return Err(Error::Message("invalid changed path".into()));
        }
    }
    Ok(trimmed.to_string())
}

/// Compute conservative incremental benchmark scope from changed paths and artifacts.
pub fn impacted_benchmarks(
    tasks: &[String],
    changed_paths: &[String],
    lineage: Option<&BTreeMap<String, LineageRule>>,
    changed_artifacts: &[String],
) -> Result<ImpactReport> {
    let mut impacted: BTreeSet<String> = BTreeSet::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut why: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut safety = false;

    for raw in changed_paths {
        let path = normalize_posix_path(raw)?;
        if path.starts_with("docs/")
            || path.starts_with(".github/")
            || path.starts_with(".hanzo/")
            || (path.starts_with("train/campaign/") && !path.starts_with("train/campaign/recipes/"))
            || path.starts_with("campaigns/")
            || path == "bench_lineage.json"
            || path.starts_with("train/tests/")
            || path.ends_with(".md")
            || path.ends_with(".rst")
            || path.ends_with("_test.py")
            || path.starts_with("tests/")
        {
            continue;
        }
        if path.starts_with("train/ground/") {
            safety = true;
            continue;
        }
        let mut specific: BTreeSet<String> = BTreeSet::new();
        for task in tasks {
            let mut patterns = vec![
                format!("train/capabilities/{task}/**"),
                format!("train/campaign/recipes/{task}/**"),
                format!("bench/tasks/{task}/**"),
            ];
            if let Some(rule) = lineage.and_then(|m| m.get(task)) {
                patterns.extend(rule.paths.iter().cloned());
            }
            if patterns.iter().any(|p| glob_match(&path, p)) {
                specific.insert(task.clone());
            }
        }
        if !specific.is_empty() {
            for t in specific {
                impacted.insert(t.clone());
                why.entry(t).or_default().push(path.clone());
            }
        } else {
            unknown.push(path);
        }
    }

    for art in changed_artifacts {
        if art.is_empty() {
            return Err(Error::Message("empty artifact identifier".into()));
        }
        let mut depends = Vec::new();
        for task in tasks {
            if let Some(rule) = lineage.and_then(|m| m.get(task)) {
                if rule.artifacts.iter().any(|a| a == art) {
                    depends.push(task.clone());
                }
            }
        }
        if !depends.is_empty() {
            for t in depends {
                impacted.insert(t.clone());
                why.entry(t).or_default().push(format!("artifact:{art}"));
            }
        } else {
            unknown.push(format!("artifact:{art}"));
        }
    }

    let has_unknown = !unknown.is_empty();
    if has_unknown {
        for t in tasks {
            impacted.insert(t.clone());
            why.entry(t.clone()).or_default().extend(unknown.clone());
        }
    }

    Ok(ImpactReport {
        tasks: impacted.into_iter().collect(),
        safety: safety || has_unknown,
        full_release_suite: has_unknown,
        reason: why,
        method: if has_unknown {
            "conservative-all".into()
        } else {
            "manifest-scoped".into()
        },
        unknown_shared_changes: unknown,
    })
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProgressiveStage {
    pub name: String,
    pub reps: usize,
    pub prompt_len: usize,
    pub gen_len: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProgressiveStageOutput {
    pub stage_name: String,
    pub baseline_samples: Vec<f64>,
    pub candidate_samples: Vec<f64>,
    pub baseline_correct: Vec<bool>,
    pub candidate_correct: Vec<bool>,
    pub example_ids: Vec<String>,
    pub dataset_manifest_sha256: String,
    pub parity_passed: bool,
    pub memory_integrity_passed: bool,
    pub max_abs_diff: Option<f64>,
    pub ulp_diff: Option<f64>,
    pub gpu_time_sec: f64,
    pub details: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProgressiveAbPlan {
    pub stages: Vec<ProgressiveStage>,
    pub higher_is_better: bool,
    pub alpha: f64,
    pub family_budget: usize,
    pub min_relative_gain: f64,
    pub min_effect_pct: Option<f64>,
    pub promising_effect_pct: Option<f64>,
    pub futility_alpha: f64,
    pub max_continuations: usize,
    pub canonical_qualification: bool,
    pub require_example_ids: bool,
}

impl Default for ProgressiveAbPlan {
    fn default() -> Self {
        Self {
            stages: vec![
                ProgressiveStage {
                    name: "micro".into(),
                    reps: 3,
                    prompt_len: 64,
                    gen_len: 16,
                },
                ProgressiveStage {
                    name: "medium".into(),
                    reps: 5,
                    prompt_len: 256,
                    gen_len: 64,
                },
                ProgressiveStage {
                    name: "qualification".into(),
                    reps: 8,
                    prompt_len: 512,
                    gen_len: 128,
                },
            ],
            higher_is_better: true,
            alpha: 0.05,
            family_budget: 1,
            min_relative_gain: 0.01,
            min_effect_pct: None,
            promising_effect_pct: None,
            futility_alpha: 0.10,
            max_continuations: 1,
            canonical_qualification: true,
            require_example_ids: false,
        }
    }
}

impl ProgressiveAbPlan {
    pub fn effective_min_effect_pct(&self) -> f64 {
        self.min_effect_pct
            .unwrap_or(self.min_relative_gain * 100.0)
    }

    pub fn effective_promising_effect_pct(&self) -> f64 {
        self.promising_effect_pct
            .unwrap_or_else(|| 0.5 * self.effective_min_effect_pct())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProgressiveAbResult {
    pub stopped_stage: usize,
    pub stage_name: String,
    pub verdict: String,
    pub early_stopped: bool,
    pub failure_diagnosis: String,
    pub looks: Vec<ThroughputLook>,
    pub final_output: ProgressiveStageOutput,
}

#[derive(Serialize, Deserialize)]
struct CheckpointEntry {
    stage_index: usize,
    output: ProgressiveStageOutput,
    look: ThroughputLook,
}

#[derive(Default, Serialize, Deserialize)]
struct CheckpointFile {
    completed: Vec<CheckpointEntry>,
}

pub struct ProgressiveAbRunner {
    pub plan: ProgressiveAbPlan,
}

impl ProgressiveAbRunner {
    pub fn new(plan: ProgressiveAbPlan) -> Self {
        Self { plan }
    }

    fn classify_non_promising(&self, effect_pct: f64, p_worse: f64) -> &'static str {
        if effect_pct <= 0.0 || p_worse <= self.plan.futility_alpha {
            "refuted"
        } else if effect_pct >= self.plan.effective_promising_effect_pct() {
            "promising_inconclusive"
        } else {
            "inconclusive"
        }
    }

    pub fn run<F>(&self, metric: &str, unit: &str, measure_stage: F) -> Result<ProgressiveAbResult>
    where
        F: FnMut(usize, &ProgressiveStage) -> Result<ProgressiveStageOutput>,
    {
        self.run_with_checkpoint(metric, unit, None, measure_stage)
    }

    pub fn run_with_checkpoint<F>(
        &self,
        metric: &str,
        unit: &str,
        checkpoint_path: Option<&Path>,
        mut measure_stage: F,
    ) -> Result<ProgressiveAbResult>
    where
        F: FnMut(usize, &ProgressiveStage) -> Result<ProgressiveStageOutput>,
    {
        let mut saved: BTreeMap<usize, (ProgressiveStageOutput, ThroughputLook)> = BTreeMap::new();
        if let Some(ckpt) = checkpoint_path.filter(|p| p.is_file()) {
            if let Ok(parsed) = serde_json::from_slice::<CheckpointFile>(&fs::read(ckpt)?) {
                for row in parsed.completed {
                    saved.insert(row.stage_index, (row.output, row.look));
                }
            }
        }

        let persist_ckpt =
            |map: &BTreeMap<usize, (ProgressiveStageOutput, ThroughputLook)>| -> Result<()> {
                let Some(ckpt) = checkpoint_path else {
                    return Ok(());
                };
                if let Some(parent) = ckpt.parent() {
                    fs::create_dir_all(parent)?;
                }
                let file_obj = CheckpointFile {
                    completed: map
                        .iter()
                        .map(|(&idx, (o, l))| CheckpointEntry {
                            stage_index: idx,
                            output: o.clone(),
                            look: l.clone(),
                        })
                        .collect(),
                };
                let tmp = ckpt.with_file_name(format!(
                    "{}.tmp",
                    ckpt.file_name().unwrap_or_default().to_string_lossy()
                ));
                let mut raw = serde_json::to_vec_pretty(&file_obj)?;
                raw.push(b'\n');
                let mut fp = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp)?;
                fp.write_all(&raw)?;
                fp.sync_data()?;
                fs::rename(tmp, ckpt)?;
                Ok(())
            };

        let eff_min_pct = self.plan.effective_min_effect_pct();
        let eff_prom_pct = self.plan.effective_promising_effect_pct();
        let min_rel_gain = eff_min_pct / 100.0;

        let mut looks = Vec::new();
        let mut last_out = ProgressiveStageOutput::default();
        let total_stages = self.plan.stages.len();

        for (zero_idx, stage) in self.plan.stages.iter().enumerate() {
            let idx = zero_idx + 1;
            let (out, look) = if let Some((so, sl)) = saved.get(&idx).cloned() {
                (so, sl)
            } else {
                let mut out = measure_stage(idx, stage)?;
                if !out.parity_passed || !out.memory_integrity_passed {
                    let diag = format!(
                        "correctness gate failed at stage {idx} ({}): parity={}, memory_integrity={}, max_abs_diff={:?}; {}",
                        stage.name,
                        out.parity_passed,
                        out.memory_integrity_passed,
                        out.max_abs_diff,
                        out.details
                    );
                    return Ok(ProgressiveAbResult {
                        stopped_stage: idx,
                        stage_name: stage.name.clone(),
                        verdict: "refuted".into(),
                        early_stopped: idx < total_stages,
                        failure_diagnosis: diag,
                        looks,
                        final_output: out,
                    });
                }
                let look = if !out.candidate_correct.is_empty() && !out.baseline_correct.is_empty()
                {
                    let eids = if out.example_ids.is_empty() {
                        None
                    } else {
                        Some(out.example_ids.as_slice())
                    };
                    let msha = if out.dataset_manifest_sha256.trim().is_empty() {
                        None
                    } else {
                        Some(out.dataset_manifest_sha256.as_str())
                    };
                    validate_paired_binary_trace(
                        &out.candidate_correct,
                        &out.baseline_correct,
                        eids,
                        msha,
                        self.plan.require_example_ids,
                    )?;
                    if out.baseline_samples.is_empty() {
                        out.baseline_samples = out
                            .baseline_correct
                            .iter()
                            .map(|&b| if b { 1.0 } else { 0.0 })
                            .collect();
                    }
                    if out.candidate_samples.is_empty() {
                        out.candidate_samples = out
                            .candidate_correct
                            .iter()
                            .map(|&c| if c { 1.0 } else { 0.0 })
                            .collect();
                    }
                    let (cand_eff, base_eff) = if self.plan.higher_is_better {
                        (&out.candidate_correct, &out.baseline_correct)
                    } else {
                        (&out.baseline_correct, &out.candidate_correct)
                    };
                    let acc_look = paired_accuracy_look_checked(
                        cand_eff,
                        base_eff,
                        idx,
                        self.plan.alpha,
                        32usize.min(cand_eff.len()),
                        min_rel_gain,
                        self.plan.family_budget.max(1),
                        Some(cand_eff.len()),
                        eids,
                        msha,
                        self.plan.require_example_ids,
                        Some(eff_min_pct),
                        Some(eff_prom_pct),
                        self.plan.futility_alpha,
                    )?;
                    let comp = paired_comparison(
                        metric,
                        unit,
                        &out.baseline_samples,
                        &out.candidate_samples,
                        self.plan.higher_is_better,
                    )?;
                    ThroughputLook {
                        decision: acc_look.decision,
                        stage_name: stage.name.clone(),
                        n: acc_look.n,
                        look_index: idx,
                        signed_delta: acc_look.delta,
                        relative_gain: acc_look.delta,
                        effect_pct: acc_look.effect_pct,
                        t_stat: 0.0,
                        one_sided_p_better: acc_look.one_sided_p_better,
                        one_sided_p_worse: acc_look.one_sided_p_worse,
                        spent_alpha_each_direction: acc_look.spent_alpha_each_direction,
                        comparison: comp,
                    }
                } else {
                    if self.plan.require_example_ids {
                        if out.baseline_samples.is_empty()
                            || out.baseline_samples.len() != out.candidate_samples.len()
                            || out
                                .baseline_samples
                                .iter()
                                .chain(out.candidate_samples.iter())
                                .any(|&v| v != 0.0 && v != 1.0)
                        {
                            return Err(Error::Message(
                                "require_example_ids=true requires binary {0,1} or boolean paired traces".into(),
                            ));
                        }
                        let c_bools: Vec<bool> =
                            out.candidate_samples.iter().map(|&v| v == 1.0).collect();
                        let b_bools: Vec<bool> =
                            out.baseline_samples.iter().map(|&v| v == 1.0).collect();
                        let eids = if out.example_ids.is_empty() {
                            None
                        } else {
                            Some(out.example_ids.as_slice())
                        };
                        let msha = if out.dataset_manifest_sha256.trim().is_empty() {
                            None
                        } else {
                            Some(out.dataset_manifest_sha256.as_str())
                        };
                        validate_paired_binary_trace(&c_bools, &b_bools, eids, msha, true)?;
                    }
                    let mut tl = paired_throughput_anytime_look(
                        metric,
                        unit,
                        &out.baseline_samples,
                        &out.candidate_samples,
                        idx,
                        self.plan.higher_is_better,
                        self.plan.alpha,
                        self.plan.family_budget,
                        min_rel_gain,
                    )?;
                    tl.stage_name = stage.name.clone();
                    tl
                };
                saved.insert(idx, (out.clone(), look.clone()));
                persist_ckpt(&saved)?;
                (out, look)
            };

            last_out = out.clone();
            looks.push(look.clone());

            let effect_pct = if look.effect_pct != 0.0 || look.relative_gain == 0.0 {
                look.effect_pct
            } else {
                look.relative_gain * 100.0
            };

            if look.decision == "futility"
                || (effect_pct < 0.0 && look.one_sided_p_worse <= self.plan.futility_alpha)
            {
                let diag = format!(
                    "early futility/regression at stage {idx} ({}): relative_gain={:+.4}, p_worse={:.6}",
                    stage.name, look.relative_gain, look.one_sided_p_worse
                );
                return Ok(ProgressiveAbResult {
                    stopped_stage: idx,
                    stage_name: stage.name.clone(),
                    verdict: "refuted".into(),
                    early_stopped: idx < total_stages,
                    failure_diagnosis: diag,
                    looks,
                    final_output: out,
                });
            }

            if idx < total_stages && look.decision != "promising" {
                let verdict = self.classify_non_promising(effect_pct, look.one_sided_p_worse);
                let diag = if verdict == "promising_inconclusive" {
                    format!(
                        "promising inconclusive screen at stage {idx} ({}): effect_pct={effect_pct:+.2}% >= {eff_prom_pct:.2}% (p_better={:.6}, p_worse={:.6} > futility_alpha={:.3}); deferring sealed qualification until bounded independent dev continuation",
                        stage.name,
                        look.one_sided_p_better,
                        look.one_sided_p_worse,
                        self.plan.futility_alpha
                    )
                } else {
                    format!(
                        "non-promising micro-screen at stage {idx} ({}): relative_gain={:+.4}, p_better={:.6} did not clear spent_alpha={:.6}; skipping heavier workloads",
                        stage.name,
                        look.relative_gain,
                        look.one_sided_p_better,
                        look.spent_alpha_each_direction
                    )
                };
                return Ok(ProgressiveAbResult {
                    stopped_stage: idx,
                    stage_name: stage.name.clone(),
                    verdict: verdict.into(),
                    early_stopped: true,
                    failure_diagnosis: diag,
                    looks,
                    final_output: out,
                });
            }
        }

        let final_look = looks
            .last()
            .ok_or_else(|| Error::Message("progressive plan must contain >= 1 stage".into()))?;
        let final_eff_pct = if final_look.effect_pct != 0.0 || final_look.relative_gain == 0.0 {
            final_look.effect_pct
        } else {
            final_look.relative_gain * 100.0
        };
        let (verdict, diag) = if final_look.decision == "promising" {
            if self.plan.canonical_qualification {
                ("proven", String::new())
            } else {
                (
                    "promising_inconclusive",
                    format!(
                        "dev-only screen crossed sequential threshold (effect_pct={final_eff_pct:+.2}%), held at promising_inconclusive because canonical_qualification=false"
                    ),
                )
            }
        } else if final_look.decision == "futility"
            || (final_eff_pct < 0.0 && final_look.one_sided_p_worse <= self.plan.futility_alpha)
        {
            (
                "refuted",
                format!(
                    "qualification regression: relative_gain={:+.4}",
                    final_look.relative_gain
                ),
            )
        } else {
            let v = self.classify_non_promising(final_eff_pct, final_look.one_sided_p_worse);
            let d = if v == "promising_inconclusive" {
                format!(
                    "promising inconclusive result: effect_pct={final_eff_pct:+.2}% >= {eff_prom_pct:.2}%, p_better={:.6}, p_worse={:.6}",
                    final_look.one_sided_p_better, final_look.one_sided_p_worse
                )
            } else {
                format!(
                    "qualification {v}: relative_gain={:+.4}",
                    final_look.relative_gain
                )
            };
            (v, d)
        };
        Ok(ProgressiveAbResult {
            stopped_stage: total_stages,
            stage_name: self
                .plan
                .stages
                .last()
                .map(|s| s.name.clone())
                .unwrap_or_default(),
            verdict: verdict.into(),
            early_stopped: false,
            failure_diagnosis: diag,
            looks,
            final_output: last_out,
        })
    }

    pub fn run_and_record<F>(
        &self,
        ledger: &LocalResearchLedger,
        exp: &mut ExperimentRecordV2,
        checkpoint_path: Option<&Path>,
        extra_artifacts: &[(&str, &[u8])],
        measure_stage: F,
    ) -> Result<(ProgressiveAbResult, SyncReceipt)>
    where
        F: FnMut(usize, &ProgressiveStage) -> Result<ProgressiveStageOutput>,
    {
        let metric = if exp.statistical_comparison.metric.is_empty() {
            "throughput".to_string()
        } else {
            exp.statistical_comparison.metric.clone()
        };
        let unit = if exp.statistical_comparison.unit.is_empty() {
            "tok/s".to_string()
        } else {
            exp.statistical_comparison.unit.clone()
        };
        let res = self.run_with_checkpoint(&metric, &unit, checkpoint_path, measure_stage)?;
        let out = &res.final_output;
        let mut comp = if let Some(last) = res.looks.last() {
            last.comparison.clone()
        } else {
            paired_comparison(
                &metric,
                &unit,
                if out.baseline_samples.is_empty() {
                    &[1.0]
                } else {
                    &out.baseline_samples
                },
                if out.candidate_samples.is_empty() {
                    &[1.0]
                } else {
                    &out.candidate_samples
                },
                self.plan.higher_is_better,
            )?
        };
        exp.config
            .entry("max_continuations".into())
            .or_insert_with(|| json!(self.plan.max_continuations));
        exp.config
            .entry("continuation_depth".into())
            .or_insert_with(|| json!(0));
        if self.plan.require_example_ids {
            exp.config.insert("require_example_ids".into(), json!(true));
        }
        if !out.example_ids.is_empty() && exp.example_ids.is_empty() {
            exp.example_ids = out.example_ids.clone();
        }
        if !out.dataset_manifest_sha256.is_empty()
            && exp.workload_manifest.manifest_sha256.is_empty()
        {
            exp.workload_manifest.manifest_sha256 = out.dataset_manifest_sha256.clone();
        }
        comp.anytime_valid = true;
        comp.verdict = res.verdict.clone();
        comp.p_value = res.looks.last().map(|l| l.one_sided_p_better);
        exp.statistical_comparison = comp.clone();
        exp.verdict = res.verdict.clone();
        exp.correctness = CorrectnessChecks {
            parity_passed: out.parity_passed,
            max_abs_diff: out.max_abs_diff,
            ulp_diff: out.ulp_diff,
            safety_passed: Some(true),
            memory_integrity_passed: out.memory_integrity_passed,
            details: out.details.clone(),
        };
        exp.raw_samples
            .insert("baseline".into(), out.baseline_samples.clone());
        exp.raw_samples
            .insert("candidate".into(), out.candidate_samples.clone());
        exp.baseline
            .insert("mean".into(), json!(comp.baseline_mean));
        exp.baseline.insert("stage".into(), json!(res.stage_name));
        exp.candidate
            .insert("mean".into(), json!(comp.candidate_mean));
        exp.candidate.insert("stage".into(), json!(res.stage_name));
        exp.gpu_time_sec = out.gpu_time_sec;
        if !res.failure_diagnosis.is_empty() {
            exp.failure_diagnosis = res.failure_diagnosis.clone();
        }
        if exp.conclusion.is_empty() {
            exp.conclusion = format!(
                "{} at stage {} ({}): cand={:.4} vs base={:.4}",
                res.verdict,
                res.stopped_stage,
                res.stage_name,
                comp.candidate_mean,
                comp.baseline_mean
            );
        }

        let looks_bytes = serde_json::to_vec_pretty(&res)?;
        let mut all_arts: Vec<(&str, &[u8])> = extra_artifacts.to_vec();
        all_arts.push(("progressive_looks_json", &looks_bytes));
        let receipt = ledger.record_experiment(exp, &all_arts)?;
        Ok((res, receipt))
    }
}
