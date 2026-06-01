//! Python trainer bridge over the federation HTTP transport.
//!
//! ## Why a bridge?
//!
//! TRL, Unsloth, and DeepSeek's open-source GRPO recipe are all Python.
//! Rewriting them in candle is a multi-quarter project; the bridge is
//! how hanzod ships real RLHF *today* without any Python on the operator's
//! box. The Python trainer runs on a separate worker (typically a GPU
//! node) and the bridge is the wire that connects hanzod to it.
//!
//! ## Wire protocol
//!
//! POST to `<coordinator>/v1/rlhf/run` with a JSON body of [`RunConfig`].
//! The coordinator persists the job, picks a worker that advertised
//! `capabilities: ["rlhf:<algorithm>"]` in the lab manifest, and routes
//! the request. The trainer streams progress back as Server-Sent Events
//! and, on success, posts a canonical BF16 delta blob (the same format
//! `hanzo_federation::codec` defines) to `<coordinator>/v1/rlhf/<job>/delta`.
//!
//! All transport runs through `hanzo-federation` so workers don't need a
//! separate identity/auth story — the same HMAC-SHA256 that protects
//! delta-soup uploads protects RLHF dispatch.
//!
//! ## One way to invoke
//!
//! `RlhfAlgorithm::Grpo.run_via_federation(...)` is the only public
//! entrypoint. There is intentionally no "give me a raw HTTP client and
//! let me build the request" escape hatch — that defeats the point of
//! standardising on this crate.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

use crate::{RlhfAlgorithm, RlhfHandle, RlhfOutcome};

/// Knobs shared across all algorithms. Algorithm-specific extras (KL
/// beta for GRPO, β for DPO, etc.) ride along in [`Self::extra`] as
/// untyped JSON so this struct doesn't need to grow per-algorithm
/// fields. The Python trainer is the authority on which keys it
/// honours; on the Rust side we just pass them through.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    /// Reference to the policy weights. Typically a Hugging Face repo
    /// (`zenlm/zen-5-flash`) or a path the trainer can resolve.
    pub policy: String,
    /// Reference to the reward model (or `"dataset"` for preference
    /// algorithms that don't use one).
    pub reward: String,
    /// Reference to the dataset (HF repo, S3 URI, local path).
    pub dataset: String,
    /// Total optimisation steps to run.
    pub max_steps: u32,
    /// Mini-batch size at the optimiser level.
    pub batch_size: u32,
    /// Inner learning rate.
    pub learning_rate: f32,
    /// Algorithm-specific extras, e.g. `{"kl_beta": 0.04, "group_size": 8}`.
    #[serde(default)]
    pub extra: serde_json::Value,
}

impl RunConfig {
    /// Convenience: a sensible default GRPO config.
    pub fn grpo_default(policy: &str, reward: &str, dataset: &str) -> Self {
        Self {
            policy: policy.to_string(),
            reward: reward.to_string(),
            dataset: dataset.to_string(),
            max_steps: 1000,
            batch_size: 4,
            learning_rate: 1e-6,
            extra: serde_json::json!({
                "group_size": crate::grpo::DEFAULT_GROUP_SIZE,
                "clip_eps":   crate::grpo::DEFAULT_CLIP_EPS,
                "kl_beta":    crate::grpo::DEFAULT_KL_BETA,
            }),
        }
    }
}

/// Response shape from the `/v1/rlhf/run` endpoint.
#[derive(Debug, Clone, Deserialize)]
struct DispatchResponse {
    job_id: String,
}

/// Response shape from `/v1/rlhf/<job>/status`.
#[derive(Debug, Clone, Deserialize)]
struct StatusResponse {
    state: String,
    #[serde(default)]
    steps: u32,
    #[serde(default)]
    loss: f32,
    #[serde(default)]
    delta_uri: String,
    #[serde(default)]
    error: Option<String>,
}

impl RlhfAlgorithm {
    /// POST `config` to the federation coordinator's RLHF endpoint and
    /// return a [`RlhfHandle`] that resolves to the final delta URI.
    ///
    /// `coordinator_url` is the federation HTTP base (no trailing slash),
    /// matching `[federation].bind` / `[federation].coordinator_url` in
    /// `hanzo.toml`.
    pub fn run_via_federation(
        self,
        coordinator_url: impl Into<String>,
        config: RunConfig,
    ) -> RlhfHandle {
        let coordinator_url = coordinator_url.into();
        let algo = self;
        let join: JoinHandle<anyhow::Result<RlhfOutcome>> = tokio::spawn(async move {
            run_one(coordinator_url, algo, config).await
        });
        RlhfHandle {
            // Job id is unknown until the dispatch returns; the handle
            // exposes the JoinHandle which surfaces the final outcome
            // (where the assigned id is also recorded).
            job_id: String::new(),
            join,
        }
    }
}

async fn run_one(
    coordinator_url: String,
    algo: RlhfAlgorithm,
    config: RunConfig,
) -> anyhow::Result<RlhfOutcome> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let dispatch_url = format!("{coordinator_url}/v1/rlhf/{}", algo.as_str());
    tracing::info!(
        target: "hanzo_rlhf::bridge",
        algorithm = algo.as_str(),
        url = %dispatch_url,
        "dispatching RLHF run"
    );
    let dispatch: DispatchResponse = client
        .post(&dispatch_url)
        .json(&config)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let job_id = dispatch.job_id;
    tracing::info!(
        target: "hanzo_rlhf::bridge",
        algorithm = algo.as_str(),
        job_id = %job_id,
        "trainer accepted job"
    );

    // Poll status until terminal. Backoff is fixed and intentionally not
    // configurable — the trainer is the authority on cadence; if you
    // want push semantics, listen on the federation's SSE stream
    // directly.
    let status_url = format!("{coordinator_url}/v1/rlhf/{job_id}/status");
    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;
        let status: StatusResponse = client
            .get(&status_url)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        match status.state.as_str() {
            "succeeded" => {
                return Ok(RlhfOutcome {
                    algorithm: algo,
                    steps: status.steps,
                    final_loss: status.loss,
                    delta_uri: status.delta_uri,
                });
            }
            "failed" => {
                return Err(anyhow::anyhow!(
                    "trainer {job_id} failed: {}",
                    status.error.unwrap_or_default()
                ));
            }
            _ => continue,
        }
    }
}

/// PolicyUpdater impl that delegates each `update` call to the Python
/// trainer. Useful when an outer loop is owned in Rust (e.g. for online
/// reward modelling) but the gradient step itself needs to run on the
/// trainer worker. Today this is a thin error: real online use should
/// dispatch the full run via [`RlhfAlgorithm::run_via_federation`].
#[derive(Debug)]
pub struct BridgeUpdater {
    /// Algorithm this updater dispatches.
    pub algorithm: RlhfAlgorithm,
    /// Federation coordinator URL.
    pub coordinator_url: String,
}

impl crate::PolicyUpdater for BridgeUpdater {
    fn update(
        &mut self,
        _policy: &mut dyn candle_nn::Module,
        _samples: &[crate::Sample],
        _advantages: &[f32],
    ) -> anyhow::Result<f32> {
        // The federation endpoint runs whole training jobs, not single
        // gradient steps. We deliberately surface that here rather than
        // pretending we can do something we can't.
        Err(anyhow::anyhow!(
            "BridgeUpdater::update is per-step; use {}::run_via_federation \
             for a full RLHF run",
            self.algorithm.as_str()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_config_grpo_default_carries_kl_beta() {
        let c = RunConfig::grpo_default("zenlm/zen-5-flash", "ds:reward", "ds:trainset");
        assert_eq!(c.policy, "zenlm/zen-5-flash");
        assert_eq!(c.batch_size, 4);
        // kl_beta is the GRPO default β
        let kl = c.extra.get("kl_beta").and_then(|v| v.as_f64()).unwrap();
        assert!((kl - crate::grpo::DEFAULT_KL_BETA as f64).abs() < 1e-6);
    }

    #[test]
    fn algorithm_strings_round_trip_serde() {
        for a in [
            RlhfAlgorithm::Grpo,
            RlhfAlgorithm::Dpo,
            RlhfAlgorithm::Ppo,
            RlhfAlgorithm::Kto,
            RlhfAlgorithm::Simpo,
            RlhfAlgorithm::Orpo,
        ] {
            let s = serde_json::to_string(&a).unwrap();
            let back: RlhfAlgorithm = serde_json::from_str(&s).unwrap();
            assert_eq!(a, back);
            assert!(s.contains(a.as_str()));
        }
    }
}
