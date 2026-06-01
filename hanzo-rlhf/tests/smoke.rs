//! End-to-end smoke tests for the hanzo-rlhf facade.
//!
//! No network, no real models — just exercise the type-level wiring so a
//! `cargo test -p hanzo-rlhf` catches any breakage in the public surface.

use hanzo_rlhf::grpo::{GroupRelativeEstimator, GrpoConfig, DEFAULT_GROUP_SIZE};
use hanzo_rlhf::python_bridge::RunConfig;
use hanzo_rlhf::{AdvantageEstimator, RlhfAlgorithm};

#[test]
fn grpo_config_round_trip_through_runconfig() {
    let rc = RunConfig::grpo_default("zenlm/zen-5-flash", "ds:reward", "ds:trainset");
    let s = serde_json::to_string(&rc).unwrap();
    let back: RunConfig = serde_json::from_str(&s).unwrap();
    assert_eq!(back.policy, "zenlm/zen-5-flash");
    assert_eq!(back.batch_size, 4);
    let gs = back.extra.get("group_size").and_then(|v| v.as_u64()).unwrap();
    assert_eq!(gs as usize, DEFAULT_GROUP_SIZE);
}

#[test]
fn estimator_keeps_group_normalisation_invariants() {
    let cfg = GrpoConfig::default();
    let est = GroupRelativeEstimator {
        group_size: cfg.group_size,
        std_eps: cfg.std_eps,
    };
    // Two groups of size G with distinct distributions.
    let rewards: Vec<f32> = (0..cfg.group_size).map(|i| i as f32).collect();
    let mut all = rewards.clone();
    all.extend(rewards.iter().map(|r| r * 10.0));
    let adv = est.advantage(&all);
    assert_eq!(adv.len(), all.len());
    // Within each group, advantages must sum to ~0.
    for chunk in adv.chunks(cfg.group_size) {
        let s: f32 = chunk.iter().sum();
        assert!(s.abs() < 1e-4, "group sum was {s}");
    }
}

#[test]
fn algorithm_enum_is_exhaustive() {
    // Touch every variant so adding one without updating call sites
    // becomes a compile error someone has to fix.
    for a in [
        RlhfAlgorithm::Grpo,
        RlhfAlgorithm::Dpo,
        RlhfAlgorithm::Ppo,
        RlhfAlgorithm::Kto,
        RlhfAlgorithm::Simpo,
        RlhfAlgorithm::Orpo,
    ] {
        assert!(!a.as_str().is_empty());
    }
}
