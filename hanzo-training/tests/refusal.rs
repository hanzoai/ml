//! The unwired training path refuses instead of fabricating.
//!
//! This crate holds two training paths. `grpo` is connected: it differentiates a
//! real `Policy` through `hanzo_ml`'s autograd and updates weights with
//! `hanzo_nn::AdamW`, and its own tests (`grpo_smoke_reward_increases`) prove
//! reward climbs, which only happens if parameters actually move.
//!
//! `Trainer` / `ModelWrapper` / `OptimizerWrapper` / the benchmarks are not
//! connected to anything. They used to *report* as though they were: a forward
//! pass returning the constant `1.0`, an optimizer step that advanced a counter
//! and no weights, and benchmarks returning a fixed perplexity of 5.2 and a fixed
//! accuracy of 0.85 for every model and corpus. Each returned `Ok`, so a run
//! ended with a loss log, a checkpoint on disk and exit status zero.
//!
//! These tests pin the refusals, so the numbers cannot come back.

use hanzo_training::config::ModelConfig;
use hanzo_training::evaluation::{AccuracyBenchmark, Benchmark, PerplexityBenchmark};
use hanzo_training::model::ModelWrapper;
use hanzo_training::optimizer::{Optimizer, OptimizerConfig, OptimizerWrapper};
use hanzo_training::TrainingConfig;

fn model_config() -> ModelConfig {
    ModelConfig {
        name: "zen-eco-3b".to_string(),
        architecture: "llama".to_string(),
        checkpoint: None,
        max_seq_length: 512,
        vocab_size: Some(32_000),
        hidden_size: Some(2048),
        num_layers: Some(16),
        num_heads: Some(16),
        custom_config: None,
    }
}

/// Every refusal names the crate and points at the connected path, so the
/// message is actionable rather than just an error.
fn assert_refusal(err: anyhow::Error, what: &str) {
    let msg = err.to_string();
    assert!(msg.contains("hanzo-training"), "{what}: {msg}");
    assert!(
        msg.contains("grpo"),
        "{what} must point at the connected path: {msg}"
    );
}

#[test]
fn a_model_with_no_weights_refuses_to_be_constructed() {
    // Not expect_err: ModelWrapper is not Debug, so the Ok side cannot be printed.
    let err = ModelWrapper::new(&model_config(), hanzo_ml::Device::Cpu)
        .err()
        .expect("ModelWrapper holds no weights and must not claim to be a model");
    assert_refusal(err, "ModelWrapper::new");
}

#[test]
fn the_forward_pass_does_not_return_a_constant_loss() {
    // The refusal at construction is what makes this unreachable in practice.
    // Asserting on the trait method too keeps the constant from being restored
    // behind a different constructor.
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/model.rs"))
        .expect("model.rs is readable");
    assert!(
        !source.contains("Tensor::new(&[1.0f32]"),
        "forward() must not synthesise a loss value"
    );
}

#[test]
fn the_optimizer_refuses_a_step_it_cannot_take() {
    let config = TrainingConfig::default();
    let mut optimizer =
        OptimizerWrapper::new(OptimizerConfig::from_training_params(&config.training))
            .expect("the wrapper itself constructs; it is the step that cannot be taken");
    let err = optimizer
        .step(vec![])
        .expect_err("a step with no gradients must not report success");
    assert_refusal(err, "OptimizerWrapper::step");

    let err = optimizer
        .zero_grad(vec![])
        .expect_err("zeroing gradients that do not exist must not report success");
    assert_refusal(err, "OptimizerWrapper::zero_grad");
}

#[test]
fn perplexity_is_not_reported_as_5_2() {
    let err = PerplexityBenchmark::new()
        .run()
        .expect_err("a benchmark with no model and no dataset must not report a score");
    assert_refusal(err, "PerplexityBenchmark::run");
}

#[test]
fn accuracy_is_not_reported_as_0_85() {
    let err = AccuracyBenchmark::new()
        .run()
        .expect_err("a benchmark with no model and no dataset must not report a score");
    assert_refusal(err, "AccuracyBenchmark::run");
}

#[test]
fn lora_is_never_described_as_enabled() {
    // `LoRAConfig` parses and validates, but no adapter consumes it anywhere in
    // the workspace. The CLI printed "LoRA: enabled (r=…, α=…)" from the config
    // alone, which reads as a capability report.
    let source = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/train.rs"))
        .expect("train.rs is readable");
    assert!(
        !source.contains("LoRA: enabled"),
        "the CLI must not report LoRA as enabled while no adapter applies it"
    );
    assert!(
        source.contains("NOT APPLIED"),
        "the CLI must say the LoRA config is read but not applied"
    );
}
