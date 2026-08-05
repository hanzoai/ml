//! Evaluation and benchmarking

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub benchmarks: Vec<String>,
    pub metrics: Vec<String>,
    pub eval_dataset: Option<String>,
    pub output_dir: Option<String>,
}

/// Evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub benchmarks: HashMap<String, BenchmarkResult>,
    pub overall_score: f64,
    pub timestamp: String,
}

/// Individual benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub score: f64,
    pub metrics: HashMap<String, f64>,
    pub details: Option<serde_json::Value>,
}

/// Benchmark trait
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self) -> Result<BenchmarkResult>;
    fn description(&self) -> &str;
}

/// Perplexity benchmark
pub struct PerplexityBenchmark {
    name: String,
}

impl Default for PerplexityBenchmark {
    fn default() -> Self {
        Self {
            name: "perplexity".to_string(),
        }
    }
}

impl PerplexityBenchmark {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Benchmark for PerplexityBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    /// Refuses: [`Benchmark::run`] takes no model and no dataset, so there is
    /// nothing here to compute a perplexity over. It previously reported the
    /// constant 5.2 for every model and every corpus.
    fn run(&self) -> Result<BenchmarkResult> {
        crate::model::unwired("the perplexity benchmark")
    }

    fn description(&self) -> &str {
        "Measures model perplexity on evaluation dataset"
    }
}

/// Accuracy benchmark
pub struct AccuracyBenchmark {
    name: String,
}

impl Default for AccuracyBenchmark {
    fn default() -> Self {
        Self {
            name: "accuracy".to_string(),
        }
    }
}

impl AccuracyBenchmark {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Benchmark for AccuracyBenchmark {
    fn name(&self) -> &str {
        &self.name
    }

    /// Refuses, for the same reason as [`PerplexityBenchmark::run`]: no model and
    /// no dataset reach it. It previously reported the constant 0.85.
    fn run(&self) -> Result<BenchmarkResult> {
        crate::model::unwired("the accuracy benchmark")
    }

    fn description(&self) -> &str {
        "Measures model accuracy on evaluation tasks"
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    benchmarks: Vec<Box<dyn Benchmark>>,
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    pub fn add_benchmark(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }

    /// Refuses on an empty runner: an overall score is the mean of the
    /// benchmarks that ran, and the mean of zero measurements is not `0.0` — it
    /// is undefined. Returning `Ok(overall_score: 0.0)` here reported a
    /// measurement that never happened.
    pub fn run_all(&self) -> Result<EvaluationResult> {
        if self.benchmarks.is_empty() {
            anyhow::bail!(
                "hanzo-training: BenchmarkRunner has no benchmarks, so there is nothing to \
                 score. Add a benchmark with add_benchmark before calling run_all."
            );
        }

        let mut benchmark_results = HashMap::new();
        let mut total_score = 0.0;

        for benchmark in &self.benchmarks {
            let result = benchmark.run()?;
            total_score += result.score;
            benchmark_results.insert(benchmark.name().to_string(), result);
        }

        let overall_score = total_score / self.benchmarks.len() as f64;

        let timestamp = chrono::Utc::now().to_rfc3339();

        Ok(EvaluationResult {
            benchmarks: benchmark_results,
            overall_score,
            timestamp,
        })
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        let mut runner = Self::new();
        runner.add_benchmark(Box::new(PerplexityBenchmark::new()));
        runner.add_benchmark(Box::new(AccuracyBenchmark::new()));
        runner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_all_refuses_with_no_benchmarks() {
        // An empty runner has measured nothing; it must not report a score.
        let runner = BenchmarkRunner::new();
        let err = runner
            .run_all()
            .expect_err("run_all must refuse when no benchmarks were added");
        assert!(
            err.to_string().contains("no benchmarks"),
            "expected the empty-runner refusal, got: {err}"
        );
    }

    #[test]
    fn run_all_propagates_a_benchmark_refusal() {
        // The default runner carries the perplexity and accuracy benchmarks,
        // both of which refuse (no model, no dataset). run_all must surface that
        // refusal, not average it away.
        let runner = BenchmarkRunner::default();
        assert!(runner.run_all().is_err());
    }
}
