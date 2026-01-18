//! Training metrics and evaluation utilities

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Training metrics collected during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub tokens_per_second: f64,
    pub elapsed_time: f64,
    pub custom_metrics: HashMap<String, f64>,
}

impl TrainingMetrics {
    pub fn new(epoch: usize, step: usize) -> Self {
        Self {
            epoch,
            step,
            loss: 0.0,
            learning_rate: 0.0,
            tokens_per_second: 0.0,
            elapsed_time: 0.0,
            custom_metrics: HashMap::new(),
        }
    }

    pub fn with_loss(mut self, loss: f64) -> Self {
        self.loss = loss;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_tokens_per_second(mut self, tps: f64) -> Self {
        self.tokens_per_second = tps;
        self
    }

    pub fn with_elapsed_time(mut self, time: f64) -> Self {
        self.elapsed_time = time;
        self
    }

    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }
}

/// Collection of metrics over time
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MetricsCollector {
    pub metrics: Vec<TrainingMetrics>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    pub fn add_metrics(&mut self, metrics: TrainingMetrics) {
        self.metrics.push(metrics);
    }

    pub fn latest(&self) -> Option<&TrainingMetrics> {
        self.metrics.last()
    }

    pub fn average_loss(&self, last_n: usize) -> Option<f64> {
        if self.metrics.is_empty() {
            return None;
        }
        
        let start = if self.metrics.len() > last_n {
            self.metrics.len() - last_n
        } else {
            0
        };
        
        let sum: f64 = self.metrics[start..].iter()
            .map(|m| m.loss)
            .sum();
        let count = self.metrics.len() - start;
        
        Some(sum / count as f64)
    }

    pub fn save_to_file(&self, path: &str) -> crate::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let collector = serde_json::from_str(&content)?;
        Ok(collector)
    }
}

/// Evaluation metrics for model validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub perplexity: f64,
    pub accuracy: f64,
    pub bleu_score: Option<f64>,
    pub rouge_scores: Option<HashMap<String, f64>>,
    pub custom_metrics: HashMap<String, f64>,
}

impl EvaluationMetrics {
    pub fn new() -> Self {
        Self {
            perplexity: 0.0,
            accuracy: 0.0,
            bleu_score: None,
            rouge_scores: None,
            custom_metrics: HashMap::new(),
        }
    }

    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = accuracy;
        self
    }
}

impl Default for EvaluationMetrics {
    fn default() -> Self {
        Self::new()
    }
}