use hanzo_ml::{Tensor, Device};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct RaftSample {
    pub question: String,
    pub oracle_passages: Vec<String>,
    pub distractors: Vec<String>,
    pub answer: String,
}

pub struct RaftTrainer {
    pub device: Device,
    pub learning_rate: f64,
}

impl RaftTrainer {
    pub fn new(device: Device) -> Self {
        Self { device, learning_rate: 2e-5 }
    }

    pub fn prepare_batch(&self, samples: Vec<RaftSample>) -> Tensor {
        // Implementation of RAFT prompt construction:
        // "Question: {q} Context: {oracle + distractors} Answer: {a}"
        todo!("Implement tensor batching for Zen model input")
    }
}
