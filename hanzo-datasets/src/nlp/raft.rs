use crate::lib::Dataset;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// RAFT (Retrieval-Augmented Fine-Tuning) Dataset structure.
/// Reference: https://arxiv.org/abs/2403.10131
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftExample {
    pub question: String,
    pub context: Vec<String>, // Includes gold and distractor documents
    pub answer: String,
    pub gold_index: Option<usize>, // Index of the relevant document
}

pub struct RaftDataset {
    pub examples: Vec<RaftExample>,
    pub path: PathBuf,
}

impl RaftDataset {
    pub fn new(path: PathBuf) -> Self {
        // In a real implementation, this would load from a JSONL file
        Self {
            examples: Vec::new(),
            path,
        }
    }

    /// Formats the example for Zen training:
    /// "Context: [D1] [D2] ... [Dn] Question: [Q] Answer: [A]"
    pub fn format_for_training(&self, example: &RaftExample) -> String {
        let mut prompt = String::from("Context:\n");
        for (i, doc) in example.context.iter().enumerate() {
            prompt.push_str(&format!("[{}] {}\n", i, doc));
        }
        prompt.push_str(&format!("\nQuestion: {}\nAnswer:", example.question));
        
        // RAFT specific: The answer should ideally include citations to the gold_index
        format!("{} {}", prompt, example.answer)
    }
}

/// Helper to generate synthetic distractors for research purposes
pub fn inject_distractors(gold: String, distractors: Vec<String>, p: f32) -> Vec<String> {
    let mut combined = distractors.clone();
    // Logic to randomly insert gold document or omit it (for negative training)
    combined.push(gold);
    combined
}
