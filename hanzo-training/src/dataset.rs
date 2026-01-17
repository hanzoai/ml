//! Dataset implementations for training

use crate::Result;
use hanzo_ml::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Training sample containing input and target tensors
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub input_ids: Tensor,
    pub attention_mask: Option<Tensor>,
    pub labels: Option<Tensor>,
    pub metadata: Option<serde_json::Value>,
}

/// Dataset trait for training data
pub trait Dataset: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get_item(&self, index: usize) -> Result<TrainingSample>;
}

/// Zen Agentic Dataset for real-world programming data
pub struct ZenAgenticDataset {
    data_path: String,
    max_seq_length: usize,
    device: Device,
    samples: Vec<AgenticSample>,
}

/// Individual agentic programming sample
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgenticSample {
    pub conversation: Vec<Message>,
    pub code_blocks: Vec<CodeBlock>,
    pub git_context: Option<GitContext>,
    pub metadata: serde_json::Value,
}

/// Message in conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: Option<String>,
    pub tools_used: Option<Vec<String>>,
}

/// Code block from session
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeBlock {
    pub language: String,
    pub content: String,
    pub file_path: Option<String>,
    pub diff: Option<String>,
}

/// Git context information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitContext {
    pub commit_hash: Option<String>,
    pub branch: Option<String>,
    pub files_changed: Option<Vec<String>>,
    pub diff: Option<String>,
}

impl ZenAgenticDataset {
    pub fn new<P: AsRef<Path>>(
        data_path: P,
        max_seq_length: usize,
        device: Device,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_string_lossy().to_string();
        let samples = Self::load_samples(&data_path)?;
        
        Ok(Self {
            data_path,
            max_seq_length,
            device,
            samples,
        })
    }

    fn load_samples(data_path: &str) -> Result<Vec<AgenticSample>> {
        let path = Path::new(data_path);
        let mut samples = Vec::new();

        if path.is_file() && path.extension().map_or(false, |ext| ext == "jsonl") {
            // Single JSONL file
            let content = std::fs::read_to_string(path)?;
            for line in content.lines() {
                if !line.trim().is_empty() {
                    let sample: AgenticSample = serde_json::from_str(line)?;
                    samples.push(sample);
                }
            }
        } else if path.is_dir() {
            // Directory containing multiple files
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let file_path = entry.path();
                
                if file_path.extension().map_or(false, |ext| ext == "jsonl") {
                    let content = std::fs::read_to_string(&file_path)?;
                    for line in content.lines() {
                        if !line.trim().is_empty() {
                            let sample: AgenticSample = serde_json::from_str(line)?;
                            samples.push(sample);
                        }
                    }
                }
            }
        } else {
            return Err(format!("Invalid dataset path: {}", data_path).into());
        }

        Ok(samples)
    }

    fn format_sample(&self, sample: &AgenticSample) -> String {
        let mut formatted = String::new();
        
        // Add conversation context
        for message in &sample.conversation {
            formatted.push_str(&format!(
                "<|{}|>\n{}\n\n",
                message.role,
                message.content
            ));
        }

        // Add code blocks
        for code_block in &sample.code_blocks {
            formatted.push_str(&format!(
                "<|code:{}|>\n{}\n\n",
                code_block.language,
                code_block.content
            ));
        }

        // Add git context if available
        if let Some(git) = &sample.git_context {
            if let Some(diff) = &git.diff {
                formatted.push_str("<|git_diff|>\n");
                formatted.push_str(diff);
                formatted.push_str("\n\n");
            }
        }

        formatted
    }

    fn tokenize(&self, text: &str) -> Result<Tensor> {
        // Simplified tokenization - in practice, use proper tokenizer
        let tokens: Vec<u32> = text
            .chars()
            .map(|c| c as u32)
            .take(self.max_seq_length)
            .collect();
        
        let tensor = Tensor::from_slice(&tokens, tokens.len(), &self.device)?
            .to_dtype(DType::U32)?;
        
        Ok(tensor)
    }
}

impl Dataset for ZenAgenticDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get_item(&self, index: usize) -> Result<TrainingSample> {
        if index >= self.samples.len() {
            return Err(format!("Index {} out of bounds for dataset of size {}", index, self.samples.len()).into());
        }

        let sample = &self.samples[index];
        let formatted_text = self.format_sample(sample);
        let input_ids = self.tokenize(&formatted_text)?;
        
        // Create labels (same as input_ids for causal LM)
        let labels = input_ids.clone();
        
        // Create attention mask (all 1s for now)
        let seq_len = input_ids.dim(0)?;
        let attention_mask = Tensor::ones((seq_len,), DType::U8, &self.device)?;

        Ok(TrainingSample {
            input_ids,
            attention_mask: Some(attention_mask),
            labels: Some(labels),
            metadata: Some(sample.metadata.clone()),
        })
    }
}

/// Zen Identity Dataset for model personality training
pub struct ZenIdentityDataset {
    data_path: String,
    max_seq_length: usize,
    device: Device,
    samples: Vec<IdentitySample>,
}

/// Identity training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdentitySample {
    pub persona: String,
    pub prompt: String,
    pub response: String,
    pub traits: Vec<String>,
    pub metadata: serde_json::Value,
}

impl ZenIdentityDataset {
    pub fn new<P: AsRef<Path>>(
        data_path: P,
        max_seq_length: usize,
        device: Device,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_string_lossy().to_string();
        let samples = Self::load_samples(&data_path)?;
        
        Ok(Self {
            data_path,
            max_seq_length,
            device,
            samples,
        })
    }

    fn load_samples(data_path: &str) -> Result<Vec<IdentitySample>> {
        let path = Path::new(data_path);
        let mut samples = Vec::new();

        if path.is_file() && path.extension().map_or(false, |ext| ext == "jsonl") {
            let content = std::fs::read_to_string(path)?;
            for line in content.lines() {
                if !line.trim().is_empty() {
                    let sample: IdentitySample = serde_json::from_str(line)?;
                    samples.push(sample);
                }
            }
        } else if path.is_dir() {
            // Load all JSONL files in directory
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let file_path = entry.path();
                
                if file_path.extension().map_or(false, |ext| ext == "jsonl") {
                    let content = std::fs::read_to_string(&file_path)?;
                    for line in content.lines() {
                        if !line.trim().is_empty() {
                            let sample: IdentitySample = serde_json::from_str(line)?;
                            samples.push(sample);
                        }
                    }
                }
            }
        } else {
            return Err(format!("Invalid dataset path: {}", data_path).into());
        }

        Ok(samples)
    }

    fn format_sample(&self, sample: &IdentitySample) -> String {
        format!(
            "<|persona|>\n{}\n\n<|user|>\n{}\n\n<|assistant|>\n{}\n",
            sample.persona,
            sample.prompt,
            sample.response
        )
    }

    fn tokenize(&self, text: &str) -> Result<Tensor> {
        // Simplified tokenization - in practice, use proper tokenizer
        let tokens: Vec<u32> = text
            .chars()
            .map(|c| c as u32)
            .take(self.max_seq_length)
            .collect();
        
        let tensor = Tensor::from_slice(&tokens, tokens.len(), &self.device)?
            .to_dtype(DType::U32)?;
        
        Ok(tensor)
    }
}

impl Dataset for ZenIdentityDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get_item(&self, index: usize) -> Result<TrainingSample> {
        if index >= self.samples.len() {
            return Err(format!("Index {} out of bounds for dataset of size {}", index, self.samples.len()).into());
        }

        let sample = &self.samples[index];
        let formatted_text = self.format_sample(sample);
        let input_ids = self.tokenize(&formatted_text)?;
        
        // Create labels (same as input_ids for causal LM)
        let labels = input_ids.clone();
        
        // Create attention mask
        let seq_len = input_ids.dim(0)?;
        let attention_mask = Tensor::ones((seq_len,), DType::U8, &self.device)?;

        Ok(TrainingSample {
            input_ids,
            attention_mask: Some(attention_mask),
            labels: Some(labels),
            metadata: Some(sample.metadata.clone()),
        })
    }
}

/// Generic JSONL dataset loader
pub struct JsonlDataset {
    data_path: String,
    max_seq_length: usize,
    device: Device,
    samples: Vec<serde_json::Value>,
}

impl JsonlDataset {
    pub fn new<P: AsRef<Path>>(
        data_path: P,
        max_seq_length: usize,
        device: Device,
    ) -> Result<Self> {
        let data_path = data_path.as_ref().to_string_lossy().to_string();
        let samples = Self::load_samples(&data_path)?;
        
        Ok(Self {
            data_path,
            max_seq_length,
            device,
            samples,
        })
    }

    fn load_samples(data_path: &str) -> Result<Vec<serde_json::Value>> {
        let content = std::fs::read_to_string(data_path)?;
        let mut samples = Vec::new();
        
        for line in content.lines() {
            if !line.trim().is_empty() {
                let sample: serde_json::Value = serde_json::from_str(line)?;
                samples.push(sample);
            }
        }
        
        Ok(samples)
    }

    fn tokenize(&self, text: &str) -> Result<Tensor> {
        // Simplified tokenization - in practice, use proper tokenizer
        let tokens: Vec<u32> = text
            .chars()
            .map(|c| c as u32)
            .take(self.max_seq_length)
            .collect();
        
        let tensor = Tensor::from_slice(&tokens, tokens.len(), &self.device)?
            .to_dtype(DType::U32)?;
        
        Ok(tensor)
    }
}

impl Dataset for JsonlDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get_item(&self, index: usize) -> Result<TrainingSample> {
        if index >= self.samples.len() {
            return Err(format!("Index {} out of bounds for dataset of size {}", index, self.samples.len()).into());
        }

        let sample = &self.samples[index];
        
        // Extract text from JSON (assuming 'text' field)
        let text = sample
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or("Missing 'text' field in sample")?;
        
        let input_ids = self.tokenize(text)?;
        let labels = input_ids.clone();
        
        let seq_len = input_ids.dim(0)?;
        let attention_mask = Tensor::ones((seq_len,), DType::U8, &self.device)?;

        Ok(TrainingSample {
            input_ids,
            attention_mask: Some(attention_mask),
            labels: Some(labels),
            metadata: Some(sample.clone()),
        })
    }
}