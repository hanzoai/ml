//! Dataset handling for training

use crate::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub path: String,
    pub format: String,
}

/// Training sample containing input and expected output
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub input: String,
    pub output: String,
}

impl TrainingSample {
    /// Convert input text to tensor (simplified tokenization)
    /// Refuses: this crate has no tokenizer.
    ///
    /// This returned `self.input.chars().map(|c| c as u32)` — Unicode code points
    /// standing in for token ids. Those are not the ids any model's embedding
    /// table was trained against, and for a 32k-vocab model most of them are out
    /// of range, so the tensor was wrong rather than approximate. The
    /// `tokenizers` crate is already a workspace dependency (used by
    /// `hanzo-datasets` and the wasm examples); a wired dataset loads the model's
    /// own tokenizer.json through it.
    pub fn input_ids(&self, _device: &hanzo_ml::Device) -> crate::Result<hanzo_ml::Tensor> {
        crate::model::unwired("dataset tokenization")
    }
}

/// Dataset trait for different dataset implementations
pub trait Dataset {
    fn name(&self) -> &str;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, index: usize) -> Result<&TrainingSample>;
    fn iter(&self) -> Box<dyn Iterator<Item = &TrainingSample> + '_>;
}

/// Basic dataset implementation
pub struct BasicDataset {
    pub name: String,
    pub samples: Vec<TrainingSample>,
}

impl BasicDataset {
    pub fn new(name: String) -> Self {
        Self {
            name,
            samples: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    /// Refuses: this never read `path`. It returned an empty dataset named after
    /// the config, which the trainer then reported as "Loaded dataset with 0
    /// samples" and trained on. A loader that reads nothing must not report
    /// success. `JsonlDataset::load` is the reader that genuinely parses a file.
    pub fn load<P: AsRef<Path>>(_path: P, _config: &DatasetConfig) -> Result<Self> {
        crate::model::unwired("loading a basic dataset")
    }
}

impl Dataset for BasicDataset {
    fn name(&self) -> &str {
        &self.name
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<&TrainingSample> {
        self.samples
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("Index {} out of bounds", index))
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &TrainingSample> + '_> {
        Box::new(self.samples.iter())
    }
}

/// Zen Agentic Dataset for training agentic AI models
pub struct ZenAgenticDataset {
    pub dataset: BasicDataset,
}

impl ZenAgenticDataset {
    /// Refuses: this logged "Loading …" and returned an empty dataset. The
    /// zen-agentic format is not parsed here, so the trainer would have reported
    /// "Loaded dataset with 0 samples" and trained on nothing. A wired loader
    /// reads the dataset through `hanzo-datasets` (already a workspace
    /// dependency) and errors when it finds no samples.
    pub fn load<P: AsRef<Path>>(_path: P) -> Result<Self> {
        crate::model::unwired("loading the zen-agentic dataset")
    }
}

impl Dataset for ZenAgenticDataset {
    fn name(&self) -> &str {
        self.dataset.name()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<&TrainingSample> {
        self.dataset.get(index)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &TrainingSample> + '_> {
        self.dataset.iter()
    }
}

/// Zen Identity Dataset for identity-aware training
pub struct ZenIdentityDataset {
    pub dataset: BasicDataset,
}

impl ZenIdentityDataset {
    /// Refuses, for the same reason as [`ZenAgenticDataset::load`]: it logged a
    /// "Loading …" line and returned an empty dataset without reading the
    /// identity data.
    pub fn load<P: AsRef<Path>>(_path: P) -> Result<Self> {
        crate::model::unwired("loading the zen-identity dataset")
    }
}

impl Dataset for ZenIdentityDataset {
    fn name(&self) -> &str {
        self.dataset.name()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<&TrainingSample> {
        self.dataset.get(index)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &TrainingSample> + '_> {
        self.dataset.iter()
    }
}

/// Generic JSONL Dataset loader
pub struct JsonlDataset {
    pub dataset: BasicDataset,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonlSample {
    pub input: String,
    pub output: String,
}

impl JsonlDataset {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut dataset = BasicDataset::new("jsonl".to_string());

        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<JsonlSample>(&line) {
                Ok(sample) => {
                    dataset.add_sample(TrainingSample {
                        input: sample.input,
                        output: sample.output,
                    });
                }
                Err(e) => {
                    log::warn!("Failed to parse line: {} - Error: {}", line, e);
                }
            }
        }

        if dataset.samples.is_empty() {
            anyhow::bail!(
                "hanzo-training: the JSONL dataset at {:?} yielded no samples \
                 (file empty, or every line failed to parse as {{\"input\":..,\"output\":..}}); \
                 a training run needs at least one sample.",
                path.as_ref()
            );
        }

        log::info!(
            "Loaded {} samples from JSONL dataset",
            dataset.samples.len()
        );

        Ok(Self { dataset })
    }
}

impl Dataset for JsonlDataset {
    fn name(&self) -> &str {
        self.dataset.name()
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Result<&TrainingSample> {
        self.dataset.get(index)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &TrainingSample> + '_> {
        self.dataset.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn input_ids_refuses_without_a_tokenizer() {
        // Pins the tokenization refusal: without a real tokenizer there are no
        // ids to build a tensor from. This must fail rather than fall back to
        // `chars() as u32`, which are not the ids any embedding table expects.
        let sample = TrainingSample {
            input: "hello".to_string(),
            output: "world".to_string(),
        };
        let err = sample
            .input_ids(&hanzo_ml::Device::Cpu)
            .expect_err("input_ids must refuse: this crate has no tokenizer");
        assert!(
            err.to_string().contains("not connected to a model"),
            "expected the unwired refusal, got: {err}"
        );
    }

    #[test]
    fn zen_agentic_load_refuses() {
        let err = ZenAgenticDataset::load("/does/not/matter")
            .err()
            .expect("the zen-agentic loader is unwired and must refuse");
        assert!(err.to_string().contains("not connected to a model"));
    }

    #[test]
    fn zen_identity_load_refuses() {
        let err = ZenIdentityDataset::load("/does/not/matter")
            .err()
            .expect("the zen-identity loader is unwired and must refuse");
        assert!(err.to_string().contains("not connected to a model"));
    }

    #[test]
    fn basic_load_refuses() {
        let config = DatasetConfig {
            name: "basic".to_string(),
            path: "/does/not/matter".to_string(),
            format: "jsonl".to_string(),
        };
        let err = BasicDataset::load("/does/not/matter", &config)
            .err()
            .expect("the basic loader is unwired and must refuse");
        assert!(err.to_string().contains("not connected to a model"));
    }

    #[test]
    fn jsonl_load_errors_on_empty_file() {
        // A reader that parses a real file must still refuse zero samples rather
        // than hand back an empty dataset the trainer would run on.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "   ").unwrap();
        let err = JsonlDataset::load(file.path())
            .err()
            .expect("an empty JSONL dataset must error, not return Ok");
        assert!(
            err.to_string().contains("no samples"),
            "expected the empty-dataset error, got: {err}"
        );
    }

    #[test]
    fn jsonl_load_reads_samples() {
        // The happy path stays green: a real line loads and reports its count.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "{{\"input\":\"a\",\"output\":\"b\"}}").unwrap();
        let dataset = JsonlDataset::load(file.path()).unwrap();
        assert_eq!(dataset.len(), 1);
    }
}
