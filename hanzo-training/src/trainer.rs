//! Core trainer implementation

use crate::{
    config::TrainingConfig, 
    dataset::{Dataset, ZenAgenticDataset, ZenIdentityDataset, JsonlDataset}, 
    model::{TrainableModel, ModelWrapper},
    optimizer::{OptimizerWrapper, OptimizerConfig},
    Result
};
use hanzo_ml::Device;
use std::time::Instant;
use tracing::{info, warn};

/// Training result containing metrics and statistics
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub final_loss: f64,
    pub total_steps: usize,
    pub training_time: f64,
    pub best_eval_loss: Option<f64>,
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Main trainer struct
pub struct Trainer {
    config: TrainingConfig,
    model: Box<dyn TrainableModel>,
    dataset: Box<dyn Dataset>,
    optimizer: Box<dyn crate::optimizer::Optimizer>,
    device: Device,
    current_step: usize,
    current_epoch: usize,
}

impl Trainer {
    /// Create a new trainer from configuration
    pub fn new(config: TrainingConfig) -> Result<Self> {
        // Validate configuration
        config.validate()?;

        // Setup device
        let device = Self::setup_device(&config)?;
        info!("Using device: {:?}", device);

        // Load dataset
        let dataset = Self::load_dataset(&config, &device)?;
        info!("Loaded dataset with {} samples", dataset.len());

        // Load model
        let model = Self::load_model(&config, &device)?;
        info!("Loaded model: {}", config.model.name);

        // Create optimizer
        let optimizer_config = OptimizerConfig::from_training_params(&config.training);
        let optimizer = Box::new(OptimizerWrapper::new(optimizer_config)?);
        info!("Created optimizer: {:?}", config.training.optimizer);

        Ok(Self {
            config,
            model,
            dataset,
            optimizer,
            device,
            current_step: 0,
            current_epoch: 0,
        })
    }

    /// Train the model
    pub fn train(&mut self) -> Result<TrainingResult> {
        info!("Starting training...");
        let start_time = Instant::now();
        
        let mut total_loss = 0.0;
        let mut step_count = 0;
        let mut best_eval_loss = None;
        let mut metrics = std::collections::HashMap::new();

        // Determine training duration
        let max_epochs = self.config.training.epochs.unwrap_or(1);
        let max_steps = self.config.training.max_steps;

        for epoch in 0..max_epochs {
            self.current_epoch = epoch;
            info!("Starting epoch {}/{}", epoch + 1, max_epochs);

            let epoch_loss = self.train_epoch()?;
            total_loss += epoch_loss;
            step_count += 1;

            // Check if we've reached max steps
            if let Some(max_steps) = max_steps {
                if self.current_step >= max_steps {
                    info!("Reached maximum steps ({}), stopping training", max_steps);
                    break;
                }
            }

            // Evaluation
            if let Some(eval_steps) = self.config.training.eval_steps {
                if self.current_step % eval_steps == 0 {
                    if let Ok(eval_loss) = self.evaluate() {
                        info!("Evaluation loss: {:.6}", eval_loss);
                        if best_eval_loss.is_none() || eval_loss < best_eval_loss.unwrap() {
                            best_eval_loss = Some(eval_loss);
                            info!("New best evaluation loss: {:.6}", eval_loss);
                        }
                    }
                }
            }

            // Save checkpoint
            if let Some(save_steps) = self.config.training.save_steps {
                if self.current_step % save_steps == 0 {
                    let checkpoint_path = std::path::PathBuf::from(format!("./checkpoint-step-{}", self.current_step));
                    if let Err(e) = self.save_checkpoint(&checkpoint_path) {
                        warn!("Failed to save checkpoint: {}", e);
                    } else {
                        info!("Saved checkpoint: {}", checkpoint_path.display());
                    }
                }
            }
        }

        let training_time = start_time.elapsed().as_secs_f64();
        let final_loss = total_loss / step_count.max(1) as f64;

        info!("Training completed!");
        info!("Final loss: {:.6}", final_loss);
        info!("Total steps: {}", self.current_step);
        info!("Training time: {:.2}s", training_time);

        metrics.insert("final_loss".to_string(), final_loss);
        metrics.insert("training_time".to_string(), training_time);
        metrics.insert("total_steps".to_string(), self.current_step as f64);

        Ok(TrainingResult {
            final_loss,
            total_steps: self.current_step,
            training_time,
            best_eval_loss,
            metrics,
        })
    }

    /// Train for one epoch
    fn train_epoch(&mut self) -> Result<f64> {
        let dataset_len = self.dataset.len();
        let batch_size = self.config.training.batch_size;
        let num_batches = (dataset_len + batch_size - 1) / batch_size;
        
        let mut epoch_loss = 0.0;
        
        for batch_idx in 0..num_batches {
            let batch_loss = self.train_step(batch_idx)?;
            epoch_loss += batch_loss;
            self.current_step += 1;

            // Logging
            if let Some(logging_steps) = self.config.training.logging_steps {
                if self.current_step % logging_steps == 0 {
                    info!(
                        "Step {}: loss = {:.6}, lr = {:.2e}",
                        self.current_step,
                        batch_loss,
                        self.optimizer.get_learning_rate()
                    );
                }
            }

            // Check max steps
            if let Some(max_steps) = self.config.training.max_steps {
                if self.current_step >= max_steps {
                    break;
                }
            }
        }

        Ok(epoch_loss / num_batches as f64)
    }

    /// Execute one training step
    fn train_step(&mut self, batch_idx: usize) -> Result<f64> {
        let batch_size = self.config.training.batch_size;
        let dataset_len = self.dataset.len();
        
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(dataset_len);
        
        let mut batch_loss = 0.0;
        let mut valid_samples = 0;

        // Accumulate gradients over batch
        for sample_idx in start_idx..end_idx {
            if let Ok(sample) = self.dataset.get_item(sample_idx) {
                let loss = self.model.forward(&sample.input_ids)?;
                self.model.backward(&loss)?;
                
                // Extract scalar loss value (simplified)
                let loss_val = loss.to_vec1::<f32>()?[0] as f64;
                batch_loss += loss_val;
                valid_samples += 1;
            }
        }

        if valid_samples > 0 {
            batch_loss /= valid_samples as f64;
            
            // Apply gradients
            self.optimizer.step(self.model.parameters())?;
            self.optimizer.zero_grad(self.model.parameters())?;
        }

        Ok(batch_loss)
    }

    /// Evaluate the model
    pub fn evaluate(&self) -> Result<f64> {
        info!("Running evaluation...");
        
        // For now, just compute loss on a subset of training data
        // In practice, you'd use a separate validation dataset
        let num_eval_samples = 100.min(self.dataset.len());
        let mut eval_loss = 0.0;
        let mut valid_samples = 0;

        for i in 0..num_eval_samples {
            if let Ok(sample) = self.dataset.get_item(i) {
                // Forward pass only (no gradients)
                let loss = self.model.forward(&sample.input_ids)?;
                let loss_val = loss.to_vec1::<f32>()?[0] as f64;
                eval_loss += loss_val;
                valid_samples += 1;
            }
        }

        if valid_samples > 0 {
            eval_loss /= valid_samples as f64;
        }

        Ok(eval_loss)
    }

    /// Save model checkpoint
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(path)?;
        self.model.save(path)?;
        info!("Checkpoint saved to: {}", path.display());
        Ok(())
    }

    /// Load model checkpoint
    pub fn load_checkpoint(&mut self, _path: &std::path::Path) -> Result<()> {
        self.model = Self::load_model(&self.config, &self.device)?;
        info!("Checkpoint loaded from model config");
        Ok(())
    }

    /// Setup device for training
    fn setup_device(config: &TrainingConfig) -> Result<Device> {
        let device_str = config.device();
        
        if device_str.starts_with("cuda") {
            let device_id = device_str
                .split(':')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            Device::cuda_if_available(device_id)
                .map_err(|e| format!("Failed to setup CUDA device: {}", e).into())
        } else if device_str == "metal" {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|e| format!("Failed to setup Metal device: {}", e).into())
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("Metal support not enabled".into())
            }
        } else {
            Ok(Device::Cpu)
        }
    }

    /// Load dataset based on configuration
    fn load_dataset(config: &TrainingConfig, device: &Device) -> Result<Box<dyn Dataset>> {
        let dataset: Box<dyn Dataset> = match config.dataset.name.as_str() {
            "zen-agentic" => Box::new(ZenAgenticDataset::new(
                &config.dataset.path,
                config.dataset.max_seq_length,
                device.clone(),
            )?),
            "zen-identity" => Box::new(ZenIdentityDataset::new(
                &config.dataset.path,
                config.dataset.max_seq_length,
                device.clone(),
            )?),
            _ => {
                // Default to generic JSONL dataset
                Box::new(JsonlDataset::new(
                    &config.dataset.path,
                    config.dataset.max_seq_length,
                    device.clone(),
                )?)
            }
        };

        Ok(dataset)
    }

    /// Load model based on configuration
    fn load_model(config: &TrainingConfig, device: &Device) -> Result<Box<dyn TrainableModel>> {
        let model = ModelWrapper::new(&config.model, device.clone())?;
        Ok(Box::new(model))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingConfig;
    use tempfile::TempDir;

    #[test]
    fn test_trainer_creation() {
        let config = TrainingConfig::default();
        // Note: This will fail without proper setup, but tests the interface
        let result = Trainer::new(config);
        assert!(result.is_err()); // Expected to fail in test environment
    }
}