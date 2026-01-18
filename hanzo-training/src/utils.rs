//! Utility functions for training

use crate::Result;
use std::path::Path;

/// Calculate estimated training time based on model size and dataset
pub fn estimate_training_time(
    model_parameters: usize,
    dataset_size: usize,
    batch_size: usize,
    epochs: usize,
    tokens_per_second: usize,
) -> f64 {
    let total_tokens = dataset_size * epochs;
    let batches = (dataset_size + batch_size - 1) / batch_size;
    let total_batches = batches * epochs;
    
    // Rough estimation based on model size and throughput
    let time_per_batch = (batch_size as f64) / (tokens_per_second as f64);
    let total_time = (total_batches as f64) * time_per_batch;
    
    total_time
}

/// Calculate memory requirements for training
pub fn estimate_memory_requirements(
    model_parameters: usize,
    batch_size: usize,
    sequence_length: usize,
    precision_bytes: usize,
) -> usize {
    // Model parameters
    let model_memory = model_parameters * precision_bytes;
    
    // Gradients (same size as model)
    let gradient_memory = model_parameters * precision_bytes;
    
    // Optimizer states (Adam: 2x model size)
    let optimizer_memory = model_parameters * precision_bytes * 2;
    
    // Activations (rough estimate)
    let activation_memory = batch_size * sequence_length * 1024 * precision_bytes;
    
    model_memory + gradient_memory + optimizer_memory + activation_memory
}

/// Format bytes into human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_index])
}

/// Format duration into human-readable string
pub fn format_duration(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u64;
    let minutes = ((seconds % 3600.0) / 60.0) as u64;
    let secs = (seconds % 60.0) as u64;
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Create output directory structure
pub fn create_output_dirs<P: AsRef<Path>>(base_dir: P) -> Result<()> {
    let base_dir = base_dir.as_ref();
    
    // Create main output directory
    std::fs::create_dir_all(base_dir)?;
    
    // Create subdirectories
    std::fs::create_dir_all(base_dir.join("checkpoints"))?;
    std::fs::create_dir_all(base_dir.join("logs"))?;
    std::fs::create_dir_all(base_dir.join("tensorboard"))?;
    std::fs::create_dir_all(base_dir.join("evaluation"))?;
    
    Ok(())
}

/// Generate a unique run name based on timestamp and config
pub fn generate_run_name(config: &crate::config::TrainingConfig) -> String {
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    format!("{}_{}_{}", config.model.name, config.dataset.name, timestamp)
}

/// Progress bar wrapper
pub struct ProgressBar {
    bar: indicatif::ProgressBar,
}

impl ProgressBar {
    pub fn new(total: u64) -> Self {
        let bar = indicatif::ProgressBar::new(total);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        Self { bar }
    }

    pub fn set_message(&self, message: &str) {
        self.bar.set_message(message.to_string());
    }

    pub fn inc(&self, delta: u64) {
        self.bar.inc(delta);
    }

    pub fn finish_with_message(&self, message: &str) {
        self.bar.finish_with_message(message.to_string());
    }
}

/// Configuration validator
pub struct ConfigValidator;

impl ConfigValidator {
    pub fn validate_paths(config: &crate::config::TrainingConfig) -> Result<()> {
        // Check if dataset path exists
        if !Path::new(&config.dataset.path).exists() {
            return Err(anyhow::anyhow!("Dataset path does not exist: {}", config.dataset.path));
        }

        // Check if checkpoint path exists (if specified)
        if let Some(checkpoint) = &config.model.checkpoint {
            if !checkpoint.starts_with("http") && !Path::new(checkpoint).exists() {
                return Err(anyhow::anyhow!("Checkpoint path does not exist: {}", checkpoint));
            }
        }

        Ok(())
    }

    pub fn validate_hardware(config: &crate::config::TrainingConfig) -> Result<()> {
        let device_str = config.device();
        
        if device_str.starts_with("cuda") {
            // Check if CUDA is available
            #[cfg(feature = "cuda")]
            {
                use hanzo_ml::Device;
                let device_id = device_str
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                
                Device::cuda_if_available(device_id)
                    .map_err(|e| anyhow::anyhow!("CUDA device not available: {}", e))?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow::anyhow!("CUDA support not compiled"));
            }
        }

        Ok(())
    }

    pub fn validate_memory_requirements(config: &crate::config::TrainingConfig) -> Result<()> {
        let estimated_memory = estimate_memory_requirements(
            config.model.hidden_size.unwrap_or(4096) * config.model.num_layers.unwrap_or(32),
            config.training.batch_size,
            config.model.max_seq_length,
            2, // Assuming bf16
        );

        // Warning for high memory usage (>= 32GB)
        if estimated_memory >= 32 * 1024 * 1024 * 1024 {
            eprintln!(
                "WARNING: Estimated memory usage: {} (may require large GPU)",
                format_bytes(estimated_memory)
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(60.0), "1m 0s");
        assert_eq!(format_duration(3665.0), "1h 1m 5s");
        assert_eq!(format_duration(45.5), "45s");
    }

    #[test]
    fn test_memory_estimation() {
        let memory = estimate_memory_requirements(1_000_000, 4, 2048, 2);
        assert!(memory > 0);
    }
}