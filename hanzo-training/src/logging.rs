//! Logging and monitoring

use crate::config::{FileLoggingConfig, LoggingConfig};
use crate::Result;

/// Logger trait
pub trait Logger: Send + Sync {
    fn log_metric(&self, name: &str, value: f64, step: usize);
    fn log_text(&self, name: &str, text: &str, step: usize);
    fn log_config(&self, config: &serde_json::Value);
    fn finish(&self);
}

/// Console logger implementation
pub struct ConsoleLogger {
    level: String,
}

impl ConsoleLogger {
    pub fn new(level: Option<String>) -> Self {
        Self {
            level: level.unwrap_or_else(|| "info".to_string()),
        }
    }
}

impl Logger for ConsoleLogger {
    fn log_metric(&self, name: &str, value: f64, step: usize) {
        println!("Step {}: {} = {:.6}", step, name, value);
    }

    fn log_text(&self, name: &str, text: &str, step: usize) {
        println!("Step {}: {} = {}", step, name, text);
    }

    fn log_config(&self, config: &serde_json::Value) {
        println!(
            "Config: {}",
            serde_json::to_string_pretty(config).unwrap_or_default()
        );
    }

    fn finish(&self) {
        println!("Logging finished");
    }
}

/// File logger implementation
#[allow(dead_code)]
pub struct FileLogger {
    path: String,
    level: String,
}

impl FileLogger {
    pub fn new(config: &FileLoggingConfig) -> Result<Self> {
        // Ensure log directory exists
        if let Some(parent) = std::path::Path::new(&config.path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        Ok(Self {
            path: config.path.clone(),
            level: config.level.clone().unwrap_or_else(|| "info".to_string()),
        })
    }

    fn write_log(&self, message: &str) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        let timestamp = chrono::Utc::now().to_rfc3339();
        writeln!(file, "[{}] {}", timestamp, message)?;
        Ok(())
    }
}

impl Logger for FileLogger {
    fn log_metric(&self, name: &str, value: f64, step: usize) {
        let message = format!("Step {}: {} = {:.6}", step, name, value);
        let _ = self.write_log(&message);
    }

    fn log_text(&self, name: &str, text: &str, step: usize) {
        let message = format!("Step {}: {} = {}", step, name, text);
        let _ = self.write_log(&message);
    }

    fn log_config(&self, config: &serde_json::Value) {
        let message = format!(
            "Config: {}",
            serde_json::to_string_pretty(config).unwrap_or_default()
        );
        let _ = self.write_log(&message);
    }

    fn finish(&self) {
        let _ = self.write_log("Logging finished");
    }
}

/// Multi-logger that combines multiple loggers
pub struct MultiLogger {
    loggers: Vec<Box<dyn Logger>>,
}

impl MultiLogger {
    pub fn new() -> Self {
        Self {
            loggers: Vec::new(),
        }
    }

    pub fn add_logger(&mut self, logger: Box<dyn Logger>) {
        self.loggers.push(logger);
    }

    pub fn from_config(config: &LoggingConfig) -> Result<Self> {
        let mut multi_logger = Self::new();

        // Add console logger
        if let Some(level) = &config.console_level {
            multi_logger.add_logger(Box::new(ConsoleLogger::new(Some(level.clone()))));
        }

        // Add file logger
        if let Some(file_config) = &config.file_logging {
            if file_config.enabled {
                multi_logger.add_logger(Box::new(FileLogger::new(file_config)?));
            }
        }

        // Add W&B logger (placeholder)
        if let Some(wandb_config) = &config.wandb {
            if wandb_config.enabled {
                // Placeholder for W&B integration
                println!("W&B logging enabled for project: {}", wandb_config.project);
            }
        }

        Ok(multi_logger)
    }
}

impl Logger for MultiLogger {
    fn log_metric(&self, name: &str, value: f64, step: usize) {
        for logger in &self.loggers {
            logger.log_metric(name, value, step);
        }
    }

    fn log_text(&self, name: &str, text: &str, step: usize) {
        for logger in &self.loggers {
            logger.log_text(name, text, step);
        }
    }

    fn log_config(&self, config: &serde_json::Value) {
        for logger in &self.loggers {
            logger.log_config(config);
        }
    }

    fn finish(&self) {
        for logger in &self.loggers {
            logger.finish();
        }
    }
}
