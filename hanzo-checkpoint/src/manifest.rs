use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckpointManifest {
    pub version: String,
    pub timestamp: u64,
    pub model_name: String,
    /// Maps tensor name to a list of chunk hashes
    pub tensors: HashMap<String, Vec<String>>,
    pub metadata: HashMap<String, String>,
}

impl CheckpointManifest {
    pub fn new(model_name: &str) -> Self {
        Self {
            version: "1.0".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model_name: model_name.to_string(),
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
}
