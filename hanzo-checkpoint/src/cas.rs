use candle_core::Tensor;
use blake3::Hasher;
use std::collections::HashMap;
use crate::Result;

/// Chunk size for redundancy detection (default 1MB)
const CHUNK_SIZE: usize = 1024 * 1024;

pub struct Chunk {
    pub hash: String,
    pub data: Vec<u8>,
}

pub struct ContentAddressableStore {
    /// Simulates VFS/S3 storage of unique chunks
    pub registry: HashMap<String, Vec<u8>>,
}

impl ContentAddressableStore {
    pub fn new() -> Self {
        Self { registry: HashMap::new() }
    }

    /// Processes a tensor and returns a list of chunk hashes
    pub fn push_tensor(&mut self, name: &str, tensor: &Tensor) -> Result<Vec<String>> {
        let bytes = tensor.to_vec1::<u8>().map_err(|_| {
            // Fallback for non-u8 tensors: flatten to raw bytes
            // In a real impl, we'd use candle's storage access
            candle_core::Error::Msg("Only u8-viewable tensors supported in MVP".to_string())
        })?;

        let mut chunk_hashes = Vec::new();

        for raw_chunk in bytes.chunks(CHUNK_SIZE) {
            let mut hasher = Hasher::new();
            hasher.update(raw_chunk);
            let hash = hasher.finalize().to_string();

            if !self.registry.contains_key(&hash) {
                self.registry.insert(hash.clone(), raw_chunk.to_vec());
            }
            chunk_hashes.push(hash);
        }

        Ok(chunk_hashes)
    }

    pub fn get_chunk(&self, hash: &str) -> Option<&Vec<u8>> {
        self.registry.get(hash)
    }
}
