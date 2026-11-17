use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheStage {
    Encoded,
    Decoded,
    Transformed,
}

/// DynamicCacheManager implements the Seneca (FAST '26) inspired elastic partitioning.
/// It tracks memory pressure across the data pipeline stages.
pub struct DynamicCacheManager {
    total_capacity_bytes: usize,
    encoded_usage: Arc<AtomicUsize>,
    decoded_usage: Arc<AtomicUsize>,
    transformed_usage: Arc<AtomicUsize>,
    gpu_idle_signals: Arc<AtomicUsize>,
}

impl DynamicCacheManager {
    pub fn new(total_ram_gb: f64) -> Self {
        let total_capacity_bytes = (total_ram_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        Self {
            total_capacity_bytes,
            encoded_usage: Arc::new(AtomicUsize::new(0)),
            decoded_usage: Arc::new(AtomicUsize::new(0)),
            transformed_usage: Arc::new(AtomicUsize::new(0)),
            gpu_idle_signals: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Reports GPU idle state to trigger cache re-partitioning
    pub fn report_gpu_starvation(&self) {
        self.gpu_idle_signals.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_usage(&self, stage: CacheStage, bytes: isize) {
        let counter = match stage {
            CacheStage::Encoded => &self.encoded_usage,
            CacheStage::Decoded => &self.decoded_usage,
            CacheStage::Transformed => &self.transformed_usage,
        };

        if bytes > 0 {
            counter.fetch_add(bytes as usize, Ordering::SeqCst);
        } else {
            counter.fetch_sub(bytes.abs() as usize, Ordering::SeqCst);
        }
    }

    /// Calculates the optimal partition ratio based on Seneca's throughput balancing
    pub fn get_target_ratio(&self) -> (f32, f32, f32) {
        let idle_count = self.gpu_idle_signals.load(Ordering::Relaxed);
        
        // If GPU is starving, we prioritize Transformed and Decoded buffers
        if idle_count > 5 {
            // Aggressive prefetching for transformed data
            (0.2, 0.3, 0.5)
        } else {
            // Balanced state
            (0.33, 0.33, 0.34)
        }
    }

    pub fn available_capacity(&self) -> usize {
        let used = self.encoded_usage.load(Ordering::Relaxed) +
                   self.decoded_usage.load(Ordering::Relaxed) +
                   self.transformed_usage.load(Ordering::Relaxed);
        if used >= self.total_capacity_bytes {
            0
        } else {
            self.total_capacity_bytes - used
        }
    }
}
