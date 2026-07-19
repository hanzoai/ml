use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Kernel function name, carried for per-op GPU-time attribution (`METAL_PROFILE_OPS`).
    /// `Arc<str>` so the per-dispatch pipeline-cache clone is a refcount bump, not an alloc.
    name: Arc<str>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub fn new(
        raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        name: &str,
    ) -> ComputePipeline {
        ComputePipeline {
            raw,
            name: Arc::from(name),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Clone the shared name handle (refcount bump, no alloc) for per-op profiling.
    pub fn name_arc(&self) -> Arc<str> {
        self.name.clone()
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

impl AsRef<ProtocolObject<dyn MTLComputePipelineState>> for ComputePipeline {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }
}
