use crate::{
    Buffer, CommandQueue, ComputePipeline, Function, Library, MTLResourceOptions, MetalKernelError,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice};
use std::{ffi::c_void, ptr};

#[derive(Clone, Debug)]
pub struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
}
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

/// Apple-silicon chip class, used by the quant kernels to pick threadgroup tunings. Detected from the
/// MTLDevice name (e.g. "Apple M4 Max"). Only `Ultra` is special-cased downstream; the rest share the
/// default path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDeviceType {
    Ultra,
    Max,
    Pro,
    Other,
}

impl AsRef<ProtocolObject<dyn MTLDevice>> for Device {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }
}

impl Device {
    pub fn registry_id(&self) -> u64 {
        self.as_ref().registryID()
    }

    /// Chip class from the device name ("Apple M4 Max" -> Max). Drives kernel threadgroup tuning.
    pub fn device_type(&self) -> MetalDeviceType {
        let name = self.as_ref().name().to_string();
        if name.contains("Ultra") {
            MetalDeviceType::Ultra
        } else if name.contains("Max") {
            MetalDeviceType::Max
        } else if name.contains("Pro") {
            MetalDeviceType::Pro
        } else {
            MetalDeviceType::Other
        }
    }

    pub fn all() -> Vec<Self> {
        MTLCreateSystemDefaultDevice()
            .into_iter()
            .map(|raw| Device { raw })
            .collect()
    }

    pub fn system_default() -> Option<Self> {
        MTLCreateSystemDefaultDevice().map(|raw| Device { raw })
    }

    pub fn new_buffer(
        &self,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        self.as_ref()
            .newBufferWithLength_options(length, options)
            .map(Buffer::new)
            .ok_or(MetalKernelError::FailedToCreateResource(
                "Buffer".to_string(),
            ))
    }

    pub fn new_buffer_with_data(
        &self,
        pointer: *const c_void,
        length: usize,
        options: MTLResourceOptions,
    ) -> Result<Buffer, MetalKernelError> {
        let pointer = ptr::NonNull::new(pointer as *mut c_void).unwrap();
        unsafe {
            self.as_ref()
                .newBufferWithBytes_length_options(pointer, length, options)
                .map(Buffer::new)
                .ok_or(MetalKernelError::FailedToCreateResource(
                    "Buffer".to_string(),
                ))
        }
    }

    pub fn new_library_with_source(
        &self,
        source: &str,
        options: Option<&MTLCompileOptions>,
    ) -> Result<Library, MetalKernelError> {
        let raw = self
            .as_ref()
            .newLibraryWithSource_options_error(&NSString::from_str(source), options)
            .unwrap();

        Ok(Library::new(raw))
    }

    pub fn new_compute_pipeline_state_with_function(
        &self,
        function: &Function,
    ) -> Result<ComputePipeline, MetalKernelError> {
        let raw = self
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .unwrap();
        Ok(ComputePipeline::new(raw))
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue, MetalKernelError> {
        let raw = self.as_ref().newCommandQueue().unwrap();
        Ok(raw)
    }

    pub fn recommended_max_working_set_size(&self) -> usize {
        self.as_ref().recommendedMaxWorkingSetSize() as usize
    }

    pub fn current_allocated_size(&self) -> usize {
        self.as_ref().currentAllocatedSize()
    }
}
