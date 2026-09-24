//! Quantized blocks resident on a ROCm device.
//!
//! The storage is the raw GGML bytes, which is what the unified quant core (`qmatvec_core`,
//! `qmmq_core`) and the MoE kernels read, so a weight exists once, on the device. The host keeps
//! no copy: on unified-memory parts (Strix Halo) a host copy beside the device copy doubled every
//! weight in the one shared pool. The few paths that compute on the CPU (dequantizing an unwired
//! type, `data`) copy the bytes back on demand.

use std::borrow::Cow;
use std::sync::Arc;

use super::{GgmlDType, QuantizedType};
use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, Result, RocmDevice, RocmStorage};

pub struct QRocmStorage {
    data: Arc<RocmStorage>,
    dtype: GgmlDType,
    size_in_bytes: usize,
    device: RocmDevice,
}

impl QRocmStorage {
    /// Upload `bytes`, GGML blocks of `dtype`.
    pub fn new(bytes: &[u8], dtype: GgmlDType, device: &RocmDevice) -> Result<Self> {
        Ok(Self {
            data: Arc::new(device.storage_from_slice(bytes)?),
            dtype,
            size_in_bytes: bytes.len(),
            device: device.clone(),
        })
    }

    /// Zeroed blocks for `elem_count` elements.
    pub fn zeros(elem_count: usize, dtype: GgmlDType, device: &RocmDevice) -> Result<Self> {
        let bytes = elem_count / dtype.block_size() * dtype.type_size();
        Self::new(&vec![0u8; bytes], dtype, device)
    }

    /// Upload host blocks.
    pub fn from_host(host: &dyn QuantizedType, device: &RocmDevice) -> Result<Self> {
        // SAFETY: `as_ptr` points at `storage_size_in_bytes` initialized bytes of `host`.
        let bytes =
            unsafe { std::slice::from_raw_parts(host.as_ptr(), host.storage_size_in_bytes()) };
        Self::new(bytes, host.dtype(), device)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn block_size(&self) -> usize {
        self.dtype.block_size()
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.size_in_bytes
    }

    pub fn device(&self) -> &RocmDevice {
        &self.device
    }

    /// The resident bytes, as the quant kernels take them.
    pub fn resident(&self) -> Arc<RocmStorage> {
        self.data.clone()
    }

    /// The bytes, copied back to the host.
    pub fn bytes(&self) -> Result<Vec<u8>> {
        match self.data.to_cpu_storage()? {
            CpuStorage::U8(bytes) => Ok(bytes),
            other => crate::bail!("resident ROCm blocks are u8, got {:?}", other.dtype()),
        }
    }

    /// Host blocks, for the paths that compute on the CPU.
    pub fn host(&self) -> Result<Box<dyn QuantizedType>> {
        Ok(self.dtype.from_data(Cow::Owned(self.bytes()?)))
    }
}
