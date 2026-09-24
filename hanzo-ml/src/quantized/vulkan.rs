//! Quantized blocks resident on a Vulkan device.
//!
//! The buffer holds the blocks in the one layout the dtype's kernels read (see [`Layout`]), so the
//! decode matvec, the prefill GEMM and the MoE matvec all bind the same resident copy. The host keeps
//! none: on unified-memory parts (Strix Halo) a host copy beside the device copy doubled every weight
//! in the one shared pool. The paths that compute on the CPU (dequantizing a type with no kernel,
//! `data`) copy the blocks back and undo the layout on demand.

use std::borrow::Cow;
use std::sync::Arc;

use super::{GgmlDType, QuantizedType};
use crate::backend::BackendStorage;
use crate::{CpuStorage, Result, VulkanDevice, VulkanStorage};

/// The bytes of `words`. The device reads the buffer little-endian, as every target host is.
fn view(words: &[u32]) -> &[u8] {
    // SAFETY: u8 has no alignment or validity requirements; the view covers exactly the words.
    unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) }
}

fn view_mut(words: &mut [u32]) -> &mut [u8] {
    // SAFETY: as `view`, with the unique borrow of `words` carried over.
    unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 4) }
}

/// How a dtype's GGML blocks sit in the device buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// The GGML bytes unchanged; the kernels byte-address them.
    Raw,
    /// Each block zero-padded at its end to a whole number of u32 words, for kernels that read the
    /// block's fields as aligned words (Q6_K 210 -> 212 B, Q3_K and IQ3_S 110 -> 112 B, ...).
    Padded,
    /// Q8_0 in 9 words: the f16 scale in the low half of word 0, the 32 int8 in words 1..9.
    Q8,
}

impl Layout {
    fn of(dtype: GgmlDType) -> Self {
        match dtype {
            GgmlDType::Q8_0 => Self::Q8,
            GgmlDType::Q6K
            | GgmlDType::Q3K
            | GgmlDType::IQ2_XXS
            | GgmlDType::IQ2_XS
            | GgmlDType::IQ2_S
            | GgmlDType::IQ1_S
            | GgmlDType::IQ1_M
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ3_S => Self::Padded,
            _ => Self::Raw,
        }
    }

    /// Words per block in the device buffer, for the block layouts.
    fn words(self, dtype: GgmlDType) -> usize {
        match self {
            Self::Q8 => 9,
            _ => dtype.type_size().div_ceil(4),
        }
    }

    /// GGML bytes to device words.
    fn pack(self, bytes: &[u8], dtype: GgmlDType) -> Vec<u32> {
        let size = dtype.type_size();
        match self {
            Self::Raw => {
                let mut words = vec![0u32; bytes.len().div_ceil(4).max(1)];
                view_mut(&mut words)[..bytes.len()].copy_from_slice(bytes);
                words
            }
            Self::Padded => {
                let stride = self.words(dtype);
                let mut words = vec![0u32; bytes.len() / size * stride];
                for (src, dst) in bytes.chunks_exact(size).zip(words.chunks_exact_mut(stride)) {
                    view_mut(dst)[..size].copy_from_slice(src);
                }
                words
            }
            Self::Q8 => {
                let mut words = vec![0u32; bytes.len() / size * 9];
                for (src, dst) in bytes.chunks_exact(size).zip(words.chunks_exact_mut(9)) {
                    dst[0] = u16::from_le_bytes([src[0], src[1]]) as u32;
                    view_mut(&mut dst[1..]).copy_from_slice(&src[2..]);
                }
                words
            }
        }
    }

    /// Device words back to `len` GGML bytes.
    fn unpack(self, words: &[u32], dtype: GgmlDType, len: usize) -> Vec<u8> {
        let size = dtype.type_size();
        let all = view(words);
        match self {
            Self::Raw => all[..len].to_vec(),
            Self::Padded => {
                let stride = self.words(dtype) * 4;
                let mut bytes = Vec::with_capacity(len);
                for blk in all.chunks_exact(stride).take(len / size) {
                    bytes.extend_from_slice(&blk[..size]);
                }
                bytes
            }
            Self::Q8 => {
                let mut bytes = Vec::with_capacity(len);
                for blk in all.chunks_exact(36).take(len / size) {
                    bytes.extend_from_slice(&blk[..2]);
                    bytes.extend_from_slice(&blk[4..36]);
                }
                bytes
            }
        }
    }
}

pub struct QVulkanStorage {
    data: Arc<VulkanStorage>,
    dtype: GgmlDType,
    size_in_bytes: usize,
    device: VulkanDevice,
}

impl QVulkanStorage {
    /// Upload `bytes`, GGML blocks of `dtype`, in the layout the dtype's kernels read.
    pub fn new(bytes: &[u8], dtype: GgmlDType, device: &VulkanDevice) -> Result<Self> {
        let words = Layout::of(dtype).pack(bytes, dtype);
        Ok(Self {
            data: Arc::new(device.upload_u32(&words)?),
            dtype,
            size_in_bytes: bytes.len(),
            device: device.clone(),
        })
    }

    /// Zeroed blocks for `elem_count` elements.
    pub fn zeros(elem_count: usize, dtype: GgmlDType, device: &VulkanDevice) -> Result<Self> {
        let bytes = elem_count / dtype.block_size() * dtype.type_size();
        Self::new(&vec![0u8; bytes], dtype, device)
    }

    /// Upload host blocks.
    pub fn from_host(host: &dyn QuantizedType, device: &VulkanDevice) -> Result<Self> {
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

    pub fn device(&self) -> &VulkanDevice {
        &self.device
    }

    /// The resident blocks, as the quant kernels take them.
    pub fn resident(&self) -> Arc<VulkanStorage> {
        self.data.clone()
    }

    /// The GGML bytes, copied back to the host.
    pub fn bytes(&self) -> Result<Vec<u8>> {
        match self.data.to_cpu_storage()? {
            CpuStorage::U32(words) => {
                Ok(Layout::of(self.dtype).unpack(&words, self.dtype, self.size_in_bytes))
            }
            other => crate::bail!("resident Vulkan blocks are u32, got {:?}", other.dtype()),
        }
    }

    /// Host blocks, for the paths that compute on the CPU.
    pub fn host(&self) -> Result<Box<dyn QuantizedType>> {
        Ok(self.dtype.from_data(Cow::Owned(self.bytes()?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 37 + 11) as u8).collect()
    }

    #[test]
    fn layouts_round_trip() {
        for dtype in [
            GgmlDType::Q8_0,
            GgmlDType::Q6K,
            GgmlDType::Q3K,
            GgmlDType::IQ3_S,
            GgmlDType::IQ2_XXS,
            GgmlDType::IQ1_M,
            GgmlDType::IQ4_NL,
            GgmlDType::IQ4_XS,
            GgmlDType::Q4_0,
            GgmlDType::Q4K,
            GgmlDType::F32,
        ] {
            let layout = Layout::of(dtype);
            let src = bytes(dtype.type_size() * 7);
            let words = layout.pack(&src, dtype);
            assert_eq!(layout.unpack(&words, dtype, src.len()), src, "{dtype:?}");
        }
    }

    #[test]
    fn q8_matches_the_kernel_layout() {
        let src = bytes(34 * 2);
        let words = Layout::Q8.pack(&src, GgmlDType::Q8_0);
        assert_eq!(words.len(), 18);
        assert_eq!(words[9], u16::from_le_bytes([src[34], src[35]]) as u32);
        assert_eq!(
            words[10],
            u32::from_le_bytes([src[36], src[37], src[38], src[39]])
        );
    }

    #[test]
    fn padded_blocks_start_on_words() {
        let src = bytes(110 * 2);
        let words = Layout::Padded.pack(&src, GgmlDType::IQ3_S);
        assert_eq!(words.len(), 56);
        let second = view(&words[28..]);
        assert_eq!(&second[..110], &src[110..]);
    }
}
