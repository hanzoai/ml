pub mod attention_residuals;
pub mod cpu_flash;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod rocm;
pub mod varlen;

use hanzo_ml::{Result, Tensor};

pub use attention_residuals::AttentionResiduals;
pub use cpu_flash::flash_attn;
pub use cpu_flash::varlen::flash_attn_varlen_cpu;
pub use rocm::{rocm_flash_attn, rocm_flash_attn_decode};
pub use varlen::flash_attn_varlen_unfused;

#[derive(Debug, Clone, Default)]
pub enum AttnMask {
    #[default]
    None,
    Causal {
        kv_offset: usize,
    },
    Mask(Tensor),
}

impl AttnMask {
    #[inline]
    pub fn causal() -> Self {
        AttnMask::Causal { kv_offset: 0 }
    }

    #[inline]
    pub fn causal_with_offset(kv_offset: usize) -> Self {
        AttnMask::Causal { kv_offset }
    }

    #[inline]
    pub fn is_causal(&self) -> bool {
        matches!(self, AttnMask::Causal { .. })
    }

    #[inline]
    pub fn kv_offset(&self) -> usize {
        match self {
            AttnMask::Causal { kv_offset } => *kv_offset,
            _ => 0,
        }
    }
}

/// Whether [`packed`] runs on `dev` in `dtype` with heads of `head` dimensions.
pub fn packs(dev: &hanzo_ml::Device, dtype: hanzo_ml::DType, head: usize) -> bool {
    match dev {
        hanzo_ml::Device::Cpu => dtype == hanzo_ml::DType::F32,
        hanzo_ml::Device::Cuda(_) => {
            cfg!(feature = "cuda") && dtype == hanzo_ml::DType::BF16 && head == 64
        }
        _ => false,
    }
}

/// [`cpu_flash::packed::packed`] on the device of `qkv`; on CUDA it carries a gradient.
#[allow(clippy::too_many_arguments)]
pub fn packed(
    qkv: &Tensor,
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    match qkv.device() {
        hanzo_ml::Device::Cpu => {
            cpu_flash::packed::packed(qkv, lens, heads, window, cos, sin, scale)
        }
        #[cfg(feature = "cuda")]
        hanzo_ml::Device::Cuda(_) => cuda::packed(qkv, lens, heads, window, cos, sin, scale),
        d => hanzo_ml::bail!("packed attention: no kernel on {d:?}"),
    }
}
