//! ROCm WMMA flash-attention (gfx11 matrix cores). Routed from the engine's ROCm attention
//! dispatch for causal prefill at long sequence lengths, where it beats rocBLAS+softmax.

use hanzo_ml::{Layout, Result, Shape, Tensor};

/// Flash-attention forward over `[B, Hq, Lq, D]` Q and `[B, Hkv, Lk, D]` K/V (GQA: `Hq % Hkv == 0`),
/// head dim D == 128, f16/bf16. ROCm-only; the kernel applies causal masking internally when set.
#[derive(Debug, Clone)]
struct RocmFlashAttn {
    scale: f32,
    causal: bool,
}

impl hanzo_ml::CustomOp3 for RocmFlashAttn {
    fn name(&self) -> &'static str {
        "rocm-flash-attn"
    }

    fn cpu_fwd(
        &self,
        _s1: &hanzo_ml::CpuStorage,
        _l1: &Layout,
        _s2: &hanzo_ml::CpuStorage,
        _l2: &Layout,
        _s3: &hanzo_ml::CpuStorage,
        _l3: &Layout,
    ) -> Result<(hanzo_ml::CpuStorage, Shape)> {
        hanzo_ml::bail!("rocm-flash-attn has no CPU path; route to naive_sdpa off ROCm")
    }

    #[cfg(feature = "rocm")]
    fn rocm_fwd(
        &self,
        s1: &hanzo_ml::RocmStorage,
        l1: &Layout,
        s2: &hanzo_ml::RocmStorage,
        l2: &Layout,
        s3: &hanzo_ml::RocmStorage,
        l3: &Layout,
    ) -> Result<(hanzo_ml::RocmStorage, Shape)> {
        let out = s1.flash_attn(l1, s2, l2, s3, l3, self.scale, self.causal)?;
        Ok((out, l1.shape().clone()))
    }
}

/// Run the WMMA flash-attention kernel. Inputs are `[B, Hq, Lq, 128]` (Q) and `[B, Hkv, Lk, 128]`
/// (K, V), contiguous, f16 or bf16, on the same ROCm device. Returns `[B, Hq, Lq, 128]`.
/// `scale` is the softmax scale (`1/sqrt(head_dim)`); `causal` enables the upper-triangle skip.
pub fn rocm_flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    q.apply_op3_no_bwd(&k, &v, &RocmFlashAttn { scale, causal })
}
