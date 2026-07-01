//! DeepSeek-V4 mHC fused Sinkhorn.
//!
//! The hyper-connection combine matrix is tiny (`[.., nh, nh]`, `nh` = n_hc = 4)
//! but the 20-iteration row/col alternation, expressed as elementwise tensor ops,
//! was ~120 tiny kernel launches per call — a swarm of 16-thread kernels that
//! dominated V4 decode (nsys: ~20% of GPU time, ~10.7k tiny kernels/token). This
//! runs the whole iteration in ONE launch, on-device, in f32 (the tensor-op path
//! ran it in bf16 — low precision for a 4×4 normalization). Semantics match the
//! reference exactly: `softmax(src) + eps`, `col_norm` (denom = colsum+eps), then
//! `(iters-1)× (row_norm, col_norm)`.

use crate::{CpuStorage, Layout, Result, Shape, Tensor};

#[derive(Debug, Clone, Copy)]
struct HcSinkhorn {
    iters: usize,
    eps: f64,
}

fn row_norm(mat: &mut [f32], nh: usize, eps: f32) {
    for r in 0..nh {
        let mut s = eps;
        for c in 0..nh {
            s += mat[r * nh + c];
        }
        for c in 0..nh {
            mat[r * nh + c] /= s;
        }
    }
}

fn col_norm(mat: &mut [f32], nh: usize, eps: f32) {
    for c in 0..nh {
        let mut s = eps;
        for r in 0..nh {
            s += mat[r * nh + c];
        }
        for r in 0..nh {
            mat[r * nh + c] /= s;
        }
    }
}

/// CPU reference (also the fallback for non-CUDA devices). Bit-for-bit the same
/// arithmetic as the CUDA kernel (both f32).
fn sinkhorn_cpu_f32(data: &[f32], nmat: usize, nh: usize, iters: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for m in 0..nmat {
        let base = m * nh * nh;
        let src = &data[base..base + nh * nh];
        let mat = &mut out[base..base + nh * nh];
        for r in 0..nh {
            let row = &src[r * nh..r * nh + nh];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for c in 0..nh {
                let e = (row[c] - mx).exp();
                mat[r * nh + c] = e;
                sum += e;
            }
            for c in 0..nh {
                mat[r * nh + c] = mat[r * nh + c] / sum + eps;
            }
        }
        col_norm(mat, nh, eps);
        for _ in 1..iters {
            row_norm(mat, nh, eps);
            col_norm(mat, nh, eps);
        }
    }
    out
}

impl crate::CustomOp1 for HcSinkhorn {
    fn name(&self) -> &'static str {
        "hc_sinkhorn"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (o1, o2) = match layout.contiguous_offsets() {
            Some(x) => x,
            None => crate::bail!("hc_sinkhorn requires a contiguous input"),
        };
        let vs = match storage {
            CpuStorage::F32(v) => &v[o1..o2],
            _ => crate::bail!("hc_sinkhorn: f32 only"),
        };
        let dims = layout.shape().dims();
        let nh = dims[dims.len() - 1];
        let nmat = vs.len() / (nh * nh);
        let out = sinkhorn_cpu_f32(vs, nmat, nh, self.iters, self.eps as f32);
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        use crate::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use crate::cuda_backend::{kernels, CudaStorageSlice as S, WrapErr};

        let dev = &storage.device;
        let (o1, o2) = match layout.contiguous_offsets() {
            Some(x) => x,
            None => crate::bail!("hc_sinkhorn requires a contiguous input"),
        };
        let src = storage.as_cuda_slice::<f32>()?.slice(o1..o2);
        let dims = layout.shape().dims();
        let nh = dims[dims.len() - 1];
        let elem = layout.shape().elem_count();
        let nmat = elem / (nh * nh);

        let dst = unsafe { dev.alloc::<f32>(elem)? };
        let func = dev.get_or_load_func("hc_sinkhorn_f32", &kernels::REDUCE)?;
        let cfg = LaunchConfig {
            grid_dim: (nmat as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: (nh * nh * std::mem::size_of::<f32>()) as u32,
        };
        let stream = dev.cuda_stream();
        let mut builder = stream.launch_builder(&func);
        let nh_i = nh as i32;
        let iters_i = self.iters as i32;
        let eps = self.eps as f32;
        builder
            .arg(&src)
            .arg(&dst)
            .arg(&nh_i)
            .arg(&iters_i)
            .arg(&eps);
        unsafe { builder.launch(cfg) }.w()?;

        let out = crate::cuda_backend::CudaStorage {
            slice: S::F32(dst),
            device: dev.clone(),
        };
        Ok((out, layout.shape().clone()))
    }
}

impl Tensor {
    /// Fused mHC Sinkhorn over the trailing `[nh, nh]` matrices: `softmax(src)+eps`,
    /// `col_norm`, then `(iters-1)× (row_norm, col_norm)`, all in f32. Runs the whole
    /// iteration in one CUDA launch (falls back to a CPU reference off-device).
    /// Returns the caller's dtype.
    pub fn hc_sinkhorn(&self, iters: usize, eps: f64) -> Result<Tensor> {
        let orig = self.dtype();
        let x = self.to_dtype(crate::DType::F32)?.contiguous()?;
        let r = x.apply_op1_no_bwd(&HcSinkhorn { iters, eps })?;
        r.to_dtype(orig)
    }
}
