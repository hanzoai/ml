// This example illustrates how to implement custom operations. These operations can provide their
// own forward pass (CPU and GPU versions) as well as their backward pass.
//
// In this example we add the RMS normalization operation and implement it for f32.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[rustfmt::skip]
#[cfg(feature = "cuda")]
mod cuda_kernels;

use clap::Parser;

use hanzo_ml_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

struct LayerNorm {
    eps: f32,
}

impl CustomOp1 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (dim1, dim2) = layout.shape().dims2()?;
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => hanzo::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        let mut dst = Vec::with_capacity(dim1 * dim2);
        for idx1 in 0..dim1 {
            let src = &src[idx1 * dim2..(idx1 + 1) * dim2];
            let variance = src.iter().map(|x| x * x).sum::<f32>();
            let s_variance = 1f32 / (variance / dim2 as f32 + self.eps).sqrt();
            dst.extend(src.iter().map(|x| x * s_variance))
        }
        let storage = hanzo::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &hanzo::CudaStorage,
        layout: &Layout,
    ) -> Result<(hanzo::CudaStorage, Shape)> {
        use hanzo_ml_core::backend::BackendStorage;
        use hanzo_ml_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use hanzo_ml_core::cuda_backend::WrapErr;
        let (d1, d2) = layout.shape().dims2()?;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        let dev = storage.device().clone();
        let slice = storage.as_cuda_slice::<f32>()?;
        let slice = match layout.contiguous_offsets() {
            None => hanzo::bail!("input has to be contiguous"),
            Some((o1, o2)) => slice.slice(o1..o2),
        };
        let elem_count = layout.shape().elem_count();
        let dst = unsafe { dev.alloc::<f32>(elem_count) }?;
        let func =
            dev.get_or_load_custom_func("rms_f32", "mymodule", cuda_kernels::LAYERNORM_KERNELS)?;
        let cfg = LaunchConfig {
            grid_dim: (d1, 1, 1),
            block_dim: (d2, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&dst);
        builder.arg(&slice);
        hanzo::builder_arg!(builder, self.eps, d1, d2);
        unsafe { builder.launch(cfg) }.w()?;

        let dst = hanzo::CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, layout.shape().clone()))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = hanzo_examples::device(args.cpu)?;
    let t = Tensor::arange(0f32, 14f32, &device)?.reshape((2, 7))?;
    println!("{t}");
    let t = t.apply_op1(LayerNorm { eps: 1e-5 })?;
    println!("{t}");
    Ok(())
}
