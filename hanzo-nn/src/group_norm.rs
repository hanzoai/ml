//! Group Normalization.
//!
//! This layer applies Group Normalization over a mini-batch of inputs.
use hanzo_ml::{DType, Layout, Result, Shape, Tensor};

#[cfg(feature = "cuda")]
use hanzo_ml::CpuStorage;

// Fused GroupNorm op: one CUDA kernel computes mean/var (f32 accum) + affine in a single launch,
// replacing the ~12-kernel reshape/cast/sum/sub/sqr/div/mul/add chain in the decomposed path.
// Inputs: contiguous NCHW activation, per-channel weight[C], per-channel bias[C].
#[cfg(feature = "cuda")]
struct GroupNormOp {
    num_groups: usize,
    channels_per_group: usize,
    spatial: usize,
    eps: f32,
}

#[cfg(feature = "cuda")]
impl hanzo_ml::CustomOp3 for GroupNormOp {
    fn name(&self) -> &'static str {
        "group-norm"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        hanzo_ml::bail!("group-norm fused op has no cpu path; use the decomposed fallback")
    }

    fn cuda_fwd(
        &self,
        s1: &hanzo_ml::CudaStorage,
        l1: &Layout,
        s2: &hanzo_ml::CudaStorage,
        _l2: &Layout,
        s3: &hanzo_ml::CudaStorage,
        _l3: &Layout,
    ) -> Result<(hanzo_ml::CudaStorage, Shape)> {
        use hanzo_ml::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use hanzo_ml::cuda_backend::{kernel_name, kernels, WrapErr};
        use hanzo_ml::{CudaDevice, WithDType};

        fn run<T: DeviceRepr + WithDType>(
            op: &GroupNormOp,
            src: &CudaSlice<T>,
            l1: &Layout,
            alpha: &CudaSlice<T>,
            beta: &CudaSlice<T>,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let src = match l1.contiguous_offsets() {
                Some((o1, o2)) => src.slice(o1..o2),
                None => hanzo_ml::bail!("group-norm input must be contiguous"),
            };
            let el = l1.shape().elem_count();
            let n_cols = op.channels_per_group * op.spatial;
            let n_rows = el / n_cols;
            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_func(&kernel_name::<T>("groupnorm"), &kernels::REDUCE)?;
            let dst = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&dst);
            builder.arg(alpha);
            builder.arg(beta);
            hanzo_ml::builder_arg!(
                builder,
                n_cols as i32,
                op.num_groups as i32,
                op.channels_per_group as i32,
                op.spatial as i32,
                block_size as i32,
                op.eps
            );
            unsafe { builder.launch(cfg) }.w()?;
            Ok(dst)
        }

        use hanzo_ml::backend::BackendStorage;
        use hanzo_ml::cuda_backend::CudaStorageSlice as S;
        let dev = s1.device().clone();
        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (S::F16(x), S::F16(a), S::F16(b)) => S::F16(run(self, x, l1, a, b, &dev)?),
            (S::BF16(x), S::BF16(a), S::BF16(b)) => S::BF16(run(self, x, l1, a, b, &dev)?),
            (S::F32(x), S::F32(a), S::F32(b)) => S::F32(run(self, x, l1, a, b, &dev)?),
            (S::F64(x), S::F64(a), S::F64(b)) => S::F64(run(self, x, l1, a, b, &dev)?),
            _ => hanzo_ml::bail!("group-norm: unsupported/mismatched dtypes"),
        };
        let dst = hanzo_ml::cuda_backend::CudaStorage { slice, device: dev };
        Ok((dst, l1.shape().clone()))
    }
}

// This group norm version handles both weight and bias so removes the mean.
#[derive(Clone, Debug)]
pub struct GroupNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    num_channels: usize,
    num_groups: usize,
}

impl GroupNorm {
    pub fn new(
        weight: Tensor,
        bias: Tensor,
        num_channels: usize,
        num_groups: usize,
        eps: f64,
    ) -> Result<Self> {
        if !num_channels.is_multiple_of(num_groups) {
            hanzo_ml::bail!(
                "GroupNorm: num_groups ({num_groups}) must divide num_channels ({num_channels})"
            )
        }
        Ok(Self {
            weight,
            bias,
            eps,
            num_channels,
            num_groups,
        })
    }
}

impl crate::Module for GroupNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_shape = x.dims();
        if x_shape.len() <= 2 {
            hanzo_ml::bail!("input rank for GroupNorm should be at least 3");
        }
        let (b_sz, n_channels) = (x_shape[0], x_shape[1]);
        let hidden_size = x_shape[2..].iter().product::<usize>() * n_channels / self.num_groups;
        if n_channels != self.num_channels {
            hanzo_ml::bail!(
                "unexpected num-channels in GroupNorm ({n_channels} <> {}",
                self.num_channels
            )
        }
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() && x.is_contiguous() {
            let spatial = x_shape[2..].iter().product::<usize>();
            let op = GroupNormOp {
                num_groups: self.num_groups,
                channels_per_group: n_channels / self.num_groups,
                spatial,
                eps: self.eps as f32,
            };
            return x.apply_op3_no_bwd(&self.weight, &self.bias, &op);
        }

        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x = x.reshape((b_sz, self.num_groups, hidden_size))?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        let mut w_dims = vec![1; x_shape.len()];
        w_dims[1] = n_channels;
        let weight = self.weight.reshape(w_dims.clone())?;
        let bias = self.bias.reshape(w_dims)?;
        x_normed
            .to_dtype(x_dtype)?
            .reshape(x_shape)?
            .broadcast_mul(&weight)?
            .broadcast_add(&bias)
    }
}

pub fn group_norm(
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    vb: crate::VarBuilder,
) -> Result<GroupNorm> {
    let weight = vb.get_with_hints(num_channels, "weight", crate::Init::Const(1.))?;
    let bias = vb.get_with_hints(num_channels, "bias", crate::Init::Const(0.))?;
    GroupNorm::new(weight, bias, num_channels, num_groups, eps)
}
