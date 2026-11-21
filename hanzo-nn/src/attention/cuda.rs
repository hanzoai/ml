//! Attention over packed sequences on CUDA: hanzo-kernels `attention.cu` as one custom op.

use half::bf16;
use hanzo_ml::backend::BackendStorage;
use hanzo_ml::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use hanzo_ml::cuda_backend::{kernels, CudaStorageSlice, WrapErr};
use hanzo_ml::{CudaStorage, CustomOp3, DType, InplaceOpN, Layout, Result, Shape, Storage, Tensor};
use std::sync::Mutex;

pub const HEAD: usize = 64;
const TILE: usize = 64;
const THREADS: u32 = 128;

/// [`super::packed`] for contiguous bf16 inputs with heads of [`HEAD`].
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
    let (t, w3) = qkv.dims2()?;
    let (p, half) = cos.dims2()?;
    if w3 != 3 * heads * HEAD || half * 2 != HEAD || sin.dims2()? != (p, half) {
        hanzo_ml::bail!(
            "packed attention: qkv {:?}, cos {:?}, {heads} heads of {HEAD}",
            qkv.shape(),
            cos.shape()
        );
    }
    if lens.iter().sum::<usize>() != t || lens.iter().any(|&l| l > p) {
        hanzo_ml::bail!("packed attention: lengths {lens:?} over {t} tokens, {p} positions");
    }
    if qkv.dtype() != DType::BF16 || cos.dtype() != DType::BF16 || sin.dtype() != DType::BF16 {
        hanzo_ml::bail!("packed attention: bf16 only on CUDA");
    }
    if !(qkv.is_contiguous() && cos.is_contiguous() && sin.is_contiguous()) {
        hanzo_ml::bail!("packed attention: contiguous inputs only");
    }
    let mut cu = Vec::with_capacity(lens.len() + 1);
    let mut tiles = Vec::new();
    let mut at = 0u32;
    for (s, &l) in lens.iter().enumerate() {
        cu.push(at);
        for i0 in (0..l).step_by(TILE) {
            tiles.extend([s as u32, i0 as u32]);
        }
        at += l as u32;
    }
    cu.push(at);
    if tiles.is_empty() {
        return Tensor::zeros((t, heads * HEAD), DType::BF16, qkv.device());
    }
    let dev = qkv.device();
    let tiles_n = tiles.len() / 2;
    let op = Flash {
        heads,
        window: window.map_or(-1, |w| w as i32),
        scale,
        total: t,
        tiles: tiles_n,
        cu: Tensor::from_vec(cu, lens.len() + 1, dev)?,
        tile: Tensor::from_vec(tiles, 2 * tiles_n, dev)?,
        lse: Mutex::new(None),
    };
    qkv.apply_op3(cos, sin, op)
}

struct Flash {
    heads: usize,
    window: i32,
    scale: f32,
    total: usize,
    tiles: usize,
    /// `[S + 1]`: each sequence's first token.
    cu: Tensor,
    /// `[NT, 2]`: each tile's sequence and first row.
    tile: Tensor,
    /// `[H, T]`: each row's log-sum-exp, from the forward for the backward.
    lse: Mutex<Option<CudaSlice<f32>>>,
}

fn hold(t: &Tensor) -> Result<(impl std::ops::Deref<Target = Storage> + '_, usize)> {
    let (s, l) = t.storage_and_layout();
    if !matches!(&*s, Storage::Cuda(_)) {
        hanzo_ml::bail!("packed attention: index tensor off the device")
    }
    Ok((s, l.start_offset()))
}

fn cuda(s: &Storage) -> &CudaStorage {
    match s {
        Storage::Cuda(c) => c,
        _ => unreachable!("checked by hold"),
    }
}

fn threads(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Flash {
    fn grid(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.tiles as u32, self.heads as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl CustomOp3 for Flash {
    fn name(&self) -> &'static str {
        "packed-attention"
    }

    fn cpu_fwd(
        &self,
        _: &hanzo_ml::CpuStorage,
        _: &Layout,
        _: &hanzo_ml::CpuStorage,
        _: &Layout,
        _: &hanzo_ml::CpuStorage,
        _: &Layout,
    ) -> Result<(hanzo_ml::CpuStorage, Shape)> {
        hanzo_ml::bail!("packed attention: the CUDA op on the CPU")
    }

    fn cuda_fwd(
        &self,
        qkv: &CudaStorage,
        ql: &Layout,
        cos: &CudaStorage,
        cl: &Layout,
        sin: &CudaStorage,
        sl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = qkv.device().clone();
        let (t, h) = (self.total, self.heads);
        let qkv = qkv.as_cuda_slice::<bf16>()?.slice(ql.start_offset()..);
        let cos = cos.as_cuda_slice::<bf16>()?.slice(cl.start_offset()..);
        let sin = sin.as_cuda_slice::<bf16>()?.slice(sl.start_offset()..);
        let ((cg, co), (tg, to)) = (hold(&self.cu)?, hold(&self.tile)?);
        let cu = cuda(&cg).as_cuda_slice::<u32>()?.slice(co..);
        let tile = cuda(&tg).as_cuda_slice::<u32>()?.slice(to..);
        // SAFETY: the kernel writes every row of both.
        let out = unsafe { dev.alloc::<bf16>(t * h * HEAD)? };
        let lse = unsafe { dev.alloc::<f32>(h * t)? };
        let func = dev.get_or_load_func("flash_fwd_bf16", &kernels::ATTENTION)?;
        let mut b = func.builder();
        b.arg(&qkv);
        b.arg(&cos);
        b.arg(&sin);
        b.arg(&cu);
        b.arg(&tile);
        hanzo_ml::builder_arg!(b, h as u32, self.window, self.scale);
        b.arg(&out);
        b.arg(&lse);
        hanzo_ml::builder_arg!(b, t as u32);
        // SAFETY: ffi.
        unsafe { b.launch(self.grid()) }.w()?;
        *self.lse.lock().expect("lse lock") = Some(lse);
        Ok((
            CudaStorage {
                slice: CudaStorageSlice::BF16(out),
                device: dev,
            },
            Shape::from_dims(&[t, h * HEAD]),
        ))
    }

    fn bwd(
        &self,
        qkv: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        out: &Tensor,
        grad: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        let dout = grad.contiguous()?;
        // SAFETY: the kernels write every element.
        let dqkv = unsafe { qkv.empty_like()? };
        dqkv.inplace_op([qkv, cos, sin, out, &dout], self)?;
        Ok((Some(dqkv), None, None))
    }
}

impl InplaceOpN<5> for Flash {
    fn name(&self) -> &'static str {
        "packed-attention-bwd"
    }

    fn cuda_fwd(
        &self,
        dqkv: &mut CudaStorage,
        dl: &Layout,
        [(qkv, ql), (cos, cl), (sin, sl), (out, ol), (dout, dol)]: [(&CudaStorage, &Layout); 5],
    ) -> Result<()> {
        let dev = dqkv.device().clone();
        let (t, h) = (self.total, self.heads);
        let lse = self.lse.lock().expect("lse lock").take().ok_or_else(|| {
            hanzo_ml::Error::Msg("packed attention: backward before forward".into())
        })?;
        let mut dqkv = dqkv
            .as_cuda_slice_mut::<bf16>()?
            .slice_mut(dl.start_offset()..);
        let qkv = qkv.as_cuda_slice::<bf16>()?.slice(ql.start_offset()..);
        let cos = cos.as_cuda_slice::<bf16>()?.slice(cl.start_offset()..);
        let sin = sin.as_cuda_slice::<bf16>()?.slice(sl.start_offset()..);
        let out = out.as_cuda_slice::<bf16>()?.slice(ol.start_offset()..);
        let dout = dout.as_cuda_slice::<bf16>()?.slice(dol.start_offset()..);
        let ((cg, co), (tg, to)) = (hold(&self.cu)?, hold(&self.tile)?);
        let cu = cuda(&cg).as_cuda_slice::<u32>()?.slice(co..);
        let tile = cuda(&tg).as_cuda_slice::<u32>()?.slice(to..);
        // SAFETY: the delta kernel writes every element.
        let delta = unsafe { dev.alloc::<f32>(h * t)? };
        let dq = dev.alloc_zeros::<f32>(t * h * HEAD)?;

        let f = dev.get_or_load_func("flash_delta_bf16", &kernels::ATTENTION)?;
        let mut b = f.builder();
        b.arg(&out);
        b.arg(&dout);
        hanzo_ml::builder_arg!(b, h as u32, t as u32);
        b.arg(&delta);
        // SAFETY: ffi.
        unsafe { b.launch(threads(t * h)) }.w()?;

        let f = dev.get_or_load_func("flash_bwd_bf16", &kernels::ATTENTION)?;
        let mut b = f.builder();
        b.arg(&qkv);
        b.arg(&cos);
        b.arg(&sin);
        b.arg(&cu);
        b.arg(&tile);
        hanzo_ml::builder_arg!(b, h as u32, self.window, self.scale);
        b.arg(&dout);
        b.arg(&lse);
        b.arg(&delta);
        b.arg(&dq);
        b.arg(&mut dqkv);
        hanzo_ml::builder_arg!(b, t as u32);
        // SAFETY: ffi.
        unsafe { b.launch(self.grid()) }.w()?;

        let f = dev.get_or_load_func("flash_dq_bf16", &kernels::ATTENTION)?;
        let mut b = f.builder();
        b.arg(&dq);
        b.arg(&cos);
        b.arg(&sin);
        b.arg(&cu);
        hanzo_ml::builder_arg!(
            b,
            (self.cu.elem_count() - 1) as u32,
            h as u32,
            self.scale,
            t as u32
        );
        b.arg(&mut dqkv);
        // SAFETY: ffi.
        unsafe { b.launch(threads(t * h * HEAD / 2)) }.w()?;
        Ok(())
    }
}
