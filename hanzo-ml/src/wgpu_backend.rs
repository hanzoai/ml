//! wgpu / WGSL GPU backend dispatching compute shaders compiled in-driver by naga.
//!
//! Milestone 1 scope: f32 contiguous matmul + native-GGML Q8_0/Q4_0 quant matvec, validated against
//! an f64 CPU reference on the GB10 (picked via the Vulkan backend). Structurally mirrors
//! `vulkan_backend.rs`: a `WgpuDevice` + `WgpuStorage`, a `dispatch` helper that gets/creates a
//! cached `ComputePipeline`, builds a bind group, records one compute pass, and submits it. The new
//! mechanism vs the raw-Vulkan backend is readback: copy to a MAP_READ staging buffer, `map_async`,
//! then block on `device.poll(Wait)`. Anything outside the covered op set bails with `Error::Msg`.
//!
//! Portability choices that differ from the Vulkan SPIR-V kernels:
//!   - Kernel params travel in a small UNIFORM buffer at `@binding(0)` (wgpu push_constants are a
//!     non-default feature, limited on D3D12/Dozen), with the storage buffers at bindings 1..N.
//!   - Pipelines use an auto-derived bind group layout (`layout: None`), so the WGSL declaration is
//!     the single source of truth for binding types.
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

// kernel name -> WGSL source (compiled in-driver by naga). Mirrors `kernel_spv` in the Vulkan
// backend. Each kernel declares a uniform `Params` at binding 0 and storage buffers at 1..N.
fn kernel_wgsl(name: &str) -> Result<&'static str> {
    let s: &'static str = match name {
        "matmul" => include_str!("wgpu/shaders/matmul.wgsl"),
        "matmul_nt" => include_str!("wgpu/shaders/matmul_nt.wgsl"),
        "mul_mat_vec_q8_0" => include_str!("wgpu/shaders/mul_mat_vec_q8_0.wgsl"),
        "mul_mat_vec_q4_0" => include_str!("wgpu/shaders/mul_mat_vec_q4_0.wgsl"),
        "mul_mat_vec_q4k" => include_str!("wgpu/shaders/mul_mat_vec_q4k.wgsl"),
        "moe_matvec_q4_0" => include_str!("wgpu/shaders/moe_matvec_q4_0.wgsl"),
        "moe_matvec_q8_0" => include_str!("wgpu/shaders/moe_matvec_q8_0.wgsl"),
        "moe_matvec_q4k" => include_str!("wgpu/shaders/moe_matvec_q4k.wgsl"),
        "flash_attn" => include_str!("wgpu/shaders/flash_attn.wgsl"),
        "strided_copy" => include_str!("wgpu/shaders/strided_copy.wgsl"),
        "copy" => include_str!("wgpu/shaders/copy.wgsl"),
        "copy2d" => include_str!("wgpu/shaders/copy2d.wgsl"),
        "const_fill" => include_str!("wgpu/shaders/const_fill.wgsl"),
        // elementwise unary
        "exp" => include_str!("wgpu/shaders/exp.wgsl"),
        "log" => include_str!("wgpu/shaders/log.wgsl"),
        "sqrt" => include_str!("wgpu/shaders/sqrt.wgsl"),
        "neg" => include_str!("wgpu/shaders/neg.wgsl"),
        "abs" => include_str!("wgpu/shaders/abs.wgsl"),
        "tanh" => include_str!("wgpu/shaders/tanh.wgsl"),
        "sigmoid" => include_str!("wgpu/shaders/sigmoid.wgsl"),
        "relu" => include_str!("wgpu/shaders/relu.wgsl"),
        "gelu" => include_str!("wgpu/shaders/gelu.wgsl"),
        "gelu_erf" => include_str!("wgpu/shaders/gelu_erf.wgsl"),
        "erf" => include_str!("wgpu/shaders/erf.wgsl"),
        "recip" => include_str!("wgpu/shaders/recip.wgsl"),
        "sign" => include_str!("wgpu/shaders/sign.wgsl"),
        "cos" => include_str!("wgpu/shaders/cos.wgsl"),
        "sin" => include_str!("wgpu/shaders/sin.wgsl"),
        "sqr" => include_str!("wgpu/shaders/sqr.wgsl"),
        "ceil" => include_str!("wgpu/shaders/ceil.wgsl"),
        "floor" => include_str!("wgpu/shaders/floor.wgsl"),
        "round" => include_str!("wgpu/shaders/round.wgsl"),
        "silu" => include_str!("wgpu/shaders/silu.wgsl"),
        "elu" => include_str!("wgpu/shaders/elu.wgsl"),
        "affine" => include_str!("wgpu/shaders/affine.wgsl"),
        "powf" => include_str!("wgpu/shaders/powf.wgsl"),
        // elementwise binary
        "add" => include_str!("wgpu/shaders/add.wgsl"),
        "sub" => include_str!("wgpu/shaders/sub.wgsl"),
        "mul" => include_str!("wgpu/shaders/mul.wgsl"),
        "div" => include_str!("wgpu/shaders/div.wgsl"),
        "maximum" => include_str!("wgpu/shaders/maximum.wgsl"),
        "minimum" => include_str!("wgpu/shaders/minimum.wgsl"),
        "silu_mul" => include_str!("wgpu/shaders/silu_mul.wgsl"),
        "cmp" => include_str!("wgpu/shaders/cmp.wgsl"),
        "where_cond" => include_str!("wgpu/shaders/where_cond.wgsl"),
        // reductions
        "reduce_sum" => include_str!("wgpu/shaders/reduce_sum.wgsl"),
        "reduce_max" => include_str!("wgpu/shaders/reduce_max.wgsl"),
        "reduce_min" => include_str!("wgpu/shaders/reduce_min.wgsl"),
        "reduce_argmin" => include_str!("wgpu/shaders/reduce_argmin.wgsl"),
        "reduce_argmax" => include_str!("wgpu/shaders/reduce_argmax.wgsl"),
        // cast
        "cast_f2u" => include_str!("wgpu/shaders/cast_f2u.wgsl"),
        "cast_u2f" => include_str!("wgpu/shaders/cast_u2f.wgsl"),
        // index ops
        "gather" => include_str!("wgpu/shaders/gather.wgsl"),
        "index_select" => include_str!("wgpu/shaders/index_select.wgsl"),
        "scatter_set" => include_str!("wgpu/shaders/scatter_set.wgsl"),
        "scatter_add_set" => include_str!("wgpu/shaders/scatter_add_set.wgsl"),
        // fused transformer ops
        "softmax_rows" => include_str!("wgpu/shaders/softmax_rows.wgsl"),
        "rms_norm" => include_str!("wgpu/shaders/rms_norm.wgsl"),
        "rope" => include_str!("wgpu/shaders/rope.wgsl"),
        _ => crate::bail!("wgpu: no WGSL kernel for `{name}`"),
    };
    Ok(s)
}

#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for WgpuError {
    fn from(e: String) -> Self {
        WgpuError::Message(e)
    }
}

impl From<WgpuError> for Error {
    fn from(e: WgpuError) -> Self {
        Error::Msg(e.to_string())
    }
}

fn wgpuerr<E: std::fmt::Display>(e: E) -> Error {
    Error::Msg(format!("wgpu: {e}"))
}

struct WgpuInner {
    #[allow(dead_code)] // held to keep the wgpu instance alive for the device's lifetime
    instance: wgpu::Instance,
    #[allow(dead_code)] // retained for adapter info / limits queries
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    gpu_id: usize,
    // Human-readable "<adapter name> [<backend>]" of the selected adapter (for Debug / reports).
    adapter_desc: String,
    // CPU-side RNG seed (kernels are deterministic; randoms are generated on the CPU then uploaded).
    seed: Mutex<u64>,
    // kernel name -> compiled compute pipeline. &'static str keys: kernel names are compile-time
    // literals. Built lazily on first dispatch, then reused (naga compiles the WGSL once). Wrapped in
    // Arc because wgpu 22's ComputePipeline is not Clone; the Arc is the cheap clone for the cache.
    pipelines: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
}

#[derive(Clone)]
pub struct WgpuDevice {
    inner: Arc<WgpuInner>,
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuDevice({}, {})", self.inner.gpu_id, self.inner.adapter_desc)
    }
}

pub struct WgpuStorage {
    // Arc so a storage can be cheaply referenced; the buffer is freed when the last Arc drops
    // (wgpu reclaims device memory on drop — no manual pool needed for M1).
    buffer: Arc<wgpu::Buffer>,
    count: usize,
    dtype: DType,
    device: WgpuDevice,
}

impl std::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuStorage(count={}, dtype={:?})", self.count, self.dtype)
    }
}

impl WgpuStorage {
    fn count(&self) -> usize {
        self.count
    }
}

// std430/std140 scalar-packing helper: tightly pack u32/f32 params into the uniform byte buffer.
fn pack_u32(v: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_ne_bytes());
    }
    b
}

// 1D workgroup width (matches the WGSL @workgroup_size(64) elementwise/matvec kernels).
const WG1D: u32 = 64;

impl WgpuDevice {
    fn dev(&self) -> &wgpu::Device {
        &self.inner.device
    }
    fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    /// The selected adapter as "<name> [<backend>]" (e.g. "NVIDIA GB10 [Vulkan]"). For reports/logs.
    pub fn adapter_description(&self) -> &str {
        &self.inner.adapter_desc
    }

    // Get or compile+cache the compute pipeline for `name`. Uses an auto-derived bind group layout
    // so the WGSL binding declarations (uniform @0, storage @1..) are the single source of truth.
    fn pipeline(&self, name: &'static str) -> Result<Arc<wgpu::ComputePipeline>> {
        if let Some(p) = self.inner.pipelines.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let src = kernel_wgsl(name)?;
        let dev = self.dev();
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = Arc::new(dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }));
        self.inner
            .pipelines
            .lock()
            .unwrap()
            .insert(name, pipeline.clone());
        Ok(pipeline)
    }

    // Dispatch kernel `name` over `bufs` (bound at 1..N) with `params` packed into a uniform buffer
    // (binding 0), and `groups` workgroup counts. One compute pass, recorded + submitted inline.
    // Convention (mirrors the Vulkan backend): the LAST storage buffer is the kernel's output.
    fn dispatch(
        &self,
        name: &'static str,
        bufs: &[&wgpu::Buffer],
        params: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = self.pipeline(name)?;
        let dev = self.dev();
        // Params -> a fresh uniform buffer at binding 0. wgpu requires a uniform binding to be at
        // least one element; pad empty params to 4 bytes. The buffer is small and transient.
        let params_padded: &[u8] = if params.is_empty() { &[0u8; 4] } else { params };
        let ubuf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: params_padded,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Bind group: entry 0 = uniform params, entries 1..N = the storage buffers in order.
        let bgl = pipeline.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::with_capacity(bufs.len() + 1);
        entries.push(wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        });
        for (i, b) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: b.as_entire_binding(),
            });
        }
        let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &bgl,
            entries: &entries,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(name) });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        self.queue().submit(std::iter::once(enc.finish()));
        Ok(())
    }

    // Allocate an uninitialized STORAGE buffer holding `count` 4-byte elements (carries COPY_SRC |
    // COPY_DST so it can be staged to/from for upload + readback).
    fn raw_buffer(&self, count: usize) -> Arc<wgpu::Buffer> {
        let size = ((count.max(1)) * 4) as u64;
        let buffer = self.dev().create_buffer(&wgpu::BufferDescriptor {
            label: Some("storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Arc::new(buffer)
    }

    // Allocate a STORAGE buffer holding `bytes` of `init` data (zero-padded to a u32 multiple). Used
    // by upload paths. Carries COPY_SRC | COPY_DST.
    fn raw_buffer_init(&self, init: &[u8]) -> Arc<wgpu::Buffer> {
        // Pad to a 4-byte multiple (storage buffers and our u32-typed kernels assume word size).
        let nwords = init.len().div_ceil(4).max(1);
        let mut words = vec![0u8; nwords * 4];
        words[..init.len()].copy_from_slice(init);
        let buffer = self.dev().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage_init"),
            contents: &words,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Arc::new(buffer)
    }

    fn alloc_f32(&self, count: usize) -> Result<WgpuStorage> {
        Ok(WgpuStorage {
            buffer: self.raw_buffer(count),
            count,
            dtype: DType::F32,
            device: self.clone(),
        })
    }

    fn alloc_u32(&self, count: usize) -> Result<WgpuStorage> {
        Ok(WgpuStorage {
            buffer: self.raw_buffer(count),
            count,
            dtype: DType::U32,
            device: self.clone(),
        })
    }

    pub(crate) fn upload_f32(&self, data: &[f32]) -> Result<WgpuStorage> {
        let bytes: &[u8] = bytemuck_cast_f32(data);
        Ok(WgpuStorage {
            buffer: self.raw_buffer_init(bytes),
            count: data.len(),
            dtype: DType::F32,
            device: self.clone(),
        })
    }

    fn upload_u32(&self, data: &[u32]) -> Result<WgpuStorage> {
        let bytes: &[u8] = bytemuck_cast_u32(data);
        Ok(WgpuStorage {
            buffer: self.raw_buffer_init(bytes),
            count: data.len(),
            dtype: DType::U32,
            device: self.clone(),
        })
    }

    /// Upload native GGML quantized weight bytes verbatim to a GPU buffer (reused across decode
    /// matvecs). The matvec kernels read the GGML block format straight out of this buffer — no CPU
    /// dequant, no re-pack. `bytes` is the exact `QTensor::data()` slice; it is zero-padded to a u32
    /// multiple for the std430 storage buffer. Mirrors `VulkanDevice::upload_qweight`.
    pub fn upload_qweight(&self, bytes: &[u8]) -> Result<WgpuStorage> {
        let nwords = bytes.len().div_ceil(4).max(1);
        Ok(WgpuStorage {
            buffer: self.raw_buffer_init(bytes),
            count: nwords, // u32 word count
            dtype: DType::U32,
            device: self.clone(),
        })
    }

    /// Native-GGML Q4_0 matvec with both operands on the GPU: `y[nout] = Wq * x[k]`, `Wq` is the raw
    /// Q4_0 block bytes from [`upload_qweight`]. `k` must be a multiple of 32.
    pub fn matvec_q4_0_gpu(
        &self,
        wq: &WgpuStorage,
        x: &WgpuStorage,
        nout: usize,
        k: usize,
    ) -> Result<WgpuStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("matvec_q4_0_gpu: k must be a multiple of 32, got {k}");
        }
        if x.count < k {
            crate::bail!("matvec_q4_0_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        self.dispatch(
            "mul_mat_vec_q4_0",
            &[&wq.buffer, &x.buffer, &out.buffer],
            &pack_u32(&[nout as u32, k as u32]),
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Native-GGML Q8_0 matvec (reads the 34-byte BlockQ8_0 directly). Both operands on the GPU:
    /// `y[nout] = Wq * x[k]`, `Wq` from [`upload_qweight`]. `k` multiple of 32.
    pub fn matvec_q8_0_gpu(
        &self,
        wq: &WgpuStorage,
        x: &WgpuStorage,
        nout: usize,
        k: usize,
    ) -> Result<WgpuStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("matvec_q8_0_gpu: k must be a multiple of 32, got {k}");
        }
        if x.count < k {
            crate::bail!("matvec_q8_0_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        self.dispatch(
            "mul_mat_vec_q8_0",
            &[&wq.buffer, &x.buffer, &out.buffer],
            &pack_u32(&[nout as u32, k as u32]),
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Host-input convenience: uploads `x` then runs [`matvec_q4_0_gpu`]. Returns the f32 result.
    pub fn matvec_q4_0(&self, wq: &WgpuStorage, x: &[f32], nout: usize, k: usize) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q4_0: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q4_0_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Host-input convenience: uploads `x` then runs [`matvec_q8_0_gpu`]. Returns the f32 result.
    pub fn matvec_q8_0(&self, wq: &WgpuStorage, x: &[f32], nout: usize, k: usize) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q8_0: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q8_0_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Native-GGML Q4_K matvec with both operands on the GPU: `y[nout] = Wq * x[k]`, `Wq` is the raw
    /// 144-byte Q4_K super-block bytes from [`upload_qweight`]. `k` must be a multiple of 256.
    pub fn matvec_q4k_gpu(
        &self,
        wq: &WgpuStorage,
        x: &WgpuStorage,
        nout: usize,
        k: usize,
    ) -> Result<WgpuStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("matvec_q4k_gpu: k must be a multiple of 256, got {k}");
        }
        if x.count < k {
            crate::bail!("matvec_q4k_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        self.dispatch(
            "mul_mat_vec_q4k",
            &[&wq.buffer, &x.buffer, &out.buffer],
            &pack_u32(&[nout as u32, k as u32]),
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Host-input convenience: uploads `x` then runs [`matvec_q4k_gpu`]. Returns the f32 result.
    pub fn matvec_q4k(&self, wq: &WgpuStorage, x: &[f32], nout: usize, k: usize) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q4k: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q4k_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Upload a `u32` index/id vector to a GPU buffer (e.g. the MoE router's per-slot expert ids).
    pub fn upload_ids(&self, ids: &[u32]) -> Result<WgpuStorage> {
        self.upload_u32(ids)
    }

    /// Fused MoE grouped quant matvec on the GPU: for each routed slot `s` and output row `r`,
    /// `y[s, r] = sum_k W[ids[s], r, k] * x[s, k]`, reading the per-expert slice from a single GGML
    /// weight bank `[E, n, k]` already uploaded via [`upload_qweight`]. The router gather and the
    /// per-expert GEMM run in one dispatch. `kernel` selects the dtype variant ("moe_matvec_q4_0" /
    /// "moe_matvec_q8_0" / "moe_matvec_q4k"). Mirrors `VulkanDevice::moe_matvec_gpu`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_matvec_gpu(
        &self,
        kernel: &'static str,
        wbank: &WgpuStorage,
        x: &WgpuStorage,
        ids: &WgpuStorage,
        nrows: usize,
        n: usize,
        k: usize,
    ) -> Result<WgpuStorage> {
        if x.count < nrows * k {
            crate::bail!("moe_matvec_gpu: x count {} < nrows*k {}", x.count, nrows * k);
        }
        if ids.count < nrows {
            crate::bail!("moe_matvec_gpu: ids count {} < nrows {nrows}", ids.count);
        }
        let total = nrows * n;
        let out = self.alloc_f32(total)?;
        self.dispatch(
            kernel,
            &[&wbank.buffer, &x.buffer, &ids.buffer, &out.buffer],
            &pack_u32(&[n as u32, k as u32, nrows as u32]),
            ((total as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Fused scaled-dot-product (flash) attention on the GPU, online-softmax. Q is `[BH, Lq, D]`,
    /// K/V are `[BH, Lk, D]` (contiguous f32 device buffers, head_dim D <= 256); output is
    /// `[BH, Lq, D]`. `scale` multiplies the QK^T scores; `causal` applies aligned causal masking.
    /// Mirrors `VulkanDevice::flash_attn_gpu`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_gpu(
        &self,
        q: &WgpuStorage,
        k: &WgpuStorage,
        v: &WgpuStorage,
        bh: usize,
        lq: usize,
        lk: usize,
        d: usize,
        scale: f32,
        causal: bool,
    ) -> Result<WgpuStorage> {
        if d > 256 {
            crate::bail!("flash_attn_gpu: head_dim {d} > 256 (kernel limit)");
        }
        if q.count < bh * lq * d {
            crate::bail!("flash_attn_gpu: q count {} < bh*lq*d {}", q.count, bh * lq * d);
        }
        if k.count < bh * lk * d || v.count < bh * lk * d {
            crate::bail!("flash_attn_gpu: k/v count too small for bh*lk*d {}", bh * lk * d);
        }
        let out = self.alloc_f32(bh * lq * d)?;
        // Uniform block matches flash_attn.wgsl: {u32 bh, lq, lk, d; f32 scale; u32 causal}.
        let mut params = pack_u32(&[bh as u32, lq as u32, lk as u32, d as u32]);
        params.extend_from_slice(&scale.to_ne_bytes());
        params.extend_from_slice(&(causal as u32).to_ne_bytes());
        let total = bh * lq;
        self.dispatch(
            "flash_attn",
            &[&q.buffer, &k.buffer, &v.buffer, &out.buffer],
            &params,
            ((total as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Host-input convenience for [`flash_attn_gpu`]: uploads Q/K/V and returns the `[BH*Lq*D]`
    /// output as an f32 vector. For tests/standalone use.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        bh: usize,
        lq: usize,
        lk: usize,
        d: usize,
        scale: f32,
        causal: bool,
    ) -> Result<Vec<f32>> {
        let qs = self.upload_f32(q)?;
        let ks = self.upload_f32(k)?;
        let vs = self.upload_f32(v)?;
        self.flash_attn_gpu(&qs, &ks, &vs, bh, lq, lk, d, scale, causal)?
            .to_vec_f32()
    }

    // Read `count` 4-byte words out of `src` to the host: copy to a MAP_READ staging buffer,
    // map_async, then block on device.poll(Wait). This is the one new mechanism vs the Vulkan
    // backend's direct host map. Returns the raw bytes (caller reinterprets as f32/u32).
    fn read_bytes(&self, src: &wgpu::Buffer, count: usize) -> Result<Vec<u8>> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = (count * 4) as u64;
        let dev = self.dev();
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc =
            dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        self.queue().submit(std::iter::once(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Drive the GPU to completion so the map callback fires; Maintain::Wait blocks until idle.
        let _ = self.dev().poll(wgpu::Maintain::Wait);
        match rx.recv() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(wgpuerr(e)),
            Err(e) => return Err(wgpuerr(e)),
        }
        let data = slice.get_mapped_range();
        let out = data.to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

impl WgpuStorage {
    // Download the whole buffer as f32 (used internally + by to_cpu_storage).
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        let bytes = self.device.read_bytes(&self.buffer, self.count)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    // Download the whole buffer as u32 (used by to_cpu_storage for U32 storage).
    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        let bytes = self.device.read_bytes(&self.buffer, self.count)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn groups_1d(n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(WG1D), 1, 1)
    }

    // Materialize any (strided/broadcast) f32 layout into a fresh contiguous buffer of
    // `layout.elem_count()` elements via the strided_copy kernel. Identity for already-contiguous
    // offset-0 layouts. rank <= 6. Mirrors VulkanStorage::contiguous.
    fn contiguous(&self, layout: &Layout) -> Result<WgpuStorage> {
        if layout.is_contiguous() && layout.start_offset() == 0 {
            // Already packed; a plain word-for-word copy keeps the contract (a fresh buffer).
            let n = layout.shape().elem_count();
            let out = self.device.alloc_f32(n)?;
            self.device.dispatch(
                "copy",
                &[&self.buffer, &out.buffer],
                &pack_u32(&[n as u32]),
                Self::groups_1d(n),
            )?;
            return Ok(out);
        }
        let dims = layout.dims();
        let rank = dims.len();
        if rank > 6 {
            crate::bail!("wgpu: contiguous supports rank <= 6, got {rank}");
        }
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let strides = layout.stride();
        // Uniform layout MUST match strided_copy.wgsl exactly: {n, rank, offset, dst_offset,
        // shape[0..6], strides[0..6]} as 16 tightly-packed u32 (the WGSL uses scalar fields, not a
        // WGSL array, precisely so this packing is byte-identical).
        let mut p = vec![n as u32, rank as u32, layout.start_offset() as u32, 0u32];
        let mut shape6 = [0u32; 6];
        let mut stride6 = [0u32; 6];
        for d in 0..rank {
            shape6[d] = dims[d] as u32;
            stride6[d] = strides[d] as u32;
        }
        p.extend_from_slice(&shape6);
        p.extend_from_slice(&stride6);
        self.device.dispatch(
            "strided_copy",
            &[&self.buffer, &out.buffer],
            &pack_u32(&p),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    // Materialize a u32 storage (ids / where_cond mask) into a fresh contiguous, offset-0 buffer.
    // Done on the CPU via the layout's strided index so it's bit-exact for arbitrary u32 values
    // (reusing the float strided_copy would reinterpret small integers as denormal floats). These
    // tensors (token/position ids, masks) are tiny so the round-trip is cheap. Mirrors Vulkan.
    fn contiguous_u32(&self, layout: &Layout) -> Result<WgpuStorage> {
        debug_assert_eq!(self.dtype, DType::U32);
        if layout.is_contiguous() && layout.start_offset() == 0 {
            return self.device.upload_u32(&self.to_vec_u32()?);
        }
        let src = self.to_vec_u32()?;
        let gathered: Vec<u32> = layout.strided_index().map(|i| src[i]).collect();
        self.device.upload_u32(&gathered)
    }

    // Buffer for a contiguous, offset-0 f32 view of `layout`: no copy when the storage already is
    // one (returns its own buffer), otherwise materializes a packed copy returned via `keep` (held
    // alive by the caller until the dispatch is recorded). Mirrors VulkanStorage::contig_buf.
    fn contig_buf(&self, layout: &Layout, keep: &mut Option<WgpuStorage>) -> Result<Arc<wgpu::Buffer>> {
        if layout.is_contiguous() && layout.start_offset() == 0 {
            Ok(self.buffer.clone())
        } else {
            let s = self.contiguous(layout)?;
            let b = s.buffer.clone();
            *keep = Some(s);
            Ok(b)
        }
    }

    // --- native fused ops (mirror VulkanStorage), used by storage.rs dispatch wrappers ---

    /// softmax over the last dim. `self` is the input; `layout` its layout. One row per thread.
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<WgpuStorage> {
        let mut xk = None;
        let xb = self.contig_buf(layout, &mut xk)?;
        let dims = layout.dims();
        let m = *dims.last().unwrap_or(&1);
        let nrows = layout.shape().elem_count() / m.max(1);
        let out = self.device.alloc_f32(nrows * m)?;
        self.device.dispatch(
            "softmax_rows",
            &[&xb, &out.buffer],
            &pack_u32(&[nrows as u32, m as u32]),
            Self::groups_1d(nrows),
        )?;
        Ok(out)
    }

    /// rms-norm over the last dim. `self`=x, `alpha`=scale [m].
    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &WgpuStorage,
        alpha_l: &Layout,
        eps: f32,
    ) -> Result<WgpuStorage> {
        let mut xk = None;
        let mut ak = None;
        let xb = self.contig_buf(layout, &mut xk)?;
        let ab = alpha.contig_buf(alpha_l, &mut ak)?;
        let dims = layout.dims();
        let m = *dims.last().unwrap_or(&1);
        let nrows = layout.shape().elem_count() / m.max(1);
        let out = self.device.alloc_f32(nrows * m)?;
        let mut params = pack_u32(&[nrows as u32, m as u32]);
        params.extend_from_slice(&eps.to_ne_bytes());
        self.device.dispatch(
            "rms_norm",
            &[&xb, &ab, &out.buffer],
            &params,
            Self::groups_1d(nrows),
        )?;
        Ok(out)
    }

    /// Fused SwiGLU: out = silu(self) * rhs, elementwise. One dispatch instead of silu + mul.
    pub fn silu_mul(
        &self,
        layout: &Layout,
        rhs: &WgpuStorage,
        rhs_l: &Layout,
    ) -> Result<WgpuStorage> {
        let mut ak = None;
        let mut bk = None;
        let ab = self.contig_buf(layout, &mut ak)?;
        let bb = rhs.contig_buf(rhs_l, &mut bk)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device.dispatch(
            "silu_mul",
            &[&ab, &bb, &out.buffer],
            &pack_u32(&[n as u32]),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    /// GPT-NeoX rotary embedding. `self`=src [b,h,t,d], cos/sin [t,d/2] or [b,t,d/2].
    pub fn rope(
        &self,
        layout: &Layout,
        cos: &WgpuStorage,
        cos_l: &Layout,
        sin: &WgpuStorage,
        sin_l: &Layout,
    ) -> Result<WgpuStorage> {
        let mut srck = None;
        let mut ck = None;
        let mut sk = None;
        let srcb = self.contig_buf(layout, &mut srck)?;
        let cb = cos.contig_buf(cos_l, &mut ck)?;
        let sb = sin.contig_buf(sin_l, &mut sk)?;
        let (b, h, t, d) = layout.shape().dims4()?;
        let unbatched = (cos_l.dims().len() == 3 && sin_l.dims().len() == 3) as u32;
        let out = self.device.alloc_f32(b * h * t * d)?;
        let pairs = b * h * t * (d / 2);
        self.device.dispatch(
            "rope",
            &[&srcb, &cb, &sb, &out.buffer],
            &pack_u32(&[b as u32, h as u32, t as u32, d as u32, unbatched]),
            Self::groups_1d(pairs),
        )?;
        Ok(out)
    }

    // Shared scatter: write/accumulate src into dst (self) along `dim` at positions `ids`.
    // dst is assumed contiguous (its layout `l` gives the dim sizes). ids/src share a shape.
    #[allow(clippy::too_many_arguments)]
    fn scatter_impl(
        &self,
        kernel: &'static str,
        l: &Layout,
        ids: &WgpuStorage,
        ids_l: &Layout,
        src: &WgpuStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if ids.dtype != DType::U32 {
            crate::bail!("wgpu: scatter requires u32 ids, got {:?}", ids.dtype);
        }
        let idc = ids.contiguous_u32(ids_l)?;
        let srcc = src.contiguous(src_l)?;
        let src_dims = src_l.dims();
        let dst_dims = l.dims();
        let right: usize = src_dims[dim + 1..].iter().product();
        let dim_src = src_dims[dim];
        let dim_dst = dst_dims[dim];
        let n = src_l.shape().elem_count();
        self.device.dispatch(
            kernel,
            &[&self.buffer, &srcc.buffer, &idc.buffer],
            &pack_u32(&[n as u32, right as u32, dim_src as u32, dim_dst as u32]),
            Self::groups_1d(n),
        )?;
        Ok(())
    }
}

// Reinterpret an f32/u32 slice as bytes. No external bytemuck dep; transmute the slice view.
fn bytemuck_cast_f32(data: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding/invalid bit patterns for raw byte reinterpretation; len*4 bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}
fn bytemuck_cast_u32(data: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding; len*4 bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(ordinal: usize) -> Result<Self> {
        // Instance restricted to Vulkan: on the GB10 box that is the NVIDIA driver (the spark target).
        // GL is excluded so we never silently land on a software rasterizer through the GL backend.
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        // Enumerate Vulkan adapters and pick the `ordinal`-th NON-CPU one (skip llvmpipe / software),
        // mirroring the raw-Vulkan backend's adapter filter. The GB10 enumerates as an integrated
        // GPU; llvmpipe enumerates as Cpu.
        let mut gpus: Vec<wgpu::Adapter> = Vec::new();
        for ad in instance.enumerate_adapters(wgpu::Backends::VULKAN) {
            let info = ad.get_info();
            let is_cpu = info.device_type == wgpu::DeviceType::Cpu
                || info.name.to_lowercase().contains("llvmpipe");
            if !is_cpu {
                gpus.push(ad);
            }
        }
        if gpus.is_empty() {
            return Err(Error::Msg("wgpu: no non-CPU Vulkan adapter".into()));
        }
        // Prefer a discrete/integrated GPU at the requested ordinal; the GB10 is integrated.
        let adapter = if ordinal < gpus.len() {
            gpus.swap_remove(ordinal)
        } else {
            return Err(Error::Msg(format!("wgpu: no adapter at ordinal {ordinal}")));
        };
        let info = adapter.get_info();
        let adapter_desc = format!("{} [{:?}]", info.name, info.backend);

        // Request the adapter's full limits so large buffers (weights, big GEMMs) are not clipped by
        // wgpu's conservative defaults (128 MiB storage binding / 256 MiB buffer).
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("hanzo-ml-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(wgpuerr)?;

        let inner = WgpuInner {
            instance,
            adapter,
            device,
            queue,
            gpu_id: ordinal,
            adapter_desc,
            seed: Mutex::new(299792458),
            pipelines: Mutex::new(HashMap::new()),
        };
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu {
            gpu_id: self.inner.gpu_id,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        // Invariant (same as Vulkan): wgpu computes in f32; f16/bf16 are represented as f32 and u8
        // as u32. zeros(bf16) yields an f32 buffer of zeros (zero is identical in any of these reps).
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => self.upload_f32(&vec![0f32; count]),
            DType::U32 | DType::U8 => self.upload_u32(&vec![0u32; count]),
            _ => crate::bail!("wgpu: only f32/u32/f16/bf16/u8 supported, got {dtype:?}"),
        }
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => self.alloc_f32(shape.elem_count()),
            DType::U32 | DType::U8 => self.alloc_u32(shape.elem_count()),
            _ => crate::bail!("wgpu: only f32/u32/f16/bf16/u8 supported, got {dtype:?}"),
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        // wgpu computes in f32; f16/bf16 are upcast on upload so real fp16/bf16 tensors load.
        match s {
            CpuStorage::F32(v) => self.upload_f32(v),
            CpuStorage::U32(v) => self.upload_u32(v),
            CpuStorage::F16(v) => self.upload_f32(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>()),
            CpuStorage::BF16(v) => self.upload_f32(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>()),
            CpuStorage::U8(v) => self.upload_u32(&v.iter().map(|&x| x as u32).collect::<Vec<_>>()),
            _ => crate::bail!("wgpu: only f32/u32/f16/bf16/u8 supported, got {:?}", s.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&s)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            crate::bail!("wgpu: rand_uniform only f32, got {dtype:?}");
        }
        use rand::Rng;
        let mut rng = rand::rng();
        let n = shape.elem_count();
        let uniform = rand::distr::Uniform::new(min as f32, max as f32).map_err(Error::wrap)?;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(rng.sample::<f32, _>(uniform));
        }
        self.upload_f32(&data)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            crate::bail!("wgpu: rand_normal only f32, got {dtype:?}");
        }
        use rand_distr::Distribution;
        let mut rng = rand::rng();
        let n = shape.elem_count();
        let normal = rand_distr::Normal::new(mean as f32, std as f32).map_err(Error::wrap)?;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(normal.sample(&mut rng));
        }
        self.upload_f32(&data)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.inner.seed.lock().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.inner.seed.lock().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        let _ = self.dev().poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

// Bail with a uniform "TODO" message for the non-milestone BackendStorage methods.
macro_rules! todo_wgpu {
    ($what:literal) => {
        crate::bail!(concat!("wgpu: TODO ", $what))
    };
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Raw device-memory copy of the whole buffer (ignores layout; assumes contiguous). Buffers
        // are just bytes (4 bytes/elem); per the upload invariant f16/bf16 live as f32 and u8 as u32.
        match self.dtype {
            DType::U32 | DType::U8 => self.device.upload_u32(&self.to_vec_u32()?),
            _ => self.device.upload_f32(&self.to_vec_f32()?),
        }
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U32 => Ok(CpuStorage::U32(self.to_vec_u32()?)),
            _ => Ok(CpuStorage::F32(self.to_vec_f32()?)),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut params = (n as u32).to_ne_bytes().to_vec();
        params.extend_from_slice(&(mul as f32).to_ne_bytes());
        params.extend_from_slice(&(add as f32).to_ne_bytes());
        self.device
            .dispatch("affine", &[&cb, &out.buffer], &params, Self::groups_1d(n))?;
        Ok(out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut params = (n as u32).to_ne_bytes().to_vec();
        params.extend_from_slice(&(e as f32).to_ne_bytes());
        self.device
            .dispatch("powf", &[&cb, &out.buffer], &params, Self::groups_1d(n))?;
        Ok(out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut params = (n as u32).to_ne_bytes().to_vec();
        params.extend_from_slice(&(alpha as f32).to_ne_bytes());
        self.device
            .dispatch("elu", &[&cb, &out.buffer], &params, Self::groups_1d(n))?;
        Ok(out)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let (kernel, is_arg) = match op {
            ReduceOp::Sum => ("reduce_sum", false),
            ReduceOp::Max => ("reduce_max", false),
            ReduceOp::Min => ("reduce_min", false),
            ReduceOp::ArgMin => ("reduce_argmin", true),
            ReduceOp::ArgMax => ("reduce_argmax", true),
        };
        let dims = layout.dims();
        let rank = dims.len();
        if rank == 0 {
            crate::bail!("wgpu: reduce_op on scalar not supported");
        }
        if sum_dims != [rank - 1] {
            crate::bail!(
                "wgpu: reduce_op only supports the last dim (got dims={sum_dims:?}, rank={rank})"
            );
        }
        let c = self.contiguous(layout)?;
        let cols = dims[rank - 1];
        let rows: usize = dims[..rank - 1].iter().product();
        let out = if is_arg {
            self.device.alloc_u32(rows)?
        } else {
            self.device.alloc_f32(rows)?
        };
        self.device.dispatch(
            kernel,
            &[&c.buffer, &out.buffer],
            &pack_u32(&[rows as u32, cols as u32]),
            Self::groups_1d(rows),
        )?;
        Ok(out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        // Elementwise compare on f32 operands -> u32 {0,1} mask.
        let lc = self.contiguous(lhs_l)?;
        let rc = rhs.contiguous(rhs_l)?;
        let n = lhs_l.shape().elem_count();
        let out = self.device.alloc_u32(n)?;
        let code: u32 = match op {
            CmpOp::Eq => 0,
            CmpOp::Ne => 1,
            CmpOp::Le => 2,
            CmpOp::Ge => 3,
            CmpOp::Lt => 4,
            CmpOp::Gt => 5,
        };
        self.device.dispatch(
            "cmp",
            &[&lc.buffer, &rc.buffer, &out.buffer],
            &pack_u32(&[n as u32, code]),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let n = layout.shape().elem_count();
        match (self.dtype, dtype) {
            // Same dtype: just contiguous-ize (respects layout/offset).
            (DType::F32, DType::F32) | (DType::U32, DType::U32) => self.contiguous(layout),
            (DType::F32, DType::U32) => {
                let c = self.contiguous(layout)?;
                let out = self.device.alloc_u32(n)?;
                self.device.dispatch(
                    "cast_f2u",
                    &[&c.buffer, &out.buffer],
                    &pack_u32(&[n as u32]),
                    Self::groups_1d(n),
                )?;
                Ok(out)
            }
            (DType::U32, DType::F32) => {
                let c = self.contiguous(layout)?;
                let out = self.device.alloc_f32(n)?;
                self.device.dispatch(
                    "cast_u2f",
                    &[&c.buffer, &out.buffer],
                    &pack_u32(&[n as u32]),
                    Self::groups_1d(n),
                )?;
                Ok(out)
            }
            // Other dtypes aren't held by this backend: CPU-convert then upload.
            _ => {
                let cpu = self.to_cpu_storage()?;
                let converted = crate::backend::BackendStorage::to_dtype(&cpu, layout, dtype)?;
                self.device.storage_from_cpu_storage(&converted)
            }
        }
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let kernel: &'static str = match B::NAME {
            "silu" => "silu",
            "gelu" => "gelu",
            "relu" => "relu",
            "exp" => "exp",
            "neg" => "neg",
            "sqr" => "sqr",
            "sqrt" => "sqrt",
            "recip" => "recip",
            "tanh" => "tanh",
            "sin" => "sin",
            "cos" => "cos",
            "log" => "log",
            "abs" => "abs",
            "floor" => "floor",
            "ceil" => "ceil",
            "round" => "round",
            "sign" => "sign",
            "erf" => "erf",
            "gelu_erf" => "gelu_erf",
            "sigmoid" => "sigmoid",
            // Anything still without a WGSL kernel: fall back to CPU (correct, slow).
            _ => {
                let cpu = self.to_cpu_storage()?;
                let r = crate::backend::BackendStorage::unary_impl::<B>(&cpu, layout)?;
                return self.device.storage_from_cpu_storage(&r);
            }
        };
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device
            .dispatch(kernel, &[&cb, &out.buffer], &pack_u32(&[n as u32]), Self::groups_1d(n))?;
        Ok(out)
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let kernel: &'static str = match B::NAME {
            "add" => "add",
            "sub" => "sub",
            "mul" => "mul",
            "div" => "div",
            "maximum" => "maximum",
            "minimum" => "minimum",
            // Anything still without a WGSL kernel: fall back to CPU (correct, slow).
            _ => {
                let lc = self.to_cpu_storage()?;
                let rc = rhs.to_cpu_storage()?;
                let r = crate::backend::BackendStorage::binary_impl::<B>(&lc, &rc, lhs_l, rhs_l)?;
                return self.device.storage_from_cpu_storage(&r);
            }
        };
        // hanzo-ml pre-broadcasts both layouts to the output shape; broadcast layouts aren't
        // contiguous so contig_buf still materializes them, but same-shape contiguous operands skip.
        let mut lk = None;
        let mut rk = None;
        let lb = self.contig_buf(lhs_l, &mut lk)?;
        let rb = rhs.contig_buf(rhs_l, &mut rk)?;
        let n = lhs_l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device
            .dispatch(kernel, &[&lb, &rb, &out.buffer], &pack_u32(&[n as u32]), Self::groups_1d(n))?;
        Ok(out)
    }

    fn where_cond(
        &self,
        l: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        // self is the condition mask (u32). The kernel reads it linearly.
        if self.dtype != DType::U32 {
            crate::bail!("wgpu: where_cond requires u32 condition, got {:?}", self.dtype);
        }
        let mut condk = None;
        let condb = if l.is_contiguous() && l.start_offset() == 0 {
            self.buffer.clone()
        } else {
            let s = self.contiguous_u32(l)?;
            let b = s.buffer.clone();
            condk = Some(s);
            b
        };
        let _ = &condk; // keep the materialized condition alive until the dispatch is recorded
        let mut tk = None;
        let mut fk = None;
        let tb = t.contig_buf(t_l, &mut tk)?;
        let fb = f.contig_buf(f_l, &mut fk)?;
        let n = l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device.dispatch(
            "where_cond",
            &[&condb, &tb, &fb, &out.buffer],
            &pack_u32(&[n as u32]),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        todo_wgpu!("conv1d")
    }

    fn conv_transpose1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        todo_wgpu!("conv_transpose1d")
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        todo_wgpu!("conv2d")
    }

    fn conv_transpose2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        todo_wgpu!("conv_transpose2d")
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if ids.dtype != DType::U32 {
            crate::bail!("wgpu: index_select requires u32 ids, got {:?}", ids.dtype);
        }
        // The kernel reads ids linearly (ids[i]), so materialize a contiguous, offset-0 copy.
        let mut idk = None;
        let ids_buf = if ids_l.is_contiguous() && ids_l.start_offset() == 0 {
            ids.buffer.clone()
        } else {
            let s = ids.contiguous_u32(ids_l)?;
            let b = s.buffer.clone();
            idk = Some(s);
            b
        };
        let _ = &idk; // keep the materialized ids alive until the dispatch is recorded
        let src_c = self.contiguous(l)?;
        let dims = l.dims();
        let left: usize = dims[..dim].iter().product();
        let dim_size = dims[dim];
        let right: usize = dims[dim + 1..].iter().product();
        let n_ids = ids_l.shape().elem_count();
        let total = left * n_ids * right;
        let out = self.device.alloc_f32(total)?;
        self.device.dispatch(
            "index_select",
            &[&ids_buf, &src_c.buffer, &out.buffer],
            &pack_u32(&[left as u32, dim_size as u32, right as u32, n_ids as u32]),
            Self::groups_1d(total),
        )?;
        Ok(out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if ids.dtype != DType::U32 {
            crate::bail!("wgpu: gather requires u32 ids, got {:?}", ids.dtype);
        }
        let src = self.contiguous(l)?;
        let idc = ids.contiguous_u32(ids_l)?;
        let out_dims = ids_l.dims();
        let src_dims = l.dims();
        let right: usize = out_dims[dim + 1..].iter().product();
        let dim_out = out_dims[dim];
        let dim_src = src_dims[dim];
        let n = ids_l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device.dispatch(
            "gather",
            &[&src.buffer, &idc.buffer, &out.buffer],
            &pack_u32(&[n as u32, right as u32, dim_out as u32, dim_src as u32]),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_impl("scatter_set", l, ids, ids_l, src, src_l, dim)
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.scatter_impl("scatter_add_set", l, ids, ids_l, src, src_l, dim)
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        todo_wgpu!("index_add")
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // lhs packed [b,m,k]: use directly if contiguous from offset 0, else materialize.
        let mut lkeep = None;
        let lc_buf = if lhs_l.is_contiguous() && lhs_l.start_offset() == 0 {
            &self.buffer
        } else {
            let s = self.contiguous(lhs_l)?;
            lkeep = Some(s);
            &lkeep.as_ref().unwrap().buffer
        };
        // rhs is [b,k,n]. A Linear passes W.t() where W is contiguous [b,n,k]; detect that
        // transposed-contiguous layout and feed W's natural [n,k] buffer to the NT kernel, skipping
        // the transpose materialization. Otherwise use the buffer directly or materialize [b,k,n].
        let d = rhs_l.dims();
        let st = rhs_l.stride();
        let nt = rhs_l.start_offset() == 0
            && ((d.len() == 2 && d[0] == k && d[1] == n && st[0] == 1 && st[1] == k)
                || (d.len() == 3
                    && d[0] == b
                    && d[1] == k
                    && d[2] == n
                    && st[0] == n * k
                    && st[1] == 1
                    && st[2] == k));
        let mut rkeep = None;
        let rc_buf = if nt || (rhs_l.is_contiguous() && rhs_l.start_offset() == 0) {
            &rhs.buffer
        } else {
            let s = rhs.contiguous(rhs_l)?;
            rkeep = Some(s);
            &rkeep.as_ref().unwrap().buffer
        };
        let _ = (&lkeep, &rkeep); // keep any materialized copies alive until dispatch is recorded
        let out = self.device.alloc_f32(b * m * n)?;
        // Push order matches matmul.wgsl / matmul_nt.wgsl: {batch, m, k, n}.
        let params = pack_u32(&[b as u32, m as u32, k as u32, n as u32]);
        let kernel = if nt { "matmul_nt" } else { "matmul" };
        let groups = ((n as u32).div_ceil(16), (m as u32).div_ceil(16), b as u32);
        self.device
            .dispatch(kernel, &[lc_buf, rc_buf, &out.buffer], &params, groups)?;
        Ok(out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let n = src_l.shape().elem_count();
        if n == 0 {
            return Ok(());
        }
        if dst.count() < dst_offset + n {
            crate::bail!(
                "wgpu: copy_strided_src dst too small ({} < {})",
                dst.count(),
                dst_offset + n
            );
        }
        let dims = src_l.dims();
        let rank = dims.len();
        if rank > 6 {
            crate::bail!("wgpu: copy_strided_src supports rank <= 6, got {rank}");
        }
        let strides = src_l.stride();
        let mut p = vec![n as u32, rank as u32, src_l.start_offset() as u32, dst_offset as u32];
        let mut shape6 = [0u32; 6];
        let mut stride6 = [0u32; 6];
        for d in 0..rank {
            shape6[d] = dims[d] as u32;
            stride6[d] = strides[d] as u32;
        }
        p.extend_from_slice(&shape6);
        p.extend_from_slice(&stride6);
        self.device.dispatch(
            "strided_copy",
            &[&self.buffer, &dst.buffer],
            &pack_u32(&p),
            Self::groups_1d(n),
        )
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let total = d1 * d2;
        if total == 0 {
            return Ok(());
        }
        let p = pack_u32(&[
            d1 as u32,
            d2 as u32,
            src_stride1 as u32,
            dst_stride1 as u32,
            src_offset as u32,
            dst_offset as u32,
        ]);
        self.device
            .dispatch("copy2d", &[&self.buffer, &dst.buffer], &p, Self::groups_1d(total))
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        // Fast path: a contiguous, offset-0 view covering the WHOLE buffer is filled on-GPU by
        // const_fill (raw 32-bit bits so one uint kernel serves f32 and u32 storage bit-exactly).
        let n = layout.shape().elem_count();
        if layout.is_contiguous() && layout.start_offset() == 0 && n == self.count {
            let bits: u32 = if self.dtype == DType::U32 || self.dtype == DType::U8 {
                s.to_f64() as u32
            } else {
                (s.to_f64() as f32).to_bits()
            };
            return self.device.dispatch(
                "const_fill",
                &[&self.buffer],
                &pack_u32(&[n as u32, bits]),
                Self::groups_1d(n),
            );
        }
        // Slow path (partial / strided / offset view): host read-modify-write so elements outside
        // the addressed set are preserved.
        if self.dtype == DType::U32 || self.dtype == DType::U8 {
            let v = s.to_f64() as u32;
            let mut data = self.to_vec_u32()?;
            for i in layout.strided_index() {
                if i >= data.len() {
                    crate::bail!("wgpu: const_set out of range");
                }
                data[i] = v;
            }
            *self = self.device.upload_u32(&data)?;
        } else {
            let v = s.to_f64() as f32;
            let mut data = self.to_vec_f32()?;
            for i in layout.strided_index() {
                if i >= data.len() {
                    crate::bail!("wgpu: const_set out of range");
                }
                data[i] = v;
            }
            *self = self.device.upload_f32(&data)?;
        }
        Ok(())
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo_wgpu!("avg_pool2d")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        todo_wgpu!("max_pool2d")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        todo_wgpu!("upsample_nearest1d")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        todo_wgpu!("upsample_nearest2d")
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        todo_wgpu!("upsample_bilinear2d")
    }
}
