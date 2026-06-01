//! Native Vulkan compute backend via `wgpu` (WebGPU -> Vulkan/DX12/Metal).
//!
//! Targets cross-vendor GPUs where CUDA/Metal are unavailable — notably the
//! AMD Strix Halo (Radeon 8060S, gfx1151) iGPU. IMPORTANT: to actually reach
//! the GPU this crate must be built **natively on Windows** (AMD Vulkan driver)
//! or on Linux with Mesa RADV. Under WSL2, wgpu only sees `llvmpipe` (CPU) —
//! see memory note `project_strix_halo_gpu_paths`.
//!
//! Design: correctness-first. `VulkanStorage` owns a `wgpu::Buffer`; buffer
//! lifecycle (alloc, upload, readback) runs on the GPU. Compute currently falls
//! back through `CpuStorage` for the long tail, with real WGSL kernels added for
//! hot ops (starting with `affine`). Each WGSL kernel that lands removes one
//! host round-trip.
#![allow(dead_code)]

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

fn vk_err(e: impl std::fmt::Display) -> Error {
    Error::Vulkan(format!("vulkan: {e}").into())
}

/// A handle to a wgpu device + queue. Cheap to clone (Arc inside).
#[derive(Clone)]
pub struct VulkanDevice {
    gpu_id: usize,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_name: Arc<String>,
    seed: Arc<Mutex<u64>>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "VulkanDevice({}, {})", self.gpu_id, self.adapter_name)
    }
}

impl VulkanDevice {
    fn bytes_per_element(dtype: DType) -> usize {
        dtype.size_in_bytes()
    }

    /// Upload raw bytes into a new STORAGE buffer (usable as compute in/out + copy src).
    fn buffer_from_bytes(&self, bytes: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("candle-vk-storage"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn empty_buffer(&self, size_bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle-vk-storage"),
            size: size_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Read a buffer's bytes back to host (blocking).
    fn read_buffer(&self, buffer: &wgpu::Buffer, size_bytes: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candle-vk-readback"),
            size: size_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, size_bytes.max(4));
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(vk_err)?
            .map_err(vk_err)?;
        let data = slice.get_mapped_range();
        let out = data[..size_bytes as usize].to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }
}

/// GPU storage: a wgpu buffer plus its logical dtype and element count.
pub struct VulkanStorage {
    buffer: Arc<wgpu::Buffer>,
    device: VulkanDevice,
    dtype: DType,
    count: usize,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "VulkanStorage({:?}, {} elems)", self.dtype, self.count)
    }
}

// ---- CpuStorage <-> raw bytes helpers ----
fn cpu_storage_to_bytes(s: &CpuStorage) -> Vec<u8> {
    match s {
        CpuStorage::U8(v) => bytemuck::cast_slice(v).to_vec(),
        CpuStorage::U32(v) => bytemuck::cast_slice(v).to_vec(),
        CpuStorage::I64(v) => bytemuck::cast_slice(v).to_vec(),
        CpuStorage::BF16(v) => bytemuck::cast_slice(v).to_vec(),
        CpuStorage::F16(v) => bytemuck::cast_slice(v).to_vec(),
        CpuStorage::F32(v) => bytemuck::cast_slice(v).to_vec(),
        CpuStorage::F64(v) => bytemuck::cast_slice(v).to_vec(),
        // float8 / other exotic dtypes: fall back to a byte reinterpret via f32 path
        other => {
            // Best-effort: treat as raw little-endian bytes of its element type.
            // candle's CpuStorage for F8E4M3 etc. is Vec<u8>-sized; handle generically.
            let _ = other;
            Vec::new()
        }
    }
}

fn bytes_to_cpu_storage(dtype: DType, bytes: &[u8]) -> Result<CpuStorage> {
    use half::{bf16, f16};
    let s = match dtype {
        DType::U8 => CpuStorage::U8(bytes.to_vec()),
        DType::U32 => CpuStorage::U32(bytemuck::cast_slice::<u8, u32>(bytes).to_vec()),
        DType::I64 => CpuStorage::I64(bytemuck::cast_slice::<u8, i64>(bytes).to_vec()),
        DType::BF16 => CpuStorage::BF16(bytemuck::cast_slice::<u8, bf16>(bytes).to_vec()),
        DType::F16 => CpuStorage::F16(bytemuck::cast_slice::<u8, f16>(bytes).to_vec()),
        DType::F32 => CpuStorage::F32(bytemuck::cast_slice::<u8, f32>(bytes).to_vec()),
        DType::F64 => CpuStorage::F64(bytemuck::cast_slice::<u8, f64>(bytes).to_vec()),
        other => return Err(vk_err(format!("unsupported dtype for vulkan readback: {other:?}"))),
    };
    Ok(s)
}

impl VulkanStorage {
    /// Read this GPU buffer back into a CpuStorage (host round-trip).
    fn cpu(&self) -> Result<CpuStorage> {
        let size = (self.count * self.dtype.size_in_bytes()) as u64;
        let bytes = self.device.read_buffer(&self.buffer, size)?;
        bytes_to_cpu_storage(self.dtype, &bytes)
    }

    /// Upload a CpuStorage into a fresh GPU buffer on `device`.
    fn from_cpu(device: &VulkanDevice, s: CpuStorage) -> Result<Self> {
        let dtype = s.dtype();
        let bytes = cpu_storage_to_bytes(&s);
        if bytes.is_empty() && s.dtype().size_in_bytes() != 0 {
            return Err(vk_err(format!("cannot serialize cpu storage of {dtype:?}")));
        }
        let count = bytes.len() / dtype.size_in_bytes().max(1);
        let buffer = device.buffer_from_bytes(&bytes);
        Ok(Self {
            buffer: Arc::new(buffer),
            device: device.clone(),
            dtype,
            count,
        })
    }

    /// Convenience: run a unary CPU op and re-upload (fallback path).
    fn via_cpu(&self, f: impl FnOnce(CpuStorage) -> Result<CpuStorage>) -> Result<Self> {
        let c = self.cpu()?;
        let r = f(c)?;
        Self::from_cpu(&self.device, r)
    }
}

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(gpu_id: usize) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))
        .ok_or_else(|| vk_err("no Vulkan/DX12 GPU adapter found"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(vk_err(format!(
                "only a CPU adapter ({}) is available — no GPU. Build natively on Windows/Linux.",
                info.name
            )));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("candle-vulkan"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(vk_err)?;
        Ok(Self {
            gpu_id,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name: Arc::new(info.name),
            seed: Arc::new(Mutex::new(299792458)),
        })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan { gpu_id: self.gpu_id }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.device, &rhs.device)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        let size = (count * dtype.size_in_bytes()) as u64;
        let buffer = self.empty_buffer(size);
        // wgpu buffers are zero-initialized.
        Ok(VulkanStorage {
            buffer: Arc::new(buffer),
            device: self.clone(),
            dtype,
            count,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // No uninit path on wgpu; zeros is safe and cheap enough.
        self.zeros_impl(shape, dtype)
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let cpu = T::to_cpu_storage(s);
        VulkanStorage::from_cpu(self, cpu)
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        VulkanStorage::from_cpu(self, s.clone())
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        VulkanStorage::from_cpu(self, s)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<Self::Storage> {
        // Generate on CPU then upload (GPU RNG kernel is a later optimization).
        let cpu = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, lo, up)?;
        VulkanStorage::from_cpu(self, cpu)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, std)?;
        VulkanStorage::from_cpu(self, cpu)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed.lock().map_err(vk_err)? = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed.lock().map_err(vk_err)?)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Copy the buffer on-device.
        let size = (self.count * self.dtype.size_in_bytes()) as u64;
        let dst = self.device.empty_buffer(size);
        let mut enc = self
            .device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.buffer, 0, &dst, 0, size.max(4));
        self.device.queue.submit(Some(enc.finish()));
        Ok(VulkanStorage {
            buffer: Arc::new(dst),
            device: self.device.clone(),
            dtype: self.dtype,
            count: self.count,
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        self.cpu()
    }

    // ---- Real WGSL GPU kernel: affine (y = x*mul + add), f32 path ----
    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype == DType::F32 && layout.is_contiguous() {
            return self.affine_f32_gpu(mul as f32, add as f32);
        }
        let l = layout.clone();
        self.via_cpu(|c| c.affine(&l, mul, add))
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let l = layout.clone();
        self.via_cpu(|c| c.powf(&l, e))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let l = layout.clone();
        self.via_cpu(|c| c.elu(&l, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, s: &[usize]) -> Result<Self> {
        let l = layout.clone();
        let s = s.to_vec();
        self.via_cpu(|c| c.reduce_op(op, &l, &s))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let (ll, rl) = (lhs_l.clone(), rhs_l.clone());
        let rhs_cpu = rhs.cpu()?;
        let lhs_cpu = self.cpu()?;
        let r = lhs_cpu.cmp(op, &rhs_cpu, &ll, &rl)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let l = layout.clone();
        self.via_cpu(|c| c.to_dtype(&l, dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let l = layout.clone();
        self.via_cpu(|c| c.unary_impl::<B>(&l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let (ll, rl) = (lhs_l.clone(), rhs_l.clone());
        let rhs_cpu = rhs.cpu()?;
        let lhs_cpu = self.cpu()?;
        let r = lhs_cpu.binary_impl::<B>(&rhs_cpu, &ll, &rl)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let (l, tl, fl) = (layout.clone(), t_l.clone(), f_l.clone());
        let (cc, tc, fc) = (self.cpu()?, t.cpu()?, f.cpu()?);
        let r = cc.where_cond(&l, &tc, &tl, &fc, &fl)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn conv1d(
        &self,
        l: &Layout,
        k: &Self,
        k_l: &Layout,
        p: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let (l, kl, p) = (l.clone(), k_l.clone(), p.clone());
        let kc = k.cpu()?;
        let r = self.cpu()?.conv1d(&l, &kc, &kl, &p)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        k: &Self,
        k_l: &Layout,
        p: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let (l, kl, p) = (l.clone(), k_l.clone(), p.clone());
        let kc = k.cpu()?;
        let r = self.cpu()?.conv_transpose1d(&l, &kc, &kl, &p)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn conv2d(
        &self,
        l: &Layout,
        k: &Self,
        k_l: &Layout,
        p: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let (l, kl, p) = (l.clone(), k_l.clone(), p.clone());
        let kc = k.cpu()?;
        let r = self.cpu()?.conv2d(&l, &kc, &kl, &p)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        k: &Self,
        k_l: &Layout,
        p: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let (l, kl, p) = (l.clone(), k_l.clone(), p.clone());
        let kc = k.cpu()?;
        let r = self.cpu()?.conv_transpose2d(&l, &kc, &kl, &p)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), st: (usize, usize)) -> Result<Self> {
        let l = l.clone();
        self.via_cpu(|c| c.avg_pool2d(&l, k, st))
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), st: (usize, usize)) -> Result<Self> {
        let l = l.clone();
        self.via_cpu(|c| c.max_pool2d(&l, k, st))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        let l = l.clone();
        self.via_cpu(|c| c.upsample_nearest1d(&l, sz))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        let l = l.clone();
        self.via_cpu(|c| c.upsample_nearest2d(&l, h, w))
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align: bool,
        sh: Option<f64>,
        sw: Option<f64>,
    ) -> Result<Self> {
        let l = l.clone();
        self.via_cpu(|c| c.upsample_bilinear2d(&l, h, w, align, sh, sw))
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let (l, il) = (l.clone(), ids_l.clone());
        let ic = ids.cpu()?;
        let r = self.cpu()?.gather(&l, &ic, &il, dim)?;
        VulkanStorage::from_cpu(&self.device, r)
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
        let (l, il, sl) = (l.clone(), ids_l.clone(), src_l.clone());
        let (ic, sc) = (ids.cpu()?, src.cpu()?);
        let mut cc = self.cpu()?;
        cc.scatter_set(&l, &ic, &il, &sc, &sl, dim)?;
        *self = VulkanStorage::from_cpu(&self.device, cc)?;
        Ok(())
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
        let (l, il, sl) = (l.clone(), ids_l.clone(), src_l.clone());
        let (ic, sc) = (ids.cpu()?, src.cpu()?);
        let mut cc = self.cpu()?;
        cc.scatter_add_set(&l, &ic, &il, &sc, &sl, dim)?;
        *self = VulkanStorage::from_cpu(&self.device, cc)?;
        Ok(())
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let (l, il) = (l.clone(), ids_l.clone());
        let ic = ids.cpu()?;
        let r = self.cpu()?.index_select(&ic, &l, &il, dim)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let (l, il, sl) = (l.clone(), ids_l.clone(), src_l.clone());
        let (ic, sc) = (ids.cpu()?, src.cpu()?);
        let r = self.cpu()?.index_add(&l, &ic, &il, &sc, &sl, dim)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // TODO: WGSL tiled matmul. Fallback via CPU for correctness first.
        let (ll, rl) = (lhs_l.clone(), rhs_l.clone());
        let rhs_cpu = rhs.cpu()?;
        let r = self.cpu()?.matmul(&rhs_cpu, bmnk, &ll, &rl)?;
        VulkanStorage::from_cpu(&self.device, r)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let sl = src_l.clone();
        let src_cpu = self.cpu()?;
        let mut dst_cpu = dst.cpu()?;
        src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, &sl)?;
        *dst = VulkanStorage::from_cpu(&self.device, dst_cpu)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        let src_cpu = self.cpu()?;
        let mut dst_cpu = dst.cpu()?;
        src_cpu.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;
        *dst = VulkanStorage::from_cpu(&self.device, dst_cpu)?;
        Ok(())
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let l = l.clone();
        let mut cc = self.cpu()?;
        cc.const_set(v, &l)?;
        *self = VulkanStorage::from_cpu(&self.device, cc)?;
        Ok(())
    }
}

// ---- Real WGSL kernels ----
impl VulkanStorage {
    /// y = x*mul + add, executed on the GPU (f32, contiguous).
    fn affine_f32_gpu(&self, mul: f32, add: f32) -> Result<Self> {
        let dev = &self.device;
        let n = self.count as u32;
        let out = dev.empty_buffer((self.count * 4) as u64);

        // params buffer: [mul, add, n_as_f32, pad]
        let params = [mul, add, f32::from_bits(n), 0.0f32];
        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("affine-params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("affine"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct P { mul: f32, add: f32, n_bits: f32, pad: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = bitcast<u32>(p.n_bits);
    if (i < n) { y[i] = x[i] * p.mul + p.add; }
}
"#
                .into(),
            ),
        });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("affine"),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });
        let bind = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
        }
        dev.queue.submit(Some(enc.finish()));

        Ok(VulkanStorage {
            buffer: Arc::new(out),
            device: dev.clone(),
            dtype: DType::F32,
            count: self.count,
        })
    }
}
