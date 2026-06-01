//! Raw-Vulkan (`ash` 0.38) GPU backend dispatching native SPIR-V kernels.
//!
//! v1 scope: f32, contiguous tensors only. Generalizes the proven `vulkan-probe`
//! `Ctx` (GPU select, host-visible buffers, descriptor sets, SPIR-V pipeline,
//! dispatch, readback) into `VulkanDevice` + `VulkanStorage`. Anything outside the
//! covered op set bails with `Error::Msg`.
use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};
use ash::vk;
use rand::Rng;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex};

// --- SPIR-V kernels (compiled by candle-core build.rs from src/vulkan/shaders/*.comp) ---
macro_rules! spv {
    ($name:literal) => {
        include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".spv"))
    };
}

// kernel name -> SPIR-V bytes. Matches the contract ABI (inputs-then-output buffers, one push block).
fn kernel_spv(name: &str) -> Result<&'static [u8]> {
    let b: &'static [u8] = match name {
        "add" => spv!("add"),
        "sub" => spv!("sub"),
        "mul" => spv!("mul"),
        "div" => spv!("div"),
        "affine" => spv!("affine"),
        "neg" => spv!("neg"),
        "exp" => spv!("exp"),
        "silu" => spv!("silu"),
        "gelu" => spv!("gelu"),
        "relu" => spv!("relu"),
        "sqr" => spv!("sqr"),
        "sqrt" => spv!("sqrt"),
        "recip" => spv!("recip"),
        "tanh" => spv!("tanh"),
        "matmul" => spv!("matmul"),
        "bmm" => spv!("bmm"),
        "bmm_reg" => spv!("bmm_reg"),
        "bmm_coopmat" => spv!("bmm_coopmat"),
        "bmm_coopmat_rb" => spv!("bmm_coopmat_rb"),
        "cast_f2h" => spv!("cast_f2h"),
        "mul_mat_vec_q8" => spv!("mul_mat_vec_q8"),
        "copy" => spv!("copy"),
        "reduce_sum" => spv!("reduce_sum"),
        "reduce_max" => spv!("reduce_max"),
        "strided_copy" => spv!("strided_copy"),
        "index_select" => spv!("index_select"),
        "where_cond" => spv!("where_cond"),
        "softmax_rows" => spv!("softmax_rows"),
        "rms_norm" => spv!("rms_norm"),
        "rope" => spv!("rope"),
        "sin" => spv!("sin"),
        "cos" => spv!("cos"),
        "log" => spv!("log"),
        "abs" => spv!("abs"),
        "floor" => spv!("floor"),
        "ceil" => spv!("ceil"),
        "round" => spv!("round"),
        "sign" => spv!("sign"),
        "erf" => spv!("erf"),
        "gelu_erf" => spv!("gelu_erf"),
        "powf" => spv!("powf"),
        "elu" => spv!("elu"),
        "maximum" => spv!("maximum"),
        "minimum" => spv!("minimum"),
        "cmp" => spv!("cmp"),
        "cast_f2u" => spv!("cast_f2u"),
        "cast_u2f" => spv!("cast_u2f"),
        "reduce_min" => spv!("reduce_min"),
        "reduce_argmin" => spv!("reduce_argmin"),
        "reduce_argmax" => spv!("reduce_argmax"),
        "gather" => spv!("gather"),
        "scatter_set" => spv!("scatter_set"),
        "scatter_add_set" => spv!("scatter_add_set"),
        "conv1d" => spv!("conv1d"),
        "conv2d" => spv!("conv2d"),
        "conv_transpose1d" => spv!("conv_transpose1d"),
        "conv_transpose2d" => spv!("conv_transpose2d"),
        "avg_pool2d" => spv!("avg_pool2d"),
        "max_pool2d" => spv!("max_pool2d"),
        "upsample_nearest1d" => spv!("upsample_nearest1d"),
        "upsample_nearest2d" => spv!("upsample_nearest2d"),
        "upsample_bilinear2d" => spv!("upsample_bilinear2d"),
        _ => crate::bail!("vulkan: no SPIR-V kernel for `{name}`"),
    };
    Ok(b)
}

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

impl From<VulkanError> for Error {
    fn from(e: VulkanError) -> Self {
        Error::Msg(e.to_string())
    }
}

// A compute pipeline + its descriptor-set layout, cached by kernel name. `n_buffers`
// is the binding count this layout was built for so callers can sanity-check.
#[derive(Clone)]
struct CachedPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    n_buffers: usize,
}

struct VkInner {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    qfi: u32,
    gpu_id: usize,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    // Cooperative-matrix (matrix-core / WMMA) availability and the chosen MxNxK tile for an
    // fp16xfp16 -> fp32 subgroup config. Present on native AMD/NV drivers (RDNA3.5 8060S),
    // absent on WSL/Dozen.
    coopmat: bool,
    cm_mnk: (u32, u32, u32),
    // Whether matmul actually uses the coopmat kernel. Opt-in via HANZO_VK_COOPMAT=1 because the
    // current naive (no shared-mem) coopmat GEMM is fp16-precision and not yet faster than the
    // tiled fp32 kernel; default off so it can't regress correctness or speed. Flip the default
    // once the shared-memory-tiled coopmat kernel lands.
    cm_use: bool,
    // CPU-side RNG seed (kernels are deterministic; randoms are generated on the CPU then uploaded).
    seed: Mutex<u64>,
    // kernel name -> built pipeline. &'static str keys: kernel names are compile-time literals.
    pipelines: Mutex<HashMap<&'static str, CachedPipeline>>,
    // Persistent per-dispatch Vulkan objects (command pool/buffer, fence, descriptor pool),
    // reset and reused each dispatch instead of created+destroyed. The whole submit path is
    // serialized through this Mutex (ops are sequential per tensor graph anyway), which both
    // makes reuse sound and kills the per-op allocation churn that dominated dispatch latency.
    submitter: Mutex<Submitter>,
    // Deferred-safe buffer reuse pool (see BufPool). Separate mutex from `submitter`: drop only
    // touches this lock, never `submitter`, so there's no lock cycle.
    bufpool: Mutex<BufPool>,
}

// Reusable submit resources, created once at device init. Many dispatches are recorded into
// the single `cmd` and submitted together on flush (see `dispatch`/`flush_locked`), so the
// per-op CPU<->GPU fence stall is paid once per batch instead of once per op.
struct Submitter {
    cpool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    dpool: vk::DescriptorPool,
    // Is `cmd` currently open with recorded-but-unsubmitted dispatches?
    recording: bool,
    // Dispatches recorded into `cmd` since it was begun (bounded by BATCH_CAP).
    n: u32,
}
// Safety: the contained handles are only ever touched while holding VkInner.submitter's Mutex.
unsafe impl Send for Submitter {}

#[derive(Clone)]
pub struct VulkanDevice {
    inner: Arc<VkInner>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanDevice({})", self.inner.gpu_id)
    }
}

pub struct VulkanStorage {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    count: usize,
    dtype: DType,
    device: VulkanDevice,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanStorage(count={}, dtype={:?})", self.count, self.dtype)
    }
}

// Deferred-safe buffer pool. Buffers are never freed inline: deferred dispatch may still hold a
// buffer's handle in an unflushed command buffer, so freeing on drop would be use-after-free. On
// drop a buffer parks in `pending`; after the next flush+fence (the awaited batch is provably done
// on the GPU) `reclaim` moves it to the size-keyed `free` list, where `raw_buffer` reuses it. This
// bounds device memory to the peak working set and reuses buffers across tokens, instead of leaking
// every allocation (which OOM'd a full unquantized forward on Dozen's ~8GB heap).
#[derive(Default)]
struct BufPool {
    pending: Vec<(u64, vk::Buffer, vk::DeviceMemory)>,
    free: HashMap<u64, Vec<(vk::Buffer, vk::DeviceMemory)>>,
}

// drop: park (size, buffer, memory) for reuse after the next fence. No Vulkan calls here, just
// bookkeeping, so it's cheap and can't race the GPU.
impl Drop for VulkanStorage {
    fn drop(&mut self) {
        if self.buffer == vk::Buffer::null() {
            return;
        }
        let bytes = ((self.count * self.dtype.size_in_bytes()).max(4)) as u64;
        if let Ok(mut pool) = self.device.inner.bufpool.lock() {
            pool.pending.push((bytes, self.buffer, self.memory));
        }
    }
}

// --- low level ash plumbing (generalized from the probe Ctx) ---
impl VulkanDevice {
    fn dev(&self) -> &ash::Device {
        &self.inner.device
    }

    /// Cooperative-matrix (matrix-core) tile `(M, N, K)` if the device supports an
    /// fp16xfp16 -> fp32 subgroup config, else `None`. Used to pick the coopmat matmul path.
    pub fn coopmat_info(&self) -> Option<(u32, u32, u32)> {
        self.inner.coopmat.then_some(self.inner.cm_mnk)
    }

    /// Quantize `W[nout x k]` (row-major fp32) to Q8_0 and upload it to the GPU once. Per 32-block:
    /// one fp16 scale + 32 int8, packed into 9 u32. Returns the device buffer to reuse across many
    /// matvecs (weights are constant during decode). `k` must be a multiple of 32.
    pub fn quantize_q8(&self, w: &[f32], nout: usize, k: usize) -> Result<VulkanStorage> {
        if k % 32 != 0 {
            crate::bail!("quantize_q8: k must be a multiple of 32, got {k}");
        }
        if w.len() != nout * k {
            crate::bail!("quantize_q8: w len {} != {nout}*{k}", w.len());
        }
        let nblocks = k / 32;
        let mut packed = vec![0u32; nout * nblocks * 9];
        for n in 0..nout {
            for b in 0..nblocks {
                let blk = &w[n * k + b * 32..n * k + b * 32 + 32];
                let amax = blk.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                let o = (n * nblocks + b) * 9;
                packed[o] = half::f16::from_f32(scale).to_bits() as u32;
                for j in 0..8 {
                    let mut word = 0u32;
                    for l in 0..4 {
                        let q = (blk[j * 4 + l] * inv).round().clamp(-127.0, 127.0) as i32 as i8;
                        word |= ((q as u8) as u32) << (l * 8);
                    }
                    packed[o + 1 + j] = word;
                }
            }
        }
        self.upload_u32(&packed)
    }

    /// Q8_0 matrix-vector: `y[nout] = Wq * x[k]` where `Wq` came from [`quantize_q8`]. The kernel
    /// reads weights at ~1.125 bytes/elem instead of 4 — the bandwidth lever for memory-bound
    /// decode on this APU. Expect ~1e-2 error (fp16-scale int8 quantization).
    pub fn matvec_q8(&self, wq: &VulkanStorage, x: &[f32], nout: usize, k: usize) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q8: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        let out = self.alloc_f32(nout)?;
        let push = push_u32(&[nout as u32, k as u32]);
        self.dispatch(
            "mul_mat_vec_q8",
            &[wq.buffer, xs.buffer, out.buffer],
            &push,
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        out.to_vec_f32()
    }

    // Allocate a host-visible+coherent storage buffer of `bytes` bytes.
    unsafe fn raw_buffer(&self, bytes: u64) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let bytes = bytes.max(4); // zero-size buffers are illegal; round up to one f32.
        // Reuse a same-size buffer reclaimed from a completed batch before allocating fresh.
        if let Some(pair) = self.inner.bufpool.lock().unwrap().free.get_mut(&bytes).and_then(Vec::pop)
        {
            return Ok(pair);
        }
        let dev = self.dev();
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = dev.create_buffer(&info, None).map_err(vkerr)?;
        let req = dev.get_buffer_memory_requirements(buf);
        let mp = &self.inner.mem_props;
        // On this unified-memory APU every host-visible type is also DEVICE_LOCAL (one 60GB
        // heap), so buffers are already in fast memory — no staging needed. We do prefer a
        // HOST_CACHED+COHERENT type when present: cached speeds up CPU readback, coherent
        // means no manual flush/invalidate. Fall back to plain HOST_VISIBLE|HOST_COHERENT.
        let base = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let cached = base | vk::MemoryPropertyFlags::HOST_CACHED;
        let usable = |i: u32, flags: vk::MemoryPropertyFlags| {
            (req.memory_type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(flags)
        };
        let idx = (0..mp.memory_type_count)
            .find(|&i| usable(i, cached))
            .or_else(|| (0..mp.memory_type_count).find(|&i| usable(i, base)))
            .ok_or_else(|| Error::Msg("vulkan: no host-visible memory type".into()))?;
        let mem = dev
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(idx),
                None,
            )
            .map_err(vkerr)?;
        dev.bind_buffer_memory(buf, mem, 0).map_err(vkerr)?;
        Ok((buf, mem))
    }

    unsafe fn write_f32(&self, mem: vk::DeviceMemory, data: &[f32]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let dev = self.dev();
        let ptr = dev
            .map_memory(mem, 0, (data.len() * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        dev.unmap_memory(mem);
        Ok(())
    }

    unsafe fn read_f32(&self, mem: vk::DeviceMemory, n: usize) -> Result<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        // Ensure all recorded GPU work that may write this buffer has completed.
        self.flush()?;
        let dev = self.dev();
        let ptr = dev
            .map_memory(mem, 0, (n * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *const f32;
        let v = std::slice::from_raw_parts(ptr, n).to_vec();
        dev.unmap_memory(mem);
        Ok(v)
    }

    // u32 mirrors of write_f32/read_f32 (buffers are just bytes; 4 bytes/elem).
    unsafe fn write_u32(&self, mem: vk::DeviceMemory, data: &[u32]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        let dev = self.dev();
        let ptr = dev
            .map_memory(mem, 0, (data.len() * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *mut u32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        dev.unmap_memory(mem);
        Ok(())
    }

    unsafe fn read_u32(&self, mem: vk::DeviceMemory, n: usize) -> Result<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        // Ensure all recorded GPU work that may write this buffer has completed.
        self.flush()?;
        let dev = self.dev();
        let ptr = dev
            .map_memory(mem, 0, (n * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *const u32;
        let v = std::slice::from_raw_parts(ptr, n).to_vec();
        dev.unmap_memory(mem);
        Ok(v)
    }

    // Build (or fetch cached) compute pipeline for `name` with `n_buffers` storage bindings
    // and a push-constant range of `push_size` bytes.
    fn pipeline(&self, name: &'static str, n_buffers: usize, push_size: usize) -> Result<CachedPipeline> {
        if let Some(p) = self.inner.pipelines.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let dev = self.dev();
        let cached = unsafe {
            let binds: Vec<_> = (0..n_buffers as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = dev
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
                    None,
                )
                .map_err(vkerr)?;
            let set_layouts = [set_layout];
            let pcr = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(push_size.max(4) as u32)];
            let layout = dev
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&pcr),
                    None,
                )
                .map_err(vkerr)?;
            let spv_bytes = kernel_spv(name)?;
            let spv = ash::util::read_spv(&mut std::io::Cursor::new(spv_bytes))
                .map_err(|e| Error::Msg(format!("vulkan: bad SPIR-V `{name}`: {e}")))?;
            let module = dev
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&spv), None)
                .map_err(vkerr)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = dev
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)],
                    None,
                )
                .map_err(|(_, e)| vkerr(e))?[0];
            dev.destroy_shader_module(module, None);
            CachedPipeline { pipeline, layout, set_layout, n_buffers }
        };
        self.inner.pipelines.lock().unwrap().insert(name, cached.clone());
        Ok(cached)
    }

    // Dispatch kernel `name` over `bufs` (bound 0..N) with raw push bytes and group counts.
    // Blocks on a fence (v1 is synchronous; matches the probe).
    fn dispatch(
        &self,
        name: &'static str,
        bufs: &[vk::Buffer],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let p = self.pipeline(name, bufs.len(), push.len())?;
        debug_assert_eq!(p.n_buffers, bufs.len(), "vulkan: kernel `{name}` binding count drift");
        let dev = self.dev();
        let queue = self.inner.queue;
        let mut s = self.inner.submitter.lock().unwrap();
        unsafe {
            if !s.recording {
                // Start a fresh batch: free the previous batch's descriptor sets (its GPU work
                // already completed at the last flush) and open the command buffer.
                dev.reset_descriptor_pool(s.dpool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(vkerr)?;
                dev.begin_command_buffer(s.cmd, &vk::CommandBufferBeginInfo::default())
                    .map_err(vkerr)?;
                s.recording = true;
                s.n = 0;
            }
            // Allocate a fresh descriptor set for this dispatch. Sets accumulate within the
            // batch (each recorded dispatch keeps its own); the pool is reset only when the
            // next batch begins, after the current one has been submitted and awaited.
            let set_layouts = [p.set_layout];
            let set = dev
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(s.dpool)
                        .set_layouts(&set_layouts),
                )
                .map_err(vkerr)?[0];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = bufs
                .iter()
                .map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)])
                .collect();
            let writes: Vec<_> = (0..bufs.len())
                .map(|i| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i])
                })
                .collect();
            dev.update_descriptor_sets(&writes, &[]);

            let cmd = s.cmd;
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
            dev.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[set], &[]);
            if !push.is_empty() {
                dev.cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            dev.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
            // Guard RAW/WAW hazards between consecutive dispatches that share a buffer. A single
            // global memory barrier is conservative (it serializes the batch on the GPU) but
            // correct for any buffer aliasing; the win here is removing host stalls, not overlap.
            let bar = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            dev.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &bar,
                &[],
                &[],
            );
            s.n += 1;
            // Bound the descriptor-set budget: a forward longer than BATCH_CAP ops just flushes
            // a handful of times instead of once.
            if s.n >= BATCH_CAP {
                flush_locked(dev, queue, &mut s)?;
                drop(s);
                self.reclaim();
            }
        }
        Ok(())
    }

    // Submit and await any pending recorded dispatches. Must be called before the host maps a
    // buffer (readback) or relies on prior GPU work having completed.
    fn flush(&self) -> Result<()> {
        let dev = self.dev();
        let queue = self.inner.queue;
        {
            let mut s = self.inner.submitter.lock().unwrap();
            flush_locked(dev, queue, &mut s)?;
        }
        self.reclaim();
        Ok(())
    }

    // Move buffers dropped before this point into the reuse pool. Sound only right after a
    // flush+fence: the awaited batch (the last that could reference them) is done on the GPU.
    fn reclaim(&self) {
        let mut pool = self.inner.bufpool.lock().unwrap();
        let pending = std::mem::take(&mut pool.pending);
        for (bytes, buf, mem) in pending {
            pool.free.entry(bytes).or_default().push((buf, mem));
        }
    }

    // Allocate an f32 storage holding `count` elements (uninitialized device memory).
    fn alloc_f32(&self, count: usize) -> Result<VulkanStorage> {
        let (buffer, memory) = unsafe { self.raw_buffer((count * 4) as u64)? };
        Ok(VulkanStorage { buffer, memory, count, dtype: DType::F32, device: self.clone() })
    }

    // Raw fp16 scratch buffer (2 bytes/elem) for cooperative-matrix matmul inputs. Returns just
    // the handle; like the rest of the backend the backing memory is not individually freed.
    fn alloc_f16(&self, count: usize) -> Result<vk::Buffer> {
        let (buffer, _memory) = unsafe { self.raw_buffer((count * 2).max(2) as u64)? };
        Ok(buffer)
    }

    pub(crate) fn upload_f32(&self, data: &[f32]) -> Result<VulkanStorage> {
        let s = self.alloc_f32(data.len())?;
        unsafe { self.write_f32(s.memory, data)? };
        Ok(s)
    }

    // u32 storage (ids/cond). 4 bytes/elem, same as f32.
    fn alloc_u32(&self, count: usize) -> Result<VulkanStorage> {
        let (buffer, memory) = unsafe { self.raw_buffer((count * 4) as u64)? };
        Ok(VulkanStorage { buffer, memory, count, dtype: DType::U32, device: self.clone() })
    }

    fn upload_u32(&self, data: &[u32]) -> Result<VulkanStorage> {
        let s = self.alloc_u32(data.len())?;
        unsafe { self.write_u32(s.memory, data)? };
        Ok(s)
    }
}

fn vkerr(e: vk::Result) -> Error {
    Error::Msg(format!("vulkan: {e:?}"))
}

// Max dispatches recorded into one command buffer before an automatic flush. Bounds the
// descriptor-set pool; this is large enough that a typical transformer forward submits once
// (or a few times), turning hundreds of per-op fence stalls into a handful.
const BATCH_CAP: u32 = 4096;

// End, submit, and block on the current command batch, then mark the submitter idle. A no-op
// when nothing is recorded. Caller must hold the submitter lock.
fn flush_locked(dev: &ash::Device, queue: vk::Queue, s: &mut Submitter) -> Result<()> {
    if !s.recording {
        return Ok(());
    }
    unsafe {
        dev.end_command_buffer(s.cmd).map_err(vkerr)?;
        dev.reset_fences(&[s.fence]).map_err(vkerr)?;
        let cmds = [s.cmd];
        dev.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], s.fence)
            .map_err(vkerr)?;
        dev.wait_for_fences(&[s.fence], true, u64::MAX).map_err(vkerr)?;
    }
    s.recording = false;
    s.n = 0;
    Ok(())
}

// push-constant helpers (std430 scalar layout: tightly packed u32/f32).
fn push_u32(v: &[u32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_ne_bytes());
    }
    b
}

const WG1D: u32 = 64;

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()
                .map_err(|e| Error::Msg(format!("vulkan: loader not found: {e}")))?;
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
            let instance = entry
                .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
                .map_err(vkerr)?;

            // Collect non-CPU adapters in enumeration order; pick the `ordinal`-th (like the probe).
            let mut gpus = Vec::new();
            for pd in instance.enumerate_physical_devices().map_err(vkerr)? {
                let p = instance.get_physical_device_properties(pd);
                let name = CStr::from_ptr(p.device_name.as_ptr()).to_string_lossy().into_owned();
                let is_cpu = p.device_type == vk::PhysicalDeviceType::CPU
                    || name.to_lowercase().contains("llvmpipe");
                if !is_cpu {
                    gpus.push(pd);
                }
            }
            if gpus.is_empty() {
                instance.destroy_instance(None);
                return Err(Error::Msg("vulkan: no non-CPU Vulkan device".into()));
            }
            let pdev = *gpus
                .get(ordinal)
                .ok_or_else(|| Error::Msg(format!("vulkan: no device at ordinal {ordinal}")))?;

            let qfi = instance
                .get_physical_device_queue_family_properties(pdev)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| Error::Msg("vulkan: no compute queue".into()))? as u32;
            let prios = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfi)
                .queue_priorities(&prios)];

            // Probe cooperative-matrix support: need the device extension AND a config with
            // fp16 A/B, fp32 C/result at subgroup scope. Guard the query on the extension being
            // advertised (the loader's fn pointer is only valid then) so WSL/Dozen stays on the
            // plain tiled path.
            let dev_exts = instance.enumerate_device_extension_properties(pdev).unwrap_or_default();
            let has_cm_ext = dev_exts.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME
            });
            let cm_mnk = if has_cm_ext {
                let cm = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
                cm.get_physical_device_cooperative_matrix_properties(pdev)
                    .ok()
                    .and_then(|props| {
                        props.into_iter().find(|p| {
                            p.a_type == vk::ComponentTypeKHR::FLOAT16
                                && p.b_type == vk::ComponentTypeKHR::FLOAT16
                                && p.c_type == vk::ComponentTypeKHR::FLOAT32
                                && p.result_type == vk::ComponentTypeKHR::FLOAT32
                                && p.scope == vk::ScopeKHR::SUBGROUP
                        })
                    })
                    .map(|p| (p.m_size, p.n_size, p.k_size))
            } else {
                None
            };
            let coopmat = cm_mnk.is_some();
            let cm_mnk = cm_mnk.unwrap_or((0, 0, 0));
            let cm_use =
                coopmat && std::env::var("HANZO_VK_COOPMAT").map(|v| v == "1").unwrap_or(false);

            // Enable the coopmat extension + the features its SPIR-V needs (cooperative matrix,
            // Vulkan memory model, fp16 arithmetic, 16-bit storage) only when supported.
            let cm_ext_names = [ash::khr::cooperative_matrix::NAME.as_ptr()];
            let mut cm_feat =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut mm_feat =
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default().vulkan_memory_model(true);
            let mut f16_feat =
                vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            let mut s16_feat = vk::PhysicalDevice16BitStorageFeatures::default()
                .storage_buffer16_bit_access(true);
            let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
            if coopmat {
                dci = dci
                    .enabled_extension_names(&cm_ext_names)
                    .push_next(&mut cm_feat)
                    .push_next(&mut mm_feat)
                    .push_next(&mut f16_feat)
                    .push_next(&mut s16_feat);
            }
            let device = instance.create_device(pdev, &dci, None).map_err(vkerr)?;
            let queue = device.get_device_queue(qfi, 0);
            let mem_props = instance.get_physical_device_memory_properties(pdev);

            // Persistent submit resources. RESET_COMMAND_BUFFER lets us re-record `cmd` each
            // dispatch; the descriptor pool holds enough STORAGE_BUFFER slots for the widest
            // kernel (where_cond/index_select use 4) and is reset (1 set freed) per dispatch.
            let cpool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qfi)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(vkerr)?;
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cpool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(vkerr)?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr)?;
            // Sized for a whole batch: one descriptor set per recorded dispatch (up to
            // BATCH_CAP), each binding up to 4 storage buffers (the widest kernel).
            let dpool_sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(BATCH_CAP * 4)];
            let dpool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&dpool_sizes)
                        .max_sets(BATCH_CAP),
                    None,
                )
                .map_err(vkerr)?;
            let submitter =
                Mutex::new(Submitter { cpool, cmd, fence, dpool, recording: false, n: 0 });

            let inner = VkInner {
                _entry: entry,
                instance,
                device,
                queue,
                qfi,
                gpu_id: ordinal,
                mem_props,
                seed: Mutex::new(299792458),
                pipelines: Mutex::new(HashMap::new()),
                submitter,
                bufpool: Mutex::new(BufPool::default()),
                coopmat,
                cm_mnk,
                cm_use,
            };
            Ok(Self { inner: Arc::new(inner) })
        }
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan { gpu_id: self.inner.gpu_id }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        match dtype {
            DType::F32 => self.upload_f32(&vec![0f32; count]),
            DType::U32 => self.upload_u32(&vec![0u32; count]),
            _ => crate::bail!("vulkan: only f32/u32 supported, got {dtype:?}"),
        }
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        match dtype {
            DType::F32 => self.alloc_f32(shape.elem_count()),
            DType::U32 => self.alloc_u32(shape.elem_count()),
            _ => crate::bail!("vulkan: only f32/u32 supported, got {dtype:?}"),
        }
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&T::to_cpu_storage(s))
    }

    fn storage_from_cpu_storage(&self, s: &CpuStorage) -> Result<Self::Storage> {
        // Vulkan computes in f32; f16/bf16 weights are upcast on upload so real
        // fp16/bf16 safetensors load on the GPU (uniform, so half tensors stay dtype-consistent).
        match s {
            CpuStorage::F32(v) => self.upload_f32(v),
            CpuStorage::U32(v) => self.upload_u32(v),
            CpuStorage::F16(v) => self.upload_f32(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>()),
            CpuStorage::BF16(v) => self.upload_f32(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>()),
            _ => crate::bail!("vulkan: only f32/u32/f16/bf16 supported, got {:?}", s.dtype()),
        }
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&s)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, min: f64, max: f64) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            crate::bail!("vulkan: rand_uniform only f32, got {dtype:?}");
        }
        // Generate on the CPU (rand 0.9 API, mirrors cpu_backend) then upload.
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
            crate::bail!("vulkan: rand_normal only f32, got {dtype:?}");
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
        self.flush()?;
        unsafe { self.dev().device_wait_idle().map_err(vkerr) }
    }
}

impl VulkanStorage {
    fn count(&self) -> usize {
        self.count
    }

    // Download the whole buffer as f32 (used internally + by to_cpu_storage).
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        unsafe { self.device.read_f32(self.memory, self.count) }
    }

    // Download the whole buffer as u32 (used by to_cpu_storage for U32 storage).
    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        unsafe { self.device.read_u32(self.memory, self.count) }
    }

    // 1D elementwise dispatch over `n` elements: ceil(n/64) workgroups.
    fn groups_1d(n: usize) -> (u32, u32, u32) {
        ((n as u32).div_ceil(WG1D), 1, 1)
    }

    // Materialize any (strided/broadcast) f32 layout into a fresh contiguous buffer
    // of `layout.elem_count()` elements via the strided_copy kernel. Identity for
    // already-contiguous layouts. rank <= 6.
    fn contiguous(&self, layout: &Layout) -> Result<VulkanStorage> {
        let dims = layout.dims();
        let rank = dims.len();
        if rank > 6 {
            crate::bail!("vulkan: contiguous supports rank <= 6, got {rank}");
        }
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let strides = layout.stride();
        let mut p = vec![n as u32, rank as u32, layout.start_offset() as u32];
        let mut shape6 = [0u32; 6];
        let mut stride6 = [0u32; 6];
        for d in 0..rank {
            shape6[d] = dims[d] as u32;
            stride6[d] = strides[d] as u32;
        }
        p.extend_from_slice(&shape6);
        p.extend_from_slice(&stride6);
        self.device
            .dispatch("strided_copy", &[self.buffer, out.buffer], &push_u32(&p), Self::groups_1d(n))?;
        Ok(out)
    }

    // --- native fused ops (replace the GPU<->CPU round-trip fallbacks) ---

    // softmax over the last dim. `self` is the input; `layout` its layout. One row per thread.
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<VulkanStorage> {
        let x = self.contiguous(layout)?;
        let dims = layout.dims();
        let m = *dims.last().unwrap_or(&1);
        let nrows = layout.shape().elem_count() / m.max(1);
        let out = self.device.alloc_f32(nrows * m)?;
        self.device.dispatch(
            "softmax_rows",
            &[x.buffer, out.buffer],
            &push_u32(&[nrows as u32, m as u32]),
            Self::groups_1d(nrows),
        )?;
        Ok(out)
    }

    // rms-norm over the last dim. `self`=x, `alpha`=scale [m]. Matches candle rms-norm.
    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &VulkanStorage,
        alpha_l: &Layout,
        eps: f32,
    ) -> Result<VulkanStorage> {
        let x = self.contiguous(layout)?;
        let a = alpha.contiguous(alpha_l)?;
        let dims = layout.dims();
        let m = *dims.last().unwrap_or(&1);
        let nrows = layout.shape().elem_count() / m.max(1);
        let out = self.device.alloc_f32(nrows * m)?;
        let mut push = push_u32(&[nrows as u32, m as u32]);
        push.extend_from_slice(&eps.to_ne_bytes());
        self.device.dispatch(
            "rms_norm",
            &[x.buffer, a.buffer, out.buffer],
            &push,
            Self::groups_1d(nrows),
        )?;
        Ok(out)
    }

    // GPT-NeoX rotary embedding. `self`=src [b,h,t,d], cos/sin [t,d/2] or [b,t,d/2].
    pub fn rope(
        &self,
        layout: &Layout,
        cos: &VulkanStorage,
        cos_l: &Layout,
        sin: &VulkanStorage,
        sin_l: &Layout,
    ) -> Result<VulkanStorage> {
        let src = self.contiguous(layout)?;
        let c = cos.contiguous(cos_l)?;
        let s = sin.contiguous(sin_l)?;
        let (b, h, t, d) = layout.shape().dims4()?;
        let unbatched = (cos_l.dims().len() == 3 && sin_l.dims().len() == 3) as u32;
        let out = self.device.alloc_f32(b * h * t * d)?;
        let pairs = b * h * t * (d / 2);
        self.device.dispatch(
            "rope",
            &[src.buffer, c.buffer, s.buffer, out.buffer],
            &push_u32(&[b as u32, h as u32, t as u32, d as u32, unbatched]),
            Self::groups_1d(pairs),
        )?;
        Ok(out)
    }

    // Shared scatter: write/accumulate src into dst (self) along `dim` at positions `ids`.
    // dst is assumed contiguous (its layout `l` gives the dim sizes). ids/src share a shape.
    fn scatter_impl(
        &self,
        kernel: &'static str,
        l: &Layout,
        ids: &VulkanStorage,
        ids_l: &Layout,
        src: &VulkanStorage,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        if ids.dtype != DType::U32 {
            crate::bail!("vulkan: scatter requires u32 ids, got {:?}", ids.dtype);
        }
        let idc = ids.contiguous(ids_l)?;
        let srcc = src.contiguous(src_l)?;
        let src_dims = src_l.dims();
        let dst_dims = l.dims();
        let right: usize = src_dims[dim + 1..].iter().product();
        let dim_src = src_dims[dim];
        let dim_dst = dst_dims[dim];
        let n = src_l.shape().elem_count();
        self.device.dispatch(
            kernel,
            &[self.buffer, srcc.buffer, idc.buffer],
            &push_u32(&[n as u32, right as u32, dim_src as u32, dim_dst as u32]),
            Self::groups_1d(n),
        )?;
        Ok(())
    }

    // Shared 2D pooling (no padding): out_h = (ih-kh)/sh + 1, out_w = (iw-kw)/sw + 1.
    fn pool2d(
        &self,
        kernel: &'static str,
        l: &Layout,
        k: (usize, usize),
        stride: (usize, usize),
    ) -> Result<VulkanStorage> {
        let inp = self.contiguous(l)?;
        let (b, c, ih, iw) = l.shape().dims4()?;
        let (kh, kw) = k;
        let (sh, sw) = stride;
        let oh = (ih - kh) / sh + 1;
        let ow = (iw - kw) / sw + 1;
        let out = self.device.alloc_f32(b * c * oh * ow)?;
        self.device.dispatch(
            kernel,
            &[inp.buffer, out.buffer],
            &push_u32(&[
                b as u32, c as u32, ih as u32, iw as u32, oh as u32, ow as u32,
                kh as u32, kw as u32, sh as u32, sw as u32,
            ]),
            Self::groups_1d(b * c * oh * ow),
        )?;
        Ok(out)
    }
}

impl BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        // Raw device-memory copy of the whole buffer (ignores layout; assumes contiguous).
        // Buffers are just bytes (4 bytes/elem), so copy regardless of dtype.
        match self.dtype {
            DType::F32 => self.device.upload_f32(&self.to_vec_f32()?),
            DType::U32 => self.device.upload_u32(&self.to_vec_u32()?),
            dt => crate::bail!("vulkan: try_clone unsupported dtype {dt:?}"),
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
        let c = self.contiguous(layout)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut push = (n as u32).to_ne_bytes().to_vec();
        push.extend_from_slice(&(mul as f32).to_ne_bytes());
        push.extend_from_slice(&(add as f32).to_ne_bytes());
        self.device
            .dispatch("affine", &[c.buffer, out.buffer], &push, Self::groups_1d(n))?;
        Ok(out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let c = self.contiguous(layout)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut push = (n as u32).to_ne_bytes().to_vec();
        push.extend_from_slice(&(e as f32).to_ne_bytes());
        self.device.dispatch("powf", &[c.buffer, out.buffer], &push, Self::groups_1d(n))?;
        Ok(out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let c = self.contiguous(layout)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut push = (n as u32).to_ne_bytes().to_vec();
        push.extend_from_slice(&(alpha as f32).to_ne_bytes());
        self.device.dispatch("elu", &[c.buffer, out.buffer], &push, Self::groups_1d(n))?;
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
        // Reduce over the last dim only, contiguously => rows x cols, one out per row.
        let dims = layout.dims();
        let rank = dims.len();
        if rank == 0 {
            crate::bail!("vulkan: reduce_op on scalar not supported");
        }
        if sum_dims != [rank - 1] {
            crate::bail!(
                "vulkan: reduce_op only supports the last dim (got dims={sum_dims:?}, rank={rank})"
            );
        }
        let c = self.contiguous(layout)?;
        let cols = dims[rank - 1];
        let rows: usize = dims[..rank - 1].iter().product();
        // arg-reductions return u32 indices; value reductions return f32.
        let out = if is_arg { self.device.alloc_u32(rows)? } else { self.device.alloc_f32(rows)? };
        let push = push_u32(&[rows as u32, cols as u32]);
        // one invocation per row
        self.device
            .dispatch(kernel, &[c.buffer, out.buffer], &push, Self::groups_1d(rows))?;
        Ok(out)
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        // Elementwise compare on f32 operands -> u32 {0,1} mask. (Backend has no U8; U32 is a
        // valid integer mask and composes with where_cond / index ops.)
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
            &[lc.buffer, rc.buffer, out.buffer],
            &push_u32(&[n as u32, code]),
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
                    &[c.buffer, out.buffer],
                    &push_u32(&[n as u32]),
                    Self::groups_1d(n),
                )?;
                Ok(out)
            }
            (DType::U32, DType::F32) => {
                let c = self.contiguous(layout)?;
                let out = self.device.alloc_f32(n)?;
                self.device.dispatch(
                    "cast_u2f",
                    &[c.buffer, out.buffer],
                    &push_u32(&[n as u32]),
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
            // Anything still without a SPIR-V kernel: fall back to CPU (correct, slow).
            _ => {
                let cpu = self.to_cpu_storage()?;
                let r = crate::backend::BackendStorage::unary_impl::<B>(&cpu, layout)?;
                return self.device.storage_from_cpu_storage(&r);
            }
        };
        let c = self.contiguous(layout)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let push = (n as u32).to_ne_bytes();
        self.device
            .dispatch(kernel, &[c.buffer, out.buffer], &push, Self::groups_1d(n))?;
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
            // Anything still without a SPIR-V kernel: fall back to CPU (correct, slow).
            _ => {
                let lc = self.to_cpu_storage()?;
                let rc = rhs.to_cpu_storage()?;
                let r = crate::backend::BackendStorage::binary_impl::<B>(&lc, &rc, lhs_l, rhs_l)?;
                return self.device.storage_from_cpu_storage(&r);
            }
        };
        // candle pre-broadcasts both layouts to the output shape (possibly with
        // stride-0 dims); contiguous() materializes each to n elements.
        let lc = self.contiguous(lhs_l)?;
        let rc = rhs.contiguous(rhs_l)?;
        let n = lhs_l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let push = (n as u32).to_ne_bytes();
        self.device
            .dispatch(kernel, &[lc.buffer, rc.buffer, out.buffer], &push, Self::groups_1d(n))?;
        Ok(out)
    }

    fn where_cond(&self, l: &Layout, t: &Self, t_l: &Layout, f: &Self, f_l: &Layout) -> Result<Self> {
        // self is the condition mask (u32). Kernel reads it directly, so it must be contiguous.
        if self.dtype != DType::U32 {
            crate::bail!("vulkan: where_cond requires u32 condition, got {:?}", self.dtype);
        }
        if !l.is_contiguous() {
            crate::bail!("vulkan: where_cond requires contiguous condition");
        }
        let tc = t.contiguous(t_l)?;
        let fc = f.contiguous(f_l)?;
        let n = l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let push = push_u32(&[n as u32]);
        self.device.dispatch(
            "where_cond",
            &[self.buffer, tc.buffer, fc.buffer, out.buffer],
            &push,
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let w = kernel.contiguous(kernel_l)?;
        let l_out = p.l_out();
        let out = self.device.alloc_f32(p.b_size * p.c_out * l_out)?;
        let push = push_u32(&[
            p.b_size as u32, p.c_in as u32, p.c_out as u32, p.l_in as u32, l_out as u32,
            p.k_size as u32, p.padding as u32, p.stride as u32, p.dilation as u32,
        ]);
        self.device.dispatch(
            "conv1d",
            &[inp.buffer, w.buffer, out.buffer],
            &push,
            Self::groups_1d(p.b_size * p.c_out * l_out),
        )?;
        Ok(out)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let w = kernel.contiguous(kernel_l)?;
        let l_out = p.l_out();
        let out = self.device.alloc_f32(p.b_size * p.c_out * l_out)?;
        let push = push_u32(&[
            p.b_size as u32, p.c_in as u32, p.c_out as u32, p.l_in as u32, l_out as u32,
            p.k_size as u32, p.padding as u32, p.stride as u32, p.dilation as u32,
        ]);
        self.device.dispatch(
            "conv_transpose1d",
            &[inp.buffer, w.buffer, out.buffer],
            &push,
            Self::groups_1d(p.b_size * p.c_out * l_out),
        )?;
        Ok(out)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let w = kernel.contiguous(kernel_l)?;
        let (oh, ow) = (p.out_h(), p.out_w());
        let out = self.device.alloc_f32(p.b_size * p.c_out * oh * ow)?;
        let push = push_u32(&[
            p.b_size as u32, p.c_in as u32, p.c_out as u32, p.i_h as u32, p.i_w as u32,
            oh as u32, ow as u32, p.k_h as u32, p.k_w as u32, p.padding as u32,
            p.stride as u32, p.dilation as u32,
        ]);
        self.device.dispatch(
            "conv2d",
            &[inp.buffer, w.buffer, out.buffer],
            &push,
            Self::groups_1d(p.b_size * p.c_out * oh * ow),
        )?;
        Ok(out)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        p: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let w = kernel.contiguous(kernel_l)?;
        let (oh, ow) = (p.out_h(), p.out_w());
        let out = self.device.alloc_f32(p.b_size * p.c_out * oh * ow)?;
        let push = push_u32(&[
            p.b_size as u32, p.c_in as u32, p.c_out as u32, p.i_h as u32, p.i_w as u32,
            oh as u32, ow as u32, p.k_h as u32, p.k_w as u32, p.padding as u32,
            p.stride as u32, p.dilation as u32,
        ]);
        self.device.dispatch(
            "conv_transpose2d",
            &[inp.buffer, w.buffer, out.buffer],
            &push,
            Self::groups_1d(p.b_size * p.c_out * oh * ow),
        )?;
        Ok(out)
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.pool2d("avg_pool2d", l, k, stride)
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.pool2d("max_pool2d", l, k, stride)
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let (b, c, l_in) = l.shape().dims3()?;
        let out = self.device.alloc_f32(b * c * sz)?;
        self.device.dispatch(
            "upsample_nearest1d",
            &[inp.buffer, out.buffer],
            &push_u32(&[b as u32, c as u32, l_in as u32, sz as u32]),
            Self::groups_1d(b * c * sz),
        )?;
        Ok(out)
    }

    fn upsample_nearest2d(&self, l: &Layout, oh: usize, ow: usize) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let (b, c, ih, iw) = l.shape().dims4()?;
        let out = self.device.alloc_f32(b * c * oh * ow)?;
        self.device.dispatch(
            "upsample_nearest2d",
            &[inp.buffer, out.buffer],
            &push_u32(&[b as u32, c as u32, ih as u32, iw as u32, oh as u32, ow as u32]),
            Self::groups_1d(b * c * oh * ow),
        )?;
        Ok(out)
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        oh: usize,
        ow: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let inp = self.contiguous(l)?;
        let (b, c, ih, iw) = l.shape().dims4()?;
        // PyTorch area_pixel scale logic, mirrored from the CPU backend.
        let sh = if align_corners {
            if oh > 1 { (ih - 1) as f64 / (oh - 1) as f64 } else { 0.0 }
        } else {
            scale_h.map(|s| 1.0 / s).unwrap_or(ih as f64 / oh as f64)
        };
        let sw = if align_corners {
            if ow > 1 { (iw - 1) as f64 / (ow - 1) as f64 } else { 0.0 }
        } else {
            scale_w.map(|s| 1.0 / s).unwrap_or(iw as f64 / ow as f64)
        };
        let out = self.device.alloc_f32(b * c * oh * ow)?;
        let mut push = push_u32(&[
            b as u32, c as u32, ih as u32, iw as u32, oh as u32, ow as u32,
            align_corners as u32,
        ]);
        push.extend_from_slice(&(sh as f32).to_ne_bytes());
        push.extend_from_slice(&(sw as f32).to_ne_bytes());
        self.device.dispatch(
            "upsample_bilinear2d",
            &[inp.buffer, out.buffer],
            &push,
            Self::groups_1d(b * c * oh * ow),
        )?;
        Ok(out)
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        if ids.dtype != DType::U32 {
            crate::bail!("vulkan: gather requires u32 ids, got {:?}", ids.dtype);
        }
        let src = self.contiguous(l)?;
        let idc = ids.contiguous(ids_l)?;
        let out_dims = ids_l.dims();
        let src_dims = l.dims();
        let right: usize = out_dims[dim + 1..].iter().product();
        let dim_out = out_dims[dim];
        let dim_src = src_dims[dim];
        let n = ids_l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device.dispatch(
            "gather",
            &[src.buffer, idc.buffer, out.buffer],
            &push_u32(&[n as u32, right as u32, dim_out as u32, dim_src as u32]),
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

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if ids.dtype != DType::U32 {
            crate::bail!("vulkan: index_select requires u32 ids, got {:?}", ids.dtype);
        }
        // Kernel reads ids directly, so it must be contiguous.
        if !ids_l.is_contiguous() {
            crate::bail!("vulkan: index_select requires contiguous ids");
        }
        let src_c = self.contiguous(l)?;
        let dims = l.dims();
        let left: usize = dims[..dim].iter().product();
        let dim_size = dims[dim];
        let right: usize = dims[dim + 1..].iter().product();
        let n_ids = ids_l.shape().elem_count();
        let total = left * n_ids * right;
        let out = self.device.alloc_f32(total)?;
        let push = push_u32(&[left as u32, dim_size as u32, right as u32, n_ids as u32]);
        self.device.dispatch(
            "index_select",
            &[ids.buffer, src_c.buffer, out.buffer],
            &push,
            Self::groups_1d(total),
        )?;
        Ok(out)
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
        crate::bail!("vulkan: index_add not implemented")
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // Materialize operands contiguous: lhs -> [b,m,k], rhs -> [b,k,n], packed.
        let lc = self.contiguous(lhs_l)?;
        let rc = rhs.contiguous(rhs_l)?;
        // C[b,m,n] = A[b,m,k] * B[b,k,n], row-major. Kernel push order is {batch,m,k,n}.
        let out = self.device.alloc_f32(b * m * n)?;

        // Matrix-core path: when the device has a 16x16x16 fp16 coopmat config and every tile
        // dimension is a multiple of 16, cast the operands to fp16 and run the WMMA GEMM. This
        // is fp16-precision matmul (as inference engines like llama.cpp do) — far faster on the
        // matrix cores than the scalar tiled kernel. Falls back below otherwise.
        if self.device.inner.cm_use
            && matches!(self.device.coopmat_info(), Some((16, 16, 16)))
            && m % 16 == 0
            && n % 16 == 0
            && k % 16 == 0
        {
            let a16 = self.device.alloc_f16(b * m * k)?;
            let b16 = self.device.alloc_f16(b * k * n)?;
            self.device.dispatch(
                "cast_f2h",
                &[lc.buffer, a16],
                &push_u32(&[(b * m * k) as u32]),
                Self::groups_1d(b * m * k),
            )?;
            self.device.dispatch(
                "cast_f2h",
                &[rc.buffer, b16],
                &push_u32(&[(b * k * n) as u32]),
                Self::groups_1d(b * k * n),
            )?;
            // Register-blocked coopmat: one subgroup owns a 2x2 grid of 16x16 tiles (32x32),
            // reusing each loaded fragment across the grid. groups cover ceil over the 2x2 block.
            let push = push_u32(&[b as u32, m as u32, k as u32, n as u32]);
            let mt = (m / 16) as u32;
            let nt = (n / 16) as u32;
            let groups = (nt.div_ceil(4), mt.div_ceil(4), b as u32);
            self.device.dispatch("bmm_coopmat_rb", &[a16, b16, out.buffer], &push, groups)?;
            return Ok(out);
        }

        // Register-blocked tiled GEMM (64x64 tile, 4x4 per thread): higher arithmetic intensity
        // than the 1-output-per-thread `bmm`, so it wins on prefill-shaped (large-M) matmuls.
        let push = push_u32(&[b as u32, m as u32, k as u32, n as u32]);
        let groups = ((n as u32).div_ceil(64), (m as u32).div_ceil(64), b as u32);
        self.device
            .dispatch("bmm_reg", &[lc.buffer, rc.buffer, out.buffer], &push, groups)?;
        Ok(out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let n = src_l.shape().elem_count();
        if n == 0 {
            return Ok(());
        }
        if dst_offset != 0 {
            crate::bail!("vulkan: copy_strided_src with non-zero dst offset not supported");
        }
        if dst.count() < n {
            crate::bail!("vulkan: copy_strided_src dst too small ({} < {n})", dst.count());
        }
        // Materialize src_l (any strided/broadcast layout) directly into dst via strided_copy.
        let dims = src_l.dims();
        let rank = dims.len();
        if rank > 6 {
            crate::bail!("vulkan: copy_strided_src supports rank <= 6, got {rank}");
        }
        let strides = src_l.stride();
        let mut p = vec![n as u32, rank as u32, src_l.start_offset() as u32];
        let mut shape6 = [0u32; 6];
        let mut stride6 = [0u32; 6];
        for d in 0..rank {
            shape6[d] = dims[d] as u32;
            stride6[d] = strides[d] as u32;
        }
        p.extend_from_slice(&shape6);
        p.extend_from_slice(&stride6);
        self.device
            .dispatch("strided_copy", &[self.buffer, dst.buffer], &push_u32(&p), Self::groups_1d(n))
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
        // CPU round-trip (bit-preserving via f32): copy d1 rows of d2 contiguous elems
        // with per-row src/dst strides. Works for any 4-byte dtype.
        let src = self.to_vec_f32()?;
        let mut dstv = dst.to_vec_f32()?;
        for i in 0..d1 {
            let so = src_offset + i * src_stride1;
            let d_o = dst_offset + i * dst_stride1;
            dstv[d_o..d_o + d2].copy_from_slice(&src[so..so + d2]);
        }
        unsafe { dst.device.write_f32(dst.memory, &dstv) }
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        if !layout.is_contiguous() {
            crate::bail!("vulkan: const_set requires contiguous layout");
        }
        let v = s.to_f64() as f32;
        // Host-visible memory: fill on the CPU then map+write the contiguous region.
        let n = layout.shape().elem_count();
        let off = layout.start_offset();
        // Read-modify-write so we don't clobber elements outside [off, off+n).
        let mut data = self.to_vec_f32()?;
        if off + n > data.len() {
            crate::bail!("vulkan: const_set out of range");
        }
        for x in &mut data[off..off + n] {
            *x = v;
        }
        unsafe { self.device.write_f32(self.memory, &data) }
    }
}
