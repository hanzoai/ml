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

// --- SPIR-V kernels (compiled by hanzo-ml build.rs from src/vulkan/shaders/*.comp) ---
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
        "silu_mul" => spv!("silu_mul"),
        "sigmoid" => spv!("sigmoid"),
        "gelu" => spv!("gelu"),
        "relu" => spv!("relu"),
        "sqr" => spv!("sqr"),
        "sqrt" => spv!("sqrt"),
        "recip" => spv!("recip"),
        "tanh" => spv!("tanh"),
        "matmul" => spv!("matmul"),
        "bmm" => spv!("bmm"),
        "bmm_reg" => spv!("bmm_reg"),
        "bmm_reg_nt" => spv!("bmm_reg_nt"),
        "bmm_coopmat" => spv!("bmm_coopmat"),
        "bmm_coopmat_rb" => spv!("bmm_coopmat_rb"),
        "bmm_coopmat_rb_nt" => spv!("bmm_coopmat_rb_nt"),
        "cast_f2h" => spv!("cast_f2h"),
        "dequant_q8_f16" => spv!("dequant_q8_f16"),
        "mul_mat_vec_q8" => spv!("mul_mat_vec_q8"),
        "mul_mat_vec_q8_sg" => spv!("mul_mat_vec_q8_sg"),
        "mul_mat_q8" => spv!("mul_mat_q8"),
        "mul_mat_q8_dp4a" => spv!("mul_mat_q8_dp4a"),
        "mul_mm_q8" => spv!("mul_mm_q8"),
        "mul_mm_q8_dp4a" => spv!("mul_mm_q8_dp4a"),
        "mul_mm_q8_dp4a_8x4" => spv!("mul_mm_q8_dp4a_8x4"),
        "mul_mm_q8_dp4a_4x8" => spv!("mul_mm_q8_dp4a_4x8"),
        "mul_mm_q8_dp4a_8x8" => spv!("mul_mm_q8_dp4a_8x8"),
        "mul_mm_q8_dp4a_2x2" => spv!("mul_mm_q8_dp4a_2x2"),
        "mul_mm_q8_dp4a_6x6" => spv!("mul_mm_q8_dp4a_6x6"),
        "mul_mm_q8_dp4a_4x4n4" => spv!("mul_mm_q8_dp4a_4x4n4"),
        "mul_mm_q8_dp4a_8x2" => spv!("mul_mm_q8_dp4a_8x2"),
        "mul_mm_q8_dp4a_12x4" => spv!("mul_mm_q8_dp4a_12x4"),
        "mul_mm_q8_dp4a_16x2" => spv!("mul_mm_q8_dp4a_16x2"),
        "mul_mm_q8_dp4a_16x4" => spv!("mul_mm_q8_dp4a_16x4"),
        "quantize_act_q8" => spv!("quantize_act_q8"),
        "mul_mat_q4k" => spv!("mul_mat_q4k"),
        "mul_mat_q4k_dp4a" => spv!("mul_mat_q4k_dp4a"),
        "mul_mat_vec_q4k" => spv!("mul_mat_vec_q4k"),
        "mul_mat_vec_q5k" => spv!("mul_mat_vec_q5k"),
        "mul_mat_vec_q6k" => spv!("mul_mat_vec_q6k"),
        "mul_mat_vec_id_q4k" => spv!("mul_mat_vec_id_q4k"),
        "mul_mat_vec_id_q4k_sg" => spv!("mul_mat_vec_id_q4k_sg"),
        "mul_mat_vec_id_q8" => spv!("mul_mat_vec_id_q8"),
        "mul_mat_vec_id_q8_sg" => spv!("mul_mat_vec_id_q8_sg"),
        "copy" => spv!("copy"),
        "copy2d" => spv!("copy2d"),
        "const_fill" => spv!("const_fill"),
        "reduce_sum" => spv!("reduce_sum"),
        "reduce_max" => spv!("reduce_max"),
        "strided_copy" => spv!("strided_copy"),
        "index_select" => spv!("index_select"),
        "where_cond" => spv!("where_cond"),
        "softmax_rows" => spv!("softmax_rows"),
        "attn_decode" => spv!("attn_decode"),
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
        "argsort" => spv!("argsort"),
        "gather" => spv!("gather"),
        "scatter_set" => spv!("scatter_set"),
        "scatter_add_set" => spv!("scatter_add_set"),
        "conv1d" => spv!("conv1d"),
        "gdn_step" => spv!("gdn_step"),
        "gdn_conv1d_step" => spv!("gdn_conv1d_step"),
        "conv2d" => spv!("conv2d"),
        "conv_transpose1d" => spv!("conv_transpose1d"),
        "conv_transpose2d" => spv!("conv_transpose2d"),
        "avg_pool2d" => spv!("avg_pool2d"),
        "max_pool2d" => spv!("max_pool2d"),
        "upsample_nearest1d" => spv!("upsample_nearest1d"),
        "upsample_nearest2d" => spv!("upsample_nearest2d"),
        "upsample_bilinear2d" => spv!("upsample_bilinear2d"),
        "gdn_recurrence" => spv!("gdn_recurrence"),
        "gdn_chunked" => spv!("gdn_chunked"),
        "gdn_conv_update" => spv!("gdn_conv_update"),
        "gdn_conv_full" => spv!("gdn_conv_full"),
        "gdn_conv_state_save" => spv!("gdn_conv_state_save"),
        "gdn_gating" => spv!("gdn_gating"),
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

// Device-memory placement strategy for storage buffers, set once at init from
// HANZO_VK_DEVICE_MEMORY_STRATEGY. On this RDNA3.5 UMA APU the host-visible "VRAM carveout" heap
// is small (a few hundred MB), while the large unified pool (~GTT, tens of GB of the 128GB system
// RAM) is exposed as a DEVICE_LOCAL-only heap. A pure host-visible policy therefore OOMs an 18.6GB
// model even though there is ample memory; we must be able to place big buffers in the large heap.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MemStrategy {
    // Only ever use host-visible memory types (legacy behaviour; correct on classic discrete GPUs
    // with a single large host-visible heap or where staging is handled elsewhere).
    HostOnly,
    // Prefer the largest DEVICE_LOCAL heap for every buffer; fall back to host-visible.
    DeviceFirst,
    // Per-allocation: use a host-visible type when its heap is large enough to hold the buffer,
    // otherwise place it in the largest usable (typically DEVICE_LOCAL) heap. Default.
    Auto,
}

struct VkInner {
    _entry: ash::Entry,
    #[allow(dead_code)] // held to keep the Vulkan instance alive for the device's lifetime
    instance: ash::Instance,
    // Physical device handle, retained so memory budget (free bytes) can be re-queried at runtime
    // for the scratch-allocation guard (heap `size` is total capacity, not what is currently free).
    pdev: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    #[allow(dead_code)] // queue family index, retained for completeness
    qfi: u32,
    gpu_id: usize,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    // Buffer-placement policy (see MemStrategy).
    mem_strategy: MemStrategy,
    // VK_EXT_memory_budget advertised: lets us query per-heap *free* bytes (heapBudget) at runtime
    // instead of only the static total heap `size`. When absent we fall back to total heap size as
    // a conservative upper bound for the scratch guard.
    has_mem_budget: bool,
    // Subgroup arithmetic support for the COMPUTE stage (queried from PhysicalDeviceSubgroupProperties
    // at init). When true the q8 mat-vec uses the subgroup-reduced kernel (one subgroup per output
    // row, fused subgroupAdd) instead of the scalar one-thread-per-row kernel; this raises memory-
    // level parallelism per row and halves the dispatch count on the decode hot path. Gated because
    // the subgroup SPIR-V needs GroupNonUniformArithmetic, which not every driver (e.g. some WSL/
    // Dozen configs) provides. `subgroup_size` is the reported subgroup width.
    subgroup_matvec: bool,
    subgroup_size: u32,
    // Cooperative-matrix (matrix-core / WMMA) availability and the chosen MxNxK tile for an
    // fp16xfp16 -> fp32 subgroup config. Present on native AMD/NV drivers (RDNA3.5 8060S),
    // absent on WSL/Dozen.
    coopmat: bool,
    cm_mnk: (u32, u32, u32),
    // Hardware int8 4x8-packed-signed dot product (GL_EXT_shader_integer_dot_product / OpSDotAccSat).
    // True iff the device advertises BOTH shaderIntegerDotProduct (feature) AND the 4x8-bit-packed-
    // signed-accelerated property (queried at init). Gates the Q8_0 PREFILL path: when set, M>1 Q8
    // matmuls run mul_mat_q8_dp4a (quantize activations to int8 once, then dp4a vs the int8 weights)
    // instead of mul_mat_q8 (f32 weight-decode + MACs) -- the compute-bound prefill lever on RDNA3.5.
    // Decode (M==1) and non-Q8 paths are untouched; devices without the feature fall back to mul_mat_q8.
    int_dot8: bool,
    // Whether matmul uses the register-blocked coopmat kernel (bmm_coopmat_rb). Default ON when the
    // device advertises coopmat (measured 1.3-2.7x over fp32 bmm_reg on the real AMD driver, full
    // forward argmax matches CPU); HANZO_VK_COOPMAT=0 forces the fp32 path.
    cm_use: bool,
    // CPU-side RNG seed (kernels are deterministic; randoms are generated on the CPU then uploaded).
    seed: Mutex<u64>,
    // Per-flush phase profiling, gated on HANZO_VK_PROFILE=1 (read once at init). When set,
    // `flush_locked` prints, per submitted batch, the time spent recording dispatches, in
    // queue_submit, in the fence wait, plus the dispatch and emitted-barrier counts; readbacks
    // print their map+copy time. Lets the 8060S show where the per-token milliseconds actually go.
    // Strictly zero-overhead when unset: the recording timer and all prints are behind this bool.
    profile: bool,
    // VK_KHR_push_descriptor device fns, present iff the driver advertises the extension (native
    // AMD/NV; typically absent on WSL/Dozen). When set, `dispatch` pushes buffer handles inline
    // into the command buffer via `vkCmdPushDescriptorSetKHR` instead of allocating + updating +
    // binding a descriptor set per op — three driver calls and two heap Vecs per dispatch collapse
    // to one recorded command, which is the dominant CPU cost on the decode hot path (the same op
    // graph, hundreds of dispatches x 28 layers, re-recorded every token). Set HANZO_VK_PUSH_DESC=0
    // to force the legacy alloc+update path. Pipelines' set layouts are created with the
    // PUSH_DESCRIPTOR_KHR flag exactly when this is `Some`, so the two paths never mix.
    push_descriptor: Option<ash::khr::push_descriptor::Device>,
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
    #[allow(dead_code)] // owns the pool that `cmd` is allocated from; freed with the device
    cpool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    dpool: vk::DescriptorPool,
    // Is `cmd` currently open with recorded-but-unsubmitted dispatches?
    recording: bool,
    // Dispatches recorded into `cmd` since it was begun (bounded by BATCH_CAP).
    n: u32,
    // Hazard tracking for selective barriers. Holds the output buffers written by dispatches
    // recorded since the last memory barrier in the current command buffer. A new dispatch needs
    // a barrier only if it READS a buffer in this set (a genuine read-after-write on data produced
    // earlier in this same batch); independent dispatches (disjoint buffers, or that only read
    // weights/inputs uploaded in an earlier already-fenced batch) record back-to-back with no
    // barrier, so the GPU can overlap their fixed launch/drain overhead. Cleared on each emitted
    // barrier and at the start of every batch. See `dispatch` for the correctness argument (WAW/WAR
    // can't occur within a batch because the buffer pool only recycles handles across fences).
    written_since_barrier: std::collections::HashSet<vk::Buffer>,
    // Per-batch profiling accumulators (HANZO_VK_PROFILE=1). `record_ns` is the wall time spent in
    // the dispatch recording path (descriptor push/update + cmd_dispatch + barrier bookkeeping)
    // since the batch began; `barriers` counts memory barriers emitted this batch. Both reset per
    // batch and are read by `flush_locked` when profiling is on. Zero-overhead otherwise (the
    // recording timer is only sampled when the device's `profile` flag is set).
    record_ns: u128,
    barriers: u32,
    // Per-batch dispatch histogram by kernel name (HANZO_VK_PROFILE=1 only). Lets flush_locked
    // print which kernels dominate the per-token dispatch/barrier count so fusion targets the
    // right ops. Reset per batch; empty/untouched when profiling is off.
    op_counts: std::collections::HashMap<&'static str, u32>,
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
    // Is `memory` HOST_VISIBLE (directly CPU-mappable)? False when the buffer was placed in a
    // DEVICE_LOCAL-only heap (the large GTT pool on this UMA APU), in which case uploads/readbacks
    // go through a transient host-visible staging buffer + a GPU copy instead of a direct map.
    host_visible: bool,
    device: VulkanDevice,
}

impl std::fmt::Debug for VulkanStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanStorage(count={}, dtype={:?})",
            self.count, self.dtype
        )
    }
}

// Deferred-safe buffer pool. Buffers are never freed inline: deferred dispatch may still hold a
// buffer's handle in an unflushed command buffer, so freeing on drop would be use-after-free. On
// drop a buffer parks in `pending`; after the next flush+fence (the awaited batch is provably done
// on the GPU) `reclaim` moves it to the size-keyed `free` list, where `raw_buffer` reuses it. This
// bounds device memory to the peak working set and reuses buffers across tokens, instead of leaking
// every allocation (which OOM'd a full unquantized forward on Dozen's ~8GB heap).
// Pool entries carry whether their backing memory is HOST_VISIBLE (mappable from the CPU). With the
// device-memory placement strategy a large buffer may be DEVICE_LOCAL-only, so reuse must preserve
// this flag — the upload/readback path branches on it (direct map vs staging copy).
#[derive(Clone, Copy)]
struct PooledBuf {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    host_visible: bool,
}

// Size-class for the reuse pool. Decode attention allocates transient tensors whose size grows by
// one token each step (score [b,h,1,cur_len], softmax, att@v); keying `free` by exact byte size
// means every token requests a never-before-seen size, so nothing is ever reused and each token
// leaks a fresh buffer -> O(seq^2) device memory -> OOM on long generations. That LEAK is bounded by
// the idle-pool cap below (reclaim() destroys idle buffers over POOL_FREE_CAP_BYTES); the cap alone
// is sufficient to prevent the OOM.
//
// CORRECTNESS: keys MUST be exact byte size. Power-of-two bucketing (rounding a >64KiB request up to
// the next pow2 so a later differently-sized request can reuse the buffer) was tried (commit
// d9b70910) and silently CORRUPTED inference: a reused bucket buffer is physically larger than the
// current logical tensor, and a consumer reads into the stale tail [logical, physical), producing
// garbled (non-deterministic) logits on EVERY model (0.6B and 8B both reproduced; CPU + native
// llama.cpp Vulkan stayed coherent on the same GPU+model, proving it was this pool, not the
// GPU/driver/model). The invariant the bucketing assumed ("kernels touch only the first n elems via
// their push-constant count, tail unused") does not hold somewhere on the readback/copy path. Exact
// keys keep physical == logical so there is never a stale tail. Re-enabling cross-size reuse for the
// growing attention-score buffers is a PERF follow-up (find the physical-size-reading consumer and
// make it honor the logical count, or zero reused buffers) -- not a correctness one; the cap below
// already bounds memory.
const POOL_EXACT_MAX: u64 = u64::MAX; // always exact: pow2 bucketing corrupts inference (see above).

const fn pool_bucket(bytes: u64) -> u64 {
    let b = if bytes < 4 { 4 } else { bytes };
    if b <= POOL_EXACT_MAX {
        b
    } else {
        b.next_power_of_two()
    }
}

// Cap the idle (free, unreferenced) pool so a workload that touches many distinct large size-classes
// can't retain them all forever. reclaim() destroys real device buffers once free exceeds this. The
// peak working set of a forward fits well under this; it only bounds the long tail. ~12 GiB.
const POOL_FREE_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;

#[derive(Default)]
struct BufPool {
    pending: Vec<(u64, PooledBuf)>,
    free: HashMap<u64, Vec<PooledBuf>>,
    // Sum of bucket sizes currently held in `free` (idle, reusable). Tracked so reclaim can enforce
    // POOL_FREE_CAP_BYTES without walking the whole map.
    free_bytes: u64,
}

// drop: park (size, buffer, memory) for reuse after the next fence. No Vulkan calls here, just
// bookkeeping, so it's cheap and can't race the GPU.
impl Drop for VulkanStorage {
    fn drop(&mut self) {
        if self.buffer == vk::Buffer::null() {
            return;
        }
        // Park under the bucket key (raw_buffer allocated this buffer at its bucket size), so the
        // physical buffer matches the key a later same-bucket request looks up.
        let bytes = pool_bucket((self.count * self.dtype.size_in_bytes()) as u64);
        if let Ok(mut pool) = self.device.inner.bufpool.lock() {
            pool.pending.push((
                bytes,
                PooledBuf {
                    buffer: self.buffer,
                    memory: self.memory,
                    host_visible: self.host_visible,
                },
            ));
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

    /// Whether the device has hardware int8 4x8-packed-signed dot product (and it is gated on). When
    /// true, the Q8_0 prefill (M>1) path should use [`matmul_q8_dp4a_gpu`] instead of [`matmul_q8_gpu`].
    pub fn int_dot8(&self) -> bool {
        self.inner.int_dot8
    }

    /// Whether a Q8 prefill GEMM of these dims should take the fp16 matrix-core (coopmat) path:
    /// device has a 16x16x16 coopmat config, it is enabled, every tile dim is a multiple of 16, and
    /// the f16 weight + activation scratch fits. Far higher arithmetic intensity than the quantized
    /// matmul (which only reuses each weight over 8 rows), closing the prefill gap to llama.cpp.
    pub fn coopmat_q8_ok(&self, m: usize, nout: usize, k: usize) -> bool {
        // Opt-in (HANZO_VK_Q8_COOPMAT=1): the f16 weight cache doubles resident weight VRAM (Q8 +
        // f16), which on this UMA part can push the decode matvec onto slower memory. Off by default
        // until the prefill win outweighs that; the path itself is correct (coherent output).
        std::env::var("HANZO_VK_Q8_COOPMAT").map(|v| v != "0").unwrap_or(false)
            && self.inner.cm_use
            && matches!(self.coopmat_info(), Some((16, 16, 16)))
            && m % 16 == 0
            && nout % 16 == 0
            && k % 16 == 0
            && self.scratch_fits(((m * k * 2) as u64).saturating_add((nout * k * 2) as u64))
    }

    /// Quantize `W[nout x k]` (row-major fp32) to Q8_0 and upload it to the GPU once. Per 32-block:
    /// one fp16 scale + 32 int8, packed into 9 u32. Returns the device buffer to reuse across many
    /// matvecs (weights are constant during decode). `k` must be a multiple of 32.
    pub fn quantize_q8(&self, w: &[f32], nout: usize, k: usize) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
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

    /// Upload an already-Q8_0-quantized weight to the GPU verbatim. `data` is the GGUF `BlockQ8_0`
    /// bytes (34 B/block = f16 scale + 32 int8), repacked LOSSLESLY into the kernel's 9-u32 layout
    /// (scale in u32[0] low half, 32 int8 four-per-word in u32[1..9]) -- the same layout
    /// [`quantize_q8`] emits, so the matvec/matmul decode is identical and no f32 round-trip is
    /// needed. This is the MoE-bank path: a Q8_0 [E,n,k] expert bank uploads once with `nout = E*n`
    /// and stays quantized. `data` must be exactly `nout * (k/32) * 34` bytes; `k` a multiple of 32.
    pub fn quantize_q8_blocks(&self, data: &[u8], nout: usize, k: usize) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("quantize_q8_blocks: k must be a multiple of 32, got {k}");
        }
        let nblocks = k / 32;
        let want = nout * nblocks * 34;
        if data.len() != want {
            crate::bail!(
                "quantize_q8_blocks: data len {} != {nout}*{nblocks}*34 = {want}",
                data.len()
            );
        }
        // Source block: bytes [0,2) = f16 scale (LE), bytes [2,34) = 32 int8. Repack each block into
        // 9 u32: u32[0] low half = the scale bits verbatim; u32[1..9] = the 32 int8, 4 per word.
        let mut packed = vec![0u32; nout * nblocks * 9];
        for blk in 0..nout * nblocks {
            let src = &data[blk * 34..blk * 34 + 34];
            let o = blk * 9;
            packed[o] = u16::from_le_bytes([src[0], src[1]]) as u32;
            for j in 0..8 {
                let b = 2 + j * 4;
                packed[o + 1 + j] = u32::from_le_bytes([src[b], src[b + 1], src[b + 2], src[b + 3]]);
            }
        }
        self.upload_u32(&packed)
    }

    /// Q8_0 matrix-vector: `y[nout] = Wq * x[k]` where `Wq` came from [`quantize_q8`]. The kernel
    /// reads weights at ~1.125 bytes/elem instead of 4 — the bandwidth lever for memory-bound
    /// decode on this APU. Expect ~1e-2 error (fp16-scale int8 quantization).
    pub fn matvec_q8(
        &self,
        wq: &VulkanStorage,
        x: &[f32],
        nout: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q8: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q8_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Q8_0 matvec with both operands already on the GPU: `y[nout] = Wq * x[k]`, no host round-trip.
    /// This is the engine decode path -- weights stay quantized in VRAM (~1.125 B/elem) instead of
    /// dequantizing to f32, so decode reads ~3.5x less memory (the bandwidth lever vs llama.cpp).
    pub fn matvec_q8_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matvec_q8_gpu_off(wq, x, nout, k, 0)
    }

    /// Fused single-query (decode) attention: `out[B,H,1,D] = softmax(scale * Q.Kᵀ) · V`, GQA-aware,
    /// in ONE dispatch (online softmax). Collapses the repeat_kv + QKᵀ bmm + softmax + ·V bmm +
    /// contiguous chain (~10 dispatches/layer) that dominates decode. Inputs row-major contiguous f32:
    /// q:[B,H,1,D], k/v:[B,Hkv,L,D]. Requires head_dim a power of two <= 128 (matches the kernel's
    /// 128-wide workgroup + shared-memory dot-product reduction); callers without that shape keep the
    /// eager bmm path.
    pub fn attn_decode_gpu(
        &self,
        q: &VulkanStorage,
        k: &VulkanStorage,
        v: &VulkanStorage,
        b: usize,
        h: usize,
        hkv: usize,
        l: usize,
        d: usize,
        scale: f32,
    ) -> Result<VulkanStorage> {
        if d == 0 || d > 128 || !d.is_power_of_two() {
            crate::bail!("attn_decode_gpu: head_dim {d} must be a power of two in 1..=128");
        }
        if h % hkv != 0 {
            crate::bail!("attn_decode_gpu: H {h} not a multiple of Hkv {hkv}");
        }
        let out = self.alloc_f32(b * h * d)?;
        let push = push_u32(&[
            b as u32,
            h as u32,
            hkv as u32,
            l as u32,
            d as u32,
            scale.to_bits(),
        ]);
        self.dispatch(
            "attn_decode",
            &[q.buffer, k.buffer, v.buffer, out.buffer],
            &push,
            ((b * h) as u32, 1, 1),
        )?;
        Ok(out)
    }

    /// Q8_0 matvec into a slice of `wq` starting at `woff` u32 words: `y[nout] = Wq[woff..] * x[k]`.
    /// `woff == 0` is bit-identical to [`matvec_q8_gpu`]; a non-zero offset selects one expert's row
    /// block of a resident MoE bank (`woff = e * nout * (k/32) * 9`), so the whole [E,n,k] Q8 bank
    /// stays uploaded once and each routed expert reads only its own slice (no per-token re-upload).
    pub fn matvec_q8_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if x.count < k {
            crate::bail!("matvec_q8_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        let push = push_u32(&[nout as u32, k as u32, woff as u32]);
        if self.inner.subgroup_matvec {
            // Subgroup-reduced kernel: each subgroup streams ROWS_PER_SG (=4, must match the shader)
            // output rows, reusing each staged x-block across them. A workgroup of WG1D (64)
            // invocations holds WG1D/subgroup_size subgroups, so it produces (subgroups * 4) rows;
            // dispatch ceil(nout / rows_per_wg) workgroups.
            const ROWS_PER_SG: u32 = 4;
            let rows_per_wg = (WG1D / self.inner.subgroup_size).max(1) * ROWS_PER_SG;
            self.dispatch(
                "mul_mat_vec_q8_sg",
                &[wq.buffer, x.buffer, out.buffer],
                &push,
                ((nout as u32).div_ceil(rows_per_wg), 1, 1),
            )?;
        } else {
            self.dispatch(
                "mul_mat_vec_q8",
                &[wq.buffer, x.buffer, out.buffer],
                &push,
                ((nout as u32).div_ceil(WG1D), 1, 1),
            )?;
        }
        Ok(out)
    }

    /// Upload Q4_K weights to the GPU once, verbatim in the GGUF super-block layout (144 B = 36 u32
    /// per 256-weight block, `nout * k/256` blocks total). No requantize: the bytes are the same
    /// blocks the CPU `k_quants::BlockQ4K` holds, so the in-shader decode matches the CPU dequant
    /// exactly. Returns the device buffer to reuse across matvecs. `data` must be exactly the
    /// `nout * (k/256) * 144` packed block bytes; `k` must be a multiple of 256.
    pub fn quantize_q4k(&self, data: &[u8], nout: usize, k: usize) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("quantize_q4k: k must be a multiple of 256, got {k}");
        }
        let nblocks = k / 256;
        let want = nout * nblocks * 144;
        if data.len() != want {
            crate::bail!(
                "quantize_q4k: data len {} != {nout}*{nblocks}*144 = {want}",
                data.len()
            );
        }
        // Reinterpret the packed block bytes as u32 words for the kernel's `uint w[]` binding. GGUF
        // blocks are 144 bytes (a multiple of 4), so the length is u32-aligned by construction.
        let words: &[u32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) };
        self.upload_u32(words)
    }

    /// Q4_K matrix-vector: `y[nout] = Wq * x[k]` where `Wq` came from [`quantize_q4k`]. Reads weights
    /// at ~4.5 bits/elem instead of 32 -- the bandwidth lever for memory-bound decode on this APU.
    /// Decode matches the CPU `BlockQ4K::to_float` so expect ~1e-3 relative error vs CPU f32 (the
    /// quantization error is already baked into the stored blocks).
    pub fn matvec_q4k(
        &self,
        wq: &VulkanStorage,
        x: &[f32],
        nout: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q4k: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q4k_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Q4_K matvec with both operands already on the GPU: `y[nout] = Wq * x[k]`, no host round-trip.
    /// Weights stay quantized in VRAM (~4.5 bits/elem) instead of dequantizing to f32, so decode
    /// reads ~7x less weight memory. One invocation per output row (scalar kernel).
    pub fn matvec_q4k_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matvec_q4k_gpu_off(wq, x, nout, k, 0)
    }

    /// Q4_K matvec into a slice of `wq` starting at `woff` u32 words: `y[nout] = Wq[woff..] * x[k]`.
    /// `woff == 0` is bit-identical to [`matvec_q4k_gpu`]; a non-zero offset selects one expert's row
    /// block of a resident MoE bank (`woff = e * nout * (k/256) * 36`), so the whole [E,n,k] Q4_K
    /// bank stays uploaded once and each routed expert reads only its own slice.
    pub fn matvec_q4k_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("matvec_q4k_gpu: k must be a multiple of 256, got {k}");
        }
        if x.count < k {
            crate::bail!("matvec_q4k_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        let push = push_u32(&[nout as u32, k as u32, woff as u32]);
        self.dispatch(
            "mul_mat_vec_q4k",
            &[wq.buffer, x.buffer, out.buffer],
            &push,
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Kernel name + dispatch grid for a fused grouped (MoE) matvec producing `m*nout` outputs. When
    /// the device supports subgroup arithmetic (`subgroup_matvec`, the same gate q8 uses) this selects
    /// the `_sg` variant — ONE SUBGROUP per (slot,col) output, lanes striding that row's blocks for
    /// coalesced reads — and a workgroup of WG1D invocations holds `WG1D/subgroup_size` subgroups, so
    /// it yields that many outputs; dispatch `ceil(m*nout / rows_per_wg)` workgroups. Otherwise the
    /// scalar kernel (one invocation per output, `ceil(m*nout / WG1D)` workgroups). `base` is the
    /// scalar kernel name; the `_sg` literal is matched (not formatted) to keep it `'static`.
    fn matvec_id_grid(
        &self,
        base: &'static str,
        m: usize,
        nout: usize,
    ) -> (&'static str, (u32, u32, u32)) {
        let total = (m * nout) as u32;
        if self.inner.subgroup_matvec {
            let sg = match base {
                "mul_mat_vec_id_q4k" => "mul_mat_vec_id_q4k_sg",
                "mul_mat_vec_id_q8" => "mul_mat_vec_id_q8_sg",
                other => other,
            };
            // subgroup_size is >=2 (checked at init) and <=64 in practice, so rows_per_wg is [1, 32].
            let rows_per_wg = (WG1D / self.inner.subgroup_size).max(1);
            (sg, (total.div_ceil(rows_per_wg), 1, 1))
        } else {
            (base, (total.div_ceil(WG1D), 1, 1))
        }
    }

    /// Fused grouped Q4_K matvec for resident-bank MoE: ONE dispatch computes `y[m*nout]` where
    /// `y[slot*nout + col] = dequant(bank[ids[slot]])[col, :] . x_flat[slot*k ..]`. `ids` is a GPU u32
    /// buffer of length `m` (the routed expert per slot), `x_flat` is `[m, k]` f32 contiguous, `bank`
    /// is the verbatim [E,n,k] Q4_K bank, `per_expert_words = n * (k/256) * 36`. Replaces the
    /// per-expert index_select/matvec/index_add loop -- and its routing-ids GPU->CPU sync -- with one
    /// kernel reading ids straight from VRAM. Decode is bit-identical to [`matvec_q4k_gpu_off`] (same
    /// in-shader block decode). Dispatches the subgroup-reduced `_sg` variant when supported (one
    /// subgroup per (slot,col)), else the scalar kernel (one invocation per output). See `matvec_id_grid`.
    #[allow(clippy::too_many_arguments)]
    pub fn mul_mat_vec_id_q4k_gpu(
        &self,
        bank: &VulkanStorage,
        x_flat: &VulkanStorage,
        ids: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
        per_expert_words: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("mul_mat_vec_id_q4k_gpu: k must be a multiple of 256, got {k}");
        }
        if x_flat.count < m * k {
            crate::bail!(
                "mul_mat_vec_id_q4k_gpu: x_flat count {} < m*k {}",
                x_flat.count,
                m * k
            );
        }
        if ids.count < m {
            crate::bail!("mul_mat_vec_id_q4k_gpu: ids count {} < m {m}", ids.count);
        }
        let out = self.alloc_f32(m * nout)?;
        let push = push_u32(&[nout as u32, k as u32, per_expert_words as u32, m as u32]);
        // Y is binding 2, not the last buffer (IDS is 3); name it so the RAW hazard tracker keys on
        // the buffer the kernel actually writes. Subgroup variant when available (one subgroup per
        // output element, coalesced reads + subgroupAdd); else the scalar fallback.
        let (kernel, groups) = self.matvec_id_grid("mul_mat_vec_id_q4k", m, nout);
        self.dispatch_out(
            kernel,
            &[bank.buffer, x_flat.buffer, out.buffer, ids.buffer],
            2,
            &push,
            groups,
        )?;
        Ok(out)
    }

    /// Fused grouped Q8_0 matvec for resident-bank MoE: ONE dispatch computes `y[m*nout]` where
    /// `y[slot*nout + col] = dequant(bank[ids[slot]])[col, :] . x_flat[slot*k ..]`. Q8_0 analog of
    /// [`mul_mat_vec_id_q4k_gpu`]; `per_expert_words = n * (k/32) * 9`. Decode is bit-identical to
    /// [`matvec_q8_gpu_off`]. Dispatches the subgroup-reduced `_sg` variant when supported (one
    /// subgroup per (slot,col)), else the scalar kernel (one invocation per output). See `matvec_id_grid`.
    #[allow(clippy::too_many_arguments)]
    pub fn mul_mat_vec_id_q8_gpu(
        &self,
        bank: &VulkanStorage,
        x_flat: &VulkanStorage,
        ids: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
        per_expert_words: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("mul_mat_vec_id_q8_gpu: k must be a multiple of 32, got {k}");
        }
        if x_flat.count < m * k {
            crate::bail!(
                "mul_mat_vec_id_q8_gpu: x_flat count {} < m*k {}",
                x_flat.count,
                m * k
            );
        }
        if ids.count < m {
            crate::bail!("mul_mat_vec_id_q8_gpu: ids count {} < m {m}", ids.count);
        }
        let out = self.alloc_f32(m * nout)?;
        let push = push_u32(&[nout as u32, k as u32, per_expert_words as u32, m as u32]);
        let (kernel, groups) = self.matvec_id_grid("mul_mat_vec_id_q8", m, nout);
        self.dispatch_out(
            kernel,
            &[bank.buffer, x_flat.buffer, out.buffer, ids.buffer],
            2,
            &push,
            groups,
        )?;
        Ok(out)
    }

    /// Q8_0 matrix-matrix (prefill): `y[m, nout] = x[m, k] * Wq^T`, weights stay quantized in VRAM
    /// (same blocks as [`quantize_q8`] / [`matvec_q8_gpu`]). Decodes each weight block once and reuses
    /// it across a tile of up to [`MATMUL_Q_MAX_M`] rows, so weight memory traffic matches a single
    /// matvec per output column instead of dequantizing the whole weight to f32 every forward. `x`
    /// must be a contiguous `[m, k]` device buffer; returns `[m, nout]` row-major.
    pub fn matmul_q8_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matmul_q8_gpu_off(wq, x, m, nout, k, 0)
    }

    /// Q8_0 prefill matmul into a slice of `wq` starting at `woff` u32 words. `woff == 0` is
    /// bit-identical to [`matmul_q8_gpu`]; a non-zero offset selects one expert's weight block of a
    /// resident MoE bank (`woff = e * nout * (k/32) * 9`) so a prefill expert with M>1 routed rows
    /// runs one banked matmul without re-uploading the weight.
    pub fn matmul_q8_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("matmul_q8_gpu: k must be a multiple of 32, got {k}");
        }
        if x.count < m * k {
            crate::bail!("matmul_q8_gpu: x count {} < m*k {}", x.count, m * k);
        }
        let out = self.alloc_f32(m * nout)?;
        let cols = (nout as u32).div_ceil(WG1D);
        let mut m0 = 0usize;
        while m0 < m {
            let mcount = (m - m0).min(MATMUL_Q_MAX_M);
            let push = push_u32(&[m0 as u32, mcount as u32, nout as u32, k as u32, woff as u32]);
            self.dispatch(
                "mul_mat_q8",
                &[wq.buffer, x.buffer, out.buffer],
                &push,
                (cols, 1, 1),
            )?;
            m0 += mcount;
        }
        Ok(out)
    }

    /// Q8_0 prefill matmul via hardware int8 dp4a -- the compute-bound prefill lever. Same result as
    /// [`matmul_q8_gpu`] (`y[m, nout] = x[m, k] * Wq^T`) but instead of decoding each weight to f32 and
    /// doing f32 MACs, it quantizes the activation tile to int8 ONCE ([`quantize_act_q8`] -> q8_1-style
    /// codes + per-block scales) then dispatches [`mul_mat_q8_dp4a`], which uses the accelerated
    /// `OpSDotAccSat` 4x8-packed signed dot vs the int8 weights. Only valid when [`int_dot8`] is true;
    /// callers should fall back to [`matmul_q8_gpu`] otherwise (this fn does NOT re-check, so the
    /// activation quant runs unconditionally -- gate at the call site). Drop-in for [`matmul_q8_gpu`].
    pub fn matmul_q8_dp4a_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matmul_q8_dp4a_gpu_off(wq, x, m, nout, k, 0)
    }

    /// [`matmul_q8_dp4a_gpu`] into a slice of `wq` starting at `woff` u32 words (`woff == 0` matches the
    /// plain form; `woff = e * nout * (k/32) * 9` selects expert `e` of a resident Q8 MoE bank). Mirrors
    /// [`matmul_q8_gpu_off`] so it is a drop-in on the banked prefill path.
    pub fn matmul_q8_dp4a_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("matmul_q8_dp4a_gpu: k must be a multiple of 32, got {k}");
        }
        if x.count < m * k {
            crate::bail!("matmul_q8_dp4a_gpu: x count {} < m*k {}", x.count, m * k);
        }
        let nblocks = k / 32;
        // Quantize the whole activation tile to int8 once (reused across all nout columns): xq is
        // m*nblocks*8 u32 (32 int8/block, 4 per word), xs is m*nblocks f32 block scales. quantize_act_q8
        // also writes xsum (dequant block sums) for the Q4_K dp4a min correction; the Q8_0 weight is
        // symmetric (no min term) so xsum is unused here, but the kernel declares 4 bindings, so bind a
        // throwaway xsum to keep the binding count consistent with the Q4_K dp4a caller.
        let xq = self.alloc_u32(m * nblocks * 8)?;
        let xs = self.alloc_f32(m * nblocks)?;
        let xsum = self.alloc_f32(m * nblocks)?;
        let qpush = push_u32(&[m as u32, k as u32]);
        // quantize_act_q8 WRITES xq(1), xs(2), xsum(3); mark all three so a later in-batch reader gets
        // the RAW barrier. The default dispatch() marks only the last binding (xsum) -- but mul_mat_q8_dp4a
        // reads xq+xs and NOT xsum, so without this the matmul races the still-running quantizer and reads
        // unwritten int8 activations (garbage). The Q4_K consumer reads xsum and so was accidentally safe.
        self.dispatch_outs(
            "quantize_act_q8",
            &[x.buffer, xq.buffer, xs.buffer, xsum.buffer],
            &[1, 2, 3],
            &qpush,
            (((m * nblocks) as u32).div_ceil(WG1D), 1, 1),
        )?;
        let out = self.alloc_f32(m * nout)?;
        let cols = (nout as u32).div_ceil(WG1D);
        let mut m0 = 0usize;
        while m0 < m {
            let mcount = (m - m0).min(MATMUL_Q_MAX_M);
            let push = push_u32(&[m0 as u32, mcount as u32, nout as u32, k as u32, woff as u32]);
            self.dispatch(
                "mul_mat_q8_dp4a",
                &[wq.buffer, xq.buffer, xs.buffer, out.buffer],
                &push,
                (cols, 1, 1),
            )?;
            m0 += mcount;
        }
        Ok(out)
    }

    /// Persistent f16 device buffer (2 B/elem), owned like [`alloc_f32`] so it survives across
    /// forwards (the dequantized-weight cache for the coopmat prefill path), unlike the transient
    /// scratch from [`alloc_f16`].
    fn alloc_f16_storage(&self, count: usize) -> Result<VulkanStorage> {
        let (buffer, memory, host_visible) = unsafe { self.raw_buffer((count * 2).max(4) as u64)? };
        Ok(VulkanStorage {
            buffer,
            memory,
            count,
            dtype: DType::F16,
            host_visible,
            device: self.clone(),
        })
    }

    /// Dequantize a resident Q8_0 weight (9 u32 / 32-block, row-major `[nout, k]`) to a fresh f16
    /// `[nout, k]` device buffer. Done once and cached so prefill can run the matrix-core GEMM
    /// instead of the bandwidth-bound quantized matmul that only reuses each weight over 8 rows.
    pub fn dequant_q8_to_f16(&self, wq: &VulkanStorage, nout: usize, k: usize) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("dequant_q8_to_f16: k must be a multiple of 32, got {k}");
        }
        let nblocks_total = nout * k / 32;
        let out = self.alloc_f16_storage(nout * k)?;
        self.dispatch(
            "dequant_q8_f16",
            &[wq.buffer, out.buffer],
            &push_u32(&[nblocks_total as u32]),
            ((nblocks_total as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Q8_0 prefill via the fp16 matrix-core (coopmat) GEMM: `y[m, nout] = x[m, k] * W[nout, k]^T`,
    /// `w16` the cached f16 weight from [`dequant_q8_to_f16`]. Casts the activations to f16 and runs
    /// `bmm_coopmat_rb_nt`, reusing each loaded fragment across the whole tile grid (vs the 8-row
    /// reuse of the quantized matmul). Caller guarantees `m % 16 == 0 && nout % 16 == 0 && k % 16 == 0`
    /// and that coopmat is available.
    pub fn matmul_q8_coopmat_gpu(
        &self,
        w16: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        if x.count < m * k {
            crate::bail!("matmul_q8_coopmat_gpu: x count {} < m*k {}", x.count, m * k);
        }
        let (a16, a16_mem, a16_hv, a16_bytes) = self.alloc_f16(m * k)?;
        self.dispatch(
            "cast_f2h",
            &[x.buffer, a16],
            &push_u32(&[(m * k) as u32]),
            (((m * k) as u32).div_ceil(WG1D), 1, 1),
        )?;
        let out = self.alloc_f32(m * nout)?;
        let push = push_u32(&[1u32, m as u32, k as u32, nout as u32]);
        let mt = (m / 16) as u32;
        let nt_tiles = (nout / 16) as u32;
        let groups = (nt_tiles.div_ceil(4), mt.div_ceil(4), 1u32);
        self.dispatch("bmm_coopmat_rb_nt", &[a16, w16.buffer, out.buffer], &push, groups)?;
        self.free_scratch(a16_bytes, a16, a16_mem, a16_hv);
        Ok(out)
    }

    /// Q8_0 prefill via the quant-aware coopmat GEMM (`mul_mm_q8`): `y[m, nout] = x[m, k] * Wq^T`.
    /// Reads the Q8 weight ONCE, dequantizing each block to f16 in shared memory before the matrix-
    /// core MulAdd -- no whole-weight f16 materialization and no per-matmul activation cast pass (vs
    /// [`matmul_q8_coopmat_gpu`]). 64x64 output tile per workgroup; caller guarantees m,n,k % 16 == 0.
    pub fn matmul_q8_mm_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        if x.count < m * k {
            crate::bail!("matmul_q8_mm_gpu: x count {} < m*k {}", x.count, m * k);
        }
        if !k.is_multiple_of(32) {
            crate::bail!("matmul_q8_mm_gpu: k must be a multiple of 32, got {k}");
        }
        let out = self.alloc_f32(m * nout)?;
        let push = push_u32(&[m as u32, k as u32, nout as u32, 0u32]); // woff = 0 (plain 2D weight)
        let groups = ((nout as u32).div_ceil(64), (m as u32).div_ceil(64), 1u32);
        self.dispatch("mul_mm_q8", &[wq.buffer, x.buffer, out.buffer], &push, groups)?;
        Ok(out)
    }

    /// Q8_0 prefill via the shared-memory-tiled int8-dot GEMM (`mul_mm_q8_dp4a`): `y[m, nout] =
    /// x[m, k] * Wq^T`. Quantizes the activation tile to int8 once, then a single dispatch where each
    /// 64x64 output tile stages its weight + activation sub-tiles in shared memory (Q8 read ONCE,
    /// reused across all 64 rows) and 256 threads each compute a 4x4 register tile. Unlike the
    /// per-column [`matmul_q8_dp4a_gpu`] this gets weight reuse AND occupancy together. m,n % 16, k % 32.
    pub fn matmul_q8_mm_dp4a_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(32) {
            crate::bail!("matmul_q8_mm_dp4a_gpu: k must be a multiple of 32, got {k}");
        }
        if x.count < m * k {
            crate::bail!("matmul_q8_mm_dp4a_gpu: x count {} < m*k {}", x.count, m * k);
        }
        let nblocks = k / 32;
        let xq = self.alloc_u32(m * nblocks * 8)?;
        let xs = self.alloc_f32(m * nblocks)?;
        let xsum = self.alloc_f32(m * nblocks)?;
        let qpush = push_u32(&[m as u32, k as u32]);
        // quantize_act_q8 writes xq(1),xs(2),xsum(3); mark 1+2 so the GEMM gets the RAW barrier on the
        // int8 acts + scales it actually reads (xsum unused for symmetric Q8).
        self.dispatch_outs(
            "quantize_act_q8",
            &[x.buffer, xq.buffer, xs.buffer, xsum.buffer],
            &[1, 2],
            &qpush,
            (((m * nblocks) as u32).div_ceil(WG1D), 1, 1),
        )?;
        let out = self.alloc_f32(m * nout)?;
        let push = push_u32(&[m as u32, k as u32, nout as u32, 0u32]); // woff = 0
        // Tile-config sweep: one build, many spv variants selected at runtime (BM=16*TM, BN=16*TN).
        // Default = 8x4 (128x64), the sweep winner. HANZO_VK_MM_CFG picks another for A/B.
        let (kern, bm, bn): (&str, u32, u32) =
            match std::env::var("HANZO_VK_MM_CFG").as_deref() {
                Ok("4x4") => ("mul_mm_q8_dp4a", 64, 64),
                Ok("4x8") => ("mul_mm_q8_dp4a_4x8", 64, 128),
                Ok("8x8") => ("mul_mm_q8_dp4a_8x8", 128, 128),
                Ok("2x2") => ("mul_mm_q8_dp4a_2x2", 32, 32),
                Ok("6x6") => ("mul_mm_q8_dp4a_6x6", 96, 96),
                Ok("4x4n4") => ("mul_mm_q8_dp4a_4x4n4", 64, 64),
                Ok("8x2") => ("mul_mm_q8_dp4a_8x2", 128, 32),
                Ok("12x4") => ("mul_mm_q8_dp4a_12x4", 192, 64),
                Ok("16x2") => ("mul_mm_q8_dp4a_16x2", 256, 32),
                Ok("16x4") => ("mul_mm_q8_dp4a_16x4", 256, 64),
                _ => ("mul_mm_q8_dp4a_8x4", 128, 64),
            };
        let groups = ((nout as u32).div_ceil(bn), (m as u32).div_ceil(bm), 1u32);
        self.dispatch(kern, &[wq.buffer, xq.buffer, xs.buffer, out.buffer], &push, groups)?;
        Ok(out)
    }

    /// Q4_K matrix-matrix (prefill): `y[m, nout] = x[m, k] * Wq^T`, weights stay quantized in VRAM
    /// (verbatim GGUF Q4_K super-blocks, same decode as [`matvec_q4k_gpu`]). Decodes each weight value
    /// once and reuses it across a tile of up to [`MATMUL_Q_MAX_M`] rows, so weight memory traffic
    /// matches a single matvec per output column instead of dequantizing the whole weight to f32 every
    /// forward. `x` must be a contiguous `[m, k]` device buffer; returns `[m, nout]` row-major.
    pub fn matmul_q4k_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matmul_q4k_gpu_off(wq, x, m, nout, k, 0)
    }

    /// Q4_K prefill matmul into a slice of `wq` starting at `woff` u32 words. `woff == 0` is
    /// bit-identical to [`matmul_q4k_gpu`]; a non-zero offset selects one expert's weight block of a
    /// resident MoE bank (`woff = e * nout * (k/256) * 36`) so a prefill expert with M>1 routed rows
    /// runs one banked matmul without re-uploading the weight.
    pub fn matmul_q4k_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("matmul_q4k_gpu: k must be a multiple of 256, got {k}");
        }
        if x.count < m * k {
            crate::bail!("matmul_q4k_gpu: x count {} < m*k {}", x.count, m * k);
        }
        let out = self.alloc_f32(m * nout)?;
        let cols = (nout as u32).div_ceil(WG1D);
        let mut m0 = 0usize;
        while m0 < m {
            let mcount = (m - m0).min(MATMUL_Q_MAX_M);
            let push = push_u32(&[m0 as u32, mcount as u32, nout as u32, k as u32, woff as u32]);
            self.dispatch(
                "mul_mat_q4k",
                &[wq.buffer, x.buffer, out.buffer],
                &push,
                (cols, 1, 1),
            )?;
            m0 += mcount;
        }
        Ok(out)
    }

    /// Q4_K prefill matmul via hardware int8 dp4a -- the compute-bound prefill lever for Q4_K weights
    /// (attention projections + MoE gate/up experts in Q4_K_M/XL). Same result as [`matmul_q4k_gpu`]
    /// (`y[m, nout] = x[m, k] * Wq^T`) but instead of decoding each weight to f32 and doing f32 MACs,
    /// it quantizes the activation tile to int8 ONCE ([`quantize_act_q8`] -> codes + per-block scales +
    /// dequant block sums) then dispatches [`mul_mat_q4k_dp4a`], which dp4a's the 4-bit codes vs the
    /// int8 activations (`d1*xs*dot - m1*xsum` per 32-sub-block). Only valid when [`int_dot8`] is true;
    /// callers fall back to [`matmul_q4k_gpu`] otherwise (this fn does NOT re-check, so the activation
    /// quant runs unconditionally -- gate at the call site). Drop-in for [`matmul_q4k_gpu`].
    pub fn matmul_q4k_dp4a_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matmul_q4k_dp4a_gpu_off(wq, x, m, nout, k, 0)
    }

    /// [`matmul_q4k_dp4a_gpu`] into a slice of `wq` starting at `woff` u32 words (`woff == 0` matches
    /// the plain form; `woff = e * nout * (k/256) * 36` selects expert `e` of a resident Q4_K MoE bank).
    /// Mirrors [`matmul_q4k_gpu_off`] so it is a drop-in on the banked prefill path.
    pub fn matmul_q4k_dp4a_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        m: usize,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("matmul_q4k_dp4a_gpu: k must be a multiple of 256, got {k}");
        }
        if x.count < m * k {
            crate::bail!("matmul_q4k_dp4a_gpu: x count {} < m*k {}", x.count, m * k);
        }
        // Activation 32-blocks (the dp4a contraction granularity), k/32 per row. The Q4_K 32-weight
        // sub-blocks align 1:1 with these, so the same quantize_act_q8 output feeds both dp4a matmuls.
        let nb32 = k / 32;
        // Quantize the whole activation tile to int8 once (reused across all nout columns): xq is
        // m*nb32*8 u32 (32 int8/block, 4 per word), xs is m*nb32 f32 block scales, xsum is m*nb32 f32
        // dequant block sums (xs*sum(codes)) for the -m*xsum min correction.
        let xq = self.alloc_u32(m * nb32 * 8)?;
        let xs = self.alloc_f32(m * nb32)?;
        let xsum = self.alloc_f32(m * nb32)?;
        let qpush = push_u32(&[m as u32, k as u32]);
        // Mark all three outputs (xq, xs, xsum) so an in-batch consumer barriers correctly regardless of
        // which it reads -- see the matching note in matmul_q8_dp4a_gpu_off (the Q8 consumer reads xq+xs
        // but not xsum, so relying on the default last-binding-only marking misses the hazard).
        self.dispatch_outs(
            "quantize_act_q8",
            &[x.buffer, xq.buffer, xs.buffer, xsum.buffer],
            &[1, 2, 3],
            &qpush,
            (((m * nb32) as u32).div_ceil(WG1D), 1, 1),
        )?;
        let out = self.alloc_f32(m * nout)?;
        let cols = (nout as u32).div_ceil(WG1D);
        let mut m0 = 0usize;
        while m0 < m {
            let mcount = (m - m0).min(MATMUL_Q_MAX_M);
            let push = push_u32(&[m0 as u32, mcount as u32, nout as u32, k as u32, woff as u32]);
            self.dispatch(
                "mul_mat_q4k_dp4a",
                &[wq.buffer, xq.buffer, xs.buffer, xsum.buffer, out.buffer],
                &push,
                (cols, 1, 1),
            )?;
            m0 += mcount;
        }
        Ok(out)
    }

    /// Gated delta-rule recurrence on the GPU (hybrid GDN / Qwen3.6 mixer), mirroring the CUDA
    /// `gated_delta_rule_recurrence` math. All operands are f32 device buffers laid out exactly as
    /// the CUDA kernel expects: `q,k` `[BH,S,K]`, `v` `[BH,S,V]`, `g,beta` `[BH,S]`, `state` `[BH,K,V]`
    /// (updated in place). Returns the output `[BH,S,V]`. `seq_len==1` is the decode path (sequential
    /// `gdn_recurrence`); a longer prefill uses the chunked kernel when `k_dim<=128`, else the
    /// sequential one (its private state array bounds `k_dim<=256`, covering shipping GDN configs).
    /// One workgroup per (V-tile, batch*head); V-tile width is 64 (matches the kernels' BV).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_recurrence_gpu(
        &self,
        q: &VulkanStorage,
        k: &VulkanStorage,
        v: &VulkanStorage,
        g: &VulkanStorage,
        beta: &VulkanStorage,
        state: &VulkanStorage,
        bh: usize,
        seq_len: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<VulkanStorage> {
        if k_dim > 256 {
            crate::bail!("gdn_recurrence_gpu: k_dim {k_dim} exceeds the kernel bound of 256");
        }
        let out = self.alloc_f32(bh * seq_len * v_dim)?;
        let push = push_u32(&[seq_len as u32, k_dim as u32, v_dim as u32]);
        let bufs = [
            q.buffer, k.buffer, v.buffer, g.buffer, beta.buffer, state.buffer, out.buffer,
        ];
        let v_tiles = (v_dim as u32).div_ceil(WG1D);
        // Chunked prefill kernel bounds its shared key array at K<=128; fall back to the sequential
        // kernel (private array bound K<=256) for the rare wider config. Decode (seq_len==1) always
        // uses the sequential kernel -- there are no chunks to amortize over a single token.
        let kernel = if seq_len > 1 && k_dim <= 128 {
            "gdn_chunked"
        } else {
            "gdn_recurrence"
        };
        // The kernel writes BOTH `state` (binding 5, in place) and `out` (binding 6). Mark both
        // live so a later dispatch in this batch that reads either inserts the RAW barrier.
        self.dispatch_multi_out(kernel, &bufs, &[5, 6], &push, (v_tiles, bh as u32, 1))?;
        Ok(out)
    }

    /// Causal conv1d single-step update (GDN decode path), mirroring CUDA `causal_conv1d_update`.
    /// `x` `[B,conv_dim,1]`, `weight` `[conv_dim,kernel_size]`, `conv_state` `[B,conv_dim,kernel_size]`
    /// (shifted+updated in place). Returns the SiLU-activated output `[B,conv_dim,1]`. f32 on this
    /// backend (f16/bf16 GDN tensors are stored as f32). `kernel_size<=8` (the kernel's window bound).
    pub fn gdn_conv_update_gpu(
        &self,
        x: &VulkanStorage,
        weight: &VulkanStorage,
        conv_state: &VulkanStorage,
        batch_size: usize,
        conv_dim: usize,
        kernel_size: usize,
    ) -> Result<VulkanStorage> {
        if !(1..=8).contains(&kernel_size) {
            crate::bail!("gdn_conv_update_gpu: kernel_size {kernel_size} out of range 1..=8");
        }
        let out = self.alloc_f32(batch_size * conv_dim)?;
        let push = push_u32(&[batch_size as u32, conv_dim as u32, kernel_size as u32]);
        let bufs = [x.buffer, weight.buffer, conv_state.buffer, out.buffer];
        // Writes conv_state (binding 2, in place) and out (binding 3); track both for hazards.
        self.dispatch_multi_out(
            "gdn_conv_update",
            &bufs,
            &[2, 3],
            &push,
            ((conv_dim as u32).div_ceil(WG1D), batch_size as u32, 1),
        )?;
        Ok(out)
    }

    /// Causal conv1d over a full sequence (GDN prefill path), mirroring CUDA `causal_conv1d_full`
    /// plus `save_conv_state`. `x` `[B,conv_dim,S]`, `weight` `[conv_dim,kernel_size]`. Returns the
    /// SiLU output `[B,conv_dim,S]` and the trailing window saved into a fresh conv_state
    /// `[B,conv_dim,kernel_size]` (left zero-padded when `S<kernel_size`). f32 throughout.
    pub fn gdn_conv_full_gpu(
        &self,
        x: &VulkanStorage,
        weight: &VulkanStorage,
        batch_size: usize,
        conv_dim: usize,
        seq_len: usize,
        kernel_size: usize,
    ) -> Result<(VulkanStorage, VulkanStorage)> {
        let out = self.alloc_f32(batch_size * conv_dim * seq_len)?;
        let cs = self.alloc_f32(batch_size * conv_dim * kernel_size)?;
        let push_full = push_u32(&[
            batch_size as u32,
            conv_dim as u32,
            seq_len as u32,
            kernel_size as u32,
        ]);
        let total = (batch_size * conv_dim * seq_len) as u32;
        self.dispatch(
            "gdn_conv_full",
            &[x.buffer, weight.buffer, out.buffer],
            &push_full,
            (total.div_ceil(WG1D), 1, 1),
        )?;
        // Independent of the conv output (reads x, writes a disjoint buffer), so no barrier needed.
        self.dispatch(
            "gdn_conv_state_save",
            &[x.buffer, cs.buffer],
            &push_full,
            ((conv_dim as u32).div_ceil(WG1D), batch_size as u32, 1),
        )?;
        Ok((out, cs))
    }

    /// Fused GDN gating, mirroring CUDA `fused_gdn_gating`: `beta=sigmoid(b)`,
    /// `g=-exp(a_log)*softplus(a+dt_bias)`. `b,a` are `[total]`; `a_log,dt_bias` are per-head
    /// `[num_heads]` (indexed by `idx % num_heads`). Returns `(beta, g)`, each `[total]`. f32.
    pub fn gdn_gating_gpu(
        &self,
        b: &VulkanStorage,
        a: &VulkanStorage,
        a_log: &VulkanStorage,
        dt_bias: &VulkanStorage,
        total: usize,
        num_heads: usize,
    ) -> Result<(VulkanStorage, VulkanStorage)> {
        let beta = self.alloc_f32(total)?;
        let g = self.alloc_f32(total)?;
        let push = push_u32(&[total as u32, num_heads as u32]);
        let bufs = [
            b.buffer,
            a.buffer,
            a_log.buffer,
            dt_bias.buffer,
            beta.buffer,
            g.buffer,
        ];
        // Writes beta (binding 4) and g (binding 5); both are consumed downstream, track both.
        self.dispatch_multi_out(
            "gdn_gating",
            &bufs,
            &[4, 5],
            &push,
            ((total as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok((beta, g))
    }

    /// Like [`Self::dispatch_out`] but the kernel writes several output bindings (named by
    /// `out_idxs`); each is marked live in the in-batch hazard set so a later dispatch reading any of
    /// them gets the RAW barrier. Used by the GDN kernels (recurrence updates state AND writes output;
    /// gating writes beta AND g; conv-update updates state AND writes output). Shares `dispatch_out`'s
    /// recording path by issuing the dispatch with the first output index, then registering the rest.
    fn dispatch_multi_out(
        &self,
        name: &'static str,
        bufs: &[vk::Buffer],
        out_idxs: &[usize],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let first = out_idxs
            .first()
            .copied()
            .unwrap_or(bufs.len().saturating_sub(1));
        self.dispatch_out(name, bufs, first, push, groups)?;
        if out_idxs.len() > 1 {
            let mut s = self.inner.submitter.lock().unwrap();
            for &i in &out_idxs[1..] {
                if let Some(&buf) = bufs.get(i) {
                    s.written_since_barrier.insert(buf);
                }
            }
        }
        Ok(())
    }

    /// Upload Q5_K weights to the GPU once, verbatim in the GGUF super-block layout (176 B = 44 u32
    /// per 256-weight block, `nout * k/256` blocks total). No requantize: the bytes are the same
    /// blocks the CPU `k_quants::BlockQ5K` holds (d, dmin, 12 scale bytes, 32 qh bytes, 128 qs
    /// bytes), so the in-shader decode matches the CPU dequant exactly. `data` must be exactly the
    /// `nout * (k/256) * 176` packed block bytes; `k` must be a multiple of 256.
    pub fn quantize_q5k(&self, data: &[u8], nout: usize, k: usize) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("quantize_q5k: k must be a multiple of 256, got {k}");
        }
        let nblocks = k / 256;
        let want = nout * nblocks * 176;
        if data.len() != want {
            crate::bail!(
                "quantize_q5k: data len {} != {nout}*{nblocks}*176 = {want}",
                data.len()
            );
        }
        // Q5_K blocks are 176 bytes (a multiple of 4), so the buffer is u32-aligned by construction
        // and can be reinterpreted verbatim for the kernel's `uint w[]` binding.
        let words: &[u32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u32, data.len() / 4) };
        self.upload_u32(words)
    }

    /// Q5_K matrix-vector: `y[nout] = Wq * x[k]` where `Wq` came from [`quantize_q5k`]. Reads weights
    /// at ~5.5 bits/elem instead of 32 -- the bandwidth lever for memory-bound decode on this APU.
    /// Decode matches the CPU `BlockQ5K::to_float`.
    pub fn matvec_q5k(
        &self,
        wq: &VulkanStorage,
        x: &[f32],
        nout: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q5k: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q5k_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Q5_K matvec with both operands already on the GPU: `y[nout] = Wq * x[k]`, no host round-trip.
    /// Weights stay quantized in VRAM (~5.5 bits/elem) instead of dequantizing to f32, so decode
    /// reads ~6x less weight memory. One invocation per output row (scalar kernel).
    pub fn matvec_q5k_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matvec_q5k_gpu_off(wq, x, nout, k, 0)
    }

    /// Q5_K matvec into a slice of `wq` starting at `woff` u32 words: `y[nout] = Wq[woff..] * x[k]`.
    /// `woff == 0` is bit-identical to [`matvec_q5k_gpu`]; a non-zero offset selects one expert's row
    /// block of a resident MoE bank (`woff = e * nout * (k/256) * 44`).
    pub fn matvec_q5k_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("matvec_q5k_gpu: k must be a multiple of 256, got {k}");
        }
        if x.count < k {
            crate::bail!("matvec_q5k_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        let push = push_u32(&[nout as u32, k as u32, woff as u32]);
        self.dispatch(
            "mul_mat_vec_q5k",
            &[wq.buffer, x.buffer, out.buffer],
            &push,
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    /// Upload Q6_K weights to the GPU once. The CPU `k_quants::BlockQ6K` block is 210 bytes
    /// (ql[128], qh[64], scales[16] i8, d f16) which is NOT u32-aligned, so each block is repacked
    /// into a PADDED 212-byte (53 u32) stride here (trailing 2 bytes unused); the shader uses the
    /// same 53 u32 stride and byte-addressed reads, so the in-shader decode matches the CPU dequant
    /// exactly. `data` must be exactly the `nout * (k/256) * 210` packed block bytes; `k` must be a
    /// multiple of 256.
    pub fn quantize_q6k(&self, data: &[u8], nout: usize, k: usize) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("quantize_q6k: k must be a multiple of 256, got {k}");
        }
        let nblocks = k / 256;
        let total = nout * nblocks;
        let want = total * 210;
        if data.len() != want {
            crate::bail!(
                "quantize_q6k: data len {} != {nout}*{nblocks}*210 = {want}",
                data.len()
            );
        }
        // Repack 210-byte source blocks into 53-u32 (212-byte) padded blocks so the device buffer is
        // u32-aligned and every block starts on a u32 boundary. Trailing 2 bytes of each block are
        // zero pad and never read by the shader.
        let mut words = vec![0u32; total * 53];
        for blk in 0..total {
            let src = &data[blk * 210..blk * 210 + 210];
            let dst = &mut words[blk * 53..blk * 53 + 53];
            // Copy the 210 bytes into the low 210 bytes of the 212-byte (53 u32) padded block.
            let dst_bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u8, 53 * 4)
            };
            dst_bytes[..210].copy_from_slice(src);
        }
        self.upload_u32(&words)
    }

    /// Q6_K matrix-vector: `y[nout] = Wq * x[k]` where `Wq` came from [`quantize_q6k`]. Reads weights
    /// at ~6.5 bits/elem (incl. pad) instead of 32 -- the bandwidth lever for memory-bound decode on
    /// this APU. Decode matches the CPU `BlockQ6K::to_float`.
    pub fn matvec_q6k(
        &self,
        wq: &VulkanStorage,
        x: &[f32],
        nout: usize,
        k: usize,
    ) -> Result<Vec<f32>> {
        if x.len() != k {
            crate::bail!("matvec_q6k: x len {} != k {k}", x.len());
        }
        let xs = self.upload_f32(x)?;
        self.matvec_q6k_gpu(wq, &xs, nout, k)?.to_vec_f32()
    }

    /// Q6_K matvec with both operands already on the GPU: `y[nout] = Wq * x[k]`, no host round-trip.
    /// Weights stay quantized in VRAM (~6.5 bits/elem incl. pad) instead of dequantizing to f32, so
    /// decode reads ~5x less weight memory. One invocation per output row (scalar kernel).
    pub fn matvec_q6k_gpu(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
    ) -> Result<VulkanStorage> {
        self.matvec_q6k_gpu_off(wq, x, nout, k, 0)
    }

    /// Q6_K matvec into a slice of `wq` starting at `woff` u32 words: `y[nout] = Wq[woff..] * x[k]`.
    /// `woff == 0` is bit-identical to [`matvec_q6k_gpu`]; a non-zero offset selects one expert's row
    /// block of a resident MoE bank (`woff = e * nout * (k/256) * 53`, 53 u32 = padded Q6_K block).
    pub fn matvec_q6k_gpu_off(
        &self,
        wq: &VulkanStorage,
        x: &VulkanStorage,
        nout: usize,
        k: usize,
        woff: usize,
    ) -> Result<VulkanStorage> {
        if !k.is_multiple_of(256) {
            crate::bail!("matvec_q6k_gpu: k must be a multiple of 256, got {k}");
        }
        if x.count < k {
            crate::bail!("matvec_q6k_gpu: x count {} < k {k}", x.count);
        }
        let out = self.alloc_f32(nout)?;
        let push = push_u32(&[nout as u32, k as u32, woff as u32]);
        self.dispatch(
            "mul_mat_vec_q6k",
            &[wq.buffer, x.buffer, out.buffer],
            &push,
            ((nout as u32).div_ceil(WG1D), 1, 1),
        )?;
        Ok(out)
    }

    // Allocate a storage buffer of `bytes` bytes, returning it plus whether its memory is
    // HOST_VISIBLE (directly CPU-mappable). Placement follows the configured MemStrategy: on this
    // UMA APU a large buffer may be placed in a DEVICE_LOCAL-only heap (the big GTT pool) that the
    // CPU cannot map — callers then upload/read back through a staging buffer. Buffers carry
    // TRANSFER_SRC|TRANSFER_DST usage so that staging GPU copy is always legal.
    unsafe fn raw_buffer(&self, bytes: u64) -> Result<(vk::Buffer, vk::DeviceMemory, bool)> {
        // Allocate at the bucket size so every pooled buffer matches its `free` key exactly; a reused
        // buffer is then always physically >= the request (kernels touch only the first `n` elems via
        // their push-constant count, and descriptors bind WHOLE_SIZE, so the extra tail is unused).
        let bytes = pool_bucket(bytes);
        // Reuse a same-bucket buffer reclaimed from a completed batch before allocating fresh.
        {
            let mut pool = self.inner.bufpool.lock().unwrap();
            if let Some(p) = pool.free.get_mut(&bytes).and_then(Vec::pop) {
                pool.free_bytes = pool.free_bytes.saturating_sub(bytes);
                return Ok((p.buffer, p.memory, p.host_visible));
            }
        }
        let dev = self.dev();
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = dev.create_buffer(&info, None).map_err(vkerr)?;
        let req = dev.get_buffer_memory_requirements(buf);
        let idx = self
            .pick_memory_type(req.memory_type_bits, req.size)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "vulkan: no usable memory type for a {req_size}-byte buffer (type_bits={bits:#x}, strategy={strat:?})",
                    req_size = req.size,
                    bits = req.memory_type_bits,
                    strat = self.inner.mem_strategy,
                ))
            })?;
        let (mem, used_idx) = match dev.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(idx),
            None,
        ) {
            Ok(m) => (m, idx),
            Err(e) => {
                // The chosen heap accounting said the type was usable, but the driver still refused
                // (fragmentation, racing allocations, or a too-optimistic budget). Try every other
                // usable type, largest heap first, before giving up — this is the concrete OOM
                // recovery for the 18.6GB-model case where the first pick is the small host-visible
                // carveout and the real space is in the DEVICE_LOCAL heap.
                let mut fallback = None;
                for alt in self.usable_types_by_heap_desc(req.memory_type_bits, req.size) {
                    if alt == idx {
                        continue;
                    }
                    if let Ok(m) = dev.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(alt),
                        None,
                    ) {
                        fallback = Some((m, alt));
                        break;
                    }
                }
                match fallback {
                    Some(p) => p,
                    None => {
                        dev.destroy_buffer(buf, None);
                        return Err(vkerr(e));
                    }
                }
            }
        };
        dev.bind_buffer_memory(buf, mem, 0).map_err(vkerr)?;
        let host_visible = self.inner.mem_props.memory_types[used_idx as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        Ok((buf, mem, host_visible))
    }

    // Free bytes available in heap `h`. Uses VK_EXT_memory_budget (heapBudget = driver's estimate of
    // what this process may still allocate from the heap) when advertised; otherwise falls back to
    // the heap's total `size` as a conservative upper bound. Budget is re-queried each call because
    // it shifts as buffers are allocated/freed (this is the point of the scratch guard below).
    fn free_heap_bytes(&self, h: u32) -> u64 {
        let mp = &self.inner.mem_props;
        let total = mp.memory_heaps[h as usize].size;
        if !self.inner.has_mem_budget {
            return total;
        }
        unsafe {
            let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            {
                // props2 mutably borrows budget via push_next; scope it so the borrow ends before
                // we read budget back.
                let mut props2 =
                    vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
                self.inner
                    .instance
                    .get_physical_device_memory_properties2(self.inner.pdev, &mut props2);
            }
            // heap_budget is 0 for heaps the driver doesn't report; treat that as "unknown" and use
            // the static size so we never under-report and wrongly refuse a valid allocation.
            let b = budget.heap_budget[h as usize];
            if b == 0 {
                total
            } else {
                b
            }
        }
    }

    // Usable memory types for `type_bits` whose heap can (per free_heap_bytes) hold `bytes`, ordered
    // largest free-heap first. The placement primitive for both pick_memory_type and the OOM retry.
    fn usable_types_by_heap_desc(&self, type_bits: u32, bytes: u64) -> Vec<u32> {
        let mp = &self.inner.mem_props;
        let mut v: Vec<u32> = (0..mp.memory_type_count)
            .filter(|&i| (type_bits & (1 << i)) != 0)
            .filter(|&i| {
                let h = mp.memory_types[i as usize].heap_index;
                self.free_heap_bytes(h) >= bytes
            })
            .collect();
        v.sort_by_key(|&i| {
            let h = mp.memory_types[i as usize].heap_index;
            std::cmp::Reverse(self.free_heap_bytes(h))
        });
        v
    }

    // Will a transient scratch buffer of `bytes` total bytes fit in the largest usable heap's free
    // space, keeping a margin so we don't allocate the very last byte (which the driver may need for
    // command-buffer / descriptor backing)? Used to decide between the fp16 coopmat path (extra
    // scratch) and the fp32 path (none) without risking an OOM abort.
    fn scratch_fits(&self, bytes: u64) -> bool {
        // 64 MiB margin or 1/16 of the request, whichever is larger.
        let margin = (bytes / 16).max(64 * 1024 * 1024);
        let need = bytes.saturating_add(margin);
        let mp = &self.inner.mem_props;
        (0..mp.memory_heap_count)
            .map(|h| self.free_heap_bytes(h))
            .max()
            .map(|free| free >= need)
            .unwrap_or(false)
    }

    // Choose a memory type index for a `bytes`-sized buffer with the given `type_bits`, honouring the
    // configured MemStrategy. Returns None only when no type's heap can fit the request.
    //
    // The shapes we must handle on the AMD 8060S UMA part:
    //  - a small HOST_VISIBLE|DEVICE_LOCAL "carveout" heap (hundreds of MB), and
    //  - a large DEVICE_LOCAL-only heap (the GTT pool, tens of GB).
    // Placing an 18.6GB weight buffer demands the large heap, which may not be host-visible; the
    // upload/readback path stages through a host-visible buffer for those (see write_/read_).
    fn pick_memory_type(&self, type_bits: u32, bytes: u64) -> Option<u32> {
        let mp = &self.inner.mem_props;
        // host-visible candidates whose heap fits, cached-coherent preferred then plain coherent.
        let host_cached = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        let host_base =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let fits = |i: u32| -> bool {
            let h = mp.memory_types[i as usize].heap_index;
            self.free_heap_bytes(h) >= bytes
        };
        let host_visible_pick = |flags: vk::MemoryPropertyFlags| -> Option<u32> {
            (0..mp.memory_type_count).find(|&i| {
                (type_bits & (1 << i)) != 0
                    && mp.memory_types[i as usize].property_flags.contains(flags)
                    && fits(i)
            })
        };
        // Largest DEVICE_LOCAL heap that fits (the GTT pool on UMA), then largest heap of any kind.
        let device_local_pick = || -> Option<u32> {
            self.usable_types_by_heap_desc(type_bits, bytes)
                .into_iter()
                .find(|&i| {
                    mp.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                })
        };
        let any_largest = || {
            self.usable_types_by_heap_desc(type_bits, bytes)
                .into_iter()
                .next()
        };

        match self.inner.mem_strategy {
            MemStrategy::HostOnly => {
                host_visible_pick(host_cached).or_else(|| host_visible_pick(host_base))
            }
            MemStrategy::DeviceFirst => device_local_pick()
                .or_else(|| host_visible_pick(host_cached))
                .or_else(|| host_visible_pick(host_base))
                .or_else(any_largest),
            // Auto: prefer host-visible when it fits (cheap upload + readback, no staging on UMA),
            // else spill to the largest DEVICE_LOCAL heap — the path that makes the 18.6GB model load.
            MemStrategy::Auto => host_visible_pick(host_cached)
                .or_else(|| host_visible_pick(host_base))
                .or_else(device_local_pick)
                .or_else(any_largest),
        }
    }

    // Allocate a buffer that is guaranteed HOST_VISIBLE (for staging). Forces the host-visible
    // policy regardless of the device strategy; staging buffers are transient and always small
    // relative to the host-visible heap (one tensor's worth at a time). Not pooled.
    unsafe fn raw_buffer_host_visible(
        &self,
        bytes: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let bytes = bytes.max(4);
        let dev = self.dev();
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = dev.create_buffer(&info, None).map_err(vkerr)?;
        let req = dev.get_buffer_memory_requirements(buf);
        let mp = &self.inner.mem_props;
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let idx = match (0..mp.memory_type_count).find(|&i| {
            (req.memory_type_bits & (1 << i)) != 0
                && mp.memory_types[i as usize].property_flags.contains(host)
        }) {
            Some(i) => i,
            None => {
                dev.destroy_buffer(buf, None);
                return Err(Error::Msg(
                    "vulkan: no host-visible memory type for staging buffer".into(),
                ));
            }
        };
        let mem = match dev.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(idx),
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                dev.destroy_buffer(buf, None);
                return Err(vkerr(e));
            }
        };
        dev.bind_buffer_memory(buf, mem, 0).map_err(vkerr)?;
        Ok((buf, mem))
    }

    // Copy `bytes` from `src[src_off..]` to `dst[dst_off..]` on a one-shot command buffer and block
    // until done. Uses the submitter's command pool + its fence. The submitter lock is held for the
    // whole copy so it can't interleave with a concurrent batch on the same `cmd`; callers flush
    // first so `cmd` is idle on entry.
    unsafe fn copy_buffer_blocking(
        &self,
        src: vk::Buffer,
        src_off: u64,
        dst: vk::Buffer,
        dst_off: u64,
        bytes: u64,
    ) -> Result<()> {
        let dev = self.dev();
        let mut s = self.inner.submitter.lock().unwrap();
        dev.reset_command_buffer(s.cmd, vk::CommandBufferResetFlags::empty())
            .map_err(vkerr)?;
        dev.begin_command_buffer(
            s.cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
        .map_err(vkerr)?;
        let region = [vk::BufferCopy::default()
            .src_offset(src_off)
            .dst_offset(dst_off)
            .size(bytes)];
        dev.cmd_copy_buffer(s.cmd, src, dst, &region);
        dev.end_command_buffer(s.cmd).map_err(vkerr)?;
        dev.reset_fences(&[s.fence]).map_err(vkerr)?;
        let cmds = [s.cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        dev.queue_submit(self.inner.queue, &submit, s.fence)
            .map_err(vkerr)?;
        dev.wait_for_fences(&[s.fence], true, u64::MAX)
            .map_err(vkerr)?;
        s.recording = false;
        s.n = 0;
        s.written_since_barrier.clear();
        Ok(())
    }

    // Bound on a single staging chunk. A non-host-visible buffer can be many GB (the 18.6GB model),
    // but the host-visible heap on this UMA part is a small carveout — so we stage in chunks of at
    // most this size, reusing one small staging buffer, instead of needing a full-size host-visible
    // mirror. 256 MiB is a good balance of per-copy submit overhead vs staging footprint.
    const STAGE_CHUNK: u64 = 256 * 1024 * 1024;

    // Upload raw `data` into device-local (non-host-visible) `dst` via a transient host-visible
    // staging buffer + GPU copy. `dst` carries TRANSFER_DST usage (set in raw_buffer). Any recorded
    // batch is flushed first (the copy mutates `dst`), then the copy runs on its own one-shot submit.
    unsafe fn staged_upload(&self, dst: vk::Buffer, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        self.flush()?;
        let dev = self.dev();
        let total = data.len() as u64;
        // One small staging buffer (<= STAGE_CHUNK), reused across chunks, so the host-visible heap
        // never needs the full buffer's worth — the lever that lets an 18.6GB DEVICE_LOCAL buffer
        // load through a few-hundred-MB host-visible carveout.
        let chunk = total.min(Self::STAGE_CHUNK);
        let (staging, staging_mem) = self.raw_buffer_host_visible(chunk)?;
        let mut off = 0u64;
        let result = (|| -> Result<()> {
            while off < total {
                let n = (total - off).min(chunk);
                let ptr = dev
                    .map_memory(staging_mem, 0, n, vk::MemoryMapFlags::empty())
                    .map_err(vkerr)? as *mut u8;
                std::ptr::copy_nonoverlapping(data.as_ptr().add(off as usize), ptr, n as usize);
                dev.unmap_memory(staging_mem);
                self.copy_buffer_blocking(staging, 0, dst, off, n)?;
                off += n;
            }
            Ok(())
        })();
        dev.destroy_buffer(staging, None);
        dev.free_memory(staging_mem, None);
        result
    }

    // Read `bytes` out of device-local (non-host-visible) `src` via a transient host-visible staging
    // buffer + GPU copy, chunked like staged_upload. Mirror of staged_upload.
    unsafe fn staged_readback(&self, src: vk::Buffer, bytes: u64) -> Result<Vec<u8>> {
        if bytes == 0 {
            return Ok(Vec::new());
        }
        self.flush()?;
        let dev = self.dev();
        let chunk = bytes.min(Self::STAGE_CHUNK);
        let (staging, staging_mem) = self.raw_buffer_host_visible(chunk)?;
        let mut out = vec![0u8; bytes as usize];
        let mut off = 0u64;
        let result = (|| -> Result<()> {
            while off < bytes {
                let n = (bytes - off).min(chunk);
                self.copy_buffer_blocking(src, off, staging, 0, n)?;
                let ptr = dev
                    .map_memory(staging_mem, 0, n, vk::MemoryMapFlags::empty())
                    .map_err(vkerr)? as *const u8;
                std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr().add(off as usize), n as usize);
                dev.unmap_memory(staging_mem);
                off += n;
            }
            Ok(())
        })();
        dev.destroy_buffer(staging, None);
        dev.free_memory(staging_mem, None);
        result.map(|()| out)
    }

    // Upload f32 `data` into a storage buffer. Host-visible memory takes the direct-map fast path;
    // DEVICE_LOCAL-only memory (big weight buffers on UMA) goes through a staging copy. `buffer` is
    // unused on the host-visible path but required to issue the staging GPU copy on the other.
    unsafe fn write_f32(
        &self,
        buffer: vk::Buffer,
        mem: vk::DeviceMemory,
        host_visible: bool,
        data: &[f32],
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        if !host_visible {
            let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            return self.staged_upload(buffer, bytes);
        }
        let dev = self.dev();
        let ptr = dev
            .map_memory(mem, 0, (data.len() * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        dev.unmap_memory(mem);
        Ok(())
    }

    unsafe fn read_f32(
        &self,
        buffer: vk::Buffer,
        mem: vk::DeviceMemory,
        host_visible: bool,
        n: usize,
    ) -> Result<Vec<f32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if !host_visible {
            let raw = self.staged_readback(buffer, (n * 4) as u64)?;
            let mut v = vec![0f32; n];
            std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, n * 4);
            return Ok(v);
        }
        // Ensure all recorded GPU work that may write this buffer has completed.
        self.flush()?;
        let dev = self.dev();
        // Time just the map+copy (the flush above already reports its own phases). On the decode
        // hot path the only readback per token is the logits, so this is the per-token readback cost.
        let t0 = self.inner.profile.then(std::time::Instant::now);
        let ptr = dev
            .map_memory(mem, 0, (n * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *const f32;
        let v = std::slice::from_raw_parts(ptr, n).to_vec();
        dev.unmap_memory(mem);
        if let Some(t) = t0 {
            eprintln!(
                "[HANZO_VK_PROFILE] readback(f32): {n} elems map+copy={:.3}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(v)
    }

    // u32 mirrors of write_f32/read_f32 (buffers are just bytes; 4 bytes/elem).
    unsafe fn write_u32(
        &self,
        buffer: vk::Buffer,
        mem: vk::DeviceMemory,
        host_visible: bool,
        data: &[u32],
    ) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        if !host_visible {
            let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            return self.staged_upload(buffer, bytes);
        }
        let dev = self.dev();
        let ptr = dev
            .map_memory(mem, 0, (data.len() * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *mut u32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        dev.unmap_memory(mem);
        Ok(())
    }

    unsafe fn read_u32(
        &self,
        buffer: vk::Buffer,
        mem: vk::DeviceMemory,
        host_visible: bool,
        n: usize,
    ) -> Result<Vec<u32>> {
        if n == 0 {
            return Ok(Vec::new());
        }
        if !host_visible {
            let raw = self.staged_readback(buffer, (n * 4) as u64)?;
            let mut v = vec![0u32; n];
            std::ptr::copy_nonoverlapping(raw.as_ptr(), v.as_mut_ptr() as *mut u8, n * 4);
            return Ok(v);
        }
        // Ensure all recorded GPU work that may write this buffer has completed.
        self.flush()?;
        let dev = self.dev();
        let t0 = self.inner.profile.then(std::time::Instant::now);
        let ptr = dev
            .map_memory(mem, 0, (n * 4) as u64, vk::MemoryMapFlags::empty())
            .map_err(vkerr)? as *const u32;
        let v = std::slice::from_raw_parts(ptr, n).to_vec();
        dev.unmap_memory(mem);
        if let Some(t) = t0 {
            eprintln!(
                "[HANZO_VK_PROFILE] readback(u32): {n} elems map+copy={:.3}ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(v)
    }

    // Build (or fetch cached) compute pipeline for `name` with `n_buffers` storage bindings
    // and a push-constant range of `push_size` bytes.
    fn pipeline(
        &self,
        name: &'static str,
        n_buffers: usize,
        push_size: usize,
    ) -> Result<CachedPipeline> {
        if let Some(p) = self.inner.pipelines.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        let dev = self.dev();
        // A set layout consumed by vkCmdPushDescriptorSetKHR must be created with the
        // PUSH_DESCRIPTOR_KHR flag; the legacy allocate-from-pool path requires it absent. The
        // backend commits to one or the other at init (`push_descriptor`), so every layout is built
        // to match and the two paths can't be crossed for a given device.
        let layout_flags = if self.inner.push_descriptor.is_some() {
            vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR
        } else {
            vk::DescriptorSetLayoutCreateFlags::empty()
        };
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
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(layout_flags)
                        .bindings(&binds),
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
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, e)| vkerr(e))?[0];
            dev.destroy_shader_module(module, None);
            CachedPipeline {
                pipeline,
                layout,
                set_layout,
                n_buffers,
            }
        };
        self.inner
            .pipelines
            .lock()
            .unwrap()
            .insert(name, cached.clone());
        Ok(cached)
    }

    // Dispatch kernel `name` over `bufs` (bound 0..N) with raw push bytes and group counts. The
    // dispatch is recorded into the current batch's command buffer (deferred; submitted on flush).
    // Convention: the LAST buffer is the kernel's output (the one it writes); all others are inputs.
    // The two scatter kernels write binding 0 instead, so they call `dispatch_out` with out_idx=0.
    fn dispatch(
        &self,
        name: &'static str,
        bufs: &[vk::Buffer],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let out_idx = bufs.len().saturating_sub(1);
        self.dispatch_outs(name, bufs, &[out_idx], push, groups)
    }

    // Like `dispatch`, but `out_idx` names which binding the kernel writes (its output). Used for
    // selective hazard barriers: only a dispatch that reads a buffer produced earlier in this same
    // batch needs a barrier before it (see `written_since_barrier`).
    fn dispatch_out(
        &self,
        name: &'static str,
        bufs: &[vk::Buffer],
        out_idx: usize,
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        self.dispatch_outs(name, bufs, &[out_idx], push, groups)
    }

    // Like `dispatch_out`, but names ALL bindings the kernel writes. A fused kernel can both update a
    // state buffer in place and write a fresh output (gdn_step); every such write must be tracked so a
    // later in-batch reader of either gets the RAW barrier. The read-side check already tests all
    // bindings, so the only generalization needed is marking each output below.
    fn dispatch_outs(
        &self,
        name: &'static str,
        bufs: &[vk::Buffer],
        out_idxs: &[usize],
        push: &[u8],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let p = self.pipeline(name, bufs.len(), push.len())?;
        debug_assert_eq!(
            p.n_buffers,
            bufs.len(),
            "vulkan: kernel `{name}` binding count drift"
        );
        let dev = self.dev();
        let queue = self.inner.queue;
        let profile = self.inner.profile;
        let rec_t0 = profile.then(std::time::Instant::now);
        let mut s = self.inner.submitter.lock().unwrap();
        unsafe {
            if !s.recording {
                // Start a fresh batch: free the previous batch's descriptor sets (its GPU work
                // already completed at the last flush) and open the command buffer. The pool reset
                // is a no-op for the push-descriptor path (no sets are allocated from it) but stays
                // harmless and keeps the legacy path correct.
                dev.reset_descriptor_pool(s.dpool, vk::DescriptorPoolResetFlags::empty())
                    .map_err(vkerr)?;
                dev.begin_command_buffer(s.cmd, &vk::CommandBufferBeginInfo::default())
                    .map_err(vkerr)?;
                s.recording = true;
                s.n = 0;
                s.written_since_barrier.clear();
                s.record_ns = 0;
                s.barriers = 0;
            }
            // RAW hazard: if this dispatch touches a buffer that an earlier dispatch in this same
            // batch wrote (and we haven't barriered since), insert ONE memory barrier first, then
            // start a new barrier-free group. Independent dispatches (disjoint buffers, or reading
            // only weights/inputs from an earlier already-fenced batch) need no barrier and the GPU
            // can overlap their fixed launch/drain overhead -- the win on the decode hot path, which
            // is hundreds of tiny dispatches/token where that overhead, not compute, dominates.
            // We test ALL bindings (not just the inputs): a fresh output handle is never already in
            // the set (the pool recycles handles only across fences, so live handles are unique
            // in-batch), so this is exact for normal ops AND also catches the in-place scatter
            // kernels' read-modify-write of binding 0.
            // Buffer-scoped RAW barrier: emit one vk::BufferMemoryBarrier per read buffer that an
            // earlier in-batch dispatch wrote, instead of a single global vk::MemoryBarrier. A global
            // barrier drains the whole compute pipeline (no op overlaps), which on the decode hot path
            // (~1k tiny dispatches/token, GPU-execution-bound) is the dominant cost; scoping to the
            // actual hazard buffers lets the driver keep disjoint-buffer dispatches in flight (e.g. the
            // next matvec prefetching its weight while the current op's output is still draining). Same
            // src/dst access masks => strict refinement, no hazard lost (we still test ALL bufs, incl
            // the in-place-scatter RMW of binding 0).
            let mut bufbars = [vk::BufferMemoryBarrier::default(); 8];
            let mut nbar = 0usize;
            for &b in bufs {
                if s.written_since_barrier.contains(&b)
                    && !bufbars[..nbar].iter().any(|x| x.buffer == b)
                {
                    bufbars[nbar] = vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(b)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    nbar += 1;
                }
            }
            if nbar > 0 {
                dev.cmd_pipeline_barrier(
                    s.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &bufbars[..nbar],
                    &[],
                );
                s.written_since_barrier.clear();
                s.barriers += 1;
            }
            // Buffer infos are needed by both paths; build them once. (WriteDescriptorSet borrows
            // these, so they must outlive the update/push call below.) Built on the stack into a
            // fixed array — never a heap Vec — because this is the decode hot path: hundreds of
            // dispatches x N layers re-recorded every token, where a per-dispatch Vec alloc+free for
            // both `infos` and `writes` was pure CPU churn the GPU then stalled on. MAX_BINDINGS (8)
            // comfortably covers the widest kernel (4 buffers: where_cond/index_select); only the
            // first `nb = bufs.len()` entries are populated and passed on, and the debug_assert
            // above (kernel binding-count drift) plus the one here pin the bound, so it can never be
            // exceeded in a correct build. The arrays live for the rest of this unsafe block,
            // satisfying the borrows the push/update calls hold on them.
            const MAX_BINDINGS: usize = 8;
            let nb = bufs.len();
            debug_assert!(
                nb <= MAX_BINDINGS,
                "vulkan: kernel `{name}` binds {nb} buffers > MAX_BINDINGS {MAX_BINDINGS}"
            );
            let mut infos = [[vk::DescriptorBufferInfo::default(); 1]; MAX_BINDINGS];
            for (i, &b) in bufs.iter().enumerate() {
                infos[i] = [vk::DescriptorBufferInfo::default()
                    .buffer(b)
                    .range(vk::WHOLE_SIZE)];
            }
            let cmd = s.cmd;
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);

            if let Some(pd) = &self.inner.push_descriptor {
                // Fast path: push buffer handles inline into the command buffer. No descriptor-set
                // object is allocated and nothing is written to pool-backed GPU memory — the driver
                // records the bindings directly, eliminating the per-op allocate + update + bind
                // (three driver calls + descriptor-pool traffic) that dominated decode CPU time.
                let mut writes = [vk::WriteDescriptorSet::default(); MAX_BINDINGS];
                for (i, w) in writes.iter_mut().enumerate().take(nb) {
                    // dst_set is ignored by vkCmdPushDescriptorSetKHR (left default/null).
                    *w = vk::WriteDescriptorSet::default()
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i]);
                }
                pd.cmd_push_descriptor_set(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    p.layout,
                    0,
                    &writes[..nb],
                );
            } else {
                // Legacy path: allocate a fresh descriptor set for this dispatch. Sets accumulate
                // within the batch (each recorded dispatch keeps its own); the pool is reset only
                // when the next batch begins, after the current one has been submitted and awaited.
                let set_layouts = [p.set_layout];
                let set = dev
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(s.dpool)
                            .set_layouts(&set_layouts),
                    )
                    .map_err(vkerr)?[0];
                let mut writes = [vk::WriteDescriptorSet::default(); MAX_BINDINGS];
                for (i, w) in writes.iter_mut().enumerate().take(nb) {
                    *w = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&infos[i]);
                }
                dev.update_descriptor_sets(&writes[..nb], &[]);
                dev.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    p.layout,
                    0,
                    &[set],
                    &[],
                );
            }
            if !push.is_empty() {
                dev.cmd_push_constants(cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            }
            dev.cmd_dispatch(cmd, groups.0, groups.1, groups.2);
            // Mark this dispatch's output live in the current barrier-free group so a later
            // dispatch that READS it triggers the barrier above. WAW/WAR can't arise within a
            // batch: a buffer is written only as some op's freshly-allocated output, the pool only
            // recycles a freed buffer's handle into a new allocation AFTER a flush+fence (reclaim
            // runs post-fence), so no two live allocations in one batch share a handle -- thus the
            // only intra-batch hazard is RAW, which this set captures. Cross-batch ordering is the
            // full queue_submit + fence wait in flush_locked.
            for &out_idx in out_idxs {
                if let Some(&out_buf) = bufs.get(out_idx) {
                    s.written_since_barrier.insert(out_buf);
                }
            }
            s.n += 1;
            if profile {
                if let Some(t0) = rec_t0 {
                    s.record_ns += t0.elapsed().as_nanos();
                }
                *s.op_counts.entry(name).or_insert(0) += 1;
            }
            // Bound the descriptor-set budget: a forward longer than BATCH_CAP ops just flushes
            // a handful of times instead of once.
            if s.n >= BATCH_CAP {
                flush_locked(dev, queue, &mut s, profile)?;
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
        let profile = self.inner.profile;
        {
            let mut s = self.inner.submitter.lock().unwrap();
            flush_locked(dev, queue, &mut s, profile)?;
        }
        self.reclaim();
        Ok(())
    }

    // Move buffers dropped before this point into the reuse pool. Sound only right after a
    // flush+fence: the awaited batch (the last that could reference them) is done on the GPU -- which
    // is also why it's safe to actually destroy buffers here when over the cap (no in-flight work can
    // still reference a free-list entry).
    fn reclaim(&self) {
        let dev = self.dev();
        let mut pool = self.inner.bufpool.lock().unwrap();
        let pending = std::mem::take(&mut pool.pending);
        for (bytes, p) in pending {
            pool.free.entry(bytes).or_default().push(p);
            pool.free_bytes += bytes;
        }
        // Enforce the idle-pool cap: destroy real device buffers (largest buckets first, since those
        // dominate the bytes and are the least likely to be reused) until back under the cap.
        while pool.free_bytes > POOL_FREE_CAP_BYTES {
            let Some(&bucket) = pool.free.keys().max() else {
                break;
            };
            let Some(bufs) = pool.free.get_mut(&bucket) else {
                break;
            };
            let Some(p) = bufs.pop() else {
                pool.free.remove(&bucket);
                continue;
            };
            if bufs.is_empty() {
                pool.free.remove(&bucket);
            }
            pool.free_bytes = pool.free_bytes.saturating_sub(bucket);
            unsafe {
                dev.destroy_buffer(p.buffer, None);
                dev.free_memory(p.memory, None);
            }
        }
    }

    // Allocate an f32 storage holding `count` elements (uninitialized device memory).
    fn alloc_f32(&self, count: usize) -> Result<VulkanStorage> {
        let (buffer, memory, host_visible) = unsafe { self.raw_buffer((count * 4) as u64)? };
        Ok(VulkanStorage {
            buffer,
            memory,
            count,
            dtype: DType::F32,
            host_visible,
            device: self.clone(),
        })
    }

    // fp16 scratch (2 bytes/elem) for coopmat matmul inputs. Returns the buffer plus its memory,
    // host-visibility, and byte size so the caller can return it to the pool via free_scratch.
    fn alloc_f16(&self, count: usize) -> Result<(vk::Buffer, vk::DeviceMemory, bool, u64)> {
        let bytes = ((count * 2).max(4)) as u64;
        let (buffer, memory, host_visible) = unsafe { self.raw_buffer(bytes)? };
        Ok((buffer, memory, host_visible, bytes))
    }

    // Return a scratch buffer to the pool. Deferred-safe like VulkanStorage::drop: it parks in
    // `pending` and is reclaimed only after the next flush+fence, by which point the dispatch that
    // referenced it has completed on the GPU.
    fn free_scratch(
        &self,
        bytes: u64,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        host_visible: bool,
    ) {
        // Park under the bucket key, matching the size raw_buffer allocated and looks up.
        let bytes = pool_bucket(bytes);
        if let Ok(mut pool) = self.inner.bufpool.lock() {
            pool.pending.push((
                bytes,
                PooledBuf {
                    buffer,
                    memory,
                    host_visible,
                },
            ));
        }
    }

    pub(crate) fn upload_f32(&self, data: &[f32]) -> Result<VulkanStorage> {
        let s = self.alloc_f32(data.len())?;
        unsafe { self.write_f32(s.buffer, s.memory, s.host_visible, data)? };
        Ok(s)
    }

    // u32 storage (ids/cond). 4 bytes/elem, same as f32.
    fn alloc_u32(&self, count: usize) -> Result<VulkanStorage> {
        let (buffer, memory, host_visible) = unsafe { self.raw_buffer((count * 4) as u64)? };
        Ok(VulkanStorage {
            buffer,
            memory,
            count,
            dtype: DType::U32,
            host_visible,
            device: self.clone(),
        })
    }

    fn upload_u32(&self, data: &[u32]) -> Result<VulkanStorage> {
        let s = self.alloc_u32(data.len())?;
        unsafe { self.write_u32(s.buffer, s.memory, s.host_visible, data)? };
        Ok(s)
    }

    // Name a CPU-fallback round-trip when HANZO_VK_PROFILE is set. Each fallback op reads its
    // operand(s) back to the host, computes on the (UNtimed) CPU, and re-uploads -- a hidden
    // bottleneck the size-only readback log can't attribute to an op. This names the culprit so a
    // GPU re-run can prioritize which op to port native next. Zero-cost when profiling is off (the
    // call is behind the `profile` bool and `op`/`extra` are cheap &str / Display formatting only
    // evaluated on the slow path). `op` is the op name; `extra` is a shape/size descriptor.
    #[inline]
    fn profile_fallback(&self, op: &str, extra: std::fmt::Arguments<'_>) {
        if self.inner.profile {
            eprintln!("[HANZO_VK_PROFILE] cpu-fallback op={op} {extra} (GPU->CPU->GPU round-trip)");
        }
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
// when nothing is recorded. Caller must hold the submitter lock. When `profile` is set, prints the
// per-batch phase breakdown (recording / submit / fence-wait time, dispatch + barrier counts) so
// the real GPU shows where per-token milliseconds go; the timers are only sampled when profiling.
fn flush_locked(dev: &ash::Device, queue: vk::Queue, s: &mut Submitter, profile: bool) -> Result<()> {
    if !s.recording {
        return Ok(());
    }
    let n = s.n;
    let barriers = s.barriers;
    let record_ms = s.record_ns as f64 / 1e6;
    let (mut submit_ms, mut wait_ms) = (0.0f64, 0.0f64);
    unsafe {
        dev.end_command_buffer(s.cmd).map_err(vkerr)?;
        dev.reset_fences(&[s.fence]).map_err(vkerr)?;
        let cmds = [s.cmd];
        let t_sub = profile.then(std::time::Instant::now);
        dev.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)],
            s.fence,
        )
        .map_err(vkerr)?;
        if let Some(t) = t_sub {
            submit_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        let t_wait = profile.then(std::time::Instant::now);
        dev.wait_for_fences(&[s.fence], true, u64::MAX)
            .map_err(vkerr)?;
        if let Some(t) = t_wait {
            wait_ms = t.elapsed().as_secs_f64() * 1e3;
        }
    }
    if profile {
        // One line per submitted batch. On decode this is ~one batch per token, so this is the
        // per-token GPU breakdown: `dispatch` = ops recorded, `barriers` = memory barriers emitted
        // (lower is better -- with selective barriers, independent ops in a row emit none).
        eprintln!(
            "[HANZO_VK_PROFILE] flush: dispatch={n} barriers={barriers} \
             record={record_ms:.3}ms submit={submit_ms:.3}ms fence_wait={wait_ms:.3}ms",
        );
        // Per-op histogram (top 14 by dispatch count) so fusion targets the dominant kernels.
        let mut ops: Vec<(&&'static str, &u32)> = s.op_counts.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        let hist = ops
            .iter()
            .take(14)
            .map(|(name, c)| format!("{name}={c}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("[HANZO_VK_PROFILE]   ops: {hist}");
    }
    s.op_counts.clear();
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

// Activation rows decoded per invocation in the quantized prefill matmul kernels (mul_mat_q8 /
// mul_mat_q4k). MUST equal the MAX_M const in those .comp shaders (their register accumulator array
// bound). The host tiles the M dimension by this so each weight block is still read once per output
// column across a tile of up to MATMUL_Q_MAX_M rows.
const MATMUL_Q_MAX_M: usize = 16;

// Compile-time per-invocation array bounds in gdn_step.comp / gdn_conv1d_step.comp (MAX_K #defines).
// head_k_dim must be <= GDN_STEP_MAX_K and the conv kernel <= GDN_CONV_MAX_K; both checked host-side.
const GDN_STEP_MAX_K: usize = 256;
const GDN_CONV_MAX_K: usize = 8;

// Compile-time MAX_COLS_PAD bound on argsort.comp's shared index scratch. cols_pad (next pow2 of the
// sorted dim) must be <= ARGSORT_MAX_COLS_PAD; wider rows fall back to the CPU argsort (not on the
// MoE routing hot path, where cols == num_experts ~128).
const ARGSORT_MAX_COLS_PAD: usize = 1024;

impl BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;

    fn new(ordinal: usize) -> Result<Self> {
        unsafe {
            let entry = ash::Entry::load()
                .map_err(|e| Error::Msg(format!("vulkan: loader not found: {e}")))?;
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
                .map_err(vkerr)?;

            // Collect non-CPU adapters in enumeration order; pick the `ordinal`-th (like the probe).
            let mut gpus = Vec::new();
            for pd in instance.enumerate_physical_devices().map_err(vkerr)? {
                let p = instance.get_physical_device_properties(pd);
                let name = CStr::from_ptr(p.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned();
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
                .ok_or_else(|| Error::Msg("vulkan: no compute queue".into()))?
                as u32;
            let prios = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(qfi)
                .queue_priorities(&prios)];

            // Probe cooperative-matrix support: need the device extension AND a config with
            // fp16 A/B, fp32 C/result at subgroup scope. Guard the query on the extension being
            // advertised (the loader's fn pointer is only valid then) so WSL/Dozen stays on the
            // plain tiled path.
            let dev_exts = instance
                .enumerate_device_extension_properties(pdev)
                .unwrap_or_default();
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
            // Default ON when the device advertises coopmat: the register-blocked kernel
            // (bmm_coopmat_rb) measured 1.3-2.7x over the fp32 bmm_reg on the real AMD driver and a
            // full Qwen3-0.6B forward's argmax matched CPU exactly (fp16 inputs, fp32 accumulate).
            // Set HANZO_VK_COOPMAT=0 to force the fp32 path (e.g. if precision matters).
            let cm_use = coopmat
                && std::env::var("HANZO_VK_COOPMAT")
                    .map(|v| v != "0")
                    .unwrap_or(true);

            // VK_KHR_push_descriptor: lets `dispatch` push buffer handles inline into the command
            // buffer (vkCmdPushDescriptorSetKHR) instead of allocating + updating + binding a
            // descriptor set per op. That per-op churn is the dominant CPU cost on the decode hot
            // path (same op graph, hundreds of dispatches x 28 layers, re-recorded every token), so
            // collapsing it to one recorded command is the lever. Enabled when advertised (native
            // AMD/NV; typically absent on WSL/Dozen, which keeps the legacy path). HANZO_VK_PUSH_DESC=0
            // forces the legacy path. The extension's guaranteed maxPushDescriptors >= 32 dwarfs our
            // widest kernel (4 storage buffers), so no per-pipeline limit check is needed.
            let has_pd_ext = dev_exts.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == ash::khr::push_descriptor::NAME
            });
            let use_pd = has_pd_ext
                && std::env::var("HANZO_VK_PUSH_DESC")
                    .map(|v| v != "0")
                    .unwrap_or(true);

            // Buffer-memory placement policy. Default `auto`: host-visible when it fits, else spill
            // big buffers (e.g. an 18.6GB model's weights) to the largest DEVICE_LOCAL heap (the GTT
            // pool on this UMA APU), which is where the real capacity lives. `host_only` restores the
            // legacy host-visible-only behaviour; `device_first` forces big-heap placement always.
            let mem_strategy = match std::env::var("HANZO_VK_DEVICE_MEMORY_STRATEGY")
                .ok()
                .as_deref()
                .map(str::trim)
            {
                Some("host_only") | Some("host") => MemStrategy::HostOnly,
                Some("device_first") | Some("device") => MemStrategy::DeviceFirst,
                Some("auto") | None | Some("") => MemStrategy::Auto,
                Some(other) => {
                    eprintln!(
                        "[vulkan] unknown HANZO_VK_DEVICE_MEMORY_STRATEGY=`{other}` (expected host_only|device_first|auto); using auto"
                    );
                    MemStrategy::Auto
                }
            };

            // VK_EXT_memory_budget: enables querying per-heap *free* bytes at runtime (the scratch
            // guard needs this; the static heap `size` is total capacity only). Enable it when the
            // device advertises it; otherwise the guard conservatively falls back to total size.
            let has_mem_budget = dev_exts.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::memory_budget::NAME
            });

            // Subgroup capability (core in Vulkan 1.1+, which the 1.3 instance guarantees). The q8
            // mat-vec subgroup kernel uses subgroupAdd (ARITHMETIC) at COMPUTE-stage scope, so we
            // require both before enabling it. HANZO_VK_SUBGROUP_MATVEC=0 forces the scalar kernel.
            let mut sg_props = vk::PhysicalDeviceSubgroupProperties::default();
            {
                // p2 mutably borrows sg_props via push_next; scope it so the borrow ends before we
                // read sg_props back below.
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg_props);
                instance.get_physical_device_properties2(pdev, &mut p2);
            }
            let sg_compute = sg_props
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE);
            let sg_arith = sg_props
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC);
            // subgroupElect needs BASIC; subgroupAdd needs ARITHMETIC. Require a sane width (>=2)
            // so the reduction is actually parallel.
            let subgroup_size = sg_props.subgroup_size;
            let subgroup_matvec = sg_compute
                && sg_arith
                && subgroup_size >= 2
                && std::env::var("HANZO_VK_SUBGROUP_MATVEC")
                    .map(|v| v != "0")
                    .unwrap_or(true);

            // Hardware int8 dot product (core in Vulkan 1.3, which the 1.3 instance guarantees, so no
            // extension to enable -- only the feature must be turned on at device creation to use
            // OpSDotAccSat). Gate the Q8_0 PREFILL dp4a path on BOTH the feature
            // (shaderIntegerDotProduct) AND the 4x8-bit-packed-signed accelerated properties: the
            // kernel uses the accumulating-saturating signed 4x8 op, so we require its acc-sat property
            // (and the plain 4x8 property) to be hardware-accelerated, else the f32-decode mul_mat_q8 is
            // as good or better. Two separate queries: properties via PhysicalDeviceProperties2, the
            // feature via PhysicalDeviceFeatures2.
            let mut idp_props = vk::PhysicalDeviceShaderIntegerDotProductProperties::default();
            {
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut idp_props);
                instance.get_physical_device_properties2(pdev, &mut p2);
            }
            let mut idp_feat = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
            {
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut idp_feat);
                instance.get_physical_device_features2(pdev, &mut f2);
            }
            let int_dot8 = idp_feat.shader_integer_dot_product == vk::TRUE
                && idp_props.integer_dot_product4x8_bit_packed_signed_accelerated == vk::TRUE
                && idp_props
                    .integer_dot_product_accumulating_saturating4x8_bit_packed_signed_accelerated
                    == vk::TRUE
                && std::env::var("HANZO_VK_INT_DOT8")
                    .map(|v| v != "0")
                    .unwrap_or(true);

            // Build the enabled-extension list dynamically: coopmat and push_descriptor are
            // independent and either may be present. push_descriptor needs no extra device feature
            // struct (just the extension + a fn-pointer load), so it's a bare name here.
            let mut ext_names: Vec<*const std::os::raw::c_char> = Vec::new();
            if coopmat {
                ext_names.push(ash::khr::cooperative_matrix::NAME.as_ptr());
            }
            if use_pd {
                ext_names.push(ash::khr::push_descriptor::NAME.as_ptr());
            }
            if has_mem_budget {
                ext_names.push(ash::ext::memory_budget::NAME.as_ptr());
            }
            // Coopmat's SPIR-V needs these features (cooperative matrix, Vulkan memory model, fp16
            // arithmetic, 16-bit storage); push_descriptor needs none.
            let mut cm_feat =
                vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut mm_feat =
                vk::PhysicalDeviceVulkanMemoryModelFeatures::default().vulkan_memory_model(true);
            let mut f16_feat =
                vk::PhysicalDeviceShaderFloat16Int8Features::default().shader_float16(true);
            let mut s16_feat =
                vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
            // Turn the int8 dot-product feature ON at device creation when gated in; OpSDotAccSat is
            // illegal otherwise. Reuse a fresh struct (the queried one carries the read-back value).
            let mut idp_enable = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
                .shader_integer_dot_product(true);
            let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci);
            if !ext_names.is_empty() {
                dci = dci.enabled_extension_names(&ext_names);
            }
            if coopmat {
                dci = dci
                    .push_next(&mut cm_feat)
                    .push_next(&mut mm_feat)
                    .push_next(&mut f16_feat)
                    .push_next(&mut s16_feat);
            }
            if int_dot8 {
                dci = dci.push_next(&mut idp_enable);
            }
            let device = instance.create_device(pdev, &dci, None).map_err(vkerr)?;
            // Load push_descriptor device fns now that the device exists with the extension enabled.
            let push_descriptor = use_pd
                .then(|| ash::khr::push_descriptor::Device::new(&instance, &device));
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
            let fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(vkerr)?;
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
            let submitter = Mutex::new(Submitter {
                cpool,
                cmd,
                fence,
                dpool,
                recording: false,
                n: 0,
                written_since_barrier: std::collections::HashSet::new(),
                record_ns: 0,
                barriers: 0,
                op_counts: std::collections::HashMap::new(),
            });

            // Phase profiling: opt-in, read once here so the hot path only checks a bool.
            let profile = std::env::var("HANZO_VK_PROFILE")
                .map(|v| v != "0")
                .unwrap_or(false);

            let inner = VkInner {
                _entry: entry,
                instance,
                pdev,
                device,
                queue,
                qfi,
                gpu_id: ordinal,
                mem_props,
                mem_strategy,
                has_mem_budget,
                subgroup_matvec,
                subgroup_size,
                seed: Mutex::new(299792458),
                profile,
                push_descriptor,
                pipelines: Mutex::new(HashMap::new()),
                submitter,
                bufpool: Mutex::new(BufPool::default()),
                coopmat,
                cm_mnk,
                cm_use,
                int_dot8,
            };
            Ok(Self {
                inner: Arc::new(inner),
            })
        }
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Vulkan {
            gpu_id: self.inner.gpu_id,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &rhs.inner)
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let count = shape.elem_count();
        // Invariant: Vulkan computes in f32; f16/bf16 are represented as f32 and u8 as u32. So
        // e.g. zeros(bf16) yields an f32 buffer of zeros (zero is the same in any of these reprs).
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => self.upload_f32(&vec![0f32; count]),
            DType::U32 | DType::U8 => self.upload_u32(&vec![0u32; count]),
            _ => crate::bail!("vulkan: only f32/u32/f16/bf16/u8 supported, got {dtype:?}"),
        }
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // Same dtype mapping as zeros_impl: f16/bf16 -> f32 storage, u8 -> u32 storage.
        match dtype {
            DType::F32 | DType::F16 | DType::BF16 => self.alloc_f32(shape.elem_count()),
            DType::U32 | DType::U8 => self.alloc_u32(shape.elem_count()),
            _ => crate::bail!("vulkan: only f32/u32/f16/bf16/u8 supported, got {dtype:?}"),
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
            CpuStorage::F16(v) => {
                self.upload_f32(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>())
            }
            CpuStorage::BF16(v) => {
                self.upload_f32(&v.iter().map(|x| x.to_f32()).collect::<Vec<_>>())
            }
            // u8 (e.g. boolean attention masks) -> u32: where_cond and casts use u32 on Vulkan.
            CpuStorage::U8(v) => self.upload_u32(&v.iter().map(|&x| x as u32).collect::<Vec<_>>()),
            _ => crate::bail!(
                "vulkan: only f32/u32/f16/bf16/u8 supported, got {:?}",
                s.dtype()
            ),
        }
    }

    fn storage_from_cpu_storage_owned(&self, s: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&s)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        min: f64,
        max: f64,
    ) -> Result<Self::Storage> {
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

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
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

    /// Number of u32 words in this buffer. For a quantized weight/bank uploaded via the `quantize_*`
    /// helpers (all `upload_u32`-backed, dtype U32) this is the block-word count -- used to assert a
    /// resident MoE bank's per-expert stride covers exactly E experts.
    pub fn len_words(&self) -> usize {
        self.count
    }

    // Download the whole buffer as f32 (used internally + by to_cpu_storage).
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        unsafe {
            self.device
                .read_f32(self.buffer, self.memory, self.host_visible, self.count)
        }
    }

    // Download the whole buffer as u32 (used by to_cpu_storage for U32 storage).
    fn to_vec_u32(&self) -> Result<Vec<u32>> {
        unsafe {
            self.device
                .read_u32(self.buffer, self.memory, self.host_visible, self.count)
        }
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
        // Push block MUST match strided_copy.comp exactly: {n, rank, offset, dst_offset, shape[6],
        // strides[6]}. dst_offset is 0 here (we materialize into a packed offset-0 buffer); it was
        // the missing 4th field that previously shifted shape/strides by one slot and silently
        // corrupted every materialized (broadcast/transpose) operand consumed in-batch.
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
            &[self.buffer, out.buffer],
            &push_u32(&p),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    // Materialize a u32 storage (ids / where_cond mask) into a fresh contiguous, offset-0 buffer.
    // Identity (raw clone) when already contiguous from offset 0. Done on the CPU via the layout's
    // strided index so it's bit-exact for arbitrary u32 values -- reusing the float strided_copy
    // would reinterpret small integers as denormal floats, which load-time flush-to-zero could
    // corrupt. These tensors (token/position ids, attention masks) are tiny, so the round-trip is
    // cheap relative to correctness.
    fn contiguous_u32(&self, layout: &Layout) -> Result<VulkanStorage> {
        debug_assert_eq!(self.dtype, DType::U32);
        if layout.is_contiguous() && layout.start_offset() == 0 {
            return self.device.upload_u32(&self.to_vec_u32()?);
        }
        let src = self.to_vec_u32()?;
        let gathered: Vec<u32> = layout.strided_index().map(|i| src[i]).collect();
        self.device.upload_u32(&gathered)
    }

    // Buffer for a contiguous, offset-0 view of `layout`: no copy when the storage already is one
    // (returns its own buffer), otherwise materializes a packed copy returned via `keep` (held
    // alive by the caller until the dispatch is recorded). Each avoided contiguous() is one fewer
    // dispatch, which dominates per-op cost in a forward. Use only for transient op inputs -- the
    // result `out` must be a fresh alloc, never this buffer (it may alias `self`).
    fn contig_buf(&self, layout: &Layout, keep: &mut Option<VulkanStorage>) -> Result<vk::Buffer> {
        if layout.is_contiguous() && layout.start_offset() == 0 {
            Ok(self.buffer)
        } else {
            let s = self.contiguous(layout)?;
            let b = s.buffer;
            *keep = Some(s);
            Ok(b)
        }
    }

    // --- native fused ops (replace the GPU<->CPU round-trip fallbacks) ---

    // Row-wise argsort over the last (contiguous) dim via the bitonic argsort kernel. Returns the u32
    // index permutation (shape == input shape) entirely on the GPU. `Ok(None)` when cols_pad exceeds
    // the shader's shared-scratch bound (ARGSORT_MAX_COLS_PAD); the caller then uses the CPU argsort.
    // MoE routing (cols == num_experts ~128) always stays on the GPU path.
    pub fn arg_sort_last_dim(
        &self,
        layout: &Layout,
        asc: bool,
        last_dim: usize,
    ) -> Result<Option<VulkanStorage>> {
        let cols = last_dim;
        let cols_pad = cols.next_power_of_two();
        if cols_pad > ARGSORT_MAX_COLS_PAD {
            return Ok(None);
        }
        let mut xk = None;
        let xb = self.contig_buf(layout, &mut xk)?;
        let n = layout.shape().elem_count();
        let rows = n / cols.max(1);
        let out = self.device.alloc_u32(n)?;
        let push = push_u32(&[rows as u32, cols as u32, cols_pad as u32, asc as u32]);
        // One workgroup per row; the shader threads cooperate over cols_pad inside it.
        self.device
            .dispatch("argsort", &[xb, out.buffer], &push, (rows as u32, 1, 1))?;
        Ok(Some(out))
    }

    // softmax over the last dim. `self` is the input; `layout` its layout. One row per thread.
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<VulkanStorage> {
        let mut xk = None;
        let xb = self.contig_buf(layout, &mut xk)?;
        let dims = layout.dims();
        let m = *dims.last().unwrap_or(&1);
        let nrows = layout.shape().elem_count() / m.max(1);
        let out = self.device.alloc_f32(nrows * m)?;
        self.device.dispatch(
            "softmax_rows",
            &[xb, out.buffer],
            &push_u32(&[nrows as u32, m as u32]),
            Self::groups_1d(nrows),
        )?;
        Ok(out)
    }

    // rms-norm over the last dim. `self`=x, `alpha`=scale [m]. Matches hanzo-ml rms-norm.
    pub fn rms_norm(
        &self,
        layout: &Layout,
        alpha: &VulkanStorage,
        alpha_l: &Layout,
        eps: f32,
    ) -> Result<VulkanStorage> {
        let mut xk = None;
        let mut ak = None;
        let xb = self.contig_buf(layout, &mut xk)?;
        let ab = alpha.contig_buf(alpha_l, &mut ak)?;
        let dims = layout.dims();
        let m = *dims.last().unwrap_or(&1);
        let nrows = layout.shape().elem_count() / m.max(1);
        let out = self.device.alloc_f32(nrows * m)?;
        let mut push = push_u32(&[nrows as u32, m as u32]);
        push.extend_from_slice(&eps.to_ne_bytes());
        self.device.dispatch(
            "rms_norm",
            &[xb, ab, out.buffer],
            &push,
            Self::groups_1d(nrows),
        )?;
        Ok(out)
    }

    // Fused SwiGLU: out = silu(self) * rhs, elementwise. One dispatch instead of silu + mul.
    pub fn silu_mul(
        &self,
        layout: &Layout,
        rhs: &VulkanStorage,
        rhs_l: &Layout,
    ) -> Result<VulkanStorage> {
        let mut ak = None;
        let mut bk = None;
        let ab = self.contig_buf(layout, &mut ak)?;
        let bb = rhs.contig_buf(rhs_l, &mut bk)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        self.device.dispatch(
            "silu_mul",
            &[ab, bb, out.buffer],
            &(n as u32).to_ne_bytes(),
            Self::groups_1d(n),
        )?;
        Ok(out)
    }

    // Unary sigmoid: out = 1 / (1 + exp(-self)). Vulkan path for hanzo_nn::ops::sigmoid.
    pub fn sigmoid(&self, layout: &Layout) -> Result<VulkanStorage> {
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let mut out = self.device.alloc_f32(n)?;
        // Preserve the input's logical dtype (storage is f32 regardless); callers multiply the
        // result against a same-dtype tensor without an intervening cast (output-gate path).
        out.dtype = self.dtype;
        self.device.dispatch(
            "sigmoid",
            &[cb, out.buffer],
            &(n as u32).to_ne_bytes(),
            Self::groups_1d(n),
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
            &[srcb, cb, sb, out.buffer],
            &push_u32(&[b as u32, h as u32, t as u32, d as u32, unbatched]),
            Self::groups_1d(pairs),
        )?;
        Ok(out)
    }

    // Gated delta rule, single decode step (seq_len==1). `self`=q, plus k,v,g,beta; `state` is the
    // recurrent state [BH, K, V], updated IN PLACE in VRAM (kept across tokens by the caller's state
    // pool). q,k: [BH, K]; v: [BH, V]; g,beta: [BH]. q must be pre-scaled by 1/sqrt(K). Returns
    // y [BH, V]. Mirrors gated_delta_rule_recurrence for seq=1 and the CUDA single-step kernel.
    // GDN_STEP_MAX_K bounds the shader's per-invocation k array; head_k_dim must not exceed it.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_step(
        &self,
        q_l: &Layout,
        k: &VulkanStorage,
        k_l: &Layout,
        v: &VulkanStorage,
        v_l: &Layout,
        g: &VulkanStorage,
        g_l: &Layout,
        beta: &VulkanStorage,
        beta_l: &Layout,
        state: &VulkanStorage,
        state_l: &Layout,
        bh: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<VulkanStorage> {
        if k_dim > GDN_STEP_MAX_K {
            crate::bail!("vulkan: gdn_step head_k_dim {k_dim} exceeds GDN_STEP_MAX_K {GDN_STEP_MAX_K}");
        }
        if !(state_l.is_contiguous() && state_l.start_offset() == 0) {
            crate::bail!("vulkan: gdn_step state must be contiguous and offset 0 (it is updated in place)");
        }
        let mut qk = None;
        let mut kk = None;
        let mut vk_ = None;
        let mut gk = None;
        let mut bk = None;
        let qb = self.contig_buf(q_l, &mut qk)?;
        let kb = k.contig_buf(k_l, &mut kk)?;
        let vb = v.contig_buf(v_l, &mut vk_)?;
        let gb = g.contig_buf(g_l, &mut gk)?;
        let betab = beta.contig_buf(beta_l, &mut bk)?;
        let out = self.device.alloc_f32(bh * v_dim)?;
        // State (binding 5) is read-modify-written in place; the fresh output (binding 6) is written
        // too. Mark BOTH so a later in-batch reader (scatter of state, RMSNorm of the output) gets the
        // RAW barrier. The read-side hazard check already tests all bindings, including the state input
        // gathered earlier this batch.
        self.device.dispatch_outs(
            "gdn_step",
            &[qb, kb, vb, gb, betab, state.buffer, out.buffer],
            &[5, 6],
            &push_u32(&[bh as u32, k_dim as u32, v_dim as u32]),
            Self::groups_1d(bh * v_dim),
        )?;
        Ok(out)
    }

    // Causal depthwise conv1d, single decode step (seq_len==1, batch==1). `self`=conv_state
    // [conv_dim, k_size], updated IN PLACE (drop oldest column, append `x`); `x` is the new column
    // [conv_dim]; `w` the weight [conv_dim, k_size]. Returns silu(conv) [conv_dim]. Mirrors
    // causal_conv1d_update for seq=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv1d_step(
        &self,
        state_l: &Layout,
        x: &VulkanStorage,
        x_l: &Layout,
        w: &VulkanStorage,
        w_l: &Layout,
        conv_dim: usize,
        k_size: usize,
    ) -> Result<VulkanStorage> {
        if k_size > GDN_CONV_MAX_K {
            crate::bail!("vulkan: gdn_conv1d_step kernel {k_size} exceeds GDN_CONV_MAX_K {GDN_CONV_MAX_K}");
        }
        if !(state_l.is_contiguous() && state_l.start_offset() == 0) {
            crate::bail!("vulkan: gdn_conv1d_step conv_state must be contiguous and offset 0 (updated in place)");
        }
        let mut xk = None;
        let mut wk = None;
        let xb = x.contig_buf(x_l, &mut xk)?;
        let wb = w.contig_buf(w_l, &mut wk)?;
        let out = self.device.alloc_f32(conv_dim)?;
        // conv_state (binding 0) is updated in place; output (binding 3) is fresh. Mark both.
        self.device.dispatch_outs(
            "gdn_conv1d_step",
            &[self.buffer, xb, wb, out.buffer],
            &[0, 3],
            &push_u32(&[conv_dim as u32, k_size as u32]),
            Self::groups_1d(conv_dim),
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
        // scatter writes (and scatter_add reads) binding 0 (`self.buffer`), not the last binding,
        // so name out_idx=0 for the hazard tracker.
        self.device.dispatch_out(
            kernel,
            &[self.buffer, srcc.buffer, idc.buffer],
            0,
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
                b as u32, c as u32, ih as u32, iw as u32, oh as u32, ow as u32, kh as u32,
                kw as u32, sh as u32, sw as u32,
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
        // Buffers are just bytes (4 bytes/elem). Vulkan storage physically holds f32 or u32; per
        // the upload invariant f16/bf16 live as f32 and u8 as u32, so clone them through the same
        // 4-byte representation rather than bailing.
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
        let mut push = (n as u32).to_ne_bytes().to_vec();
        push.extend_from_slice(&(mul as f32).to_ne_bytes());
        push.extend_from_slice(&(add as f32).to_ne_bytes());
        self.device
            .dispatch("affine", &[cb, out.buffer], &push, Self::groups_1d(n))?;
        Ok(out)
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut push = (n as u32).to_ne_bytes().to_vec();
        push.extend_from_slice(&(e as f32).to_ne_bytes());
        self.device
            .dispatch("powf", &[cb, out.buffer], &push, Self::groups_1d(n))?;
        Ok(out)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let c = self.contiguous(layout)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let mut push = (n as u32).to_ne_bytes().to_vec();
        push.extend_from_slice(&(alpha as f32).to_ne_bytes());
        self.device
            .dispatch("elu", &[c.buffer, out.buffer], &push, Self::groups_1d(n))?;
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
        // The SPIR-V reduce kernel collapses the last (contiguous) dim into one output per row. For
        // any other single axis, permute it to the end first so the same kernel runs on the GPU; the
        // resulting row-major order matches the framework's keep-dim wrap exactly.
        let (c, cols) = if sum_dims == [rank - 1] {
            (self.contiguous(layout)?, dims[rank - 1])
        } else if sum_dims.len() == 1 {
            let d = sum_dims[0];
            let mut perm: Vec<usize> = (0..rank).filter(|&x| x != d).collect();
            perm.push(d);
            (self.contiguous(&layout.permute(&perm)?)?, dims[d])
        } else {
            crate::bail!("vulkan: reduce over multiple axes at once not supported (got {sum_dims:?})");
        };
        let rows: usize = layout.shape().elem_count() / cols;
        // arg-reductions return u32 indices; value reductions return f32.
        let out = if is_arg {
            self.device.alloc_u32(rows)?
        } else {
            self.device.alloc_f32(rows)?
        };
        let push = push_u32(&[rows as u32, cols as u32]);
        // one invocation per row
        self.device.dispatch(
            kernel,
            &[c.buffer, out.buffer],
            &push,
            Self::groups_1d(rows),
        )?;
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
            // f16/bf16 are represented as f32 in this backend (see zeros_impl/upload invariant), so any
            // conversion among the float dtypes is a representation no-op -- the f32 buffer already holds
            // full-precision values. Just contiguous-ize. This eliminates the GPU->CPU->GPU round-trip
            // that `to_dtype F32<->BF16` otherwise incurs on every layer (the dominant decode/prefill cost).
            (
                DType::F32 | DType::F16 | DType::BF16,
                DType::F32 | DType::F16 | DType::BF16,
            ) => self.contiguous(layout),
            (DType::U32, DType::U32) => self.contiguous(layout),
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
                self.device.profile_fallback(
                    "to_dtype",
                    format_args!("{:?}->{:?} elems={n}", self.dtype, dtype),
                );
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
                self.device.profile_fallback(
                    B::NAME,
                    format_args!("unary elems={}", layout.shape().elem_count()),
                );
                let cpu = self.to_cpu_storage()?;
                let r = crate::backend::BackendStorage::unary_impl::<B>(&cpu, layout)?;
                return self.device.storage_from_cpu_storage(&r);
            }
        };
        let mut ck = None;
        let cb = self.contig_buf(layout, &mut ck)?;
        let n = layout.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let push = (n as u32).to_ne_bytes();
        self.device
            .dispatch(kernel, &[cb, out.buffer], &push, Self::groups_1d(n))?;
        Ok(out)
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let kernel: &'static str = match B::NAME {
            "add" => "add",
            "sub" => "sub",
            "mul" => "mul",
            "div" => "div",
            "maximum" => "maximum",
            "minimum" => "minimum",
            // Anything still without a SPIR-V kernel: fall back to CPU (correct, slow).
            _ => {
                self.device.profile_fallback(
                    B::NAME,
                    format_args!("binary elems={}", lhs_l.shape().elem_count()),
                );
                let lc = self.to_cpu_storage()?;
                let rc = rhs.to_cpu_storage()?;
                let r = crate::backend::BackendStorage::binary_impl::<B>(&lc, &rc, lhs_l, rhs_l)?;
                return self.device.storage_from_cpu_storage(&r);
            }
        };
        // hanzo-ml pre-broadcasts both layouts to the output shape (possibly with stride-0 dims);
        // broadcast layouts aren't contiguous so contig_buf still materializes them, but
        // same-shape contiguous operands skip the copy (one fewer dispatch each).
        let mut lk = None;
        let mut rk = None;
        let lb = self.contig_buf(lhs_l, &mut lk)?;
        let rb = rhs.contig_buf(rhs_l, &mut rk)?;
        let n = lhs_l.shape().elem_count();
        let out = self.device.alloc_f32(n)?;
        let push = (n as u32).to_ne_bytes();
        self.device
            .dispatch(kernel, &[lb, rb, out.buffer], &push, Self::groups_1d(n))?;
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
        // self is the condition mask (u32). The kernel reads it linearly, so materialize a
        // contiguous, offset-0 copy when the condition layout isn't already one.
        if self.dtype != DType::U32 {
            crate::bail!(
                "vulkan: where_cond requires u32 condition, got {:?}",
                self.dtype
            );
        }
        let mut condk = None;
        let condb = if l.is_contiguous() && l.start_offset() == 0 {
            self.buffer
        } else {
            let s = self.contiguous_u32(l)?;
            let b = s.buffer;
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
        let push = push_u32(&[n as u32]);
        self.device.dispatch(
            "where_cond",
            &[condb, tb, fb, out.buffer],
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
            p.b_size as u32,
            p.c_in as u32,
            p.c_out as u32,
            p.l_in as u32,
            l_out as u32,
            p.k_size as u32,
            p.padding as u32,
            p.stride as u32,
            p.dilation as u32,
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
            p.b_size as u32,
            p.c_in as u32,
            p.c_out as u32,
            p.l_in as u32,
            l_out as u32,
            p.k_size as u32,
            p.padding as u32,
            p.stride as u32,
            p.dilation as u32,
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
            p.b_size as u32,
            p.c_in as u32,
            p.c_out as u32,
            p.i_h as u32,
            p.i_w as u32,
            oh as u32,
            ow as u32,
            p.k_h as u32,
            p.k_w as u32,
            p.padding as u32,
            p.stride as u32,
            p.dilation as u32,
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
            p.b_size as u32,
            p.c_in as u32,
            p.c_out as u32,
            p.i_h as u32,
            p.i_w as u32,
            oh as u32,
            ow as u32,
            p.k_h as u32,
            p.k_w as u32,
            p.padding as u32,
            p.stride as u32,
            p.dilation as u32,
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
            &push_u32(&[
                b as u32, c as u32, ih as u32, iw as u32, oh as u32, ow as u32,
            ]),
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
            if oh > 1 {
                (ih - 1) as f64 / (oh - 1) as f64
            } else {
                0.0
            }
        } else {
            scale_h.map(|s| 1.0 / s).unwrap_or(ih as f64 / oh as f64)
        };
        let sw = if align_corners {
            if ow > 1 {
                (iw - 1) as f64 / (ow - 1) as f64
            } else {
                0.0
            }
        } else {
            scale_w.map(|s| 1.0 / s).unwrap_or(iw as f64 / ow as f64)
        };
        let out = self.device.alloc_f32(b * c * oh * ow)?;
        let mut push = push_u32(&[
            b as u32,
            c as u32,
            ih as u32,
            iw as u32,
            oh as u32,
            ow as u32,
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
        // The kernel reads ids linearly (ids[i]), so materialize a contiguous, offset-0 copy when
        // the ids layout isn't already one.
        let mut idk = None;
        let ids_buf = if ids_l.is_contiguous() && ids_l.start_offset() == 0 {
            ids.buffer
        } else {
            let s = ids.contiguous_u32(ids_l)?;
            let b = s.buffer;
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
        let push = push_u32(&[left as u32, dim_size as u32, right as u32, n_ids as u32]);
        self.device.dispatch(
            "index_select",
            &[ids_buf, src_c.buffer, out.buffer],
            &push,
            Self::groups_1d(total),
        )?;
        Ok(out)
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
        // index_add: out = self.clone(); out[.., ids[i], ..] += src[.., i, ..] along `dim`. ids is
        // 1D of length src.dims()[dim]. We reuse the device scatter_add kernel, which wants ids
        // shaped like src, so broadcast the 1D ids across src's pre/post dims first. ids is tiny
        // (one entry per src dim slot) so the broadcast is built on the host and re-uploaded.
        if ids.dtype != DType::U32 {
            crate::bail!("vulkan: index_add requires u32 ids, got {:?}", ids.dtype);
        }
        let ids_host = ids.to_vec_u32()?;
        let ids_host = match ids_l.contiguous_offsets() {
            Some((a, b)) => &ids_host[a..b],
            None => crate::bail!("vulkan: index_add requires contiguous ids"),
        };
        let src_dims = src_l.dims();
        let dim_src = src_dims[dim];
        if ids_host.len() != dim_src {
            crate::bail!(
                "vulkan: index_add ids len {} != src dim {dim} size {dim_src}",
                ids_host.len()
            );
        }
        let pre: usize = src_dims[..dim].iter().product();
        let post: usize = src_dims[dim + 1..].iter().product();
        let mut ids_full = vec![0u32; pre * dim_src * post];
        for p in 0..pre {
            for (s, &id) in ids_host.iter().enumerate() {
                let base = (p * dim_src + s) * post;
                for r in 0..post {
                    ids_full[base + r] = id;
                }
            }
        }
        let ids_full = self.device.upload_u32(&ids_full)?;
        let ids_full_layout = Layout::contiguous(src_dims);
        // Fresh contiguous copy of self to accumulate into (scatter writes binding 0 in place).
        let mut out = self.contiguous(l)?;
        out.scatter_add_set(
            &Layout::contiguous(l.dims()),
            &ids_full,
            &ids_full_layout,
            src,
            src_l,
            dim,
        )?;
        Ok(out)
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
            self.buffer
        } else {
            let s = self.contiguous(lhs_l)?;
            let bf = s.buffer;
            lkeep = Some(s);
            bf
        };
        // rhs is [b,k,n]. A Linear passes W.t() where W is contiguous [b,n,k]; detect that
        // transposed-contiguous layout and feed W's natural [n,k] buffer to an NT kernel -- skipping
        // the transpose strided_copy, which the bench showed is 14-23x the matmul itself and the
        // dominant per-matmul cost in a forward. Otherwise use the buffer directly (already packed)
        // or materialize a [b,k,n] copy.
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
            rhs.buffer
        } else {
            let s = rhs.contiguous(rhs_l)?;
            let bf = s.buffer;
            rkeep = Some(s);
            bf
        };
        let _ = (&lkeep, &rkeep); // keep any materialized copies alive until the dispatch is recorded
                                  // C[b,m,n] = A[b,m,k] * B[b,k,n] (B = W[n,k]^T when nt), row-major. Push order {batch,m,k,n}.
        let out = self.device.alloc_f32(b * m * n)?;
        let push = push_u32(&[b as u32, m as u32, k as u32, n as u32]);

        // Matrix-core path: 16x16x16 fp16 coopmat when every tile dim is a multiple of 16. Casts
        // operands to fp16 (as llama.cpp does) and runs the register-blocked WMMA GEMM.
        //
        // The fp16 scratch (a16 + b16) is an extra, transient 2 B/elem allocation on top of the
        // already-resident operands and output. On the UMA part a large model already fills most of
        // the GTT heap, so a big GEMM's scratch can be the allocation that tips over the edge. Guard
        // it: if the two scratch buffers won't fit in the largest usable heap's *free* bytes (plus a
        // margin), fall through to the fp32 tiled GEMM, which needs no extra f16 buffers — correct
        // result, just slower, instead of an OOM abort.
        let scratch_bytes = ((b * m * k * 2).max(4) as u64).saturating_add((b * k * n * 2).max(4) as u64);
        if self.device.inner.cm_use
            && self.device.scratch_fits(scratch_bytes)
            && matches!(self.device.coopmat_info(), Some((16, 16, 16)))
            && m % 16 == 0
            && n % 16 == 0
            && k % 16 == 0
        {
            let (a16, a16_mem, a16_hv, a16_bytes) = self.device.alloc_f16(b * m * k)?;
            let (b16, b16_mem, b16_hv, b16_bytes) = self.device.alloc_f16(b * k * n)?;
            self.device.dispatch(
                "cast_f2h",
                &[lc_buf, a16],
                &push_u32(&[(b * m * k) as u32]),
                Self::groups_1d(b * m * k),
            )?;
            self.device.dispatch(
                "cast_f2h",
                &[rc_buf, b16],
                &push_u32(&[(b * k * n) as u32]),
                Self::groups_1d(b * k * n),
            )?;
            let mt = (m / 16) as u32;
            let nt_tiles = (n / 16) as u32;
            let groups = (nt_tiles.div_ceil(4), mt.div_ceil(4), b as u32);
            let kernel = if nt {
                "bmm_coopmat_rb_nt"
            } else {
                "bmm_coopmat_rb"
            };
            self.device
                .dispatch(kernel, &[a16, b16, out.buffer], &push, groups)?;
            // Scratch is dead after the kernel reads it; return to the pool (reclaimed post-flush).
            self.device.free_scratch(a16_bytes, a16, a16_mem, a16_hv);
            self.device.free_scratch(b16_bytes, b16, b16_mem, b16_hv);
            return Ok(out);
        }

        // Register-blocked fp32 tiled GEMM (64x64 tile, 4x4 per thread); NT variant reads W[n,k].
        let groups = ((n as u32).div_ceil(64), (m as u32).div_ceil(64), b as u32);
        let kernel = if nt { "bmm_reg_nt" } else { "bmm_reg" };
        self.device
            .dispatch(kernel, &[lc_buf, rc_buf, out.buffer], &push, groups)?;
        Ok(out)
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let n = src_l.shape().elem_count();
        if n == 0 {
            return Ok(());
        }
        if dst.count() < dst_offset + n {
            crate::bail!(
                "vulkan: copy_strided_src dst too small ({} < {})",
                dst.count(),
                dst_offset + n
            );
        }
        // Materialize src_l (any strided/broadcast layout) into dst starting at dst_offset (the
        // KV-cache append pattern: write new tokens' K/V at an offset into the cache buffer).
        let dims = src_l.dims();
        let rank = dims.len();
        if rank > 6 {
            crate::bail!("vulkan: copy_strided_src supports rank <= 6, got {rank}");
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
            &[self.buffer, dst.buffer],
            &push_u32(&p),
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
        // Native on-GPU 2D strided block copy: d1 rows of d2 contiguous elems, per-row
        // src/dst strides + base offsets. This is the cat / slice_set primitive (KV-cache
        // append + GQA repeat_kv every layer), previously the single hottest GPU<->CPU
        // round-trip in the forward (each call read both buffers to the host, copied on CPU,
        // re-uploaded). The `copy2d` kernel is uint-typed so it's bit-exact for f32 AND u32
        // storage. The kernel only writes the addressed elements, leaving the rest of `dst`
        // intact, exactly like the previous read-modify-write, so successive copies compose.
        let total = d1 * d2;
        if total == 0 {
            return Ok(());
        }
        if self.device.inner.profile {
            eprintln!(
                "[HANZO_VK_PROFILE] copy2d(native): d1={d1} d2={d2} elems={total} \
                 (was a CPU round-trip; now on-GPU)"
            );
        }
        let push = push_u32(&[
            d1 as u32,
            d2 as u32,
            src_stride1 as u32,
            dst_stride1 as u32,
            src_offset as u32,
            dst_offset as u32,
        ]);
        self.device.dispatch(
            "copy2d",
            &[self.buffer, dst.buffer],
            &push,
            Self::groups_1d(total),
        )
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        // Fast path: a contiguous, offset-0 view that covers the WHOLE buffer (the
        // Tensor::full / ones / fill primitive) is filled on-GPU by const_fill, no readback.
        // The value is passed as raw 32-bit bits so one uint kernel serves f32 and u32 storage
        // bit-exactly. Every element is written, so nothing outside the addressed set exists to
        // preserve -- the read-modify-write the slow path needs is unnecessary here.
        let n = layout.shape().elem_count();
        if layout.is_contiguous() && layout.start_offset() == 0 && n == self.count {
            let bits: u32 = if self.dtype == DType::U32 || self.dtype == DType::U8 {
                s.to_f64() as u32
            } else {
                (s.to_f64() as f32).to_bits()
            };
            return self.device.dispatch(
                "const_fill",
                &[self.buffer],
                &push_u32(&[n as u32, bits]),
                Self::groups_1d(n),
            );
        }
        // Slow path (partial / strided / offset view): host-visible read-modify-write so elements
        // outside the addressed set are preserved. u32 storage is set as an integer; everything
        // else (f32, and f16/bf16 which live as f32) as a float. Named for the profiler since it
        // still round-trips.
        self.device.profile_fallback(
            "const_set",
            format_args!("elems={n} buf={} (partial/strided)", self.count),
        );
        if self.dtype == DType::U32 || self.dtype == DType::U8 {
            let v = s.to_f64() as u32;
            let mut data = self.to_vec_u32()?;
            for i in layout.strided_index() {
                if i >= data.len() {
                    crate::bail!("vulkan: const_set out of range");
                }
                data[i] = v;
            }
            unsafe {
                self.device
                    .write_u32(self.buffer, self.memory, self.host_visible, &data)
            }
        } else {
            let v = s.to_f64() as f32;
            let mut data = self.to_vec_f32()?;
            for i in layout.strided_index() {
                if i >= data.len() {
                    crate::bail!("vulkan: const_set out of range");
                }
                data[i] = v;
            }
            unsafe {
                self.device
                    .write_f32(self.buffer, self.memory, self.host_visible, &data)
            }
        }
    }
}
