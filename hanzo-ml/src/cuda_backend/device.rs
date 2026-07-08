use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use cudarc;
use cudarc::driver::{
    result as cuda_result, sys as cuda_sys, CudaFunction, DevicePtr, UnifiedSlice,
};
use float8::F8E4M3;
use half::{bf16, f16};
pub use hanzo_kernels as kernels;
use std::any::{Any, TypeId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

const CUDA_GRAPH_HTOD_CACHE_MAX_BYTES: usize = 4096;

type CudaGraphHtodCacheKey = (DeviceId, TypeId, Vec<u8>);
type CudaGraphHtodCache = HashMap<CudaGraphHtodCacheKey, Box<dyn Any>>;

thread_local! {
    static CUDA_GRAPH_HTOD_CACHE: RefCell<CudaGraphHtodCache> =
        RefCell::new(HashMap::new());
    static CUDA_GRAPH_HTOD_CACHE_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[must_use]
pub struct CudaGraphHtodCacheGuard;

impl Drop for CudaGraphHtodCacheGuard {
    fn drop(&mut self) {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| {
            debug_assert!(depth.get() > 0);
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

pub struct ModuleStore {
    mdls: [Option<Arc<cudarc::driver::CudaModule>>; kernels::ALL_IDS.len()],
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    context: Arc<cudarc::driver::CudaContext>,
    modules: Arc<std::sync::RwLock<ModuleStore>>,
    custom_modules: Arc<std::sync::RwLock<HashMap<String, Arc<cudarc::driver::CudaModule>>>>,
    stream: Arc<cudarc::driver::CudaStream>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    curand: Arc<Mutex<CudaRng>>,
    seed_value: Arc<RwLock<u64>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl CudaDevice {
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn alloc<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc::<T>(len).w()
    }

    pub fn alloc_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len).w()
    }

    pub fn enable_cuda_graph_htod_cache(&self) -> CudaGraphHtodCacheGuard {
        CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| depth.set(depth.get() + 1));
        CudaGraphHtodCacheGuard
    }

    pub fn memcpy_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        if cuda_graph_htod_cache_enabled() && !src.is_empty() {
            return self.memcpy_htod_capture_cached(src, dst);
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    pub fn clone_dtoh<T: cudarc::driver::DeviceRepr, Src: cudarc::driver::DevicePtr<T>>(
        &self,
        src: &Src,
    ) -> Result<Vec<T>> {
        self.stream.clone_dtoh(src).w()
    }

    pub fn memcpy_dtod<
        T,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtod(src, dst).w()
    }

    pub fn memcpy_dtoh<
        T: cudarc::driver::DeviceRepr,
        Src: cudarc::driver::DevicePtr<T>,
        Dst: cudarc::driver::HostSlice<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        self.stream.memcpy_dtoh(src, dst).w()
    }

    pub fn clone_htod<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if cuda_graph_htod_cache_enabled() && !src.is_empty() {
            return self.clone_htod_capture_cached(src);
        }
        self.stream.clone_htod(src).w()
    }

    fn memcpy_htod_capture_cached<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &Src,
        dst: &mut Dst,
    ) -> Result<()> {
        assert!(dst.len() >= src.len());
        if let Some(cached) = self.cached_htod_slice(src)? {
            return self.stream.memcpy_dtod(&cached, dst).w();
        }
        self.stream.memcpy_htod(src, dst).w()
    }

    fn clone_htod_capture_cached<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        if let Some(cached) = self.cached_htod_slice(src)? {
            return Ok(cached);
        }
        self.stream.clone_htod(src).w()
    }

    fn cached_htod_slice<
        T: cudarc::driver::DeviceRepr + 'static,
        Src: cudarc::driver::HostSlice<T> + ?Sized,
    >(
        &self,
        src: &Src,
    ) -> Result<Option<cudarc::driver::CudaSlice<T>>> {
        let (src_slice, _sync) = unsafe { src.stream_synced_slice(&self.stream) };
        let byte_len = std::mem::size_of_val(src_slice);
        if byte_len > CUDA_GRAPH_HTOD_CACHE_MAX_BYTES {
            if self.cuda_graph_capture_active() {
                crate::bail!("CUDA graph capture cannot upload uncached host data");
            }
            return Ok(None);
        }

        let bytes =
            unsafe { std::slice::from_raw_parts(src_slice.as_ptr().cast::<u8>(), byte_len) }
                .to_vec();
        let key = (self.id, TypeId::of::<T>(), bytes);
        if let Some(cached) = CUDA_GRAPH_HTOD_CACHE.with(|cache| {
            cache.borrow().get(&key).and_then(|cached| {
                cached
                    .downcast_ref::<cudarc::driver::CudaSlice<T>>()
                    .cloned()
            })
        }) {
            return Ok(Some(cached));
        }
        if self.cuda_graph_capture_active() {
            crate::bail!("CUDA graph capture missing cached host data");
        }

        let cached = self.stream.clone_htod(src).w()?;
        CUDA_GRAPH_HTOD_CACHE.with(|cache| {
            cache.borrow_mut().insert(key, Box::new(cached.clone()));
        });
        Ok(Some(cached))
    }

    fn cuda_graph_capture_active(&self) -> bool {
        matches!(
            self.stream.capture_status(),
            Ok(status) if status != cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        )
    }
}

fn cuda_graph_htod_cache_enabled() -> bool {
    CUDA_GRAPH_HTOD_CACHE_DEPTH.with(|depth| depth.get() > 0)
}

pub struct CudaFunc {
    func: CudaFunction,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl std::ops::Deref for CudaFunc {
    type Target = CudaFunction;

    fn deref(&self) -> &Self::Target {
        &self.func
    }
}

impl CudaFunc {
    pub fn into_cuda_function(self) -> CudaFunction {
        self.func
    }
}

#[macro_export]
macro_rules! builder_arg {
    ($b:ident, $($arg:expr),*) => {
        $(
            let __arg = $arg;
            $b.arg(&__arg);
        )*
    };
}

impl CudaFunc {
    pub fn builder(&self) -> cudarc::driver::LaunchArgs<'_> {
        self.stream.launch_builder(&self.func)
    }
}

impl CudaDevice {
    pub fn cuda_stream(&self) -> Arc<cudarc::driver::CudaStream> {
        self.stream.clone()
    }

    /// When turned on, all cuda tensors **created after calling this function** will
    /// not track uses via cuda events.
    ///
    /// # Safety
    ///
    /// It is up to the user to ensure proper synchronization between multiple streams:
    /// - Ensure that no tensor is freed before a use on another stream is finished.
    /// - Ensure that a tensor is not used on another stream before allocation on the
    ///   allocating stream finishes.
    /// - Ensure that a tensor is not written two concurrently by multiple streams.
    pub unsafe fn disable_event_tracking(&self) {
        self.context.disable_event_tracking()
    }

    pub fn is_event_tracking(&self) -> bool {
        self.context.is_event_tracking()
    }

    #[cfg(all(feature = "ug", not(target_arch = "wasm32")))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: hanzo_ug::lang::ssa::Kernel,
    ) -> Result<CudaFunc> {
        let mut buf = vec![];
        hanzo_ug::cuda::code_gen::gen(&mut buf, func_name, &kernel)?;
        let cuda_code = String::from_utf8(buf)?;
        let opts = cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::safe::compile_ptx_with_opts(cuda_code, opts).w()?;
        let module = self.context.load_module(ptx).w()?;
        let func = module.load_function(func_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn get_or_load_custom_func(
        &self,
        fn_name: &str,
        module_name: &str,
        ptx: &str,
    ) -> Result<CudaFunc> {
        let ms = self.custom_modules.read().unwrap();
        if let Some(mdl) = ms.get(module_name).as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.custom_modules.write().unwrap();
        let cuda_module = self.context.load_module(ptx.into()).w()?;
        ms.insert(module_name.to_string(), cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn get_or_load_func(&self, fn_name: &str, mdl: &kernels::Module) -> Result<CudaFunc> {
        let ms = self.modules.read().unwrap();
        if let Some(mdl) = ms.mdls[mdl.index()].as_ref() {
            let func = mdl.load_function(fn_name).w()?;
            return Ok(CudaFunc {
                func,
                stream: self.stream.clone(),
            });
        }
        drop(ms);
        let mut ms = self.modules.write().unwrap();
        let cuda_module = self.context.load_module(mdl.ptx().into()).w()?;
        ms.mdls[mdl.index()] = Some(cuda_module.clone());
        let func = cuda_module.load_function(fn_name).w()?;
        Ok(CudaFunc {
            func,
            stream: self.stream.clone(),
        })
    }

    pub fn cublas_handle(&self) -> Arc<cudarc::cublas::CudaBlas> {
        self.blas.clone()
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.new_stream().w()?;
        Self::from_context_and_stream(context, stream)
    }

    fn from_context_and_stream(
        context: Arc<cudarc::driver::CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        let blas = cudarc::cublas::CudaBlas::new(stream.clone()).w()?;
        let curand = cudarc::curand::CudaRng::new(299792458, stream.clone()).w()?;
        let module_store = ModuleStore {
            mdls: [const { None }; kernels::ALL_IDS.len()],
        };
        Ok(Self {
            id: DeviceId::new(),
            context,
            stream,
            blas: Arc::new(blas),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
            modules: Arc::new(std::sync::RwLock::new(module_store)),
            custom_modules: Arc::new(std::sync::RwLock::new(HashMap::new())),
            seed_value: Arc::new(RwLock::new(299792458)),
        })
    }
}

// ---------------------------------------------------------------------------
// Unified / managed memory (CUDA UMA path for GB10-class coherent devices).
//
// A discrete GPU keeps model state in host RAM *and* a separate VRAM copy; on a
// physically unified device (e.g. GB10 DGX Spark, 128 GiB coherent memory) that
// double-counting caps everything >= ~80 GiB. This section adds, additively:
//   * detection of the unified / managed capability (mirrors the ROCm APU
//     `RocmDevice::is_integrated()` precedent),
//   * a size-gated `cuMemAllocManaged`-backed allocation path (cudarc's
//     `UnifiedSlice`, which frees with the correct `cuMemFree` — a managed
//     pointer wrapped in a device-malloc `CudaSlice` would be freed with the
//     invalid `cuMemFreeAsync`), and
//   * `cuMemAdvise` / `cuMemPrefetchAsync` (v2) hint helpers, usable on managed
//     buffers *and* on an mmap'd weight region (the HMM-direct loader seam).
//
// Discrete behaviour is unchanged: a non-unified device reports `is_unified() ==
// false`, `should_use_managed()` returns false, and the managed entry points are
// simply never called.
//
// Toolkit pinning: the v2 advise/prefetch entry points require CUDA >= 12.2, and
// the `CUmemLocation { type_, id }` literal matches the cudarc binding layout for
// CUDA < 13.2. The stack targets CUDA 13.0; on 13.2 (cudarc feature
// `cuda-13020`) the `id` field moves into an anonymous union and this section
// needs a one-line cfg.
// ---------------------------------------------------------------------------

/// Default managed-allocation size gate: a buffer at least this large on a
/// unified device is backed by `cuMemAllocManaged` instead of device-malloc.
/// Mirrors the ds4 reference's ">= 8 GiB KV" rule. Override (bytes) with
/// `ML_CUDA_UMA_THRESHOLD`.
const DEFAULT_UMA_THRESHOLD_BYTES: usize = 8 * 1024 * 1024 * 1024;

/// Process-wide managed-allocation threshold, read once from the environment.
fn uma_threshold_bytes() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("ML_CUDA_UMA_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(DEFAULT_UMA_THRESHOLD_BYTES)
    })
}

/// Pure managed-allocation policy: managed memory is chosen only on a unified
/// device that supports it, for buffers at or above the threshold. Factored out
/// so the decision is testable without a GPU.
fn want_managed(
    is_unified: bool,
    supports_managed: bool,
    num_bytes: usize,
    threshold: usize,
) -> bool {
    is_unified && supports_managed && num_bytes >= threshold
}

impl CudaDevice {
    fn device_attr(&self, attr: cuda_sys::CUdevice_attribute) -> i32 {
        // Fail safe: an unreadable attribute reads as 0 ("not supported"), which
        // keeps the device on the conservative discrete-GPU path.
        self.context.attribute(attr).unwrap_or(0)
    }

    /// True when the device can coherently access pageable host memory
    /// (`cudaDevAttrPageableMemoryAccess`) — i.e. a physically unified / HMM
    /// device such as GB10. This is the signal that the managed and HMM-direct
    /// paths are worthwhile; a discrete GPU returns false and stays on
    /// device-malloc.
    pub fn is_unified(&self) -> bool {
        self.device_attr(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS)
            != 0
    }

    /// True for an integrated GPU that shares physical RAM with the CPU
    /// (`cudaDevAttrIntegrated`). Mirrors `RocmDevice::is_integrated()`.
    pub fn is_integrated(&self) -> bool {
        self.device_attr(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0
    }

    /// True when the device supports `cuMemAllocManaged`
    /// (`cudaDevAttrManagedMemory`).
    pub fn supports_managed_memory(&self) -> bool {
        self.device_attr(cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY) != 0
    }

    /// True when the device may access managed memory concurrently with the host
    /// (`cudaDevAttrConcurrentManagedAccess`); required for device prefetch.
    pub fn concurrent_managed_access(&self) -> bool {
        self.device_attr(
            cuda_sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
        ) != 0
    }

    /// The managed-allocation size gate in bytes (see `ML_CUDA_UMA_THRESHOLD`).
    pub fn unified_alloc_threshold(&self) -> usize {
        uma_threshold_bytes()
    }

    /// Whether a buffer of `num_bytes` should be backed by managed memory on this
    /// device. The single source of truth for the discrete-vs-managed decision;
    /// the loader / storage layer calls this and routes large, long-lived buffers
    /// to [`Self::alloc_unified`].
    pub fn should_use_managed(&self, num_bytes: usize) -> bool {
        want_managed(
            self.is_unified(),
            self.supports_managed_memory(),
            num_bytes,
            uma_threshold_bytes(),
        )
    }

    /// Allocate an uninitialised managed (`cuMemAllocManaged`) buffer attached
    /// globally, so any stream and the host may access it. It is freed with the
    /// correct `cuMemFree` via `UnifiedSlice`'s drop.
    ///
    /// # Safety
    /// The memory is returned uninitialised; it must be written before it is read
    /// (same contract as the device-malloc [`Self::alloc`]).
    pub unsafe fn alloc_unified<T: cudarc::driver::DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<UnifiedSlice<T>> {
        unsafe { self.context.alloc_unified::<T>(len, true) }.w()
    }

    /// Allocate a zeroed managed buffer. The fill runs on the host view of the
    /// coherent buffer, matching the device-malloc `alloc_zeros` result.
    pub fn alloc_unified_zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<UnifiedSlice<T>> {
        // SAFETY: written immediately below before any read.
        let mut buf = unsafe { self.alloc_unified::<T>(len) }?;
        {
            let host = buf.as_mut_slice().w()?;
            let num_bytes = std::mem::size_of_val(host);
            // SAFETY: the all-zero byte pattern is valid for `T: ValidAsZeroBits`,
            // and `num_bytes` is exactly this buffer's extent.
            unsafe { std::ptr::write_bytes(host.as_mut_ptr().cast::<u8>(), 0u8, num_bytes) };
        }
        Ok(buf)
    }

    fn unified_location(&self, to_device: bool) -> cuda_sys::CUmemLocation {
        if to_device {
            cuda_sys::CUmemLocation {
                type_: cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: self.context.ordinal() as i32,
            }
        } else {
            cuda_sys::CUmemLocation {
                type_: cuda_sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_HOST,
                id: 0,
            }
        }
    }

    fn mem_advise_raw(
        &self,
        dptr: cuda_sys::CUdeviceptr,
        num_bytes: usize,
        advice: cuda_sys::CUmem_advise,
        to_device: bool,
    ) -> Result<()> {
        let location = self.unified_location(to_device);
        // SAFETY: the caller guarantees `dptr`/`num_bytes` cover a single managed
        // or HMM-accessible range; `cuMemAdvise` only sets migration hints.
        unsafe { cuda_result::mem_advise(dptr, num_bytes, advice, location) }.w()
    }

    fn mem_prefetch_raw(
        &self,
        dptr: cuda_sys::CUdeviceptr,
        num_bytes: usize,
        to_device: bool,
    ) -> Result<()> {
        let location = self.unified_location(to_device);
        // SAFETY: as `mem_advise_raw`; prefetch is a stream-ordered migration.
        unsafe {
            cuda_result::mem_prefetch_async(dptr, num_bytes, location, self.stream.cu_stream())
        }
        .w()
    }

    /// Hint that a managed buffer is read-mostly (duplicate-on-read, cheap GPU
    /// reads) — the correct advice for read-only model weights.
    pub fn advise_read_mostly<T>(&self, buf: &UnifiedSlice<T>) -> Result<()> {
        let (dptr, _guard) = buf.device_ptr(&self.stream);
        self.mem_advise_raw(
            dptr,
            buf.num_bytes(),
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY,
            true,
        )
    }

    /// Hint that a managed buffer's preferred residency is this device — keeps
    /// hot, write-heavy state (e.g. the KV cache) from migrating back to host.
    pub fn advise_preferred_location_device<T>(&self, buf: &UnifiedSlice<T>) -> Result<()> {
        let (dptr, _guard) = buf.device_ptr(&self.stream);
        self.mem_advise_raw(
            dptr,
            buf.num_bytes(),
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
            true,
        )
    }

    /// Hint that this device accesses a managed buffer, establishing a direct
    /// mapping so reads resolve over the coherent link without faulting.
    pub fn advise_accessed_by_device<T>(&self, buf: &UnifiedSlice<T>) -> Result<()> {
        let (dptr, _guard) = buf.device_ptr(&self.stream);
        self.mem_advise_raw(
            dptr,
            buf.num_bytes(),
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_ACCESSED_BY,
            true,
        )
    }

    /// Stream-ordered prefetch of a managed buffer onto this device. Requires
    /// `concurrent_managed_access()`; otherwise CUDA returns an error.
    pub fn prefetch_to_device<T>(&self, buf: &UnifiedSlice<T>) -> Result<()> {
        let (dptr, _guard) = buf.device_ptr(&self.stream);
        self.mem_prefetch_raw(dptr, buf.num_bytes(), true)
    }

    /// Stream-ordered prefetch of a managed buffer back to host memory.
    pub fn prefetch_to_host<T>(&self, buf: &UnifiedSlice<T>) -> Result<()> {
        let (dptr, _guard) = buf.device_ptr(&self.stream);
        self.mem_prefetch_raw(dptr, buf.num_bytes(), false)
    }

    /// HMM-direct loader seam: advise an mmap'd (pageable) weight region as
    /// read-mostly so the GPU reads it in place over the coherent link, with no
    /// host-to-device copy. Meaningful only on a unified device (`is_unified()`).
    ///
    /// # Safety
    /// `ptr`/`num_bytes` must describe a single, currently-mapped host range
    /// (e.g. an mmap'd safetensors region) that outlives the GPU work using it.
    pub unsafe fn advise_hmm_read_mostly(
        &self,
        ptr: *const c_void,
        num_bytes: usize,
    ) -> Result<()> {
        self.mem_advise_raw(
            ptr as cuda_sys::CUdeviceptr,
            num_bytes,
            cuda_sys::CUmem_advise::CU_MEM_ADVISE_SET_READ_MOSTLY,
            true,
        )
    }

    /// HMM-direct loader seam: prefetch an mmap'd (pageable) weight region onto
    /// this device. Meaningful only on a unified device with concurrent managed
    /// access.
    ///
    /// # Safety
    /// As [`Self::advise_hmm_read_mostly`].
    pub unsafe fn prefetch_hmm_to_device(
        &self,
        ptr: *const c_void,
        num_bytes: usize,
    ) -> Result<()> {
        self.mem_prefetch_raw(ptr as cuda_sys::CUdeviceptr, num_bytes, true)
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let context = cudarc::driver::CudaContext::new(ordinal).w()?;
        let stream = context.per_thread_stream();
        Self::from_context_and_stream(context, stream)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.stream.clone()).w()?;
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.context.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc_zeros::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc_zeros::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count)? };
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_uniform",
                })
                .w()?
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/hanzoai/ml/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round)? };
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round)? };
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                Err(CudaError::UnsupportedDtype {
                    dtype,
                    op: "rand_normal",
                })
                .w()?
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count)?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count)?;
                CudaStorageSlice::U32(data)
            }
            DType::I16 => {
                let data = self.alloc::<i16>(elem_count)?;
                CudaStorageSlice::I16(data)
            }
            DType::I32 => {
                let data = self.alloc::<i32>(elem_count)?;
                CudaStorageSlice::I32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count)?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count)?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count)?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count)?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count)?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<F8E4M3>(elem_count)?;
                CudaStorageSlice::F8E4M3(data)
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(
                    CudaError::InternalError("Dummy types not supported in CUDA backend").into(),
                )
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorageRef::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorageRef::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorageRef::F4(_)
            | CpuStorageRef::F6E2M3(_)
            | CpuStorageRef::F6E3M2(_)
            | CpuStorageRef::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: T::DTYPE,
                    op: "storage_from_slice",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I16(data)
            }
            CpuStorage::I32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F64(data)
            }
            CpuStorage::F8E4M3(storage) => {
                let data = self.clone_htod(&storage)?;
                CudaStorageSlice::F8E4M3(data)
            }
            CpuStorage::F4(_)
            | CpuStorage::F6E2M3(_)
            | CpuStorage::F6E3M2(_)
            | CpuStorage::F8E8M0(_) => {
                return Err(CudaError::UnsupportedDtype {
                    dtype: storage.dtype(),
                    op: "storage_from_cpu_storage_owned",
                }
                .into());
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.stream.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}

#[cfg(test)]
mod uma_tests {
    use super::*;

    // Pure managed-allocation policy — runs without a GPU.
    #[test]
    fn want_managed_policy() {
        let t = DEFAULT_UMA_THRESHOLD_BYTES;
        // Discrete device: never managed, regardless of size.
        assert!(!want_managed(false, true, t, t));
        // Unified but no managed support: never.
        assert!(!want_managed(true, false, t, t));
        // Unified + managed, below the gate: stay on device-malloc.
        assert!(!want_managed(true, true, t - 1, t));
        // Unified + managed, at / above the gate: use managed memory.
        assert!(want_managed(true, true, t, t));
        assert!(want_managed(true, true, t + 1, t));
    }

    // Detection + managed alloc/advise/prefetch roundtrip on real hardware.
    // Skips cleanly when no CUDA device (or no managed support) is present.
    #[test]
    fn unified_alloc_roundtrip_and_hints() {
        let dev = match CudaDevice::new(0) {
            Ok(d) => d,
            Err(_) => return, // no device in this environment
        };

        // Detection must never panic.
        let unified = dev.is_unified();
        let integrated = dev.is_integrated();
        let managed = dev.supports_managed_memory();
        let concurrent = dev.concurrent_managed_access();
        eprintln!(
            "cuda uma: unified={unified} integrated={integrated} managed={managed} \
             concurrent={concurrent} threshold={}",
            dev.unified_alloc_threshold()
        );

        if !managed {
            return; // device cannot allocate managed memory
        }

        let n = 4096usize;
        // SAFETY: written before any read, below.
        let mut buf = unsafe { dev.alloc_unified::<f32>(n) }.unwrap();
        for (i, x) in buf.as_mut_slice().unwrap().iter_mut().enumerate() {
            *x = i as f32;
        }

        // Exercise the v2 advise/prefetch hint path on the live device.
        dev.advise_read_mostly(&buf).unwrap();
        dev.advise_preferred_location_device(&buf).unwrap();
        dev.advise_accessed_by_device(&buf).unwrap();
        if concurrent {
            dev.prefetch_to_device(&buf).unwrap();
        }
        dev.synchronize().unwrap();

        // Read back through the coherent host view: same values.
        for (i, &x) in buf.as_slice().unwrap().iter().enumerate() {
            assert_eq!(x, i as f32);
        }

        // Zeroed managed allocation.
        let zeros = dev.alloc_unified_zeros::<f32>(n).unwrap();
        assert!(zeros.as_slice().unwrap().iter().all(|&x| x == 0.0));
    }
}
