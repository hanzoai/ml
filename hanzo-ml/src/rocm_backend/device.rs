use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, Layout, Result, Shape};
use half::{bf16, f16};
use hanzo_rocm_kernels::compile::KernelCache;
use std::sync::{Arc, Mutex, RwLock};

#[cfg(feature = "rocm-miopen")]
use super::wrappers::SendSyncMIOpenHandle;
use super::wrappers::{
    DevicePool, SendSyncDeviceMemory, SendSyncPseudoRng, SendSyncRocblasHandle, SendSyncStream,
};
use super::{Affine, RocmError, RocmStorage, RocmStorageSlice};
use rocm_rs::hip::Device as HipDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

#[derive(Clone)]
pub struct RocmDevice {
    id: DeviceId,
    device: Arc<HipDevice>,
    pub(crate) stream: Arc<SendSyncStream>,
    rocrand: Arc<Mutex<Option<SendSyncPseudoRng>>>,
    seed_value: Arc<RwLock<u64>>,
    pub(crate) blas: Arc<SendSyncRocblasHandle>,
    #[cfg(feature = "rocm-miopen")]
    pub(crate) miopen: Arc<SendSyncMIOpenHandle>,
    kernel_manager: Arc<Mutex<KernelCache>>,
    /// Caching allocator pool (avoids the synchronizing per-op hipMalloc).
    pool: DevicePool,
}

impl std::fmt::Debug for RocmDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RocmDevice({:?})", self.id)
    }
}

/// Fixed rocBLAS device workspace (bytes). By default rocBLAS manages its own GEMM scratch and
/// `hipMalloc`s it lazily on first use; that allocation lands OUTSIDE our caching pool and, when it
/// happens during a hipGraph capture, gets baked into a graph node yet is freed/reused across
/// replays -> corrupted decode (the MoE F32 router-gate matmul is the one dense rocBLAS op in the
/// captured forward). Pinning a single process-lifetime workspace makes rocBLAS reuse it for every
/// GEMM, so no per-call `hipMalloc` ever escapes into a captured graph. 64 MiB comfortably covers
/// gemm_ex Standard on every shape this engine issues.
const ROCBLAS_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;

/// Pin a fixed, process-lifetime rocBLAS workspace on `blas` (see [`ROCBLAS_WORKSPACE_BYTES`]). The
/// buffer is a raw `hipMalloc` deliberately leaked for the process lifetime, mirroring the leaked
/// rocBLAS handle: rocBLAS frees any prior managed workspace when we set ours, so it must outlive
/// the handle, and the OS reclaims it at exit anyway.
fn pin_rocblas_workspace(blas: &SendSyncRocblasHandle) -> Result<()> {
    use rocm_rs::hip::bindings;
    use rocm_rs::rocblas::ffi;
    let mut ptr = std::ptr::null_mut();
    let err = unsafe { bindings::hipMalloc(&mut ptr, ROCBLAS_WORKSPACE_BYTES) };
    if err != bindings::hipError_t_hipSuccess {
        return Err(crate::Error::Msg(format!(
            "Failed to allocate rocBLAS workspace: {err:?}"
        )));
    }
    let status = unsafe { ffi::rocblas_set_workspace(blas.as_raw(), ptr, ROCBLAS_WORKSPACE_BYTES) };
    if status != ffi::rocblas_status__rocblas_status_success {
        unsafe {
            let _ = bindings::hipFree(ptr);
        }
        return Err(RocmError::Rocblas(format!("rocblas_set_workspace failed: {status}")).into());
    }
    Ok(())
}

impl RocmDevice {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = HipDevice::new(device_id as i32)?;
        device.set_current()?;
        let stream = device.get_stream()?;

        // rocrand's generator destructor SIGSEGVs under WSL (rocRAND poisson manager teardown).
        // Inference samples on the host, so create the generator lazily and only if a device RNG
        // op actually runs; left unused it stays None and never hits the broken destructor.
        let seed = 299792458u64;

        let blas = SendSyncRocblasHandle::new().map_err(|e| RocmError::Rocblas(e.to_string()))?;
        blas.set_stream(&stream)
            .map_err(|e| RocmError::Rocblas(e.to_string()))?;
        pin_rocblas_workspace(&blas)?;

        #[cfg(feature = "rocm-miopen")]
        let miopen =
            SendSyncMIOpenHandle::new(&stream).map_err(|e| RocmError::MIOpen(e.to_string()))?;

        let kernel_manager =
            Arc::new(Mutex::new(KernelCache::new(&device).map_err(|e| {
                crate::Error::Msg(format!("Failed to create kernel cache: {}", e))
            })?));

        // rocBLAS (and MIOpen) handle destructors SIGSEGV under WSL at process teardown. They are
        // process-lifetime handles, so leak one Arc ref to skip the broken destructor; the OS
        // reclaims the device on exit anyway.
        let blas = Arc::new(blas);
        std::mem::forget(blas.clone());
        #[cfg(feature = "rocm-miopen")]
        let miopen = {
            let miopen = Arc::new(miopen);
            std::mem::forget(miopen.clone());
            miopen
        };

        Ok(Self {
            id: DeviceId::new(),
            device: Arc::new(device),
            stream: Arc::new(SendSyncStream(stream)),
            rocrand: Arc::new(Mutex::new(None)),
            seed_value: Arc::new(RwLock::new(seed)),
            blas,
            #[cfg(feature = "rocm-miopen")]
            miopen,
            kernel_manager,
            pool: Arc::new(Mutex::new(super::wrappers::PoolInner::default())),
        })
    }

    fn lock_rng(&self) -> Result<std::sync::MutexGuard<'_, Option<SendSyncPseudoRng>>> {
        let mut guard = self.rocrand.lock().unwrap();
        if guard.is_none() {
            let mut rng = SendSyncPseudoRng::new(rocm_rs::rocrand::rng_type::PSEUDO_DEFAULT)
                .map_err(|e| {
                    crate::Error::Msg(format!("Failed to create rocrand generator: {}", e))
                })?;
            rng.set_seed(*self.seed_value.read().unwrap())
                .map_err(|e| crate::Error::Msg(format!("Failed to set rocrand seed: {}", e)))?;
            *guard = Some(rng);
        }
        Ok(guard)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn ordinal(&self) -> i32 {
        self.device.id()
    }

    /// True for APUs (gfx1151 Strix Halo etc.) where GPU and CPU share physical RAM. The device
    /// mapper uses this to treat total VRAM as unified instead of slicing it per-layer.
    pub fn is_integrated(&self) -> bool {
        rocm_rs::hip::get_device_properties(self.device.id())
            .map(|p| p.integrated != 0)
            .unwrap_or(false)
    }

    pub fn alloc<T>(&self, len: usize) -> Result<SendSyncDeviceMemory<T>> {
        SendSyncDeviceMemory::new_pooled(len, Some(self.pool.clone()))
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))
    }

    pub fn alloc_zeros<T: Default + Clone>(&self, len: usize) -> Result<SendSyncDeviceMemory<T>> {
        let mut mem = SendSyncDeviceMemory::new_pooled(len, Some(self.pool.clone()))
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))?;
        // Capture-safe: enqueue the zero-fill asynchronously on the device's single
        // stream. The synchronizing `hipMemset` trips hipGraph capture (HIP 906);
        // `hipMemsetAsync` is recordable, and single-stream ordering keeps the math
        // identical to the prior blocking version.
        mem.memset_async(0, self.stream.0.as_raw())
            .map_err(|e| crate::Error::Msg(format!("Failed to memset: {}", e)))?;
        Ok(mem)
    }

    pub fn clone_htod<T: Clone>(&self, src: &[T]) -> Result<SendSyncDeviceMemory<T>> {
        let count = src.len();
        let mut dst = SendSyncDeviceMemory::new_pooled(count, Some(self.pool.clone()))
            .map_err(|e| crate::Error::Msg(format!("Failed to allocate ROCm memory: {}", e)))?;
        dst.copy_from_host(src)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy host to device: {}", e)))?;
        Ok(dst)
    }

    pub fn clone_dtoh<T: Default + Clone>(&self, src: &SendSyncDeviceMemory<T>) -> Result<Vec<T>> {
        let count = src.count();
        let mut dst: Vec<T> = vec![T::default(); count];
        src.copy_to_host(&mut dst)
            .map_err(|e| crate::Error::Msg(format!("Failed to copy device to host: {}", e)))?;
        Ok(dst)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| crate::Error::Msg(format!("Synchronize failed: {}", e)))
    }

    /// Open a hipGraph-capture reservation scope on the caching pool. Buffers
    /// allocated while the scope is open are reserved (never recycled) on Drop so
    /// the captured graph's device pointers can never be handed to a later
    /// allocation — preventing the stale-replay corruption. Must be paired with
    /// [`Self::end_graph_capture_scope`]. See `wrappers::PoolInner`.
    pub fn begin_graph_capture_scope(&self) {
        if let Ok(mut inner) = self.pool.lock() {
            inner.begin_capture();
        }
    }

    /// Close a hipGraph-capture reservation scope opened by
    /// [`Self::begin_graph_capture_scope`].
    pub fn end_graph_capture_scope(&self) {
        if let Ok(mut inner) = self.pool.lock() {
            inner.end_capture();
        }
    }

    pub(crate) fn kernel_manager(&self) -> &std::sync::Mutex<KernelCache> {
        &self.kernel_manager
    }

    #[cfg(feature = "rocm-miopen")]
    pub(crate) fn miopen(&self) -> &Arc<SendSyncMIOpenHandle> {
        &self.miopen
    }

    /// Get a reference to the underlying HIP stream.
    /// This is public so that hanzo-nn and other crates can launch custom kernels.
    pub fn stream(&self) -> &rocm_rs::hip::Stream {
        &self.stream.0
    }

    /// Get or load a kernel function from the cache.
    /// This is public so that hanzo-nn and other crates can launch custom kernels.
    pub fn get_or_load_func(
        &self,
        kernel_name: &'static str,
        source: &'static str,
    ) -> crate::Result<rocm_rs::hip::Function> {
        let raw = {
            let kernel_manager = self
                .kernel_manager
                .lock()
                .map_err(|_| crate::Error::Msg("Failed to lock kernel manager".to_string()))?;
            kernel_manager
                .get_func_raw(kernel_name, source, kernel_name)
                .map_err(|e| crate::Error::Msg(e.to_string()))?
        };
        Ok(unsafe { rocm_rs::hip::Function::from_raw(raw as _) })
    }
}

macro_rules! dispatch_dtypes {
    ($method:ident, ($self:expr, $elem_count:expr, $dtype:expr) -> |$slice:ident| $body:expr) => {
        match $dtype {
            DType::U8 => {
                let $slice = RocmStorageSlice::U8($self.$method::<u8>($elem_count)?);
                $body
            }
            DType::U32 => {
                let $slice = RocmStorageSlice::U32($self.$method::<u32>($elem_count)?);
                $body
            }
            DType::I16 => {
                let $slice = RocmStorageSlice::I16($self.$method::<i16>($elem_count)?);
                $body
            }
            DType::I32 => {
                let $slice = RocmStorageSlice::I32($self.$method::<i32>($elem_count)?);
                $body
            }
            DType::I64 => {
                let $slice = RocmStorageSlice::I64($self.$method::<i64>($elem_count)?);
                $body
            }
            DType::BF16 => {
                let $slice = RocmStorageSlice::BF16($self.$method::<bf16>($elem_count)?);
                $body
            }
            DType::F16 => {
                let $slice = RocmStorageSlice::F16($self.$method::<f16>($elem_count)?);
                $body
            }
            DType::F32 => {
                let $slice = RocmStorageSlice::F32($self.$method::<f32>($elem_count)?);
                $body
            }
            DType::F64 => {
                let $slice = RocmStorageSlice::F64($self.$method::<f64>($elem_count)?);
                $body
            }
            DType::F8E4M3 => {
                let $slice = RocmStorageSlice::F8E4M3($self.$method::<u8>($elem_count)?);
                $body
            }
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported for ROCm",
                    $dtype
                )));
            }
        }
    };
}

macro_rules! dispatch_cpu_storage {
    ($storage:expr, $self:expr, |$data:ident, $variant:ident| $body:expr) => {
        match $storage {
            CpuStorage::U8($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::U8(mem);
                $body
            }
            CpuStorage::U32($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::U32(mem);
                $body
            }
            CpuStorage::I16($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::I16(mem);
                $body
            }
            CpuStorage::I32($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::I32(mem);
                $body
            }
            CpuStorage::I64($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::I64(mem);
                $body
            }
            CpuStorage::BF16($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::BF16(mem);
                $body
            }
            CpuStorage::F16($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::F16(mem);
                $body
            }
            CpuStorage::F32($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::F32(mem);
                $body
            }
            CpuStorage::F64($data) => {
                let mem = $self.clone_htod($data.as_slice())?;
                let $variant = RocmStorageSlice::F64(mem);
                $body
            }
            _ => {
                return Err(crate::Error::Msg(format!(
                    "CpuStorage variant not yet supported for ROCm"
                )));
            }
        }
    };
}

impl BackendDevice for RocmDevice {
    type Storage = RocmStorage;

    fn new(device_id: usize) -> Result<Self> {
        Self::new(device_id)
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Rocm {
            gpu_id: self.device.id() as usize,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        // Same physical GPU (by ordinal) == same device. The per-instance DeviceId
        // counter is too strict: separately-constructed RocmDevices for gpu 0 must
        // compare equal, otherwise matmul rejects two tensors on the same GPU.
        self.device.id() == other.device.id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        dispatch_dtypes!(alloc_zeros, (self, elem_count, dtype) -> |slice| {
            Ok(RocmStorage {
                slice,
                device: self.clone(),
            })
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        dispatch_dtypes!(alloc, (self, elem_count, dtype) -> |slice| {
            Ok(RocmStorage {
                slice,
                device: self.clone(),
            })
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let mem = self.clone_htod(data)?;
        let slice = match T::DTYPE {
            DType::U8 => RocmStorageSlice::U8(unsafe { std::mem::transmute(mem) }),
            DType::U32 => RocmStorageSlice::U32(unsafe { std::mem::transmute(mem) }),
            DType::I16 => RocmStorageSlice::I16(unsafe { std::mem::transmute(mem) }),
            DType::I32 => RocmStorageSlice::I32(unsafe { std::mem::transmute(mem) }),
            DType::I64 => RocmStorageSlice::I64(unsafe { std::mem::transmute(mem) }),
            DType::BF16 => RocmStorageSlice::BF16(unsafe { std::mem::transmute(mem) }),
            DType::F16 => RocmStorageSlice::F16(unsafe { std::mem::transmute(mem) }),
            DType::F32 => RocmStorageSlice::F32(unsafe { std::mem::transmute(mem) }),
            DType::F64 => RocmStorageSlice::F64(unsafe { std::mem::transmute(mem) }),
            DType::F8E4M3 => RocmStorageSlice::F8E4M3(unsafe { std::mem::transmute(mem) }),
            dtype => {
                return Err(crate::Error::Msg(format!(
                    "DType {:?} not yet supported for ROCm storage_from_slice",
                    dtype
                )));
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        dispatch_cpu_storage!(storage, self, |data, slice| {
            Ok(RocmStorage {
                slice,
                device: self.clone(),
            })
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut guard = self.lock_rng()?;
        let rocrand = guard.as_mut().unwrap();
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I16
            | DType::I32
            | DType::I64
            | DType::F16
            | DType::BF16
            | DType::F8E4M3
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "dtype {:?} not supported for rocm rand_uniform",
                    dtype
                )));
            }
            DType::F32 => {
                let mut data = rocm_rs::hip::DeviceMemory::<f32>::new(elem_count)
                    .map_err(|e| crate::Error::Msg(format!("rocm rand alloc failed: {}", e)))?;
                rocrand.generate_uniform(&mut data).map_err(|e| {
                    crate::Error::Msg(format!("rocrand generate_uniform failed: {}", e))
                })?;
                RocmStorageSlice::F32(SendSyncDeviceMemory::from_device_memory(data))
            }
            DType::F64 => {
                let mut data = rocm_rs::hip::DeviceMemory::<f64>::new(elem_count)
                    .map_err(|e| crate::Error::Msg(format!("rocm rand alloc failed: {}", e)))?;
                rocrand.generate_uniform_double(&mut data).map_err(|e| {
                    crate::Error::Msg(format!("rocrand generate_uniform_double failed: {}", e))
                })?;
                RocmStorageSlice::F64(SendSyncDeviceMemory::from_device_memory(data))
            }
        };
        let slice = if lo == 0. && hi == 1.0 {
            slice
        } else {
            let layout = Layout::contiguous(shape);
            Affine(hi - lo, lo).map(&slice, self, &layout)?
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let mut guard = self.lock_rng()?;
        let rocrand = guard.as_mut().unwrap();
        // rocrand can only generate an even number of normal values.
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
            | DType::BF16
            | DType::F8E4M3
            | DType::F6E2M3
            | DType::F6E3M2
            | DType::F4
            | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "dtype {:?} not supported for rocm rand_normal",
                    dtype
                )));
            }
            DType::F32 => {
                let mut data = rocm_rs::hip::DeviceMemory::<f32>::new(elem_count_round)
                    .map_err(|e| crate::Error::Msg(format!("rocm rand alloc failed: {}", e)))?;
                rocrand
                    .generate_normal(&mut data, mean as f32, std as f32)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_normal failed: {}", e))
                    })?;
                RocmStorageSlice::F32(SendSyncDeviceMemory::from_device_memory(data))
            }
            DType::F64 => {
                let mut data = rocm_rs::hip::DeviceMemory::<f64>::new(elem_count_round)
                    .map_err(|e| crate::Error::Msg(format!("rocm rand alloc failed: {}", e)))?;
                rocrand
                    .generate_normal_double(&mut data, mean, std)
                    .map_err(|e| {
                        crate::Error::Msg(format!("rocrand generate_normal_double failed: {}", e))
                    })?;
                RocmStorageSlice::F64(SendSyncDeviceMemory::from_device_memory(data))
            }
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        if let Some(rocrand) = self.rocrand.lock().unwrap().as_mut() {
            rocrand
                .set_seed(seed)
                .map_err(|e| crate::Error::Msg(format!("Failed to set rocrand seed: {}", e)))?;
        }
        *self.seed_value.write().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed_value.read().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        self.synchronize()
    }
}
