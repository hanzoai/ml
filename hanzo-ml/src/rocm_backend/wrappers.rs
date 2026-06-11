use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use rocm_rs::hip::error::Error as HipError;
use rocm_rs::hip::{bindings, DeviceMemory, Stream};
#[cfg(feature = "rocm-miopen")]
use rocm_rs::miopen::Handle;
use rocm_rs::rocrand::PseudoRng;

/// Caching device-memory pool: byte-size -> free device pointers (as usize),
/// plus a hipGraph-capture reservation depth.
///
/// Reusing freed buffers avoids per-op `hipMalloc`, which is a *synchronizing*
/// call on HIP — it blocks the CPU until all pending GPU work completes. Without
/// pooling, every tensor op (each allocates its output) silently serializes the
/// stream, so the GPU sits ~95% idle waiting on the CPU: the WSL-bridge ~1 tok/s
/// killer. This mirrors hanzo-ml-cuda's caching allocator. Buffers are returned to
/// the pool on Drop instead of being freed.
///
/// # hipGraph capture reservation (`capture_depth`)
///
/// A captured decode forward bakes the *device pointer* of every output buffer
/// into the graph's kernel nodes. After capture those buffers are normally
/// dropped back into the freelist and handed out to later eager allocations —
/// but the graph still writes to / reads from those exact pointers on every
/// replay. That aliases the graph's captured storage with unrelated live
/// tensors, so each replay clobbers (and recomputes against) corrupted scratch
/// and the logits never advance: the fluent-but-stale single-token loop.
///
/// CUDA dodges this because its caching allocator is graph-capture aware and
/// keeps captured buffers reserved. The ROCm plain-`hipMalloc` pool was not.
/// While `capture_depth > 0`, every buffer allocated from the pool is *leaked*
/// out of the pool on Drop (its pointer is recorded in `reserved`, neither freed
/// nor recycled) so no later allocation can ever reuse a pointer the in-flight
/// graph captured. The reservation is permanent for the process — acceptable
/// because the decode graph cache is bounded and each bucket's working set is
/// captured exactly once.
pub struct PoolInner {
    /// byte-size -> free device pointers (as usize).
    free: HashMap<usize, Vec<usize>>,
    /// Number of nested in-flight hipGraph captures. While > 0, buffers
    /// allocated from this pool are reserved (leaked) on Drop instead of
    /// recycled, so a captured graph's pointers are never reused.
    capture_depth: usize,
    /// Device pointers reserved by captures (kept out of the freelist forever).
    reserved: Vec<usize>,
}

impl Default for PoolInner {
    fn default() -> Self {
        Self {
            free: HashMap::new(),
            capture_depth: 0,
            reserved: Vec::new(),
        }
    }
}

pub type DevicePool = Arc<Mutex<PoolInner>>;

const MAX_POOLED_PER_SIZE: usize = 64;

/// A device buffer, optionally backed by a caching pool (see [`DevicePool`]).
pub struct SendSyncDeviceMemory<T> {
    ptr: *mut std::ffi::c_void,
    size: usize, // bytes
    pool: Option<DevicePool>,
    /// True when this buffer was allocated while a hipGraph capture was in
    /// flight: on Drop it is reserved (leaked) rather than recycled so the
    /// captured graph's pointer is never handed to another tensor.
    capture_reserved: bool,
    phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for SendSyncDeviceMemory<T> {}
unsafe impl<T: Sync> Sync for SendSyncDeviceMemory<T> {}

impl<T> SendSyncDeviceMemory<T> {
    pub fn new(len: usize) -> Result<Self, HipError> {
        Self::new_pooled(len, None)
    }

    /// Allocate `len` elements, reusing a pooled buffer of the same byte size when
    /// available (avoids the synchronizing `hipMalloc` on the hot path).
    pub fn new_pooled(len: usize, pool: Option<DevicePool>) -> Result<Self, HipError> {
        let size = len * std::mem::size_of::<T>();
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                size: 0,
                pool,
                capture_reserved: false,
                phantom: PhantomData,
            });
        }
        // Whether a hipGraph capture is in flight. Read under the same lock we use
        // to pop a free buffer so the decision is consistent with the pool state.
        let mut capture_reserved = false;
        if let Some(p) = &pool {
            if let Ok(mut inner) = p.lock() {
                capture_reserved = inner.capture_depth > 0;
                if let Some(v) = inner.free.get_mut(&size) {
                    if let Some(raw) = v.pop() {
                        return Ok(Self {
                            ptr: raw as *mut std::ffi::c_void,
                            size,
                            pool: pool.clone(),
                            capture_reserved,
                            phantom: PhantomData,
                        });
                    }
                }
            }
        }
        let mut ptr = std::ptr::null_mut();
        let err = unsafe { bindings::hipMalloc(&mut ptr, size) };
        if err != bindings::hipError_t_hipSuccess {
            return Err(HipError::new(err));
        }
        Ok(Self {
            ptr,
            size,
            pool,
            capture_reserved,
            phantom: PhantomData,
        })
    }

    /// Take ownership of a rocm-rs `DeviceMemory` (e.g. a rocrand output buffer).
    /// Not pooled — freed on Drop. Lets us reuse rocrand's typed fill API without
    /// reimplementing it.
    pub fn from_device_memory(mem: DeviceMemory<T>) -> Self {
        let ptr = mem.as_ptr();
        let size = mem.size();
        std::mem::forget(mem); // we own the ptr now; our Drop frees it (pool = None)
        Self {
            ptr,
            size,
            pool: None,
            capture_reserved: false,
            phantom: PhantomData,
        }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn count(&self) -> usize {
        let es = std::mem::size_of::<T>();
        if es == 0 {
            0
        } else {
            self.size / es
        }
    }

    /// Pointer advanced by `offset` ELEMENTS (not bytes). The base is a `c_void`
    /// (byte) pointer, so cast to the element type first or `.add` advances by bytes.
    ///
    /// # Safety
    /// `offset` must be within the allocation.
    pub unsafe fn offset_ptr(&self, offset: usize) -> *mut std::ffi::c_void {
        (self.ptr as *mut T).add(offset) as *mut std::ffi::c_void
    }

    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), HipError> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }
        let n = std::cmp::min(self.size, data.len() * std::mem::size_of::<T>());
        let err = unsafe {
            bindings::hipMemcpy(
                self.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                n,
                bindings::hipMemcpyKind_hipMemcpyHostToDevice,
            )
        };
        if err != bindings::hipError_t_hipSuccess {
            return Err(HipError::new(err));
        }
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), HipError> {
        if self.ptr.is_null() || data.is_empty() {
            return Ok(());
        }
        let n = std::cmp::min(self.size, data.len() * std::mem::size_of::<T>());
        let err = unsafe {
            bindings::hipMemcpy(
                data.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                n,
                bindings::hipMemcpyKind_hipMemcpyDeviceToHost,
            )
        };
        if err != bindings::hipError_t_hipSuccess {
            return Err(HipError::new(err));
        }
        Ok(())
    }

    pub fn copy_from_device(&mut self, src: &Self) -> Result<(), HipError> {
        if self.ptr.is_null() || src.ptr.is_null() {
            return Ok(());
        }
        let n = std::cmp::min(self.size, src.size);
        let err = unsafe {
            bindings::hipMemcpy(
                self.ptr,
                src.ptr,
                n,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
            )
        };
        if err != bindings::hipError_t_hipSuccess {
            return Err(HipError::new(err));
        }
        Ok(())
    }

    pub fn memset(&mut self, value: i32) -> Result<(), HipError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let err = unsafe { bindings::hipMemset(self.ptr, value, self.size) };
        if err != bindings::hipError_t_hipSuccess {
            return Err(HipError::new(err));
        }
        Ok(())
    }

    /// Capture-safe memset: enqueues `hipMemsetAsync` on `stream` instead of the
    /// synchronizing `hipMemset`. The destination is device memory, so this op is
    /// fully recordable inside a hipGraph capture (no host-side sync). `stream` is
    /// the raw `hipStream_t` from `device.stream().as_raw()`. Ordering on the
    /// backend's single stream preserves correctness.
    pub fn memset_async(
        &mut self,
        value: i32,
        stream: bindings::hipStream_t,
    ) -> Result<(), HipError> {
        if self.ptr.is_null() {
            return Ok(());
        }
        let err = unsafe { bindings::hipMemsetAsync(self.ptr, value, self.size, stream) };
        if err != bindings::hipError_t_hipSuccess {
            return Err(HipError::new(err));
        }
        Ok(())
    }
}

impl<T> Drop for SendSyncDeviceMemory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Some(p) = &self.pool {
            if let Ok(mut inner) = p.lock() {
                if self.capture_reserved {
                    // This pointer was baked into an in-flight hipGraph capture.
                    // Reserve it permanently (never freed, never recycled) so no
                    // later allocation can alias the graph's captured storage and
                    // corrupt a replay. See [`PoolInner`].
                    inner.reserved.push(self.ptr as usize);
                    return;
                }
                let v = inner.free.entry(self.size).or_default();
                if v.len() < MAX_POOLED_PER_SIZE {
                    v.push(self.ptr as usize);
                    return;
                }
            }
        }
        unsafe {
            let _ = bindings::hipFree(self.ptr);
        }
    }
}

impl PoolInner {
    /// Enter a hipGraph capture scope: every buffer allocated from this pool
    /// while the scope is open is reserved (leaked) on Drop instead of being
    /// recycled, so the captured graph's device pointers are never reused.
    pub fn begin_capture(&mut self) {
        self.capture_depth += 1;
    }

    /// Leave a hipGraph capture scope.
    pub fn end_capture(&mut self) {
        self.capture_depth = self.capture_depth.saturating_sub(1);
    }
}

pub struct SendSyncStream(pub Stream);

unsafe impl Send for SendSyncStream {}
unsafe impl Sync for SendSyncStream {}

impl Deref for SendSyncStream {
    type Target = Stream;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SendSyncRocblasHandle(pub rocm_rs::rocblas::Handle);

unsafe impl Send for SendSyncRocblasHandle {}
unsafe impl Sync for SendSyncRocblasHandle {}

impl SendSyncRocblasHandle {
    pub fn new() -> Result<Self, rocm_rs::rocblas::error::Error> {
        Ok(Self(rocm_rs::rocblas::Handle::new()?))
    }
}

impl Deref for SendSyncRocblasHandle {
    type Target = rocm_rs::rocblas::Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SendSyncPseudoRng(pub PseudoRng);

unsafe impl Send for SendSyncPseudoRng {}
unsafe impl Sync for SendSyncPseudoRng {}

impl SendSyncPseudoRng {
    pub fn new(rng_type: u32) -> Result<Self, rocm_rs::rocrand::error::Error> {
        Ok(Self(PseudoRng::new(rng_type)?))
    }
}

impl Deref for SendSyncPseudoRng {
    type Target = PseudoRng;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SendSyncPseudoRng {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(feature = "rocm-miopen")]
pub struct SendSyncMIOpenHandle(pub Handle);

#[cfg(feature = "rocm-miopen")]
unsafe impl Send for SendSyncMIOpenHandle {}
#[cfg(feature = "rocm-miopen")]
unsafe impl Sync for SendSyncMIOpenHandle {}

#[cfg(feature = "rocm-miopen")]
impl SendSyncMIOpenHandle {
    pub fn new(stream: &Stream) -> Result<Self, rocm_rs::miopen::error::Error> {
        let handle = Handle::with_stream(stream)?;
        Ok(Self(handle))
    }
}

#[cfg(feature = "rocm-miopen")]
impl Deref for SendSyncMIOpenHandle {
    type Target = Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
