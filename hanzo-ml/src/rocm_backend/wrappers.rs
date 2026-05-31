use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use rocm_rs::hip::error::Error as HipError;
use rocm_rs::hip::{bindings, DeviceMemory, Stream};
use rocm_rs::miopen::Handle;
use rocm_rs::rocrand::PseudoRng;

/// Caching device-memory pool: byte-size -> free device pointers (as usize).
///
/// Reusing freed buffers avoids per-op `hipMalloc`, which is a *synchronizing*
/// call on HIP — it blocks the CPU until all pending GPU work completes. Without
/// pooling, every tensor op (each allocates its output) silently serializes the
/// stream, so the GPU sits ~95% idle waiting on the CPU: the WSL-bridge ~1 tok/s
/// killer. This mirrors candle-cuda's caching allocator. Buffers are returned to
/// the pool on Drop instead of being freed.
pub type DevicePool = Arc<Mutex<HashMap<usize, Vec<usize>>>>;

const MAX_POOLED_PER_SIZE: usize = 64;

/// A device buffer, optionally backed by a caching pool (see [`DevicePool`]).
pub struct SendSyncDeviceMemory<T> {
    ptr: *mut std::ffi::c_void,
    size: usize, // bytes
    pool: Option<DevicePool>,
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
                phantom: PhantomData,
            });
        }
        if let Some(p) = &pool {
            if let Ok(mut map) = p.lock() {
                if let Some(v) = map.get_mut(&size) {
                    if let Some(raw) = v.pop() {
                        return Ok(Self {
                            ptr: raw as *mut std::ffi::c_void,
                            size,
                            pool: pool.clone(),
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
}

impl<T> Drop for SendSyncDeviceMemory<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        if let Some(p) = &self.pool {
            if let Ok(mut map) = p.lock() {
                let v = map.entry(self.size).or_default();
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

pub struct SendSyncMIOpenHandle(pub Handle);

unsafe impl Send for SendSyncMIOpenHandle {}
unsafe impl Sync for SendSyncMIOpenHandle {}

impl SendSyncMIOpenHandle {
    pub fn new(stream: &Stream) -> Result<Self, rocm_rs::miopen::error::Error> {
        let handle = Handle::with_stream(stream)?;
        Ok(Self(handle))
    }
}

impl Deref for SendSyncMIOpenHandle {
    type Target = Handle;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
