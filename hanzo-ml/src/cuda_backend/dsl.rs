//! CUDA DSL kernel artifact registry, single-compile NVRTC loader, and dispatch seam.
//!
//! Mirrors `metal_backend::kernel_msl` + `dispatch_dsl`: every `.cu` file under `src/cuda/dsl/` is
//! generated from `hanzo-kernel` via `kernel-gen --target cuda --out hanzo-ml/src/cuda/dsl` and
//! drift-checked with `--check`. Compiled once per `(device, artifact)` via NVRTC and cached.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig, PushKernelArg};

use super::device::{CudaDevice, CudaFunc, DeviceId};
use super::{CudaStorage, CudaStorageSlice, WrapErr};
use crate::{DType, Error, Result};

/// A committed CUDA DSL artifact: generated `.cu` source and its `extern "C" __global__` entry name.
pub struct CudaDslArtifact {
    pub cu: &'static str,
    pub entry: &'static str,
}

macro_rules! cuda_dsl_artifacts {
    ($($name:literal => $entry:literal),* $(,)?) => {
        /// Every committed CUDA DSL artifact name, for diagnostics and drift tests.
        pub const CUDA_DSL_ARTIFACTS: &[&str] = &[$($name),*];
        /// Resolve a committed CUDA DSL artifact by name.
        pub fn kernel_cu(name: &str) -> Result<CudaDslArtifact> {
            match name {
                $($name => Ok(CudaDslArtifact {
                    cu: include_str!(concat!("../cuda/dsl/", $name, ".cu")),
                    entry: $entry,
                }),)*
                _ => Err(Error::Msg(format!(
                    "no committed CUDA DSL artifact for '{name}'"
                ))),
            }
        }
    };
}

cuda_dsl_artifacts! {
    "matvec_pq2_0_gguf_nsg2_nr2_f32" => "matvec_pq2_0_gguf_f_f32",
    "matvec_pq2_0_gguf_nsg2_nr4_f32" => "matvec_pq2_0_gguf_f_f32",
    "matvec_pq2_0_gguf_nsg2_nr8_f32" => "matvec_pq2_0_gguf_f_f32",
    "matvec_pq2_0_gguf_nsg2_nr2_bf16" => "matvec_pq2_0_gguf_f_",
    "matvec_pq2_0_gguf_nsg2_nr4_bf16" => "matvec_pq2_0_gguf_f_",
    "matvec_pq2_0_gguf_nsg2_nr8_bf16" => "matvec_pq2_0_gguf_f_",
    "matvec_pq2_0_gguf_nsg4_nr2_f32" => "matvec_pq2_0_gguf_f_f32",
    "matvec_pq2_0_gguf_nsg4_nr4_f32" => "matvec_pq2_0_gguf_f_f32",
    "matvec_pq2_0_gguf_nsg4_nr8_f32" => "matvec_pq2_0_gguf_f_f32",
    "matvec_pq2_0_gguf_nsg4_nr2_bf16" => "matvec_pq2_0_gguf_f_",
    "matvec_pq2_0_gguf_nsg4_nr4_bf16" => "matvec_pq2_0_gguf_f_",
    "matvec_pq2_0_gguf_nsg4_nr8_bf16" => "matvec_pq2_0_gguf_f_",
    "matmul_pq2_0_gguf_bm32_f32" => "matmul_pq2_0_gguf_f_f32",
    "matmul_pq2_0_gguf_bm32_bf16" => "matmul_pq2_0_gguf_f_",
    "matmul_pq2_0_gguf_bm64_f32" => "matmul_pq2_0_gguf_f_f32",
    "matmul_pq2_0_gguf_bm64_bf16" => "matmul_pq2_0_gguf_f_",
    "matvec_ptq1_0_gguf_nsg2_nr2_f32" => "matvec_ptq1_0_gguf_f_f32",
    "matvec_ptq1_0_gguf_nsg2_nr4_f32" => "matvec_ptq1_0_gguf_f_f32",
    "matvec_ptq1_0_gguf_nsg2_nr8_f32" => "matvec_ptq1_0_gguf_f_f32",
    "matvec_ptq1_0_gguf_nsg2_nr2_bf16" => "matvec_ptq1_0_gguf_f_",
    "matvec_ptq1_0_gguf_nsg2_nr4_bf16" => "matvec_ptq1_0_gguf_f_",
    "matvec_ptq1_0_gguf_nsg2_nr8_bf16" => "matvec_ptq1_0_gguf_f_",
    "matvec_ptq1_0_gguf_nsg4_nr2_f32" => "matvec_ptq1_0_gguf_f_f32",
    "matvec_ptq1_0_gguf_nsg4_nr4_f32" => "matvec_ptq1_0_gguf_f_f32",
    "matvec_ptq1_0_gguf_nsg4_nr8_f32" => "matvec_ptq1_0_gguf_f_f32",
    "matvec_ptq1_0_gguf_nsg4_nr2_bf16" => "matvec_ptq1_0_gguf_f_",
    "matvec_ptq1_0_gguf_nsg4_nr4_bf16" => "matvec_ptq1_0_gguf_f_",
    "matvec_ptq1_0_gguf_nsg4_nr8_bf16" => "matvec_ptq1_0_gguf_f_",
    "matmul_ptq1_0_gguf_bm32_f32" => "matmul_ptq1_0_gguf_f_f32",
    "matmul_ptq1_0_gguf_bm32_bf16" => "matmul_ptq1_0_gguf_f_",
    "matmul_ptq1_0_gguf_bm64_f32" => "matmul_ptq1_0_gguf_f_f32",
    "matmul_ptq1_0_gguf_bm64_bf16" => "matmul_ptq1_0_gguf_f_",
    // Production GDN (F32 / F16 / BF16 I/O, internal F32 math & state)
    "gdn_conv1d_step_k4_f32" => "gdn_conv1d_step_f_f32",
    "gdn_conv1d_step_k4_f16" => "gdn_conv1d_step_f_",
    "gdn_conv1d_step_k4_bf16" => "gdn_conv1d_step_f_",
    "gdn_conv1d_stateful_k4_f32" => "gdn_conv1d_stateful_f_f32",
    "gdn_conv1d_stateful_k4_f16" => "gdn_conv1d_stateful_f_",
    "gdn_conv1d_stateful_k4_bf16" => "gdn_conv1d_stateful_f_",
    "gdn_gating_io_f32" => "gdn_gating_io_f_f32",
    "gdn_gating_io_f16" => "gdn_gating_io_f_",
    "gdn_gating_io_bf16" => "gdn_gating_io_f_",
    "gdn_scan_lanes_io_k128_v128_l4_f32" => "gdn_scan_lanes_io_f_f32",
    "gdn_scan_lanes_io_k128_v128_l4_f16" => "gdn_scan_lanes_io_f_",
    "gdn_scan_lanes_io_k128_v128_l4_bf16" => "gdn_scan_lanes_io_f_",
    "gdn_scan_lanes_io_k64_v64_l4_f32" => "gdn_scan_lanes_io_f_f32",
    "gdn_scan_lanes_io_k64_v64_l4_f16" => "gdn_scan_lanes_io_f_",
    "gdn_scan_lanes_io_k64_v64_l4_bf16" => "gdn_scan_lanes_io_f_",
    // Production Paged & Contiguous FlashAttention (d=64/128/256, F32/F16/BF16)
    "flash_attn_paged_d64_f32" => "flash_attn_paged_f_f32",
    "flash_attn_paged_d64_f16" => "flash_attn_paged_f_",
    "flash_attn_paged_d64_bf16" => "flash_attn_paged_f_",
    "flash_attn_paged_d128_f32" => "flash_attn_paged_f_f32",
    "flash_attn_paged_d128_f16" => "flash_attn_paged_f_",
    "flash_attn_paged_d128_bf16" => "flash_attn_paged_f_",
    "flash_attn_paged_d256_f32" => "flash_attn_paged_f_f32",
    "flash_attn_paged_d256_f16" => "flash_attn_paged_f_",
    "flash_attn_paged_d256_bf16" => "flash_attn_paged_f_",
    // Production Fast Walsh-Hadamard Transform (B=64..2048, F32/F16/BF16, fused signs)
    "fwht_hadamard_b64_f32" => "fwht_hadamard_f_f32",
    "fwht_hadamard_b64_f16" => "fwht_hadamard_f_",
    "fwht_hadamard_b64_bf16" => "fwht_hadamard_f_",
    "fwht_hadamard_b128_f32" => "fwht_hadamard_f_f32",
    "fwht_hadamard_b128_f16" => "fwht_hadamard_f_",
    "fwht_hadamard_b128_bf16" => "fwht_hadamard_f_",
    "fwht_hadamard_b256_f32" => "fwht_hadamard_f_f32",
    "fwht_hadamard_b256_f16" => "fwht_hadamard_f_",
    "fwht_hadamard_b256_bf16" => "fwht_hadamard_f_",
    "fwht_hadamard_b512_f32" => "fwht_hadamard_f_f32",
    "fwht_hadamard_b512_f16" => "fwht_hadamard_f_",
    "fwht_hadamard_b512_bf16" => "fwht_hadamard_f_",
    "fwht_hadamard_b1024_f32" => "fwht_hadamard_f_f32",
    "fwht_hadamard_b1024_f16" => "fwht_hadamard_f_",
    "fwht_hadamard_b1024_bf16" => "fwht_hadamard_f_",
    "fwht_hadamard_b2048_f32" => "fwht_hadamard_f_f32",
    "fwht_hadamard_b2048_f16" => "fwht_hadamard_f_",
    "fwht_hadamard_b2048_bf16" => "fwht_hadamard_f_",
}

/// Trailing `const __grid_constant__ info_st info` parameter emitted by cubecl-cpp CUDA kernels
/// (`static_meta[2 * num_buffers]`).
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaDslInfoN<const N: usize>([u32; N]);

// SAFETY: plain `[u32; N]` POD struct matching `struct info_st { uint32 static_meta[N]; };`.
unsafe impl<const N: usize> DeviceRepr for CudaDslInfoN<N> {}

/// Compute exact dynamic shared memory bytes declared by cubecl's `// Shared array size: N, B bytes`
/// + `&dynamic_shared_mem[OFF]` prologue comments.
fn parse_dynamic_shared_mem_bytes(cu: &str) -> u32 {
    let mut max_end = 0u32;
    let mut pending_bytes = 0u32;
    for line in cu.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("// Shared array size:") {
            if let Some(after_comma) = rest.split(',').nth(1) {
                if let Some(num) = after_comma.trim().split_whitespace().next() {
                    pending_bytes = num.parse().unwrap_or(0);
                }
            }
        } else if let Some(idx) = t.find("dynamic_shared_mem[") {
            let after = &t[idx + "dynamic_shared_mem[".len()..];
            if let Some(end) = after.find(']') {
                if let Ok(off) = after[..end].parse::<u32>() {
                    max_end = max_end.max(off + pending_bytes);
                    pending_bytes = 0;
                }
            }
        }
    }
    max_end
}

fn cuda_include_paths() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in [
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDA_ROOT",
        "CUDA_HOME",
        "CUDA_PATH",
    ] {
        if let Ok(v) = std::env::var(var) {
            roots.push(PathBuf::from(v));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/usr/local/cuda-13.0"));
    roots.push(PathBuf::from("/usr/local/cuda-12.8"));
    let mut out = Vec::new();
    for root in roots {
        let inc = root.join("include");
        if inc.join("cuda_fp16.h").exists() {
            let cccl = inc.join("cccl");
            out.push(inc);
            if cccl.exists() {
                out.push(cccl);
            }
            break;
        }
    }
    if out.is_empty() && PathBuf::from("/usr/include/cuda_fp16.h").exists() {
        out.push(PathBuf::from("/usr/include"));
    }
    out
}

fn compile_cu_to_ptx(src: &str, cc: i32) -> Result<String> {
    let major = cc / 100;
    let minor = (cc % 100) / 10;
    let arch = if major >= 9 {
        format!("--gpu-architecture=sm_{major}{minor}a")
    } else {
        format!("--gpu-architecture=sm_{major}{minor}")
    };
    let inc_opts: Vec<String> = cuda_include_paths()
        .into_iter()
        .map(|p| format!("--include-path={}", p.display()))
        .collect();
    let mut opts: Vec<&str> = Vec::with_capacity(2 + inc_opts.len());
    opts.push(&arch);
    for inc in &inc_opts {
        opts.push(inc.as_str());
    }
    opts.push("-lineinfo");

    let csrc = CString::new(src).map_err(|e| Error::Msg(format!("NVRTC CString: {e}")))?;
    // SAFETY: NVRTC FFI lifecycle; program is destroyed before returning.
    unsafe {
        let prog = cudarc::nvrtc::result::create_program(csrc.as_c_str(), None)
            .map_err(|e| Error::Msg(format!("NVRTC create_program: {e:?}")))?;
        let compile_res = cudarc::nvrtc::result::compile_program(prog, &opts);
        if let Err(err) = compile_res {
            let log = cudarc::nvrtc::result::get_program_log(prog)
                .ok()
                .and_then(|raw| {
                    CStr::from_ptr(raw.as_ptr())
                        .to_str()
                        .ok()
                        .map(str::to_owned)
                })
                .unwrap_or_default();
            let _ = cudarc::nvrtc::result::destroy_program(prog);
            return Err(Error::Msg(format!(
                "NVRTC compile failed ({err:?}, arch={arch}):\n{log}"
            )));
        }
        let ptx_raw = cudarc::nvrtc::result::get_ptx(prog);
        let _ = cudarc::nvrtc::result::destroy_program(prog);
        let ptx_bytes = ptx_raw.map_err(|e| Error::Msg(format!("NVRTC get_ptx: {e:?}")))?;
        let ptx = CStr::from_ptr(ptx_bytes.as_ptr())
            .to_str()
            .map_err(|e| Error::Msg(format!("NVRTC PTX utf8: {e}")))?
            .to_owned();
        Ok(ptx)
    }
}

type ShmemCache = HashMap<(DeviceId, String), u32>;
type MetaCache = HashMap<(DeviceId, [u32; 3]), Arc<CudaSlice<u32>>>;
type U32VecCache = HashMap<(DeviceId, Vec<u32>), Arc<CudaSlice<u32>>>;
type F32VecCache = HashMap<(DeviceId, Vec<u32>), Arc<CudaSlice<f32>>>;

static DSL_SHMEM_CACHE: OnceLock<Mutex<ShmemCache>> = OnceLock::new();
static DSL_META_CACHE: OnceLock<Mutex<MetaCache>> = OnceLock::new();
static DSL_U32_VEC_CACHE: OnceLock<Mutex<U32VecCache>> = OnceLock::new();
static DSL_F32_VEC_CACHE: OnceLock<Mutex<F32VecCache>> = OnceLock::new();

impl CudaDevice {
    /// Compile (once per device + artifact) and return `(CudaFunc, shared_mem_bytes)` for a
    /// registered CUDA DSL artifact.
    pub fn dsl_func(&self, name: &str) -> Result<(CudaFunc, u32)> {
        let art = kernel_cu(name)?;
        let mod_key = format!("dsl::{name}");
        let shmem_map = DSL_SHMEM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(&shmem) = shmem_map.lock().unwrap().get(&(self.id(), mod_key.clone())) {
            // Module is already loaded in `custom_modules`; empty ptx slice is unused on hit.
            let func = self.get_or_load_custom_func(art.entry, &mod_key, "")?;
            return Ok((func, shmem));
        }
        let cc = crate::quantized::fast_mmq::compute_cap(self);
        let ptx = compile_cu_to_ptx(art.cu, cc)?;
        let shmem = parse_dynamic_shared_mem_bytes(art.cu);
        let func = self.get_or_load_custom_func(art.entry, &mod_key, &ptx)?;
        shmem_map
            .lock()
            .unwrap()
            .insert((self.id(), mod_key), shmem);
        crate::quantized::dsl::note_compile(name);
        Ok((func, shmem))
    }

    /// Cached 3-word `[u32; 3]` device metadata buffer for DSL dispatches (zero alloc per op).
    pub fn dsl_const_u32(&self, words: &[u32; 3]) -> Result<Arc<CudaSlice<u32>>> {
        let cache = DSL_META_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (self.id(), *words);
        if let Some(buf) = cache.lock().unwrap().get(&key) {
            return Ok(buf.clone());
        }
        let slice = Arc::new(self.clone_htod(words.as_slice())?);
        cache.lock().unwrap().insert(key, slice.clone());
        Ok(slice)
    }

    /// Cached arbitrary-length `u32` metadata buffer for DSL dispatches (bounded to small keys).
    pub fn dsl_const_u32_slice(&self, words: &[u32]) -> Result<Arc<CudaSlice<u32>>> {
        let cache = DSL_U32_VEC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (self.id(), words.to_vec());
        if let Some(buf) = cache.lock().unwrap().get(&key) {
            return Ok(buf.clone());
        }
        let slice = Arc::new(self.clone_htod(words)?);
        let mut guard = cache.lock().unwrap();
        if guard.len() > 2048 {
            guard.clear();
        }
        guard.insert(key, slice.clone());
        Ok(slice)
    }

    /// Cached small `f32` metadata buffer (e.g. `[scale, softcap]`) for DSL dispatches.
    pub fn dsl_const_f32_slice(&self, vals: &[f32]) -> Result<Arc<CudaSlice<f32>>> {
        let cache = DSL_F32_VEC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let bits: Vec<u32> = vals.iter().map(|v| v.to_bits()).collect();
        let key = (self.id(), bits);
        if let Some(buf) = cache.lock().unwrap().get(&key) {
            return Ok(buf.clone());
        }
        let slice = Arc::new(self.clone_htod(vals)?);
        let mut guard = cache.lock().unwrap();
        if guard.len() > 512 {
            guard.clear();
        }
        guard.insert(key, slice.clone());
        Ok(slice)
    }

    /// Dispatch a registered CUDA DSL kernel with `ptrs.len()` buffer pointers followed by
    /// `info_st { uint32 static_meta[NINFO]; }` (`NINFO = 2 * ptrs.len()`).
    ///
    /// # Safety
    /// Caller must ensure every pointer in `ptrs` is a valid device pointer for the stream's
    /// lifetime and hold any `SyncOnDrop` guards until launch completes.
    pub unsafe fn dispatch_dsl_raw<const NINFO: usize>(
        &self,
        name: &str,
        ptrs: &[u64],
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> Result<()> {
        let (func, shared_mem_bytes) = self.dsl_func(name)?;
        let cfg = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes,
        };
        let info = CudaDslInfoN([0u32; NINFO]);
        let mut builder = func.builder();
        for p in ptrs {
            builder.arg(p);
        }
        builder.arg(&info);
        builder.launch(cfg).w()?;
        Ok(())
    }

    /// Dispatch a registered 4-buffer CUDA DSL kernel `(wq, x, out, meta, info_st)`.
    ///
    /// # Safety
    /// Caller must ensure `ptrs[0..4]` are valid device pointers for the stream's lifetime and
    /// respect any active `SyncOnDrop` guards until launch completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_dsl_4(
        &self,
        name: &str,
        ptrs: [u64; 4],
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
    ) -> Result<()> {
        self.dispatch_dsl_raw::<8>(name, &ptrs, grid_dim, block_dim)
    }

    fn alloc_dsl_float(&self, elems: usize, dtype: DType) -> Result<CudaStorage> {
        let slice = match dtype {
            DType::F32 => CudaStorageSlice::F32(unsafe { self.alloc::<f32>(elems)? }),
            DType::F16 => CudaStorageSlice::F16(unsafe { self.alloc::<half::f16>(elems)? }),
            DType::BF16 => CudaStorageSlice::BF16(unsafe { self.alloc::<half::bf16>(elems)? }),
            other => crate::bail!("cuda dsl alloc: unsupported dtype {other:?}"),
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    /// Production DSL single-token GDN conv1d step (F32/F16/BF16, k=4, updates `state` in place + SiLU).
    pub fn gdn_conv1d_step_dsl(
        &self,
        dtype: DType,
        x: (&CudaStorage, usize),
        weight: (&CudaStorage, usize),
        state: (&mut CudaStorage, usize),
        batch: usize,
        conv_dim: usize,
    ) -> Result<CudaStorage> {
        let artifact = match dtype {
            DType::F32 => "gdn_conv1d_step_k4_f32",
            DType::F16 => "gdn_conv1d_step_k4_f16",
            DType::BF16 => "gdn_conv1d_step_k4_bf16",
            other => crate::bail!("cuda gdn_conv1d_step_dsl: unsupported dtype {other:?}"),
        };
        let total = batch * conv_dim;
        if total == 0 {
            crate::bail!("cuda gdn_conv1d_step_dsl: empty shape ({batch}, {conv_dim})");
        }
        let mut out = self.alloc_dsl_float(total, dtype)?;
        let dims = self.dsl_const_u32_slice(&[batch as u32, conv_dim as u32])?;
        let stream = self.cuda_stream();
        {
            let (xp, _gx) = cuda_storage_ptr(x.0, x.1, dtype, &stream)?;
            let (wp, _gw) = cuda_storage_ptr(weight.0, weight.1, dtype, &stream)?;
            let (sp, _gs) = cuda_storage_ptr_mut(state.0, state.1, dtype, &stream)?;
            let (op, _go) = cuda_storage_ptr_mut(&mut out, 0, dtype, &stream)?;
            let (dp, _gd) = dims.device_ptr(&stream);
            unsafe {
                self.dispatch_dsl_raw::<10>(
                    artifact,
                    &[xp, wp, sp, op, dp],
                    ((total as u32).div_ceil(64), 1, 1),
                    (64, 1, 1),
                )?;
            }
        }
        crate::quantized::dsl::note_dispatch(
            crate::quantized::dsl::QkImpl::Dsl,
            artifact,
            conv_dim,
            batch,
        );
        Ok(out)
    }

    /// Production DSL stateful GDN conv1d prefill/chunked step (F32/F16/BF16, k=4).
    /// Returns `(out, state_out)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv1d_stateful_dsl(
        &self,
        dtype: DType,
        x: (&CudaStorage, usize),
        weight: (&CudaStorage, usize),
        state_in: (&CudaStorage, usize),
        batch: usize,
        conv_dim: usize,
        seq_len: usize,
        has_state: bool,
    ) -> Result<(CudaStorage, CudaStorage)> {
        let artifact = match dtype {
            DType::F32 => "gdn_conv1d_stateful_k4_f32",
            DType::F16 => "gdn_conv1d_stateful_k4_f16",
            DType::BF16 => "gdn_conv1d_stateful_k4_bf16",
            other => crate::bail!("cuda gdn_conv1d_stateful_dsl: unsupported dtype {other:?}"),
        };
        let total = batch * conv_dim * seq_len;
        if total == 0 {
            crate::bail!(
                "cuda gdn_conv1d_stateful_dsl: empty shape ({batch}, {conv_dim}, {seq_len})"
            );
        }
        let mut out = self.alloc_dsl_float(total, dtype)?;
        let mut state_out = self.alloc_dsl_float(batch * conv_dim * 4, dtype)?;
        let dims = self.dsl_const_u32_slice(&[
            batch as u32,
            conv_dim as u32,
            seq_len as u32,
            u32::from(has_state),
        ])?;
        let stream = self.cuda_stream();
        {
            let (xp, _gx) = cuda_storage_ptr(x.0, x.1, dtype, &stream)?;
            let (wp, _gw) = cuda_storage_ptr(weight.0, weight.1, dtype, &stream)?;
            let (sip, _gsi) = cuda_storage_ptr(state_in.0, state_in.1, dtype, &stream)?;
            let (sop, _gso) = cuda_storage_ptr_mut(&mut state_out, 0, dtype, &stream)?;
            let (op, _go) = cuda_storage_ptr_mut(&mut out, 0, dtype, &stream)?;
            let (dp, _gd) = dims.device_ptr(&stream);
            unsafe {
                self.dispatch_dsl_raw::<12>(
                    artifact,
                    &[xp, wp, sip, sop, op, dp],
                    ((total as u32).div_ceil(64), 1, 1),
                    (64, 1, 1),
                )?;
            }
        }
        crate::quantized::dsl::note_dispatch(
            crate::quantized::dsl::QkImpl::Dsl,
            artifact,
            conv_dim,
            seq_len,
        );
        Ok((out, state_out))
    }

    /// Production DSL typed GDN gating (F32/F16/BF16 I/O, F32 `a_log`/`dt_bias`).
    /// Returns `(beta_out, g_out)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gating_io_dsl(
        &self,
        dtype: DType,
        b_in: (&CudaStorage, usize),
        a_in: (&CudaStorage, usize),
        a_log: (&CudaStorage, usize),
        dt_bias: (&CudaStorage, usize),
        batch_tokens: usize,
        num_heads: usize,
    ) -> Result<(CudaStorage, CudaStorage)> {
        let artifact = match dtype {
            DType::F32 => "gdn_gating_io_f32",
            DType::F16 => "gdn_gating_io_f16",
            DType::BF16 => "gdn_gating_io_bf16",
            other => crate::bail!("cuda gdn_gating_io_dsl: unsupported dtype {other:?}"),
        };
        let total = batch_tokens * num_heads;
        if total == 0 {
            crate::bail!("cuda gdn_gating_io_dsl: empty shape ({batch_tokens}, {num_heads})");
        }
        let mut beta_out = self.alloc_dsl_float(total, dtype)?;
        let mut g_out = self.alloc_dsl_float(total, dtype)?;
        let dims = self.dsl_const_u32_slice(&[total as u32, num_heads as u32])?;
        let stream = self.cuda_stream();
        {
            let (bp, _gb) = cuda_storage_ptr(b_in.0, b_in.1, dtype, &stream)?;
            let (ap, _ga) = cuda_storage_ptr(a_in.0, a_in.1, dtype, &stream)?;
            let (alp, _gal) = cuda_storage_ptr(a_log.0, a_log.1, DType::F32, &stream)?;
            let (dtp, _gdt) = cuda_storage_ptr(dt_bias.0, dt_bias.1, DType::F32, &stream)?;
            let (bop, _gbo) = cuda_storage_ptr_mut(&mut beta_out, 0, dtype, &stream)?;
            let (gop, _ggo) = cuda_storage_ptr_mut(&mut g_out, 0, dtype, &stream)?;
            let (dp, _gd) = dims.device_ptr(&stream);
            unsafe {
                self.dispatch_dsl_raw::<14>(
                    artifact,
                    &[bp, ap, alp, dtp, bop, gop, dp],
                    ((total as u32).div_ceil(64), 1, 1),
                    (64, 1, 1),
                )?;
            }
        }
        crate::quantized::dsl::note_dispatch(
            crate::quantized::dsl::QkImpl::Dsl,
            artifact,
            num_heads,
            batch_tokens,
        );
        Ok((beta_out, g_out))
    }

    /// Production DSL typed GDN recurrence scan (F32/F16/BF16 I/O, F32 in-place `state`).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_lanes_io_dsl(
        &self,
        dtype: DType,
        q: (&CudaStorage, usize),
        k: (&CudaStorage, usize),
        v: (&CudaStorage, usize),
        g: (&CudaStorage, usize),
        beta: (&CudaStorage, usize),
        state: (&mut CudaStorage, usize),
        bh: usize,
        seq: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<CudaStorage> {
        let artifact = match (k_dim, v_dim, dtype) {
            (128, 128, DType::F32) => "gdn_scan_lanes_io_k128_v128_l4_f32",
            (128, 128, DType::F16) => "gdn_scan_lanes_io_k128_v128_l4_f16",
            (128, 128, DType::BF16) => "gdn_scan_lanes_io_k128_v128_l4_bf16",
            (64, 64, DType::F32) => "gdn_scan_lanes_io_k64_v64_l4_f32",
            (64, 64, DType::F16) => "gdn_scan_lanes_io_k64_v64_l4_f16",
            (64, 64, DType::BF16) => "gdn_scan_lanes_io_k64_v64_l4_bf16",
            _ => crate::bail!(
                "cuda gdn_scan_lanes_io_dsl: unsupported k={k_dim} v={v_dim} dtype={dtype:?}"
            ),
        };
        if bh == 0 || seq == 0 {
            crate::bail!("cuda gdn_scan_lanes_io_dsl: empty shape ({bh}, {seq})");
        }
        let mut out = self.alloc_dsl_float(bh * seq * v_dim, dtype)?;
        let dims = self.dsl_const_u32_slice(&[bh as u32, seq as u32])?;
        let total_threads = bh * v_dim * 4;
        let stream = self.cuda_stream();
        {
            let (qp, _gq) = cuda_storage_ptr(q.0, q.1, dtype, &stream)?;
            let (kp, _gk) = cuda_storage_ptr(k.0, k.1, dtype, &stream)?;
            let (vp, _gv) = cuda_storage_ptr(v.0, v.1, dtype, &stream)?;
            let (gp, _gg) = cuda_storage_ptr(g.0, g.1, dtype, &stream)?;
            let (bp, _gb) = cuda_storage_ptr(beta.0, beta.1, dtype, &stream)?;
            let (sp, _gs) = cuda_storage_ptr_mut(state.0, state.1, DType::F32, &stream)?;
            let (op, _go) = cuda_storage_ptr_mut(&mut out, 0, dtype, &stream)?;
            let (dp, _gd) = dims.device_ptr(&stream);
            unsafe {
                self.dispatch_dsl_raw::<16>(
                    artifact,
                    &[qp, kp, vp, gp, bp, sp, op, dp],
                    ((total_threads as u32).div_ceil(64), 1, 1),
                    (64, 1, 1),
                )?;
            }
        }
        crate::quantized::dsl::note_dispatch(crate::quantized::dsl::QkImpl::Dsl, artifact, bh, seq);
        Ok(out)
    }

    /// Production DSL Paged & Contiguous FlashAttention (d=64/128/256, F32/F16/BF16).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_paged_dsl(
        &self,
        dtype: DType,
        q: (&CudaStorage, usize),
        k_cache: (&CudaStorage, usize),
        v_cache: (&CudaStorage, usize),
        block_tables: (&CudaStorage, usize),
        context_lens: (&CudaStorage, usize),
        batch: usize,
        n_heads: usize,
        n_kv: usize,
        seq_q: usize,
        d: usize,
        scale: f32,
        softcap: f32,
        causal_mode: u32,
        block_size: usize,
        max_blocks: usize,
        kv_layout: usize,
        x_vec: usize,
        sliding_window: usize,
        q_offset_override: usize,
    ) -> Result<CudaStorage> {
        let artifact = match (d, dtype) {
            (64, DType::F32) => "flash_attn_paged_d64_f32",
            (64, DType::F16) => "flash_attn_paged_d64_f16",
            (64, DType::BF16) => "flash_attn_paged_d64_bf16",
            (128, DType::F32) => "flash_attn_paged_d128_f32",
            (128, DType::F16) => "flash_attn_paged_d128_f16",
            (128, DType::BF16) => "flash_attn_paged_d128_bf16",
            (256, DType::F32) => "flash_attn_paged_d256_f32",
            (256, DType::F16) => "flash_attn_paged_d256_f16",
            (256, DType::BF16) => "flash_attn_paged_d256_bf16",
            _ => {
                crate::bail!("cuda flash_attn_paged_dsl: unsupported head_dim={d} dtype={dtype:?}")
            }
        };
        if batch == 0 || n_heads == 0 || n_kv == 0 || seq_q == 0 {
            crate::bail!("cuda flash_attn_paged_dsl: empty shape");
        }
        let mut out = self.alloc_dsl_float(batch * n_heads * seq_q * d, dtype)?;
        let scale_cfg = self.dsl_const_f32_slice(&[scale, softcap])?;
        let meta = self.dsl_const_u32_slice(&[
            seq_q as u32,
            n_heads as u32,
            n_kv as u32,
            causal_mode,
            block_size as u32,
            max_blocks as u32,
            kv_layout as u32,
            x_vec as u32,
            sliding_window as u32,
            q_offset_override as u32,
            0u32,
            0u32,
        ])?;
        let blocks = (batch * n_heads * seq_q.div_ceil(16)) as u32;
        let stream = self.cuda_stream();
        {
            let (qp, _gq) = cuda_storage_ptr(q.0, q.1, dtype, &stream)?;
            let (kp, _gk) = cuda_storage_ptr(k_cache.0, k_cache.1, dtype, &stream)?;
            let (vp, _gv) = cuda_storage_ptr(v_cache.0, v_cache.1, dtype, &stream)?;
            let (btp, _gbt) =
                cuda_storage_ptr(block_tables.0, block_tables.1, DType::U32, &stream)?;
            let (clp, _gcl) =
                cuda_storage_ptr(context_lens.0, context_lens.1, DType::U32, &stream)?;
            let (op, _go) = cuda_storage_ptr_mut(&mut out, 0, dtype, &stream)?;
            let (scp, _gsc) = scale_cfg.device_ptr(&stream);
            let (mp, _gm) = meta.device_ptr(&stream);
            unsafe {
                self.dispatch_dsl_raw::<16>(
                    artifact,
                    &[qp, kp, vp, btp, clp, op, scp, mp],
                    (blocks, 1, 1),
                    (32, 1, 1),
                )?;
            }
        }
        crate::quantized::dsl::note_dispatch(
            crate::quantized::dsl::QkImpl::Dsl,
            artifact,
            n_heads,
            d,
        );
        Ok(out)
    }

    /// Production DSL Fast Walsh-Hadamard Transform (`B=64..2048`, F32/F16/BF16, fused optional signs).
    #[allow(clippy::too_many_arguments)]
    pub fn fwht_hadamard_dsl(
        &self,
        dtype: DType,
        x: (&CudaStorage, usize),
        signs: Option<(&CudaStorage, usize, usize)>,
        n_blocks: usize,
        block_size: usize,
        inverse: bool,
    ) -> Result<CudaStorage> {
        let artifact = match (block_size, dtype) {
            (64, DType::F32) => "fwht_hadamard_b64_f32",
            (64, DType::F16) => "fwht_hadamard_b64_f16",
            (64, DType::BF16) => "fwht_hadamard_b64_bf16",
            (128, DType::F32) => "fwht_hadamard_b128_f32",
            (128, DType::F16) => "fwht_hadamard_b128_f16",
            (128, DType::BF16) => "fwht_hadamard_b128_bf16",
            (256, DType::F32) => "fwht_hadamard_b256_f32",
            (256, DType::F16) => "fwht_hadamard_b256_f16",
            (256, DType::BF16) => "fwht_hadamard_b256_bf16",
            (512, DType::F32) => "fwht_hadamard_b512_f32",
            (512, DType::F16) => "fwht_hadamard_b512_f16",
            (512, DType::BF16) => "fwht_hadamard_b512_bf16",
            (1024, DType::F32) => "fwht_hadamard_b1024_f32",
            (1024, DType::F16) => "fwht_hadamard_b1024_f16",
            (1024, DType::BF16) => "fwht_hadamard_b1024_bf16",
            (2048, DType::F32) => "fwht_hadamard_b2048_f32",
            (2048, DType::F16) => "fwht_hadamard_b2048_f16",
            (2048, DType::BF16) => "fwht_hadamard_b2048_bf16",
            _ => crate::bail!(
                "cuda fwht_hadamard_dsl: unsupported block_size={block_size} dtype={dtype:?}"
            ),
        };
        if n_blocks == 0 {
            crate::bail!("cuda fwht_hadamard_dsl: empty input");
        }
        let mut out = self.alloc_dsl_float(n_blocks * block_size, dtype)?;
        let (sign_w, has_signs) = match signs {
            Some((_, _, w)) => (w, 1u32),
            None => (block_size, 0u32),
        };
        let dims = self.dsl_const_u32_slice(&[
            n_blocks as u32,
            sign_w as u32,
            has_signs,
            u32::from(inverse),
        ])?;
        let dummy_signs = if signs.is_none() {
            Some(self.dsl_const_f32_slice(&[1.0f32])?)
        } else {
            None
        };
        let nt = (block_size / 4).clamp(16, 256) as u32;
        let stream = self.cuda_stream();
        {
            let (xp, _gx) = cuda_storage_ptr(x.0, x.1, dtype, &stream)?;
            let (sp, _gs_storage, _gs_dummy) = match signs {
                Some((ss, so, _)) => {
                    let (p, g) = cuda_storage_ptr(ss, so, DType::F32, &stream)?;
                    (p, Some(g), None)
                }
                None => {
                    let (p, g) = dummy_signs.as_ref().unwrap().device_ptr(&stream);
                    (p, None, Some(g))
                }
            };
            let (op, _go) = cuda_storage_ptr_mut(&mut out, 0, dtype, &stream)?;
            let (dp, _gd) = dims.device_ptr(&stream);
            unsafe {
                self.dispatch_dsl_raw::<8>(
                    artifact,
                    &[xp, sp, op, dp],
                    (n_blocks as u32, 1, 1),
                    (nt, 1, 1),
                )?;
            }
        }
        crate::quantized::dsl::note_dispatch(
            crate::quantized::dsl::QkImpl::Dsl,
            artifact,
            n_blocks,
            block_size,
        );
        Ok(out)
    }
}

fn cuda_storage_ptr<'a>(
    s: &'a CudaStorage,
    elem_offset: usize,
    dtype: DType,
    stream: &'a Arc<cudarc::driver::CudaStream>,
) -> Result<(u64, cudarc::driver::SyncOnDrop<'a>)> {
    let (base, guard) = match (&s.slice, dtype) {
        (CudaStorageSlice::F32(sl), DType::F32) => sl.device_ptr(stream),
        (CudaStorageSlice::F16(sl), DType::F16) => sl.device_ptr(stream),
        (CudaStorageSlice::BF16(sl), DType::BF16) => sl.device_ptr(stream),
        (CudaStorageSlice::U32(sl), DType::U32) => sl.device_ptr(stream),
        _ => crate::bail!("cuda_storage_ptr: expected dtype {dtype:?}"),
    };
    Ok((base + (elem_offset * dtype.size_in_bytes()) as u64, guard))
}

fn cuda_storage_ptr_mut<'a>(
    s: &'a mut CudaStorage,
    elem_offset: usize,
    dtype: DType,
    stream: &'a Arc<cudarc::driver::CudaStream>,
) -> Result<(u64, cudarc::driver::SyncOnDrop<'a>)> {
    let (base, guard) = match (&mut s.slice, dtype) {
        (CudaStorageSlice::F32(sl), DType::F32) => sl.device_ptr_mut(stream),
        (CudaStorageSlice::F16(sl), DType::F16) => sl.device_ptr_mut(stream),
        (CudaStorageSlice::BF16(sl), DType::BF16) => sl.device_ptr_mut(stream),
        (CudaStorageSlice::U32(sl), DType::U32) => sl.device_ptr_mut(stream),
        _ => crate::bail!("cuda_storage_ptr_mut: expected dtype {dtype:?}"),
    };
    Ok((base + (elem_offset * dtype.size_in_bytes()) as u64, guard))
}
