//! CUDA DSL kernel artifact registry, single-compile NVRTC loader, and dispatch seam.
//!
//! Mirrors `metal_backend::kernel_msl` + `dispatch_dsl`: every `.cu` file under `src/cuda/dsl/` is
//! generated from `hanzo-kernel` via `kernel-gen --target cuda --out hanzo-ml/src/cuda/dsl` and
//! drift-checked with `--check`. Compiled once per `(device, artifact)` via NVRTC and cached.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};

use super::device::{CudaDevice, CudaFunc, DeviceId};
use super::WrapErr;
use crate::{Error, Result};

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
}

/// Trailing `const __grid_constant__ info_st info` parameter emitted by cubecl-cpp CUDA kernels.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaDslInfo([u32; 8]);

// SAFETY: plain `[u32; 8]` POD struct matching `struct info_st { uint32 static_meta[8]; };`.
unsafe impl DeviceRepr for CudaDslInfo {}

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
    for var in ["CUDA_TOOLKIT_ROOT_DIR", "CUDA_ROOT", "CUDA_HOME", "CUDA_PATH"] {
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
                .and_then(|raw| CStr::from_ptr(raw.as_ptr()).to_str().ok().map(str::to_owned))
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

static DSL_SHMEM_CACHE: OnceLock<Mutex<ShmemCache>> = OnceLock::new();
static DSL_META_CACHE: OnceLock<Mutex<MetaCache>> = OnceLock::new();

impl CudaDevice {
    /// Compile (once per device + artifact) and return `(CudaFunc, shared_mem_bytes)` for a
    /// registered CUDA DSL artifact.
    pub fn dsl_func(&self, name: &str) -> Result<(CudaFunc, u32)> {
        let art = kernel_cu(name)?;
        let mod_key = format!("dsl::{name}");
        let shmem_map = DSL_SHMEM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(&shmem) = shmem_map
            .lock()
            .unwrap()
            .get(&(self.id(), mod_key.clone()))
        {
            // Module is already loaded in `custom_modules`; empty ptx slice is unused on hit.
            let func = self.get_or_load_custom_func(art.entry, &mod_key, "")?;
            return Ok((func, shmem));
        }
        let cc = crate::quantized::fast_mmq::compute_cap(self);
        let ptx = compile_cu_to_ptx(art.cu, cc)?;
        let shmem = parse_dynamic_shared_mem_bytes(art.cu);
        let func = self.get_or_load_custom_func(art.entry, &mod_key, &ptx)?;
        shmem_map.lock().unwrap().insert((self.id(), mod_key), shmem);
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
        let (func, shared_mem_bytes) = self.dsl_func(name)?;
        let cfg = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes,
        };
        let info = CudaDslInfo([0u32; 8]);
        let mut builder = func.builder();
        builder.arg(&ptrs[0]);
        builder.arg(&ptrs[1]);
        builder.arg(&ptrs[2]);
        builder.arg(&ptrs[3]);
        builder.arg(&info);
        builder.launch(cfg).w()?;
        Ok(())
    }
}
