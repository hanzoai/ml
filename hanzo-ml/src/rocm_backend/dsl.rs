//! ROCm/HIP DSL kernel artifact registry, single-compile disk-cached loader, and dispatch seam.
//!
//! Mirrors `cuda_backend::dsl`: every `.hip` file under `src/rocm/dsl/` is generated from
//! `hanzo-kernel` via `kernel-gen --target rocm --out hanzo-ml/src/rocm/dsl` and drift-checked
//! with `--check`. Compiled once per `(arch, rocm_version, artifact)` via `hanzo-rocm-kernels`'s
//! `KernelCache` (which caches `.cso` code objects on disk and resolved `hipFunction_t` handles in
//! memory).

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use rocm_rs::hip::Dim3;

use super::device::{DeviceId, RocmDevice};
use super::wrappers::SendSyncDeviceMemory;
use super::{launch_kernel_shmem, RocmStorage, RocmStorageSlice};
use crate::quantized::dsl::{self, QkImpl};
use crate::quantized::GgmlDType;
use crate::{DType, Error, Result};

/// A committed ROCm DSL artifact: generated `.hip` source, cache-safe module name, and entry name.
pub struct RocmDslArtifact {
    pub name: &'static str,
    pub hip: &'static str,
    pub entry: &'static str,
}

macro_rules! rocm_dsl_artifacts {
    ($($name:literal => $entry:literal),* $(,)?) => {
        /// Every committed ROCm DSL artifact name, for diagnostics and drift tests.
        pub const ROCM_DSL_ARTIFACTS: &[&str] = &[$($name),*];
        /// Resolve a committed ROCm DSL artifact by name.
        pub fn kernel_hip(name: &str) -> Result<RocmDslArtifact> {
            match name {
                $($name => Ok(RocmDslArtifact {
                    name: $name,
                    hip: include_str!(concat!("../rocm/dsl/", $name, ".hip")),
                    entry: $entry,
                }),)*
                _ => Err(Error::Msg(format!(
                    "no committed ROCm DSL artifact for '{name}'"
                ))),
            }
        }
    };
}

rocm_dsl_artifacts! {
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
    "flash_attn_paged_split_kv_d64_f32" => "flash_attn_paged_split_kv_f_f32",
    "flash_attn_paged_split_kv_d64_f16" => "flash_attn_paged_split_kv_f_",
    "flash_attn_paged_split_kv_d64_bf16" => "flash_attn_paged_split_kv_f_",
    "flash_attn_paged_split_kv_d128_f32" => "flash_attn_paged_split_kv_f_f32",
    "flash_attn_paged_split_kv_d128_f16" => "flash_attn_paged_split_kv_f_",
    "flash_attn_paged_split_kv_d128_bf16" => "flash_attn_paged_split_kv_f_",
    "flash_attn_paged_split_kv_d256_f32" => "flash_attn_paged_split_kv_f_f32",
    "flash_attn_paged_split_kv_d256_f16" => "flash_attn_paged_split_kv_f_",
    "flash_attn_paged_split_kv_d256_bf16" => "flash_attn_paged_split_kv_f_",
    "flash_attn_split_kv_reduce_d64_f32" => "flash_attn_split_kv_reduce_f_f32",
    "flash_attn_split_kv_reduce_d64_f16" => "flash_attn_split_kv_reduce_f_",
    "flash_attn_split_kv_reduce_d64_bf16" => "flash_attn_split_kv_reduce_f_",
    "flash_attn_split_kv_reduce_d128_f32" => "flash_attn_split_kv_reduce_f_f32",
    "flash_attn_split_kv_reduce_d128_f16" => "flash_attn_split_kv_reduce_f_",
    "flash_attn_split_kv_reduce_d128_bf16" => "flash_attn_split_kv_reduce_f_",
    "flash_attn_split_kv_reduce_d256_f32" => "flash_attn_split_kv_reduce_f_f32",
    "flash_attn_split_kv_reduce_d256_f16" => "flash_attn_split_kv_reduce_f_",
    "flash_attn_split_kv_reduce_d256_bf16" => "flash_attn_split_kv_reduce_f_",
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

/// Compute exact dynamic shared memory bytes declared by cubecl's `// Shared array size: N, B bytes`
/// + `&dynamic_shared_mem[OFF]` prologue comments.
fn parse_dynamic_shared_mem_bytes(hip: &str) -> u32 {
    let mut max_end = 0u32;
    let mut pending_bytes = 0u32;
    for line in hip.lines() {
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

type ShmemCache = HashMap<(DeviceId, &'static str), u32>;
type ZeroInfoCache = HashMap<DeviceId, Arc<SendSyncDeviceMemory<u32>>>;
type U32VecCache = HashMap<(DeviceId, Vec<u32>), Arc<SendSyncDeviceMemory<u32>>>;
type F32VecCache = HashMap<(DeviceId, Vec<u32>), Arc<SendSyncDeviceMemory<f32>>>;

static ROCM_DSL_SHMEM_CACHE: OnceLock<Mutex<ShmemCache>> = OnceLock::new();
static ROCM_DSL_ZERO_INFO: OnceLock<Mutex<ZeroInfoCache>> = OnceLock::new();
static ROCM_DSL_U32_VEC_CACHE: OnceLock<Mutex<U32VecCache>> = OnceLock::new();
static ROCM_DSL_F32_VEC_CACHE: OnceLock<Mutex<F32VecCache>> = OnceLock::new();

impl RocmDevice {
    /// Cached 16-word (64 B) zeroed device `info_st` buffer for HIP DSL launches
    /// (`const info_st *__restrict__ info_ptr`). Allocated once per device (hipGraph-capture safe).
    fn dsl_zero_info(&self) -> Result<Arc<SendSyncDeviceMemory<u32>>> {
        let cache = ROCM_DSL_ZERO_INFO.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(buf) = cache.lock().unwrap().get(&self.id()) {
            return Ok(buf.clone());
        }
        let zeros = [0u32; 32];
        let mem = Arc::new(self.clone_htod(&zeros)?);
        cache.lock().unwrap().insert(self.id(), mem.clone());
        Ok(mem)
    }

    /// Cached small `u32` metadata buffer for HIP DSL dispatches (zero alloc on cache hit).
    pub fn dsl_const_u32_slice(&self, words: &[u32]) -> Result<Arc<SendSyncDeviceMemory<u32>>> {
        let cache = ROCM_DSL_U32_VEC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (self.id(), words.to_vec());
        if let Some(buf) = cache.lock().unwrap().get(&key) {
            return Ok(buf.clone());
        }
        let mem = Arc::new(self.clone_htod(words)?);
        let mut guard = cache.lock().unwrap();
        if guard.len() > 2048 {
            guard.clear();
        }
        guard.insert(key, mem.clone());
        Ok(mem)
    }

    /// Cached small `f32` metadata buffer (e.g. `[scale, softcap]`) for HIP DSL dispatches.
    pub fn dsl_const_f32_slice(&self, vals: &[f32]) -> Result<Arc<SendSyncDeviceMemory<f32>>> {
        let cache = ROCM_DSL_F32_VEC_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let bits: Vec<u32> = vals.iter().map(|v| v.to_bits()).collect();
        let key = (self.id(), bits);
        if let Some(buf) = cache.lock().unwrap().get(&key) {
            return Ok(buf.clone());
        }
        let mem = Arc::new(self.clone_htod(vals)?);
        let mut guard = cache.lock().unwrap();
        if guard.len() > 512 {
            guard.clear();
        }
        guard.insert(key, mem.clone());
        Ok(mem)
    }

    fn alloc_dsl_float(&self, elems: usize, dtype: DType) -> Result<RocmStorage> {
        let slice = match dtype {
            DType::F32 => RocmStorageSlice::F32(self.alloc::<f32>(elems)?),
            DType::F16 => RocmStorageSlice::F16(self.alloc::<half::f16>(elems)?),
            DType::BF16 => RocmStorageSlice::BF16(self.alloc::<half::bf16>(elems)?),
            other => crate::bail!("rocm dsl alloc: unsupported dtype {other:?}"),
        };
        Ok(RocmStorage {
            slice,
            device: self.clone(),
        })
    }

    /// Dispatch a registered ROCm DSL `.hip` kernel with `buf_ptrs` followed by `info_ptr`.
    ///
    /// # Safety
    /// Caller must ensure every pointer in `buf_ptrs` is a valid device pointer for the stream's
    /// lifetime.
    pub unsafe fn dispatch_dsl_raw(
        &self,
        name: &str,
        buf_ptrs: &[*mut c_void],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
    ) -> Result<()> {
        let art = kernel_hip(name)?;
        let shmem_map = ROCM_DSL_SHMEM_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let (shmem, first_compile) = {
            let guard = shmem_map.lock().unwrap();
            if let Some(&s) = guard.get(&(self.id(), art.name)) {
                (s, false)
            } else {
                (parse_dynamic_shared_mem_bytes(art.hip), true)
            }
        };
        let info = self.dsl_zero_info()?;
        let mut ptrs: Vec<*mut c_void> = Vec::with_capacity(buf_ptrs.len() + 1);
        ptrs.extend_from_slice(buf_ptrs);
        ptrs.push(info.as_ptr());
        let mut arg_ptrs: Vec<*mut c_void> = ptrs
            .iter_mut()
            .map(|p| p as *mut *mut c_void as *mut c_void)
            .collect();
        launch_kernel_shmem(
            self,
            art.name,
            art.hip,
            art.entry,
            Dim3 {
                x: grid.0,
                y: grid.1,
                z: grid.2,
            },
            Dim3 {
                x: block.0,
                y: block.1,
                z: block.2,
            },
            shmem,
            &mut arg_ptrs,
        )?;
        if first_compile {
            shmem_map
                .lock()
                .unwrap()
                .insert((self.id(), art.name), shmem);
            dsl::note_compile(art.name);
        }
        Ok(())
    }

    /// Try canonical DSL `PQ2_0` / `PTQ1_0` decode matvec (`m == 1`) or prefill GEMM (`m > 1`) on
    /// ROCm (`F32` or `BF16` activations). Returns `Ok(None)` when disabled (`HANZO_*_IMPL=legacy`)
    /// or ineligible so caller falls back cleanly to legacy ROCm kernels.
    #[allow(clippy::too_many_arguments)]
    pub fn try_ternary_dsl_matmul(
        &self,
        dtype: GgmlDType,
        wq: &RocmStorage,
        wq_bytes: usize,
        x: &RocmStorage,
        x_offset: usize,
        x_dtype: DType,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Option<RocmStorage>> {
        let active = match dtype {
            GgmlDType::PQ2_0 => dsl::pq2_impl() == QkImpl::Dsl,
            GgmlDType::PTQ1_0 => dsl::ptq1_impl() == QkImpl::Dsl,
            _ => return Ok(None),
        };
        if !active || m == 0 || n == 0 || k == 0 {
            return Ok(None);
        }
        let src1_bf16 = match x_dtype {
            DType::F32 => false,
            DType::BF16 => true,
            _ => return Ok(None),
        };
        let elig = match dtype {
            GgmlDType::PQ2_0 => dsl::pq2_dsl_eligible(n, k, wq_bytes),
            GgmlDType::PTQ1_0 => dsl::ptq1_dsl_eligible(n, k, wq_bytes),
            _ => return Ok(None),
        };
        if elig.is_err() {
            return Ok(None);
        }

        let nwords = wq_bytes / 4;
        let (artifact, meta_words, grid, block) = if m == 1 {
            let (a, nsg, nr0) = match dtype {
                GgmlDType::PTQ1_0 => dsl::ptq1_matvec_artifact(n, src1_bf16),
                _ => dsl::pq2_matvec_artifact(n, src1_bf16),
            };
            (
                a,
                [k as u32, n as u32, nwords as u32],
                (n.div_ceil(nsg * nr0) as u32, 1, 1),
                (32u32, nsg as u32, 1),
            )
        } else {
            let (a, bm) = match dtype {
                GgmlDType::PTQ1_0 => dsl::ptq1_matmul_artifact(m, n, src1_bf16),
                _ => dsl::pq2_matmul_artifact(m, n, src1_bf16),
            };
            (
                a,
                [m as u32, n as u32, k as u32],
                (m.div_ceil(bm) as u32, n.div_ceil(64) as u32, 1),
                (128u32, 1, 1),
            )
        };

        let meta = self.dsl_const_u32_slice(&meta_words)?;
        let out = self.alloc_dsl_float(m * n, x_dtype)?;
        let w_ptr = wq.slice.as_ptr();
        let x_ptr = rocm_storage_ptr(x, x_offset, x_dtype)?;
        let o_ptr = rocm_storage_ptr(&out, 0, x_dtype)?;
        let m_ptr = meta.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(artifact, &[w_ptr, x_ptr, o_ptr, m_ptr], grid, block)?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, n, k);
        Ok(Some(out))
    }

    /// Production DSL single-token GDN conv1d step (F32/F16/BF16, k=4, updates `state` in place + SiLU).
    pub fn gdn_conv1d_step_dsl(
        &self,
        dtype: DType,
        x: (&RocmStorage, usize),
        weight: (&RocmStorage, usize),
        state: (&mut RocmStorage, usize),
        batch: usize,
        conv_dim: usize,
    ) -> Result<RocmStorage> {
        let artifact = match dtype {
            DType::F32 => "gdn_conv1d_step_k4_f32",
            DType::F16 => "gdn_conv1d_step_k4_f16",
            DType::BF16 => "gdn_conv1d_step_k4_bf16",
            other => crate::bail!("rocm gdn_conv1d_step_dsl: unsupported dtype {other:?}"),
        };
        let total = batch * conv_dim;
        if total == 0 {
            crate::bail!("rocm gdn_conv1d_step_dsl: empty shape ({batch}, {conv_dim})");
        }
        let out = self.alloc_dsl_float(total, dtype)?;
        let dims = self.dsl_const_u32_slice(&[batch as u32, conv_dim as u32])?;
        let xp = rocm_storage_ptr(x.0, x.1, dtype)?;
        let wp = rocm_storage_ptr(weight.0, weight.1, dtype)?;
        let sp = rocm_storage_ptr(state.0, state.1, dtype)?;
        let op = rocm_storage_ptr(&out, 0, dtype)?;
        let dp = dims.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(
                artifact,
                &[xp, wp, sp, op, dp],
                ((total as u32).div_ceil(64), 1, 1),
                (64, 1, 1),
            )?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, conv_dim, batch);
        Ok(out)
    }

    /// Production DSL stateful GDN conv1d prefill/chunked step (F32/F16/BF16, k=4).
    /// Returns `(out, state_out)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv1d_stateful_dsl(
        &self,
        dtype: DType,
        x: (&RocmStorage, usize),
        weight: (&RocmStorage, usize),
        state_in: (&RocmStorage, usize),
        batch: usize,
        conv_dim: usize,
        seq_len: usize,
        has_state: bool,
    ) -> Result<(RocmStorage, RocmStorage)> {
        let artifact = match dtype {
            DType::F32 => "gdn_conv1d_stateful_k4_f32",
            DType::F16 => "gdn_conv1d_stateful_k4_f16",
            DType::BF16 => "gdn_conv1d_stateful_k4_bf16",
            other => crate::bail!("rocm gdn_conv1d_stateful_dsl: unsupported dtype {other:?}"),
        };
        let total = batch * conv_dim * seq_len;
        if total == 0 {
            crate::bail!(
                "rocm gdn_conv1d_stateful_dsl: empty shape ({batch}, {conv_dim}, {seq_len})"
            );
        }
        let out = self.alloc_dsl_float(total, dtype)?;
        let state_out = self.alloc_dsl_float(batch * conv_dim * 4, dtype)?;
        let dims = self.dsl_const_u32_slice(&[
            batch as u32,
            conv_dim as u32,
            seq_len as u32,
            u32::from(has_state),
        ])?;
        let xp = rocm_storage_ptr(x.0, x.1, dtype)?;
        let wp = rocm_storage_ptr(weight.0, weight.1, dtype)?;
        let sip = rocm_storage_ptr(state_in.0, state_in.1, dtype)?;
        let sop = rocm_storage_ptr(&state_out, 0, dtype)?;
        let op = rocm_storage_ptr(&out, 0, dtype)?;
        let dp = dims.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(
                artifact,
                &[xp, wp, sip, sop, op, dp],
                ((total as u32).div_ceil(64), 1, 1),
                (64, 1, 1),
            )?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, conv_dim, seq_len);
        Ok((out, state_out))
    }

    /// Production DSL typed GDN gating (F32/F16/BF16 I/O, F32 `a_log`/`dt_bias`).
    /// Returns `(beta_out, g_out)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gating_io_dsl(
        &self,
        dtype: DType,
        b_in: (&RocmStorage, usize),
        a_in: (&RocmStorage, usize),
        a_log: (&RocmStorage, usize),
        dt_bias: (&RocmStorage, usize),
        batch_tokens: usize,
        num_heads: usize,
    ) -> Result<(RocmStorage, RocmStorage)> {
        let artifact = match dtype {
            DType::F32 => "gdn_gating_io_f32",
            DType::F16 => "gdn_gating_io_f16",
            DType::BF16 => "gdn_gating_io_bf16",
            other => crate::bail!("rocm gdn_gating_io_dsl: unsupported dtype {other:?}"),
        };
        let total = batch_tokens * num_heads;
        if total == 0 {
            crate::bail!("rocm gdn_gating_io_dsl: empty shape ({batch_tokens}, {num_heads})");
        }
        let beta_out = self.alloc_dsl_float(total, dtype)?;
        let g_out = self.alloc_dsl_float(total, dtype)?;
        let dims = self.dsl_const_u32_slice(&[total as u32, num_heads as u32])?;
        let bp = rocm_storage_ptr(b_in.0, b_in.1, dtype)?;
        let ap = rocm_storage_ptr(a_in.0, a_in.1, dtype)?;
        let alp = rocm_storage_ptr(a_log.0, a_log.1, DType::F32)?;
        let dtp = rocm_storage_ptr(dt_bias.0, dt_bias.1, DType::F32)?;
        let bop = rocm_storage_ptr(&beta_out, 0, dtype)?;
        let gop = rocm_storage_ptr(&g_out, 0, dtype)?;
        let dp = dims.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(
                artifact,
                &[bp, ap, alp, dtp, bop, gop, dp],
                ((total as u32).div_ceil(64), 1, 1),
                (64, 1, 1),
            )?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, num_heads, batch_tokens);
        Ok((beta_out, g_out))
    }

    /// Production DSL typed GDN recurrence scan (F32/F16/BF16 I/O, F32 in-place `state`).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_lanes_io_dsl(
        &self,
        dtype: DType,
        q: (&RocmStorage, usize),
        k: (&RocmStorage, usize),
        v: (&RocmStorage, usize),
        g: (&RocmStorage, usize),
        beta: (&RocmStorage, usize),
        state: (&mut RocmStorage, usize),
        bh: usize,
        seq: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<RocmStorage> {
        let artifact = match (k_dim, v_dim, dtype) {
            (128, 128, DType::F32) => "gdn_scan_lanes_io_k128_v128_l4_f32",
            (128, 128, DType::F16) => "gdn_scan_lanes_io_k128_v128_l4_f16",
            (128, 128, DType::BF16) => "gdn_scan_lanes_io_k128_v128_l4_bf16",
            (64, 64, DType::F32) => "gdn_scan_lanes_io_k64_v64_l4_f32",
            (64, 64, DType::F16) => "gdn_scan_lanes_io_k64_v64_l4_f16",
            (64, 64, DType::BF16) => "gdn_scan_lanes_io_k64_v64_l4_bf16",
            _ => crate::bail!(
                "rocm gdn_scan_lanes_io_dsl: unsupported k={k_dim} v={v_dim} dtype={dtype:?}"
            ),
        };
        if bh == 0 || seq == 0 {
            crate::bail!("rocm gdn_scan_lanes_io_dsl: empty shape ({bh}, {seq})");
        }
        let out = self.alloc_dsl_float(bh * seq * v_dim, dtype)?;
        let dims = self.dsl_const_u32_slice(&[bh as u32, seq as u32])?;
        let total_threads = bh * v_dim * 4;
        let qp = rocm_storage_ptr(q.0, q.1, dtype)?;
        let kp = rocm_storage_ptr(k.0, k.1, dtype)?;
        let vp = rocm_storage_ptr(v.0, v.1, dtype)?;
        let gp = rocm_storage_ptr(g.0, g.1, dtype)?;
        let bp = rocm_storage_ptr(beta.0, beta.1, dtype)?;
        let sp = rocm_storage_ptr(state.0, state.1, DType::F32)?;
        let op = rocm_storage_ptr(&out, 0, dtype)?;
        let dp = dims.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(
                artifact,
                &[qp, kp, vp, gp, bp, sp, op, dp],
                ((total_threads as u32).div_ceil(64), 1, 1),
                (64, 1, 1),
            )?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, bh, seq);
        Ok(out)
    }

    /// Production DSL Paged & Contiguous FlashAttention (d=64/128/256, F32/F16/BF16).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_paged_dsl(
        &self,
        dtype: DType,
        q: (&RocmStorage, usize),
        k_cache: (&RocmStorage, usize),
        v_cache: (&RocmStorage, usize),
        block_tables: (&RocmStorage, usize),
        context_lens: (&RocmStorage, usize),
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
    ) -> Result<RocmStorage> {
        let (artifact, split_artifact, reduce_artifact) = match (d, dtype) {
            (64, DType::F32) => (
                "flash_attn_paged_d64_f32",
                "flash_attn_paged_split_kv_d64_f32",
                "flash_attn_split_kv_reduce_d64_f32",
            ),
            (64, DType::F16) => (
                "flash_attn_paged_d64_f16",
                "flash_attn_paged_split_kv_d64_f16",
                "flash_attn_split_kv_reduce_d64_f16",
            ),
            (64, DType::BF16) => (
                "flash_attn_paged_d64_bf16",
                "flash_attn_paged_split_kv_d64_bf16",
                "flash_attn_split_kv_reduce_d64_bf16",
            ),
            (128, DType::F32) => (
                "flash_attn_paged_d128_f32",
                "flash_attn_paged_split_kv_d128_f32",
                "flash_attn_split_kv_reduce_d128_f32",
            ),
            (128, DType::F16) => (
                "flash_attn_paged_d128_f16",
                "flash_attn_paged_split_kv_d128_f16",
                "flash_attn_split_kv_reduce_d128_f16",
            ),
            (128, DType::BF16) => (
                "flash_attn_paged_d128_bf16",
                "flash_attn_paged_split_kv_d128_bf16",
                "flash_attn_split_kv_reduce_d128_bf16",
            ),
            (256, DType::F32) => (
                "flash_attn_paged_d256_f32",
                "flash_attn_paged_split_kv_d256_f32",
                "flash_attn_split_kv_reduce_d256_f32",
            ),
            (256, DType::F16) => (
                "flash_attn_paged_d256_f16",
                "flash_attn_paged_split_kv_d256_f16",
                "flash_attn_split_kv_reduce_d256_f16",
            ),
            (256, DType::BF16) => (
                "flash_attn_paged_d256_bf16",
                "flash_attn_paged_split_kv_d256_bf16",
                "flash_attn_split_kv_reduce_d256_bf16",
            ),
            _ => {
                crate::bail!("rocm flash_attn_paged_dsl: unsupported head_dim={d} dtype={dtype:?}")
            }
        };
        if batch == 0 || n_heads == 0 || n_kv == 0 || seq_q == 0 {
            crate::bail!("rocm flash_attn_paged_dsl: empty shape");
        }
        let total_qh = batch * n_heads * seq_q;
        let out = self.alloc_dsl_float(total_qh * d, dtype)?;
        let scale_cfg = self.dsl_const_f32_slice(&[scale, softcap])?;
        if seq_q <= 4 {
            let max_ctx_est = if kv_layout == 2 {
                max_blocks.max(1)
            } else {
                max_blocks.saturating_mul(block_size)
            };
            let num_splits = crate::flash_attn::choose_flash_split_kv(
                batch,
                n_heads,
                seq_q,
                d,
                max_ctx_est,
                sliding_window,
            );
            let partial_len = if num_splits > 1 {
                total_qh * num_splits * (d + 2)
            } else {
                1
            };
            let partial_buf = self.alloc_dsl_float(partial_len, DType::F32)?;
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
                num_splits as u32,
            ])?;
            let split_blocks = (total_qh * num_splits) as u32;
            let qp = rocm_storage_ptr(q.0, q.1, dtype)?;
            let kp = rocm_storage_ptr(k_cache.0, k_cache.1, dtype)?;
            let vp = rocm_storage_ptr(v_cache.0, v_cache.1, dtype)?;
            let btp = rocm_storage_ptr(block_tables.0, block_tables.1, DType::U32)?;
            let clp = rocm_storage_ptr(context_lens.0, context_lens.1, DType::U32)?;
            let op = rocm_storage_ptr(&out, 0, dtype)?;
            let pp = rocm_storage_ptr(&partial_buf, 0, DType::F32)?;
            let scp = scale_cfg.as_ptr();
            let mp = meta.as_ptr();
            unsafe {
                self.dispatch_dsl_raw(
                    split_artifact,
                    &[qp, kp, vp, btp, clp, op, pp, scp, mp],
                    (split_blocks, 1, 1),
                    (32, 1, 1),
                )?;
            }
            dsl::note_dispatch(QkImpl::Dsl, split_artifact, n_heads, d);
            if num_splits > 1 {
                let rmeta = self.dsl_const_u32_slice(&[
                    total_qh as u32,
                    num_splits as u32,
                    0u32,
                    0u32,
                ])?;
                let rmp = rmeta.as_ptr();
                unsafe {
                    self.dispatch_dsl_raw(
                        reduce_artifact,
                        &[pp, op, rmp],
                        (total_qh as u32, 1, 1),
                        (32, 1, 1),
                    )?;
                }
                dsl::note_dispatch(QkImpl::Dsl, reduce_artifact, n_heads, d);
            }
            return Ok(out);
        }
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
        let qp = rocm_storage_ptr(q.0, q.1, dtype)?;
        let kp = rocm_storage_ptr(k_cache.0, k_cache.1, dtype)?;
        let vp = rocm_storage_ptr(v_cache.0, v_cache.1, dtype)?;
        let btp = rocm_storage_ptr(block_tables.0, block_tables.1, DType::U32)?;
        let clp = rocm_storage_ptr(context_lens.0, context_lens.1, DType::U32)?;
        let op = rocm_storage_ptr(&out, 0, dtype)?;
        let scp = scale_cfg.as_ptr();
        let mp = meta.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(
                artifact,
                &[qp, kp, vp, btp, clp, op, scp, mp],
                (blocks, 1, 1),
                (32, 1, 1),
            )?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, n_heads, d);
        Ok(out)
    }

    /// Production DSL Fast Walsh-Hadamard Transform (`B=64..2048`, F32/F16/BF16, fused optional signs).
    #[allow(clippy::too_many_arguments)]
    pub fn fwht_hadamard_dsl(
        &self,
        dtype: DType,
        x: (&RocmStorage, usize),
        signs: Option<(&RocmStorage, usize, usize)>,
        n_blocks: usize,
        block_size: usize,
        inverse: bool,
    ) -> Result<RocmStorage> {
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
                "rocm fwht_hadamard_dsl: unsupported block_size={block_size} dtype={dtype:?}"
            ),
        };
        if n_blocks == 0 {
            crate::bail!("rocm fwht_hadamard_dsl: empty input");
        }
        let out = self.alloc_dsl_float(n_blocks * block_size, dtype)?;
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
        let xp = rocm_storage_ptr(x.0, x.1, dtype)?;
        let sp = match signs {
            Some((ss, so, _)) => rocm_storage_ptr(ss, so, DType::F32)?,
            None => dummy_signs.as_ref().unwrap().as_ptr(),
        };
        let op = rocm_storage_ptr(&out, 0, dtype)?;
        let dp = dims.as_ptr();
        unsafe {
            self.dispatch_dsl_raw(
                artifact,
                &[xp, sp, op, dp],
                (n_blocks as u32, 1, 1),
                (nt, 1, 1),
            )?;
        }
        dsl::note_dispatch(QkImpl::Dsl, artifact, n_blocks, block_size);
        Ok(out)
    }
}

fn rocm_storage_ptr(s: &RocmStorage, elem_offset: usize, dtype: DType) -> Result<*mut c_void> {
    if s.slice.dtype() != dtype {
        crate::bail!(
            "rocm_storage_ptr: expected dtype {dtype:?}, got {:?}",
            s.slice.dtype()
        );
    }
    Ok(unsafe { s.slice.offset_ptr(elem_offset) })
}
