//! The one selection point for quantized kernels that have a DSL implementation (`hanzo-kernel`).
//!
//! Backend-independent on purpose: this module decides DSL vs legacy, picks the schedule (which is
//! also the pipeline identity, i.e. the generated artifact name), checks eligibility, and records
//! what actually ran. A backend's quantized storage asks here and then executes; `hanzo-engine`
//! keeps calling `QMatMul` and never learns which implementation served it.
//!
//! Switch: `HANZO_PQ2_IMPL=dsl|legacy` (default `legacy` until the 27B qualification makes the DSL
//! path the default). `set_pq2_impl` overrides it in-process for A/B tests and benches.
//!
//! Evidence: every dispatch is counted per (impl, artifact, n, k) and every DSL pipeline compile per
//! artifact (`stats`, `report`). `HANZO_QK_STATS=1` logs the first dispatch of each key and a full
//! report at every power-of-two total from 1024 on, so a long benchmark run shows its counts in the
//! log without an exit hook.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};

/// Which implementation serves a quantized op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkImpl {
    /// The hand kernel that shipped before the DSL path (kept as the fallback).
    Legacy,
    /// The generated DSL kernel.
    Dsl,
}

impl QkImpl {
    pub fn name(self) -> &'static str {
        match self {
            QkImpl::Legacy => "legacy",
            QkImpl::Dsl => "dsl",
        }
    }
}

const UNSET: u8 = u8::MAX;
const OVERRIDE_NONE: u8 = 0;
const OVERRIDE_LEGACY: u8 = 1;
const OVERRIDE_DSL: u8 = 2;
const OVERRIDE_AUTO: u8 = 3;

static PQ2_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static PTQ1_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static GDN_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static FLASH_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static FWHT_IMPL: AtomicU8 = AtomicU8::new(UNSET);

static PQ2_DECODE_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static PQ2_PREFILL_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static PTQ1_DECODE_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static PTQ1_PREFILL_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static GDN_DECODE_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static GDN_PREFILL_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static FLASH_DECODE_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static FLASH_PREFILL_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static DSL_POLICY_AUTO: AtomicU8 = AtomicU8::new(UNSET);

/// Hardware backend executing a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkBackend {
    Metal,
    Cuda,
    Rocm,
    Vulkan,
    Cpu,
}

impl QkBackend {
    pub fn from_device(dev: &crate::Device) -> Self {
        match dev {
            crate::Device::Cpu => QkBackend::Cpu,
            crate::Device::Cuda(_) => QkBackend::Cuda,
            crate::Device::Metal(_) => QkBackend::Metal,
            #[cfg(feature = "rocm")]
            crate::Device::Rocm(_) => QkBackend::Rocm,
            #[cfg(feature = "vulkan")]
            crate::Device::Vulkan(_) => QkBackend::Vulkan,
            #[cfg(feature = "wgpu")]
            crate::Device::Wgpu(_) => QkBackend::Vulkan,
        }
    }
}

/// Execution phase (`Decode` single/small-token vs `Prefill` multi-token).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkPhase {
    /// Single-token decode (`m <= 1` or `seq_q <= 4`).
    Decode,
    /// Multi-token prefill (`m > 1` or `seq_q > 4`).
    Prefill,
}

impl QkPhase {
    pub fn from_m(m: usize) -> Self {
        if m <= 1 {
            QkPhase::Decode
        } else {
            QkPhase::Prefill
        }
    }

    pub fn from_seq_q(seq_q: usize) -> Self {
        if seq_q <= 4 {
            QkPhase::Decode
        } else {
            QkPhase::Prefill
        }
    }
}

/// Operation + shape descriptor for phase- and backend-aware kernel selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkOp {
    Fwht {
        block_size: usize,
    },
    Pq2 {
        m: usize,
        n: usize,
        k: usize,
    },
    Ptq1 {
        m: usize,
        n: usize,
        k: usize,
    },
    Gdn {
        seq_len: usize,
        k_dim: usize,
        v_dim: usize,
    },
    FlashAttn {
        batch: usize,
        n_heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
    },
}

fn encode(i: QkImpl) -> u8 {
    match i {
        QkImpl::Legacy => OVERRIDE_LEGACY,
        QkImpl::Dsl => OVERRIDE_DSL,
    }
}

fn encode_opt(i: Option<QkImpl>) -> u8 {
    match i {
        None => OVERRIDE_NONE,
        Some(QkImpl::Legacy) => OVERRIDE_LEGACY,
        Some(QkImpl::Dsl) => OVERRIDE_DSL,
    }
}

fn read_ml_or_legacy_env(env_var: &str) -> Result<String, std::env::VarError> {
    if let Some(suffix) = env_var.strip_prefix("HANZO_") {
        let ml_key = format!("ML_{suffix}");
        if let Ok(val) = std::env::var(&ml_key) {
            return Ok(val);
        }
    }
    std::env::var(env_var)
}

fn read_phase_override(atom: &AtomicU8, env_var: &str) -> Option<QkImpl> {
    match atom.load(Ordering::Relaxed) {
        OVERRIDE_NONE => None,
        OVERRIDE_LEGACY => Some(QkImpl::Legacy),
        OVERRIDE_DSL => Some(QkImpl::Dsl),
        _ => {
            let parsed = match read_ml_or_legacy_env(env_var).as_deref() {
                Ok("dsl") => Some(QkImpl::Dsl),
                Ok("legacy") => Some(QkImpl::Legacy),
                _ => None,
            };
            atom.store(encode_opt(parsed), Ordering::Relaxed);
            parsed
        }
    }
}

fn read_op_override(atom: &AtomicU8, env_var: &str) -> u8 {
    let cur = atom.load(Ordering::Relaxed);
    match cur {
        OVERRIDE_NONE | OVERRIDE_LEGACY | OVERRIDE_DSL | OVERRIDE_AUTO => cur,
        _ => {
            let v = match read_ml_or_legacy_env(env_var).as_deref() {
                Ok("dsl") => OVERRIDE_DSL,
                Ok("legacy") => OVERRIDE_LEGACY,
                Ok("auto") => OVERRIDE_AUTO,
                _ => OVERRIDE_NONE,
            };
            atom.store(v, Ordering::Relaxed);
            v
        }
    }
}

fn dsl_policy_auto_enabled() -> bool {
    match DSL_POLICY_AUTO.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let on = matches!(
                std::env::var("ML_DSL_POLICY")
                    .or_else(|_| std::env::var("HANZO_DSL_POLICY"))
                    .as_deref(),
                Ok("auto") | Ok("qualified") | Ok("metal-promoted")
            );
            DSL_POLICY_AUTO.store(u8::from(on), Ordering::Relaxed);
            on
        }
    }
}

/// Enable or disable the automatic hardware/phase-qualified DSL selection policy in-process.
pub fn set_dsl_policy_auto(enabled: bool) {
    DSL_POLICY_AUTO.store(u8::from(enabled), Ordering::Relaxed);
}

fn read_impl_env(atom: &AtomicU8, env_var: &str) -> QkImpl {
    read_impl_env_default(atom, env_var, QkImpl::Legacy)
}

fn read_impl_env_default(atom: &AtomicU8, env_var: &str, default: QkImpl) -> QkImpl {
    match read_op_override(atom, env_var) {
        OVERRIDE_LEGACY => QkImpl::Legacy,
        OVERRIDE_DSL => QkImpl::Dsl,
        _ => default,
    }
}

/// Qualified hardware-, phase-, context-, and shape-specific kernel policy:
/// - **FWHT**: `Dsl` across all GPU backends (`Metal`, `Cuda`, `Rocm`, `Vulkan`) for supported power-of-two blocks (`64..=2048`).
/// - **PQ2_0**: `Dsl` on `Metal` for both decode GEMV (`m == 1`, `+37.1%`) and prefill GEMM (`m > 1`, `+10.1%`); `Legacy` on other backends unless explicitly overridden.
/// - **PTQ1_0**: Phase-split on `Metal` — `Dsl` for prefill GEMM (`m > 1`, `+5.3%` over FWHT baseline) and `Legacy` for decode GEMV (`m == 1`, avoiding `-3.6%` decode slowdown); `Legacy` on other backends.
/// - **GDN**: `Dsl` on `Metal` for supported head dimensions (`(128, 128)` and `(64, 64)`, `+35.8%` decode / `+7.5%` prefill); `Legacy` on other backends.
/// - **FlashAttention**: Context-, batch-, and head-dim-aware on `Metal`: selects single-pass warp-register DSL decode (`Dsl`) for short contexts (`batch == 1`, `seq_q <= 4`, `0 < seq_k <= 128`, `head_dim in {64, 128, 256}`, measured `+10.3%` on M4 Max 27B), while keeping `Legacy` at longer contexts (`512..32K`) and prefill where legacy remains faster until further Split-KV partition tuning closes the gap.
pub fn qualified_policy_impl(backend: QkBackend, op: QkOp) -> QkImpl {
    match op {
        QkOp::Fwht { block_size } => {
            if matches!(block_size, 64 | 128 | 256 | 512 | 1024 | 2048)
                && matches!(
                    backend,
                    QkBackend::Metal | QkBackend::Cuda | QkBackend::Rocm | QkBackend::Vulkan
                )
            {
                QkImpl::Dsl
            } else {
                QkImpl::Legacy
            }
        }
        QkOp::Pq2 { m, n, k } => {
            if backend == QkBackend::Metal && m > 0 && n > 0 && k >= 128 && k.is_multiple_of(128) {
                QkImpl::Dsl
            } else {
                QkImpl::Legacy
            }
        }
        QkOp::Ptq1 { m, n, k } => {
            if backend == QkBackend::Metal
                && QkPhase::from_m(m) == QkPhase::Prefill
                && n > 0
                && k >= 128
                && k.is_multiple_of(128)
            {
                QkImpl::Dsl
            } else {
                QkImpl::Legacy
            }
        }
        QkOp::Gdn {
            seq_len,
            k_dim,
            v_dim,
        } => {
            if backend == QkBackend::Metal
                && seq_len > 0
                && matches!((k_dim, v_dim), (128, 128) | (64, 64))
            {
                QkImpl::Dsl
            } else {
                QkImpl::Legacy
            }
        }
        QkOp::FlashAttn {
            batch,
            n_heads,
            seq_q,
            seq_k,
            head_dim,
        } => {
            if backend == QkBackend::Metal
                && batch == 1
                && n_heads > 0
                && QkPhase::from_seq_q(seq_q) == QkPhase::Decode
                && seq_k > 0
                && seq_k <= 128
                && matches!(head_dim, 64 | 128 | 256)
            {
                QkImpl::Dsl
            } else {
                QkImpl::Legacy
            }
        }
    }
}

/// Phase-, operation-, shape-, and backend-aware kernel selection with full env-var and in-process
/// override precedence:
/// 1. Phase-specific override (`HANZO_{PQ2,PTQ1,GDN,FLASH}_{DECODE,PREFILL}_IMPL` / `set_*_{decode,prefill}_impl`)
/// 2. Operation-level override (`HANZO_{PQ2,PTQ1,GDN,FLASH,FWHT}_IMPL=dsl|legacy` / `set_*_impl`)
/// 3. Qualified hardware/phase policy (`qualified_policy_impl`) when `HANZO_*_IMPL=auto` or `HANZO_DSL_POLICY=auto`
/// 4. Conservative global default (`Dsl` for FWHT; `Legacy` for PQ2, PTQ1, GDN, FlashAttention)
pub fn select_impl(backend: QkBackend, op: QkOp) -> QkImpl {
    match op {
        QkOp::Fwht { .. } => match read_op_override(&FWHT_IMPL, "HANZO_FWHT_IMPL") {
            OVERRIDE_LEGACY => QkImpl::Legacy,
            OVERRIDE_DSL => QkImpl::Dsl,
            OVERRIDE_AUTO => qualified_policy_impl(backend, op),
            _ => QkImpl::Dsl,
        },
        QkOp::Pq2 { m, .. } => {
            let phase_ov = match QkPhase::from_m(m) {
                QkPhase::Decode => read_phase_override(&PQ2_DECODE_IMPL, "HANZO_PQ2_DECODE_IMPL"),
                QkPhase::Prefill => {
                    read_phase_override(&PQ2_PREFILL_IMPL, "HANZO_PQ2_PREFILL_IMPL")
                }
            };
            if let Some(forced) = phase_ov {
                return forced;
            }
            match read_op_override(&PQ2_IMPL, "HANZO_PQ2_IMPL") {
                OVERRIDE_LEGACY => QkImpl::Legacy,
                OVERRIDE_DSL => QkImpl::Dsl,
                OVERRIDE_AUTO => qualified_policy_impl(backend, op),
                _ if dsl_policy_auto_enabled() => qualified_policy_impl(backend, op),
                _ => QkImpl::Legacy,
            }
        }
        QkOp::Ptq1 { m, .. } => {
            let phase_ov = match QkPhase::from_m(m) {
                QkPhase::Decode => read_phase_override(&PTQ1_DECODE_IMPL, "HANZO_PTQ1_DECODE_IMPL"),
                QkPhase::Prefill => {
                    read_phase_override(&PTQ1_PREFILL_IMPL, "HANZO_PTQ1_PREFILL_IMPL")
                }
            };
            if let Some(forced) = phase_ov {
                return forced;
            }
            match read_op_override(&PTQ1_IMPL, "HANZO_PTQ1_IMPL") {
                OVERRIDE_LEGACY => QkImpl::Legacy,
                OVERRIDE_DSL => QkImpl::Dsl,
                OVERRIDE_AUTO => qualified_policy_impl(backend, op),
                _ if dsl_policy_auto_enabled() => qualified_policy_impl(backend, op),
                _ => QkImpl::Legacy,
            }
        }
        QkOp::Gdn { seq_len, .. } => {
            let phase_ov = match QkPhase::from_m(seq_len) {
                QkPhase::Decode => read_phase_override(&GDN_DECODE_IMPL, "HANZO_GDN_DECODE_IMPL"),
                QkPhase::Prefill => {
                    read_phase_override(&GDN_PREFILL_IMPL, "HANZO_GDN_PREFILL_IMPL")
                }
            };
            if let Some(forced) = phase_ov {
                return forced;
            }
            match read_op_override(&GDN_IMPL, "HANZO_GDN_IMPL") {
                OVERRIDE_LEGACY => QkImpl::Legacy,
                OVERRIDE_DSL => QkImpl::Dsl,
                OVERRIDE_AUTO => qualified_policy_impl(backend, op),
                _ if dsl_policy_auto_enabled() => qualified_policy_impl(backend, op),
                _ => QkImpl::Legacy,
            }
        }
        QkOp::FlashAttn { seq_q, .. } => {
            let phase_ov = match QkPhase::from_seq_q(seq_q) {
                QkPhase::Decode => {
                    read_phase_override(&FLASH_DECODE_IMPL, "HANZO_FLASH_DECODE_IMPL")
                }
                QkPhase::Prefill => {
                    read_phase_override(&FLASH_PREFILL_IMPL, "HANZO_FLASH_PREFILL_IMPL")
                }
            };
            if let Some(forced) = phase_ov {
                return forced;
            }
            match read_op_override(&FLASH_IMPL, "HANZO_FLASH_IMPL") {
                OVERRIDE_LEGACY => QkImpl::Legacy,
                OVERRIDE_DSL => QkImpl::Dsl,
                OVERRIDE_AUTO => qualified_policy_impl(backend, op),
                _ if dsl_policy_auto_enabled() => qualified_policy_impl(backend, op),
                _ => QkImpl::Legacy,
            }
        }
    }
}

/// Set or clear the phase-specific PQ2_0 decode (`m == 1`) override in-process.
pub fn set_pq2_decode_impl(i: Option<QkImpl>) {
    PQ2_DECODE_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific PQ2_0 prefill (`m > 1`) override in-process.
pub fn set_pq2_prefill_impl(i: Option<QkImpl>) {
    PQ2_PREFILL_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific PTQ1_0 decode (`m == 1`) override in-process.
pub fn set_ptq1_decode_impl(i: Option<QkImpl>) {
    PTQ1_DECODE_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific PTQ1_0 prefill (`m > 1`) override in-process.
pub fn set_ptq1_prefill_impl(i: Option<QkImpl>) {
    PTQ1_PREFILL_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific GDN decode (`seq_len == 1`) override in-process.
pub fn set_gdn_decode_impl(i: Option<QkImpl>) {
    GDN_DECODE_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific GDN prefill (`seq_len > 1`) override in-process.
pub fn set_gdn_prefill_impl(i: Option<QkImpl>) {
    GDN_PREFILL_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific FlashAttention decode (`seq_q <= 4`) override in-process.
pub fn set_flash_decode_impl(i: Option<QkImpl>) {
    FLASH_DECODE_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Set or clear the phase-specific FlashAttention prefill (`seq_q > 4`) override in-process.
pub fn set_flash_prefill_impl(i: Option<QkImpl>) {
    FLASH_PREFILL_IMPL.store(encode_opt(i), Ordering::Relaxed);
}

/// Clear all operation-level in-process overrides so `HANZO_DSL_POLICY` / `qualified_policy_impl`
/// can be evaluated directly.
pub fn clear_op_overrides() {
    for a in [&PQ2_IMPL, &PTQ1_IMPL, &GDN_IMPL, &FLASH_IMPL, &FWHT_IMPL] {
        a.store(OVERRIDE_NONE, Ordering::Relaxed);
    }
    set_pq2_decode_impl(None);
    set_pq2_prefill_impl(None);
    set_ptq1_decode_impl(None);
    set_ptq1_prefill_impl(None);
    set_gdn_decode_impl(None);
    set_gdn_prefill_impl(None);
    set_flash_decode_impl(None);
    set_flash_prefill_impl(None);
}

/// The implementation PQ2_0 ops use: `set_pq2_impl` if called, else `HANZO_PQ2_IMPL`.
pub fn pq2_impl() -> QkImpl {
    read_impl_env(&PQ2_IMPL, "HANZO_PQ2_IMPL")
}

/// Force the PQ2_0 implementation for this process (A/B tests, interleaved benches).
pub fn set_pq2_impl(i: QkImpl) {
    PQ2_IMPL.store(encode(i), Ordering::Relaxed);
}

/// The implementation PTQ1_0 ops use: `set_ptq1_impl` if called, else `HANZO_PTQ1_IMPL`.
pub fn ptq1_impl() -> QkImpl {
    read_impl_env(&PTQ1_IMPL, "HANZO_PTQ1_IMPL")
}

/// Force the PTQ1_0 implementation for this process (A/B tests, interleaved benches).
pub fn set_ptq1_impl(i: QkImpl) {
    PTQ1_IMPL.store(encode(i), Ordering::Relaxed);
}

/// The implementation GDN ops use: `set_gdn_impl` if called, else `HANZO_GDN_IMPL` (default `legacy`).
pub fn gdn_impl() -> QkImpl {
    read_impl_env(&GDN_IMPL, "HANZO_GDN_IMPL")
}

/// Force the GDN implementation for this process (A/B tests, interleaved benches).
pub fn set_gdn_impl(i: QkImpl) {
    GDN_IMPL.store(encode(i), Ordering::Relaxed);
}

/// The implementation FlashAttention ops use: `set_flash_impl` if called, else `HANZO_FLASH_IMPL` (default `legacy`).
pub fn flash_impl() -> QkImpl {
    read_impl_env(&FLASH_IMPL, "HANZO_FLASH_IMPL")
}

/// Force the FlashAttention implementation for this process (A/B tests, interleaved benches).
pub fn set_flash_impl(i: QkImpl) {
    FLASH_IMPL.store(encode(i), Ordering::Relaxed);
}

/// The implementation FWHT Hadamard rotation ops use: `set_fwht_impl` if called, else
/// `HANZO_FWHT_IMPL` (default `dsl`, replacing the dense O(B^2) matmul fallback on Metal/Vulkan;
/// set `HANZO_FWHT_IMPL=legacy` to force fallback for A/B).
pub fn fwht_impl() -> QkImpl {
    read_impl_env_default(&FWHT_IMPL, "HANZO_FWHT_IMPL", QkImpl::Dsl)
}

/// Force the FWHT implementation for this process (A/B tests, interleaved benches).
pub fn set_fwht_impl(i: QkImpl) {
    FWHT_IMPL.store(encode(i), Ordering::Relaxed);
}

/// The production PQ2_0 decode schedule `(nsg, nr0)` for `rows` outputs (NSG = 2, NR0 = 4
/// everywhere, from the `pq2_dsl_nr0_sweep` measurement on M4 Max). Mirrors
/// `hanzo_kernel::ternary::pq2_0_gguf_schedule` (a dev-dependency test holds them equal), which is
/// what `kernel-gen` generates artifacts for.
pub fn pq2_schedule(rows: usize) -> (usize, usize) {
    let _ = rows;
    (2, 4)
}

/// Diagnostic schedule override for measurement only: `HANZO_PQ2_NR0=2|4|8` and
/// `HANZO_PQ2_NSG=2|4` force every PQ2_0 decode matvec onto that generated artifact (`set_pq2_nr0` /
/// `set_pq2_nsg` do the same in-process). Production schedules live in `pq2_schedule`.
static NR0_FORCE: AtomicU8 = AtomicU8::new(UNSET);
static NSG_FORCE: AtomicU8 = AtomicU8::new(UNSET);

fn nr0_override() -> Option<usize> {
    match NR0_FORCE.load(Ordering::Relaxed) {
        UNSET => {
            let v = match std::env::var("HANZO_PQ2_NR0").as_deref() {
                Ok("2") => 2,
                Ok("4") => 4,
                Ok("8") => 8,
                _ => 0,
            };
            NR0_FORCE.store(v, Ordering::Relaxed);
            (v != 0).then_some(v as usize)
        }
        0 => None,
        v => Some(v as usize),
    }
}

fn nsg_override() -> Option<usize> {
    match NSG_FORCE.load(Ordering::Relaxed) {
        UNSET => {
            let v = match std::env::var("HANZO_PQ2_NSG").as_deref() {
                Ok("2") => 2,
                Ok("4") => 4,
                _ => 0,
            };
            NSG_FORCE.store(v, Ordering::Relaxed);
            (v != 0).then_some(v as usize)
        }
        0 => None,
        v => Some(v as usize),
    }
}

/// Force (Some) or clear (None) the diagnostic NR0 override in-process.
pub fn set_pq2_nr0(nr0: Option<usize>) {
    NR0_FORCE.store(nr0.map_or(0, |v| v as u8), Ordering::Relaxed);
}

/// Force (Some) or clear (None) the diagnostic NSG override in-process.
pub fn set_pq2_nsg(nsg: Option<usize>) {
    NSG_FORCE.store(nsg.map_or(0, |v| v as u8), Ordering::Relaxed);
}

fn ternary_matvec_schedule(rows: usize) -> (usize, usize) {
    let (mut nsg, mut nr0) = pq2_schedule(rows);
    if let Some(forced) = nsg_override() {
        nsg = forced;
    }
    if let Some(forced) = nr0_override() {
        nr0 = forced;
    }
    (nsg, nr0)
}

/// The generated artifact (= pipeline identity) for a PQ2_0 decode matvec, with its schedule.
pub fn pq2_matvec_artifact(rows: usize, bf16: bool) -> (&'static str, usize, usize) {
    let (nsg, nr0) = ternary_matvec_schedule(rows);
    let a = match (nsg, nr0, bf16) {
        (2, 2, false) => "matvec_pq2_0_gguf_nsg2_nr2_f32",
        (2, 4, false) => "matvec_pq2_0_gguf_nsg2_nr4_f32",
        (2, _, false) => "matvec_pq2_0_gguf_nsg2_nr8_f32",
        (2, 2, true) => "matvec_pq2_0_gguf_nsg2_nr2_bf16",
        (2, 4, true) => "matvec_pq2_0_gguf_nsg2_nr4_bf16",
        (2, _, true) => "matvec_pq2_0_gguf_nsg2_nr8_bf16",
        (_, 2, false) => "matvec_pq2_0_gguf_nsg4_nr2_f32",
        (_, 4, false) => "matvec_pq2_0_gguf_nsg4_nr4_f32",
        (_, _, false) => "matvec_pq2_0_gguf_nsg4_nr8_f32",
        (_, 2, true) => "matvec_pq2_0_gguf_nsg4_nr2_bf16",
        (_, 4, true) => "matvec_pq2_0_gguf_nsg4_nr4_bf16",
        (_, _, true) => "matvec_pq2_0_gguf_nsg4_nr8_bf16",
    };
    (a, nsg, nr0)
}

/// The generated artifact (= pipeline identity) for a PTQ1_0 decode matvec, with its schedule.
pub fn ptq1_matvec_artifact(rows: usize, bf16: bool) -> (&'static str, usize, usize) {
    let (nsg, nr0) = ternary_matvec_schedule(rows);
    let a = match (nsg, nr0, bf16) {
        (2, 2, false) => "matvec_ptq1_0_gguf_nsg2_nr2_f32",
        (2, 4, false) => "matvec_ptq1_0_gguf_nsg2_nr4_f32",
        (2, _, false) => "matvec_ptq1_0_gguf_nsg2_nr8_f32",
        (2, 2, true) => "matvec_ptq1_0_gguf_nsg2_nr2_bf16",
        (2, 4, true) => "matvec_ptq1_0_gguf_nsg2_nr4_bf16",
        (2, _, true) => "matvec_ptq1_0_gguf_nsg2_nr8_bf16",
        (_, 2, false) => "matvec_ptq1_0_gguf_nsg4_nr2_f32",
        (_, 4, false) => "matvec_ptq1_0_gguf_nsg4_nr4_f32",
        (_, _, false) => "matvec_ptq1_0_gguf_nsg4_nr8_f32",
        (_, 2, true) => "matvec_ptq1_0_gguf_nsg4_nr2_bf16",
        (_, 4, true) => "matvec_ptq1_0_gguf_nsg4_nr4_bf16",
        (_, _, true) => "matvec_ptq1_0_gguf_nsg4_nr8_bf16",
    };
    (a, nsg, nr0)
}

/// The generated artifact (= pipeline identity) and `BM` tile height (`32` for small prefills
/// `<= 48` or narrow weights `n < 2048` for higher grid parallelism; `64` otherwise to halve
/// weight dequantization bandwidth) for a PQ2_0 prefill GEMM.
pub fn pq2_matmul_artifact(m: usize, n: usize, bf16: bool) -> (&'static str, usize) {
    let bm = if m > 48 && n >= 2048 { 64 } else { 32 };
    let a = match (bm, bf16) {
        (32, false) => "matmul_pq2_0_gguf_bm32_f32",
        (32, true) => "matmul_pq2_0_gguf_bm32_bf16",
        (_, false) => "matmul_pq2_0_gguf_bm64_f32",
        (_, true) => "matmul_pq2_0_gguf_bm64_bf16",
    };
    (a, bm)
}

/// The generated artifact (= pipeline identity) and `BM` tile height for a PTQ1_0 prefill GEMM.
pub fn ptq1_matmul_artifact(m: usize, n: usize, bf16: bool) -> (&'static str, usize) {
    let bm = if m > 48 && n >= 2048 { 64 } else { 32 };
    let a = match (bm, bf16) {
        (32, false) => "matmul_ptq1_0_gguf_bm32_f32",
        (32, true) => "matmul_ptq1_0_gguf_bm32_bf16",
        (_, false) => "matmul_ptq1_0_gguf_bm64_f32",
        (_, true) => "matmul_ptq1_0_gguf_bm64_bf16",
    };
    (a, bm)
}

fn gguf_dsl_eligible(
    n: usize,
    k: usize,
    buf_bytes: usize,
    block_bytes: usize,
) -> Result<usize, &'static str> {
    if !k.is_multiple_of(128) {
        return Err("k not a multiple of 128");
    }
    let bytes = n * (k / 128) * block_bytes;
    if !bytes.is_multiple_of(4) {
        return Err("odd block count: tensor is not a whole number of u32 words");
    }
    if buf_bytes < bytes {
        return Err("weight buffer shorter than the tensor");
    }
    if bytes / 4 > u32::MAX as usize {
        return Err("tensor exceeds u32 word indexing");
    }
    Ok(bytes / 4)
}

/// Whether the native-GGUF DSL kernels can serve an `n x k` PQ2_0 weight (`34 B/block`).
pub fn pq2_dsl_eligible(n: usize, k: usize, buf_bytes: usize) -> Result<usize, &'static str> {
    gguf_dsl_eligible(n, k, buf_bytes, 34)
}

/// Whether the native-GGUF DSL kernels can serve an `n x k` PTQ1_0 weight (`28 B/block`).
pub fn ptq1_dsl_eligible(n: usize, k: usize, buf_bytes: usize) -> Result<usize, &'static str> {
    gguf_dsl_eligible(n, k, buf_bytes, 28)
}

type Key = (&'static str, &'static str, usize, usize);

fn dispatches() -> &'static Mutex<BTreeMap<Key, u64>> {
    static M: OnceLock<Mutex<BTreeMap<Key, u64>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn compiles() -> &'static Mutex<BTreeMap<String, u64>> {
    static M: OnceLock<Mutex<BTreeMap<String, u64>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(BTreeMap::new()))
}

static TOTAL: AtomicU64 = AtomicU64::new(0);

fn stats_log() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("HANZO_QK_STATS").is_ok_and(|v| v == "1"))
}

/// Record one dispatch of `artifact` (a DSL artifact or a legacy kernel name) by `imp`.
pub fn note_dispatch(imp: QkImpl, artifact: &'static str, n: usize, k: usize) {
    let first = {
        let mut m = dispatches().lock().unwrap_or_else(|e| e.into_inner());
        let c = m.entry((imp.name(), artifact, n, k)).or_insert(0);
        *c += 1;
        *c == 1
    };
    let total = TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    if stats_log() {
        if first {
            eprintln!(
                "[hanzo-qk] first dispatch impl={} artifact={artifact} n={n} k={k}",
                imp.name()
            );
        }
        if total >= 1024 && total.is_power_of_two() {
            eprintln!("[hanzo-qk] after {total} dispatches:\n{}", report());
        }
    }
}

/// Record one DSL pipeline compilation of `artifact` (called by the backend's pipeline cache).
pub fn note_compile(artifact: &str) {
    let mut m = compiles().lock().unwrap_or_else(|e| e.into_inner());
    *m.entry(artifact.to_string()).or_insert(0) += 1;
    if stats_log() {
        eprintln!("[hanzo-qk] compiled pipeline artifact={artifact}");
    }
}

/// A snapshot of what ran.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    /// (impl, artifact, n, k, calls)
    pub dispatches: Vec<(&'static str, &'static str, usize, usize, u64)>,
    /// (artifact, compiles)
    pub compiles: Vec<(String, u64)>,
}

impl Stats {
    pub fn calls(&self, imp: QkImpl) -> u64 {
        self.dispatches
            .iter()
            .filter(|d| d.0 == imp.name())
            .map(|d| d.4)
            .sum()
    }
    pub fn compiles_of(&self, artifact: &str) -> u64 {
        self.compiles
            .iter()
            .find(|c| c.0 == artifact)
            .map_or(0, |c| c.1)
    }
}

pub fn stats() -> Stats {
    let d = dispatches().lock().unwrap_or_else(|e| e.into_inner());
    let c = compiles().lock().unwrap_or_else(|e| e.into_inner());
    Stats {
        dispatches: d.iter().map(|(k, v)| (k.0, k.1, k.2, k.3, *v)).collect(),
        compiles: c.iter().map(|(k, v)| (k.clone(), *v)).collect(),
    }
}

pub fn reset_stats() {
    dispatches()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
    compiles().lock().unwrap_or_else(|e| e.into_inner()).clear();
    TOTAL.store(0, Ordering::Relaxed);
}

/// Human-readable dispatch + compile table.
pub fn report() -> String {
    let s = stats();
    let mut out = String::from("impl    calls       n      k  artifact\n");
    for (imp, a, n, k, c) in &s.dispatches {
        out.push_str(&format!("{imp:<7} {c:>6} {n:>7} {k:>6}  {a}\n"));
    }
    for (a, c) in &s.compiles {
        out.push_str(&format!("compiled {c}x  {a}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_dsl_source_of_truth() {
        for rows in [1, 64, 2048, 2049, 5120, 16383, 16384, 17408, 248_320] {
            assert_eq!(
                pq2_schedule(rows),
                hanzo_kernel::ternary::pq2_0_gguf_schedule(rows),
                "rows={rows}"
            );
        }
    }

    #[test]
    fn artifacts_are_distinct_per_schedule_and_dtype() {
        let mut seen = std::collections::HashSet::new();
        for nsg in [2usize, 4] {
            set_pq2_nsg(Some(nsg));
            for nr0 in [2usize, 4, 8] {
                set_pq2_nr0(Some(nr0));
                for bf16 in [false, true] {
                    let (a, got_nsg, got_nr0) = pq2_matvec_artifact(5000, bf16);
                    assert_eq!((got_nsg, got_nr0), (nsg, nr0));
                    assert!(a.contains(&format!("_nsg{nsg}_nr{nr0}_")), "{a}");
                    assert!(a.ends_with(if bf16 { "_bf16" } else { "_f32" }), "{a}");
                    assert!(seen.insert(a), "duplicate pipeline identity {a}");

                    let (p, got_nsg_p, got_nr0_p) = ptq1_matvec_artifact(5000, bf16);
                    assert_eq!((got_nsg_p, got_nr0_p), (nsg, nr0));
                    assert!(seen.insert(p), "duplicate pipeline identity {p}");
                }
            }
        }
        for (m, n, want_bm) in [
            (16usize, 5120usize, 32usize),
            (64, 1024, 32),
            (64, 5120, 64),
        ] {
            for bf16 in [false, true] {
                let (a, bm) = pq2_matmul_artifact(m, n, bf16);
                assert_eq!(bm, want_bm);
                seen.insert(a);
                let (p, bm_p) = ptq1_matmul_artifact(m, n, bf16);
                assert_eq!(bm_p, want_bm);
                seen.insert(p);
            }
        }
        assert_eq!(seen.len(), 24 + 8);
        set_pq2_nsg(None);
        set_pq2_nr0(None);
    }

    #[test]
    fn eligibility() {
        assert_eq!(
            pq2_dsl_eligible(5120, 5120, 5120 * 40 * 34),
            Ok(5120 * 40 * 34 / 4)
        );
        assert!(pq2_dsl_eligible(3, 128, 3 * 34).is_err()); // odd block count
        assert!(pq2_dsl_eligible(4, 256, 10).is_err()); // short buffer
        assert!(pq2_dsl_eligible(4, 100, 1 << 20).is_err());
        assert_eq!(
            ptq1_dsl_eligible(3, 128, 3 * 28),
            Ok(3 * 7) // 28 B = 7 u32 words, even for odd block counts
        );
        assert!(ptq1_dsl_eligible(4, 256, 10).is_err());
    }

    #[test]
    fn phase_and_backend_selection_policy_and_overrides() {
        clear_op_overrides();
        set_dsl_policy_auto(false);

        // 1. Conservative default when policy auto is off and no overrides are set:
        //    FWHT=Dsl; PQ2/PTQ1/GDN/Flash=Legacy.
        assert_eq!(
            select_impl(QkBackend::Metal, QkOp::Fwht { block_size: 1024 }),
            QkImpl::Dsl
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Pq2 {
                    m: 1,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Legacy
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Ptq1 {
                    m: 64,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Legacy
        );

        // 2. Qualified hardware/phase policy (`HANZO_DSL_POLICY=auto` / `set_dsl_policy_auto(true)`):
        //    - Metal: FWHT=Dsl, PQ2 decode & prefill=Dsl, PTQ1 decode=Legacy / prefill=Dsl,
        //             GDN=Dsl, FlashAttn=Legacy.
        //    - CUDA/ROCm/Vulkan: FWHT=Dsl, PQ2/PTQ1/GDN/FlashAttn=Legacy unless overridden.
        set_dsl_policy_auto(true);
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Pq2 {
                    m: 1,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Dsl
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Pq2 {
                    m: 128,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Dsl
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Ptq1 {
                    m: 1,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Legacy,
            "PTQ1 decode GEMV (m=1) must stay on Legacy on Metal under qualified policy"
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Ptq1 {
                    m: 512,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Dsl,
            "PTQ1 prefill GEMM (m>1) must select Dsl on Metal under qualified policy"
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::Gdn {
                    seq_len: 1,
                    k_dim: 128,
                    v_dim: 128
                }
            ),
            QkImpl::Dsl
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::FlashAttn {
                    batch: 1,
                    n_heads: 24,
                    seq_q: 1,
                    seq_k: 64,
                    head_dim: 256,
                }
            ),
            QkImpl::Dsl,
            "Short-context single-request Metal decode (seq_k<=128) must select single-pass warp-register DSL"
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::FlashAttn {
                    batch: 1,
                    n_heads: 24,
                    seq_q: 1,
                    seq_k: 4096,
                    head_dim: 256,
                }
            ),
            QkImpl::Legacy,
            "4K Metal decode must retain Legacy under qualified policy until Split-KV partition tuning surpasses Legacy"
        );
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::FlashAttn {
                    batch: 4,
                    n_heads: 24,
                    seq_q: 1,
                    seq_k: 64,
                    head_dim: 256,
                }
            ),
            QkImpl::Legacy
        );
        for b in [QkBackend::Cuda, QkBackend::Rocm, QkBackend::Vulkan] {
            assert_eq!(select_impl(b, QkOp::Fwht { block_size: 1024 }), QkImpl::Dsl);
            assert_eq!(
                select_impl(
                    b,
                    QkOp::Pq2 {
                        m: 1,
                        n: 5120,
                        k: 5120
                    }
                ),
                QkImpl::Legacy
            );
            assert_eq!(
                select_impl(
                    b,
                    QkOp::Ptq1 {
                        m: 512,
                        n: 5120,
                        k: 5120
                    }
                ),
                QkImpl::Legacy
            );
            assert_eq!(
                select_impl(
                    b,
                    QkOp::Gdn {
                        seq_len: 1,
                        k_dim: 128,
                        v_dim: 128
                    }
                ),
                QkImpl::Legacy
            );
        }

        // 3. Phase-specific overrides & emergency rollback take precedence over operation-level overrides and auto policy:
        set_ptq1_impl(QkImpl::Legacy);
        set_ptq1_prefill_impl(Some(QkImpl::Dsl));
        assert_eq!(
            select_impl(
                QkBackend::Cuda,
                QkOp::Ptq1 {
                    m: 1,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Legacy
        );
        assert_eq!(
            select_impl(
                QkBackend::Cuda,
                QkOp::Ptq1 {
                    m: 64,
                    n: 5120,
                    k: 5120
                }
            ),
            QkImpl::Dsl
        );
        set_flash_decode_impl(Some(QkImpl::Legacy));
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::FlashAttn {
                    batch: 1,
                    n_heads: 24,
                    seq_q: 1,
                    seq_k: 64,
                    head_dim: 256,
                }
            ),
            QkImpl::Legacy,
            "Explicit phase-specific rollback must override auto policy"
        );
        set_flash_decode_impl(Some(QkImpl::Dsl));
        assert_eq!(
            select_impl(
                QkBackend::Metal,
                QkOp::FlashAttn {
                    batch: 1,
                    n_heads: 24,
                    seq_q: 1,
                    seq_k: 32768,
                    head_dim: 256,
                }
            ),
            QkImpl::Dsl
        );

        clear_op_overrides();
        set_dsl_policy_auto(false);
    }
}
