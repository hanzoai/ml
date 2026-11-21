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
static PQ2_IMPL: AtomicU8 = AtomicU8::new(UNSET);
static PTQ1_IMPL: AtomicU8 = AtomicU8::new(UNSET);

fn encode(i: QkImpl) -> u8 {
    match i {
        QkImpl::Legacy => 0,
        QkImpl::Dsl => 1,
    }
}

fn read_impl_env(atom: &AtomicU8, env_var: &str) -> QkImpl {
    match atom.load(Ordering::Relaxed) {
        0 => QkImpl::Legacy,
        1 => QkImpl::Dsl,
        _ => {
            let i = match std::env::var(env_var).as_deref() {
                Ok("dsl") => QkImpl::Dsl,
                _ => QkImpl::Legacy,
            };
            atom.store(encode(i), Ordering::Relaxed);
            i
        }
    }
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

pub fn ptq1_schedule(rows: usize) -> (usize, usize) {
    if rows <= 2048 {
        (2, 2)
    } else {
        (2, 4)
    }
}

/// The generated artifact (= pipeline identity) for a PTQ1_0 decode matvec, with its schedule.
pub fn ptq1_matvec_artifact(rows: usize, bf16: bool) -> (&'static str, usize, usize) {
    let (mut nsg, mut nr0) = ptq1_schedule(rows);
    if let Some(forced) = nsg_override() {
        nsg = forced;
    }
    if let Some(forced) = nr0_override() {
        nr0 = forced;
    }
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
    if k % 128 != 0 {
        return Err("k not a multiple of 128");
    }
    let bytes = n * (k / 128) * block_bytes;
    if bytes % 4 != 0 {
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
        for (m, n, want_bm) in [(16usize, 5120usize, 32usize), (64, 1024, 32), (64, 5120, 64)] {
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
}
