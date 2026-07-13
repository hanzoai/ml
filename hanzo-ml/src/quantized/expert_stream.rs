//! Disk-streaming LRU expert cache for large MoE GGUFs (the colibri "low memory mode").
//!
//! A GGUF MoE stores each layer's routed experts as three stacked banks -- `ffn_gate_exps`,
//! `ffn_up_exps`, `ffn_down_exps` -- each a `[n_experts, n, k]` quantized tensor. Loading them
//! resident is what makes GLM-5.2 (202 GB) need 202 GB of RAM. This module keeps them on NVMe and
//! streams one expert's `[n, k]` slice on demand: pin -> per-bank LRU -> `pread` (+ `fadvise`
//! DONTNEED so the page cache can't balloon). Resident RAM becomes `pinned + LRU` owned slabs --
//! bounded and under our control, not at the mercy of the OS page cache. With a warm cache and the
//! hot experts pinned, decode is matmul/bandwidth bound just like the resident path.
//!
//! Faithful to colibri (`c/glm.c`): `expert_load` (pread coalesced + fadvise), `cap_for_ram`
//! (auto-size the cache from MemAvailable, auto-raise to fill big-RAM boxes), `pin_load`/AUTOPIN
//! (learn the hottest experts into a sidecar and pin them at startup). Here the natural unit is one
//! bank per (layer x projection) rather than colibri's per-expert triple, because the three
//! projections are three separate GGUF tensors -- more orthogonal, same mechanism.
//!
//! Bit-exact by construction: a streamed expert is the identical file bytes the resident slice
//! would hold, so `indexed_moe_forward` produces the identical output; only the fetch path differs.

use crate::Result;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::GgmlDType;

/// Runtime gate. Streaming is OFF by default so resident behaviour is unchanged for models that
/// fit; the loader flips it on for `--stream-experts`. Compile-time-always, runtime-gated -- one way.
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Reserve left to the kernel page cache + activations/logits, mirroring colibri's honest slack.
const PAGE_CACHE_RESERVE: u64 = 2_500_000_000;
const ACTIVATION_RESERVE: u64 = 1_200_000_000;
/// Fraction of MemAvailable the cache may claim (rest breathes for OS + wrapper); colibri's 0.88.
const BUDGET_FRACTION: f64 = 0.88;
/// Default hot-set: fraction of a bank's cap auto-pinned from the learned usage sidecar.
const PIN_FRACTION: f64 = 0.25;

/// One streaming expert bank = one stacked `[n_experts, n, k]` GGUF tensor kept on disk.
pub struct ExpertStreamBank {
    /// Tensor name (`blk.{L}.ffn_gate_exps.weight`); the stable key for the usage sidecar.
    name: String,
    file: File,
    /// Byte offset of expert 0 within the file (`tensor_data_offset + tensor.offset`).
    base_offset: u64,
    /// Bytes per expert = `n*k / block_size * type_size`.
    expert_bytes: usize,
    n_experts: usize,
    dtype: GgmlDType,
    n: usize,
    k: usize,
    inner: Mutex<BankState>,
}

struct BankState {
    /// Max experts held in the LRU (the pinned hot-set is separate and unbounded by `cap`).
    cap: usize,
    clock: u64,
    lru: HashMap<u32, LruEntry>,
    pinned: HashMap<u32, Arc<[u8]>>,
    /// Per-expert routing frequency this run; persisted to the sidecar for AUTOPIN next run.
    usage: Vec<u64>,
    hits: u64,
    misses: u64,
}

struct LruEntry {
    bytes: Arc<[u8]>,
    used: u64,
}

impl ExpertStreamBank {
    /// Open a bank over `path`. Registers it so the global budget can size every bank at once.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        name: String,
        path: &std::path::Path,
        base_offset: u64,
        expert_bytes: usize,
        n_experts: usize,
        dtype: GgmlDType,
        n: usize,
        k: usize,
    ) -> Result<Arc<Self>> {
        let file = File::open(path)?;
        let bank = Arc::new(Self {
            name,
            file,
            base_offset,
            expert_bytes,
            n_experts,
            dtype,
            n,
            k,
            inner: Mutex::new(BankState {
                // Provisional cap of 1 until `finalize` sizes it from RAM; never zero (a miss must
                // always be admittable so a fetch can't deadlock waiting for room).
                cap: 1,
                clock: 0,
                lru: HashMap::new(),
                pinned: HashMap::new(),
                usage: vec![0; n_experts],
                hits: 0,
                misses: 0,
            }),
        });
        registry().lock().expect("registry lock").register(&bank);
        Ok(bank)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Per-expert matrix dims `[n, k]` (the shape a fetched expert's `QTensor` takes).
    pub fn expert_dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    pub fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }

    /// Full logical size of the whole bank as if resident (for shape/size accounting only).
    pub fn logical_bytes(&self) -> usize {
        self.expert_bytes * self.n_experts
    }

    /// Fetch expert `eid`'s raw GGML bytes: pinned hot-set -> LRU -> disk. Bit-identical to the
    /// resident slice; records the access for AUTOPIN learning.
    pub fn fetch(&self, eid: u32) -> Result<Arc<[u8]>> {
        let mut guard = self.inner.lock().expect("expert bank lock poisoned");
        // Reborrow as `&mut BankState` so disjoint fields (hits/clock vs pinned/lru) can be borrowed
        // independently -- the MutexGuard's Deref would otherwise treat every field access as the
        // whole guard.
        let st = &mut *guard;
        if (eid as usize) < st.usage.len() {
            st.usage[eid as usize] = st.usage[eid as usize].saturating_add(1);
        }
        if let Some(bytes) = st.pinned.get(&eid) {
            st.hits += 1;
            return Ok(bytes.clone());
        }
        if let Some(entry) = st.lru.get_mut(&eid) {
            st.hits += 1;
            st.clock += 1;
            entry.used = st.clock;
            return Ok(entry.bytes.clone());
        }
        st.misses += 1;
        let bytes = self.read_expert(eid)?;
        self.admit(st, eid, bytes.clone());
        Ok(bytes)
    }

    /// Insert into the LRU, evicting the least-recently-used entry when at cap.
    fn admit(&self, st: &mut BankState, eid: u32, bytes: Arc<[u8]>) {
        if st.lru.len() >= st.cap {
            if let Some(&victim) = st
                .lru
                .iter()
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| k)
            {
                st.lru.remove(&victim);
            }
        }
        st.clock += 1;
        let used = st.clock;
        st.lru.insert(eid, LruEntry { bytes, used });
    }

    /// Positional read of one expert's bytes (thread-safe, no shared seek), then advise the kernel
    /// to drop those file pages so the page cache stays bounded (colibri's `POSIX_FADV_DONTNEED`).
    fn read_expert(&self, eid: u32) -> Result<Arc<[u8]>> {
        let off = self.base_offset + eid as u64 * self.expert_bytes as u64;
        let mut buf = vec![0u8; self.expert_bytes];
        read_exact_at(&self.file, &mut buf, off)?;
        fadvise_dontneed(&self.file, off, self.expert_bytes as u64);
        Ok(Arc::from(buf.into_boxed_slice()))
    }

    /// Pin expert `eid` into the hot-set (loads it if absent). Idempotent.
    pub fn pin(&self, eid: u32) -> Result<()> {
        let mut st = self.inner.lock().expect("expert bank lock poisoned");
        if st.pinned.contains_key(&eid) {
            return Ok(());
        }
        let bytes = if let Some(entry) = st.lru.remove(&eid) {
            entry.bytes
        } else {
            drop(st);
            let b = self.read_expert(eid)?;
            st = self.inner.lock().expect("expert bank lock poisoned");
            b
        };
        st.pinned.insert(eid, bytes);
        Ok(())
    }

    /// Set the LRU cap (max non-pinned experts held resident). Sized by [`finalize`] from RAM; also
    /// settable directly (tests, manual budgets).
    pub fn set_cap(&self, cap: usize) {
        let mut st = self.inner.lock().expect("expert bank lock poisoned");
        st.cap = cap.max(1);
    }

    /// `(pinned, cached, cap, hits, misses)` snapshot for the resident-RAM / hit-rate proof.
    pub fn stats(&self) -> (usize, usize, usize, u64, u64) {
        let st = self.inner.lock().expect("expert bank lock poisoned");
        (st.pinned.len(), st.lru.len(), st.cap, st.hits, st.misses)
    }

    /// Bytes currently held resident by this bank (pinned + cached slabs).
    pub fn resident_bytes(&self) -> usize {
        let st = self.inner.lock().expect("expert bank lock poisoned");
        (st.pinned.len() + st.lru.len()) * self.expert_bytes
    }

    fn usage_snapshot(&self) -> Vec<(u32, u64)> {
        let st = self.inner.lock().expect("expert bank lock poisoned");
        st.usage
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(e, &c)| (e as u32, c))
            .collect()
    }
}

/// Global registry: every live bank + the learned-usage sidecar path. Lets one budget size all
/// banks together (colibri `cap_for_ram`) and one pass learn/pin the hottest experts.
struct Registry {
    banks: Vec<Weak<ExpertStreamBank>>,
    sidecar: Option<PathBuf>,
}

fn registry() -> &'static Mutex<Registry> {
    static REG: OnceLock<Mutex<Registry>> = OnceLock::new();
    REG.get_or_init(|| {
        Mutex::new(Registry {
            banks: Vec::new(),
            sidecar: None,
        })
    })
}

impl Registry {
    fn register(&mut self, bank: &Arc<ExpertStreamBank>) {
        self.banks.push(Arc::downgrade(bank));
    }

    fn live(&self) -> Vec<Arc<ExpertStreamBank>> {
        self.banks.iter().filter_map(Weak::upgrade).collect()
    }
}

/// Point the learning-pin at a sidecar file (`<model_dir>/.hanzo_experts_usage`).
pub fn set_usage_sidecar(path: PathBuf) {
    registry().lock().expect("registry lock").sidecar = Some(path);
}

/// Total bytes held resident across all live streaming banks (the low-RAM proof).
pub fn total_resident_bytes() -> usize {
    registry()
        .lock()
        .expect("registry lock")
        .live()
        .iter()
        .map(|b| b.resident_bytes())
        .sum()
}

/// Size every bank's LRU cap from available RAM and auto-pin the hottest experts from the sidecar.
/// Call once after the dense weights are resident and all expert banks are open.
pub fn finalize() {
    let banks = registry().lock().expect("registry lock").live();
    if banks.is_empty() {
        return;
    }
    // Largest per-expert stride across banks -> the conservative divisor (banks are near-equal).
    let expert_bytes = banks.iter().map(|b| b.expert_bytes).max().unwrap_or(1).max(1);
    let n_banks = banks.len() as u64;

    let avail = mem_available_bytes();
    let budget = (avail as f64 * BUDGET_FRACTION) as u64;
    let slack = PAGE_CACHE_RESERVE + ACTIVATION_RESERVE;
    let for_cache = budget.saturating_sub(slack);
    let mut cap = (for_cache / (n_banks * expert_bytes as u64)) as usize;

    // Never below 1 (a miss must be admittable), never above n_experts (a bank can't fill more).
    let max_experts = banks.iter().map(|b| b.n_experts).max().unwrap_or(1);
    cap = cap.clamp(1, max_experts);
    for b in &banks {
        b.set_cap(cap);
    }

    let pinned = load_and_pin(&banks, cap);
    register_atexit_save();

    eprintln!(
        "[stream-experts] {} banks x {:.1} MB/expert; MemAvailable {:.1} GB -> per-bank cap {} \
         (projected cache {:.1} GB), auto-pinned {} experts",
        banks.len(),
        expert_bytes as f64 / 1e6,
        avail as f64 / 1e9,
        cap,
        (cap as u64 * n_banks * expert_bytes as u64) as f64 / 1e9,
        pinned,
    );
}

/// Read the usage sidecar (if any) and pin each bank's hottest experts, up to `PIN_FRACTION * cap`.
fn load_and_pin(banks: &[Arc<ExpertStreamBank>], cap: usize) -> usize {
    let sidecar = match registry().lock().expect("registry lock").sidecar.clone() {
        Some(p) if p.exists() => p,
        _ => return 0,
    };
    let text = match std::fs::read_to_string(&sidecar) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    // `name eid count` per line.
    let mut by_name: HashMap<&str, Vec<(u32, u64)>> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(name), Some(eid), Some(cnt)) = (it.next(), it.next(), it.next()) else {
            continue;
        };
        if let (Ok(eid), Ok(cnt)) = (eid.parse::<u32>(), cnt.parse::<u64>()) {
            by_name.entry(name).or_default().push((eid, cnt));
        }
    }
    let pin_budget = ((cap as f64 * PIN_FRACTION) as usize).max(1);
    let mut pinned = 0usize;
    for b in banks {
        let Some(rows) = by_name.get(b.name.as_str()) else {
            continue;
        };
        let mut rows = rows.clone();
        rows.sort_by(|a, c| c.1.cmp(&a.1));
        for (eid, _) in rows.into_iter().take(pin_budget) {
            if eid as usize >= b.n_experts {
                continue;
            }
            if b.pin(eid).is_ok() {
                pinned += 1;
            }
        }
    }
    pinned
}

/// Persist this run's per-expert routing frequencies (merged with any prior counts) so the next
/// startup can AUTOPIN the hottest. Call at shutdown.
pub fn save_usage() -> Result<()> {
    let (banks, sidecar) = {
        let reg = registry().lock().expect("registry lock");
        (reg.live(), reg.sidecar.clone())
    };
    let Some(path) = sidecar else {
        return Ok(());
    };
    // Merge with existing counts so the learned distribution accumulates across runs.
    let mut merged: HashMap<(String, u32), u64> = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(&path) {
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if let (Some(name), Some(eid), Some(cnt)) = (it.next(), it.next(), it.next()) {
                if let (Ok(eid), Ok(cnt)) = (eid.parse::<u32>(), cnt.parse::<u64>()) {
                    *merged.entry((name.to_string(), eid)).or_default() += cnt;
                }
            }
        }
    }
    for b in &banks {
        for (eid, cnt) in b.usage_snapshot() {
            *merged.entry((b.name.clone(), eid)).or_default() += cnt;
        }
    }
    let mut out = String::new();
    for ((name, eid), cnt) in merged {
        out.push_str(&format!("{name} {eid} {cnt}\n"));
    }
    std::fs::write(&path, out)?;
    Ok(())
}

/// Persist the learned usage on clean process exit so the next run can AUTOPIN, without threading a
/// shutdown hook through the engine. Registered once; a SIGKILL won't fire it, a normal exit/SIGTERM
/// will. Best-effort: a write failure only means no learning carried forward.
fn register_atexit_save() {
    #[cfg(unix)]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            extern "C" fn on_exit() {
                let _ = save_usage();
            }
            unsafe {
                libc::atexit(on_exit);
            }
        });
    }
}

// ---- platform: positional read + page-cache drop ---------------------------------------------

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)?;
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(_file: &File, _buf: &mut [u8], _offset: u64) -> Result<()> {
    crate::bail!("streaming experts require a unix positional-read (pread) platform")
}

/// Advise the kernel to drop the just-read file pages so the page cache can't grow unbounded while
/// we stream the whole model past it. Best-effort: a failure only means more page cache, not wrong
/// results. Offset/len need no alignment for `POSIX_FADV_DONTNEED`.
#[cfg(unix)]
fn fadvise_dontneed(file: &File, offset: u64, len: u64) {
    use std::os::unix::io::AsRawFd;
    // SAFETY: fd is valid for the call; posix_fadvise only advises, never mutates our memory.
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(unix))]
fn fadvise_dontneed(_file: &File, _offset: u64, _len: u64) {}

/// MemAvailable in bytes. Linux reads `/proc/meminfo` (the true reclaimable ceiling); elsewhere a
/// conservative fallback keeps the cache small rather than risking OOM.
fn mem_available_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    // `MemAvailable:   12345678 kB`
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok())
                    {
                        return kb.saturating_mul(1024);
                    }
                }
            }
        }
    }
    // Fallback (non-linux or unreadable): assume 8 GB free, matching colibri's floor.
    static WARNED: AtomicU64 = AtomicU64::new(0);
    if WARNED.swap(1, Ordering::Relaxed) == 0 {
        eprintln!("[stream-experts] MemAvailable unreadable; assuming 8 GB free");
    }
    8_000_000_000
}
