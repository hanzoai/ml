//! Disk-streaming LRU expert cache: the low-memory mode for large MoE GGUFs.
//!
//! Each GGUF MoE layer stores its routed experts as three stacked `[n_experts, n, k]` banks
//! (`ffn_{gate,up,down}_exps`). Instead of loading them resident, keep them on NVMe and stream one
//! expert's `[n, k]` slice on demand: pin -> per-bank LRU -> pread (+ fadvise DONTNEED). Resident
//! RAM is then `pinned + LRU` slabs -- bounded, not the whole model. One bank per layer x projection.
//!
//! Bit-exact: a streamed expert is the same file bytes the resident slice holds, so
//! `indexed_moe_forward` yields identical output; only the fetch path changes.

use crate::Result;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::GgmlDType;

/// Runtime gate, default OFF so resident behaviour is unchanged. The loader flips it on.
static ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Reserved out of the RAM budget for page cache and activations.
const PAGE_CACHE_RESERVE: u64 = 2_500_000_000;
const ACTIVATION_RESERVE: u64 = 1_200_000_000;
/// Fraction of MemAvailable the cache may claim.
const BUDGET_FRACTION: f64 = 0.88;
/// Fraction of a bank's cap auto-pinned from the learned usage sidecar.
const PIN_FRACTION: f64 = 0.25;
/// Max hot-set swaps per [`repin`] pass. Four-swap cap: bounds work and
/// stops a single turn from churning the whole tier.
const REPIN_MAX_SWAPS: usize = 4;

/// One stacked `[n_experts, n, k]` GGUF expert bank kept on disk.
pub struct ExpertStreamBank {
    /// Tensor name; the sidecar key for learned pinning.
    name: String,
    file: File,
    /// Byte offset of expert 0 in the file.
    base_offset: u64,
    /// Bytes per expert (`n*k / block_size * type_size`).
    expert_bytes: usize,
    n_experts: usize,
    dtype: GgmlDType,
    n: usize,
    k: usize,
    inner: Mutex<BankState>,
}

struct BankState {
    /// Max experts in the LRU; the pinned set is separate.
    cap: usize,
    clock: u64,
    lru: HashMap<u32, LruEntry>,
    pinned: HashMap<u32, Arc<[u8]>>,
    /// Per-expert routing frequency this run, persisted for pinning next run. The long-term
    /// signal: never decayed, so `.coli_usage`-style pinning stays stable across runs.
    usage: Vec<u64>,
    /// Per-expert session heat: the short-term signal `repin` adapts on. Halved each pass
    /// (`tier_decay`) so recent routing dominates and a warm expert cools once it stops
    /// being picked. Distinct from `usage` precisely because it is decayed.
    heat: Vec<u32>,
    hits: u64,
    misses: u64,
}

struct LruEntry {
    bytes: Arc<[u8]>,
    used: u64,
}

impl ExpertStreamBank {
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
            // cap starts at 1 (never 0: a miss must be admittable); `finalize` sizes it from RAM.
            inner: Mutex::new(BankState {
                cap: 1,
                clock: 0,
                lru: HashMap::new(),
                pinned: HashMap::new(),
                usage: vec![0; n_experts],
                heat: vec![0; n_experts],
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

    pub fn expert_dims(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    pub fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }

    /// Full size of the bank as if resident (shape/size accounting only).
    pub fn logical_bytes(&self) -> usize {
        self.expert_bytes * self.n_experts
    }

    /// pinned -> LRU -> disk.
    pub fn fetch(&self, eid: u32) -> Result<Arc<[u8]>> {
        let mut guard = self.inner.lock().expect("expert bank lock poisoned");
        // Reborrow so disjoint fields (hits/clock vs pinned/lru) can be borrowed independently.
        let st = &mut *guard;
        if (eid as usize) < st.usage.len() {
            st.usage[eid as usize] = st.usage[eid as usize].saturating_add(1);
            st.heat[eid as usize] = st.heat[eid as usize].saturating_add(1);
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

    fn admit(&self, st: &mut BankState, eid: u32, bytes: Arc<[u8]>) {
        if st.lru.len() >= st.cap {
            if let Some(&victim) = st.lru.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k) {
                st.lru.remove(&victim);
            }
        }
        st.clock += 1;
        let used = st.clock;
        st.lru.insert(eid, LruEntry { bytes, used });
    }

    fn read_expert(&self, eid: u32) -> Result<Arc<[u8]>> {
        let off = self.base_offset + eid as u64 * self.expert_bytes as u64;
        let mut buf = vec![0u8; self.expert_bytes];
        read_exact_at(&self.file, &mut buf, off)?;
        fadvise_dontneed(&self.file, off, self.expert_bytes as u64);
        Ok(Arc::from(buf.into_boxed_slice()))
    }

    /// Load `eid` into the pinned hot-set (idempotent).
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

    /// Live tier adaptation (`--repin`). Decay the session heat, then swap up to
    /// [`REPIN_MAX_SWAPS`] of the coldest pinned experts for the hottest streamed ones, each
    /// only when the heat gap clears the hysteresis margin ([`tier_pick_swap`]). Correctness
    /// neutral: the pinned *set* changes, never the bytes -- a promoted expert is the same
    /// file slice it would stream, so `indexed_moe_forward` output is bit-identical. Cold
    /// experts fall back to the LRU (still warm), hot experts come from the LRU or disk.
    /// Call at a turn boundary. Returns the number of swaps performed.
    pub fn repin(&self, max_swaps: usize) -> Result<usize> {
        let mut guard = self.inner.lock().expect("expert bank lock poisoned");
        let st = &mut *guard;
        tier_decay(&mut st.heat);
        let mut swaps = 0usize;
        while swaps < max_swaps {
            let pinned: Vec<u32> = st.pinned.keys().copied().collect();
            let Some((slot, hot, _gain)) = tier_pick_swap(&st.heat, &pinned) else {
                break;
            };
            let cold = pinned[slot];
            // Demote the cold expert to the LRU: still resident, still a bit-identical hit.
            if let Some(bytes) = st.pinned.remove(&cold) {
                self.admit(st, cold, bytes);
            }
            // Promote the hot expert: reuse an LRU copy if present, else read it from disk.
            let bytes = match st.lru.remove(&hot) {
                Some(entry) => entry.bytes,
                None => self.read_expert(hot)?,
            };
            st.pinned.insert(hot, bytes);
            swaps += 1;
        }
        Ok(swaps)
    }

    /// Background-thread readahead (router-lookahead): warm `eid` into the LRU ahead of the
    /// `fetch` that needs it, overlapping disk I/O with compute. The read owns its bytes in an
    /// `Arc` (unlike a bare `WILLNEED` page-cache hint, which memory pressure can re-evict), so
    /// a later `fetch` is a guaranteed hit on identical bytes -- correctness neutral. Skips
    /// experts already resident. Runs on the shared prefetch worker; never on the caller.
    fn prefetch_resident(&self, eid: u32) -> Result<()> {
        {
            let st = self.inner.lock().expect("expert bank lock poisoned");
            if st.pinned.contains_key(&eid) || st.lru.contains_key(&eid) {
                return Ok(());
            }
        }
        // Read without the lock so concurrent fetches on this bank are not stalled by disk.
        let bytes = self.read_expert(eid)?;
        let mut guard = self.inner.lock().expect("expert bank lock poisoned");
        let st = &mut *guard;
        // Re-check: a fetch may have admitted it while we were reading.
        if st.pinned.contains_key(&eid) || st.lru.contains_key(&eid) {
            return Ok(());
        }
        self.admit(st, eid, bytes);
        Ok(())
    }

    /// Enqueue a router-lookahead prefetch of `eid`. Best effort and
    /// non-blocking: dropped when disabled, out of range, or the worker queue is full -- a
    /// missed prefetch only costs a later on-demand `fetch`, never correctness.
    pub fn prefetch(self: &Arc<Self>, eid: u32) {
        if !prefetch_enabled() || eid as usize >= self.n_experts {
            return;
        }
        let _ = prefetcher().try_send(PrefetchJob {
            bank: Arc::downgrade(self),
            eid,
        });
    }

    /// Sized by [`finalize`] from RAM; also settable directly.
    pub fn set_cap(&self, cap: usize) {
        let mut st = self.inner.lock().expect("expert bank lock poisoned");
        st.cap = cap.max(1);
    }

    /// `(pinned, cached, cap, hits, misses)`.
    pub fn stats(&self) -> (usize, usize, usize, u64, u64) {
        let st = self.inner.lock().expect("expert bank lock poisoned");
        (st.pinned.len(), st.lru.len(), st.cap, st.hits, st.misses)
    }

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

/// Every live bank + the usage sidecar path, so one budget sizes all banks and one pass pins.
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

pub fn set_usage_sidecar(path: PathBuf) {
    registry().lock().expect("registry lock").sidecar = Some(path);
}

pub fn total_resident_bytes() -> usize {
    registry()
        .lock()
        .expect("registry lock")
        .live()
        .iter()
        .map(|b| b.resident_bytes())
        .sum()
}

// ---------------------------------------------------------------------------
// Live tier adaptation: a pure heat-swap decision + decay, so
// the swap policy is testable in isolation from the I/O and locking around it.
// ---------------------------------------------------------------------------

/// Pick one pinned slot to replace with the hottest streamed expert, or `None` if no swap
/// clears the hysteresis margin. Pure and total: the caller owns all I/O and locking.
///
/// `heat[e]` is expert `e`'s session heat; `pinned` lists the currently pinned expert ids.
/// Heat-swap rule: coldest pinned vs hottest non-resident, admitted only
/// when `hot > cold + cold/4 + 4`. The `cold/4` (25%) margin stops ping-pong between two
/// near-equal experts; the `+4` covers tiny samples where the ratio alone is noisy. Returns
/// `(slot, hot_eid, gain)` where `slot` indexes `pinned` and `gain = hot_heat - cold_heat`.
fn tier_pick_swap(heat: &[u32], pinned: &[u32]) -> Option<(usize, u32, i64)> {
    if heat.is_empty() || pinned.is_empty() {
        return None;
    }
    // Coldest pinned slot (first minimum wins on ties, strict `<`).
    let cold = pinned
        .iter()
        .enumerate()
        .min_by_key(|&(_, &p)| heat[p as usize])
        .map(|(z, _)| z)
        .expect("pinned is non-empty");
    // Hottest non-resident expert (first maximum wins on ties, strict `>`).
    let mut hot: Option<usize> = None;
    let mut hot_heat = 0u32;
    for (e, &h) in heat.iter().enumerate() {
        let resident = pinned.iter().any(|&p| p as usize == e);
        if !resident && h > hot_heat {
            hot_heat = h;
            hot = Some(e);
        }
    }
    let hot = hot?;
    let cold_heat = heat[pinned[cold] as usize];
    if hot_heat <= cold_heat + (cold_heat >> 2) + 4 {
        return None;
    }
    Some((cold, hot as u32, hot_heat as i64 - cold_heat as i64))
}

/// Halve every expert's session heat (heat decay): recent routing keeps its lead,
/// stale heat fades toward zero so a once-hot expert eventually loses its pin.
fn tier_decay(heat: &mut [u32]) {
    for h in heat.iter_mut() {
        *h >>= 1;
    }
}

/// Tokens between [`repin_all`] passes (`--repin N`); `0`/unset disables live tier
/// adaptation, so the hot-set stays exactly as [`finalize`] pinned it. Read once.
pub fn repin_interval() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("STREAM_EXPERTS_REPIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Adapt every live bank's hot-set to recent routing heat. No-op unless `STREAM_EXPERTS_REPIN`
/// is set. The engine calls this at turn boundaries every [`repin_interval`] tokens. Returns
/// the total number of swaps performed across all banks.
pub fn repin_all() -> usize {
    if repin_interval() == 0 {
        return 0;
    }
    // Snapshot the live banks and DROP the registry guard before iterating: repin() takes each
    // bank's own Mutex across up to REPIN_MAX_SWAPS blocking preads, so holding the global
    // registry lock across all of them would serialize every bank behind one. Same idiom as
    // finalize()/save_usage() below.
    let banks = registry().lock().expect("registry lock").live();
    banks
        .iter()
        .map(|b| b.repin(REPIN_MAX_SWAPS).unwrap_or(0))
        .sum()
}

// ---------------------------------------------------------------------------
// Async prefetch (router-lookahead): one shared I/O worker warms experts into the
// LRU ahead of the fetch that needs them, overlapping disk with compute.
// ---------------------------------------------------------------------------

/// Whether router-lookahead prefetch is on (`STREAM_EXPERTS_PREFETCH`, default OFF). Read once.
pub fn prefetch_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("STREAM_EXPERTS_PREFETCH")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

struct PrefetchJob {
    bank: Weak<ExpertStreamBank>,
    eid: u32,
}

/// The single background readahead worker, started on first use. Bounded queue: `try_send`
/// drops a job when full (ring buffer: full = drop), so a saturated disk can
/// never back-pressure or block the compute thread. `Weak` keeps a queued job from pinning a
/// bank whose model has been dropped.
fn prefetcher() -> &'static SyncSender<PrefetchJob> {
    static P: OnceLock<SyncSender<PrefetchJob>> = OnceLock::new();
    P.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<PrefetchJob>(256);
        std::thread::Builder::new()
            .name("stream-experts-prefetch".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    if let Some(bank) = job.bank.upgrade() {
                        let _ = bank.prefetch_resident(job.eid);
                    }
                }
            })
            .expect("spawn stream-experts prefetch worker");
        tx
    })
}

/// Size every bank's LRU cap from available RAM and pin the learned-hot experts. Call once after
/// the dense weights are resident and all banks are open.
pub fn finalize() {
    let banks = registry().lock().expect("registry lock").live();
    if banks.is_empty() {
        return;
    }
    let expert_bytes = banks.iter().map(|b| b.expert_bytes).max().unwrap_or(1).max(1);
    let n_banks = banks.len() as u64;

    // STREAM_EXPERTS_RAM_GB forces the cache budget; otherwise size from live MemAvailable.
    let forced_gb = std::env::var("STREAM_EXPERTS_RAM_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|g| *g > 0.0);
    let avail = match forced_gb {
        Some(gb) => (gb * 1e9) as u64,
        None => mem_available_bytes(),
    };
    let budget = (avail as f64 * BUDGET_FRACTION) as u64;
    let slack = PAGE_CACHE_RESERVE + ACTIVATION_RESERVE;
    let for_cache = budget.saturating_sub(slack);
    let mut cap = (for_cache / (n_banks * expert_bytes as u64)) as usize;

    let max_experts = banks.iter().map(|b| b.n_experts).max().unwrap_or(1);
    cap = cap.clamp(1, max_experts);
    for b in &banks {
        b.set_cap(cap);
    }

    let pinned = load_and_pin(&banks, cap);
    register_atexit_save();

    eprintln!(
        "[stream-experts] {} banks x {:.1} MB/expert; budget {:.1} GB -> cap {}/bank \
         (cache {:.1} GB), pinned {}",
        banks.len(),
        expert_bytes as f64 / 1e6,
        avail as f64 / 1e9,
        cap,
        (cap as u64 * n_banks * expert_bytes as u64) as f64 / 1e9,
        pinned,
    );
}

/// Pin each bank's hottest experts (up to `PIN_FRACTION * cap`) from the usage sidecar.
fn load_and_pin(banks: &[Arc<ExpertStreamBank>], cap: usize) -> usize {
    let sidecar = match registry().lock().expect("registry lock").sidecar.clone() {
        Some(p) if p.exists() => p,
        _ => return 0,
    };
    let text = match std::fs::read_to_string(&sidecar) {
        Ok(t) => t,
        Err(_) => return 0,
    };
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

/// Persist this run's routing frequencies (merged with prior counts) for next-run pinning.
pub fn save_usage() -> Result<()> {
    let (banks, sidecar) = {
        let reg = registry().lock().expect("registry lock");
        (reg.live(), reg.sidecar.clone())
    };
    let Some(path) = sidecar else {
        return Ok(());
    };
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

/// Persist usage on clean exit so the next run can pin. A SIGKILL won't fire it; a normal exit will.
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

/// Drop the just-read file pages so the page cache stays bounded. Best-effort; no alignment needed.
/// Linux-only: Darwin has no `posix_fadvise` (its nearest hint, `F_NOCACHE`, changes fd semantics
/// rather than advising a range), so everywhere else this is a no-op and the page cache self-evicts.
#[cfg(target_os = "linux")]
fn fadvise_dontneed(file: &File, offset: u64, len: u64) {
    use std::os::unix::io::AsRawFd;
    unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn fadvise_dontneed(_file: &File, _offset: u64, _len: u64) {}

/// MemAvailable in bytes: `/proc/meminfo` on Linux, a conservative fallback elsewhere.
fn mem_available_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse::<u64>().ok())
                    {
                        return kb.saturating_mul(1024);
                    }
                }
            }
        }
    }
    static WARNED: AtomicU64 = AtomicU64::new(0);
    if WARNED.swap(1, Ordering::Relaxed) == 0 {
        eprintln!("[stream-experts] MemAvailable unreadable; assuming 8 GB free");
    }
    8_000_000_000
}

#[cfg(test)]
mod tests {
    use super::{tier_decay, tier_pick_swap, REPIN_MAX_SWAPS};

    /// Apply the swap policy to a pinned set exactly as `repin` does (cold slot -> hot expert),
    /// minus the disk I/O, so we can assert on convergence and the swap cap in isolation.
    fn simulate(heat: &[u32], pinned: &mut Vec<u32>, max_swaps: usize) -> usize {
        let mut swaps = 0;
        while swaps < max_swaps {
            match tier_pick_swap(heat, pinned) {
                Some((slot, hot, _gain)) => {
                    pinned[slot] = hot;
                    swaps += 1;
                }
                None => break,
            }
        }
        swaps
    }

    #[test]
    fn swaps_coldest_pinned_for_hottest_streamed() {
        // heat: e0=100, e1=5 (pinned, cold), e2=200 (streamed, hot), e3=50.
        let heat = [100, 5, 200, 50];
        let pinned = [0, 1];
        let (slot, hot, gain) = tier_pick_swap(&heat, &pinned).expect("beneficial swap");
        assert_eq!(slot, 1, "coldest pinned is at slot 1 (eid 1)");
        assert_eq!(hot, 2, "hottest streamed is eid 2");
        assert_eq!(gain, 195, "gain is hot_heat - cold_heat = 200 - 5");
    }

    #[test]
    fn hysteresis_blocks_near_equal_experts() {
        // Cold pinned heat 100 -> threshold 100 + 25 + 4 = 129; a streamed expert at 110 is
        // hotter but inside the margin, so no swap (no ping-pong on noise).
        let heat = [100, 100, 110];
        let pinned = [0, 1];
        assert!(tier_pick_swap(&heat, &pinned).is_none());
    }

    #[test]
    fn hysteresis_margin_is_exclusive_at_the_boundary() {
        // cold_heat 100 -> threshold = 100 + (100>>2=25) + 4 = 129.
        assert!(
            tier_pick_swap(&[100, 100, 129], &[0, 1]).is_none(),
            "equal to the threshold must not swap"
        );
        let (slot, hot, gain) =
            tier_pick_swap(&[100, 100, 130], &[0, 1]).expect("one over the threshold swaps");
        assert_eq!((slot, hot, gain), (0, 2, 30));
    }

    #[test]
    fn no_swap_when_every_expert_is_already_pinned() {
        assert!(tier_pick_swap(&[10, 20], &[0, 1]).is_none());
    }

    #[test]
    fn empty_inputs_are_safe() {
        assert!(tier_pick_swap(&[], &[]).is_none());
        assert!(tier_pick_swap(&[1, 2, 3], &[]).is_none());
        assert!(tier_pick_swap(&[], &[0, 1]).is_none());
    }

    #[test]
    fn picks_the_coldest_slot_and_hottest_candidate_among_many() {
        // pinned e0=50, e2=30 (cold), e4=80; streamed e1=10, e3=200 (hot), e5=5.
        let heat = [50, 10, 30, 200, 80, 5];
        let pinned = [0, 2, 4];
        let (slot, hot, gain) = tier_pick_swap(&heat, &pinned).expect("swap");
        assert_eq!(slot, 1, "eid 2 (heat 30) is the coldest pinned, at slot 1");
        assert_eq!(hot, 3, "eid 3 (heat 200) is the hottest streamed");
        assert_eq!(gain, 170);
    }

    #[test]
    fn decay_halves_every_expert() {
        let mut heat = [10, 3, 0, 255, 1];
        tier_decay(&mut heat);
        assert_eq!(heat, [5, 1, 0, 127, 0]);
    }

    #[test]
    fn repeated_swaps_converge_then_stop_under_hysteresis() {
        // Two genuinely hot streamed experts (100, 90) displace two cold pins (5, 4); once both
        // hot experts are pinned the margin blocks further churn well under the swap cap.
        let heat = [100, 90, 5, 4];
        let mut pinned = vec![2, 3];
        let swaps = simulate(&heat, &mut pinned, REPIN_MAX_SWAPS);
        assert_eq!(swaps, 2, "exactly the two hot experts get pinned");
        pinned.sort_unstable();
        assert_eq!(pinned, vec![0, 1], "hot-set converged to the two hottest experts");
        // A second pass over the same (now settled) heat is a no-op: stable, no ping-pong.
        assert_eq!(simulate(&heat, &mut pinned, REPIN_MAX_SWAPS), 0);
    }

    #[test]
    fn one_dominant_expert_settles_after_a_single_swap() {
        // A lone hot streamed expert takes one cold slot, then hysteresis stops the pass -- the
        // other pinned experts are far enough ahead of the remaining stream to stay put.
        let heat = [50, 10, 30, 200, 80, 5];
        let mut pinned = vec![0, 2, 4];
        assert_eq!(simulate(&heat, &mut pinned, REPIN_MAX_SWAPS), 1);
        pinned.sort_unstable();
        assert_eq!(pinned, vec![0, 3, 4]);
    }
}
