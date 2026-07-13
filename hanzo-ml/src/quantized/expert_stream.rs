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
    /// Per-expert routing frequency this run, persisted for pinning next run.
    usage: Vec<u64>,
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
#[cfg(unix)]
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

#[cfg(not(unix))]
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
