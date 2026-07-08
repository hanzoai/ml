//! Autotune: **one source per op, the schedule picked per device.**
//!
//! The "collapse" program. A backend used to fork a kernel per device to hand-tune the schedule
//! (tile size, vector width, unroll factor, cube dim). That is the "same op, N forks, N drifting
//! numeric behaviors" bug class the DSL exists to kill. The DSL already gives ONE source lowered to
//! every backend; autotune closes the loop on the schedule: expose the schedule as `#[comptime]`
//! knobs on that one source (each knob value is a distinct compiled kernel — cubecl monomorphizes on
//! comptime state), enumerate a small VARIANT set, time each once per `(device, op, shape-class)`, and
//! cache the winner. No forks: the source is one; only the chosen comptime tuple differs per device.
//!
//! # The algebra (three orthogonal values)
//!
//!   * [`Variant`] — a named schedule of an op: a closure that launches it and returns `(output, ms)`.
//!     The closure takes an iteration count so the SAME closure both benchmarks (many iters) and
//!     executes (one iter). This is the crate's established `*_bench` shape (`-> (Vec<f32>, f64)`).
//!   * [`Tuned`] — the ergonomic builder that collects variants for one `(op, shape-key)` and runs.
//!     Reads like `fuse::Fuse` / `dag::Tape`: fluent, no macro, no ceremony.
//!   * [`Tuner`] — the cache: an in-memory `(device, op, key) -> winner` map backed by a per-device
//!     on-disk file under the XDG cache dir. Pure value; takes an explicit cache root, so it is fully
//!     unit-testable without a GPU, without env, and without global state. The process-wide default
//!     ([`global`]) is XDG-rooted.
//!
//! # Why first-party, not `cubecl_runtime::tune`
//!
//! cubecl-runtime HAS an autotuner (`LocalTuner`/`TunableSet`/`Tunable`) and it was studied. It is the
//! wrong fit for three concrete reasons, not NIH: (1) it is re-exported only from `cubecl_runtime`, so
//! adopting it forces a *second* engine crate into a facade that is architected so only `Cargo.toml`
//! names the engine; (2) its persistent cache lives under `target/` (via the `dirs` crate + an MD5
//! checksum + serde), not the XDG path this layer is required to use; (3) its `TuneInputs::At<'a>` GAT +
//! `for<'a> Fn(I::At<'a>)` + `KeyGenerator`/`InputGenerator` traits are a lot of surface for "pick the
//! fastest of N named closures." The concepts below are deliberately the same as cubecl's, just sized
//! to this crate and pointed at XDG — a thin analog, not a new philosophy.
//!
//! # Proven on the CPU runtime (no GPU)
//!
//! [`norm::rms_norm_autotuned`](crate::norm::rms_norm_autotuned) and
//! [`quant::matvec_q8_dp4a_autotuned`](crate::quant::matvec_q8_dp4a_autotuned) each expose one comptime-
//! knobbed source with a 4-variant set. The tests show: every variant is bit-exact vs the CPU oracle,
//! the tuner selects + caches a winner (in-memory AND on-disk), a second call skips the benchmarking of
//! the losers, and a fresh `Tuner` reloads the winner from disk. Timing on the CPU runtime is enough to
//! prove the mechanism; a GPU run additionally makes the *winner* meaningful (on CPU the schedules are
//! near-equivalent, so the selected winner is arbitrary-but-stable — the point under test is the
//! select/cache machinery, not which schedule wins).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::prelude::{ComputeClient, Runtime};

/// One named schedule of an op. `f(iters)` launches the schedule `iters` times (after a fixed warmup)
/// and returns `(output, mean_ms_per_dispatch)` — the crate's `*_bench` contract. Timing uses many
/// iters; producing an output on a cache hit uses one.
pub struct Variant<'a, T> {
    /// Human-readable schedule name (e.g. `"b64_v4"`). Stable — it is the on-disk cache value.
    pub name: &'static str,
    f: Box<dyn Fn(usize) -> (T, f64) + 'a>,
}

impl<'a, T> Variant<'a, T> {
    /// Build a variant from its name and a `|iters| -> (output, ms)` launcher.
    pub fn new(name: &'static str, f: impl Fn(usize) -> (T, f64) + 'a) -> Self {
        Variant { name, f: Box::new(f) }
    }
}

/// The outcome of a tuned run: the winning output plus enough evidence to see the mechanism work.
#[derive(Debug)]
pub struct Pick<T> {
    /// The winning variant's output.
    pub output: T,
    /// The winning variant's name.
    pub winner: String,
    /// `true` if the winner came from the cache (in-memory or disk); `false` if this call tuned.
    pub from_cache: bool,
    /// How many variants were benchmarked on this call: `N` on a miss, `0` on a hit. This is the
    /// observable "second call skips timing".
    pub benched: usize,
    /// `(name, ms)` for every variant benchmarked (empty on a cache hit). Sorted fastest-first.
    pub timings: Vec<(String, f64)>,
}

/// Number of timed iterations per variant during tuning. Warmup is handled inside each `*_bench`.
const TUNE_ITERS: usize = 25;

// ============================================================================================
// The cache.
// ============================================================================================

#[derive(Default)]
struct Inner {
    /// `(device, op, key) -> winner name`.
    winners: HashMap<(String, String, String), String>,
    /// Devices whose on-disk file has already been merged into `winners` (load-once).
    loaded: HashSet<String>,
}

/// The autotune cache: an in-memory winner map backed by a per-device file under `cache_root`.
///
/// A `Tuner` is a plain value — construct one over any directory and it is fully testable offline. The
/// process-wide default ([`global`]) roots at the XDG cache dir. The disk format is one TSV file per
/// device (`<cache_root>/hanzo-kernel/autotune/<device>.tsv`), each line `op\tkey\twinner` — no serde,
/// no new dependency. Cross-process writes append and are idempotent (the loader keeps the last line
/// for a key), matching how an advisory autotune cache should behave.
pub struct Tuner {
    cache_root: PathBuf,
    inner: Mutex<Inner>,
}

impl Tuner {
    /// A tuner rooted at `cache_root` (the parent of `hanzo-kernel/autotune/`).
    pub fn new(cache_root: impl Into<PathBuf>) -> Self {
        Tuner { cache_root: cache_root.into(), inner: Mutex::new(Inner::default()) }
    }

    /// The on-disk file for a device's winners.
    fn device_file(&self, device: &str) -> PathBuf {
        self.cache_root.join("hanzo-kernel").join("autotune").join(format!("{}.tsv", sanitize(device)))
    }

    /// Select (and cache) the fastest variant for `(device, op, key)`.
    ///
    /// Cache hit: launch ONLY the winner once (to produce its output); the losers are never launched.
    /// Cache miss: launch every variant `TUNE_ITERS` times, keep the fastest, persist it, and return
    /// its already-computed output (no redundant re-run of the winner).
    ///
    /// If a cached winner name is no longer among the variants (the set changed), it is treated as a
    /// miss and re-tuned.
    pub fn select<T>(&self, device: &str, op: &str, key: &str, variants: Vec<Variant<'_, T>>) -> Pick<T> {
        assert!(!variants.is_empty(), "tune::select needs at least one variant");
        self.ensure_loaded(device);

        let ck = (device.to_string(), op.to_string(), key.to_string());
        let cached = self.inner.lock().expect("tuner cache poisoned").winners.get(&ck).cloned();

        if let Some(winner) = cached {
            if let Some(v) = variants.iter().find(|v| v.name == winner) {
                let (output, _ms) = (v.f)(1);
                return Pick { output, winner, from_cache: true, benched: 0, timings: Vec::new() };
            }
            // Winner name not in the current set -> the variant set changed; fall through and re-tune.
        }

        // Miss: benchmark every variant once, keep the fastest.
        let mut timings: Vec<(String, f64)> = Vec::with_capacity(variants.len());
        let mut best: Option<(usize, f64)> = None;
        let mut outputs: Vec<Option<T>> = Vec::with_capacity(variants.len());
        for (i, v) in variants.iter().enumerate() {
            let (out, ms) = (v.f)(TUNE_ITERS);
            timings.push((v.name.to_string(), ms));
            outputs.push(Some(out));
            if best.map(|(_, bms)| ms < bms).unwrap_or(true) {
                best = Some((i, ms));
            }
        }
        let (bi, _) = best.expect("at least one variant timed");
        let winner = variants[bi].name.to_string();
        let output = outputs[bi].take().expect("winner output present");

        self.record(device, op, key, &winner);
        timings.sort_by(|a, b| a.1.total_cmp(&b.1));
        Pick { output, winner, from_cache: false, benched: variants.len(), timings }
    }

    /// Merge a device's on-disk winners into memory exactly once.
    fn ensure_loaded(&self, device: &str) {
        {
            let inner = self.inner.lock().expect("tuner cache poisoned");
            if inner.loaded.contains(device) {
                return;
            }
        }
        let entries = read_device_file(&self.device_file(device));
        let mut inner = self.inner.lock().expect("tuner cache poisoned");
        for (op, key, winner) in entries {
            inner.winners.insert((device.to_string(), op, key), winner);
        }
        inner.loaded.insert(device.to_string());
    }

    /// Record a winner in memory and append it to the device's on-disk file.
    fn record(&self, device: &str, op: &str, key: &str, winner: &str) {
        {
            let mut inner = self.inner.lock().expect("tuner cache poisoned");
            inner
                .winners
                .insert((device.to_string(), op.to_string(), key.to_string()), winner.to_string());
        }
        append_device_file(&self.device_file(device), op, key, winner);
    }

    /// Winner currently cached for `(device, op, key)`, if any (after loading disk). For tests/introspection.
    pub fn cached_winner(&self, device: &str, op: &str, key: &str) -> Option<String> {
        self.ensure_loaded(device);
        self.inner
            .lock()
            .expect("tuner cache poisoned")
            .winners
            .get(&(device.to_string(), op.to_string(), key.to_string()))
            .cloned()
    }
}

/// The process-wide default tuner, rooted at the XDG cache dir. Used by [`Tuned::run`].
pub fn global() -> &'static Tuner {
    static G: OnceLock<Tuner> = OnceLock::new();
    G.get_or_init(|| Tuner::new(xdg_cache_dir()))
}

/// `$XDG_CACHE_HOME`, else `$HOME/.cache`, else the current dir. No new env var is introduced — only the
/// standard XDG one is read.
pub fn xdg_cache_dir() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
        if !x.is_empty() {
            return PathBuf::from(x);
        }
    }
    if let Some(h) = std::env::var_os("HOME") {
        if !h.is_empty() {
            return PathBuf::from(h).join(".cache");
        }
    }
    PathBuf::from(".")
}

/// Backend identifier for `R` (the last `::` segment of its type name, e.g. `"CpuRuntime"`). This is the
/// device granularity that matters for schedule selection — a winner is valid per backend. A finer
/// physical-GPU key can be appended from `client` device properties without changing this API.
pub fn device_id<R: Runtime>(_client: &ComputeClient<R>) -> String {
    let name = std::any::type_name::<R>();
    name.rsplit("::").next().unwrap_or(name).to_string()
}

// --- disk helpers (TSV, std-only) -----------------------------------------------------------

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

/// A TSV cell may not contain a tab or newline; collapse any to a space so a line always round-trips.
fn cell(s: &str) -> String {
    s.chars().map(|c| if c == '\t' || c == '\n' || c == '\r' { ' ' } else { c }).collect()
}

fn read_device_file(path: &Path) -> Vec<(String, String, String)> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    // Last line for a given (op,key) wins; a HashMap over (op,key) dedups append history.
    let mut map: HashMap<(String, String), String> = HashMap::new();
    for line in text.lines() {
        let mut it = line.splitn(3, '\t');
        if let (Some(op), Some(key), Some(winner)) = (it.next(), it.next(), it.next()) {
            map.insert((op.to_string(), key.to_string()), winner.to_string());
        }
    }
    map.into_iter().map(|((op, key), winner)| (op, key, winner)).collect()
}

fn append_device_file(path: &Path, op: &str, key: &str, winner: &str) {
    use std::io::Write;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}\t{}\t{}", cell(op), cell(key), cell(winner));
    }
}

// ============================================================================================
// The ergonomic builder.
// ============================================================================================

/// Fluent builder for a tuned op: collect variants for one `(op, shape-key)`, then `run`.
///
/// ```ignore
/// let y = Tuned::new("rms_norm", format!("rows={rows},n={n}"))
///     .variant("b64_r1",  |it| rms_norm_tuned_bench(&c, x, w, rows, n, eps, 1, 64,  it))
///     .variant("b128_r1", |it| rms_norm_tuned_bench(&c, x, w, rows, n, eps, 1, 128, it))
///     .variant("b64_r2",  |it| rms_norm_tuned_bench(&c, x, w, rows, n, eps, 2, 64,  it))
///     .variant("b256_r1", |it| rms_norm_tuned_bench(&c, x, w, rows, n, eps, 1, 256, it))
///     .run(&c);   // times each once per (device, op, shape), caches the winner, thereafter runs it
/// ```
pub struct Tuned<'a, T> {
    op: &'static str,
    key: String,
    variants: Vec<Variant<'a, T>>,
}

impl<'a, T> Tuned<'a, T> {
    /// Start a variant set for op `op` at shape-class `key`.
    pub fn new(op: &'static str, key: impl Into<String>) -> Self {
        Tuned { op, key: key.into(), variants: Vec::new() }
    }

    /// Add a named schedule variant. `f(iters) -> (output, ms)` is the crate's `*_bench` shape.
    pub fn variant(mut self, name: &'static str, f: impl Fn(usize) -> (T, f64) + 'a) -> Self {
        self.variants.push(Variant::new(name, f));
        self
    }

    /// Tune (or read the cache) on an explicit tuner + device, returning the full [`Pick`]. This is the
    /// offline-testable core: no client, no globals.
    pub fn pick_with(self, tuner: &Tuner, device: &str) -> Pick<T> {
        tuner.select(device, self.op, &self.key, self.variants)
    }

    /// Tune (or read the cache) via the process-wide XDG tuner, keyed by `client`'s backend. Returns
    /// the full [`Pick`] (winner + timings + cache flags).
    pub fn pick<R: Runtime>(self, client: &ComputeClient<R>) -> Pick<T> {
        let device = device_id::<R>(client);
        self.pick_with(global(), &device)
    }

    /// Tune (or read the cache) via the process-wide XDG tuner and return just the winning output — the
    /// production surface. One source per op, autotuned per device, no forks.
    pub fn run<R: Runtime>(self, client: &ComputeClient<R>) -> T {
        self.pick(client).output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core select/cache mechanism, with pure closures (no GPU): a variant "produces" a constant
    /// output and reports a fixed ms, so the fastest is deterministic. Proves: miss tunes all, hit skips
    /// timing, a fresh Tuner reloads the winner from disk, and a changed variant set re-tunes.
    #[test]
    fn select_caches_and_reloads_from_disk() {
        let dir = tmp_dir("select");
        let variants = || {
            vec![
                Variant::new("slow", |_it| (10u32, 9.0)),
                Variant::new("fast", |_it| (20u32, 1.0)), // fastest
                Variant::new("mid", |_it| (30u32, 5.0)),
            ]
        };

        // First tuner: miss -> benchmarks all 3, picks "fast".
        let t1 = Tuner::new(&dir);
        let p = t1.select("cpu", "op", "s1", variants());
        assert!(!p.from_cache);
        assert_eq!(p.benched, 3);
        assert_eq!(p.winner, "fast");
        assert_eq!(p.output, 20);
        assert_eq!(p.timings.first().unwrap().0, "fast"); // sorted fastest-first

        // Same tuner again: in-memory hit -> benchmarks nothing, still returns the winner's output.
        let p = t1.select("cpu", "op", "s1", variants());
        assert!(p.from_cache);
        assert_eq!(p.benched, 0);
        assert_eq!(p.winner, "fast");
        assert_eq!(p.output, 20);

        // Fresh tuner, same dir: cold in-memory cache -> must reload the winner from disk, skip timing.
        let t2 = Tuner::new(&dir);
        assert_eq!(t2.cached_winner("cpu", "op", "s1").as_deref(), Some("fast"));
        let p = t2.select("cpu", "op", "s1", variants());
        assert!(p.from_cache);
        assert_eq!(p.benched, 0);
        assert_eq!(p.winner, "fast");

        // A different shape-class key is an independent decision (still a miss).
        let p = t2.select("cpu", "op", "s2", variants());
        assert!(!p.from_cache);
        assert_eq!(p.benched, 3);

        // A changed variant set (winner "fast" absent) re-tunes rather than trusting a stale name.
        let shrunk = vec![Variant::new("slow", |_it| (10u32, 9.0)), Variant::new("mid", |_it| (30u32, 2.0))];
        let p = t2.select("cpu", "op", "s1", shrunk);
        assert!(!p.from_cache);
        assert_eq!(p.winner, "mid");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The disk format round-trips and dedups append history (last write for a key wins).
    #[test]
    fn disk_format_round_trips() {
        let dir = tmp_dir("disk");
        let t = Tuner::new(&dir);
        let file = t.device_file("CpuRuntime");
        append_device_file(&file, "rms_norm", "rows=8,n=4", "b64_r1");
        append_device_file(&file, "matvec", "rows=8,k=64", "b128_v2");
        append_device_file(&file, "rms_norm", "rows=8,n=4", "b256_r1"); // supersedes the first line

        let mut got = read_device_file(&file);
        got.sort();
        assert_eq!(
            got,
            vec![
                ("matvec".to_string(), "rows=8,k=64".to_string(), "b128_v2".to_string()),
                ("rms_norm".to_string(), "rows=8,n=4".to_string(), "b256_r1".to_string()),
            ]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The device file path is `<root>/hanzo-kernel/autotune/<sanitized-device>.tsv`.
    #[test]
    fn device_file_path_is_xdg_shaped() {
        let t = Tuner::new("/x/y");
        assert_eq!(t.device_file("Cuda::Runtime").to_str().unwrap(), "/x/y/hanzo-kernel/autotune/Cuda__Runtime.tsv");
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "hanzo-kernel-tune-{tag}-{}-{}",
            std::process::id(),
            // a per-call nonce so repeated runs don't collide
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
