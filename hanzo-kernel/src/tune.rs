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

// ============================================================================================
// Evolutionary search over a Space of comptime schedules.
// ============================================================================================
//
// [`Tuned`] enumerates a hand-listed variant set and times every one. That is right when the set is a
// handful. A real kernel's schedule is a PRODUCT of knobs (tile rows x cols x warps x LDS pad x ...),
// tens to hundreds of feasible points, each a distinct compiled kernel. Timing all of them on the GPU
// is the cost we cannot pay. So this layer SEARCHES the product instead of enumerating it: an
// evolutionary hunt over a [`Space`], with a MULTI-FIDELITY [`Evaluator`] that spends a free static
// tier (compile + shader-stat read) to reject infeasible schedules before it spends the expensive GPU
// tier (a sustained on-device timing) on the survivors.
//
// It is the same algebra as [`Tuned`], one level up: a [`Config`] is a point in the space; its
// canonical [`Config::name`] is exactly a [`Tuner`] cache value, so a hunt's winner persists and
// replays through the same TSV cache ([`Tuner::evolve`]). No global state, no `rand`, no `Date::now` --
// a seeded [`Rng`] makes a hunt reproducible, which is the difference between a tuner and a slot machine.
//
// # Negative priors
//
// The refutation log is knowledge: a knob value a prior hunt already proved bad is not worth a GPU
// dispatch to re-disprove. A [`Space`] carries a deny-list (inspectable data: partial assignments) so
// those subspaces are never proposed. This is a search-domain prior, distinct from a hard feasibility
// [constraint](Space::constraint) (which encodes what the hardware/compiler can express at all).

use std::hash::Hash;

/// Seeded xorshift64* PRNG. Deterministic and self-contained -- a hunt keyed by a seed reproduces
/// exactly, which the unit tests depend on and a slot-machine `rand`/`Date::now` source would destroy.
/// Not cryptographic; a schedule search does not need it to be.
pub struct Rng(u64);

impl Rng {
    /// Seed the generator. The state is forced nonzero (xorshift's one fixed point).
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E3779B97F4A7C15 | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform integer in `[0, n)`. Requires `n > 0`.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    /// A coin that comes up `true` with probability `p` in `[0, 1]`.
    pub fn chance(&mut self, p: f64) -> bool {
        // 53-bit uniform in [0,1); < p is a fair Bernoulli(p).
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64) < p
    }
}

/// One search parameter: a name and its allowed discrete values (the alphabet a gene mutates over).
pub struct Param {
    /// Parameter name, e.g. `"NWARP"`. Appears in [`Config::name`] and the deny-list.
    pub name: &'static str,
    /// The finite set of legal values (the gene alphabet).
    pub values: Vec<i64>,
}

/// A point in the [`Space`]: one chosen value per parameter, in Space order. This is the GA genome.
/// `Eq`/`Hash` make it the key of the evaluated-set (dedup) and the memo (never re-measure a point).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    vals: Vec<i64>,
}

impl Config {
    /// The value chosen for `name` (panics if `name` is not a parameter of `space` -- a program bug).
    pub fn get(&self, space: &Space, name: &str) -> i64 {
        let i = space.index(name).unwrap_or_else(|| panic!("no param `{name}` in space"));
        self.vals[i]
    }
    /// Canonical name `"P1=v1,P2=v2,..."` in Space order. Stable -- it is the [`Tuner`] cache value and
    /// round-trips through [`Space::parse`].
    pub fn name(&self, space: &Space) -> String {
        space
            .params
            .iter()
            .zip(&self.vals)
            .map(|(p, v)| format!("{}={}", p.name, v))
            .collect::<Vec<_>>()
            .join(",")
    }
}

type Constraint = Box<dyn Fn(&Config, &Space) -> bool>;

/// The search domain: named parameters, hard feasibility constraints, and negative priors (deny-list).
///
/// A [`Config`] is *admissible* iff every value is in its parameter's domain, no deny entry matches, and
/// every constraint holds. The GA only ever holds admissible configs -- constraints and priors prune the
/// domain up front, for free, before any compile or dispatch. Built fluently, no macro:
///
/// ```ignore
/// let space = Space::new()
///     .param("NWARP", [2, 4, 8])
///     .param("RM",    [1, 2, 4])
///     .param("RN",    [8, 16])
///     .constraint(|c, s| c.get(s, "NWARP") * c.get(s, "RM") * 16 == 128) // BM == 128
///     .deny([("RN", 4)])        // BN=64 coopmat -- refuted
///     .deny([("double_buffer", 1)]); // 2x LDS caps occupancy -- refuted
/// ```
pub struct Space {
    params: Vec<Param>,
    constraints: Vec<Constraint>,
    /// Each entry is a partial assignment; a config matching ALL of an entry's `(name, value)` pairs is
    /// denied. Pure data so the priors are inspectable and unit-testable, and read like the refutation log.
    denies: Vec<Vec<(&'static str, i64)>>,
}

impl Default for Space {
    fn default() -> Self {
        Space { params: Vec::new(), constraints: Vec::new(), denies: Vec::new() }
    }
}

impl Space {
    /// An empty space. Add parameters, constraints and priors fluently.
    pub fn new() -> Self {
        Space::default()
    }

    /// Add a parameter and its discrete value set.
    pub fn param(mut self, name: &'static str, values: impl IntoIterator<Item = i64>) -> Self {
        self.params.push(Param { name, values: values.into_iter().collect() });
        self
    }

    /// Add a hard feasibility constraint (relational or over derived quantities, e.g. `BM == 128`). A
    /// config is infeasible if any constraint returns `false`.
    pub fn constraint(mut self, f: impl Fn(&Config, &Space) -> bool + 'static) -> Self {
        self.constraints.push(Box::new(f));
        self
    }

    /// Add a negative prior: a partial assignment whose full match denies a config. `deny([("RN", 4)])`
    /// removes the whole `RN=4` subspace; `deny([("a", 1), ("b", 2)])` removes only their intersection.
    pub fn deny(mut self, subspace: impl IntoIterator<Item = (&'static str, i64)>) -> Self {
        self.denies.push(subspace.into_iter().collect());
        self
    }

    /// Number of parameters.
    pub fn arity(&self) -> usize {
        self.params.len()
    }

    fn index(&self, name: &str) -> Option<usize> {
        self.params.iter().position(|p| p.name == name)
    }

    /// Build a config from a full set of named assignments (every parameter must appear exactly once).
    /// A test/seed helper; panics on a missing or unknown name so a typo fails loudly.
    pub fn config(&self, assignments: &[(&str, i64)]) -> Config {
        let mut vals = vec![i64::MIN; self.params.len()];
        let mut set = vec![false; self.params.len()];
        for (name, v) in assignments {
            let i = self.index(name).unwrap_or_else(|| panic!("no param `{name}`"));
            vals[i] = *v;
            set[i] = true;
        }
        assert!(set.iter().all(|&b| b), "config() must assign every parameter");
        Config { vals }
    }

    /// Parse a [`Config::name`] back into a config, validating names and value domains. `None` if the
    /// string names an unknown parameter, omits one, or carries an out-of-domain value (a stale cache).
    pub fn parse(&self, name: &str) -> Option<Config> {
        let mut vals = vec![i64::MIN; self.params.len()];
        let mut set = vec![false; self.params.len()];
        for tok in name.split(',') {
            let (k, v) = tok.split_once('=')?;
            let i = self.index(k)?;
            let v: i64 = v.parse().ok()?;
            if !self.params[i].values.contains(&v) {
                return None;
            }
            vals[i] = v;
            set[i] = true;
        }
        if !set.iter().all(|&b| b) {
            return None;
        }
        Some(Config { vals })
    }

    /// Is `c` denied by a negative prior?
    pub fn denied(&self, c: &Config) -> bool {
        self.denies.iter().any(|entry| {
            entry.iter().all(|(name, val)| self.index(name).is_some_and(|i| c.vals[i] == *val))
        })
    }

    /// Is `c` admissible? In-domain values, not denied, all constraints satisfied.
    pub fn feasible(&self, c: &Config) -> bool {
        if c.vals.len() != self.params.len() {
            return false;
        }
        if !self.params.iter().zip(&c.vals).all(|(p, v)| p.values.contains(v)) {
            return false;
        }
        if self.denied(c) {
            return false;
        }
        self.constraints.iter().all(|f| f(c, self))
    }

    /// Every admissible config, by cartesian product then feasibility filter. For a SMALL space this is
    /// the exhaustive baseline a hunt is judged against (and lets a tiny genome be compile-swept offline);
    /// it is exponential in arity, so it is a diagnostic, not the search itself.
    pub fn enumerate(&self) -> Vec<Config> {
        let mut out = vec![Config { vals: Vec::with_capacity(self.params.len()) }];
        for p in &self.params {
            let mut next = Vec::with_capacity(out.len() * p.values.len());
            for base in &out {
                for &v in &p.values {
                    let mut vals = base.vals.clone();
                    vals.push(v);
                    next.push(Config { vals });
                }
            }
            out = next;
        }
        out.retain(|c| self.feasible(c));
        out
    }

    /// Draw one uniformly-random config and accept it only if admissible; up to `tries` attempts, else
    /// `None` (a heavily-constrained space). Deterministic in `rng`.
    pub fn random(&self, rng: &mut Rng, tries: usize) -> Option<Config> {
        for _ in 0..tries {
            let vals = self.params.iter().map(|p| p.values[rng.below(p.values.len())]).collect();
            let c = Config { vals };
            if self.feasible(&c) {
                return Some(c);
            }
        }
        None
    }
}

/// The verdict of the free static tier: admit this config to GPU timing, or reject it (with a reason)
/// so the search spends no dispatch on it and never proposes it again.
#[derive(Clone, Debug)]
pub enum Verdict {
    /// Admit to the GPU tier.
    Pass,
    /// Reject before any dispatch; the string is the human-readable cause (compile error, register
    /// spill, LDS over the occupancy budget).
    Reject(String),
}

/// A two-tier (multi-fidelity) fitness function over a [`Space`].
///
/// `static_check` is the FREE tier: for a GPU kernel it compiles the variant and reads its shader stats
/// (VGPR / LDS / scratch), rejecting a schedule that cannot fit the occupancy budget before it costs a
/// dispatch. `measure` is the EXPENSIVE tier: it runs the survivor on the device and returns SUSTAINED
/// milliseconds (lower is better) -- warmed past the boost window, because a burst number is a lie for a
/// throttling part. The search only ever `measure`s a config that `static_check` passed.
///
/// # `measure` MUST hold the deployment regime, not a warm microbench
///
/// `measure` is the oracle the whole search trusts; if it measures the wrong regime every winner is a
/// lie, and a lying oracle is worse than no tuner. The failure mode is concrete and recorded: a single-
/// tile microbench reuses one weight, so after the first iteration that weight is served from cache and
/// the kernel runs at its isolated COMPUTE ceiling -- where an occupancy knob (more subgroups) looks like
/// a win. The deployment never sees that ceiling: decode streams every weight cache-cold from GTT
/// (bandwidth-bound) and prefill drains a barrier between dispatches (utilization-slaved DVFS downclock),
/// and BOTH impose an occupancy-INDEPENDENT floor the extra occupancy cannot lift (it changes occupancy,
/// not bytes moved or dispatch count). So `measure` MUST hold the config in its deployment memory regime
/// -- stream the weights cache-cold from a working set past the last-level cache, evicted between timed
/// iterations, not a cache-resident single tile. A warm oracle once scored an over-occupied coopmat
/// schedule +24% that measured flat in-engine; the golden gate `cold_oracle_scores_nwarp8_flat` is the
/// committed regression on exactly this, and `hanzo-ml`'s `CoopmatEval` is the on-device realization
/// (a weight bank sized past the MALL, streamed pipelined so the GTT link stays saturated).
pub trait Evaluator {
    /// Free tier. Called first for every distinct config.
    fn static_check(&self, cfg: &Config) -> Verdict;
    /// Expensive tier. Called only on configs that passed `static_check`. Returns sustained ms measured
    /// in the config's DEPLOYMENT memory regime (cold-weight-streamed past the cache), never a cache-warm
    /// single tile -- see the trait note; a warm number crowns occupancy wins the deployment never sees.
    fn measure(&self, cfg: &Config, iters: usize) -> f64;
}

/// One config's evaluated fitness, memoized so it is computed at most once.
#[derive(Clone)]
struct Cell {
    /// Comparable fitness: the sustained ms, or `+inf` for a statically-rejected config.
    fitness: f64,
    /// `Some(ms)` if it reached the GPU tier.
    ms: Option<f64>,
    /// `Some(reason)` if the static tier rejected it.
    reason: Option<String>,
}

/// The outcome of a hunt: the winner plus the full evidence trail.
#[derive(Debug)]
pub struct EvoReport {
    /// The winning config (lowest sustained ms). If every config was statically rejected, this is the
    /// first config evaluated and `best_ms` is `+inf`.
    pub best: Config,
    /// The winner's canonical name -- exactly a [`Tuner`] cache value.
    pub best_name: String,
    /// The winner's sustained ms (`+inf` if nothing was measured).
    pub best_ms: f64,
    /// Every measured config as `(name, ms)`, fastest-first. Length = GPU-tier evaluations.
    pub measured: Vec<(String, f64)>,
    /// Every statically-rejected config as `(name, reason)`. Length = configs pruned for free.
    pub rejected: Vec<(String, String)>,
    /// Distinct configs that reached fitness (measured + rejected) -- the search's total footprint.
    pub evaluated: usize,
    /// Generations actually run.
    pub generations: usize,
}

/// An evolutionary hunt over a [`Space`]: population, tournament selection, uniform crossover, point
/// mutation, elitism, and an evaluated-set that dedups so no point is ever measured twice. Fluent, no
/// macro -- reads like [`Tuned`]. Deterministic given a seed.
pub struct Evolution {
    pop: usize,
    generations: usize,
    tournament: usize,
    mutation: f64,
    elitism: usize,
    measure_iters: usize,
    seeds: Vec<Config>,
}

impl Default for Evolution {
    fn default() -> Self {
        Evolution {
            pop: 16,
            generations: 8,
            tournament: 3,
            mutation: 0.25,
            elitism: 2,
            measure_iters: TUNE_ITERS,
            seeds: Vec::new(),
        }
    }
}

impl Evolution {
    /// A hunt with sensible defaults (pop 16, 8 generations, tournament 3, mutation 0.25, elitism 2).
    pub fn new() -> Self {
        Evolution::default()
    }
    /// Population size (points held per generation).
    pub fn population(mut self, n: usize) -> Self {
        self.pop = n.max(1);
        self
    }
    /// Number of generations to run.
    pub fn generations(mut self, n: usize) -> Self {
        self.generations = n;
        self
    }
    /// Tournament size for parent selection (higher = greedier).
    pub fn tournament(mut self, k: usize) -> Self {
        self.tournament = k.max(1);
        self
    }
    /// Per-gene mutation probability in `[0, 1]`.
    pub fn mutation(mut self, p: f64) -> Self {
        self.mutation = p.clamp(0.0, 1.0);
        self
    }
    /// How many best configs survive unchanged into the next generation.
    pub fn elitism(mut self, k: usize) -> Self {
        self.elitism = k;
        self
    }
    /// Iteration count handed to [`Evaluator::measure`].
    pub fn measure_iters(mut self, n: usize) -> Self {
        self.measure_iters = n;
        self
    }
    /// Inject a config into the initial population (e.g. the shipped incumbent, so a hunt always has the
    /// A/B baseline in-frame). Infeasible seeds are dropped.
    pub fn seed_config(mut self, c: Config) -> Self {
        self.seeds.push(c);
        self
    }

    /// Compare two evaluated configs by fitness, breaking ties by name so the ordering (and therefore
    /// the winner and the whole trajectory) is deterministic.
    fn cmp_fit(space: &Space, memo: &HashMap<Config, Cell>, a: &Config, b: &Config) -> std::cmp::Ordering {
        let fa = memo[a].fitness;
        let fb = memo[b].fitness;
        fa.total_cmp(&fb).then_with(|| a.name(space).cmp(&b.name(space)))
    }

    /// Evaluate one config through the two tiers, memoized. Records first-seen order for a deterministic
    /// report. Returns the comparable fitness.
    fn evaluate<E: Evaluator>(
        &self,
        eval: &E,
        c: &Config,
        memo: &mut HashMap<Config, Cell>,
        order: &mut Vec<Config>,
    ) -> f64 {
        if let Some(cell) = memo.get(c) {
            return cell.fitness;
        }
        let cell = match eval.static_check(c) {
            Verdict::Reject(reason) => Cell { fitness: f64::INFINITY, ms: None, reason: Some(reason) },
            Verdict::Pass => {
                let ms = eval.measure(c, self.measure_iters);
                Cell { fitness: ms, ms: Some(ms), reason: None }
            }
        };
        let f = cell.fitness;
        memo.insert(c.clone(), cell);
        order.push(c.clone());
        f
    }

    /// Tournament selection: sample `tournament` population members and return the fittest.
    fn select<'p>(&self, space: &Space, memo: &HashMap<Config, Cell>, pop: &'p [Config], rng: &mut Rng) -> &'p Config {
        let mut best = &pop[rng.below(pop.len())];
        for _ in 1..self.tournament {
            let c = &pop[rng.below(pop.len())];
            if Self::cmp_fit(space, memo, c, best).is_lt() {
                best = c;
            }
        }
        best
    }

    /// Uniform crossover: each gene comes from parent `a` or `b` by a fair coin.
    fn crossover(&self, a: &Config, b: &Config, rng: &mut Rng) -> Config {
        let vals = a
            .vals
            .iter()
            .zip(&b.vals)
            .map(|(&va, &vb)| if rng.chance(0.5) { va } else { vb })
            .collect();
        Config { vals }
    }

    /// Point mutation: each gene, with probability `mutation`, jumps to a different value in its domain.
    fn mutate(&self, space: &Space, c: &mut Config, rng: &mut Rng) {
        for (i, p) in space.params.iter().enumerate() {
            if p.values.len() > 1 && rng.chance(self.mutation) {
                let cur = c.vals[i];
                // Draw uniformly from the other values (index skip keeps it a true "jump").
                let mut j = rng.below(p.values.len() - 1);
                if p.values[j] == cur {
                    j = p.values.len() - 1;
                }
                c.vals[i] = p.values[j];
            }
        }
    }

    /// Bring a bred child back into the feasible set: point-mutate up to a cap, else fall back to a fresh
    /// random feasible draw. Guarantees the population is always admissible.
    fn repair(&self, space: &Space, mut c: Config, rng: &mut Rng) -> Option<Config> {
        for _ in 0..64 {
            if space.feasible(&c) {
                return Some(c);
            }
            self.mutate(space, &mut c, rng);
        }
        space.random(rng, 1024)
    }

    /// Run the hunt. Deterministic in `seed`. Calls the evaluator's free tier on every distinct config
    /// and its GPU tier only on the survivors.
    pub fn hunt<E: Evaluator>(&self, space: &Space, eval: &E, seed: u64) -> EvoReport {
        let mut rng = Rng::new(seed);
        let mut memo: HashMap<Config, Cell> = HashMap::new();
        let mut order: Vec<Config> = Vec::new();

        // Generation 0: feasible seeds first (incumbent A/B baseline), then random fill, deduped.
        let mut population: Vec<Config> = Vec::new();
        let mut seen: HashSet<Config> = HashSet::new();
        for s in &self.seeds {
            if space.feasible(s) && seen.insert(s.clone()) {
                population.push(s.clone());
            }
        }
        let mut fill_tries = 0usize;
        while population.len() < self.pop && fill_tries < self.pop * 128 {
            fill_tries += 1;
            match space.random(&mut rng, 256) {
                Some(c) if seen.insert(c.clone()) => population.push(c),
                Some(_) => {} // duplicate, retry
                None => break, // domain infeasible/exhausted
            }
        }
        if population.is_empty() {
            // Nothing feasible at all: report an empty hunt rather than panic.
            return EvoReport {
                best: Config { vals: vec![] },
                best_name: "<none>".to_string(),
                best_ms: f64::INFINITY,
                measured: Vec::new(),
                rejected: Vec::new(),
                evaluated: 0,
                generations: 0,
            };
        }
        for c in &population.clone() {
            self.evaluate(eval, c, &mut memo, &mut order);
        }

        let mut gens_run = 0usize;
        for _ in 0..self.generations {
            gens_run += 1;
            // Rank the current population; elites survive unchanged.
            let mut ranked = population.clone();
            ranked.sort_by(|a, b| Self::cmp_fit(space, &memo, a, b));

            let mut next: Vec<Config> = Vec::new();
            let mut nseen: HashSet<Config> = HashSet::new();
            for e in ranked.iter().take(self.elitism) {
                if nseen.insert(e.clone()) {
                    next.push(e.clone());
                }
            }
            let mut tries = 0usize;
            while next.len() < self.pop && tries < self.pop * 128 {
                tries += 1;
                let a = self.select(space, &memo, &population, &mut rng).clone();
                let b = self.select(space, &memo, &population, &mut rng).clone();
                let mut child = self.crossover(&a, &b, &mut rng);
                self.mutate(space, &mut child, &mut rng);
                let child = match self.repair(space, child, &mut rng) {
                    Some(c) => c,
                    None => continue,
                };
                if nseen.insert(child.clone()) {
                    next.push(child);
                }
            }
            // A small/constrained domain may under-fill; that is fine, the population just shrinks.
            if next.is_empty() {
                break;
            }
            for c in &next.clone() {
                self.evaluate(eval, c, &mut memo, &mut order);
            }
            population = next;
        }

        // Build the report from the deterministic first-seen order.
        let mut measured: Vec<(String, f64)> = order
            .iter()
            .filter_map(|c| memo[c].ms.map(|ms| (c.name(space), ms)))
            .collect();
        measured.sort_by(|x, y| x.1.total_cmp(&y.1).then_with(|| x.0.cmp(&y.0)));
        let rejected: Vec<(String, String)> = order
            .iter()
            .filter_map(|c| memo[c].reason.clone().map(|r| (c.name(space), r)))
            .collect();

        let best = order
            .iter()
            .filter(|c| memo[*c].ms.is_some())
            .min_by(|a, b| Self::cmp_fit(space, &memo, a, b))
            .cloned()
            .unwrap_or_else(|| order[0].clone());
        let best_ms = memo[&best].ms.unwrap_or(f64::INFINITY);
        let best_name = best.name(space);

        EvoReport {
            best,
            best_name,
            best_ms,
            measured,
            rejected,
            evaluated: order.len(),
            generations: gens_run,
        }
    }
}

/// The outcome of [`Tuner::evolve`]: the winning config name (which round-trips through [`Space::parse`])
/// and, when a hunt actually ran, its full [`EvoReport`].
pub struct Evolved {
    /// Winning config's canonical name (from cache or from the hunt).
    pub winner: String,
    /// `true` if the winner was read from the persistent cache and no hunt ran.
    pub from_cache: bool,
    /// The hunt's evidence trail, or `None` on a cache hit.
    pub report: Option<EvoReport>,
}

impl Tuner {
    /// Evolve `space` on `eval` for `(device, op, key)` -- but skip the whole hunt if a winner is already
    /// cached (parse it and return). Mirrors [`Tuner::select`] for the evolutionary strategy: same TSV
    /// cache, keyed by the winner's canonical [`Config::name`]. On a miss it hunts, records the winner,
    /// and returns the report.
    pub fn evolve<E: Evaluator>(
        &self,
        device: &str,
        op: &str,
        key: &str,
        space: &Space,
        eval: &E,
        evo: &Evolution,
        seed: u64,
    ) -> Evolved {
        if let Some(w) = self.cached_winner(device, op, key) {
            if space.parse(&w).is_some() {
                return Evolved { winner: w, from_cache: true, report: None };
            }
            // Stale cache (space changed): fall through and re-hunt.
        }
        let report = evo.hunt(space, eval, seed);
        self.record(device, op, key, &report.best_name);
        Evolved { winner: report.best_name.clone(), from_cache: false, report: Some(report) }
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

    // --- evolutionary search ------------------------------------------------------------------

    /// A three-parameter grid (10^3 = 1000 points) with a narrow deep global well and a WIDE shallow
    /// local well -- the classic deception: the local basin has the larger attraction area, so greedy
    /// search settles there, and only real exploration finds the narrow global optimum. `measure` is the
    /// landscape value (lower is better); the evaluator holds the space so it reads genes by name.
    struct Deceptive<'a> {
        space: &'a Space,
    }
    impl<'a> Evaluator for Deceptive<'a> {
        fn static_check(&self, _c: &Config) -> Verdict {
            Verdict::Pass
        }
        fn measure(&self, c: &Config, _iters: usize) -> f64 {
            let a = c.get(self.space, "a") as f64;
            let b = c.get(self.space, "b") as f64;
            let cc = c.get(self.space, "c") as f64;
            // global (7,2,5): depth 10, width 3.5 (narrow). local (2,7,4): depth 6, width 18 (wide).
            let g = ((a - 7.0).powi(2) + (b - 2.0).powi(2) + (cc - 5.0).powi(2)) / 3.5;
            let l = ((a - 2.0).powi(2) + (b - 7.0).powi(2) + (cc - 4.0).powi(2)) / 18.0;
            10.0 - 10.0 * (-g).exp() - 6.0 * (-l).exp()
        }
    }

    fn grid3() -> Space {
        Space::new().param("a", 0..=9).param("b", 0..=9).param("c", 0..=9)
    }

    /// Brute-force argmin over the grid subject to a predicate (the oracle the GA must match).
    fn brute_argmin(space: &Space, eval: &Deceptive, keep: impl Fn(&Config) -> bool) -> (Config, f64) {
        let mut best: Option<(Config, f64)> = None;
        for a in 0..=9 {
            for b in 0..=9 {
                for c in 0..=9 {
                    let cfg = space.config(&[("a", a), ("b", b), ("c", c)]);
                    if !keep(&cfg) {
                        continue;
                    }
                    let ms = eval.measure(&cfg, 1);
                    // tie-break by name to match cmp_fit exactly.
                    let better = match &best {
                        None => true,
                        Some((bc, bms)) => {
                            ms < *bms || (ms == *bms && cfg.name(space) < bc.name(space))
                        }
                    };
                    if better {
                        best = Some((cfg, ms));
                    }
                }
            }
        }
        best.unwrap()
    }

    #[test]
    fn evo_converges_on_deceptive_landscape_and_is_deterministic() {
        let space = grid3();
        let eval = Deceptive { space: &space };
        let (opt, opt_ms) = brute_argmin(&space, &eval, |_| true);
        assert_eq!(opt.name(&space), "a=7,b=2,c=5"); // the narrow global well
        let local_ms = eval.measure(&space.config(&[("a", 2), ("b", 7), ("c", 4)]), 1); // the deceptive trap

        let evo = Evolution::new().population(28).generations(22).tournament(3).mutation(0.35).elitism(2);

        // Converges to the GLOBAL optimum (exact ms), escaping the deceptive local well...
        let r = evo.hunt(&space, &eval, 0xC0FFEE);
        assert_eq!(r.best_name, "a=7,b=2,c=5", "GA trapped in the local well; best={}", r.best_name);
        assert!((r.best_ms - opt_ms).abs() < 1e-9);
        assert!(r.best_ms < local_ms - 1.0, "did not escape the local trap");
        // ...as a real search, not an enumeration of all 1000 points.
        assert!(r.evaluated < 700, "evaluated {} of 1000 -- not searching", r.evaluated);
        assert_eq!(r.rejected.len(), 0); // this landscape rejects nothing

        // Deterministic under seed: same seed => identical winner, footprint, and full timing trail.
        let r2 = evo.hunt(&space, &eval, 0xC0FFEE);
        assert_eq!(r.best_name, r2.best_name);
        assert_eq!(r.evaluated, r2.evaluated);
        assert_eq!(r.measured, r2.measured);

        // Systematic (not lucky) escape: across many seeds the hunt reaches the narrow global well the
        // large majority of the time. A deceptive landscape with a finite budget can still trap a
        // minority of seeds -- which is exactly why the production hunt multi-seeds and keeps the best;
        // no single run is trusted. We assert the rate, not every seed.
        let seeds: [u64; 10] = [1, 42, 7777, 0xABCDEF, 0xC0FFEE, 2, 3, 99, 12345, 0xDEADBEEF];
        let hits = seeds.iter().filter(|&&s| evo.hunt(&space, &eval, s).best_name == "a=7,b=2,c=5").count();
        assert!(hits >= 8, "only {hits}/10 seeds reached the global optimum -- escape not systematic");
    }

    #[test]
    fn evo_respects_the_deny_list() {
        // Deny the whole a=7 plane -- which contains the global optimum. The hunt must never propose it
        // and must settle on the best admissible config instead.
        let space = grid3().deny([("a", 7)]);
        let eval = Deceptive { space: &space };

        // Feasibility: a denied point is infeasible; a neighbour is fine.
        assert!(!space.feasible(&space.config(&[("a", 7), ("b", 2), ("c", 5)])));
        assert!(space.denied(&space.config(&[("a", 7), ("b", 0), ("c", 0)])));
        assert!(space.feasible(&space.config(&[("a", 6), ("b", 2), ("c", 5)])));

        // Random draws never land in the denied subspace.
        let mut rng = Rng::new(5);
        for _ in 0..1000 {
            let c = space.random(&mut rng, 256).unwrap();
            assert_ne!(c.get(&space, "a"), 7);
        }

        // With the global column blocked, the hunt must produce an ADMISSIBLE result that escaped the far
        // deceptive well (rather than settling there because the global was denied) and sits near the best
        // admissible point -- but the deny-list is the hard contract: not one evaluated config may touch
        // the denied plane.
        let (_opt, opt_ms) = brute_argmin(&space, &eval, |c| c.get(&space, "a") != 7);
        let local_ms = eval.measure(&space.config(&[("a", 2), ("b", 7), ("c", 4)]), 1);
        let evo = Evolution::new().population(28).generations(22).tournament(3).mutation(0.35).elitism(2);
        let r = evo.hunt(&space, &eval, 99);
        assert!(space.feasible(&r.best) && r.best.get(&space, "a") != 7, "best is denied: {}", r.best_name);
        assert!(r.best_ms < local_ms - 0.1, "did not escape the deceptive well: best_ms={}", r.best_ms);
        assert!(r.best_ms <= opt_ms + 2.0, "best far from the admissible optimum: {}", r.best_ms);
        // The hard contract: not one evaluated config touched the denied plane.
        for (name, _) in r.measured.iter() {
            let c = space.parse(name).unwrap();
            assert_ne!(c.get(&space, "a"), 7, "measured a denied config: {name}");
        }
    }

    /// The free static tier must gate the GPU tier: a rejected config is never `measure`d. Here
    /// `static_check` rejects `a+b+c > 20`, and `measure` asserts it is never handed such a config.
    struct Gated<'a> {
        space: &'a Space,
        measured: std::cell::RefCell<usize>,
    }
    impl<'a> Evaluator for Gated<'a> {
        fn static_check(&self, c: &Config) -> Verdict {
            let sum = c.get(self.space, "a") + c.get(self.space, "b") + c.get(self.space, "c");
            if sum > 20 {
                Verdict::Reject(format!("synthetic spill: sum={sum} > 20"))
            } else {
                Verdict::Pass
            }
        }
        fn measure(&self, c: &Config, _iters: usize) -> f64 {
            let sum = c.get(self.space, "a") + c.get(self.space, "b") + c.get(self.space, "c");
            assert!(sum <= 20, "measure() ran on a statically-rejected config: sum={sum}");
            *self.measured.borrow_mut() += 1;
            // Fitness pulls toward the boundary so the search actively probes near the rejected region.
            -(sum as f64)
        }
    }

    #[test]
    fn evo_multi_fidelity_gates_the_gpu_tier() {
        let space = grid3();
        let eval = Gated { space: &space, measured: std::cell::RefCell::new(0) };
        let evo = Evolution::new().population(24).generations(15).mutation(0.35);
        let r = evo.hunt(&space, &eval, 2024);

        // Some configs were pruned for free (the space has many sum>20 points), and the winner is a
        // best-admissible boundary config (sum == 20), never a rejected one.
        assert!(!r.rejected.is_empty(), "nothing was statically rejected");
        assert!(r.rejected.iter().all(|(_, why)| why.contains("spill")));
        assert!(r.best_ms.is_finite());
        assert_eq!(r.best_ms, -20.0, "winner should sit on the sum==20 boundary");
        // measured count == GPU-tier calls == the length of the measured trail.
        assert_eq!(*eval.measured.borrow(), r.measured.len());
        // measured and rejected are disjoint and together are the whole footprint.
        assert_eq!(r.measured.len() + r.rejected.len(), r.evaluated);
    }

    #[test]
    fn space_name_parse_round_trips() {
        let space = Space::new().param("NWARP", [2, 4, 8]).param("RM", [1, 2, 4]);
        let c = space.config(&[("NWARP", 4), ("RM", 2)]);
        assert_eq!(c.name(&space), "NWARP=4,RM=2");
        assert_eq!(space.parse("NWARP=4,RM=2").as_ref(), Some(&c));
        assert!(space.parse("NWARP=4").is_none()); // missing a parameter
        assert!(space.parse("NWARP=3,RM=2").is_none()); // out-of-domain value
        assert!(space.parse("BOGUS=1,RM=2").is_none()); // unknown parameter
    }

    #[test]
    fn space_enumerate_lists_only_feasible() {
        let space = Space::new()
            .param("a", [1, 2, 3])
            .param("b", [1, 2, 3])
            .constraint(|c, s| c.get(s, "a") + c.get(s, "b") <= 4)
            .deny([("a", 2)]);
        let all = space.enumerate();
        assert!(all.iter().all(|c| space.feasible(c)));
        assert!(all.iter().all(|c| c.get(&space, "a") != 2)); // deny honoured
        // a=1 -> b in {1,2,3}; a=3 -> b in {1}; a=2 denied. = 4.
        assert_eq!(all.len(), 4);
    }

    /// A unimodal bowl (unique min at the origin corner) -- enough to test the cache bridge deterministically.
    struct Bowl<'a> {
        space: &'a Space,
        calls: std::cell::RefCell<usize>,
    }
    impl<'a> Evaluator for Bowl<'a> {
        fn static_check(&self, _c: &Config) -> Verdict {
            Verdict::Pass
        }
        fn measure(&self, c: &Config, _iters: usize) -> f64 {
            *self.calls.borrow_mut() += 1;
            (c.get(self.space, "a") + c.get(self.space, "b")) as f64
        }
    }

    #[test]
    fn tuner_evolve_caches_and_skips_the_hunt() {
        let dir = tmp_dir("evolve");
        let space = Space::new().param("a", 0..=4).param("b", 0..=4);
        let eval = Bowl { space: &space, calls: std::cell::RefCell::new(0) };
        let evo = Evolution::new().population(8).generations(5);

        // First call misses: a hunt runs, records the winner, and spends GPU-tier calls.
        let t1 = Tuner::new(&dir);
        let r1 = t1.evolve("cpu", "coopmat", "k1", &space, &eval, &evo, 7);
        assert!(!r1.from_cache);
        assert!(r1.report.is_some());
        assert_eq!(r1.winner, "a=0,b=0"); // the bowl's unique minimum
        let after_first = *eval.calls.borrow();
        assert!(after_first > 0);

        // Second call hits the in-memory cache: NO hunt, NO new GPU-tier calls.
        let r2 = t1.evolve("cpu", "coopmat", "k1", &space, &eval, &evo, 7);
        assert!(r2.from_cache);
        assert!(r2.report.is_none());
        assert_eq!(r2.winner, r1.winner);
        assert_eq!(*eval.calls.borrow(), after_first, "cache hit still ran the hunt");

        // A fresh tuner over the same dir reloads the winner from disk and still skips.
        let t2 = Tuner::new(&dir);
        let r3 = t2.evolve("cpu", "coopmat", "k1", &space, &eval, &evo, 7);
        assert!(r3.from_cache);
        assert_eq!(r3.winner, r1.winner);
        assert_eq!(*eval.calls.borrow(), after_first);

        std::fs::remove_dir_all(&dir).ok();
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
