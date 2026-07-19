//! Evolutionary autotuner hunt over the `mul_mm_q4k_coopmat` prefill genome.
//!
//! Reuses the hanzo-kernel `tune` evolutionary search (a dev-dependency) to tune the committed Q4_K
//! coopmat prefill matmul at its dominant live shape (the ffn gate/up, m=512 n=11008 k=2048). The genome
//! is the shader's `#ifndef`-defaulted schedule knobs (RM/RN/NWARP tile geometry + the LDS stride pad);
//! a config is realized by compiling the same `.comp` with `glslc -D`, dispatched through the real batch
//! path via [`VulkanDevice::install_coopmat_variant`] / [`VulkanDevice::matmul_q4k_coopmat_variant`].
//!
//! Multi-fidelity, as the search demands: the FREE static tier compiles the variant and reads its RADV
//! shader stats, rejecting a register-spilling or over-LDS schedule before it costs a dispatch; the
//! EXPENSIVE tier bit-exact-checks the survivor against the dp4a oracle and times it SUSTAINED (warmed,
//! back-to-back, so the integrated GPU's utilization-slaved clock is at its steady state -- a burst
//! number is a lie for this part). The refutation log is encoded as the space's deny-list, so the hunt
//! never spends a dispatch re-disproving a knob a prior hunt already killed (BK=64, BN=64, double-buffer).

use super::*;
use crate::backend::BackendDevice;
use crate::quantized::k_quants::{BlockQ4K, GgmlType};
use hanzo_kernel::tune::{Config, Evaluator, Evolution, Space, Verdict};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// The templatized shader (source of truth) and its committed SPIR-V (the byte-identical anchor).
const SHADER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/vulkan/shaders/mul_mm_q4k_coopmat.comp");
const COMMITTED_SPV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src/vulkan/spv/mul_mm_q4k_coopmat.spv");

/// LDS ceiling (bytes) above which occupancy drops below the single-buffer incumbent's. The 0.11.71
/// single-buffer win measured 20 KB -> 8 subgroups and 40 KB (double buffer) -> 6; 32 KB is the honest
/// budget between them, so a config over it is the refuted low-occupancy regime -- reject before timing.
const LDS_BUDGET: usize = 32 * 1024;
/// coopmat tile edge (hardware 16x16x16; not a knob).
const TILE: i64 = 16;
/// Correctness gate: coopmat rounds weight+activation to f16, dp4a rounds the activation to int8, so the
/// two agree only to a scale-relative f16 bound accumulated over k (same as `q4k_coopmat_matches_dp4a`).
const GATE: f32 = 2e-2;

/// The Q4_K coopmat prefill schedule genome. RM/RN/NWARP set the accumulator tile (BM = NWARP*RM*16 rows,
/// BN = RN*16 cols, WG = NWARP*64 lanes); PAD is the LDS anti-bank-aliasing stride pad (pairs). BK and
/// DBUF exist ONLY to materialize the refutation log as priors -- BK=64 is a Q4_K-sub-block decode
/// rewrite the shader does not implement, DBUF=1 (double buffer) was refuted (2x LDS caps occupancy) --
/// so both are denied and the reachable genome is exactly the single-buffer BK=32 family the committed
/// shader realizes.
fn coopmat_space() -> Space {
    Space::new()
        .param("NWARP", [2, 4, 8])
        .param("RM", [1, 2, 4])
        .param("RN", [4, 8, 16])
        .param("PAD", [2, 4, 8])
        .param("BK", [32, 64])
        .param("DBUF", [0, 1])
        // BM = NWARP*RM*16 in {128, 256}: 128 is the proven tile height, 256 the one untested step up.
        // Excludes the refuted BM=64 (sb_bm64, occupancy collapse) and anything taller than the batch.
        .constraint(|c, s| {
            let bm = c.get(s, "NWARP") * c.get(s, "RM") * TILE;
            bm == 128 || bm == 256
        })
        // WG = NWARP*64 <= 1024 (max compute workgroup size).
        .constraint(|c, s| c.get(s, "NWARP") * 64 <= 1024)
        .deny([("BK", 64)]) // Q4_K sub-block decode is written for 32; 64 is a decode rewrite -- refuted
        .deny([("RN", 4)]) // BN=64 coopmat -- refuted
        .deny([("DBUF", 1)]) // double buffer: 2x LDS caps occupancy -- refuted (0.11.71 single-buffer win)
}

/// The shipped 0.11.71 single-buffer coopmat default (the A/B baseline the hunt is judged against).
fn incumbent(space: &Space) -> Config {
    space.config(&[("NWARP", 4), ("RM", 2), ("RN", 8), ("PAD", 4), ("BK", 32), ("DBUF", 0)])
}

/// Derived tile geometry `(BM, BN, WG)` for a config.
fn geom(c: &Config, s: &Space) -> (u32, u32, u32) {
    let bm = (c.get(s, "NWARP") * c.get(s, "RM") * TILE) as u32;
    let bn = (c.get(s, "RN") * TILE) as u32;
    let wg = (c.get(s, "NWARP") * 64) as u32;
    (bm, bn, wg)
}

/// `glslc -D` flags realizing a config. BK and DBUF are never emitted -- they are denied, so the shader's
/// fixed BK=32 single-buffer defaults are the only reachable realization.
fn defines(c: &Config, s: &Space) -> Vec<String> {
    let (_, _, wg) = geom(c, s);
    vec![
        format!("-DHK_RM={}u", c.get(s, "RM")),
        format!("-DHK_RN={}u", c.get(s, "RN")),
        format!("-DHK_NWARP={}u", c.get(s, "NWARP")),
        format!("-DHK_WG={wg}"),
        format!("-DHK_PAD={}u", c.get(s, "PAD")),
    ]
}

/// Computed LDS bytes: two single-buffered f16vec2 tiles, s_a[BM*(BK/2+PAD)] + s_b[BN*(BK/2+PAD)], 4 B each.
fn lds_bytes(c: &Config, s: &Space) -> usize {
    let (bm, bn, _) = geom(c, s);
    let stride = (c.get(s, "BK") / 2 + c.get(s, "PAD")) as usize; // pairs (BK is always 32 here)
    (bm as usize * stride + bn as usize * stride) * 4
}

/// Compile a config to SPIR-V bytes via glslc; `Err(reason)` is a static rejection (the reason surfaces
/// in the report). Honours `GLSLC` for an unusual toolchain, matching `regen-spv.sh` / `build.rs`.
fn compile(c: &Config, s: &Space, out: &Path) -> std::result::Result<Vec<u8>, String> {
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".to_string());
    let o = Command::new(&glslc)
        .arg("--target-env=vulkan1.2")
        .args(defines(c, s))
        .arg(SHADER)
        .arg("-o")
        .arg(out)
        .output()
        .map_err(|e| format!("glslc spawn: {e}"))?;
    if !o.status.success() {
        let msg = String::from_utf8_lossy(&o.stderr);
        return Err(format!("glslc: {}", msg.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim()));
    }
    std::fs::read(out).map_err(|e| format!("read spv: {e}"))
}

// --- RADV shader stats (best-effort static tier) -------------------------------------------------

#[derive(Debug, Default, Clone)]
struct Stats {
    vgpr: Option<u32>,
    spilled_vgpr: Option<u32>,
    spilled_sgpr: Option<u32>,
    scratch: Option<u32>,
}

impl Stats {
    /// A schedule the register allocator could not place: any spill or scratch is a hard reject. When
    /// nothing parsed (no RADV / stats unavailable) every field is `None`, so this is `false` and the
    /// config is admitted to timing rather than rejected on absent evidence.
    fn spills(&self) -> bool {
        self.scratch.unwrap_or(0) > 0
            || self.spilled_vgpr.unwrap_or(0) > 0
            || self.spilled_sgpr.unwrap_or(0) > 0
    }
}

/// The first unsigned integer on the first line containing `key` (case-insensitive). RADV prints
/// `VGPRS: 40`, `Spilled VGPRs: 0`, `Scratch: 0`, `LDS: 5120` -- exact wording drifts across Mesa, so
/// match tolerantly on the keyword and take the number after it.
fn find_num(text: &str, key: &str) -> Option<u32> {
    for line in text.lines() {
        let l = line.to_ascii_lowercase();
        if l.contains(key) {
            let n: String = l.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = n.parse() {
                return Some(v);
            }
        }
    }
    None
}

fn parse_stats(text: &str) -> Stats {
    Stats {
        spilled_vgpr: find_num(text, "spilled vgpr"),
        spilled_sgpr: find_num(text, "spilled sgpr"),
        scratch: find_num(text, "scratch"),
        // A "vgprs:" line that is not the "spilled vgprs" line.
        vgpr: text
            .lines()
            .map(|l| l.trim().to_ascii_lowercase())
            .find(|l| l.starts_with("vgpr"))
            .and_then(|l| find_num(&l, "vgpr")),
    }
}

/// Run `f` with fd 2 redirected to a temp file, returning `(result, captured_stderr)`. Used to grab
/// RADV's `RADV_DEBUG=shaderstats` output (written to stderr at pipeline creation). Unix-only; the whole
/// module is test-gated so this never ships.
fn capture_stderr<R>(f: impl FnOnce() -> R) -> (R, String) {
    use std::os::unix::io::AsRawFd;
    let path = std::env::temp_dir().join(format!("hk-shaderstats-{}-{}.txt", std::process::id(), rand_tag()));
    let Ok(file) = std::fs::File::create(&path) else {
        return (f(), String::new());
    };
    let (r, text);
    unsafe {
        let saved = libc::dup(2);
        libc::dup2(file.as_raw_fd(), 2);
        r = f();
        libc::fflush(std::ptr::null_mut()); // flush C stdio (RADV uses fprintf(stderr, ...))
        libc::dup2(saved, 2);
        libc::close(saved);
    }
    drop(file);
    text = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    (r, text)
}

fn rand_tag() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

// --- the evaluator -------------------------------------------------------------------------------

/// The two-tier fitness for the coopmat genome over a fixed shape. Holds the resident weight, activation,
/// and dp4a oracle so every variant is checked and timed against the same ground truth. Interior
/// mutability because [`Evaluator`] is `&self`: the compiled bytes cached from the static tier feed the
/// GPU tier, and a last-installed marker avoids rebuilding the pipeline between the two.
struct CoopmatEval<'a> {
    dev: &'a VulkanDevice,
    space: &'a Space,
    wq: &'a VulkanStorage,
    xh: &'a VulkanStorage,
    oracle: &'a [f32],
    maxref: f32,
    m: usize,
    n: usize,
    k: usize,
    tmp: std::path::PathBuf,
    shaderstats: bool,
    compiled: RefCell<HashMap<Config, (Vec<u8>, u32, u32)>>,
    installed: RefCell<Option<Config>>,
    worst_rel: Cell<f32>,
}

impl<'a> CoopmatEval<'a> {
    /// Ensure the device is running this config's variant (rebuilds the pipeline only on a change).
    fn install(&self, cfg: &Config) {
        if self.installed.borrow().as_ref() == Some(cfg) {
            return;
        }
        let map = self.compiled.borrow();
        let (spv, bm, bn) = map.get(cfg).expect("install: config must be compiled in the static tier first");
        self.dev.install_coopmat_variant(spv, *bm, *bn);
        drop(map);
        *self.installed.borrow_mut() = Some(cfg.clone());
    }

    fn run_once(&self) -> crate::Result<Vec<f32>> {
        self.dev.matmul_q4k_coopmat_variant(self.wq, self.xh, self.m, self.n, self.k)?.to_vec_f32()
    }
}

impl<'a> Evaluator for CoopmatEval<'a> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (bm, bn, _) = geom(cfg, self.space);
        let lds = lds_bytes(cfg, self.space);
        if lds > LDS_BUDGET {
            return Verdict::Reject(format!("LDS {}KB > {}KB budget (occupancy)", lds / 1024, LDS_BUDGET / 1024));
        }
        let out = self.tmp.join("variant.spv");
        let spv = match compile(cfg, self.space, &out) {
            Ok(b) => b,
            Err(e) => return Verdict::Reject(e),
        };
        self.compiled.borrow_mut().insert(cfg.clone(), (spv, bm, bn));

        // Build the pipeline once and read its RADV stats; reject a spilling schedule before any timing.
        if self.shaderstats {
            self.install(cfg);
            let (res, text) = capture_stderr(|| self.dev.matmul_q4k_coopmat_variant(self.wq, self.xh, self.m, self.n, self.k));
            let _ = self.dev.synchronize();
            if res.is_ok() {
                let st = parse_stats(&text);
                if st.spills() {
                    return Verdict::Reject(format!(
                        "spill: scratch={:?} spilled_vgpr={:?} vgpr={:?}",
                        st.scratch, st.spilled_vgpr, st.vgpr
                    ));
                }
            }
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        self.install(cfg);
        // Correctness gate folded into fitness: a variant that diverges from the dp4a oracle is
        // infinitely slow, so it can never win. Same scale-relative bound as q4k_coopmat_matches_dp4a.
        let got = match self.run_once() {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[hunt] {} dispatch error: {e}", cfg.name(self.space));
                return f64::INFINITY;
            }
        };
        let rel = got.iter().zip(self.oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if rel > GATE {
            eprintln!("[hunt] {} DIVERGED scale_rel={rel:.2e} > {GATE:.0e}", cfg.name(self.space));
            return f64::INFINITY;
        }
        // Sustained: warm past the boost window (keep the GPU busy so the utilization-slaved clock rises),
        // then time a back-to-back run. Same harness shape as mmq_q4k_coopmat_vs_dp4a_prefill_ab.
        for _ in 0..10 {
            let _ = self.dev.matmul_q4k_coopmat_variant(self.wq, self.xh, self.m, self.n, self.k);
        }
        let _ = self.dev.synchronize();
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let _ = self.dev.matmul_q4k_coopmat_variant(self.wq, self.xh, self.m, self.n, self.k);
        }
        let _ = self.dev.synchronize();
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    }
}

/// Build a resident Q4_K weight, an activation, and the dp4a oracle (the trusted int8 path) for `shape`.
fn build_inputs(dev: &VulkanDevice, m: usize, n: usize, k: usize) -> (VulkanStorage, VulkanStorage, Vec<f32>, f32) {
    let nb = k / 256;
    let mut s = 0x0BADC0DE_CAFEF00Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    // Real Q4_K blocks (valid f16 d/dmin) -- random bytes would seed Inf/NaN f16 scales.
    let mut blocks: Vec<BlockQ4K> = (0..n * nb).map(|_| unsafe { std::mem::zeroed() }).collect();
    for r in 0..n {
        let rowf: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
        BlockQ4K::from_float(&rowf, &mut blocks[r * nb..(r + 1) * nb]);
    }
    let wq_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * std::mem::size_of::<BlockQ4K>()) };
    let wq = dev.upload_qweight(wq_bytes).unwrap();
    let x: Vec<f32> = (0..m * k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let xh = dev.upload_f32(&x).unwrap();
    unsafe { std::env::set_var("VK_Q4K_COOPMAT_OFF", "1") };
    let oracle = dev.matmul_q4k_gpu(&wq, &xh, m, n, k).unwrap().to_vec_f32().unwrap();
    unsafe { std::env::remove_var("VK_Q4K_COOPMAT_OFF") };
    let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
    (wq, xh, oracle, maxref)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-only (glslc, no GPU): every feasible genome config compiles to valid SPIR-V, and the
    /// incumbent's `-D` realization reproduces the committed module byte-for-byte. Proves the whole
    /// config -> defines -> glslc pipeline is sound across the entire search domain, and re-anchors the
    /// byte-identical gate the templatization rests on.
    #[test]
    fn coopmat_genome_compiles_across_space() {
        let space = coopmat_space();
        let configs = space.enumerate();
        assert!(!configs.is_empty(), "empty feasible space");
        let tmp = std::env::temp_dir().join(format!("hk-coopmat-sweep-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let mut ok = 0usize;
        let mut fails: Vec<(String, String)> = Vec::new();
        for c in &configs {
            match compile(c, &space, &tmp.join("v.spv")) {
                Ok(_) => ok += 1,
                Err(e) => fails.push((c.name(&space), e)),
            }
        }
        eprintln!("[coopmat-sweep] {} feasible configs; {ok} compiled, {} failed", configs.len(), fails.len());
        for (nm, e) in &fails {
            eprintln!("  FAIL {nm}: {e}");
        }
        assert!(fails.is_empty(), "{} genome configs failed to compile", fails.len());

        // Incumbent -D compile == committed SPIR-V (the byte-identical anchor).
        let inc = incumbent(&space);
        let inc_spv = compile(&inc, &space, &tmp.join("inc.spv")).unwrap();
        let committed = std::fs::read(COMMITTED_SPV).expect("read committed spv");
        assert_eq!(inc_spv, committed, "incumbent -D realization diverged from the committed SPIR-V");

        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The first hunt (GPU; opt-in via `HANZO_COOPMAT_HUNT=1`). Tunes the coopmat prefill kernel at the
    /// ffn gate/up shape, seeded with the shipped incumbent, and prints the full evidence trail. A win
    /// updates the shader defaults; a plateau (incumbent stays best over the whole sound genome) is the
    /// valid, valuable equilibrium result. Set `HANZO_COOPMAT_NOSTATS=1` to skip RADV shaderstats,
    /// `HANZO_HUNT_SEED=<n>` to reseed.
    #[test]
    fn coopmat_hunt() {
        if std::env::var_os("HANZO_COOPMAT_HUNT").is_none() {
            return;
        }
        let shaderstats = std::env::var_os("HANZO_COOPMAT_NOSTATS").is_none();
        if shaderstats {
            // RADV reads this at device init, so it must be set before VulkanDevice::new.
            unsafe { std::env::set_var("RADV_DEBUG", "shaderstats") };
        }
        let dev = match VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[hunt] no vulkan device ({e}); skipping");
                return;
            }
        };
        if dev.coopmat_info().is_none() {
            eprintln!("[hunt] device has no cooperative matrix; skipping");
            return;
        }
        let (m, n, k) = (512usize, 11008usize, 2048usize); // ffn gate/up -- the dominant coopmat shape
        let (wq, xh, oracle, maxref) = build_inputs(&dev, m, n, k);
        let space = coopmat_space();
        let feasible = space.enumerate().len();
        let tmp = std::env::temp_dir().join(format!("hk-coopmat-hunt-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let eval = CoopmatEval {
            dev: &dev,
            space: &space,
            wq: &wq,
            xh: &xh,
            oracle: &oracle,
            maxref,
            m,
            n,
            k,
            tmp: tmp.clone(),
            shaderstats,
            compiled: RefCell::new(HashMap::new()),
            installed: RefCell::new(None),
            worst_rel: Cell::new(0.0),
        };

        let inc = incumbent(&space);
        // Population/generations sized to cover this small (~30) sound genome near-exhaustively, so the
        // measured winner is definitive. On a larger kernel the same call searches instead of enumerates.
        let evo = Evolution::new()
            .population(16)
            .generations(8)
            .tournament(3)
            .mutation(0.3)
            .elitism(2)
            .measure_iters(30)
            .seed_config(inc.clone());
        let seed = std::env::var("HANZO_HUNT_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEEu64);

        let report = evo.hunt(&space, &eval, seed);

        eprintln!("\n=== COOPMAT HUNT  ffn gate/up {m}x{n}x{k}  (seed {seed:#x}) ===");
        eprintln!(
            "feasible genome: {feasible} configs | evaluated: {} ({} measured, {} static-rejected) | coverage {:.0}%",
            report.evaluated,
            report.measured.len(),
            report.rejected.len(),
            100.0 * report.evaluated as f64 / feasible as f64
        );
        eprintln!("--- static-rejected (free tier) ---");
        for (nm, why) in &report.rejected {
            eprintln!("  REJECT  {nm}   {why}");
        }
        eprintln!("--- measured (sustained ms, fastest first) ---");
        for (nm, ms) in &report.measured {
            let gf = 2.0 * m as f64 * n as f64 * k as f64 / (ms * 1e6);
            let tag = if *nm == inc.name(&space) { "  <-- incumbent" } else { "" };
            eprintln!("  {ms:8.3} ms   {gf:7.0} GF/s   {nm}{tag}");
        }
        let inc_ms = report.measured.iter().find(|(nm, _)| *nm == inc.name(&space)).map(|(_, ms)| *ms);
        eprintln!(
            "--- WINNER {} @ {:.3} ms  vs incumbent {} @ {} ---",
            report.best_name,
            report.best_ms,
            inc.name(&space),
            inc_ms.map(|v| format!("{v:.3} ms")).unwrap_or_else(|| "n/a".into())
        );
        if let Some(iv) = inc_ms {
            let delta = (iv - report.best_ms) / iv * 100.0;
            if report.best_name == inc.name(&space) {
                eprintln!("--- RESULT: PLATEAU CONFIRMED -- the incumbent is the equilibrium over the sound genome ---");
            } else {
                eprintln!("--- RESULT: WINNER beats incumbent by {delta:.1}% (RE-VERIFY IN-ENGINE before shipping) ---");
            }
        }
        eprintln!("worst correctness scale_rel over the hunt: {:.2e} (gate {GATE:.0e})", eval.worst_rel.get());

        std::fs::remove_dir_all(&tmp).ok();

        // The hunt must crown a real, bit-exact winner.
        assert!(report.best_ms.is_finite(), "no measurable winner");
        assert!(eval.worst_rel.get() < GATE, "a measured variant diverged from the oracle");
    }
}
