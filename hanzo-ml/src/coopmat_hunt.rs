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
//! EXPENSIVE tier bit-exact-checks the survivor against the dp4a oracle and times it COLD -- rotating a
//! weight bank larger than the GPU's last-level cache so every timed GEMM streams its weight cold from
//! GTT, isolated by a synchronize (the barrier-drained regime the real forward pays). A single reused
//! tile fits the cache and measures on-chip occupancy the deployment never sees; that warm oracle is the
//! campaign's #1 scar -- it crowned NWARP=8 as a +24% win that measured flat in-engine (refutation #27).
//! The [`cold_oracle_agrees_with_in_engine`] test is the committed gate on this fitness. The refutation
//! log is encoded as the space's deny-list, so the hunt never spends a dispatch re-disproving a knob a
//! prior hunt already killed (BK=64, BN=64, double-buffer).

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
/// Cold-stream timing passes per fitness eval; the minimum is taken. The integrated GPU's clock is
/// utilization-slaved and drifts across a hunt, so one pass ranks by the clock a config happened to catch;
/// the min over a few passes is the least-drift-polluted time, making configs measured apart comparable.
const MEASURE_REPEATS: usize = 3;
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

/// The two-tier fitness for the coopmat genome over a fixed shape. Holds the resident cold-weight bank,
/// activation, and dp4a oracle so every variant is checked and timed against the same ground truth. The
/// timing tier rotates the whole bank (cold-weight streaming); `bank[0]` is the correctness weight the
/// oracle was computed from. Interior mutability because [`Evaluator`] is `&self`: the compiled bytes
/// cached from the static tier feed the GPU tier, and a last-installed marker avoids rebuilding the
/// pipeline between the two.
struct CoopmatEval<'a> {
    dev: &'a VulkanDevice,
    space: &'a Space,
    bank: &'a [VulkanStorage],
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
        self.dev.matmul_q4k_coopmat_variant(&self.bank[0], self.xh, self.m, self.n, self.k)?.to_vec_f32()
    }

    /// One cold-weight-streaming, GTT-saturated timing pass over the installed variant: rotate the whole
    /// >2xMALL bank so every GEMM reads its weight COLD from GTT, PIPELINED (no per-dispatch sync) so
    /// consecutive GEMMs overlap and keep the GTT link saturated -- the memory-bandwidth-bound regime the
    /// real prefill runs in. A warmup of two full rotations drives the utilization-slaved clock to its
    /// steady high state (still cold per-tile: rotation guarantees eviction) before the timed window.
    fn cold_pipe_ms(&self, iters: usize) -> f64 {
        let nb = self.bank.len();
        for i in 0..(2 * nb) {
            let _ = self.dev.matmul_q4k_coopmat_variant(&self.bank[i % nb], self.xh, self.m, self.n, self.k);
        }
        let _ = self.dev.synchronize();
        let t = std::time::Instant::now();
        for i in 0..iters {
            let _ = self.dev.matmul_q4k_coopmat_variant(&self.bank[i % nb], self.xh, self.m, self.n, self.k);
        }
        let _ = self.dev.synchronize();
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
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
            let (res, text) = capture_stderr(|| self.dev.matmul_q4k_coopmat_variant(&self.bank[0], self.xh, self.m, self.n, self.k));
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
        // COLD-WEIGHT-STREAMING, GTT-SATURATED fitness -- the campaign's #1 scar made a gate. A single
        // reused tile fits the 32 MB MALL and is served warm after iter 1, measuring on-chip occupancy the
        // deployment never sees -- that is why the warm winner NWARP=8 scored +24% but was flat in-engine.
        // [`cold_pipe_ms`] instead streams cold weights through a saturated GTT link, the memory-bandwidth-
        // bound regime the real prefill runs in, where extra occupancy (NWARP 4->8) converts to nothing so
        // the oracle correctly does NOT crown NWARP=8. Take the MINIMUM over several passes: the integrated
        // GPU's clock is utilization-slaved and drifts across a long hunt, so a single pass ranks configs by
        // whatever clock they caught (a false +24% for a config measured hot). The minimum is each config's
        // least-drift-polluted cold-stream time, making configs measured minutes apart comparable. The
        // whole-forward DVFS downclock is still not reproducible in a busy microbench, so a genuine winner
        // must be confirmed in-engine before it changes a default.
        (0..MEASURE_REPEATS).map(|_| self.cold_pipe_ms(iters)).fold(f64::INFINITY, f64::min)
    }
}

/// The gfx1151 last-level ("MALL"/Infinity) cache is 32 MB (rocminfo L3). A weight working-set that fits
/// it is served warm after the first touch -- so a single reused tile measures on-chip occupancy, not the
/// cold GTT weight stream the real forward pays. The cold oracle sizes its bank to exceed this.
const MALL_BYTES: usize = 32 * 1024 * 1024;

/// Bytes of one Q4_K weight tile of `n` rows x `k` cols (n*(k/256) blocks of 144 B).
fn tile_bytes(n: usize, k: usize) -> usize {
    n * (k / 256) * std::mem::size_of::<BlockQ4K>()
}

/// Build one resident Q4_K weight tile (`n` rows x `k` cols) from `seed`. Distinct seeds give distinct
/// device buffers at distinct addresses, so a bank of them does not alias in cache. Real Q4_K blocks
/// (valid f16 d/dmin) -- random bytes would seed Inf/NaN f16 scales.
fn build_weight(dev: &VulkanDevice, seed: u64, n: usize, k: usize) -> VulkanStorage {
    let nb = k / 256;
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut blocks: Vec<BlockQ4K> = (0..n * nb).map(|_| unsafe { std::mem::zeroed() }).collect();
    for r in 0..n {
        let rowf: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
        BlockQ4K::from_float(&rowf, &mut blocks[r * nb..(r + 1) * nb]);
    }
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, blocks.len() * std::mem::size_of::<BlockQ4K>()) };
    dev.upload_qweight(bytes).unwrap()
}

/// A cold-weight bank: enough distinct tiles that rotating through them evicts each before its next use,
/// so every timed GEMM reads its weight COLD from GTT (the deployment regime). Sized past 2x the MALL so
/// the tiles touched between two uses of any one tile (>= MALL) guarantee its eviction; >= 4 tiles minimum
/// so a short rotation still cycles cold.
fn build_bank(dev: &VulkanDevice, n: usize, k: usize) -> Vec<VulkanStorage> {
    let n_tiles = ((2 * MALL_BYTES).div_ceil(tile_bytes(n, k)) + 1).max(4);
    (0..n_tiles).map(|i| build_weight(dev, 0xC01D_0000u64.wrapping_add(i as u64).wrapping_mul(0x9E3779B1), n, k)).collect()
}

/// Build the cold-weight bank (whose `bank[0]` doubles as the correctness weight), an activation, and the
/// dp4a oracle (the trusted int8 path) computed on `bank[0]`. The bank exceeds the MALL so the timing tier
/// streams cold weights; the oracle anchors the bit-exact gate folded into fitness.
fn build_bank_and_oracle(
    dev: &VulkanDevice,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<VulkanStorage>, VulkanStorage, Vec<f32>, f32) {
    let bank = build_bank(dev, n, k);
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let x: Vec<f32> = (0..m * k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let xh = dev.upload_f32(&x).unwrap();
    unsafe { std::env::set_var("VK_Q4K_COOPMAT_OFF", "1") };
    let oracle = dev.matmul_q4k_gpu(&bank[0], &xh, m, n, k).unwrap().to_vec_f32().unwrap();
    unsafe { std::env::remove_var("VK_Q4K_COOPMAT_OFF") };
    let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
    (bank, xh, oracle, maxref)
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
        let (bank, xh, oracle, maxref) = build_bank_and_oracle(&dev, m, n, k);
        eprintln!(
            "[hunt] cold bank: {} tiles x {} MB = {} MB (MALL {} MB) -- timing streams cold weights",
            bank.len(),
            tile_bytes(n, k) >> 20,
            (bank.len() * tile_bytes(n, k)) >> 20,
            MALL_BYTES >> 20
        );
        let space = coopmat_space();
        let feasible = space.enumerate().len();
        let tmp = std::env::temp_dir().join(format!("hk-coopmat-hunt-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let eval = CoopmatEval {
            dev: &dev,
            space: &space,
            bank: &bank,
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

    /// THE ACCEPTANCE GATE for the fitness oracle (GPU; runs whenever a coopmat device is present, skips
    /// gracefully otherwise). The cold oracle must make the DEPLOYMENT decision on the warm hunt's winner
    /// `NWARP=8,RM=1` vs the shipped incumbent `NWARP=4,RM=2`: DO NOT crown NWARP=8. The CTO measured that
    /// variant in-engine at coopmat op 181 ms UNCHANGED / pp512 1750.4 vs 1745.5 = flat (refutation #27),
    /// while the old WARM microbench crowned it a +24% win. The cold-pipelined oracle streams cold weights
    /// through a saturated GTT (the memory-bandwidth-bound regime the real prefill runs in) and scores
    /// NWARP=8 at ~-10% -- i.e. NOT a win, so a hunt keeps the incumbent, matching the in-engine action.
    /// (An isolated per-dispatch-synchronized loop instead spuriously credits NWARP=8's occupancy ~+11% at
    /// an unsaturated link; the on-chip DVFS downclock of the whole forward is not reproducible in a busy
    /// microbench, so a real winner must still be in-engine-confirmed.) If this ever scores NWARP=8 a win
    /// past the band, the oracle has regressed to a warm/unsaturated regime and no winner it produces can be
    /// trusted to change a default. Encodes the golden negative-control as a committed assertion.
    #[test]
    fn cold_oracle_agrees_with_in_engine() {
        let dev = match VulkanDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("[cold-oracle] no vulkan device ({e}); skipping");
                return;
            }
        };
        if dev.coopmat_info().is_none() {
            eprintln!("[cold-oracle] no coopmat; skipping");
            return;
        }
        let (m, n, k) = (512usize, 11008usize, 2048usize); // ffn gate/up -- the shape refutation #27 measured
        let (bank, xh, oracle, maxref) = build_bank_and_oracle(&dev, m, n, k);
        let space = coopmat_space();
        let tmp = std::env::temp_dir().join(format!("hk-cold-oracle-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let eval = CoopmatEval {
            dev: &dev,
            space: &space,
            bank: &bank,
            xh: &xh,
            oracle: &oracle,
            maxref,
            m,
            n,
            k,
            tmp: tmp.clone(),
            shaderstats: false,
            compiled: RefCell::new(HashMap::new()),
            installed: RefCell::new(None),
            worst_rel: Cell::new(0.0),
        };
        assert!(
            bank.len() * tile_bytes(n, k) > 2 * MALL_BYTES,
            "cold bank {} MB must exceed 2x the {} MB MALL to guarantee eviction",
            (bank.len() * tile_bytes(n, k)) >> 20,
            MALL_BYTES >> 20
        );

        let incumbent_cfg = incumbent(&space); // NWARP=4,RM=2 -- the shipped default
        let warm_winner = space.config(&[("NWARP", 8), ("RM", 1), ("RN", 8), ("PAD", 4), ("BK", 32), ("DBUF", 0)]);
        // Both must reach the GPU tier (compile + pass static) for the comparison to be meaningful.
        for cfg in [&incumbent_cfg, &warm_winner] {
            match eval.static_check(cfg) {
                Verdict::Pass => {}
                Verdict::Reject(why) => panic!("golden config {} statically rejected: {why}", cfg.name(&space)),
            }
        }
        // measure() already takes the min over MEASURE_REPEATS cold-stream passes (drift-resistant), so one
        // call per config is the honest fitness -- exactly what the hunt uses.
        let ms_inc = eval.measure(&incumbent_cfg, 30);
        let ms_warm = eval.measure(&warm_winner, 30);
        let delta = (ms_inc - ms_warm) / ms_inc * 100.0; // + => NWARP=8 scored faster (the illusion to reject)
        eprintln!(
            "[cold-oracle] {m}x{n}x{k}  incumbent {} = {ms_inc:.3} ms | warm-winner {} = {ms_warm:.3} ms | delta {delta:+.1}%  (warm microbench crowned it +24%; cold-pipe scores it NOT-a-win, matching in-engine flat)",
            incumbent_cfg.name(&space),
            warm_winner.name(&space)
        );
        std::fs::remove_dir_all(&tmp).ok();

        assert!(ms_inc.is_finite() && ms_warm.is_finite(), "a golden config failed to measure");
        assert!(eval.worst_rel.get() < GATE, "a golden variant diverged from the oracle (scale_rel {:.2e})", eval.worst_rel.get());
        // The cold-pipelined (saturated) oracle scores NWARP=8 at ~-10% (NOT a win) -- it makes the same
        // decision as in-engine (keep the incumbent). The warm microbench inflated it to +17-24%, and an
        // isolated per-dispatch-synchronized loop credits it ~+11%; both would wrongly crown NWARP=8. The
        // 6% ceiling passes the saturated-cold verdict with margin while failing every regime that crowns
        // NWARP=8 -- exceed it and the oracle has regressed to a warm or unsaturated fitness.
        const CROWN_CEILING: f64 = 6.0;
        assert!(
            delta < CROWN_CEILING,
            "cold oracle scored the warm winner NWARP=8/RM=1 a {delta:+.1}% win over the incumbent -- the \
             ORACLE regressed to a warm/unsaturated regime (in-engine measured this pair flat, refutation \
             #27; the saturated cold-pipe oracle scores it ~-10%). Fix the cold-stream fitness."
        );
    }
}
