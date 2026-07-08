//! Intrinsic islands — kill the whole-kernel escape hatch.
//!
//! When one inner loop of a kernel needs a target-specific instruction or idiom, you do NOT fork a
//! parallel per-backend kernel. You write a scoped, target-gated `island! { ... }` block *inside* the
//! one DSL kernel. The kernel keeps ONE signature, ONE launch contract, ONE CPU oracle, ONE test.
//!
//! ```ignore
//! #[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
//! pub fn matvec_island<F: Float>(
//!     wq: &Array<Vector<i8, Const<4>>>,
//!     xq: &Array<Vector<i8, Const<4>>>,
//!     wd: &Array<F>,
//!     out: &mut Array<F>,
//!     #[comptime] k: usize,
//!     #[comptime] target: Target,   // threaded by the launch layer; not a data input
//! ) {
//!     // ...
//!     let dp = island! {
//!         // accelerated arm: the hardware 4-wide int8 dot (dp4a / OpSDotAccSat)
//!         cuda | rocm | metal | vulkan => { w.dot(x) }
//!         // NORMATIVE fallback — the CPU-oracle semantics. Every unlisted target takes this.
//!         default => { w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] }
//!     };
//!     // ...
//! }
//! ```
//!
//! # The mechanism (target selection at comptime)
//!
//! cubecl compiles the SAME target-agnostic IR for every backend; at `#[cube]` *expand* time there is
//! no brand string for "which backend am I", only a capability set (`device_properties()`). The
//! backend brand IS known on the HOST, via `Runtime::name(client)` (`"cpu"`, `"cuda"`, `"hip"`,
//! `"wgpu<spirv>"`, `"wgpu<msl>"`). So islands are selected by a `#[comptime] target: Target` tag that
//! the launch layer derives from the runtime and threads in; the `#[kernel]` macro rewrites
//! `island! { ... }` into a comptime `match target { ... }`. A comptime match (a const scrutinee)
//! lowers ONLY the selected arm — every backend compiles exactly its own arm, nothing else.
//!
//! The tag is out of the authoring surface twice over: the branch site names no target (it lists
//! backends), and the launch wrapper fills the tag (`Target::of::<R>`), so callers never pass it.
//!
//! # The oracle contract
//!
//! `default` is NORMATIVE: islands must be semantically equivalent to it, and the CPU runtime is always
//! tagged `Target::Cpu` — a variant no accelerated arm claims — so on CPU every island resolves to
//! `default`. The CPU runtime is therefore the single bit-exact oracle for every island: run the
//! kernel on CPU with any accelerated tag and it MUST equal the `default` result byte-for-byte. The
//! tests below prove exactly that.

use crate::prelude::*;

/// The compile target an [`island!`] selects on. A `#[comptime]` tag: it rides in the kernel's
/// parameter block, so it derives `Eq + Hash` (the lowering engine keys the compiled kernel on it) and
/// `Copy` (it is a plain value). It is NOT a data input — the launch layer fills it from the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    /// The CPU runtime — always the `default` (oracle) arm.
    Cpu,
    /// NVIDIA CUDA.
    Cuda,
    /// AMD ROCm/HIP.
    Rocm,
    /// Apple Metal.
    Metal,
    /// Vulkan (SPIR-V).
    Vulkan,
    /// WebGPU / WGSL.
    WebGpu,
}

impl Target {
    /// The target the runtime `R` compiles for, derived on the HOST from `Runtime::name`. This is the
    /// launch layer's job; the kernel author never calls it. The CPU runtime maps to [`Target::Cpu`],
    /// which pins the oracle contract.
    pub fn of<R: Runtime>(client: &ComputeClient<R>) -> Target {
        Target::from_runtime_name(R::name(client))
    }

    /// Map a cubecl `Runtime::name` string to a [`Target`]. Split out so the host selection mechanism
    /// is unit-testable without instantiating any runtime. Names per the cubecl backends:
    /// `"cpu"`, `"cuda"`, `"hip"`, and wgpu's `"wgpu<spirv>"` / `"wgpu<msl>"` / `"wgpu<wgsl>"`.
    pub fn from_runtime_name(name: &str) -> Target {
        match name {
            "cuda" => Target::Cuda,
            "hip" => Target::Rocm,
            n if n.contains("spirv") => Target::Vulkan,
            n if n.contains("msl") => Target::Metal,
            n if n.starts_with("wgpu") => Target::WebGpu,
            _ => Target::Cpu,
        }
    }
}

// ================================================================================================
// The demo: ONE dp4a matvec kernel with ONE island. The accelerated arm uses the hardware 4-wide
// int8 dot (`Line<i32>.dot` -> dp4a / OpSDotAccSat); the `default` arm expresses the identical
// integer contraction as portable scalar lane MACs. Same math, two idioms — the island's whole point.
// ================================================================================================

/// dp4a matvec, one invocation per output row: `out[row] = sum_g wd[block(g)] * dot(wq_g, xq_g)`.
/// The inner 4-way int8 dot is an [`island!`]: accelerated targets issue the hardware dot instruction;
/// `default` (the CPU oracle, and any unlisted backend) computes the same sum with scalar lane MACs.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_island<F: Float>(
    wq: &Array<Vector<i8, Const<4>>>, // int8 weights, packed x4   [rows * k/4]
    xq: &Array<Vector<i8, Const<4>>>, // int8 activation, packed x4 [k/4]
    wd: &Array<F>,                    // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] target: Target,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let ng = k / 4;
        let nb = k / 32;
        let wbase = row * ng;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for g in 0..ng {
            // Widen the packed int8 lanes to i32 (lane-wise sign-extend) so the dot is exact.
            let w = Vector::<i32, Const<4>>::cast_from(wq[wbase + g]);
            let x = Vector::<i32, Const<4>>::cast_from(xq[g]);
            let dp = island! {
                // Accelerated: the hardware 4-wide int8 dot (dp4a / OpSDotAccSat), one instruction.
                cuda | rocm | metal | vulkan => { w.dot(x) }
                // NORMATIVE oracle: the identical integer contraction, portable scalar lane MACs.
                default => { w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] }
            };
            acc += wd[dbase + g / 8] * F::cast_from(dp);
        }
        out[row] = acc;
    }
}

/// Host launch — the PRODUCTION surface. Derives the island tag from the runtime (`Target::of`) and
/// hides it: callers pass only data. One signature, one launch, every backend.
pub fn matvec_island_run<R: Runtime>(
    client: &ComputeClient<R>,
    wq: &[i8],
    xq: &[i8],
    wd: &[f32],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    matvec_island_run_with(client, wq, xq, wd, rows, k, Target::of(client))
}

/// Host launch with an explicit tag — used by the oracle tests to force an accelerated arm onto the
/// CPU runtime and prove it equals `default` byte-for-byte. Production goes through [`matvec_island_run`].
pub fn matvec_island_run_with<R: Runtime>(
    client: &ComputeClient<R>,
    wq: &[i8],
    xq: &[i8],
    wd: &[f32],
    rows: usize,
    k: usize,
    target: Target,
) -> Vec<f32> {
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    let ng = k / 4;
    unsafe {
        matvec_island::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
            target,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle: the trusted plain-Rust reference the island's `default` arm mirrors, bit-for-bit (the
/// integer dot is exact; the f32 accumulation is in the same order as the kernel).
pub fn matvec_island_ref(wq: &[i8], xq: &[i8], wd: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    (0..rows)
        .map(|row| {
            let mut acc = 0.0f32;
            for g in 0..k / 4 {
                let mut dp = 0i32;
                for l in 0..4 {
                    dp += wq[row * k + g * 4 + l] as i32 * xq[g * 4 + l] as i32;
                }
                acc += wd[row * nb + g / 8] * dp as f32;
            }
            acc
        })
        .collect()
}

/// Deterministic valid test data: `wq` int8 `[rows*k]`, `xq` int8 `[k]`, `wd` f32 scales `[rows*k/32]`.
pub fn gen_island(rows: usize, k: usize) -> (Vec<i8>, Vec<i8>, Vec<f32>) {
    let mut s = 0xD1B5_4A32_D192_ED03u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let wq: Vec<i8> = (0..rows * k).map(|_| next() as i8).collect();
    let xq: Vec<i8> = (0..k).map(|_| next() as i8).collect();
    // f16-round the scale so the reference uses the precision a real GGUF scale would carry.
    let wd: Vec<f32> = (0..rows * (k / 32))
        .map(|_| half::f16::from_f32((next() % 1000) as f32 / 8000.0 + 0.01).to_f32())
        .collect();
    (wq, xq, wd)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- The host target-selection mechanism (no runtime needed) ----------------------------------

    #[test]
    fn runtime_name_maps_to_target() {
        // The exact cubecl `Runtime::name` strings, one per backend.
        assert_eq!(Target::from_runtime_name("cpu"), Target::Cpu);
        assert_eq!(Target::from_runtime_name("cuda"), Target::Cuda);
        assert_eq!(Target::from_runtime_name("hip"), Target::Rocm);
        assert_eq!(Target::from_runtime_name("wgpu<spirv>"), Target::Vulkan);
        assert_eq!(Target::from_runtime_name("wgpu<msl>"), Target::Metal);
        assert_eq!(Target::from_runtime_name("wgpu<wgsl>"), Target::WebGpu);
        // Anything unknown falls to the oracle arm — fail safe, never a wrong accelerated arm.
        assert_eq!(Target::from_runtime_name("something-else"), Target::Cpu);
    }

    // --- The oracle: island arms == default, bit-exact, on the CPU runtime ------------------------

    #[cfg(feature = "cpu")]
    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        CpuRuntime::client(&CpuDevice)
    }

    #[cfg(feature = "cpu")]
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The island's accelerated arm, FORCED onto the CPU runtime, is byte-for-byte the `default`
    /// (oracle) arm AND the plain-Rust reference — for every accelerated target. This is the whole
    /// contract: one kernel, one CPU oracle, every island proven equivalent.
    #[cfg(feature = "cpu")]
    #[test]
    fn island_arm_equals_default_oracle_bit_exact() {
        use cubecl::cpu::CpuRuntime;
        let client = cpu_client();
        let (rows, k) = (12usize, 256usize);
        let (wq, xq, wd) = gen_island(rows, k);

        // The oracle: the `default` arm (CPU is tagged Cpu -> default), and the plain-Rust reference.
        let default = matvec_island_run_with::<CpuRuntime>(&client, &wq, &xq, &wd, rows, k, Target::Cpu);
        let reference = matvec_island_ref(&wq, &xq, &wd, rows, k);
        assert_eq!(bits(&default), bits(&reference), "default arm != plain-Rust oracle");

        // Every accelerated arm, executed on the CPU runtime, must equal the oracle byte-for-byte.
        for accel in [Target::Cuda, Target::Rocm, Target::Metal, Target::Vulkan, Target::WebGpu] {
            let out = matvec_island_run_with::<CpuRuntime>(&client, &wq, &xq, &wd, rows, k, accel);
            assert_eq!(bits(&out), bits(&default), "island arm {accel:?} diverged from default");
        }
    }

    /// The PRODUCTION surface (`matvec_island_run`, which derives the tag from the runtime) resolves to
    /// the oracle on CPU without the caller ever naming a target.
    #[cfg(feature = "cpu")]
    #[test]
    fn production_run_selects_oracle_on_cpu() {
        use cubecl::cpu::CpuRuntime;
        let client = cpu_client();
        let (rows, k) = (7usize, 128usize);
        let (wq, xq, wd) = gen_island(rows, k);

        let out = matvec_island_run::<CpuRuntime>(&client, &wq, &xq, &wd, rows, k);
        assert_eq!(bits(&out), bits(&matvec_island_ref(&wq, &xq, &wd, rows, k)));
    }
}
