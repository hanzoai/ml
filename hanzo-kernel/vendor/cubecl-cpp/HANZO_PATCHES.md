# Hanzo patches to cubecl-cpp 0.10.0

Vendored from crates.io `cubecl-cpp 0.10.0` and wired in by `[patch.crates-io]` in
`hanzo-kernel/Cargo.toml`. `diff -r ~/.cargo/registry/src/*/cubecl-cpp-0.10.0 vendor/cubecl-cpp`
shows only the hunks marked `HANZO PATCH` (plus this file). Delete the vendor directory and the
patch entry once upstream ships the fix in a release we can take.

## bitcast-of-literal (`src/shared/instruction.rs`, `Instruction::Bitcast`)

**Bug.** A bitcast whose input is a literal (`Variable::Constant`) is emitted as
`const float l_71 = reinterpret_cast<thread float const&>(uint(947912704));`, a reference cast of
an rvalue. MSL rejects it ("reinterpret_cast from rvalue to reference type"). The source is
`f32::reinterpret(113u32 << 23)` after constant folding, in `hanzo_kernel::quant::f16lo_to_f32`.
Any DSL kernel calling it failed to build on Metal (wgpu reported only "Failed to generate the
backend-specific code"; `xcrun metal -c` names the line).

**Fix.** Bind the literal to a named local first:
`const uint l_71_bits = uint(947912704); const float l_71 = reinterpret_cast<thread float const&>(l_71_bits);`
Non-constant inputs are emitted exactly as before.

**Regression gate.** `hanzo_kernel::ternary::f16_decode_probe` decodes 4096 f16 patterns
(zeros, both subnormal ends, normals, max, infinities) and compares with `half` bit-for-bit.
`pq2-check --backend <b>` runs it first on each backend, through that backend's real compiler:

| backend | compiler path | result (2026-11-18) |
|---|---|---|
| metal (dbc M4 Max) | cubecl-cpp MSL, patched branch | 0 / 4096 differ |
| cuda (dgx GB10) | cubecl-cpp CUDA, same shared emitter | 0 / 4096 differ |
| vulkan (dgx GB10) | cubecl-spirv, not affected | (SPIR-V, unaffected) |

**Upstream status.** As of tracel-ai/cubecl `main` at `68fff47` (2026-11-18), the crate has been
restructured. Its Metal `ReinterpretCastOp` (`crates/cubecl-cpp/src/metal/unary.rs`) still emits
`reinterpret_cast<const thread {ty}&>(input)` for non-pointer inputs, so the bug is probably not
fixed there, unless the new IR always gives constants a name before this point. That still needs
checking against a 0.11 build.

Issue to file upstream (not filed yet; needs owner approval to post):

> **MSL: bitcast of a constant emits a reference cast of an rvalue**
> `f32::reinterpret(C)` where `C` is a compile-time constant lowers on Metal to
> `reinterpret_cast<thread float const&>(uint(C))`, which the Metal compiler rejects
> ("reinterpret_cast from rvalue to reference type"). Repro: a `#[cube]` fn returning
> `f32::reinterpret(113u32 << 23)`, launched on wgpu<msl>. Suggested fix: emit `as_type<T>(x)`
> for MSL scalar/vector bitcasts (valid for rvalues), or bind constant inputs to a named local.
