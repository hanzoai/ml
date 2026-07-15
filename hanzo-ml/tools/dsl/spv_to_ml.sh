#!/usr/bin/env bash
# spv_to_ml.sh <raw_cubecl.spv> <out_ml.spv>
#
# Turn a raw CubeCL-emitted compute SPIR-V into one hanzo-ml's Vulkan dispatch runs unchanged.
# This is the reusable codegen seam for the kernel-DSL migration: any comptime-dims #[kernel]
# becomes a drop-in .spv for VulkanDevice::dispatch, no manual surgery.
#
# Full pipeline (per kernel):
#   1. Author the kernel in hanzo-kernel: `#[kernel(targets(...))]`, dims as #[comptime] (no runtime
#      .len()) so cubecl emits exactly the data buffers at bindings 0,1,2 (its info/scalar UBO ends
#      up unused). Runtime scalars are NOT push constants in cubecl -- they become a buffer binding;
#      for those, bind a small scalar SSBO in ml instead of using comptime dims.
#   2. Dump raw SPIR-V:  CUBECL_DEBUG_SPIRV=<dir> cargo run --release --bin <driver>   (needs the
#      cubecl `spirv-dump` feature). Produces <Kernel>_WgpuRuntime_<hash>.spv.
#   3. Run this script:  it renames the entry point to `main` (ml hardcodes .name(c"main")) and drops
#      cubecl's unused info variable from the entry interface (spirv-opt). Vulkan ignores the
#      now-non-interface binding, so ml binds only the real data buffers.
#   4. include_bytes! the result, register it in kernel_spv(), dispatch("<name>", &[bufs], &[], grid).
#
# Proven: dsl_mul (elementwise) and dsl_matvec (reduction) both dispatch through ml bit-exact
# (see vulkan_backend.rs mod dsl_dispatch_proof).
#   3b. Integer dot (dp4a) kernels: cubecl emits `OpCapability DotProduct` but never the
#      `SPV_KHR_integer_dot_product` OpExtension the capability requires, so the module is invalid
#      SPIR-V. Declare it here (idempotent, and only when the capability is actually used) rather than
#      hand-patching each kernel -- this script is the one codegen seam every dp4a .spv passes through.
set -euo pipefail
raw="$1"; out="$2"
name=$(spirv-dis "$raw" | grep -oE 'OpEntryPoint GLCompute %[A-Za-z0-9_]+ "[^"]+"' | grep -oE '"[^"]+"' | tr -d '"' | head -1)
[ -z "$name" ] && { echo "no GLCompute entry in $raw" >&2; exit 1; }
tmp=$(mktemp --suffix=.spv)
# Insert the missing OpExtension after the last OpCapability (SPIR-V requires OpExtension to follow
# OpCapability and precede OpExtInstImport), but only for modules that declare DotProduct and lack it.
spirv-dis "$raw" | sed "s/\"$name\"/\"main\"/g" | awk '
  /OpCapability DotProduct/ { dp = 1 }
  /OpExtension "SPV_KHR_integer_dot_product"/ { have = 1 }
  { line[NR] = $0; if ($0 ~ /OpCapability/) last_cap = NR }
  END {
    for (i = 1; i <= NR; i++) {
      print line[i]
      if (i == last_cap && dp && !have) print "               OpExtension \"SPV_KHR_integer_dot_product\""
    }
  }' | spirv-as --target-env vulkan1.2 -o "$tmp" -
spirv-opt --remove-unused-interface-variables --eliminate-dead-code-aggressive "$tmp" -o "$out"
rm -f "$tmp"
echo "spv_to_ml: '$name' -> 'main', cubecl info-var removed from entry interface -> $out ($(stat -c%s "$out") bytes)"
