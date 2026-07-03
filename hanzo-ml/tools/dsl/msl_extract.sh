#!/usr/bin/env bash
# Metal analog of spv_to_ml.sh: extract the cubecl-generated MSL for a #[kernel], for dispatch
# through ml's Metal backend. The prior scoping ("no MSL dump in cubecl 0.10") was WRONG --
# cubecl's Metal path is cubecl-cpp's MslCompiler (direct MSL source), and CUBECL_DEBUG_LOG dumps it.
#
# PIECE 1 -- EXTRACTION (this script). Run any DSL #[kernel] under the metal feature with
# CUBECL_DEBUG_LOG=1; cubecl writes the generated MSL source to /tmp/cubecl.log. Each kernel is a
# block: `#include <metal_stdlib>` ... `struct info_st { uint static_meta[N]; };` ... `[[kernel]] void
# <name>(...)`. Build with `--no-default-features --features metal` (drop cpu: LLVM-bundler blocker).
#
#   usage: CUBECL_DEBUG_LOG=1 <run the kernel>; ./msl_extract.sh <entry_name_substr>
#
# PIECE 2 -- DISPATCH PRIMITIVE (to build in metal_backend, mirrors the ug DSL at device.rs:112 +
# custom_op.rs:683). cubecl MSL binding convention (see matvec.metal.reference):
#   - data buffers at [[buffer(0..N-1)]] (const device T* / device T*)
#   - `constant info_st& info [[buffer(N)]]` -- metadata: uint static_meta[K] holds array lengths;
#     kernels read e.g. info.static_meta[7] for a bounds guard. ml must populate this small uint
#     buffer (the Metal analog of Vulkan's info buffer; on Vulkan it was stripped when unused, on
#     Metal it is `constant`, so populate it from the buffer element-counts).
#   - entry is cubecl-named (e.g. matvec_q8_f_f32), NOT "main" -- Metal get_function takes the name
#     directly, so NO rename step is needed (unlike the Vulkan seam).
#   - thread built-ins: [[thread_position_in_grid]] / [[threads_per_threadgroup]] /
#     [[threadgroups_per_grid]] -> dispatch_thread_groups(grid, threadgroup).
# Primitive: new_library_with_source(msl) -> get_function(entry) -> pipeline; encoder
# set_compute_pipeline_state; set_buffer(i, data_i, 0) for 0..N; set_buffer(N, info_buf, 0);
# dispatch_thread_groups. The dispatch_out(out_idx) lesson from Vulkan applies: `out` is not the last
# binding (info trails it), so track the RAW hazard on the real data-output index, not the last buffer.
#
# STATUS: piece 1 solved + verified (matvec MSL bit-exact-runs standalone on M4 Max, 2.4e-7). Piece 2
# is a straight mirror of the existing ug Metal dispatch; the only new logic is populating info_st.
set -euo pipefail
LOG="${CUBECL_LOG:-/tmp/cubecl.log}"; NAME="${1:-}"
[ -f "$LOG" ] || { echo "no $LOG -- run the kernel with CUBECL_DEBUG_LOG=1 first" >&2; exit 1; }
awk -v n="$NAME" '/#include <metal_stdlib>/{buf=""} {buf=buf $0 "\n"} /\[\[kernel\]\]/{k=1}
  k&&/^}$/{if(n=="" || buf ~ ("void [a-z_]*" n)) {printf "%s", buf; exit} k=0}' "$LOG"
