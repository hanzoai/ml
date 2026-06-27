#!/usr/bin/env python3
"""Generate the CUDA i-quant codebook headers from the Rust source-of-truth.

ONE source of truth: hanzo-ml/src/quantized/iq_grids.rs (the CPU to_float oracle's
codebooks). Each GPU backend (HIP, Vulkan, CUDA) materializes the SAME values in its
own kernel language -- generated, never hand-copied. Emits TWO CUDA headers from the one
source:
  argv[2] iquant_grids.cuh     -- `__device__ __constant__ <NAME>_D[]` for the dp4a DECODE
                                   (iquant_mmvq.cu / the ported qdp4a<IQ*> structs).
  argv[3] iquant_grids_mmq.cuh -- `static const __device__ <name>[]` for the int8-WMMA MMQ
                                   PREFILL tile loaders (mmq_common.cuh / load_tiles_iq*),
                                   the IQ2/IQ3 codebooks only (lowercase ggml-cuda names).
"""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])          # iq_grids.rs
OUT = Path(sys.argv[2])          # iquant_grids.cuh
OUT_MMQ = Path(sys.argv[3]) if len(sys.argv) > 3 else None  # iquant_grids_mmq.cuh

# The IQ2/IQ3 codebooks the MMQ load_tiles_iq* read (lowercase ggml-cuda names == NAME.lower()).
# IQ1 uses a distinct GPU-packed table (iq1s_grid_gpu) not derivable here, so it is MMQ-excluded.
MMQ_GRIDS = {"IQ2XXS_GRID", "IQ2XS_GRID", "IQ2S_GRID", "IQ3XXS_GRID", "IQ3S_GRID"}

# Rust name -> (CUDA elem type, value suffix). CUDA array name = Rust name + "_D".
TYPES = {
    "u64": ("uint64_t", "ULL"),
    "u32": ("uint32_t", "U"),
    "u8":  ("uint8_t",  ""),
}

text = SRC.read_text()
# pub static NAME: [TYPE; N] = [ ... ];
pat = re.compile(
    r"pub static (\w+):\s*\[(\w+);\s*(\d+)\]\s*=\s*\[(.*?)\];",
    re.DOTALL,
)

emit = [
    "// AUTO-GENERATED from hanzo-ml/src/quantized/iq_grids.rs (the CPU to_float oracle's",
    "// codebooks). DO NOT EDIT BY HAND. Regenerate with scratchpad/gen_cuda_grids.py.",
    "// The single source of truth is the Rust grid; this is the CUDA materialization.",
    "#pragma once",
    "#include <stdint.h>",
    "",
]

emit_mmq = [
    "// AUTO-GENERATED from hanzo-ml/src/quantized/iq_grids.rs. DO NOT EDIT BY HAND.",
    "// The IQ2/IQ3 codebooks the int8-WMMA MMQ load_tiles_iq* read (ggml-cuda lowercase names).",
    "#pragma once",
    "#include <stdint.h>",
    "",
]


def rows(vals):
    return ["    " + ", ".join(vals[i:i + 8]) + "," for i in range(0, len(vals), 8)]


count = 0
mmq_count = 0
for m in pat.finditer(text):
    name, rty, n, body = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    if rty not in TYPES:
        continue
    cty, suf = TYPES[rty]
    vals = [v.strip() for v in body.split(",") if v.strip()]
    assert len(vals) == n, f"{name}: token count {len(vals)} != declared {n}"
    svals = [v + suf for v in vals] if suf else vals
    emit.append(f"__device__ __constant__ {cty} {name}_D[{n}] = {{")
    emit.extend(rows(svals))
    emit.append("};")
    emit.append("")
    count += 1
    if name in MMQ_GRIDS:
        emit_mmq.append(f"static const __device__ {cty} {name.lower()}[{n}] = {{")
        emit_mmq.extend(rows(svals))
        emit_mmq.append("};")
        emit_mmq.append("")
        mmq_count += 1

OUT.write_text("\n".join(emit) + "\n")
print(f"emitted {count} tables -> {OUT}")
if OUT_MMQ is not None:
    OUT_MMQ.write_text("\n".join(emit_mmq) + "\n")
    print(f"emitted {mmq_count} MMQ tables -> {OUT_MMQ}")
