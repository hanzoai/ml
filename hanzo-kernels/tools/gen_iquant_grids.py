#!/usr/bin/env python3
"""Generate the CUDA i-quant codebook header from the Rust source-of-truth.

ONE source of truth: hanzo-ml/src/quantized/iq_grids.rs (the CPU to_float oracle's
codebooks). Each GPU backend (HIP, Vulkan, CUDA) materializes the SAME values in its
own kernel language -- generated, never hand-copied. This emits the CUDA `__constant__`
tables with the `_D` suffix the ported `qdp4a<IQ*>` structs index.
"""
import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])          # iq_grids.rs
OUT = Path(sys.argv[2])          # iquant_grids.cuh

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

count = 0
for m in pat.finditer(text):
    name, rty, n, body = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    if rty not in TYPES:
        continue
    cty, suf = TYPES[rty]
    vals = [v.strip() for v in body.split(",") if v.strip()]
    assert len(vals) == n, f"{name}: token count {len(vals)} != declared {n}"
    if suf:
        vals = [v + suf for v in vals]
    emit.append(f"__device__ __constant__ {cty} {name}_D[{n}] = {{")
    for i in range(0, len(vals), 8):
        emit.append("    " + ", ".join(vals[i:i + 8]) + ",")
    emit.append("};")
    emit.append("")
    count += 1

OUT.write_text("\n".join(emit) + "\n")
print(f"emitted {count} tables -> {OUT}")
