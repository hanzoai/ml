#!/usr/bin/env bash
# Standalone correctness + microbench harness for the unified native ROCm prefill GEMM
# (qmmq_core<WTYPE> in ../src/kernels/quant.hip). Builds ONE code object from the production kernels
# (+ two dequant-to-f16 kernels for the fallback A/B), loads it via the HIP module API (the same path
# the Rust rocm_backend uses), and runs:
#   qmmq_oracle  -- numeric gate: GPU prefill vs CPU dequant->f32 GEMM, asserts nbad==0 (Q4_K, Q6_K).
#   qmmq_bench   -- native vs dequant-f16+rocBLAS-hgemm on production FFN shapes + a decode-cost probe.
# Lives outside the Rust workspace on purpose: the ml workspace has wasm-only members that break a
# plain `cargo test -p hanzo-ml --features rocm`, so the canonical Rust oracle was relocated to
# hanzo-cli (engine). This harness tests the SAME shipped kernels with zero workspace coupling.
set -euo pipefail
cd "$(dirname "$0")"

export ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.13}
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:/opt/rocm/lib
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-11.5.1}
ARCH=${ROCM_GFX_ARCH:-gfx1151}

echo ">> building qmmq_kernels.hsaco (production kernels + fallback dequant) for $ARCH"
hipcc --genco -O3 --offload-arch="$ARCH" qmmq_bench_kernels.hip -o qmmq_kernels.hsaco

echo ">> building host programs"
hipcc -O3 --offload-arch="$ARCH" qmmq_oracle.hip -o qmmq_oracle
hipcc -O3 --offload-arch="$ARCH" qmmq_bench.hip  -lrocblas -o qmmq_bench

echo ">> NUMERIC ORACLE (must report ALL PASS / nbad=0)"
./qmmq_oracle

echo ">> MICROBENCH (native int8-WMMA prefill vs dequant-f16 + rocBLAS hgemm)"
./qmmq_bench
