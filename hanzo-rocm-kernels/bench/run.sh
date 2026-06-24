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

echo ">> A/B DECODE: pre-SWAR (byte loop) vs shipped SWAR decode, thermally interleaved"
GITROOT=$(git -C . rev-parse --show-toplevel 2>/dev/null || echo ..)
BASE_REV=$(git -C "$GITROOT" log -1 --format=%H --skip=1 -- hanzo-rocm-kernels/src/kernels/quant.hip 2>/dev/null || true)
if [ -n "$BASE_REV" ]; then
  git -C "$GITROOT" show "$BASE_REV:hanzo-rocm-kernels/src/kernels/quant.hip" > /tmp/qmmq_quant_base.hip
  hipcc --genco -O3 --offload-arch="$ARCH" /tmp/qmmq_quant_base.hip -o k_old.hsaco
  hipcc --genco -O3 --offload-arch="$ARCH" ../src/kernels/quant.hip      -o k_new.hsaco
  hipcc -O3 --offload-arch="$ARCH" qmmq_ab_decode.hip -o qmmq_ab_decode
  ./qmmq_ab_decode 512 5120 8192
  ./qmmq_ab_decode 512 4096 12288   # FFN gate/up
  ./qmmq_ab_decode 512 12288 4096   # FFN down
else
  echo "  (skipped: could not resolve baseline quant.hip revision from git)"
fi

echo ">> WMMA PEAK: raw iu8 vs f16 matrix-core ceiling (proves iu8 == f16 on gfx1151, ~1.0x)"
hipcc -O3 --offload-arch="$ARCH" qmmq_wmma_peak.hip -o qmmq_wmma_peak
./qmmq_wmma_peak

if command -v rocprofv3 >/dev/null 2>&1; then
  echo ">> rocprofv3 on Q8_0 qmmq (clean MMA machine): LDS/VALU/occupancy/bank-conflict counters"
  hipcc --genco -O3 --offload-arch="$ARCH" ../src/kernels/quant.hip -o qmmq_kernels.hsaco
  hipcc -O3 --offload-arch="$ARCH" qmmq_prof.hip -o qmmq_prof
  rm -rf prof_out
  rocprofv3 --pmc SQ_INSTS_VALU SQ_INSTS_LDS LDSBankConflict OccupancyPercent MemUnitBusy GRBM_GUI_ACTIVE \
    -d prof_out -o q8 -f csv -- ./qmmq_prof q8_0 40 qmmq_kernels.hsaco >/dev/null 2>&1 || true
  if [ -f prof_out/q8_counter_collection.csv ]; then
    python3 - <<'PY'
import csv,collections
agg=collections.defaultdict(list)
for r in csv.DictReader(open('prof_out/q8_counter_collection.csv')):
    if 'qmmq' in r.get('Kernel_Name',''): agg[r['Counter_Name']].append(float(r['Counter_Value']))
for c,v in sorted(agg.items()): print(f"  {c:18s} mean={sum(v)/len(v):.4g}")
PY
  fi
fi
