// CUDA 13 build-compat shim (force-included via nvcc `-include`).
//
// CUTLASS 3.5's cuda_host_adapter.hpp references the driver typedefs
// `PFN_cuTensorMapEncodeTiled` / `PFN_cuTensorMapEncodeIm2col`. CUDA 12 defined
// those unversioned names; CUDA 13 ships only the versioned `_v12000` forms
// (cudaTypedefs.h) and drops the unversioned aliases, so the header fails to
// compile. Re-introduce the unversioned aliases when building against CUDA 13+.
//
// This touches no kernel and no CUTLASS source — it only restores a typedef
// name the toolkit renamed, so the FA3 kernels compile unchanged under nvcc 13.
#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
#if !defined(PFN_cuTensorMapEncodeTiled)
using PFN_cuTensorMapEncodeTiled = PFN_cuTensorMapEncodeTiled_v12000;
#endif
#if !defined(PFN_cuTensorMapEncodeIm2col)
using PFN_cuTensorMapEncodeIm2col = PFN_cuTensorMapEncodeIm2col_v12000;
#endif
#endif
