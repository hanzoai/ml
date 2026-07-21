// build.rs — FlashAttention-3 (Hopper) kernel compilation, hard arch-gated.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// FlashAttention-3 (Shah et al., arXiv:2407.08608) uses Hopper-only hardware
// (warp-specialized producer/consumer, TMA, wgmma, GEMM<->softmax pingpong), so
// its kernels are compiled ONLY for `sm_90a`.
//
// Arch gate — the single safety property of this crate:
//   * The kernels are built only when the `cuda` cargo feature is set. Without
//     it this script returns immediately, nothing is compiled or linked, and
//     `src/lib.rs` is a pure-Rust stub. Every non-datacenter target (sm_121 /
//     GB10, ROCm, Metal, Vulkan, CPU) is therefore untouched.
//   * When enabled, the kernels are cross-compiled for `sm_90a` regardless of
//     the build host's own GPU: `nvcc -gencode arch=compute_90a,code=sm_90a`
//     emits Hopper SASS on any CUDA host, so a datacenter binary can be built
//     anywhere. The host compute capability is never consulted.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

/// CUTLASS / CuTe revision the sm_90 kernels are written against. Fetched
/// on-demand (or taken from `CUTLASS_DIR`) by `hanzo-flash-attn-build`.
const CUTLASS_COMMIT: &str = "4c42f73fdab5787e3bb57717f35a8cb1b3c0dc6d";

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");

    // Arch gate: no `cuda` feature -> no kernels, crate is a stub.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return Ok(());
    }

    println!("cargo:rerun-if-env-changed=FLASH_ATTN_V3_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=FLASH_ATTN_V3_NVCC_JOBS");
    println!("cargo:rerun-if-env-changed=FLASH_ATTN_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let hkernel = manifest.join("hkernel");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").context("OUT_DIR not set")?);

    // Optional persistent cache dir for the (slow) kernel objects.
    let build_dir = match std::env::var("FLASH_ATTN_BUILD_DIR") {
        Ok(d) => PathBuf::from(d)
            .canonicalize()
            .with_context(|| "FLASH_ATTN_BUILD_DIR does not exist")?,
        Err(_) => out_dir.clone(),
    };
    std::fs::create_dir_all(&build_dir)?;

    // Kernel sources = every `.cu` in hkernel/ (flash_api.cu + the instantiation
    // files). Globbing keeps this in lockstep with the directory: adding an
    // hdim/dtype means dropping in the `.cu`, not editing a list here.
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut newest_header = std::time::UNIX_EPOCH;
    for entry in std::fs::read_dir(&hkernel)? {
        let path = entry?.path();
        match path.extension().and_then(|e| e.to_str()) {
            Some("cu") => {
                println!(
                    "cargo:rerun-if-changed=hkernel/{}",
                    path.file_name().unwrap().to_string_lossy()
                );
                sources.push(path);
            }
            Some("h") | Some("hpp") | Some("cuh") => {
                println!(
                    "cargo:rerun-if-changed=hkernel/{}",
                    path.file_name().unwrap().to_string_lossy()
                );
                newest_header = newest_header.max(mtime(&path));
            }
            _ => {}
        }
    }
    sources.sort();
    if sources.is_empty() {
        bail!("no .cu kernel sources found in {}", hkernel.display());
    }

    // CUTLASS include tree (env `CUTLASS_DIR`, cache, or download).
    let cutlass = hanzo_flash_attn_build::fetch_cutlass(&out_dir, CUTLASS_COMMIT)?;
    let cutlass_inc = hanzo_flash_attn_build::cutlass_include_arg(&cutlass);

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    // Datacenter arch. `90a` (Hopper) is the only architecture these kernels
    // support; overridable for future datacenter parts once ported.
    let arch = std::env::var("FLASH_ATTN_V3_CUDA_ARCH").unwrap_or_else(|_| "90a".to_string());
    let gencode = format!("arch=compute_{arch},code=sm_{arch}");
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_msvc = target.contains("msvc");
    let hkernel_inc = format!("-I{}", hkernel.display());

    let jobs: usize = std::env::var("FLASH_ATTN_V3_NVCC_JOBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()
        .context("failed to build nvcc thread pool")?;

    let objects: Result<Vec<PathBuf>> = pool.install(|| {
        sources
            .par_iter()
            .map(|src| {
                let stem = src.file_name().unwrap().to_string_lossy();
                let obj = build_dir.join(format!("{stem}.sm{arch}.o"));
                // Skip when a cached object is newer than its source and every header.
                if obj.exists() && mtime(&obj) >= mtime(src).max(newest_header) {
                    return Ok(obj);
                }
                let mut cmd = Command::new(&nvcc);
                cmd.arg("-c")
                    .arg(src)
                    .arg("-o")
                    .arg(&obj)
                    // Force-include the CUDA-13 compat shim before any CUTLASS header.
                    .arg("-include")
                    .arg(hkernel.join("cuda_compat.h"))
                    .arg("-gencode")
                    .arg(&gencode)
                    .arg("-std=c++17")
                    .arg("-O3")
                    .arg(&cutlass_inc)
                    .arg(&hkernel_inc)
                    .arg("-U__CUDA_NO_HALF_OPERATORS__")
                    .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
                    .arg("-U__CUDA_NO_BFLOAT16_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
                    .arg("-U__CUDA_NO_BFLOAT162_OPERATORS__")
                    .arg("-U__CUDA_NO_BFLOAT162_CONVERSIONS__")
                    .arg("-D_USE_MATH_DEFINES")
                    .args(["--default-stream", "per-thread"])
                    .arg("--expt-relaxed-constexpr")
                    .arg("--expt-extended-lambda")
                    .arg("--use_fast_math");
                if !is_msvc {
                    cmd.arg("-Xcompiler").arg("-fPIC");
                }
                let status = cmd
                    .status()
                    .with_context(|| format!("failed to spawn nvcc for {stem}"))?;
                if !status.success() {
                    bail!("nvcc failed to compile {stem} for sm_{arch}");
                }
                Ok(obj)
            })
            .collect()
    });
    let objects = objects?;

    // Archive the objects into one static lib.
    let lib = build_dir.join("libflashattentionv3.a");
    let _ = std::fs::remove_file(&lib);
    let mut ar = Command::new(std::env::var("AR").unwrap_or_else(|_| "ar".to_string()));
    ar.arg("crus").arg(&lib).args(&objects);
    if !ar.status().context("failed to spawn ar")?.success() {
        bail!("ar failed to archive FA3 objects");
    }

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=flashattentionv3");
    // The kernels reference the CUDA runtime (cudaFuncSetAttribute, cudaGetDevice…).
    if let Some(libdir) = cuda_lib_dir(&nvcc) {
        println!("cargo:rustc-link-search=native={}", libdir.display());
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_msvc {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}

fn mtime(p: &Path) -> std::time::SystemTime {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::UNIX_EPOCH)
}

/// Resolve the CUDA runtime library directory from `CUDA_HOME`/`CUDA_PATH`, else
/// from the nvcc location (`<toolkit>/bin/nvcc` -> `<toolkit>/lib64`).
fn cuda_lib_dir(nvcc: &str) -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let lib = PathBuf::from(root).join("lib64");
            if lib.is_dir() {
                return Some(lib);
            }
        }
    }
    let nvcc_path = which(nvcc)?;
    let lib = nvcc_path.parent()?.parent()?.join("lib64");
    lib.is_dir().then_some(lib)
}

fn which(bin: &str) -> Option<PathBuf> {
    let p = PathBuf::from(bin);
    if p.is_absolute() {
        return Some(p);
    }
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(bin))
            .find(|c| c.is_file())
    })
}
