use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // Recompile when ANY CUDA source or header changes. The statically-compiled kernels live in
    // subdirs (src/mmq_gguf, src/moe) and #include shared headers (mmq_gguf.cuh, mmq_common.cuh, ...);
    // without a rerun-if-changed for each, cargo silently reuses stale objects when only a .cu/.cuh
    // changes (build.rs itself is unchanged), so an edited kernel never recompiles.
    for pat in ["src/**/*.cu", "src/**/*.cuh"] {
        for entry in glob::glob(pat).expect("invalid glob").flatten() {
            println!("cargo::rerun-if-changed={}", entry.display());
        }
    }

    // Statically-compiled kernels. These provide `extern "C"` host launchers
    // (consumed via `hanzo-kernels/src/ffi.rs`) rather than standalone PTX
    // entry points, so they are compiled into `libmoe.a` and excluded from the
    // PTX build below.
    let static_kernels: Vec<PathBuf> = {
        let mut v = vec![
            PathBuf::from("src/moe/moe_gguf.cu"),
            PathBuf::from("src/moe/moe_wmma.cu"),
            PathBuf::from("src/moe/moe_wmma_gguf.cu"),
            PathBuf::from("src/mmvq_gguf.cu"),
            PathBuf::from("src/fattn_ds4.cu"),
        ];
        // Dense GGUF MMQ kernels (one matmul instance per quant type + quantize).
        let mut mmq: Vec<PathBuf> = glob::glob("src/mmq_gguf/*.cu")
            .expect("invalid glob")
            .map(|p| p.expect("invalid path"))
            .collect();
        mmq.sort();
        v.append(&mut mmq);
        v
    };

    // PTX kernels: every top-level `src/*.cu` that is not statically compiled.
    let ptx_kernels: Vec<PathBuf> = {
        let mut v: Vec<PathBuf> = glob::glob("src/*.cu")
            .expect("invalid glob")
            .map(|p| p.expect("invalid path"))
            .filter(|p| !static_kernels.iter().any(|s| s == p))
            .collect();
        v.sort();
        v
    };

    // Build for PTX (only the standalone kernels above).
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut builder = bindgen_cuda::Builder::default()
        .kernel_paths(ptx_kernels)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math");
    println!("cargo::warning={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let builder = builder.kernel_paths(static_kernels);
    println!("cargo::warning={builder:?}");
    builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
