// Build script for hanzo-ml.
//
// Vulkan backend: provides every GLSL compute kernel under
// `src/vulkan/shaders/*.comp` to the backend as `$OUT_DIR/<name>.spv`, which
// it loads via `include_bytes!(concat!(env!("OUT_DIR"), "/<name>.spv"))`.
//
// Two ways to obtain the SPIR-V, in priority order:
//   1. If `glslc` (shaderc) is available, compile each `*.comp` fresh
//      (`--target-env=vulkan1.2`). The `.comp` sources are the source of truth.
//   2. Otherwise fall back to the pre-compiled `.spv` checked in under
//      `src/vulkan/spv/<name>.spv`. This lets the crate build on hosts WITHOUT
//      a Vulkan SDK / glslc (e.g. native Windows) — only a glslc-equipped host
//      (any Linux dev box) needs to regenerate the committed `.spv` when a
//      shader changes (see `src/vulkan/regen-spv.sh`).
//
// The whole step is guarded so it never breaks a non-Vulkan build:
//   - It only runs when the `vulkan` cargo feature is enabled
//     (`CARGO_FEATURE_VULKAN` is set by cargo), AND
//   - it only runs when the shaders directory actually exists.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Always re-run if the build script itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    let vulkan_feature = std::env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let shaders_dir = Path::new("src/vulkan/shaders");
    let prebuilt_dir = Path::new("src/vulkan/spv");

    // No-op for non-vulkan builds, or if the shaders haven't been added yet.
    if !vulkan_feature || !shaders_dir.is_dir() {
        return;
    }

    // Re-run if a shader is added/removed from the directory.
    println!("cargo:rerun-if-changed={}", shaders_dir.display());
    println!("cargo:rerun-if-changed={}", prebuilt_dir.display());

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");

    let mut entries: Vec<_> = std::fs::read_dir(shaders_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", shaders_dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|e| e == "comp").unwrap_or(false))
        .collect();
    // Deterministic compile order.
    entries.sort();

    if entries.is_empty() {
        return;
    }

    // Resolve glslc once (allow override via GLSLC for unusual toolchains).
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".to_string());
    let have_glslc = Command::new(&glslc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    // Second-choice compiler: glslangValidator (the `glslang-tools` package, present on hosts
    // that ship glslang but not shaderc/glslc). `-V` emits Vulkan SPIR-V; the same `.comp`
    // sources compile under it (incl. GL_EXT_shader_explicit_arithmetic_types_float16). Only
    // probed when glslc is missing so the glslc fast path is unchanged.
    let glslang = std::env::var("GLSLANG").unwrap_or_else(|_| "glslangValidator".to_string());
    let have_glslang = !have_glslc
        && Command::new(&glslang)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

    if !have_glslc && !have_glslang {
        println!(
            "cargo:warning=neither glslc nor glslangValidator found; using pre-compiled SPIR-V from {}",
            prebuilt_dir.display()
        );
    } else if have_glslang {
        println!("cargo:warning=glslc not found; compiling shaders with glslangValidator");
    }

    for src in &entries {
        // <name>.comp -> <name>.spv
        let stem = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("invalid shader file name: {}", src.display()));
        let dst = format!("{out_dir}/{stem}.spv");
        let src_str = src.to_str().expect("non-UTF8 shader path");
        println!("cargo:rerun-if-changed={src_str}");

        if have_glslc {
            // Source of truth: compile the .comp fresh.
            let status = Command::new(&glslc)
                .args(["--target-env=vulkan1.2", src_str, "-o", &dst])
                .status()
                .unwrap_or_else(|e| panic!("failed to run `{glslc}`: {e}"));
            assert!(
                status.success(),
                "glslc failed to compile {src_str} (exit status: {status})"
            );
        } else if have_glslang {
            // Compile fresh with glslangValidator (`-V` => Vulkan SPIR-V binary).
            let status = Command::new(&glslang)
                .args(["--target-env", "vulkan1.2", "-V", src_str, "-o", &dst])
                .status()
                .unwrap_or_else(|e| panic!("failed to run `{glslang}`: {e}"));
            assert!(
                status.success(),
                "glslangValidator failed to compile {src_str} (exit status: {status})"
            );
        } else {
            // Fallback: copy the committed pre-compiled SPIR-V.
            let prebuilt: PathBuf = prebuilt_dir.join(format!("{stem}.spv"));
            println!("cargo:rerun-if-changed={}", prebuilt.display());
            std::fs::copy(&prebuilt, &dst).unwrap_or_else(|e| {
                panic!(
                    "no glslc and no pre-compiled SPIR-V at {} (run src/vulkan/regen-spv.sh on a host with glslc): {e}",
                    prebuilt.display()
                )
            });
        }
    }
}
