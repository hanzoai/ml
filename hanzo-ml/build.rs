// Build script for candle-core.
//
// Vulkan backend: compiles every GLSL compute kernel under
// `src/vulkan/shaders/*.comp` to `$OUT_DIR/<name>.spv` using `glslc`
// (`--target-env=vulkan1.2`). The backend then loads them via
// `include_bytes!(concat!(env!("OUT_DIR"), "/<name>.spv"))`.
//
// This step is guarded two ways so it never breaks a non-Vulkan build:
//   1. It only runs when the `vulkan` cargo feature is enabled
//      (`CARGO_FEATURE_VULKAN` is set by cargo), AND
//   2. it only runs when the shaders directory actually exists.
// If either is missing, the build script is a no-op.

use std::path::Path;
use std::process::Command;

fn main() {
    // Always re-run if the build script itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    let vulkan_feature = std::env::var_os("CARGO_FEATURE_VULKAN").is_some();
    let shaders_dir = Path::new("src/vulkan/shaders");

    // No-op for non-vulkan builds, or if the shaders haven't been added yet.
    if !vulkan_feature || !shaders_dir.is_dir() {
        return;
    }

    // Re-run if a shader is added/removed from the directory.
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

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

    for src in &entries {
        // <name>.comp -> <name>.spv
        let stem = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("invalid shader file name: {}", src.display()));
        let spv = format!("{out_dir}/{stem}.spv");
        let src_str = src.to_str().expect("non-UTF8 shader path");

        println!("cargo:rerun-if-changed={src_str}");

        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.2", src_str, "-o", &spv])
            .status()
            .unwrap_or_else(|e| {
                panic!("failed to run `{glslc}` (install shaderc / glslc): {e}")
            });
        assert!(
            status.success(),
            "glslc failed to compile {src_str} (exit status: {status})"
        );
    }
}
