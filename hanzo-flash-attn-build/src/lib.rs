//! Build-time helper for the hanzo flash-attention CUDA kernels.
//!
//! The flash-attention kernels are templated on top of NVIDIA CUTLASS / CuTe.
//! CUTLASS is a large header-only library that we do **not** want to vendor into
//! this repository, so instead we fetch the headers on demand at a pinned commit
//! and hand the resulting include path back to the kernel compiler.
//!
//! Two entry points are exposed and consumed by `hanzo-flash-attn/build.rs`:
//!
//! * [`fetch_cutlass`] resolves the CUTLASS source tree for a pinned commit and
//!   returns the path to its root directory (the directory that contains
//!   `include/cutlass/cutlass.h`).
//! * [`cutlass_include_arg`] turns that root directory into the `-I<...>` nvcc
//!   include flag(s) needed to compile against CUTLASS.
//!
//! Resolution order for `fetch_cutlass`:
//! 1. `CUTLASS_DIR` env var, if set and pointing at a usable CUTLASS checkout —
//!    used verbatim (lets CI / offline builds pre-stage the headers).
//! 2. A previously downloaded copy under `<out_dir>/cutlass-<commit>` (cache).
//! 3. Download `https://github.com/NVIDIA/cutlass/archive/<commit>.tar.gz`,
//!    extract it under `out_dir`, and use the extracted tree.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};

/// Upstream CUTLASS repository. GitHub serves an immutable source archive for
/// any commit-ish at `<repo>/archive/<commit>.tar.gz`.
const CUTLASS_REPO: &str = "https://github.com/NVIDIA/cutlass";

/// Resolve the CUTLASS source tree pinned at `commit` and return its root
/// directory (the directory containing `include/cutlass/cutlass.h`).
///
/// `out_dir` is the cargo `OUT_DIR` of the crate driving the build; downloaded
/// and extracted artifacts are cached there so repeated builds are cheap.
pub fn fetch_cutlass(out_dir: &Path, commit: &str) -> Result<PathBuf> {
    // 1. Honor an explicit, pre-staged checkout. This is the escape hatch for
    //    air-gapped / CI builds that ship their own CUTLASS.
    if let Ok(dir) = std::env::var("CUTLASS_DIR") {
        let dir = PathBuf::from(dir);
        if is_valid_cutlass(&dir) {
            // Recompile if the operator points us at a different tree.
            println!("cargo::rerun-if-env-changed=CUTLASS_DIR");
            return Ok(dir);
        }
        bail!(
            "CUTLASS_DIR is set to {} but that directory does not contain \
             include/cutlass/cutlass.h",
            dir.display()
        );
    }
    println!("cargo::rerun-if-env-changed=CUTLASS_DIR");

    // GitHub's source archive for a commit extracts to `cutlass-<commit>/`.
    let extracted_root = out_dir.join(format!("cutlass-{commit}"));

    // 2. Cache hit: a complete, previously extracted tree.
    if is_valid_cutlass(&extracted_root) {
        return Ok(extracted_root);
    }

    // Stale / partial extraction from a previous interrupted build: clear it so
    // we re-extract cleanly.
    if extracted_root.exists() {
        fs::remove_dir_all(&extracted_root).with_context(|| {
            format!(
                "failed to remove stale CUTLASS directory {}",
                extracted_root.display()
            )
        })?;
    }

    // 3. Download + extract.
    fs::create_dir_all(out_dir)
        .with_context(|| format!("failed to create out dir {}", out_dir.display()))?;
    download_and_extract(out_dir, commit)?;

    if !is_valid_cutlass(&extracted_root) {
        bail!(
            "CUTLASS was downloaded and extracted to {} but the expected header \
             include/cutlass/cutlass.h is missing",
            extracted_root.display()
        );
    }

    Ok(extracted_root)
}

/// Build the nvcc include flag for a resolved CUTLASS root directory.
///
/// CUTLASS ships both its core API headers (`cutlass/...`) and the CuTe headers
/// (`cute/...`) under a single `include/` directory, and that is everything the
/// sm80 flash-attention forward kernels include. The returned value is exactly
/// **one** nvcc argument token, e.g. `-I/abs/path/cutlass-<commit>/include`.
///
/// A single token matters: the caller forwards this straight to
/// `bindgen_cuda::Builder::arg`, which pushes it as one element of the nvcc
/// argv. Joining several `-I` flags with spaces here would produce a single
/// malformed include path, so only the one required include root is returned.
pub fn cutlass_include_arg(cutlass_dir: &Path) -> String {
    let include = cutlass_dir.join("include");
    format!("-I{}", include.display())
}

/// A directory is a usable CUTLASS checkout iff its primary public header is
/// present. This is cheap and robust against partial downloads.
fn is_valid_cutlass(dir: &Path) -> bool {
    dir.join("include").join("cutlass").join("cutlass.h").is_file()
}

/// Download the pinned CUTLASS source archive from GitHub and extract it into
/// `out_dir`. GitHub redirects `archive/<commit>.tar.gz` to a `codeload`
/// host, so redirects must be followed (ureq does so by default).
fn download_and_extract(out_dir: &Path, commit: &str) -> Result<()> {
    let url = format!("{CUTLASS_REPO}/archive/{commit}.tar.gz");
    println!("cargo:warning=fetching CUTLASS {commit} from {url}");

    // Stream the gzipped tarball straight into the gzip+tar decoders so we never
    // hold the whole (tens of MB) archive in memory. Generous timeouts cover the
    // large transfer over slow CI links.
    let resp = ureq::builder()
        .timeout_connect(Duration::from_secs(30))
        .timeout(Duration::from_secs(600))
        .redirects(10)
        .build()
        .get(&url)
        .call()
        .with_context(|| format!("failed to download CUTLASS archive from {url}"))?;

    let reader = resp.into_reader();
    let gz = flate2::read::GzDecoder::new(reader);
    let mut archive = tar::Archive::new(gz);

    // Extract underneath out_dir; the archive's own top-level directory is
    // `cutlass-<commit>/`, matching the path `fetch_cutlass` expects.
    archive
        .unpack(out_dir)
        .with_context(|| format!("failed to extract CUTLASS archive into {}", out_dir.display()))?;

    Ok(())
}

/// Read an entire reader into a byte vector. Retained as a small utility used by
/// the integration test below; kept out of the hot download path which streams.
#[allow(dead_code)]
fn read_all(mut r: impl Read) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).context("failed to read response body")?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn include_arg_includes_core_include_dir() {
        let tmp = std::env::temp_dir().join("hanzo_cutlass_test_root");
        let inc = tmp.join("include").join("cutlass");
        fs::create_dir_all(&inc).unwrap();
        let arg = cutlass_include_arg(&tmp);
        assert!(arg.contains("-I"));
        assert!(arg.contains(&tmp.join("include").display().to_string()));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn include_arg_is_single_token() {
        let tmp = std::env::temp_dir().join("hanzo_cutlass_test_single");
        fs::create_dir_all(tmp.join("include")).unwrap();
        let arg = cutlass_include_arg(&tmp);
        // Exactly one -I flag and no embedded spaces -> one argv token for nvcc.
        assert_eq!(arg.matches("-I").count(), 1);
        assert!(!arg.contains(' '));
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn validity_requires_primary_header() {
        let tmp = std::env::temp_dir().join("hanzo_cutlass_test_valid");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(tmp.join("include")).unwrap();
        assert!(!is_valid_cutlass(&tmp));
        fs::create_dir_all(tmp.join("include").join("cutlass")).unwrap();
        fs::write(tmp.join("include").join("cutlass").join("cutlass.h"), b"// hdr").unwrap();
        assert!(is_valid_cutlass(&tmp));
        let _ = fs::remove_dir_all(&tmp);
    }
}
