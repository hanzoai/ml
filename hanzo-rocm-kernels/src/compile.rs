use crate::error::KernelError;
use crate::wrappers::SendSyncModule;
use rocm_rs::hip::Device;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

/// Single unified cache for compiled kernel modules.
///
/// Combines the functionality of the old CacheManager (disk cache)
/// and KernelManager (module cache) into one simpler struct.
pub struct KernelCache {
    cache_dir: PathBuf,
    arch: String,
    rocm_version: String,
    modules: Mutex<HashMap<&'static str, Arc<SendSyncModule>>>,
    /// Resolved kernel function handles (hipFunction_t as usize), keyed by func name.
    /// hipModuleGetFunction is a slow driver round-trip through the WSL bridge (~90us),
    /// so resolving once per function (not per op) is a major decode speedup. The handle
    /// stays valid because its module is held forever in `modules`.
    functions: Mutex<HashMap<String, usize>>,
}

impl KernelCache {
    /// Create a new KernelCache for the given device
    pub fn new(device: &Device) -> Result<Self, KernelError> {
        let arch = detect_gpu_arch(device)?;
        let rocm_version = detect_rocm_version()?;
        let cache_dir = get_cache_dir()?;

        // Create cache directory structure: ~/.cache/hanzo-ml-rocm/{arch}-{rocm_version}/
        let arch_version = format!("{}-{}", arch, rocm_version);
        let kernel_dir = cache_dir.join(&arch_version);
        fs::create_dir_all(&kernel_dir).map_err(|e| {
            KernelError::Io(format!(
                "Failed to create cache directory {}: {}",
                kernel_dir.display(),
                e
            ))
        })?;

        Ok(Self {
            cache_dir: kernel_dir,
            arch,
            rocm_version,
            modules: Mutex::new(HashMap::new()),
            functions: Mutex::new(HashMap::new()),
        })
    }

    /// Get a resolved kernel function handle (hipFunction_t as usize), caching it so
    /// hipModuleGetFunction runs once per function instead of once per op.
    pub fn get_func_raw(
        &self,
        module_name: &'static str,
        source: &'static str,
        func_name: &str,
    ) -> Result<usize, KernelError> {
        {
            let funcs = self
                .functions
                .lock()
                .map_err(|_| KernelError::Internal("Failed to lock functions cache".to_string()))?;
            if let Some(&ptr) = funcs.get(func_name) {
                return Ok(ptr);
            }
        }
        let module = self.get_or_load(module_name, source)?;
        let func = module.get_function(func_name).map_err(|e| {
            KernelError::Compilation(format!("Kernel function {} not found: {}", func_name, e))
        })?;
        let raw = func.as_raw() as usize;
        self.functions
            .lock()
            .map_err(|_| KernelError::Internal("Failed to lock functions cache".to_string()))?
            .insert(func_name.to_string(), raw);
        Ok(raw)
    }

    /// Get or compile a kernel module.
    ///
    /// This method checks the in-memory cache first, then the disk cache,
    /// and compiles from source if needed.
    pub fn get_or_load(
        &self,
        name: &'static str,
        source: &'static str,
    ) -> Result<Arc<SendSyncModule>, KernelError> {
        // Check in-memory cache first
        {
            let modules = self
                .modules
                .lock()
                .map_err(|_| KernelError::Internal("Failed to lock modules cache".to_string()))?;
            if let Some(module) = modules.get(name) {
                return Ok(module.clone());
            }
        }

        // Compute hash of source to version the cache
        let source_hash = compute_source_hash(source);
        let cache_file = self.cache_dir.join(format!("{}_{}.cso", name, source_hash));

        // Try to load from disk cache or compile
        let binary = if cache_file.exists() {
            fs::read(&cache_file).map_err(|e| {
                KernelError::Io(format!(
                    "Failed to read cached binary {}: {}",
                    cache_file.display(),
                    e
                ))
            })?
        } else {
            let binary = compile_kernel(name, source, &self.arch, &cache_file)?;
            fs::write(&cache_file, &binary).map_err(|e| {
                KernelError::Io(format!(
                    "Failed to write cache file {}: {}",
                    cache_file.display(),
                    e
                ))
            })?;
            binary
        };

        // Load module from binary
        let module = SendSyncModule::load_data(&binary).map_err(|e| {
            KernelError::Compilation(format!(
                "Failed to load module {} from compiled binary: {}",
                name, e
            ))
        })?;

        let module = Arc::new(module);

        // Store in memory cache
        {
            let mut modules = self
                .modules
                .lock()
                .map_err(|_| KernelError::Internal("Failed to lock modules cache".to_string()))?;
            modules.insert(name, module.clone());
        }

        Ok(module)
    }

    /// Get the cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Get GPU architecture
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Get ROCm version
    pub fn rocm_version(&self) -> &str {
        &self.rocm_version
    }
}

/// Detect the GPU architecture (e.g., "gfx908", "gfx90a", "gfx942")
fn detect_gpu_arch(_device: &Device) -> Result<String, KernelError> {
    // First try environment variables (useful for build machines and for Windows, where
    // rocminfo is typically absent). ROCM_GFX_ARCH matches hanzo-engine's build.rs.
    for var in ["CANDLE_ROCM_ARCH", "ROCM_GFX_ARCH"] {
        if let Ok(arch) = std::env::var(var) {
            let arch = arch.trim();
            if !arch.is_empty() {
                return Ok(arch.to_string());
            }
        }
    }

    // Try to use rocminfo to detect the architecture
    match Command::new("rocminfo").arg("-a").output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Look for "Name:" line with gfxXXXX
            for line in stdout.lines() {
                if line.contains("Name:") && line.contains("gfx") {
                    if let Some(start) = line.find("gfx") {
                        let arch = &line[start..];
                        // Extract just the gfxXXXX part
                        let end = arch
                            .find(|c: char| !c.is_alphanumeric())
                            .unwrap_or(arch.len());
                        return Ok(arch[..end].to_string());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to run rocminfo: {}", e);
        }
    }

    // Try hipcc to get default arch
    match Command::new("hipcc").args(&["--version"]).output() {
        Ok(_) => {
            eprintln!("Warning: Could not detect GPU architecture, defaulting to gfx1151 (set ROCM_GFX_ARCH to override)");
            Ok("gfx1151".to_string())
        }
        Err(e) => Err(KernelError::Compilation(format!(
            "hipcc not found: {}. Please install ROCm or set CANDLE_ROCM_ARCH environment variable",
            e
        ))),
    }
}

/// Detect ROCm version
fn detect_rocm_version() -> Result<String, KernelError> {
    // Try to get from environment variable first
    if let Ok(version) = std::env::var("CANDLE_ROCM_VERSION") {
        return Ok(version);
    }

    // Try to get from hipcc --version
    match Command::new("hipcc").args(&["--version"]).output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse version from output like "HIP version: 6.1.0"
            for line in stdout.lines() {
                if line.contains("HIP version:") || line.contains("HIP_VERSION:") {
                    if let Some(v) = line.split(':').nth(1) {
                        let version = v.trim().split('.').take(2).collect::<Vec<_>>().join(".");
                        return Ok(version);
                    }
                }
            }
            // If we can't parse, return a default
            Ok("6.0".to_string())
        }
        Err(e) => Err(KernelError::Compilation(format!(
            "hipcc not found: {}. Please install ROCm or set CANDLE_ROCM_VERSION environment variable",
            e
        ))),
    }
}

/// Get the base cache directory
fn get_cache_dir() -> Result<PathBuf, KernelError> {
    let home = dirs::cache_dir()
        .or_else(|| std::env::var("HOME").ok().map(PathBuf::from))
        .ok_or_else(|| KernelError::Internal("Could not determine cache directory".to_string()))?;

    Ok(home.join("hanzo-ml-rocm"))
}

/// Compute a hash of the source code
fn compute_source_hash(source: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(source.as_bytes());
    let result = hasher.finalize();
    // Use first 16 characters of hex as hash
    format!("{:x}", result)[..16].to_string()
}

/// Compile a kernel using hipcc
fn compile_kernel(
    name: &str,
    source: &str,
    arch: &str,
    output_path: &Path,
) -> Result<Vec<u8>, KernelError> {
    let temp_dir = std::env::temp_dir();
    let source_hash = compute_source_hash(source);
    let source_file = temp_dir.join(format!("hanzo_{}_{}.hip", name, source_hash));
    let obj_file = temp_dir.join(format!("hanzo_{}_{}.o", name, source_hash));
    let fatbin_file = temp_dir.join(format!("hanzo_{}_{}.fatbin", name, source_hash));
    let hsaco_file = temp_dir.join(format!("hanzo_{}_{}.hsaco", name, source_hash));

    // Clean up temp files on any error
    let _cleanup = TempFileCleanup {
        files: vec![
            source_file.clone(),
            obj_file.clone(),
            fatbin_file.clone(),
            hsaco_file.clone(),
        ],
    };

    fs::write(&source_file, source).map_err(|e| {
        KernelError::Io(format!(
            "Failed to write source file {}: {}",
            source_file.display(),
            e
        ))
    })?;

    // Compile HIP straight to a code-object bundle with `hipcc --genco`. This avoids the
    // Linux-only `objcopy -j .hip_fatbin` extraction (Windows COFF objects don't yield a
    // bundler-parseable section); --genco emits a __CLANG_OFFLOAD_BUNDLE__ that the
    // offload-bundler below unbundles for the gfx target. `-fPIC` is omitted (the
    // windows-msvc clang target rejects it; code is position-independent there by default).
    let _ = &obj_file; // no longer needed (was the `-c` object output)
    let offload_arg = format!("--offload-arch={}", arch);
    let fatbin_str = fatbin_file.to_str().unwrap();
    let src_str = source_file.to_str().unwrap();
    let mut hipcc_args: Vec<&str> = vec![offload_arg.as_str(), "-O3", "--genco"];
    if !cfg!(target_os = "windows") {
        hipcc_args.push("-fPIC");
    }
    hipcc_args.extend_from_slice(&["-o", fatbin_str, src_str]);
    let output = Command::new("hipcc")
        .args(&hipcc_args)
        .output()
        .map_err(|e| {
            KernelError::Compilation(format!("Failed to execute hipcc: {}. Is hipcc in PATH?", e))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(KernelError::Compilation(format!(
            "hipcc --genco failed for {}:\n{}",
            name, stderr
        )));
    }

    // Step 3: Unbundle the code object for specific architecture
    let target = format!("hipv4-amdgcn-amd-amdhsa--{}", arch);
    let bundler_path = find_rocm_tool("clang-offload-bundler")?;
    let unbundle_output = Command::new(&bundler_path)
        .args(&[
            "--unbundle",
            "--type=o",
            "--input",
            fatbin_file.to_str().unwrap(),
            "--targets",
            &target,
            "--output",
            hsaco_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| {
            KernelError::Compilation(format!(
                "Failed to execute clang-offload-bundler: {}. Is ROCm in PATH?",
                e
            ))
        })?;

    if !unbundle_output.status.success() {
        let stderr = String::from_utf8_lossy(&unbundle_output.stderr);
        return Err(KernelError::Compilation(format!(
            "clang-offload-bundler extraction failed for {}:\n{}",
            name, stderr
        )));
    }

    // Read the final code object
    let binary = fs::read(&hsaco_file).map_err(|e| {
        KernelError::Io(format!(
            "Failed to read code object {}: {}",
            hsaco_file.display(),
            e
        ))
    })?;

    // Write to cache location
    fs::write(output_path, &binary).map_err(|e| {
        KernelError::Io(format!(
            "Failed to write cache file {}: {}",
            output_path.display(),
            e
        ))
    })?;

    Ok(binary)
}

/// Find an ROCm tool using hipcc
fn find_rocm_tool(tool_name: &str) -> Result<String, KernelError> {
    // ROCm tools live in ROCM_PATH/bin; hipcc's --print-prog-name does not resolve all of
    // them (especially on Windows). Prefer the explicit path, then hipcc, then PATH.
    if let Ok(rocm) = std::env::var("ROCM_PATH") {
        for cand in [
            format!("{rocm}/bin/{tool_name}.exe"),
            format!("{rocm}/bin/{tool_name}"),
        ] {
            if PathBuf::from(&cand).exists() {
                return Ok(cand);
            }
        }
    }
    let output = Command::new("hipcc")
        .args(&["--print-prog-name", tool_name])
        .output()
        .map_err(|e| KernelError::Compilation(format!("Failed to run hipcc: {}", e)))?;
    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() && PathBuf::from(&path).exists() {
            return Ok(path);
        }
    }
    // Last resort: rely on PATH (ROCm bin is typically on PATH).
    Ok(tool_name.to_string())
}

/// Helper struct to clean up temporary files
struct TempFileCleanup {
    files: Vec<PathBuf>,
}

impl Drop for TempFileCleanup {
    fn drop(&mut self) {
        for file in &self.files {
            let _ = fs::remove_file(file);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_hash() {
        let source1 = "__global__ void test() {}";
        let source2 = "__global__ void test() {}";
        let source3 = "__global__ void test2() {}";

        let hash1 = compute_source_hash(source1);
        let hash2 = compute_source_hash(source2);
        let hash3 = compute_source_hash(source3);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
