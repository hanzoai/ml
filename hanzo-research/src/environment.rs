//! What a binary was built from ([`Build`], stamped by [`stamp`] in its build script) and where
//! it runs ([`Environment`], measured, HIP-1334 §6.1). A field that cannot be measured is empty
//! or `None`; nothing is filled in.

use crate::canonical;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The source a binary was compiled from, stamped at build time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub name: String,
    pub version: String,
    /// The commit, 40 hex; empty when built outside a git checkout.
    pub git_sha: String,
    /// The tree differed from the commit, or there was no commit to compare with.
    pub dirty: bool,
    /// RFC 3339 UTC: `SOURCE_DATE_EPOCH` when set, else the build's clock.
    pub built: String,
    /// `sha256:` of the workspace's `Cargo.lock`.
    pub lockfile_hash: String,
    /// The hanzoai/ml commit the lockfile pins, empty when it pins none.
    pub ml: String,
    pub profile: String,
    pub flags: Vec<String>,
}

impl Build {
    /// One line for `--version` and logs.
    pub fn line(&self) -> String {
        format!(
            "{} {} (git {}{}, built {}, lock {}, ml {}, {})",
            self.name,
            self.version,
            if self.git_sha.is_empty() {
                "none"
            } else {
                &self.git_sha
            },
            if self.dirty { " dirty" } else { "" },
            self.built,
            self.lockfile_hash,
            if self.ml.is_empty() { "none" } else { &self.ml },
            self.profile,
        )
    }
}

/// This crate's [`Build`], from what [`stamp`] set when it was compiled.
#[macro_export]
macro_rules! build {
    () => {
        $crate::environment::Build {
            name: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            git_sha: env!("HANZO_BUILD_SHA").into(),
            dirty: env!("HANZO_BUILD_DIRTY") == "true",
            built: env!("HANZO_BUILD_TIME").into(),
            lockfile_hash: env!("HANZO_BUILD_LOCK").into(),
            ml: env!("HANZO_BUILD_ML").into(),
            profile: env!("HANZO_BUILD_PROFILE").into(),
            flags: env!("HANZO_BUILD_FLAGS")
                .split(' ')
                .filter(|f| !f.is_empty())
                .map(String::from)
                .collect(),
        }
    };
}

/// This crate's version and stamp as one `&'static str`, for clap's `version`.
#[macro_export]
macro_rules! version {
    () => {
        concat!(
            env!("CARGO_PKG_VERSION"),
            " (git ",
            env!("HANZO_BUILD_SHA"),
            ", dirty ",
            env!("HANZO_BUILD_DIRTY"),
            ", built ",
            env!("HANZO_BUILD_TIME"),
            ", lock ",
            env!("HANZO_BUILD_LOCK"),
            ")"
        )
    };
}

/// For a build script: stamp the crate with the checkout it is compiled from, and rerun when a
/// tracked file, the index or the commit changes.
pub fn stamp() {
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let d = dir.to_string_lossy().to_string();
    let top = git(&d, &["rev-parse", "--show-toplevel"]);
    let sha = git(&d, &["rev-parse", "HEAD"]);
    let status = Command::new("git")
        .args(["-C", &d, "status", "--porcelain"])
        .output();
    let dirty =
        sha.is_empty() || !matches!(&status, Ok(o) if o.status.success() && o.stdout.is_empty());
    let lock = dir
        .ancestors()
        .map(|a| a.join("Cargo.lock"))
        .find(|p| p.exists());
    let (lock_hash, ml) = match &lock {
        Some(p) => {
            let bytes = std::fs::read(p).expect("Cargo.lock");
            println!("cargo:rerun-if-changed={}", p.display());
            (
                format!("sha256:{}", canonical::sha256(&bytes)),
                ml_commit(&String::from_utf8_lossy(&bytes)),
            )
        }
        None => (String::new(), String::new()),
    };
    let built = match std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
    {
        Some(s) => crate::wire::rfc3339(s),
        None => crate::wire::now(),
    };
    let flags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .split('\u{1f}')
        .filter(|f| !f.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    for (k, v) in [
        ("HANZO_BUILD_SHA", sha.as_str()),
        ("HANZO_BUILD_DIRTY", if dirty { "true" } else { "false" }),
        ("HANZO_BUILD_TIME", built.as_str()),
        ("HANZO_BUILD_LOCK", lock_hash.as_str()),
        ("HANZO_BUILD_ML", ml.as_str()),
        (
            "HANZO_BUILD_PROFILE",
            &std::env::var("PROFILE").unwrap_or_default(),
        ),
        ("HANZO_BUILD_FLAGS", flags.as_str()),
    ] {
        println!("cargo:rustc-env={k}={v}");
    }
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    if top.is_empty() {
        return;
    }
    // Rerun on any tracked file, so `dirty` never describes a tree other than the one compiled.
    let files = git(&top, &["ls-files", "-z"]);
    for f in files.split('\0').filter(|f| !f.is_empty()) {
        println!(
            "cargo:rerun-if-changed={}",
            Path::new(&top).join(f).display()
        );
    }
    let git_dir = PathBuf::from(git(&top, &["rev-parse", "--absolute-git-dir"]));
    let common = PathBuf::from(git(
        &top,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    ));
    let head = git(&top, &["symbolic-ref", "-q", "HEAD"]);
    let mut watch = vec![
        git_dir.join("HEAD"),
        git_dir.join("index"),
        common.join("packed-refs"),
    ];
    if !head.is_empty() {
        watch.push(common.join(&head));
    }
    for p in watch.into_iter().filter(|p| p.exists()) {
        println!("cargo:rerun-if-changed={}", p.display());
    }
}

/// The hanzoai/ml commit a `Cargo.lock` pins: the `#<sha>` of its first git source there.
pub fn ml_commit(lock: &str) -> String {
    lock.lines()
        .filter_map(|l| {
            l.trim()
                .strip_prefix("source = \"git+https://github.com/hanzoai/ml")
        })
        .find_map(|s| {
            s.split_once('#')
                .map(|(_, sha)| sha.trim_end_matches('"').to_string())
        })
        .unwrap_or_default()
}

/// The accelerator API and its versions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Accelerator {
    /// `cuda`, `rocm`, `metal` or `cpu`.
    pub api: String,
    /// The CUDA or ROCm toolkit's version, empty when none is installed.
    pub version: String,
    /// The kernel driver's version, empty when unread.
    pub driver: String,
}

/// One GPU.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    pub uuid: String,
    pub name: String,
    /// `sm_121`, `gfx1151`, …; empty when unread.
    pub arch: String,
}

/// Where a run executes, measured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Environment {
    /// The producing tree's commit, 40 hex; empty outside a checkout.
    pub git_sha: String,
    /// The tree differed from the commit, or there was no commit.
    pub dirty: bool,
    /// `sha256:` of the lockfile's bytes, empty when there is none.
    pub lockfile_hash: String,
    /// The hanzoai/ml commit the lockfile pins.
    pub ml: String,
    pub accelerator: Accelerator,
    pub gpus: Vec<Gpu>,
    pub compiler_flags: Vec<String>,
    pub host: String,
    pub os: String,
    /// The container image digest; `None` outside a container or when the runtime names none.
    pub image: Option<String>,
    /// The binary that measured it: `<name> <version>`.
    pub tool: String,
}

impl Environment {
    /// The environment of a binary built as `b`: its source from the stamp, the host measured.
    pub fn of(b: &Build) -> Environment {
        Environment {
            git_sha: b.git_sha.clone(),
            dirty: b.dirty,
            lockfile_hash: b.lockfile_hash.clone(),
            ml: b.ml.clone(),
            compiler_flags: b.flags.clone(),
            tool: format!("{} {}", b.name, b.version),
            ..host()
        }
    }

    /// The environment a build of the checkout at `repo` would run in: its source measured
    /// now, the host measured, the flags `RUSTFLAGS` would give.
    pub fn repo(repo: &Path, tool: &Build) -> Environment {
        let r = repo.to_string_lossy().to_string();
        let top = git(&r, &["rev-parse", "--show-toplevel"]);
        let sha = git(&r, &["rev-parse", "HEAD"]);
        let clean = Command::new("git")
            .args(["-C", &r, "status", "--porcelain"])
            .output()
            .is_ok_and(|o| o.status.success() && o.stdout.is_empty());
        let lock = (!top.is_empty())
            .then(|| std::fs::read(Path::new(&top).join("Cargo.lock")).ok())
            .flatten();
        Environment {
            dirty: sha.is_empty() || !clean,
            git_sha: sha,
            lockfile_hash: lock.as_ref().map_or(String::new(), |b| {
                format!("sha256:{}", canonical::sha256(b))
            }),
            ml: lock.map_or(String::new(), |b| ml_commit(&String::from_utf8_lossy(&b))),
            compiler_flags: std::env::var("RUSTFLAGS")
                .unwrap_or_default()
                .split_whitespace()
                .map(String::from)
                .collect(),
            tool: format!("{} {}", tool.name, tool.version),
            ..host()
        }
    }

    /// The `sha256:` hash of its canonical JSON.
    pub fn hash(&self) -> crate::Result<String> {
        Ok(canonical::hash(self)?)
    }
}

/// The host's part: accelerator, GPUs, name, OS and image.
fn host() -> Environment {
    let (accelerator, gpus) = accelerator();
    Environment {
        git_sha: String::new(),
        dirty: true,
        lockfile_hash: String::new(),
        ml: String::new(),
        accelerator,
        gpus,
        compiler_flags: Vec::new(),
        host: run("uname", &["-n"]),
        os: format!("{} {}", run("uname", &["-s"]), run("uname", &["-r"])),
        image: image(),
        tool: String::new(),
    }
}

fn accelerator() -> (Accelerator, Vec<Gpu>) {
    let smi = run(
        "nvidia-smi",
        &[
            "--query-gpu=uuid,name,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
    );
    if !smi.is_empty() {
        let mut driver = String::new();
        let gpus = smi
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split(',').map(str::trim).collect();
                driver = f.get(2).unwrap_or(&"").to_string();
                Gpu {
                    uuid: f.first().unwrap_or(&"").to_string(),
                    name: f.get(1).unwrap_or(&"").to_string(),
                    arch: f
                        .get(3)
                        .map_or(String::new(), |c| format!("sm_{}", c.replace('.', ""))),
                }
            })
            .collect();
        let nvcc = run("nvcc", &["--version"]);
        let version = nvcc
            .lines()
            .find_map(|l| l.split("release ").nth(1))
            .and_then(|v| v.split(',').next())
            .unwrap_or("")
            .to_string();
        return (
            Accelerator {
                api: "cuda".into(),
                version,
                driver,
            },
            gpus,
        );
    }
    let rocm = std::fs::read_to_string("/opt/rocm/.info/version").unwrap_or_default();
    if !rocm.trim().is_empty() {
        let names = run(
            "rocm-smi",
            &["--showproductname", "--showuniqueid", "--csv"],
        );
        let gpus = names
            .lines()
            .skip(1)
            .filter(|l| l.starts_with("card"))
            .map(|l| {
                let f: Vec<&str> = l.split(',').map(str::trim).collect();
                Gpu {
                    uuid: f.last().unwrap_or(&"").to_string(),
                    name: f.get(1).unwrap_or(&"").to_string(),
                    arch: f
                        .iter()
                        .find(|x| x.starts_with("gfx"))
                        .unwrap_or(&"")
                        .to_string(),
                }
            })
            .collect();
        let driver = std::fs::read_to_string("/sys/module/amdgpu/version").unwrap_or_default();
        return (
            Accelerator {
                api: "rocm".into(),
                version: rocm.trim().into(),
                driver: driver.trim().into(),
            },
            gpus,
        );
    }
    if cfg!(target_os = "macos") {
        let chip = run("sysctl", &["-n", "machdep.cpu.brand_string"]);
        return (
            Accelerator {
                api: "metal".into(),
                version: run("sw_vers", &["-productVersion"]),
                driver: String::new(),
            },
            vec![Gpu {
                uuid: String::new(),
                name: chip,
                arch: String::new(),
            }],
        );
    }
    (
        Accelerator {
            api: "cpu".into(),
            ..Accelerator::default()
        },
        Vec::new(),
    )
}

/// The image digest the container runtime names (`HANZO_IMAGE`, `<repo>@sha256:…`, set by the
/// deployment), `None` outside a container or when none is named.
fn image() -> Option<String> {
    let contained = Path::new("/.dockerenv").exists()
        || Path::new("/run/.containerenv").exists()
        || std::fs::read_to_string("/proc/1/cgroup").is_ok_and(|c| c.contains("kubepods"));
    if !contained {
        return None;
    }
    std::env::var("HANZO_IMAGE")
        .ok()
        .filter(|i| i.contains("@sha256:"))
}

fn run(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn git(dir: &str, args: &[&str]) -> String {
    let mut a = vec!["-C", dir];
    a.extend_from_slice(args);
    run("git", &a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ml_from_lock() {
        let lock = "[[package]]\nname = \"serde\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\n\n\
                    [[package]]\nname = \"hanzo-ml\"\nversion = \"0.11.94\"\n\
                    source = \"git+https://github.com/hanzoai/ml?rev=4feecaed6c1de6fe49a3472de381f009d941a9a8#4feecaed6c1de6fe49a3472de381f009d941a9a8\"\n";
        assert_eq!(ml_commit(lock), "4feecaed6c1de6fe49a3472de381f009d941a9a8");
        assert_eq!(ml_commit("[[package]]\nname = \"x\"\n"), "");
    }

    #[test]
    fn repo_is_measured() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let tool = Build {
            name: "t".into(),
            version: "0".into(),
            git_sha: String::new(),
            dirty: true,
            built: String::new(),
            lockfile_hash: String::new(),
            ml: String::new(),
            profile: String::new(),
            flags: vec![],
        };
        let e = Environment::repo(here, &tool);
        let sha = git(&here.to_string_lossy(), &["rev-parse", "HEAD"]);
        assert_eq!(e.git_sha, sha);
        assert_eq!(e.tool, "t 0");
        assert!(!e.accelerator.api.is_empty());
        assert!(e.hash().unwrap().starts_with("sha256:"));
        let outside = Environment::repo(&std::env::temp_dir(), &tool);
        assert!(outside.git_sha.is_empty() && outside.dirty && outside.lockfile_hash.is_empty());
    }
}
