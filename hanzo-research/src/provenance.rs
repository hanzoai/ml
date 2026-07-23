//! Zero-config provenance capture — the caller supplies nothing; the run's environment
//! is read. The Rust peer of the Python SDK's `provenance` module, producing the
//! identical record shape: git sha/branch/dirty, the commit-message narrative (the story
//! of what changed), the resolved crate versions, and the host.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use std::process::Command;

use serde::Serialize;

/// The producing repo's git identity at run time.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Git {
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
}

/// The box the run executed on — enough to correlate a result to hardware.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Host {
    pub hostname: String,
    pub platform: String,
}

/// Run a command, returning trimmed stdout, or "" on any failure. Provenance is
/// best-effort — a missing tool or non-repo directory yields empty, never an error.
fn run(cmd: &str, args: &[&str]) -> String {
    Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// Run `git -C <repo> <args...>`.
fn git(repo: &str, args: &[&str]) -> String {
    let mut full: Vec<&str> = vec!["-C", repo];
    full.extend_from_slice(args);
    run("git", &full)
}

/// The git sha, branch, and dirty flag of the producing repo.
pub fn git_state(repo: &str) -> Git {
    Git {
        git_sha: git(repo, &["rev-parse", "HEAD"]),
        git_branch: git(repo, &["rev-parse", "--abbrev-ref", "HEAD"]),
        git_dirty: !git(repo, &["status", "--porcelain"]).is_empty(),
    }
}

/// The commit-subject narrative — the "what changed + why" story. Subjects SINCE the
/// last recorded run's sha when known (`<since>..HEAD`), else the last `window` commits.
/// Research self-documents as a side effect of running: commit normally, nothing extra.
pub fn commit_narrative(repo: &str, since_sha: &str, window: usize) -> Vec<String> {
    // since_sha can arrive from a server response and reaches `git log <since>..HEAD`; only a
    // git object id (hex) is accepted, so it can never be read as a flag or a path — anything
    // else falls back to the recent window. Parity with the C++ port.
    let raw = if is_object_id(since_sha) {
        git(repo, &["log", &format!("{since_sha}..HEAD"), "--format=%s"])
    } else {
        git(repo, &["log", &format!("-{window}"), "--format=%s"])
    };
    raw.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// True when `s` is a git object id — hex only, 1..=64 chars (sha1=40, sha256=64). Gating a
/// server-supplied `since` to hex means it can never be read as a git flag (a leading `-`)
/// or a path (`/`), which closes the arg-injection.
fn is_object_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `{crate: version}` resolved from the repo's `Cargo.lock`, for the named crates that
/// are present. The Rust analog of Python's installed-version lookup: the run's ACTUAL
/// resolved dependency versions, so a kernel-perf result is pinned to the versions that
/// produced it — the longitudinal "which version regressed X" record.
pub fn lib_versions(repo: &str, names: &[&str]) -> BTreeMap<String, String> {
    let lock = std::fs::read_to_string(Path::new(repo).join("Cargo.lock")).unwrap_or_default();
    lib_versions_from_lock(&lock, names)
}

/// Parse a `Cargo.lock`'s `[[package]]` blocks (name immediately precedes version) and
/// keep the wanted crates. Split out so it is testable without a filesystem.
pub(crate) fn lib_versions_from_lock(lock: &str, names: &[&str]) -> BTreeMap<String, String> {
    let want: HashSet<&str> = names.iter().copied().collect();
    let mut out = BTreeMap::new();
    let mut name = "";
    for line in lock.lines() {
        let l = line.trim();
        if let Some(v) = l.strip_prefix("name = ") {
            name = v.trim_matches('"');
        } else if let Some(v) = l.strip_prefix("version = ") {
            if want.contains(name) {
                out.insert(name.to_string(), v.trim_matches('"').to_string());
            }
            name = "";
        }
    }
    out
}

/// The host: hostname and platform, mirroring Python's `socket.gethostname()` +
/// `os.uname().sysname` via `uname -n` / `uname -s`, with std fallbacks.
pub fn host() -> Host {
    Host {
        hostname: {
            let h = run("uname", &["-n"]);
            if h.is_empty() {
                std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string())
            } else {
                h
            }
        },
        platform: {
            let p = run("uname", &["-s"]);
            if p.is_empty() {
                std::env::consts::OS.to_string()
            } else {
                p
            }
        },
    }
}

/// The git repo the caller runs in (the toplevel), auto-detected from the CWD.
pub fn find_repo() -> String {
    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(str::to_string))
        .unwrap_or_else(|| ".".to_string());
    let top = git(&cwd, &["rev-parse", "--show-toplevel"]);
    if top.is_empty() {
        cwd
    } else {
        top
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wanted_versions_from_a_lock() {
        // A minimal Cargo.lock: only the wanted crates are kept; order-independent.
        let lock = r#"
# This file is automatically @generated by Cargo.
version = 4

[[package]]
name = "hanzo-ml"
version = "0.11.94"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "serde"
version = "1.0.228"
dependencies = [
 "serde_derive",
]

[[package]]
name = "hanzo-engine"
version = "1.7.92"
"#;
        let got = lib_versions_from_lock(lock, &["hanzo-ml", "hanzo-engine"]);
        assert_eq!(got.get("hanzo-ml").map(String::as_str), Some("0.11.94"));
        assert_eq!(got.get("hanzo-engine").map(String::as_str), Some("1.7.92"));
        // A crate not asked for is not captured.
        assert!(!got.contains_key("serde"));
        // A crate asked for but absent is simply skipped (no panic, no empty entry).
        let miss = lib_versions_from_lock(lock, &["does-not-exist"]);
        assert!(miss.is_empty());
    }

    #[test]
    fn since_must_be_a_git_object_id() {
        // A real object id drives `<since>..HEAD`.
        assert!(is_object_id("abc123"));
        assert!(is_object_id(&"a".repeat(40))); // sha1
        assert!(is_object_id(&"F".repeat(64))); // sha256
                                                // Anything else falls back to the window — no git flag/path can be injected.
        assert!(!is_object_id("")); // empty
        assert!(!is_object_id("--output=/tmp/pwned")); // flag injection
        assert!(!is_object_id("HEAD~5")); // ~ not hex
        assert!(!is_object_id("../etc")); // path chars
        assert!(!is_object_id("a b")); // whitespace
        assert!(!is_object_id(&"a".repeat(65))); // too long
    }
}
