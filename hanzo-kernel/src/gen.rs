//! kernel-gen: the reproducible DSL -> backend-source generator.
//!
//! A `#[kernel]` is authored once in this crate. `hanzo-ml` does not depend on the DSL runtime at
//! inference time; it dispatches committed, human-diffable backend source (`src/metal/dsl/*.metal`
//! through `kernel_msl`, SPIR-V through `kernel_spv`). This module is how that source is produced, so
//! no artifact is ever hand-extracted or hand-edited.
//!
//! How: the generator launches each registered kernel specialization once on the target runtime with
//! cubecl's compilation logger pointed at a scratch file (`CUBECL_DEBUG_LOG=<file>`), then parses the
//! `[START_KERNEL_COMPILATION] ... source: ... [END_KERNEL_COMPILATION]` blocks the runtime itself wrote.
//! The captured text is `CompiledKernel::source`, i.e. exactly what the runtime compiled for that
//! device -- not a re-derivation that could drift from it. (Offline `define()` is not usable: cubecl
//! 0.10's `KernelDefinition` reads `client.properties()`, so the IR is device-dependent by design.)
//!
//! Drift gate: `kernel-gen --target <t> --check <dir>` regenerates every artifact for that target and
//! fails if any committed file differs. It runs where the target device exists (Metal on Apple
//! silicon, CUDA/ROCm on their GPUs); the release checklist runs it per backend.

use crate::prelude::*;

/// One generated artifact: a kernel specialization and where its source lands.
pub struct Spec<R: Runtime> {
    /// Artifact file stem under the backend's dsl dir (`<artifact>.metal`).
    pub artifact: &'static str,
    /// Entry-point name the runtime emits (cubecl names it after the kernel + elem suffix). Also the
    /// `kernel_msl` registry key.
    pub entry: &'static str,
    /// Launches the specialization once on `client` (small valid buffers; output is discarded).
    pub launch: fn(&ComputeClient<R>),
}

/// One `[START_KERNEL_COMPILATION]` block from a cubecl compilation log.
#[derive(Debug, Clone, PartialEq)]
pub struct Captured {
    pub lang: String,
    pub source: String,
}

impl Captured {
    /// Entry-point name declared in the source, for MSL (`[[kernel]]\nvoid <name>(`) and CUDA/HIP
    /// (`__global__ void <name>(`).
    pub fn entry(&self) -> Option<&str> {
        let s = &self.source;
        let at = s.find("[[kernel]]").or_else(|| s.find("__global__"))?;
        let rest = &s[at..];
        let v = rest.find("void ")? + "void ".len();
        let rest = &rest[v..];
        let end = rest.find('(')?;
        Some(rest[..end].trim())
    }
}

/// A bare local declaration with no initializer, e.g. `float l_mut_2;` or `uint l_12;`.
fn bare_local_decl_id(line: &str) -> Option<u64> {
    let t = line.trim();
    let body = t.strip_suffix(';')?;
    if body.contains('=') || body.contains('(') || body.contains('{') {
        return None;
    }
    let name = body.rsplit(' ').next()?;
    let digits = name
        .strip_prefix("l_mut_")
        .or_else(|| name.strip_prefix("l_"))?;
    digits.parse().ok()
}

/// Make captured source byte-stable across runs.
///
/// cubecl-cpp 0.10 declares mutable locals by iterating a hash set, so the ORDER of the bare
/// `T l_mut_N;` lines changes from run to run (measured on dbc: 5 runs, 2 orders for `gemv`). Their
/// order has no meaning in C/MSL, so each contiguous run is sorted by `N`. Leading blank lines (the
/// logger's formatting) are dropped. Nothing else is touched.
pub fn canonicalize(src: &str) -> String {
    let lines: Vec<&str> = src.trim_start_matches('\n').lines().collect();
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        if bare_local_decl_id(lines[i]).is_some() {
            let mut j = i;
            while j < lines.len() && bare_local_decl_id(lines[j]).is_some() {
                j += 1;
            }
            let mut run: Vec<&str> = lines[i..j].to_vec();
            run.sort_by_key(|l| bare_local_decl_id(l).unwrap_or(u64::MAX));
            out.extend(run);
            i = j;
        } else {
            out.push(lines[i]);
            i += 1;
        }
    }
    let mut s = out.join("\n");
    s.push('\n');
    sort_prologue_structs(&s)
}

/// cubecl-cpp also emits the prologue's type definitions (`struct alignas(4) half_2 {..};`,
/// `struct info_st {..};`) by iterating a hash set. Sort those blocks by text, but only when no struct
/// names another one in its body, so a dependent declaration order can never be broken.
fn sort_prologue_structs(src: &str) -> String {
    let Some(kernel_at) = src.find("[[kernel]]").or_else(|| src.find("__global__")) else {
        return src.to_string();
    };
    let (pro, rest) = src.split_at(kernel_at);
    let lines: Vec<&str> = pro.lines().collect();
    let mut blocks: Vec<String> = Vec::new();
    let mut head: Vec<&str> = Vec::new();
    let mut tail: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("struct ") && lines[i].trim_end().ends_with('{') {
            let mut j = i;
            while j < lines.len() && lines[j].trim() != "};" {
                j += 1;
            }
            if j == lines.len() {
                return src.to_string();
            }
            blocks.push(lines[i..=j].join("\n"));
            i = j + 1;
            // swallow one blank separator after a block
            if i < lines.len() && lines[i].trim().is_empty() {
                i += 1;
            }
        } else if blocks.is_empty() {
            head.push(lines[i]);
            i += 1;
        } else {
            tail.push(lines[i]);
            i += 1;
        }
    }
    if blocks.len() < 2 {
        return src.to_string();
    }
    let names: Vec<String> = blocks
        .iter()
        .filter_map(|b| {
            let first = b.lines().next()?;
            let name = first.trim_end_matches('{').split_whitespace().last()?;
            Some(name.to_string())
        })
        .collect();
    let dependent = blocks.iter().any(|b| {
        let body: String = b.lines().skip(1).collect::<Vec<_>>().join("\n");
        names.iter().any(|n| body.contains(n.as_str()))
    });
    if dependent {
        return src.to_string();
    }
    blocks.sort();
    let mut out = head.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    for b in &blocks {
        out.push_str(b);
        out.push_str("\n\n");
    }
    out.push_str(&tail.join("\n"));
    if !tail.is_empty() {
        out.push('\n');
    }
    out.push_str(rest);
    out
}

const START: &str = "[START_KERNEL_COMPILATION]";
const END: &str = "[END_KERNEL_COMPILATION]";

/// Parse every compilation block out of a cubecl log. Interleaved profiling/autotune lines are
/// ignored; only text between the source fence and `[END_KERNEL_COMPILATION]` is kept.
pub fn parse_log(log: &str) -> Vec<Captured> {
    let mut out = Vec::new();
    let mut rest = log;
    while let Some(s) = rest.find(START) {
        let block = &rest[s + START.len()..];
        let Some(e) = block.find(END) else { break };
        let body = &block[..e];
        rest = &block[e + END.len()..];
        let Some(src_at) = body.find("\nsource:\n```") else {
            continue;
        };
        let after = &body[src_at + "\nsource:\n```".len()..];
        let Some(nl) = after.find('\n') else { continue };
        let lang = after[..nl].trim().to_string();
        let code = &after[nl + 1..];
        // The fence closes with "\n```\n" right before END.
        let Some(close) = code.rfind("\n```") else {
            continue;
        };
        out.push(Captured {
            lang,
            source: canonicalize(&code[..close]),
        });
    }
    out
}

/// Run `specs` on `client`, capturing each one's compiled source from `log_path` (which must be the
/// file `CUBECL_DEBUG_LOG` named before the first client was created). Returns (spec index, source).
/// Fails if a spec's entry point never appears in the log.
pub fn capture<R: Runtime>(
    client: &ComputeClient<R>,
    specs: &[Spec<R>],
    log_path: &std::path::Path,
) -> Result<Vec<(usize, String)>, String> {
    let mut got = Vec::new();
    for (i, spec) in specs.iter().enumerate() {
        let before = std::fs::read_to_string(log_path).unwrap_or_default().len();
        (spec.launch)(client);
        let log = std::fs::read_to_string(log_path)
            .map_err(|e| format!("read {}: {e}", log_path.display()))?;
        let fresh = &log[before.min(log.len())..];
        // Only THIS launch's compilations count: specializations of one kernel share an entry name
        // (comptime params do not rename it), so matching anywhere else in the log could hand back a
        // different specialization's source.
        let all = parse_log(fresh);
        let hits: Vec<Captured> = all
            .iter()
            .filter(|c| c.entry() == Some(spec.entry))
            .cloned()
            .collect();
        if hits.len() != 1 {
            let seen: Vec<&str> = all.iter().filter_map(|c| c.entry()).collect();
            return Err(format!(
                "{}: its launch compiled `{}` {} times, expected exactly 1; entries it did compile: {seen:?}",
                spec.artifact,
                spec.entry,
                hits.len()
            ));
        }
        let hit = hits.into_iter().next().unwrap();
        got.push((i, hit.source));
    }
    Ok(got)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = "noise\n[START_KERNEL_COMPILATION]\nname: k\nsource:\n```cpp\n#include <metal_stdlib>\nusing namespace metal;\n\n[[kernel]]\nvoid gemv_f_f32(\n    const device float * buffer_0 [[buffer(0)]]\n) {\n}\n```\n[END_KERNEL_COMPILATION]\n| profile line\n[START_KERNEL_COMPILATION]\nsource:\n```cuda\nextern \"C\" __global__ void add_f32(float* a) {}\n```\n[END_KERNEL_COMPILATION]\n";

    #[test]
    fn parses_blocks_and_entries() {
        let c = parse_log(LOG);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].lang, "cpp");
        assert_eq!(c[0].entry(), Some("gemv_f_f32"));
        assert!(c[0].source.starts_with("#include <metal_stdlib>\n"));
        assert!(c[0].source.ends_with("}\n"));
        assert_eq!(c[1].entry(), Some("add_f32"));
    }

    #[test]
    fn canonicalize_sorts_only_bare_decl_runs() {
        let a = "\nint x = 1;\nuint l_mut_12;\nfloat l_mut_2;\nuint l_mut_3;\nl_mut_2 = 0.0;\n";
        let b = "\nint x = 1;\nfloat l_mut_2;\nuint l_mut_3;\nuint l_mut_12;\nl_mut_2 = 0.0;\n";
        assert_eq!(canonicalize(a), canonicalize(b));
        assert_eq!(
            canonicalize(a),
            "int x = 1;\nfloat l_mut_2;\nuint l_mut_3;\nuint l_mut_12;\nl_mut_2 = 0.0;\n"
        );
    }

    #[test]
    fn prologue_structs_sort_when_independent() {
        let a = "#include <metal_stdlib>\nusing namespace metal;\n\nstruct alignas(4) half_2 {\n    half i_0;\n};\n\nstruct info_st {\n    uint m[8];\n};\n\n[[kernel]]\nvoid k() {\n}\n";
        let b = "#include <metal_stdlib>\nusing namespace metal;\n\nstruct info_st {\n    uint m[8];\n};\n\nstruct alignas(4) half_2 {\n    half i_0;\n};\n\n[[kernel]]\nvoid k() {\n}\n";
        assert_eq!(canonicalize(a), canonicalize(b));
        assert!(canonicalize(a).contains("[[kernel]]\nvoid k()"));
        // A struct that names another must keep its order.
        let dep = "struct b_t {\n    a_t x;\n};\n\nstruct a_t {\n    uint y;\n};\n\n[[kernel]]\nvoid k() {\n}\n";
        assert_eq!(canonicalize(dep), dep);
    }

    #[test]
    fn truncated_block_is_ignored() {
        let c = parse_log("[START_KERNEL_COMPILATION]\nsource:\n```cpp\nvoid x(");
        assert!(c.is_empty());
    }
}
