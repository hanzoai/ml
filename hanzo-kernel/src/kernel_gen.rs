//! kernel-gen: regenerate (or drift-check) the committed DSL backend artifacts hanzo-ml dispatches.
//!
//!   kernel-gen --target metal --out ../hanzo-ml/src/metal/dsl          # write artifacts
//!   kernel-gen --target metal --check ../hanzo-ml/src/metal/dsl        # fail on any drift
//!
//! Each artifact is captured from the runtime's own compilation log (see `hanzo_kernel::gen`), so the
//! committed text is exactly what that backend compiled. Run on a host that has the target device.

use hanzo_kernel::gen::{capture, Spec};
use hanzo_kernel::prelude::*;
use std::path::{Path, PathBuf};

fn usage() -> ! {
    eprintln!("usage: kernel-gen --target metal (--out DIR | --check DIR)");
    std::process::exit(2);
}

/// One PQ2_0 matvec specialization, launched on a 3-row x 256-col tensor (odd block count, so the
/// half-word path and the clamped tail are both compiled in).
#[cfg(feature = "metal")]
macro_rules! pq2_spec {
    ($artifact:literal, $entry:literal, $e:ty, $conv:expr, $nr0:literal) => {
        Spec {
            artifact: $artifact,
            entry: $entry,
            launch: |c| {
                use hanzo_kernel::ternary::*;
                let (rows, k) = (3usize, 256usize);
                let (codes, d) = pq2_0_synth(rows, k, 1);
                let words = pq2_0_gguf_words(&pq2_0_gguf_bytes(&codes, &d, rows, k));
                let x: Vec<$e> = (0..k).map(|i| $conv(i as f32 * 1e-3)).collect();
                let _ = matvec_pq2_0_gguf_run_as::<$e, cubecl::wgpu::WgpuRuntime>(
                    c, &words, &x, rows, k, 4, $nr0,
                );
            },
        }
    };
}

/// Every Metal artifact hanzo-ml commits under `src/metal/dsl/`. Adding a DSL kernel to the Metal
/// path = one entry here + one `kernel_msl` arm in hanzo-ml.
#[cfg(feature = "metal")]
fn metal_specs() -> Vec<Spec<cubecl::wgpu::WgpuRuntime>> {
    use cubecl::wgpu::WgpuRuntime as R;
    vec![
        Spec {
            artifact: "rms_norm_blk",
            entry: "rms_norm_blk_f_f32",
            launch: |c| {
                let (rows, n) = (2usize, 128usize);
                let x = vec![0.5f32; rows * n];
                let w = vec![1.0f32; n];
                let _ =
                    hanzo_kernel::norm::rms_norm_blk_run::<f32, R>(c, &x, &w, rows, n, 1e-5, 128);
            },
        },
        Spec {
            artifact: "gemv",
            entry: "gemv_f_f32",
            launch: |c| {
                let (n, k) = (4usize, 64usize);
                let _ = hanzo_kernel::quant::gemv_run::<R>(
                    c,
                    &vec![1.0; n * k],
                    &vec![1.0; k],
                    n,
                    k,
                    32,
                );
            },
        },
        // PQ2_0 decode matvec over native GGUF blocks: one artifact per production schedule
        // (`ternary::pq2_0_gguf_schedule`: NSG=4, NR0 in {2,4,8}) x activation dtype. The entry name
        // is the same for every NR0 (cubecl names it after kernel + elem), so the ARTIFACT name is
        // the pipeline identity hanzo-ml caches on.
        pq2_spec!(
            "matvec_pq2_0_gguf_nsg4_nr2_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            2
        ),
        pq2_spec!(
            "matvec_pq2_0_gguf_nsg4_nr4_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            4
        ),
        pq2_spec!(
            "matvec_pq2_0_gguf_nsg4_nr8_f32",
            "matvec_pq2_0_gguf_f_f32",
            f32,
            std::convert::identity,
            8
        ),
        pq2_spec!(
            "matvec_pq2_0_gguf_nsg4_nr2_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            2
        ),
        pq2_spec!(
            "matvec_pq2_0_gguf_nsg4_nr4_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            4
        ),
        pq2_spec!(
            "matvec_pq2_0_gguf_nsg4_nr8_bf16",
            "matvec_pq2_0_gguf_f_",
            half::bf16,
            half::bf16::from_f32,
            8
        ),
    ]
}

fn write_or_check(dir: &Path, check: bool, ext: &str, items: &[(&str, String)]) -> bool {
    let mut ok = true;
    for (stem, src) in items {
        let path = dir.join(format!("{stem}.{ext}"));
        if check {
            match std::fs::read_to_string(&path) {
                Ok(cur) if cur == *src => eprintln!("ok     {}", path.display()),
                Ok(cur) => {
                    ok = false;
                    let first = cur
                        .lines()
                        .zip(src.lines())
                        .position(|(a, b)| a != b)
                        .unwrap_or(cur.lines().count().min(src.lines().count()));
                    eprintln!(
                        "DRIFT  {} (first differing line {}; committed {} lines, generated {} lines)",
                        path.display(),
                        first + 1,
                        cur.lines().count(),
                        src.lines().count()
                    );
                }
                Err(e) => {
                    ok = false;
                    eprintln!("MISSING {}: {e}", path.display());
                }
            }
        } else {
            std::fs::create_dir_all(dir).expect("create out dir");
            std::fs::write(&path, src).expect("write artifact");
            eprintln!("wrote  {}", path.display());
        }
    }
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut target = None;
    let mut dir: Option<(PathBuf, bool)> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--target" => target = it.next().cloned(),
            "--out" => dir = it.next().map(|d| (PathBuf::from(d), false)),
            "--check" => dir = it.next().map(|d| (PathBuf::from(d), true)),
            _ => usage(),
        }
    }
    let (Some(target), Some((dir, check))) = (target, dir) else {
        usage()
    };

    // Must be set before the first client exists: cubecl reads its config once.
    let log = std::env::temp_dir().join(format!("hanzo-kernel-gen-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    std::env::set_var("CUBECL_DEBUG_LOG", &log);

    let ok = match target.as_str() {
        #[cfg(feature = "metal")]
        "metal" => {
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
            let client = WgpuRuntime::client(&WgpuDevice::default());
            let name = WgpuRuntime::name(&client);
            if !name.contains("msl") {
                eprintln!("runtime is `{name}`, not wgpu<msl>: refusing to write Metal artifacts");
                std::process::exit(1);
            }
            let specs = metal_specs();
            let got = capture(&client, &specs, &log).unwrap_or_else(|e| {
                eprintln!("{e}");
                std::process::exit(1)
            });
            let items: Vec<(&str, String)> = got
                .into_iter()
                .map(|(i, s)| (specs[i].artifact, s))
                .collect();
            write_or_check(&dir, check, "metal", &items)
        }
        t => {
            eprintln!("target `{t}` not built in (enable its cargo feature) or not supported yet");
            std::process::exit(2);
        }
    };
    let _ = std::fs::remove_file(&log);
    std::process::exit(if ok { 0 } else { 1 });
}
