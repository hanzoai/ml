//! RoPE (rotary position embedding) in the DSL, both rotation styles, one source -> every backend.
//!
//! The kernel is a pure rotation: cos/sin are precomputed per row `[rows, d/2]` (exactly the cache an
//! engine builds), so position -> frequency -> cos/sin stays on the host and the kernel has no transcendentals.
//! Two styles, which is the RoPE footgun (engine CLAUDE.md pitfall #8):
//!   - `rope_half`: GPT-NeoX / `is_gptx=true` -- pairs `(x[j], x[j + d/2])`.
//!   - `rope_interleaved`: GPT-J / `is_gptx=false` -- adjacent pairs `(x[2j], x[2j+1])`.
//! Using the wrong one gives cosine ~0.02 with the reference; both live here so a model picks by name.

use crate::prelude::*;

/// Half-split rotation (GPT-NeoX): rotate `(x[j], x[j+d/2])` by `(cos[j], sin[j])`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn rope_half<F: Float>(
    x: &Array<F>,
    cos: &Array<F>,
    sin: &Array<F>,
    out: &mut Array<F>,
    #[comptime] d: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / d {
        let d2 = d / 2;
        let xb = row * d;
        let cb = row * d2;
        for j in 0..d2 {
            let c = cos[cb + j];
            let s = sin[cb + j];
            let a = x[xb + j];
            let b = x[xb + j + d2];
            out[xb + j] = a * c - b * s;
            out[xb + j + d2] = a * s + b * c;
        }
    }
}

/// Interleaved rotation (GPT-J): rotate adjacent pairs `(x[2j], x[2j+1])` by `(cos[j], sin[j])`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn rope_interleaved<F: Float>(
    x: &Array<F>,
    cos: &Array<F>,
    sin: &Array<F>,
    out: &mut Array<F>,
    #[comptime] d: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / d {
        let d2 = d / 2;
        let xb = row * d;
        let cb = row * d2;
        for j in 0..d2 {
            let c = cos[cb + j];
            let s = sin[cb + j];
            let a = x[xb + 2 * j];
            let b = x[xb + 2 * j + 1];
            out[xb + 2 * j] = a * c - b * s;
            out[xb + 2 * j + 1] = a * s + b * c;
        }
    }
}

/// Host launch. `interleaved` picks the GPT-J style; otherwise GPT-NeoX half-split.
pub fn rope_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    rows: usize,
    d: usize,
    interleaved: bool,
) -> Vec<f32> {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let ch = client.create_from_slice(f32::as_bytes(cos));
    let sh = client.create_from_slice(f32::as_bytes(sin));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * d]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        let (g, b) = (Grid::Static(grid, 1, 1), Block::new_1d(block));
        let (xa, ca, sa, oa) = (
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ch.clone(), cos.len()),
            ArrayArg::from_raw_parts(sh.clone(), sin.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * d),
        );
        if interleaved {
            rope_interleaved::launch_unchecked::<f32, R>(client, g, b, xa, ca, sa, oa, d);
        } else {
            rope_half::launch_unchecked::<f32, R>(client, g, b, xa, ca, sa, oa, d);
        }
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle for RoPE (both styles), the reference the DSL kernel is gated against.
pub fn rope_ref(x: &[f32], cos: &[f32], sin: &[f32], rows: usize, d: usize, interleaved: bool) -> Vec<f32> {
    let d2 = d / 2;
    let mut out = vec![0.0f32; rows * d];
    for row in 0..rows {
        let (xb, cb) = (row * d, row * d2);
        for j in 0..d2 {
            let (c, s) = (cos[cb + j], sin[cb + j]);
            let (ai, bi) = if interleaved { (xb + 2 * j, xb + 2 * j + 1) } else { (xb + j, xb + j + d2) };
            let (a, b) = (x[ai], x[bi]);
            out[ai] = a * c - b * s;
            out[bi] = a * s + b * c;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a real RoPE cos/sin cache: theta_j = pos * base^(-2j/d), per row=position.
    fn caches(rows: usize, d: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
        let d2 = d / 2;
        let mut cos = vec![0.0f32; rows * d2];
        let mut sin = vec![0.0f32; rows * d2];
        for p in 0..rows {
            for j in 0..d2 {
                let freq = base.powf(-2.0 * j as f32 / d as f32);
                let theta = p as f32 * freq;
                cos[p * d2 + j] = theta.cos();
                sin[p * d2 + j] = theta.sin();
            }
        }
        (cos, sin)
    }

    fn xdata(rows: usize, d: usize) -> Vec<f32> {
        let mut s = 0x9E3779B9_7F4A7C15u64;
        (0..rows * d)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / x.abs().max(1e-6)).fold(0.0, f32::max)
    }

    fn run_style<R: Runtime>(c: &ComputeClient<R>, interleaved: bool, tag: &str) {
        let (rows, d) = (16, 64);
        let x = xdata(rows, d);
        let (cos, sin) = caches(rows, d, 10000.0);
        let got = rope_run::<R>(c, &x, &cos, &sin, rows, d, interleaved);
        let want = rope_ref(&x, &cos, &sin, rows, d, interleaved);
        let rel = max_rel(&want, &got);
        eprintln!("[rope {tag}] {rows}x{d} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "rope {tag} max_rel {rel}");
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn rope_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        run_style::<CpuRuntime>(&c, false, "half CPU");
        run_style::<CpuRuntime>(&c, true, "interleaved CPU");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn rope_metal_bit_exact() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        run_style::<WgpuRuntime>(&c, false, "half METAL");
        run_style::<WgpuRuntime>(&c, true, "interleaved METAL");
    }
}
