//! The optimizer's kernels over F32 parameters: one AdamW step in place, and a gradient's sum of
//! squares as per-block partials a caller sums in a fixed order.

use crate::prelude::*;

/// One AdamW step in place: `p = [lr, beta1, beta2, eps, decay, scale_m, scale_v, grad_scale]`,
/// `scale_*` the bias corrections.
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn adamw(
    w: &mut Array<f32>,
    g: &Array<f32>,
    m: &mut Array<f32>,
    v: &mut Array<f32>,
    p: &Array<f32>,
    n: &Array<u32>,
) {
    let i = ABSOLUTE_POS;
    if i < n[0] as usize {
        let gi = g[i] * p[7];
        let mi = p[1] * m[i] + (1.0f32 - p[1]) * gi;
        let vi = p[2] * v[i] + (1.0f32 - p[2]) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        let step = (mi * p[5]) / ((vi * p[6]).sqrt() + p[3]);
        w[i] = w[i] * (1.0f32 - p[0] * p[4]) - p[0] * step;
    }
}

/// `partial[base + b]`: the sum of squares of block `b`'s grid-stride share of `x[..n]`,
/// `meta = [n, base]`; block of `nt` threads.
#[kernel(targets(cuda, rocm, metal), unchecked)]
pub fn sumsq(x: &Array<f32>, partial: &mut Array<f32>, meta: &Array<u32>, #[comptime] nt: usize) {
    let n = meta[0] as usize;
    let t = UNIT_POS as usize;
    let step = CUBE_COUNT as usize * CUBE_DIM as usize;
    let mut acc = 0f32;
    let mut i = ABSOLUTE_POS as usize;
    while i < n {
        let xi = x[i];
        acc += xi * xi;
        i += step;
    }
    let mut smem = SharedMemory::<f32>::new(nt);
    smem[t] = acc;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            let v = smem[(UNIT_POS + stride) as usize];
            smem[t] += v;
        }
        sync_cube();
        stride /= 2;
    }
    let base = meta[1] as usize;
    if t == 0 {
        partial[base + CUBE_POS as usize] = smem[0];
    }
}

/// [`adamw`] on `client` from host data; returns `(w, m, v)`.
#[allow(clippy::too_many_arguments)]
pub fn adamw_run<R: Runtime>(
    client: &ComputeClient<R>,
    w: &[f32],
    g: &[f32],
    m: &[f32],
    v: &[f32],
    p: [f32; 8],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = w.len();
    let h = |x: &[f32]| client.create_from_slice(f32::as_bytes(x));
    let (wh, gh, mh, vh, ph) = (h(w), h(g), h(m), h(v), h(&p));
    let nh = client.create_from_slice(u32::as_bytes(&[n as u32]));
    unsafe {
        adamw::launch_unchecked::<R>(
            client,
            Grid::Static((n as u32).div_ceil(256), 1, 1),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(wh.clone(), n),
            ArrayArg::from_raw_parts(gh.clone(), n),
            ArrayArg::from_raw_parts(mh.clone(), n),
            ArrayArg::from_raw_parts(vh.clone(), n),
            ArrayArg::from_raw_parts(ph.clone(), 8),
            ArrayArg::from_raw_parts(nh.clone(), 1),
        );
    }
    let read = |x| f32::from_bytes(&client.read_one_unchecked(x)).to_vec();
    (read(wh), read(mh), read(vh))
}

/// [`sumsq`] on `client`: `blocks` partials of `x`'s sum of squares.
pub fn sumsq_run<R: Runtime>(client: &ComputeClient<R>, x: &[f32], blocks: u32) -> Vec<f32> {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let ph = client.create_from_slice(f32::as_bytes(&vec![0f32; blocks as usize]));
    let mh = client.create_from_slice(u32::as_bytes(&[x.len() as u32, 0]));
    unsafe {
        sumsq::launch_unchecked::<R>(
            client,
            Grid::Static(blocks, 1, 1),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ph.clone(), blocks as usize),
            ArrayArg::from_raw_parts(mh.clone(), 2),
            256,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(ph)).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 20001) as f32 / 10000.0 - 1.0
            })
            .collect()
    }

    const P: [f32; 8] = [2e-5, 0.9, 0.98, 1e-6, 0.01, 1.9, 7.3, 0.37];

    /// The step in f64, rounded once.
    fn adamw_ref(w: &[f32], g: &[f32], m: &[f32], v: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let p: Vec<f64> = P.iter().map(|&x| x as f64).collect();
        let mut out = (
            vec![0f32; w.len()],
            vec![0f32; w.len()],
            vec![0f32; w.len()],
        );
        for i in 0..w.len() {
            let gi = g[i] as f64 * p[7];
            let mi = p[1] * m[i] as f64 + (1.0 - p[1]) * gi;
            let vi = p[2] * v[i] as f64 + (1.0 - p[2]) * gi * gi;
            let step = mi * p[5] / ((vi * p[6]).sqrt() + p[3]);
            out.0[i] = (w[i] as f64 * (1.0 - p[0] * p[4]) - p[0] * step) as f32;
            out.1[i] = mi as f32;
            out.2[i] = vi as f32;
        }
        out
    }

    fn check<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let n = 1025 * 7;
        let (w, g, m) = (rnd(n, 3), rnd(n, 5), rnd(n, 7));
        let v: Vec<f32> = rnd(n, 11).iter().map(|x| x * x).collect();
        let want = adamw_ref(&w, &g, &m, &v);
        let got = adamw_run(client, &w, &g, &m, &v, P);
        for (what, a, b) in [
            ("w", &got.0, &want.0),
            ("m", &got.1, &want.1),
            ("v", &got.2, &want.2),
        ] {
            let top = b.iter().fold(0f32, |x, y| x.max(y.abs()));
            let d = a.iter().zip(b).fold(0f32, |x, (p, q)| x.max((p - q).abs()));
            eprintln!("[adamw {name}] {what}: max |Δ| {d:.2e} at scale {top:.2}");
            assert!(d <= 2e-7 * top, "{what}: {d}");
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn adamw_is_the_step_on_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        check(&CpuRuntime::client(&CpuDevice::default()), "cpu");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn adamw_is_the_step_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        check(&CudaRuntime::client(&CudaDevice::default()), "cuda");
    }

    fn sums<R: Runtime>(c: &ComputeClient<R>, name: &str) {
        let x = rnd(100_003, 13);
        let want: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
        for blocks in [1u32, 7, 64] {
            let got: f64 = sumsq_run(c, &x, blocks).iter().map(|&v| v as f64).sum();
            eprintln!("[sumsq {name}] {blocks} blocks: {got} against {want}");
            assert!((got - want).abs() <= 1e-5 * want);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn sumsq_sums_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        sums(&CudaRuntime::client(&CudaDevice::default()), "cuda");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn sumsq_sums_on_metal() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        sums(&WgpuRuntime::client(&WgpuDevice::default()), "metal");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn adamw_is_the_step_on_metal() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        check(&WgpuRuntime::client(&WgpuDevice::default()), "metal");
    }
}
