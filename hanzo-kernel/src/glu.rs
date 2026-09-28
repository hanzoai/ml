//! GeGLU, `y = gelu(g) u` with `gelu(x) = x Φ(x)`, and its gradient, computed in F32.

use crate::prelude::*;

/// `y = gelu(g) u`, elementwise over `n = meta[0]` values.
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn geglu<F: Float>(g: &Array<F>, u: &Array<F>, y: &mut Array<F>, meta: &Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < meta[0] as usize {
        let x = f32::cast_from(g[i]);
        let c = 0.5f32 * (1.0f32 + (x * 0.70710678f32).erf());
        y[i] = F::cast_from(x * c * f32::cast_from(u[i]));
    }
}

/// `dg = dy u gelu'(g)`, `du = dy gelu(g)`, `gelu'(x) = Φ(x) + x φ(x)`.
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn geglu_back<F: Float>(
    g: &Array<F>,
    u: &Array<F>,
    dy: &Array<F>,
    dg: &mut Array<F>,
    du: &mut Array<F>,
    meta: &Array<u32>,
) {
    let i = ABSOLUTE_POS;
    if i < meta[0] as usize {
        let x = f32::cast_from(g[i]);
        let d = f32::cast_from(dy[i]);
        let c = 0.5f32 * (1.0f32 + (x * 0.70710678f32).erf());
        let phi = 0.39894228f32 * (-0.5f32 * x * x).exp();
        dg[i] = F::cast_from(d * f32::cast_from(u[i]) * (c + x * phi));
        du[i] = F::cast_from(d * x * c);
    }
}

/// [`geglu`] then [`geglu_back`] on `client` from host data: `(y, dg, du)`.
pub fn geglu_run<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    g: &[F],
    u: &[F],
    dy: &[F],
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let n = g.len();
    let h = |x: &[F]| client.create_from_slice(F::as_bytes(x));
    let z = || client.create_from_slice(&vec![0u8; n * std::mem::size_of::<F>()]);
    let (gh, uh, dyh, yh, dgh, duh) = (h(g), h(u), h(dy), z(), z(), z());
    let mh = client.create_from_slice(u32::as_bytes(&[n as u32]));
    let grid = Grid::Static((n as u32).div_ceil(256), 1, 1);
    unsafe {
        geglu::launch_unchecked::<F, R>(
            client,
            grid.clone(),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(gh.clone(), n),
            ArrayArg::from_raw_parts(uh.clone(), n),
            ArrayArg::from_raw_parts(yh.clone(), n),
            ArrayArg::from_raw_parts(mh.clone(), 1),
        );
        geglu_back::launch_unchecked::<F, R>(
            client,
            grid,
            Block::new_1d(256),
            ArrayArg::from_raw_parts(gh.clone(), n),
            ArrayArg::from_raw_parts(uh.clone(), n),
            ArrayArg::from_raw_parts(dyh.clone(), n),
            ArrayArg::from_raw_parts(dgh.clone(), n),
            ArrayArg::from_raw_parts(duh.clone(), n),
            ArrayArg::from_raw_parts(mh.clone(), 1),
        );
    }
    let read = |x| F::from_bytes(&client.read_one_unchecked(x)).to_vec();
    (read(yh), read(dgh), read(duh))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20001) as f32 / 10000.0 - 1.0) * amp
            })
            .collect()
    }

    /// erf by its Maclaurin series, exact to f64 over the |x| < 3 these tests draw.
    fn erf(x: f64) -> f64 {
        let (mut term, mut sum, x2) = (x, x, x * x);
        for n in 1..80 {
            term *= -x2 / n as f64;
            sum += term / (2 * n + 1) as f64;
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    }

    fn check<R: Runtime>(client: &ComputeClient<R>, name: &str) {
        let n = 4099;
        let (g, u, dy) = (rnd(n, 3, 4.0), rnd(n, 5, 2.0), rnd(n, 7, 1.0));
        let (y, dg, du) = geglu_run::<R, f32>(client, &g, &u, &dy);
        let mut worst = [0f64; 3];
        for i in 0..n {
            let x = g[i] as f64;
            let c = 0.5 * (1.0 + erf(x / 2f64.sqrt()));
            let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
            let want = [
                x * c * u[i] as f64,
                dy[i] as f64 * u[i] as f64 * (c + x * phi),
                dy[i] as f64 * x * c,
            ];
            for (k, (a, b)) in [y[i], dg[i], du[i]].iter().zip(want).enumerate() {
                worst[k] = worst[k].max((*a as f64 - b).abs());
            }
        }
        eprintln!(
            "[geglu {name}] max |Δ| y {:.2e}, dg {:.2e}, du {:.2e}",
            worst[0], worst[1], worst[2]
        );
        assert!(worst.iter().all(|&w| w < 1e-5), "{worst:?}");
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn geglu_is_the_composite_on_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        check(&CpuRuntime::client(&CpuDevice::default()), "cpu");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn geglu_is_the_composite_on_metal() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        check(&WgpuRuntime::client(&WgpuDevice::default()), "metal");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn geglu_is_the_composite_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        check(&CudaRuntime::client(&CudaDevice::default()), "cuda");
    }
}
