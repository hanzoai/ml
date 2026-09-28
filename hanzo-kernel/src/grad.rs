//! Backward passes of the last-dim softmax and LayerNorm, a block of `nt` threads per row of `d`,
//! accumulated in F32.

use crate::prelude::*;

/// The sum of `v` over the block, returned to every thread; `smem` holds `nt` values.
#[device]
fn total(v: f32, smem: &mut SharedMemory<f32>) -> f32 {
    let t = UNIT_POS as usize;
    sync_cube();
    smem[t] = v;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            let x = smem[(UNIT_POS + stride) as usize];
            smem[t] += x;
        }
        sync_cube();
        stride /= 2;
    }
    smem[0]
}

/// `dx = y (dy − ⟨dy, y⟩)` per row; `meta = [d]`.
#[kernel(targets(cuda, rocm, metal), unchecked)]
pub fn softmax_back<F: Float>(
    y: &Array<F>,
    dy: &Array<F>,
    dx: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] nt: usize,
) {
    let d = meta[0] as usize;
    let base = CUBE_POS as usize * d;
    let t = UNIT_POS as usize;
    let mut smem = SharedMemory::<f32>::new(nt);
    let mut acc = 0f32;
    let mut i = t;
    while i < d {
        acc += f32::cast_from(dy[base + i]) * f32::cast_from(y[base + i]);
        i += nt;
    }
    let dot = total(acc, &mut smem);
    let mut o = t;
    while o < d {
        let yi = f32::cast_from(y[base + o]);
        dx[base + o] = F::cast_from(yi * (f32::cast_from(dy[base + o]) - dot));
        o += nt;
    }
}

/// LayerNorm per row, `x̂ = (x − μ) r`, `r = (var + eps)^-½`, `u = dy α`:
/// `dx = r (u − mean(u) − x̂ mean(u x̂))`; each row's `(μ, r)` into `stats`. `meta = [d]`.
#[kernel(targets(cuda, rocm, metal), unchecked)]
pub fn layer_norm_back<F: Float>(
    x: &Array<F>,
    dy: &Array<F>,
    alpha: &Array<F>,
    dx: &mut Array<F>,
    stats: &mut Array<f32>,
    eps: &Array<f32>,
    meta: &Array<u32>,
    #[comptime] nt: usize,
) {
    let d = meta[0] as usize;
    let row = CUBE_POS as usize;
    let base = row * d;
    let t = UNIT_POS as usize;
    let n = f32::cast_from(d as u32);
    let mut smem = SharedMemory::<f32>::new(nt);
    let mut s = 0f32;
    let mut i = t;
    while i < d {
        s += f32::cast_from(x[base + i]);
        i += nt;
    }
    let mean = total(s, &mut smem) / n;
    let mut q = 0f32;
    let mut i = t;
    while i < d {
        let c = f32::cast_from(x[base + i]) - mean;
        q += c * c;
        i += nt;
    }
    let r = 1.0f32 / (total(q, &mut smem) / n + eps[0]).sqrt();
    let mut su = 0f32;
    let mut sux = 0f32;
    let mut i = t;
    while i < d {
        let xh = (f32::cast_from(x[base + i]) - mean) * r;
        let u = f32::cast_from(dy[base + i]) * f32::cast_from(alpha[i]);
        su += u;
        sux += u * xh;
        i += nt;
    }
    let su = total(su, &mut smem) / n;
    let sux = total(sux, &mut smem) / n;
    let mut i = t;
    while i < d {
        let xh = (f32::cast_from(x[base + i]) - mean) * r;
        let u = f32::cast_from(dy[base + i]) * f32::cast_from(alpha[i]);
        dx[base + i] = F::cast_from(r * (u - su - xh * sux));
        i += nt;
    }
    if t == 0 {
        stats[2 * row] = mean;
        stats[2 * row + 1] = r;
    }
}

/// LayerNorm's parameter gradients as per-chunk partials: unit `(j, c)` sums column `j` over rows
/// `c·per .. (c + 1)·per`, `dα` into `partial[c][0][j]`, `dβ` into `partial[c][1][j]`;
/// `meta = [rows, d, per]`.
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn layer_norm_params<F: Float>(
    x: &Array<F>,
    dy: &Array<F>,
    stats: &Array<f32>,
    partial: &mut Array<f32>,
    meta: &Array<u32>,
) {
    let rows = meta[0] as usize;
    let d = meta[1] as usize;
    let per = meta[2] as usize;
    let j = ABSOLUTE_POS_X as usize;
    let c = CUBE_POS_Y as usize;
    if j < d {
        let r0 = c * per;
        let mut r1 = r0 + per;
        if r1 > rows {
            r1 = rows;
        }
        let mut da = 0f32;
        let mut db = 0f32;
        let mut r = r0;
        while r < r1 {
            let g = f32::cast_from(dy[r * d + j]);
            da += g * (f32::cast_from(x[r * d + j]) - stats[2 * r]) * stats[2 * r + 1];
            db += g;
            r += 1;
        }
        partial[c * 2 * d + j] = da;
        partial[c * 2 * d + d + j] = db;
    }
}

/// [`softmax_back`] on `client` from host data, `rows × d`.
pub fn softmax_back_run<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    y: &[F],
    dy: &[F],
    d: usize,
) -> Vec<F> {
    let rows = y.len() / d;
    let (yh, dyh) = (
        client.create_from_slice(F::as_bytes(y)),
        client.create_from_slice(F::as_bytes(dy)),
    );
    let dxh = client.create_from_slice(&vec![0u8; y.len() * std::mem::size_of::<F>()]);
    let mh = client.create_from_slice(u32::as_bytes(&[d as u32]));
    unsafe {
        softmax_back::launch_unchecked::<F, R>(
            client,
            Grid::Static(rows as u32, 1, 1),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(yh.clone(), y.len()),
            ArrayArg::from_raw_parts(dyh.clone(), y.len()),
            ArrayArg::from_raw_parts(dxh.clone(), y.len()),
            ArrayArg::from_raw_parts(mh.clone(), 1),
            256,
        );
    }
    F::from_bytes(&client.read_one_unchecked(dxh)).to_vec()
}

/// [`layer_norm_back`] and [`layer_norm_params`] on `client` from host data, `rows × d`:
/// `(dx, dα, dβ)`, the parameter partials summed on the host in chunk order.
pub fn layer_norm_back_run<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    x: &[F],
    dy: &[F],
    alpha: &[F],
    eps: f32,
) -> (Vec<F>, Vec<f32>, Vec<f32>) {
    let d = alpha.len();
    let rows = x.len() / d;
    let h = |v: &[F]| client.create_from_slice(F::as_bytes(v));
    let (xh, dyh, ah) = (h(x), h(dy), h(alpha));
    let dxh = client.create_from_slice(&vec![0u8; x.len() * std::mem::size_of::<F>()]);
    let sh = client.create_from_slice(f32::as_bytes(&vec![0f32; 2 * rows]));
    let eh = client.create_from_slice(f32::as_bytes(&[eps]));
    let chunks = rows.div_ceil(64).clamp(1, 256);
    let per = rows.div_ceil(chunks);
    let ph = client.create_from_slice(f32::as_bytes(&vec![0f32; chunks * 2 * d]));
    let mh = client.create_from_slice(u32::as_bytes(&[d as u32]));
    let pmh = client.create_from_slice(u32::as_bytes(&[rows as u32, d as u32, per as u32]));
    unsafe {
        layer_norm_back::launch_unchecked::<F, R>(
            client,
            Grid::Static(rows as u32, 1, 1),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(dyh.clone(), x.len()),
            ArrayArg::from_raw_parts(ah.clone(), d),
            ArrayArg::from_raw_parts(dxh.clone(), x.len()),
            ArrayArg::from_raw_parts(sh.clone(), 2 * rows),
            ArrayArg::from_raw_parts(eh.clone(), 1),
            ArrayArg::from_raw_parts(mh.clone(), 1),
            256,
        );
        layer_norm_params::launch_unchecked::<F, R>(
            client,
            Grid::Static((d as u32).div_ceil(256), chunks as u32, 1),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(dyh.clone(), x.len()),
            ArrayArg::from_raw_parts(sh.clone(), 2 * rows),
            ArrayArg::from_raw_parts(ph.clone(), chunks * 2 * d),
            ArrayArg::from_raw_parts(pmh.clone(), 3),
        );
    }
    let dx = F::from_bytes(&client.read_one_unchecked(dxh)).to_vec();
    let partial = f32::from_bytes(&client.read_one_unchecked(ph)).to_vec();
    let (mut da, mut db) = (vec![0f32; d], vec![0f32; d]);
    for c in 0..chunks {
        for j in 0..d {
            da[j] += partial[c * 2 * d + j];
            db[j] += partial[c * 2 * d + d + j];
        }
    }
    (dx, da, db)
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

    fn gap(a: &[f32], b: &[f64]) -> f64 {
        let top = b.iter().fold(0f64, |m, x| m.max(x.abs())).max(1e-12);
        a.iter()
            .zip(b)
            .fold(0f64, |m, (x, y)| m.max((*x as f64 - y).abs()))
            / top
    }

    fn rows<R: Runtime>(c: &ComputeClient<R>, name: &str) {
        let (rows, d) = (70usize, 768usize);
        let z = rnd(rows * d, 3, 3.0);
        let dy = rnd(rows * d, 5, 1.0);
        let mut y = vec![0f32; rows * d];
        let mut want = vec![0f64; rows * d];
        for r in 0..rows {
            let row = &z[r * d..(r + 1) * d];
            let m = row.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
            let sum: f64 = row.iter().map(|&x| (x as f64 - m).exp()).sum();
            let p: Vec<f64> = row.iter().map(|&x| (x as f64 - m).exp() / sum).collect();
            for i in 0..d {
                y[r * d + i] = p[i] as f32;
            }
            let dot: f64 = (0..d)
                .map(|i| dy[r * d + i] as f64 * y[r * d + i] as f64)
                .sum();
            for i in 0..d {
                want[r * d + i] = y[r * d + i] as f64 * (dy[r * d + i] as f64 - dot);
            }
        }
        let got = softmax_back_run::<R, f32>(c, &y, &dy, d);
        let g = gap(&got, &want);
        eprintln!("[softmax back {name}] {g:.2e}");
        assert!(g < 1e-5);

        let x = rnd(rows * d, 7, 2.0);
        let alpha = rnd(d, 11, 1.0);
        let eps = 1e-5f64;
        let (mut wdx, mut wda, mut wdb) = (vec![0f64; rows * d], vec![0f64; d], vec![0f64; d]);
        for r in 0..rows {
            let row: Vec<f64> = x[r * d..(r + 1) * d].iter().map(|&v| v as f64).collect();
            let mean = row.iter().sum::<f64>() / d as f64;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d as f64;
            let rs = 1.0 / (var + eps).sqrt();
            let xh: Vec<f64> = row.iter().map(|v| (v - mean) * rs).collect();
            let u: Vec<f64> = (0..d)
                .map(|i| dy[r * d + i] as f64 * alpha[i] as f64)
                .collect();
            let su = u.iter().sum::<f64>() / d as f64;
            let sux = u.iter().zip(&xh).map(|(a, b)| a * b).sum::<f64>() / d as f64;
            for i in 0..d {
                wdx[r * d + i] = rs * (u[i] - su - xh[i] * sux);
                wda[i] += dy[r * d + i] as f64 * xh[i];
                wdb[i] += dy[r * d + i] as f64;
            }
        }
        let (dx, da, db) = layer_norm_back_run::<R, f32>(c, &x, &dy, &alpha, eps as f32);
        let (gx, ga, gb) = (gap(&dx, &wdx), gap(&da, &wda), gap(&db, &wdb));
        eprintln!("[layer norm back {name}] dx {gx:.2e}, dα {ga:.2e}, dβ {gb:.2e}");
        assert!(gx < 1e-5 && ga < 1e-5 && gb < 1e-5);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn row_gradients_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        rows(&CudaRuntime::client(&CudaDevice::default()), "cuda");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn row_gradients_on_metal() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        rows(&WgpuRuntime::client(&WgpuDevice::default()), "metal");
    }

    /// Mean ms a launch of each small training kernel at the sizes hanzo-nn's `fused_speed` times,
    /// buffers resident: `cargo test --features cuda small_speed -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn small_speed_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        use half::bf16;
        type R = CudaRuntime;
        let c = R::client(&CudaDevice::default());
        let narrow = |v: Vec<f32>| -> Vec<bf16> { v.into_iter().map(bf16::from_f32).collect() };
        let h16 = |v: Vec<f32>| c.create_from_slice(bf16::as_bytes(&narrow(v)));
        let h32 = |v: &[f32]| c.create_from_slice(f32::as_bytes(v));
        let hu = |v: &[u32]| c.create_from_slice(u32::as_bytes(v));
        let sync = |c: &ComputeClient<R>| {
            let _ = c.read_one_unchecked(c.create_from_slice(u32::as_bytes(&[0u32])));
        };
        let time = |f: &dyn Fn()| -> f64 {
            for _ in 0..3 {
                f();
            }
            sync(&c);
            let clock = std::time::Instant::now();
            for _ in 0..50 {
                f();
            }
            sync(&c);
            clock.elapsed().as_secs_f64() * 1e3 / 50.0
        };
        let arg = |h: &cubecl::server::Handle, n: usize| unsafe {
            ArrayArg::from_raw_parts(h.clone(), n)
        };

        let n = 8192 * 1152;
        let (g, u, y) = (h16(rnd(n, 3, 2.0)), h16(rnd(n, 5, 1.0)), h16(vec![0f32; n]));
        let (dg, du) = (h16(vec![0f32; n]), h16(vec![0f32; n]));
        let mn = hu(&[n as u32]);
        let grid = Grid::Static((n as u32).div_ceil(256), 1, 1);
        let fwd = time(&|| unsafe {
            crate::glu::geglu::launch_unchecked::<bf16, R>(
                &c,
                grid.clone(),
                Block::new_1d(256),
                arg(&g, n),
                arg(&u, n),
                arg(&y, n),
                arg(&mn, 1),
            );
        });
        let back = time(&|| unsafe {
            crate::glu::geglu_back::launch_unchecked::<bf16, R>(
                &c,
                grid.clone(),
                Block::new_1d(256),
                arg(&g, n),
                arg(&u, n),
                arg(&y, n),
                arg(&dg, n),
                arg(&du, n),
                arg(&mn, 1),
            );
        });
        eprintln!("[small speed dsl] geglu {n}: forward {fwd:.3} ms, backward {back:.3} ms");

        let (rows, d) = (16384usize, 1024usize);
        let (sy, sdy, sdx) = (
            h16(rnd(rows * d, 7, 1.0)),
            h16(rnd(rows * d, 9, 1.0)),
            h16(vec![0f32; rows * d]),
        );
        let md = hu(&[d as u32]);
        let ms = time(&|| unsafe {
            softmax_back::launch_unchecked::<bf16, R>(
                &c,
                Grid::Static(rows as u32, 1, 1),
                Block::new_1d(256),
                arg(&sy, rows * d),
                arg(&sdy, rows * d),
                arg(&sdx, rows * d),
                arg(&md, 1),
                256,
            );
        });
        eprintln!("[small speed dsl] softmax backward {rows}x{d}: {ms:.3} ms");

        let (rows, d) = (8192usize, 768usize);
        let (x, dy, al) = (
            h16(rnd(rows * d, 11, 1.0)),
            h16(rnd(rows * d, 13, 1.0)),
            h16(rnd(d, 17, 1.0)),
        );
        let (dx, st) = (h16(vec![0f32; rows * d]), h32(&vec![0f32; 2 * rows]));
        let eps = h32(&[1e-5]);
        let chunks = rows.div_ceil(64).clamp(1, 256);
        let per = rows.div_ceil(chunks);
        let part = h32(&vec![0f32; chunks * 2 * d]);
        let (md, pm) = (hu(&[d as u32]), hu(&[rows as u32, d as u32, per as u32]));
        let ms = time(&|| unsafe {
            layer_norm_back::launch_unchecked::<bf16, R>(
                &c,
                Grid::Static(rows as u32, 1, 1),
                Block::new_1d(256),
                arg(&x, rows * d),
                arg(&dy, rows * d),
                arg(&al, d),
                arg(&dx, rows * d),
                arg(&st, 2 * rows),
                arg(&eps, 1),
                arg(&md, 1),
                256,
            );
            layer_norm_params::launch_unchecked::<bf16, R>(
                &c,
                Grid::Static((d as u32).div_ceil(256), chunks as u32, 1),
                Block::new_1d(256),
                arg(&x, rows * d),
                arg(&dy, rows * d),
                arg(&st, 2 * rows),
                arg(&part, chunks * 2 * d),
                arg(&pm, 3),
            );
        });
        eprintln!("[small speed dsl] layer norm backward {rows}x{d}: {ms:.3} ms");

        let n = 768 * 2304;
        let (w, gw, m) = (
            h32(&rnd(n, 19, 1.0)),
            h32(&rnd(n, 23, 1.0)),
            h32(&rnd(n, 29, 0.1)),
        );
        let v = h32(&rnd(n, 31, 0.1).iter().map(|x| x * x).collect::<Vec<_>>());
        let p = h32(&[2e-5, 0.9, 0.98, 1e-6, 0.01, 1.9, 7.3, 1.0]);
        let nn = hu(&[n as u32]);
        let ms = time(&|| unsafe {
            crate::optim::adamw::launch_unchecked::<R>(
                &c,
                Grid::Static((n as u32).div_ceil(256), 1, 1),
                Block::new_1d(256),
                arg(&w, n),
                arg(&gw, n),
                arg(&m, n),
                arg(&v, n),
                arg(&p, 8),
                arg(&nn, 1),
            );
        });
        let blocks = (n as u32).div_ceil(256).min(4096);
        let (part, sm) = (h32(&vec![0f32; blocks as usize]), hu(&[n as u32, 0]));
        let ss = time(&|| unsafe {
            crate::optim::sumsq::launch_unchecked::<R>(
                &c,
                Grid::Static(blocks, 1, 1),
                Block::new_1d(256),
                arg(&gw, n),
                arg(&part, blocks as usize),
                arg(&sm, 2),
                256,
            );
        });
        eprintln!("[small speed dsl] adamw {n}: {ms:.3} ms, sum of squares {ss:.3} ms");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn row_gradients_on_rocm() {
        use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
        rows(&HipRuntime::client(&AmdDevice::default()), "rocm");
    }
}
