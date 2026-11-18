//! Ternary weight ops in the DSL: Prism PQ2_0, written once for every backend.
//!
//! PQ2_0 is the 2-bit codec of the Bonsai-2 / Zen6 line: 128 weights per block, one f16 scale `d`,
//! and a 2-bit code per weight, `code in {0,1,2,3}` standing for `{-1, 0, +1, +2}`:
//!
//! ```text
//!   w = d * (code - 1)
//! ```
//!
//! So a row dotted with an int8-quantized activation never needs the weights sign-extended: with
//! `q8` blocks of 32 (`xq`, scale `xs`, `xsum = xs * Σ xq`) the block contribution is
//!
//! ```text
//!   d * ( xs * Σ code·xq  -  xsum )
//! ```
//!
//! The `-1` rides on a sum the activation quantizer already produced, so the codes feed `dp4a`
//! straight off a 2-bit-to-int8 spread (four codes of a byte become the four int8 lanes of a word).
//! The decode kernel reads 2.125 bits per weight and does one dot-product instruction per four.
//!
//! Layout is planar (what a loader produces from the 34-byte GGUF blocks without shuffling): `wq`
//! holds the 2-bit codes, 16 per `u32` (element `16w + i` at bits `2i`, exactly the little-endian
//! reading of the GGUF `qs` bytes), row-major; `wd` holds the block scales as f32 (rounded through the
//! f16 the block stores).
//!
//! The schedule (threads per block, sub-blocks per thread) is a [`Space`] the tuner searches, so the
//! winner is a measured per-device fact: [`pq2_0_space`], [`Pq2MatvecEval`], `matvec-check pq2sweep`.

use crate::prelude::*;
use crate::tune::{Config, Evaluator, Space, Verdict};

/// Weights per PQ2_0 block.
pub const QK_PQ2_0: usize = 128;

/// Pack row-major 2-bit `codes` (`rows * k`, each `0..=3`) and per-block scales `d`
/// (`rows * k / 128`, rounded through f16 as the GGUF block stores them) into the planar
/// `(wq, wd)` the kernel reads: codes 16 per `u32`, scales as f32.
pub fn pq2_0_pack(codes: &[u8], d: &[f32], rows: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    assert_eq!(codes.len(), rows * k);
    assert!(k % 128 == 0, "k must be a multiple of the 128-weight block");
    let mut wq = vec![0u32; rows * k / 16];
    for (w, word) in wq.iter_mut().enumerate() {
        for i in 0..16 {
            *word |= (codes[w * 16 + i] as u32 & 3) << (2 * i);
        }
    }
    assert_eq!(d.len(), rows * k / QK_PQ2_0);
    let wd = d.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
    (wq, wd)
}

/// Deterministic synthetic weights: uniform codes and block scales that are exactly f16.
pub fn pq2_0_synth(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut x = seed | 1;
    let mut next = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let codes = (0..rows * k).map(|_| (next() >> 33) as u8 & 3).collect();
    let d = (0..rows * k / QK_PQ2_0)
        .map(|_| {
            half::f16::from_f32(0.005 + 0.03 * ((next() >> 40) as f32 / (1u32 << 24) as f32))
                .to_f32()
        })
        .collect();
    (codes, d)
}

/// The kernel's math on the host, in f64, from the SAME quantized activation the kernel reads
/// (`xq` packed four int8 per word, `xs`, `xsum` from [`crate::quant::quant_act_q8_cpu`]); the
/// only differences from the device are float summation order.
pub fn pq2_0_matvec_ref(
    codes: &[u8],
    d: &[f32],
    xq: &[u32],
    xs: &[f32],
    xsum: &[f32],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    let nsub = k / 32;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for sb in 0..nsub {
                let mut idot = 0i64;
                for e in 0..32 {
                    let word = xq[sb * 8 + e / 4];
                    let x = ((word >> (8 * (e % 4))) & 0xFF) as u8 as i8 as i64;
                    idot += codes[r * k + sb * 32 + e] as i64 * x;
                }
                let blk = r * (k / QK_PQ2_0) + sb / 4;
                acc += d[blk] as f64 * (xs[sb] as f64 * idot as f64 - xsum[sb] as f64);
            }
            acc as f32
        })
        .collect()
}

/// PQ2_0 decode matvec on int8 `dp4a`: one cube per output row, `nt` threads cooperating over the
/// row's 32-element sub-blocks, `chunk` consecutive sub-blocks per thread per step, then a shared
/// memory tree reduction. Each 32-bit word of codes covers 16 weights; each of its four bytes is
/// spread so its four 2-bit codes occupy the four int8 lanes of a word, and one `dot` against the
/// matching four activation int8s is a single `OpSDot` / `dp4a`.
///
/// `wq`/`wd`: planar weights (see the module docs). `xq`: int8 activation, 8 words per 32-block.
/// `xs`: per-block scale. `xsum`: per-block `xs * Σ xq`. `meta[0]` = `k`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn matvec_pq2_0_dp4a_blk<F: Float>(
    wq: &Array<u32>,
    wd: &Array<F>,
    xq: &Array<u32>,
    xs: &Array<F>,
    xsum: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] nt: usize,
    #[comptime] chunk: usize,
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let k = meta[0] as usize;
    let nsub = k / 32;
    let per = (nsub + nt * chunk - 1) / (nt * chunk);
    let mut partial = F::new(0.0f32);
    for j in 0..per {
        let base = (j * nt + t) * chunk;
        #[unroll]
        for c in 0..chunk {
            let sb = base + c;
            if sb < nsub {
                let blk = sb / 4;
                let d = wd[row * (k / 128) + blk];
                let wb = row * (k / 16) + sb * 2;
                let xg = sb * 8;
                let mut idot = 0i32;
                #[unroll]
                for wi in 0..2usize {
                    let w = wq[wb + wi];
                    #[unroll]
                    for e in 0..4usize {
                        let b = (w >> ((8 * e) as u32)) & 0xFFu32;
                        let spread = (b & 3u32)
                            | ((b & 0xCu32) << 6u32)
                            | ((b & 0x30u32) << 12u32)
                            | ((b & 0xC0u32) << 18u32);
                        let wv = Vector::<i32, Const<4>>::cast_from(
                            Vector::<i8, Const<4>>::reinterpret::<u32>(spread),
                        );
                        let xv = Vector::<i32, Const<4>>::cast_from(
                            Vector::<i8, Const<4>>::reinterpret::<u32>(xq[xg + wi * 4 + e]),
                        );
                        idot += wv.dot(xv);
                    }
                }
                partial += d * (xs[sb] * F::cast_from(idot) - xsum[sb]);
            }
        }
    }
    let mut smem = SharedMemory::<F>::new(nt);
    smem[t] = partial;
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
    if t == 0 {
        out[row] = smem[0];
    }
}

/// Schedule space of [`matvec_pq2_0_dp4a_blk`]: threads per row and sub-blocks per thread per step.
pub fn pq2_0_space() -> Space {
    Space::new()
        .param("NT", [32, 64, 128, 256])
        .param("CHUNK", [1, 2, 4])
}

/// Cold-weight, oracle-gated fitness for [`pq2_0_space`]: resident weight banks (distinct buffers, so a
/// rotation reads cold the way a layer-by-layer decode does), one q8 activation, the host reference.
pub struct Pq2MatvecEval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<(cubecl::server::Handle, cubecl::server::Handle)>,
    xq: cubecl::server::Handle,
    xs: cubecl::server::Handle,
    xsum: cubecl::server::Handle,
    meta: cubecl::server::Handle,
    out: cubecl::server::Handle,
    lens: (usize, usize, usize, usize), // wq, wd, xq words, blocks
    rows: usize,
    oracle: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> Pq2MatvecEval<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        codes: &[u8],
        d: &[f32],
        x: &[f32],
        rows: usize,
        k: usize,
        nbanks: usize,
        repeats: usize,
    ) -> Self {
        let (wq, wd) = pq2_0_pack(codes, d, rows, k);
        let (xq, xs, xsum) = crate::quant::quant_act_q8_cpu(x, 1, k);
        let banks = (0..nbanks.max(1))
            .map(|_| {
                (
                    client.create_from_slice(u32::as_bytes(&wq)),
                    client.create_from_slice(f32::as_bytes(&wd)),
                )
            })
            .collect();
        let oracle = pq2_0_matvec_ref(codes, d, &xq, &xs, &xsum, rows, k);
        let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        Self {
            client,
            space,
            banks,
            xq: client.create_from_slice(u32::as_bytes(&xq)),
            xs: client.create_from_slice(f32::as_bytes(&xs)),
            xsum: client.create_from_slice(f32::as_bytes(&xsum)),
            meta: client.create_from_slice(u32::as_bytes(&[k as u32])),
            out: client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows])),
            lens: (wq.len(), wd.len(), xq.len(), xs.len()),
            rows,
            oracle,
            maxref,
            repeats,
            worst_rel: std::cell::Cell::new(0.0),
        }
    }

    pub fn nbanks(&self) -> usize {
        self.banks.len()
    }

    pub fn worst_rel(&self) -> f32 {
        self.worst_rel.get()
    }

    fn cfg(&self, c: &Config) -> (usize, usize) {
        (
            c.get(self.space, "NT") as usize,
            c.get(self.space, "CHUNK") as usize,
        )
    }

    fn dispatch(&self, bank: usize, nt: usize, chunk: usize) {
        let (a, b) = &self.banks[bank];
        unsafe {
            matvec_pq2_0_dp4a_blk::launch_unchecked::<f32, R>(
                self.client,
                Grid::Static(self.rows as u32, 1, 1),
                Block::new_1d(nt as u32),
                ArrayArg::from_raw_parts(a.clone(), self.lens.0),
                ArrayArg::from_raw_parts(b.clone(), self.lens.1),
                ArrayArg::from_raw_parts(self.xq.clone(), self.lens.2),
                ArrayArg::from_raw_parts(self.xs.clone(), self.lens.3),
                ArrayArg::from_raw_parts(self.xsum.clone(), self.lens.3),
                ArrayArg::from_raw_parts(self.out.clone(), self.rows),
                ArrayArg::from_raw_parts(self.meta.clone(), 1),
                nt,
                chunk,
            );
        }
    }

    fn read_out(&self) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(self.out.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for Pq2MatvecEval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (nt, _) = self.cfg(cfg);
        let hw = &self.client.properties().hardware;
        if nt as u32 > hw.max_units_per_cube {
            return Verdict::Reject(format!(
                "{nt} threads exceed {} units per cube",
                hw.max_units_per_cube
            ));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (nt, chunk) = self.cfg(cfg);
        self.dispatch(0, nt, chunk);
        let rel = self
            .read_out()
            .iter()
            .zip(&self.oracle)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if !(rel <= 1e-3) {
            return f64::INFINITY;
        }
        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(i % nb, nt, chunk);
        }
        let _ = self.read_out();
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(i % nb, nt, chunk);
                }
                let _ = self.read_out();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Host f64 reference for direct float-activation PQ2_0 matvec (`w = d * (code - 1)`).
pub fn pq2_0_float_ref(codes: &[u8], d: &[f32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for b in 0..(k / QK_PQ2_0) {
                let scale = d[r * (k / QK_PQ2_0) + b] as f64;
                let base = b * QK_PQ2_0;
                let mut bacc = 0.0f64;
                for i in 0..QK_PQ2_0 {
                    let c = codes[r * k + base + i] as f64 - 1.0;
                    bacc += c * x[base + i] as f64;
                }
                acc += scale * bacc;
            }
            acc as f32
        })
        .collect()
}

/// Multi-row simdgroup/plane PQ2_0 decode matvec (`tg128` schedule family for SIMD4 float backends
/// such as Apple Silicon Metal where `dot(float4, float4)` outpaces emulated `dp4a`).
///
/// One 32-lane plane cooperates on `4` consecutive 128-weight blocks per step (`8` lanes/block,
/// `16` weights = `1` `u32` word per lane) across `nr0` output rows simultaneously, decoding each
/// 2-bit code via IEEE-754 binary16 exponent-25 (`as_type<f16>(0x6400 | code) - 1025.0 == code - 1`
/// bit-exactly, zero `uitofp`, zero `floor`, zero `sumy` cancellation) and reducing across the
/// plane with `plane_sum` (no shared memory, no barriers). Requires hardware `plane_size == 32`.
///
/// Launch geometry: `Block::new_2d(32, nsg as u32)` (`128` threads at `nsg = 4`),
/// `Grid::Static(rows.div_ceil(nsg * nr0) as u32, 1, 1)`. `meta = [k, rows]`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn matvec_pq2_0_plane<F: Float>(
    wq: &Array<u32>,
    wd: &Array<F>,
    x: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] nsg: usize,
    #[comptime] nr0: usize,
) {
    let lane = UNIT_POS_X as usize;
    let sg = UNIT_POS_Y as usize;
    let k = meta[0] as usize;
    let nout = meta[1] as usize;
    let nb = k / 128usize;
    let first_row = (CUBE_POS as usize * nsg + sg) * nr0;

    let ix = lane / 8usize;
    let il = lane % 8usize;

    let mut sumf = Array::<F>::new(nr0);
    #[unroll]
    for r in 0usize..nr0 {
        sumf[r] = F::new(0.0f32);
    }

    if first_row < nout {
        let mut ib = ix;
        while ib < nb {
            let xb = ib * 128usize + il * 16usize;
            let mut y = Array::<F>::new(16usize);
            #[unroll]
            for i in 0usize..16usize {
                y[i] = x[xb + i];
            }
            #[unroll]
            for r in 0usize..nr0 {
                let crow = if first_row + r < nout {
                    first_row + r
                } else {
                    nout - 1usize
                };
                let d = wd[crow * nb + ib];
                let w32 = wq[crow * (nb * 8usize) + ib * 8usize + il];
                let mut acc = F::new(0.0f32);
                #[unroll]
                for i in 0usize..16usize {
                    let c_bits = ((w32 >> ((2 * i) as u32)) & 3u32) | 0x4B00_0000u32;
                    let w_val =
                        F::cast_from(f32::reinterpret(c_bits) - 8_388_609.0f32);
                    acc += w_val * y[i];
                }
                sumf[r] += d * acc;
            }
            ib += 4usize;
        }
    }

    #[unroll]
    for r in 0usize..nr0 {
        let tot = plane_sum(sumf[r]);
        if lane == 0usize && first_row + r < nout {
            out[first_row + r] = tot;
        }
    }
}

/// Schedule space of [`matvec_pq2_0_plane`]: simdgroups per threadgroup (`NSG=4` -> `tg128`) and
/// output rows per simdgroup (`NR0`).
pub fn pq2_0_plane_space() -> Space {
    Space::new().param("NSG", [2, 4]).param("NR0", [2, 4, 8])
}

/// Cold-weight, oracle-gated fitness for [`pq2_0_plane_space`].
pub struct Pq2PlaneMatvecEval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<(cubecl::server::Handle, cubecl::server::Handle)>,
    x: cubecl::server::Handle,
    meta: cubecl::server::Handle,
    out: cubecl::server::Handle,
    lens: (usize, usize, usize),
    rows: usize,
    oracle: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> Pq2PlaneMatvecEval<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        codes: &[u8],
        d: &[f32],
        x: &[f32],
        rows: usize,
        k: usize,
        nbanks: usize,
        repeats: usize,
    ) -> Self {
        let (wq, wd) = pq2_0_pack(codes, d, rows, k);
        let banks = (0..nbanks.max(1))
            .map(|_| {
                (
                    client.create_from_slice(u32::as_bytes(&wq)),
                    client.create_from_slice(f32::as_bytes(&wd)),
                )
            })
            .collect();
        let oracle = pq2_0_float_ref(codes, &wd, x, rows, k);
        let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        Self {
            client,
            space,
            banks,
            x: client.create_from_slice(f32::as_bytes(x)),
            meta: client.create_from_slice(u32::as_bytes(&[k as u32, rows as u32])),
            out: client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows])),
            lens: (wq.len(), wd.len(), x.len()),
            rows,
            oracle,
            maxref,
            repeats,
            worst_rel: std::cell::Cell::new(0.0),
        }
    }

    pub fn nbanks(&self) -> usize {
        self.banks.len()
    }

    pub fn worst_rel(&self) -> f32 {
        self.worst_rel.get()
    }

    fn cfg(&self, c: &Config) -> (usize, usize) {
        (
            c.get(self.space, "NSG") as usize,
            c.get(self.space, "NR0") as usize,
        )
    }

    fn dispatch(&self, bank: usize, nsg: usize, nr0: usize) {
        let (a, b) = &self.banks[bank];
        let grid_x = self.rows.div_ceil(nsg * nr0) as u32;
        unsafe {
            matvec_pq2_0_plane::launch_unchecked::<f32, R>(
                self.client,
                Grid::Static(grid_x, 1, 1),
                Block::new_2d(32, nsg as u32),
                ArrayArg::from_raw_parts(a.clone(), self.lens.0),
                ArrayArg::from_raw_parts(b.clone(), self.lens.1),
                ArrayArg::from_raw_parts(self.x.clone(), self.lens.2),
                ArrayArg::from_raw_parts(self.out.clone(), self.rows),
                ArrayArg::from_raw_parts(self.meta.clone(), 2),
                nsg,
                nr0,
            );
        }
    }

    fn read_out(&self) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(self.out.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for Pq2PlaneMatvecEval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (nsg, _) = self.cfg(cfg);
        let hw = &self.client.properties().hardware;
        if hw.plane_size_min != 32 || hw.plane_size_max != 32 {
            return Verdict::Reject(format!(
                "matvec_pq2_0_plane requires 32-lane hardware planes, got {}..{}",
                hw.plane_size_min, hw.plane_size_max
            ));
        }
        if (32 * nsg) as u32 > hw.max_units_per_cube {
            return Verdict::Reject(format!(
                "{} threads exceed {} units per cube",
                32 * nsg,
                hw.max_units_per_cube
            ));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (nsg, nr0) = self.cfg(cfg);
        self.dispatch(0, nsg, nr0);
        let rel = self
            .read_out()
            .iter()
            .zip(&self.oracle)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if !(rel <= 1e-5) {
            return f64::INFINITY;
        }
        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(i % nb, nsg, nr0);
        }
        let _ = self.read_out();
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(i % nb, nsg, nr0);
                }
                let _ = self.read_out();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
}
