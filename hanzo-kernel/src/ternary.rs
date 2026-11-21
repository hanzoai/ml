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
                    let w_val = F::cast_from(f32::reinterpret(c_bits) - 8_388_609.0f32);
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

// ================================================================================================
// Native-layout PQ2_0: the kernel reads the GGUF `block_pq2_0` bytes the engine already holds.
// ================================================================================================
//
// `block_pq2_0` is 34 bytes: `half d` then `u8 qs[32]` (element `j` at `qs[j/4] >> 2*(j%4)`). That is
// 2-byte aligned, so a `u32` view of a block alternates between word-aligned and half-word-shifted.
// Two consecutive blocks are 68 bytes = exactly 17 words, so block `g` sits in pair `p = g/2`:
//
//   half h = g%2 | scale d                  | lane il's 16 codes (qs bytes 4il..4il+3)
//   -------------+--------------------------+----------------------------------------------
//   0 (even)     | low  16 bits of w[17p]   | (w[17p+il] >> 16) | (w[17p+il+1] << 16)
//   1 (odd)      | high 16 bits of w[17p+8] |  w[17p+9+il]
//
// Both rows are one branch-free expression with `s = 16*(1-h)`:
//   `(a >> s) | ((b << 15) << (17 - s))`   (s = 0 shifts `b` out in two legal sub-32 shifts)
// so every lane of a plane runs the same instructions whatever its block's parity. The `b` index is
// clamped to the last word, so no read leaves the tensor's `ceil(34*G/4)` words. When `G` is odd the
// final word is half outside the tensor; the binding must cover `ceil(34*G/4)` words (hanzo-ml's
// buffers are allocated in >= 4-byte granules).

/// The 32-bit little-endian word of 16 codes lane `il` (0..8) owns in global block `g`.
/// Even blocks (`h = 0`) straddle `wq[17p + il]` and `wq[17p + il + 1]` (max `17p + 8 < nwords`
/// since `nwords >= 17p + 9`); odd blocks (`h = 1`) are word-aligned at `wq[17p + 9 + il]`, so
/// `bi = ai + (1 - h)` never leaves the tensor and needs no runtime `nwords` branch.
#[device]
pub fn pq2_gguf_codes(wq: &Array<u32>, g: usize, il: usize) -> u32 {
    let p = g / 2usize;
    let h = g % 2usize;
    let ai = 17usize * p + 9usize * h + il;
    let bi = ai + (1usize - h);
    let s = (16usize * (1usize - h)) as u32;
    let a = wq[ai];
    let b = wq[bi];
    (a >> s) | ((b << 15u32) << (17u32 - s))
}

/// The f16 block scale of global block `g`, as f32.
#[device]
pub fn pq2_gguf_scale(wq: &Array<u32>, g: usize) -> f32 {
    let p = g / 2usize;
    let h = g % 2usize;
    let word = wq[17usize * p + 8usize * h];
    crate::quant::f16lo_to_f32((word >> ((16usize * h) as u32)) & 0xFFFFu32)
}

/// PQ2_0 decode matvec over native GGUF blocks: `out[r] = Σ_j d·(code-1)·x[j]`.
///
/// Same plane schedule as [`matvec_pq2_0_plane`] (8 lanes per block, 4 blocks per plane step, `nr0`
/// rows per plane, `nsg` planes per cube, `plane_sum` reduce, no shared memory), reading the
/// engine's resident bytes directly. Accumulates in f32 for every `F`.
///
/// Codes decode bit-exactly via exponent-24 (`f32::reinterpret(0x4B00_0000 | c) - 8388609.0 == c - 1`)
/// with zero cancellation; row bounds are pre-clamped outside the `K` loop and word-straddle loads
/// are branch-free (`bi = ai + (1 - h)`). The block scale is hardware f16->f32 on `metal | cuda | rocm`,
/// with the portable `f16lo_to_f32` fallback elsewhere.
///
/// Launch: `Block::new_2d(32, nsg)`, `Grid::Static(rows.div_ceil(nsg*nr0), 1, 1)`,
/// `meta = [k, rows, nwords]` with `nwords = ceil(34 * rows * k/128 / 4)`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn matvec_pq2_0_gguf<F: Float>(
    wq: &Array<u32>,
    x: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] nsg: usize,
    #[comptime] nr0: usize,
    #[comptime] tgt: Target,
) {
    let lane = UNIT_POS_X as usize;
    let sg = UNIT_POS_Y as usize;
    let k = meta[0] as usize;
    let nout = meta[1] as usize;
    let nb = k / 128usize;
    let first_row = (CUBE_POS as usize * nsg + sg) * nr0;
    let ix = lane / 8usize;
    let il = lane % 8usize;

    let mut sumf = Array::<f32>::new(nr0);
    let mut crows = Array::<usize>::new(nr0);
    #[unroll]
    for r in 0usize..nr0 {
        sumf[r] = 0.0f32;
        let mut row = first_row + r;
        if row >= nout {
            row = nout - 1usize;
        }
        crows[r] = row;
    }

    if first_row < nout {
        let mut ib = ix;
        while ib < nb {
            let xb = ib * 128usize + il * 16usize;
            let mut y = Array::<f32>::new(16usize);
            #[unroll]
            for i in 0usize..16usize {
                y[i] = f32::cast_from(x[xb + i]);
            }
            #[unroll]
            for r in 0usize..nr0 {
                let g = crows[r] * nb + ib;
                let sword = wq[17usize * (g / 2usize) + 8usize * (g % 2usize)];
                let sbits = (sword >> ((16usize * (g % 2usize)) as u32)) & 0xFFFFu32;
                let d = island! {
                    metal | cuda | rocm => {
                        f32::cast_from(Vector::<half::f16, Const<2>>::reinterpret::<u32>(sbits)[0])
                    }
                    default => { crate::quant::f16lo_to_f32(sbits) }
                };
                let w32 = pq2_gguf_codes(wq, g, il);
                let mut acc = 0.0f32;
                #[unroll]
                for i in 0usize..16usize {
                    let c_bits = ((w32 >> ((2 * i) as u32)) & 3u32) | 0x4B00_0000u32;
                    acc += (f32::reinterpret(c_bits) - 8_388_609.0f32) * y[i];
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
            out[first_row + r] = F::cast_from(tot);
        }
    }
}

/// Serialize row-major `codes` + f16-exact scales `d` into GGUF `block_pq2_0` bytes (34 B/block),
/// bit-for-bit what the engine loads from a `.gguf`.
pub fn pq2_0_gguf_bytes(codes: &[u8], d: &[f32], rows: usize, k: usize) -> Vec<u8> {
    assert_eq!(codes.len(), rows * k);
    assert!(k % QK_PQ2_0 == 0);
    let nblk = rows * k / QK_PQ2_0;
    assert_eq!(d.len(), nblk);
    let mut out = Vec::with_capacity(nblk * 34);
    for b in 0..nblk {
        out.extend_from_slice(&half::f16::from_f32(d[b]).to_le_bytes());
        for q in 0..32 {
            let base = b * QK_PQ2_0 + q * 4;
            let mut byte = 0u8;
            for s in 0..4 {
                byte |= (codes[base + s] & 3) << (2 * s);
            }
            out.push(byte);
        }
    }
    out
}

/// View GGUF bytes as the `u32` words the kernel binds (`ceil(len/4)`, zero tail).
pub fn pq2_0_gguf_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

/// The oracle: decode the GGUF bytes exactly as hanzo-ml's `BlockPQ2_0::to_float` does
/// (`(q - 1) * d.to_f32()`), dot in f64.
pub fn pq2_0_gguf_ref(bytes: &[u8], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / QK_PQ2_0;
    (0..rows)
        .map(|r| {
            let mut acc = 0f64;
            for b in 0..nb {
                let blk = &bytes[(r * nb + b) * 34..(r * nb + b + 1) * 34];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32() as f64;
                for j in 0..QK_PQ2_0 {
                    let q = ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) as f64;
                    acc += (q - 1.0) * d * x[b * QK_PQ2_0 + j] as f64;
                }
            }
            acc as f32
        })
        .collect()
}

/// Launch [`matvec_pq2_0_gguf`] once and read the result back (f32 activation).
#[allow(clippy::too_many_arguments)]
pub fn matvec_pq2_0_gguf_run<R: Runtime>(
    client: &ComputeClient<R>,
    words: &[u32],
    x: &[f32],
    rows: usize,
    k: usize,
    nsg: usize,
    nr0: usize,
) -> Vec<f32> {
    matvec_pq2_0_gguf_run_as::<f32, R>(client, words, x, rows, k, nsg, nr0)
}

/// Launch [`matvec_pq2_0_gguf`] with activation/output element `E` (f32 or bf16).
#[allow(clippy::too_many_arguments)]
pub fn matvec_pq2_0_gguf_run_as<E: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    words: &[u32],
    x: &[E],
    rows: usize,
    k: usize,
    nsg: usize,
    nr0: usize,
) -> Vec<E> {
    let wh = client.create_from_slice(u32::as_bytes(words));
    let xh = client.create_from_slice(E::as_bytes(x));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::default(); rows]));
    let mh = client.create_from_slice(u32::as_bytes(&[k as u32, rows as u32, words.len() as u32]));
    unsafe {
        matvec_pq2_0_gguf::launch_unchecked::<E, R>(
            client,
            Grid::Static(rows.div_ceil(nsg * nr0) as u32, 1, 1),
            Block::new_2d(32, nsg as u32),
            ArrayArg::from_raw_parts(wh.clone(), words.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            ArrayArg::from_raw_parts(mh.clone(), 3),
            nsg,
            nr0,
            Target::of(client),
        );
    }
    E::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// The production schedule for a PQ2_0 decode matvec with `rows` outputs: NSG = 2 (64 threads),
/// NR0 = 4 at every size. Measured for THIS kernel on dbc M4 Max (`hanzo-ml` test
/// `pq2_dsl_nr0_sweep`, 27B decode shapes, cold-bank rotation, interleaved): `(2, 4)` beats legacy
/// on 7 of 8 shapes (+4.0% to +59%: 1024x5120 13.8 vs 22.0 us, 5120x6144 30.3 vs 34.0 us,
/// 6144x5120 30.2 vs 33.1 us, 10240x5120 43.9 vs 45.9 us, 12288x5120 51.8 vs 54.5 us,
/// 17408x5120 70.0 vs 75.0 us, 5120x17408 75.4 vs 78.4 us) and is within 2.6% at 248320x5120
/// (849.0 vs 826.7 us). The single source of truth hanzo-ml's dispatch and kernel-gen both read.
pub fn pq2_0_gguf_schedule(rows: usize) -> (usize, usize) {
    let _ = rows;
    (2, 4)
}

/// Shapes every backend is gated on: odd block counts (the half-word path and the clamped tail),
/// a single block, and Zen6-27B-class projection shapes.
pub const PQ2_GGUF_SHAPES: &[(usize, usize)] = &[
    (1, 128),
    (3, 128),
    (7, 384),
    (33, 640),
    (257, 1152),
    (2048, 2048),
    (5120, 5120),
    (17408, 5120),
    (5120, 17408),
];

/// Correctness board for one runtime: every shape x every schedule, scale-relative error vs the
/// f64 oracle. Returns (rows, k, nsg, nr0, rel) per case.
pub fn pq2_0_gguf_board<R: Runtime>(
    client: &ComputeClient<R>,
) -> Vec<(usize, usize, usize, usize, f32)> {
    let mut res = Vec::new();
    for &(rows, k) in PQ2_GGUF_SHAPES {
        let (codes, d) = pq2_0_synth(rows, k, 0x9E37_79B9 ^ (rows * 31 + k) as u64);
        let bytes = pq2_0_gguf_bytes(&codes, &d, rows, k);
        let words = pq2_0_gguf_words(&bytes);
        let x: Vec<f32> = (0..k)
            .map(|i| (((i * 7919) % 2003) as f32 / 1001.5) - 1.0)
            .collect();
        let want = pq2_0_gguf_ref(&bytes, &x, rows, k);
        let maxref = want.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-30);
        for nsg in [2usize, 4] {
            for nr0 in [2usize, 4, 8] {
                let got = matvec_pq2_0_gguf_run(client, &words, &x, rows, k, nsg, nr0);
                let rel = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max)
                    / maxref;
                res.push((rows, k, nsg, nr0, rel));
            }
        }
    }
    res
}

#[cfg(test)]
mod gguf_layout_tests {
    use super::*;

    /// The host-side model of `pq2_gguf_codes` / `pq2_gguf_scale` must reproduce the byte decode for
    /// every block parity, lane, and the clamped tail -- proven on the CPU before any GPU runs it.
    #[test]
    fn word_view_matches_byte_decode() {
        for &(rows, k) in &[(1usize, 128usize), (3, 128), (7, 384), (5, 256)] {
            let (codes, d) = pq2_0_synth(rows, k, 7);
            let bytes = pq2_0_gguf_bytes(&codes, &d, rows, k);
            let w = pq2_0_gguf_words(&bytes);
            let nwords = w.len();
            for g in 0..rows * k / QK_PQ2_0 {
                let (p, h) = (g / 2, g % 2);
                let sc = (w[17 * p + 8 * h] >> (16 * h)) & 0xFFFF;
                assert_eq!(
                    half::f16::from_bits(sc as u16).to_f32(),
                    half::f16::from_f32(d[g]).to_f32()
                );
                for il in 0..8 {
                    let ai = 17 * p + 9 * h + il;
                    let bi = (ai + 1).min(nwords - 1);
                    let s = 16 * (1 - h) as u32;
                    let v = (w[ai] >> s) | ((w[bi] << 15) << (17 - s));
                    for i in 0..16 {
                        let want = codes[g * QK_PQ2_0 + il * 16 + i] as u32 & 3;
                        assert_eq!((v >> (2 * i)) & 3, want, "g={g} il={il} i={i}");
                    }
                }
            }
        }
    }

    #[test]
    fn oracle_matches_planar_decode() {
        let (rows, k) = (5, 384);
        let (codes, d) = pq2_0_synth(rows, k, 11);
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
        let bytes = pq2_0_gguf_bytes(&codes, &d, rows, k);
        let (_, wd) = pq2_0_pack(&codes, &d, rows, k);
        let a = pq2_0_gguf_ref(&bytes, &x, rows, k);
        let b = pq2_0_float_ref(&codes, &wd, &x, rows, k);
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() <= 1e-5 * v.abs().max(1.0), "{u} vs {v}");
        }
    }
}

// ================================================================================================
// Compiler regression fixture: the vendored cubecl-cpp `HANZO PATCH (bitcast-of-literal)`.
// ================================================================================================

/// Decode f16 bit patterns through [`crate::quant::f16lo_to_f32`]. Its subnormal path bitcasts a
/// constant (`f32::reinterpret(113 << 23)`), which upstream cubecl-cpp lowered to an invalid MSL
/// reference cast of an rvalue. Running this through each backend's real compiler is the regression
/// gate for the patch: Metal (patched path), CUDA/HIP (same shared emitter, must stay equivalent),
/// Vulkan (SPIR-V, untouched).
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn f16_decode_probe(bits: &Array<u32>, out: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < bits.len() {
        out[i] = crate::quant::f16lo_to_f32(bits[i]);
    }
}

/// Every f16 class: zeros, smallest/largest subnormals, normals, max, infinities, and a run of
/// random patterns (NaNs are excluded: their payload is not part of the contract).
pub fn f16_probe_bits() -> Vec<u32> {
    let mut v: Vec<u32> = vec![
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00, 0x3555,
        0x7BFF, 0xFBFF, 0x7C00, 0xFC00,
    ];
    let mut x = 0x2545_F491u32;
    while v.len() < 4096 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        let h = x & 0xFFFF;
        if (h & 0x7C00) != 0x7C00 || (h & 0x03FF) == 0 {
            v.push(h);
        }
    }
    v
}

/// Run the probe; returns the number of patterns whose decode differs from `half` bit-for-bit.
pub fn f16_decode_probe_run<R: Runtime>(client: &ComputeClient<R>) -> usize {
    let bits = f16_probe_bits();
    let bh = client.create_from_slice(u32::as_bytes(&bits));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0f32; bits.len()]));
    unsafe {
        f16_decode_probe::launch_unchecked::<R>(
            client,
            Grid::Static(bits.len().div_ceil(256) as u32, 1, 1),
            Block::new_1d(256),
            ArrayArg::from_raw_parts(bh, bits.len()),
            ArrayArg::from_raw_parts(oh.clone(), bits.len()),
        );
    }
    let got = f32::from_bytes(&client.read_one_unchecked(oh)).to_vec();
    bits.iter()
        .zip(&got)
        .filter(|(b, g)| half::f16::from_bits(**b as u16).to_f32().to_bits() != g.to_bits())
        .count()
}

// ================================================================================================
// Prefill GEMM (M > 1) over native GGUF PQ2_0 blocks: one DSL source for Metal, CUDA, Vulkan, ROCm.
// ================================================================================================

/// Stage a `bm x 32` activation tile `x[m0..m0+bm, k0..k0+32]` into stride-40 padded shared
/// memory (`+8 half` skew avoids bank conflicts during `cmma` loads).
#[device]
#[allow(clippy::too_many_arguments)]
fn stage_sa_tile<F: Float>(
    x: &Array<F>,
    sa: &mut SharedMemory<half::f16>,
    m0: usize,
    m: usize,
    k: usize,
    k0: usize,
    tid: usize,
    #[comptime] bm: usize,
) {
    let ak_base = (tid % 4usize) * 8usize;
    let tid_m = tid / 4usize;
    #[unroll]
    for mr in 0usize..(bm / 32usize) {
        let local_m = mr * 32usize + tid_m;
        let arow = m0 + local_m;
        let sa_base = local_m * 40usize + ak_base;
        if arow < m {
            let x_base = arow * k + k0 + ak_base;
            #[unroll]
            for i in 0usize..8usize {
                sa[sa_base + i] = half::f16::cast_from(f32::cast_from(x[x_base + i]));
            }
        } else {
            #[unroll]
            for i in 0usize..8usize {
                sa[sa_base + i] = half::f16::cast_from(0.0f32);
            }
        }
    }
}

/// Stage one `bm x 32` activation tile `sa` and `32 x 64` dequantized PQ2_0 weight tile `sb`
/// (both `half::f16`) for block-K step `kb` (`k0 = kb * 32`).
#[device]
#[allow(clippy::too_many_arguments)]
fn pq2_stage_ab_tile<F: Float>(
    wq: &Array<u32>,
    x: &Array<F>,
    sa: &mut SharedMemory<half::f16>,
    sb: &mut SharedMemory<half::f16>,
    m0: usize,
    n0: usize,
    m: usize,
    n: usize,
    k: usize,
    kb: usize,
    tid: usize,
    #[comptime] bm: usize,
    #[comptime] tgt: Target,
) {
    let k0 = kb * 32usize;
    let nb = k / 128usize;
    stage_sa_tile(x, sa, m0, m, k, k0, tid, bm);
    // Stage 32 x 64 [K, N] RowMajor weight tile into stride-72 padded buffer (+8 half skew).
    let local_n = tid / 2usize;
    let half_k = tid % 2usize;
    let nrow = n0 + local_n;
    let sb_k0 = half_k * 16usize;
    if nrow < n {
        let g = nrow * nb + (kb / 4usize);
        let sword = wq[17usize * (g / 2usize) + 8usize * (g % 2usize)];
        let sbits = (sword >> ((16usize * (g % 2usize)) as u32)) & 0xFFFFu32;
        let hw_f16 = comptime!(matches!(tgt, Target::Metal | Target::Cuda | Target::Rocm));
        let d = if hw_f16 {
            f32::cast_from(Vector::<half::f16, Const<2>>::reinterpret::<u32>(sbits)[0])
        } else {
            crate::quant::f16lo_to_f32(sbits)
        };
        let il = 2usize * (kb % 4usize) + half_k;
        let w32 = pq2_gguf_codes(wq, g, il);
        #[unroll]
        for i in 0usize..16usize {
            let c_bits = ((w32 >> ((2 * i) as u32)) & 3u32) | 0x4B00_0000u32;
            let w = (f32::reinterpret(c_bits) - 8_388_609.0f32) * d;
            sb[(sb_k0 + i) * 72usize + local_n] = half::f16::cast_from(w);
        }
    } else {
        #[unroll]
        for i in 0usize..16usize {
            sb[(sb_k0 + i) * 72usize + local_n] = half::f16::cast_from(0.0f32);
        }
    }
}

/// Cooperatively flush one warp's `16 x 32` accumulator staging tile `sc` to `out[m, n]`.
#[device]
#[allow(clippy::too_many_arguments)]
fn pq2_flush_warp_tile<F: Float>(
    sc: &SharedMemory<f32>,
    out: &mut Array<F>,
    tid: usize,
    tile_m0: usize,
    tile_n0: usize,
    m: usize,
    n: usize,
    #[comptime] plane: usize,
) {
    sync_cube();
    if comptime!(plane == 32usize) {
        #[unroll]
        for i in 0usize..4usize {
            let idx = tid + i * 128usize;
            let orow = tile_m0 + idx / 32usize;
            let ocol = tile_n0 + idx % 32usize;
            if orow < m && ocol < n {
                out[orow * n + ocol] = F::cast_from(sc[idx]);
            }
        }
    } else if tid < 128usize {
        #[unroll]
        for i in 0usize..4usize {
            let idx = tid + i * 128usize;
            let orow = tile_m0 + idx / 32usize;
            let ocol = tile_n0 + idx % 32usize;
            if orow < m && ocol < n {
                out[orow * n + ocol] = F::cast_from(sc[idx]);
            }
        }
    }
    sync_cube();
}

/// PQ2_0 prefill GEMM over native GGUF blocks: `out[m, n] = x[m, k] * W[n, k]^T`.
///
/// Tiling: `bm` (`32` or `64`) tokens, `BN = 64` output rows, `BK = 32` contraction step; `4`
/// planes (`4 * plane` threads: `128` for 32-lane warps/wave32, `256` for 64-lane AMD Vulkan
/// subgroups) arranged `2 x 2`, each owning a `(bm/2) x 32` output sub-tile.
/// Both `sa` (`[bm, BK]`, stride `40`) and `sb` (`[BK, BN]`, stride `72`) are `+8 half`-skewed
/// `RowMajor` so `cmma` loads use `transpose = false` without shared-bank conflicts, and the
/// epilogue shares one `16 x 32` (`2 KB`) `sc` tile across warps (`9216 B` at `bm=32`, `11776 B` at `bm=64`).
/// - `metal`: `8x8x8` `simdgroup_matrix` (`(bm/16) x 4` accumulators per warp, 4 `dk` steps per `BK=32`).
/// - `cuda | rocm | vulkan`: `16x16x16` cooperative matrix (`(bm/32) x 2` accumulators per warp, 2 `dk` steps).
/// - `default`: scalar f16-staged reference contraction on the same shared tiles.
///
/// Launch: `Block::new_1d(4 * plane)`, `Grid::Static(m.div_ceil(bm) as u32, n.div_ceil(64) as u32, 1)`,
/// `meta = [m, n, k]`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm, cpu), unchecked)]
pub fn matmul_pq2_0_gguf<F: Float>(
    wq: &Array<u32>,
    x: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] bm: usize,
    #[comptime] plane: usize,
    #[comptime] tgt: Target,
) {
    let tid = UNIT_POS as usize;
    let warp = tid / plane;
    let warp_m = warp / 2usize;
    let warp_n = warp % 2usize;
    let m = meta[0] as usize;
    let n = meta[1] as usize;
    let k = meta[2] as usize;
    let m0 = CUBE_POS_X as usize * bm;
    let n0 = CUBE_POS_Y as usize * 64usize;
    let kb_count = k / 32usize;
    let warp_rows = bm / 2usize;

    let mut sa = SharedMemory::<half::f16>::new(bm * 40usize);
    let mut sb = SharedMemory::<half::f16>::new(32usize * 72usize);
    let mut sc = SharedMemory::<f32>::new(16usize * 32usize);

    island! {
        metal => {
            let wm_tiles = bm / 16usize;
            let mut acc = Sequence::<cmma::Matrix<f32>>::new();
            #[unroll]
            for _i in 0usize..(wm_tiles * 4usize) {
                acc.push(cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    8usize,
                    8usize,
                    8usize,
                    cmma::MatrixLayout::Undefined,
                    0.0f32,
                ));
            }
            for kb in 0..kb_count {
                pq2_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                sync_cube();
                #[unroll]
                for dk in 0usize..4usize {
                    let mut mb = Sequence::<cmma::Matrix<half::f16>>::new();
                    #[unroll]
                    for jn in 0usize..4usize {
                        let b_col = warp_n * 32usize + jn * 8usize;
                        mb.push(cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::B,
                            8usize,
                            8usize,
                            8usize,
                            cmma::MatrixLayout::RowMajor,
                            &sb.slice((dk * 8usize) * 72usize + b_col, 32usize * 72usize),
                            72u32,
                        ));
                    }
                    #[unroll]
                    for im in 0usize..wm_tiles {
                        let a_row = warp_m * warp_rows + im * 8usize;
                        let ma = cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::A,
                            8usize,
                            8usize,
                            8usize,
                            cmma::MatrixLayout::RowMajor,
                            &sa.slice(a_row * 40usize + dk * 8usize, bm * 40usize),
                            40u32,
                        );
                        #[unroll]
                        for jn in 0usize..4usize {
                            let c = acc.index(im * 4usize + jn);
                            cmma::execute::<half::f16, half::f16, f32, f32>(&ma, mb.index(jn), c, c);
                        }
                    }
                }
                sync_cube();
            }
            #[unroll]
            for w in 0usize..4usize {
                #[unroll]
                for rh in 0usize..(bm / 32usize) {
                    if warp == w {
                        #[unroll]
                        for im in 0usize..2usize {
                            #[unroll]
                            for jn in 0usize..4usize {
                                let c_off = (im * 8usize) * 32usize + jn * 8usize;
                                cmma::store(
                                    &mut sc.slice_mut(c_off, 16usize * 32usize),
                                    acc.index((rh * 2usize + im) * 4usize + jn),
                                    32u32,
                                    cmma::MatrixLayout::RowMajor,
                                );
                            }
                        }
                    }
                    pq2_flush_warp_tile(
                        &sc,
                        out,
                        tid,
                        m0 + (w / 2usize) * warp_rows + rh * 16usize,
                        n0 + (w % 2usize) * 32usize,
                        m,
                        n,
                        plane,
                    );
                }
            }
        }
        cuda | rocm | vulkan => {
            let wm_tiles = bm / 32usize;
            let mut acc = Sequence::<cmma::Matrix<f32>>::new();
            #[unroll]
            for _i in 0usize..(wm_tiles * 2usize) {
                acc.push(cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0.0f32,
                ));
            }
            for kb in 0..kb_count {
                if comptime!(plane == 32usize) {
                    pq2_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                } else if tid < 128usize {
                    pq2_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                }
                sync_cube();
                #[unroll]
                for dk in 0usize..2usize {
                    let mut mb = Sequence::<cmma::Matrix<half::f16>>::new();
                    #[unroll]
                    for jn in 0usize..2usize {
                        let b_col = warp_n * 32usize + jn * 16usize;
                        mb.push(cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::B,
                            16usize,
                            16usize,
                            16usize,
                            cmma::MatrixLayout::RowMajor,
                            &sb.slice((dk * 16usize) * 72usize + b_col, 32usize * 72usize),
                            72u32,
                        ));
                    }
                    #[unroll]
                    for im in 0usize..wm_tiles {
                        let a_row = warp_m * warp_rows + im * 16usize;
                        let ma = cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::A,
                            16usize,
                            16usize,
                            16usize,
                            cmma::MatrixLayout::RowMajor,
                            &sa.slice(a_row * 40usize + dk * 16usize, bm * 40usize),
                            40u32,
                        );
                        #[unroll]
                        for jn in 0usize..2usize {
                            let c = acc.index(im * 2usize + jn);
                            cmma::execute::<half::f16, half::f16, f32, f32>(&ma, mb.index(jn), c, c);
                        }
                    }
                }
                sync_cube();
            }
            #[unroll]
            for w in 0usize..4usize {
                #[unroll]
                for rh in 0usize..wm_tiles {
                    if warp == w {
                        #[unroll]
                        for jn in 0usize..2usize {
                            cmma::store(
                                &mut sc.slice_mut(jn * 16usize, 16usize * 32usize),
                                acc.index(rh * 2usize + jn),
                                32u32,
                                cmma::MatrixLayout::RowMajor,
                            );
                        }
                    }
                    pq2_flush_warp_tile(
                        &sc,
                        out,
                        tid,
                        m0 + (w / 2usize) * warp_rows + rh * 16usize,
                        n0 + (w % 2usize) * 32usize,
                        m,
                        n,
                        plane,
                    );
                }
            }
        }
        default => {
            let mut acc = Array::<f32>::new(warp_rows);
            #[unroll]
            for i in 0usize..warp_rows {
                acc[i] = 0.0f32;
            }
            let half_m = tid / 64usize;
            let col = tid % 64usize;
            for kb in 0..kb_count {
                pq2_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                sync_cube();
                #[unroll]
                for i in 0usize..warp_rows {
                    let row = half_m * warp_rows + i;
                    let mut sum = 0.0f32;
                    for kk in 0usize..32usize {
                        sum += f32::cast_from(sa[row * 40usize + kk])
                            * f32::cast_from(sb[kk * 72usize + col]);
                    }
                    acc[i] += sum;
                }
                sync_cube();
            }
            #[unroll]
            for i in 0usize..warp_rows {
                let orow = m0 + half_m * warp_rows + i;
                let ocol = n0 + col;
                if orow < m && ocol < n {
                    out[orow * n + ocol] = F::cast_from(acc[i]);
                }
            }
        }
    }
}

/// Host f64 reference for PQ2_0 GEMM (`out[m, n] = x[m, k] * W[n, k]^T`).
pub fn pq2_0_gguf_gemm_ref(
    bytes: &[u8],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let nb = k / QK_PQ2_0;
    let mut w = vec![0.0f64; n * k];
    for r in 0..n {
        for b in 0..nb {
            let blk = &bytes[(r * nb + b) * 34..(r * nb + b + 1) * 34];
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32() as f64;
            for j in 0..QK_PQ2_0 {
                let q = ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) as f64;
                w[r * k + b * QK_PQ2_0 + j] = (q - 1.0) * d;
            }
        }
    }
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += x[row * k + j] as f64 * w[col * k + j];
            }
            out[row * n + col] = acc as f32;
        }
    }
    out
}

/// Launch [`matmul_pq2_0_gguf`] with explicit `bm` (`32` or `64`) once and read `[m, n]` back.
pub fn matmul_pq2_0_gguf_bm_run_as<E: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    words: &[u32],
    x: &[E],
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
) -> Vec<E> {
    let tgt = Target::of(client);
    let plane = if matches!(tgt, Target::Vulkan) {
        client.properties().hardware.plane_size_max as usize
    } else {
        32usize
    };
    let wh = client.create_from_slice(u32::as_bytes(words));
    let xh = client.create_from_slice(E::as_bytes(x));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::default(); m * n]));
    let mh = client.create_from_slice(u32::as_bytes(&[m as u32, n as u32, k as u32]));
    unsafe {
        matmul_pq2_0_gguf::launch_unchecked::<E, R>(
            client,
            Grid::Static(m.div_ceil(bm) as u32, n.div_ceil(64) as u32, 1),
            Block::new_1d((4 * plane) as u32),
            ArrayArg::from_raw_parts(wh.clone(), words.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            ArrayArg::from_raw_parts(mh.clone(), 3),
            bm,
            plane,
            tgt,
        );
    }
    E::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Launch [`matmul_pq2_0_gguf`] picking `bm = 32` for small `m` (`<= 48`) and `64` above.
pub fn matmul_pq2_0_gguf_run_as<E: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    words: &[u32],
    x: &[E],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<E> {
    let bm = if m <= 48 { 32 } else { 64 };
    matmul_pq2_0_gguf_bm_run_as(client, words, x, m, n, k, bm)
}

/// Shapes every backend is gated on for PQ2_0 prefill GEMM: ragged small tiles + 27B prefill tiles.
pub const PQ2_GGUF_GEMM_SHAPES: &[(usize, usize, usize)] = &[
    (1, 64, 128),
    (17, 73, 256),
    (33, 257, 384),
    (64, 1024, 5120),
    (128, 5120, 5120),
    (512, 6144, 5120),
];

/// Correctness board for [`matmul_pq2_0_gguf`]: scale-relative error vs f64 oracle (`f32` and `bf16`).
pub fn pq2_0_gguf_gemm_board<R: Runtime>(
    client: &ComputeClient<R>,
) -> Vec<(usize, usize, usize, f32, f32)> {
    let mut res = Vec::new();
    for &(m, n, k) in PQ2_GGUF_GEMM_SHAPES {
        let (codes, d) = pq2_0_synth(n, k, 0xA5A5_1234 ^ (m * 17 + n * 31 + k) as u64);
        let bytes = pq2_0_gguf_bytes(&codes, &d, n, k);
        let words = pq2_0_gguf_words(&bytes);
        let x: Vec<f32> = (0..m * k)
            .map(|i| (((i * 7919) % 2003) as f32 / 1001.5) - 1.0)
            .collect();
        let want = pq2_0_gguf_gemm_ref(&bytes, &x, m, n, k);
        let maxref = want.iter().fold(0f32, |mx, &v| mx.max(v.abs())).max(1e-30);
        let got32 = matmul_pq2_0_gguf_run_as::<f32, R>(client, &words, &x, m, n, k);
        let rel32 = got32
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / maxref;
        let xb: Vec<half::bf16> = x.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let xb_f: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
        let want_b = pq2_0_gguf_gemm_ref(&bytes, &xb_f, m, n, k);
        let maxref_b = want_b
            .iter()
            .fold(0f32, |mx, &v| mx.max(v.abs()))
            .max(1e-30);
        let got_b = matmul_pq2_0_gguf_run_as::<half::bf16, R>(client, &words, &xb, m, n, k);
        let rel_b = got_b
            .iter()
            .zip(&want_b)
            .map(|(a, b)| (a.to_f32() - b).abs())
            .fold(0f32, f32::max)
            / maxref_b;
        res.push((m, n, k, rel32, rel_b));
    }
    res
}

// ================================================================================================
// PTQ1_0 (GGUF type 143, 128-element base-3 ternary in 28 bytes = 7 u32 words) in the Hanzo DSL.
// ================================================================================================

/// Elements per native GGUF `block_ptq1_0`.
pub const QK_PTQ1_0: usize = 128;

/// Decode one trit (`-1.0, 0.0, +1.0`) from packed byte `b` (`0..=255`) at base-3 digit power `pow3`
/// (`1, 3, 9, 27, 81`) via exact integer extraction + exponent-24 bitcast (`zero floor/cancel`).
#[device]
#[inline]
fn ptq1_trit(b: u32, pow3: u32) -> f32 {
    let t = (((b * pow3) & 0xFFu32) * 3u32) >> 8u32;
    f32::reinterpret(t | 0x4B00_0000u32) - 8_388_609.0f32
}

/// Dot product of the 5 packed trits in byte `b` (`0..=255`) using base-3 collapse against
/// precomputed activation differences `c0 = y0 - 3*y1`, ..., `c3 = y3 - 3*y4`, `c4 = y4`.
/// Exact in fp32 since `1/256 = 2^-8` and `243 * 255 = 61,965 < 2^24`; caller subtracts `sumy`.
#[device]
#[inline]
#[allow(clippy::too_many_arguments)]
fn ptq1_dot_5_collapse(b: u32, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> f32 {
    let u = f32::cast_from(b) * (1.0f32 / 256.0f32);
    f32::floor(3.0f32 * u) * c0
        + f32::floor(9.0f32 * u) * c1
        + f32::floor(27.0f32 * u) * c2
        + f32::floor(81.0f32 * u) * c3
        + f32::floor(243.0f32 * u) * c4
}

/// PTQ1_0 decode matvec over native GGUF `block_ptq1_0` (`28 B = 7 u32` words): `out[r] = Σ_j d·t·x[j]`.
///
/// Zero word-straddling (`7 * g` aligned `u32` words), base-3 collapse (`floor(3^k * b / 256)`) with
/// precomputed activation differences and `sumy` shared across `NR0` output rows per plane.
///
/// Launch: `Block::new_2d(32, nsg)`, `Grid::Static(rows.div_ceil(nsg*nr0), 1, 1)`,
/// `meta = [k, rows, nwords]` with `nwords = 7 * rows * (k / 128)`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn matvec_ptq1_0_gguf<F: Float>(
    wq: &Array<u32>,
    x: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] nsg: usize,
    #[comptime] nr0: usize,
    #[comptime] _tgt: Target,
) {
    let lane = UNIT_POS_X as usize;
    let sg = UNIT_POS_Y as usize;
    let k = meta[0] as usize;
    let nout = meta[1] as usize;
    let nb = k / 128usize;
    let first_row = (CUBE_POS as usize * nsg + sg) * nr0;
    let ix = lane / 8usize;
    let it = lane % 8usize;
    let qh_shift = it / 2usize;
    let mut qh_p = 1.0f32;
    if qh_shift == 1usize {
        qh_p = 3.0f32;
    } else if qh_shift == 2usize {
        qh_p = 9.0f32;
    } else if qh_shift == 3usize {
        qh_p = 27.0f32;
    }

    let qh_p3 = 3.0f32 * qh_p;
    let inv256 = 1.0f32 / 256.0f32;

    let mut sumf = Array::<f32>::new(nr0);
    #[unroll]
    for r in 0usize..nr0 {
        sumf[r] = 0.0f32;
    }
    let row_stride = 7usize * nb;
    let mut max_r = 0usize;
    if first_row < nout {
        max_r = nout - 1usize - first_row;
    }
    let max_off = max_r * row_stride;
    let base0 = 7usize * (first_row * nb + ix);
    let mut row_off = Array::<usize>::new(nr0);
    #[unroll]
    for r in 0usize..nr0 {
        let mut ro = r * row_stride;
        if r > max_r {
            ro = max_off;
        }
        row_off[r] = ro;
    }

    let m0 = 2usize * it;
    let w01_off = it / 2usize;
    let w01_sh0 = (16usize * (it % 2usize)) as u32;
    let w01_sh1 = w01_sh0 + 8u32;
    let wmid_off = 4usize + it / 4usize;
    let wmid_sh = (8usize * (it % 4usize)) as u32;
    let bh_sh = (8usize * (it % 2usize)) as u32;

    if first_row < nout {
        let mut ib = ix;
        let mut yb = ix * 128usize;
        let mut cur0 = base0;
        while ib < nb {
            let y00 = f32::cast_from(x[yb + m0]);
            let y01 = f32::cast_from(x[yb + 16usize + m0]);
            let y02 = f32::cast_from(x[yb + 32usize + m0]);
            let y03 = f32::cast_from(x[yb + 48usize + m0]);
            let y04 = f32::cast_from(x[yb + 64usize + m0]);
            let y10 = f32::cast_from(x[yb + m0 + 1usize]);
            let y11 = f32::cast_from(x[yb + 16usize + m0 + 1usize]);
            let y12 = f32::cast_from(x[yb + 32usize + m0 + 1usize]);
            let y13 = f32::cast_from(x[yb + 48usize + m0 + 1usize]);
            let y14 = f32::cast_from(x[yb + 64usize + m0 + 1usize]);
            let y20 = f32::cast_from(x[yb + 80usize + it]);
            let y21 = f32::cast_from(x[yb + 88usize + it]);
            let y22 = f32::cast_from(x[yb + 96usize + it]);
            let y23 = f32::cast_from(x[yb + 104usize + it]);
            let y24 = f32::cast_from(x[yb + 112usize + it]);
            let yt = f32::cast_from(x[yb + 120usize + it]);

            let sumy = (y00 + y01 + y02 + y03 + y04)
                + (y10 + y11 + y12 + y13 + y14)
                + (y20 + y21 + y22 + y23 + y24)
                + yt;
            let c00 = y00 - 3.0f32 * y01;
            let c01 = y01 - 3.0f32 * y02;
            let c02 = y02 - 3.0f32 * y03;
            let c03 = y03 - 3.0f32 * y04;
            let c10 = y10 - 3.0f32 * y11;
            let c11 = y11 - 3.0f32 * y12;
            let c12 = y12 - 3.0f32 * y13;
            let c13 = y13 - 3.0f32 * y14;
            let c20 = y20 - 3.0f32 * y21;
            let c21 = y21 - 3.0f32 * y22;
            let c22 = y22 - 3.0f32 * y23;
            let c23 = y23 - 3.0f32 * y24;

            #[unroll]
            for r in 0usize..nr0 {
                let base = cur0 + row_off[r];
                let w_01 = wq[base + w01_off];
                let b0 = (w_01 >> w01_sh0) & 0xFFu32;
                let b1 = (w_01 >> w01_sh1) & 0xFFu32;
                let w_mid = wq[base + wmid_off];
                let b2 = (w_mid >> wmid_sh) & 0xFFu32;
                let w_tail = wq[base + 6usize];
                let bh = (w_tail >> bh_sh) & 0xFFu32;
                let sbits_hi = w_tail >> 16u32;
                let d = island! {
                    metal | cuda | rocm => {
                        f32::cast_from(Vector::<half::f16, Const<2>>::reinterpret::<u32>(sbits_hi)[0])
                    }
                    default => { crate::quant::f16lo_to_f32(sbits_hi & 0xFFFFu32) }
                };
                let uq = f32::cast_from(bh) * inv256;
                let qh_digit = f32::floor(qh_p3 * uq) - 3.0f32 * f32::floor(qh_p * uq);
                let acc = ptq1_dot_5_collapse(b0, c00, c01, c02, c03, y04)
                    + ptq1_dot_5_collapse(b1, c10, c11, c12, c13, y14)
                    + ptq1_dot_5_collapse(b2, c20, c21, c22, c23, y24)
                    + qh_digit * yt;
                sumf[r] += d * (acc - sumy);
            }
            ib += 4usize;
            yb += 512usize;
            cur0 += 28usize;
        }
    }

    #[unroll]
    for r in 0usize..nr0 {
        let tot = plane_sum(sumf[r]);
        if lane == 0usize && first_row + r < nout {
            out[first_row + r] = F::cast_from(tot);
        }
    }
}

/// Stage `bm x 32` activation tile `sa` and `32 x 64` dequantized PTQ1_0 weight tile `sb`.
#[device]
#[allow(clippy::too_many_arguments)]
fn ptq1_stage_ab_tile<F: Float>(
    wq: &Array<u32>,
    x: &Array<F>,
    sa: &mut SharedMemory<half::f16>,
    sb: &mut SharedMemory<half::f16>,
    m0: usize,
    n0: usize,
    m: usize,
    n: usize,
    k: usize,
    kb: usize,
    tid: usize,
    #[comptime] bm: usize,
    #[comptime] tgt: Target,
) {
    let k0 = kb * 32usize;
    let nb = k / 128usize;
    stage_sa_tile(x, sa, m0, m, k, k0, tid, bm);
    let local_n = tid / 2usize;
    let half_k = tid % 2usize;
    let nrow = n0 + local_n;
    let sb_k0 = half_k * 16usize;
    if nrow < n {
        let base = 7usize * (nrow * nb + (kb / 4usize));
        let w6 = wq[base + 6usize];
        let sbits = (w6 >> 16u32) & 0xFFFFu32;
        let hw_f16 = comptime!(matches!(tgt, Target::Metal | Target::Cuda | Target::Rocm));
        let d = if hw_f16 {
            f32::cast_from(Vector::<half::f16, Const<2>>::reinterpret::<u32>(sbits)[0])
        } else {
            crate::quant::f16lo_to_f32(sbits)
        };
        let il = 2usize * (kb % 4usize) + half_k;
        if il < 5usize {
            let w0 = wq[base];
            let w1 = wq[base + 1usize];
            let w2 = wq[base + 2usize];
            let w3 = wq[base + 3usize];
            let mut p3 = 1u32;
            if il == 1usize {
                p3 = 3u32;
            } else if il == 2usize {
                p3 = 9u32;
            } else if il == 3usize {
                p3 = 27u32;
            } else if il == 4usize {
                p3 = 81u32;
            }
            #[unroll]
            for j in 0usize..4usize {
                let sh = (8usize * j) as u32;
                sb[(sb_k0 + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit((w0 >> sh) & 0xFFu32, p3) * d);
                sb[(sb_k0 + 4usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit((w1 >> sh) & 0xFFu32, p3) * d);
                sb[(sb_k0 + 8usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit((w2 >> sh) & 0xFFu32, p3) * d);
                sb[(sb_k0 + 12usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit((w3 >> sh) & 0xFFu32, p3) * d);
            }
        } else if il < 7usize {
            let w4 = wq[base + 4usize];
            let w5 = wq[base + 5usize];
            let mut p_lo = 1u32;
            let mut p_hi = 3u32;
            if il == 6usize {
                p_lo = 9u32;
                p_hi = 27u32;
            }
            #[unroll]
            for j in 0usize..4usize {
                let sh = (8usize * j) as u32;
                let b4 = (w4 >> sh) & 0xFFu32;
                let b5 = (w5 >> sh) & 0xFFu32;
                sb[(sb_k0 + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit(b4, p_lo) * d);
                sb[(sb_k0 + 4usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit(b5, p_lo) * d);
                sb[(sb_k0 + 8usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit(b4, p_hi) * d);
                sb[(sb_k0 + 12usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit(b5, p_hi) * d);
            }
        } else {
            let w4 = wq[base + 4usize];
            let w5 = wq[base + 5usize];
            let qh0 = w6 & 0xFFu32;
            let qh1 = (w6 >> 8u32) & 0xFFu32;
            let mut qh_p = 1u32;
            #[unroll]
            for j in 0usize..4usize {
                let sh = (8usize * j) as u32;
                sb[(sb_k0 + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit((w4 >> sh) & 0xFFu32, 81u32) * d);
                sb[(sb_k0 + 4usize + j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit((w5 >> sh) & 0xFFu32, 81u32) * d);
                sb[(sb_k0 + 8usize + 2usize * j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit(qh0, qh_p) * d);
                sb[(sb_k0 + 9usize + 2usize * j) * 72usize + local_n] =
                    half::f16::cast_from(ptq1_trit(qh1, qh_p) * d);
                qh_p *= 3u32;
            }
        }
    } else {
        #[unroll]
        for i in 0usize..16usize {
            sb[(sb_k0 + i) * 72usize + local_n] = half::f16::cast_from(0.0f32);
        }
    }
}

/// PTQ1_0 prefill GEMM over native GGUF blocks: `out[m, n] = x[m, k] * W[n, k]^T`.
///
/// Same `cmma` tile schedule (`BM = 32|64, BN = 64, BK = 32`) as [`matmul_pq2_0_gguf`], staging
/// dequantized `block_ptq1_0` weights via [`ptq1_stage_ab_tile`].
///
/// Launch: `Block::new_1d(4 * plane)`, `Grid::Static(m.div_ceil(bm) as u32, n.div_ceil(64) as u32, 1)`,
/// `meta = [m, n, k]`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm, cpu), unchecked)]
pub fn matmul_ptq1_0_gguf<F: Float>(
    wq: &Array<u32>,
    x: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] bm: usize,
    #[comptime] plane: usize,
    #[comptime] tgt: Target,
) {
    let tid = UNIT_POS as usize;
    let warp = tid / plane;
    let warp_m = warp / 2usize;
    let warp_n = warp % 2usize;
    let m = meta[0] as usize;
    let n = meta[1] as usize;
    let k = meta[2] as usize;
    let m0 = CUBE_POS_X as usize * bm;
    let n0 = CUBE_POS_Y as usize * 64usize;
    let kb_count = k / 32usize;
    let warp_rows = bm / 2usize;

    let mut sa = SharedMemory::<half::f16>::new(bm * 40usize);
    let mut sb = SharedMemory::<half::f16>::new(32usize * 72usize);
    let mut sc = SharedMemory::<f32>::new(16usize * 32usize);

    island! {
        metal => {
            let wm_tiles = bm / 16usize;
            let mut acc = Sequence::<cmma::Matrix<f32>>::new();
            #[unroll]
            for _i in 0usize..(wm_tiles * 4usize) {
                acc.push(cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    8usize,
                    8usize,
                    8usize,
                    cmma::MatrixLayout::Undefined,
                    0.0f32,
                ));
            }
            for kb in 0..kb_count {
                ptq1_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                sync_cube();
                #[unroll]
                for dk in 0usize..4usize {
                    let mut mb = Sequence::<cmma::Matrix<half::f16>>::new();
                    #[unroll]
                    for jn in 0usize..4usize {
                        let b_col = warp_n * 32usize + jn * 8usize;
                        mb.push(cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::B,
                            8usize,
                            8usize,
                            8usize,
                            cmma::MatrixLayout::RowMajor,
                            &sb.slice((dk * 8usize) * 72usize + b_col, 32usize * 72usize),
                            72u32,
                        ));
                    }
                    #[unroll]
                    for im in 0usize..wm_tiles {
                        let a_row = warp_m * warp_rows + im * 8usize;
                        let ma = cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::A,
                            8usize,
                            8usize,
                            8usize,
                            cmma::MatrixLayout::RowMajor,
                            &sa.slice(a_row * 40usize + dk * 8usize, bm * 40usize),
                            40u32,
                        );
                        #[unroll]
                        for jn in 0usize..4usize {
                            let c = acc.index(im * 4usize + jn);
                            cmma::execute::<half::f16, half::f16, f32, f32>(&ma, mb.index(jn), c, c);
                        }
                    }
                }
                sync_cube();
            }
            #[unroll]
            for w in 0usize..4usize {
                #[unroll]
                for rh in 0usize..(bm / 32usize) {
                    if warp == w {
                        #[unroll]
                        for im in 0usize..2usize {
                            #[unroll]
                            for jn in 0usize..4usize {
                                let c_off = (im * 8usize) * 32usize + jn * 8usize;
                                cmma::store(
                                    &mut sc.slice_mut(c_off, 16usize * 32usize),
                                    acc.index((rh * 2usize + im) * 4usize + jn),
                                    32u32,
                                    cmma::MatrixLayout::RowMajor,
                                );
                            }
                        }
                    }
                    pq2_flush_warp_tile(
                        &sc,
                        out,
                        tid,
                        m0 + (w / 2usize) * warp_rows + rh * 16usize,
                        n0 + (w % 2usize) * 32usize,
                        m,
                        n,
                        plane,
                    );
                }
            }
        }
        cuda | rocm | vulkan => {
            let wm_tiles = bm / 32usize;
            let mut acc = Sequence::<cmma::Matrix<f32>>::new();
            #[unroll]
            for _i in 0usize..(wm_tiles * 2usize) {
                acc.push(cmma::Matrix::<f32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0.0f32,
                ));
            }
            for kb in 0..kb_count {
                if comptime!(plane == 32usize) {
                    ptq1_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                } else if tid < 128usize {
                    ptq1_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                }
                sync_cube();
                #[unroll]
                for dk in 0usize..2usize {
                    let mut mb = Sequence::<cmma::Matrix<half::f16>>::new();
                    #[unroll]
                    for jn in 0usize..2usize {
                        let b_col = warp_n * 32usize + jn * 16usize;
                        mb.push(cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::B,
                            16usize,
                            16usize,
                            16usize,
                            cmma::MatrixLayout::RowMajor,
                            &sb.slice((dk * 16usize) * 72usize + b_col, 32usize * 72usize),
                            72u32,
                        ));
                    }
                    #[unroll]
                    for im in 0usize..wm_tiles {
                        let a_row = warp_m * warp_rows + im * 16usize;
                        let ma = cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::A,
                            16usize,
                            16usize,
                            16usize,
                            cmma::MatrixLayout::RowMajor,
                            &sa.slice(a_row * 40usize + dk * 16usize, bm * 40usize),
                            40u32,
                        );
                        #[unroll]
                        for jn in 0usize..2usize {
                            let c = acc.index(im * 2usize + jn);
                            cmma::execute::<half::f16, half::f16, f32, f32>(&ma, mb.index(jn), c, c);
                        }
                    }
                }
                sync_cube();
            }
            #[unroll]
            for w in 0usize..4usize {
                #[unroll]
                for rh in 0usize..wm_tiles {
                    if warp == w {
                        #[unroll]
                        for jn in 0usize..2usize {
                            cmma::store(
                                &mut sc.slice_mut(jn * 16usize, 16usize * 32usize),
                                acc.index(rh * 2usize + jn),
                                32u32,
                                cmma::MatrixLayout::RowMajor,
                            );
                        }
                    }
                    pq2_flush_warp_tile(
                        &sc,
                        out,
                        tid,
                        m0 + (w / 2usize) * warp_rows + rh * 16usize,
                        n0 + (w % 2usize) * 32usize,
                        m,
                        n,
                        plane,
                    );
                }
            }
        }
        default => {
            let mut acc = Array::<f32>::new(warp_rows);
            #[unroll]
            for i in 0usize..warp_rows {
                acc[i] = 0.0f32;
            }
            let half_m = tid / 64usize;
            let col = tid % 64usize;
            for kb in 0..kb_count {
                ptq1_stage_ab_tile(wq, x, &mut sa, &mut sb, m0, n0, m, n, k, kb, tid, bm, tgt);
                sync_cube();
                #[unroll]
                for i in 0usize..warp_rows {
                    let row = half_m * warp_rows + i;
                    let mut sum = 0.0f32;
                    for kk in 0usize..32usize {
                        sum += f32::cast_from(sa[row * 40usize + kk])
                            * f32::cast_from(sb[kk * 72usize + col]);
                    }
                    acc[i] += sum;
                }
                sync_cube();
            }
            #[unroll]
            for i in 0usize..warp_rows {
                let orow = m0 + half_m * warp_rows + i;
                let ocol = n0 + col;
                if orow < m && ocol < n {
                    out[orow * n + ocol] = F::cast_from(acc[i]);
                }
            }
        }
    }
}

/// Deterministic ternary codes in `{0, 1, 2}` (`-1, 0, +1`) + f16-representable scales for PTQ1_0.
pub fn ptq1_0_synth(rows: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(k % QK_PTQ1_0, 0);
    let nb = (rows * k) / QK_PTQ1_0;
    let mut s = seed ^ 0x517C_C1B7_2722_0A95;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let codes = (0..rows * k).map(|_| (next() % 3) as u8).collect();
    let d = (0..nb)
        .map(|_| {
            half::f16::from_f32(0.005 + 0.03 * ((next() >> 40) as f32 / (1u32 << 24) as f32))
                .to_f32()
        })
        .collect();
    (codes, d)
}

/// Pack `(codes in {0, 1, 2}, d)` into exact 28-byte GGUF `block_ptq1_0` bytes (`iq_quants.rs` order).
pub fn ptq1_0_gguf_bytes(codes: &[u8], d: &[f32], rows: usize, k: usize) -> Vec<u8> {
    let nb = k / QK_PTQ1_0;
    let mut out = Vec::with_capacity(rows * nb * 28);
    for b in 0..rows * nb {
        let c = &codes[b * QK_PTQ1_0..(b + 1) * QK_PTQ1_0];
        let mut qs = [0u8; 24];
        for m in 0..16 {
            let val = 81 * c[m] as usize
                + 27 * c[16 + m] as usize
                + 9 * c[32 + m] as usize
                + 3 * c[48 + m] as usize
                + c[64 + m] as usize;
            qs[m] = ((val * 256 + 242) / 243) as u8;
        }
        for m in 0..8 {
            let val = 81 * c[80 + m] as usize
                + 27 * c[88 + m] as usize
                + 9 * c[96 + m] as usize
                + 3 * c[104 + m] as usize
                + c[112 + m] as usize;
            qs[16 + m] = ((val * 256 + 242) / 243) as u8;
        }
        let mut qh = [0u8; 2];
        for m in 0..2 {
            let val = 27 * c[120 + m] as usize
                + 9 * c[122 + m] as usize
                + 3 * c[124 + m] as usize
                + c[126 + m] as usize;
            qh[m] = ((val * 256 + 80) / 81) as u8;
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&half::f16::from_f32(d[b]).to_le_bytes());
    }
    out
}

/// View 28-byte GGUF `block_ptq1_0` bytes as little-endian `u32` words.
pub fn ptq1_0_gguf_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Decode one 28-byte GGUF `block_ptq1_0` into 128 f64 weights matching `BlockPTQ1_0::to_float`.
fn decode_ptq1_block_f64(blk: &[u8]) -> [f64; QK_PTQ1_0] {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let d = half::f16::from_le_bytes([blk[26], blk[27]]).to_f32() as f64;
    let mut w = [0.0f64; QK_PTQ1_0];
    for n in 0..5 {
        for m in 0..16 {
            let q = blk[m].wrapping_mul(POW3[n]);
            let xi = (((q as u16) * 3) >> 8) as i32 - 1;
            w[n * 16 + m] = xi as f64 * d;
        }
    }
    for n in 0..5 {
        for m in 0..8 {
            let q = blk[16 + m].wrapping_mul(POW3[n]);
            let xi = (((q as u16) * 3) >> 8) as i32 - 1;
            w[80 + n * 8 + m] = xi as f64 * d;
        }
    }
    for n in 0..4 {
        for m in 0..2 {
            let q = blk[24 + m].wrapping_mul(POW3[n]);
            let xi = (((q as u16) * 3) >> 8) as i32 - 1;
            w[120 + n * 2 + m] = xi as f64 * d;
        }
    }
    w
}

/// Host f64 reference for PTQ1_0 matvec (`rows x k` GGUF bytes).
pub fn ptq1_0_gguf_ref(bytes: &[u8], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / QK_PTQ1_0;
    (0..rows)
        .map(|r| {
            let mut acc = 0.0f64;
            for b in 0..nb {
                let w = decode_ptq1_block_f64(&bytes[(r * nb + b) * 28..(r * nb + b + 1) * 28]);
                for (j, &wv) in w.iter().enumerate() {
                    acc += wv * x[b * QK_PTQ1_0 + j] as f64;
                }
            }
            acc as f32
        })
        .collect()
}

/// Host f64 reference for PTQ1_0 GEMM (`out[m, n] = x[m, k] * W[n, k]^T`).
pub fn ptq1_0_gguf_gemm_ref(
    bytes: &[u8],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let nb = k / QK_PTQ1_0;
    let mut w = vec![0.0f64; n * k];
    for r in 0..n {
        for b in 0..nb {
            let bw = decode_ptq1_block_f64(&bytes[(r * nb + b) * 28..(r * nb + b + 1) * 28]);
            w[r * k + b * QK_PTQ1_0..(r * k + (b + 1) * QK_PTQ1_0)].copy_from_slice(&bw);
        }
    }
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for j in 0..k {
                acc += x[row * k + j] as f64 * w[col * k + j];
            }
            out[row * n + col] = acc as f32;
        }
    }
    out
}

/// Launch [`matvec_ptq1_0_gguf`] once and read back `rows` outputs.
pub fn matvec_ptq1_0_gguf_run_as<E: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    words: &[u32],
    x: &[E],
    rows: usize,
    k: usize,
    nsg: usize,
    nr0: usize,
) -> Vec<E> {
    let wh = client.create_from_slice(u32::as_bytes(words));
    let xh = client.create_from_slice(E::as_bytes(x));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::default(); rows]));
    let mh = client.create_from_slice(u32::as_bytes(&[
        k as u32,
        rows as u32,
        words.len() as u32,
    ]));
    unsafe {
        matvec_ptq1_0_gguf::launch_unchecked::<E, R>(
            client,
            Grid::Static((rows as u32).div_ceil((nsg * nr0) as u32), 1, 1),
            Block::new_2d(32, nsg as u32),
            ArrayArg::from_raw_parts(wh.clone(), words.len()),
            ArrayArg::from_raw_parts(xh.clone(), k),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            ArrayArg::from_raw_parts(mh.clone(), 3),
            nsg,
            nr0,
            Target::of(client),
        );
    }
    E::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Correctness board for [`matvec_ptq1_0_gguf`] across `PQ2_GGUF_SHAPES` and schedules.
pub fn ptq1_0_gguf_board<R: Runtime>(
    client: &ComputeClient<R>,
) -> Vec<(usize, usize, usize, usize, f32)> {
    let mut res = Vec::new();
    for &(rows, k) in PQ2_GGUF_SHAPES {
        let (codes, d) = ptq1_0_synth(rows, k, 0x3A17 ^ (rows * 19 + k) as u64);
        let bytes = ptq1_0_gguf_bytes(&codes, &d, rows, k);
        let words = ptq1_0_gguf_words(&bytes);
        let x: Vec<f32> = (0..k)
            .map(|j| ((j % 29) as f32 - 14.0) * 0.03 + 0.01)
            .collect();
        let want = ptq1_0_gguf_ref(&bytes, &x, rows, k);
        let maxref = want.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-30);
        for &nsg in &[2usize, 4usize] {
            for &nr0 in &[2usize, 4usize, 8usize] {
                let got = matvec_ptq1_0_gguf_run_as::<f32, R>(client, &words, &x, rows, k, nsg, nr0);
                let rel = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max)
                    / maxref;
                res.push((rows, k, nsg, nr0, rel));
            }
        }
    }
    res
}

/// Launch [`matmul_ptq1_0_gguf`] with explicit `bm` (`32` or `64`) once and read `[m, n]` back.
pub fn matmul_ptq1_0_gguf_bm_run_as<E: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    words: &[u32],
    x: &[E],
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
) -> Vec<E> {
    let tgt = Target::of(client);
    let plane = if matches!(tgt, Target::Vulkan) {
        client.properties().hardware.plane_size_max as usize
    } else {
        32usize
    };
    let wh = client.create_from_slice(u32::as_bytes(words));
    let xh = client.create_from_slice(E::as_bytes(x));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::default(); m * n]));
    let mh = client.create_from_slice(u32::as_bytes(&[m as u32, n as u32, k as u32]));
    unsafe {
        matmul_ptq1_0_gguf::launch_unchecked::<E, R>(
            client,
            Grid::Static(m.div_ceil(bm) as u32, n.div_ceil(64) as u32, 1),
            Block::new_1d((4 * plane) as u32),
            ArrayArg::from_raw_parts(wh.clone(), words.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            ArrayArg::from_raw_parts(mh.clone(), 3),
            bm,
            plane,
            tgt,
        );
    }
    E::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Correctness board for [`matmul_ptq1_0_gguf`]: scale-relative error vs f64 oracle (`f32` and `bf16`).
pub fn ptq1_0_gguf_gemm_board<R: Runtime>(
    client: &ComputeClient<R>,
) -> Vec<(usize, usize, usize, f32, f32)> {
    let mut res = Vec::new();
    for &(m, n, k) in PQ2_GGUF_GEMM_SHAPES {
        let (codes, d) = ptq1_0_synth(n, k, 0x5B5B_4321 ^ (m * 17 + n * 31 + k) as u64);
        let bytes = ptq1_0_gguf_bytes(&codes, &d, n, k);
        let words = ptq1_0_gguf_words(&bytes);
        let bm = if m > 48 && n >= 2048 { 64 } else { 32 };
        let x: Vec<f32> = (0..m * k)
            .map(|i| (((i * 7919) % 2003) as f32 / 1001.5) - 1.0)
            .collect();
        let want = ptq1_0_gguf_gemm_ref(&bytes, &x, m, n, k);
        let maxref = want.iter().fold(0f32, |mx, &v| mx.max(v.abs())).max(1e-30);
        let got32 = matmul_ptq1_0_gguf_bm_run_as::<f32, R>(client, &words, &x, m, n, k, bm);
        let rel32 = got32
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
            / maxref;
        let xb: Vec<half::bf16> = x.iter().map(|&v| half::bf16::from_f32(v)).collect();
        let xb_f: Vec<f32> = xb.iter().map(|v| v.to_f32()).collect();
        let want_b = ptq1_0_gguf_gemm_ref(&bytes, &xb_f, m, n, k);
        let maxref_b = want_b
            .iter()
            .fold(0f32, |mx, &v| mx.max(v.abs()))
            .max(1e-30);
        let got_b =
            matmul_ptq1_0_gguf_bm_run_as::<half::bf16, R>(client, &words, &xb, m, n, k, bm);
        let rel_b = got_b
            .iter()
            .zip(&want_b)
            .map(|(a, b)| (a.to_f32() - b).abs())
            .fold(0f32, f32::max)
            / maxref_b;
        res.push((m, n, k, rel32, rel_b));
    }
    res
}


