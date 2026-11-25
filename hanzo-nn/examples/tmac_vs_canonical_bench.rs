//! Reproducible same-model full-shape & end-to-end prompt/decode comparative benchmark:
//! `hanzo_nn::ternary_lut` (`TmacLut`, opt-in) vs canonical `hanzo-ml` ternary formats
//! (`PQ2_0`, `PTQ1_0`, and `BitcosMatrix`) on identical ternary weights, identical per-row
//! f16 scales, identical prompt/decode workloads, and matched thread settings (`1T` and `MT`
//! via `hanzo_ml::utils::barrier_pool()`).
//!
//! No analytical watt/energy estimate is computed or claimed (`energy_telemetry: null`).
//!
//! Run with:
//! `cargo run --release -p hanzo-nn --example tmac_vs_canonical_bench`
//! Optional JSON output:
//! `cargo run --release -p hanzo-nn --example tmac_vs_canonical_bench -- --json /tmp/tmac_bench.json`

use half::f16;
use hanzo_ml::quantized::bitcos::{pack_trits_to_pq2_0, pack_trits_to_ptq1_0, BitcosMatrix};
use hanzo_ml::quantized::ggml_file::qtensor_from_ggml;
use hanzo_ml::quantized::iq_quants::{BlockPQ2_0, BlockPTQ1_0, QK_PQ2_0};
use hanzo_ml::quantized::k_quants::{BlockQ8_0, GgmlType, QK8_0};
use hanzo_ml::quantized::{GgmlDType, QMatMul};
use hanzo_ml::{Device, Module, Result, Tensor};
use hanzo_nn::ternary_lut::{
    ternary_lut_matvec_f32, ternary_lut_matvec_f32_parallel, HardwareMetadata, PackedTernaryMatrix,
    TmacKernelBackend,
};
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
struct ShapeSpec {
    name: &'static str,
    rows: usize,
    cols: usize,
    iters: usize,
}

#[derive(Serialize)]
struct MatvecBenchRow {
    shape: String,
    rows: usize,
    cols: usize,
    zero_density: f64,
    format: String,
    threads: String,
    weight_bytes: usize,
    effective_bpw: f64,
    mean_us: f64,
    stdev_us: f64,
    cv_pct: f64,
    p50_us: f64,
    p95_us: f64,
    gops: f64,
    effective_gb_s: f64,
    cosine_vs_f32: f64,
    max_rel_err_vs_f32: f64,
}

#[derive(Clone, Copy)]
struct E2eProfile {
    name: &'static str,
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    vocab_size: usize,
}

#[derive(Serialize)]
struct E2eBenchRow {
    profile: String,
    format: String,
    threads: String,
    runs: usize,
    tokens_per_run: usize,
    weight_bytes: usize,
    effective_bpw: f64,
    wall_ms_mean: f64,
    wall_ms_stdev: f64,
    tok_s_mean: f64,
    tok_s_stdev: f64,
    tok_s_cv_pct: f64,
    max_logit_err_vs_f32: f32,
    cosine_vs_f32: f64,
    token_agreement_vs_f32: f64,
}

#[derive(Serialize)]
struct BenchReport {
    metadata: HardwareMetadata,
    barrier_pool_threads: usize,
    matvec_rows: Vec<MatvecBenchRow>,
    e2e_rows: Vec<E2eBenchRow>,
}

fn generate_trits(len: usize, zero_pct: u64, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bucket = (s >> 40) % 100;
            if bucket < zero_pct {
                0
            } else if ((s >> 32) & 1) == 0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

fn generate_f32(len: usize, scale: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            let raw = ((s >> 40) % 2001) as f32 - 1000.0;
            raw * 0.001 * scale
        })
        .collect()
}

fn exact_f32_matvec(
    trits: &[i8],
    row_scales: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
) {
    for r in 0..rows {
        let row = &trits[r * cols..(r + 1) * cols];
        let mut acc = 0.0f64;
        for c in 0..cols {
            acc += f64::from(row[c]) * f64::from(x[c]);
        }
        out[r] = (acc * f64::from(row_scales[r])) as f32;
    }
}

fn accuracy_metrics(cand: &[f32], reference: &[f32]) -> (f64, f64, f32) {
    let mut dot = 0.0f64;
    let mut nc = 0.0f64;
    let mut nr = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut max_ref = 0.0f64;
    for (&c, &r) in cand.iter().zip(reference.iter()) {
        let cf = f64::from(c);
        let rf = f64::from(r);
        dot += cf * rf;
        nc += cf * cf;
        nr += rf * rf;
        max_abs = max_abs.max((cf - rf).abs());
        max_ref = max_ref.max(rf.abs());
    }
    let cos = if nc > 0.0 && nr > 0.0 {
        dot / (nc.sqrt() * nr.sqrt())
    } else {
        1.0
    };
    let rel = if max_ref > 0.0 {
        max_abs / max_ref
    } else {
        0.0
    };
    (cos, rel, max_abs as f32)
}

fn sample_stats(samples: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = if samples.len() > 1 {
        samples
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };
    let stdev = var.sqrt();
    let cv_pct = if mean > 0.0 {
        (stdev / mean) * 100.0
    } else {
        0.0
    };
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = sorted[(sorted.len() - 1) / 2];
    let p95 = sorted[((sorted.len() - 1) * 95) / 100];
    (mean, stdev, cv_pct, p50, p95)
}

fn time_samples_us<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> Vec<f64> {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    samples
}

fn pq2_matvec_1t(blocks: &[BlockPQ2_0], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let bpr = cols / QK_PQ2_0;
    let mut x_q8 = vec![BlockQ8_0::zeros(); cols / QK8_0];
    BlockQ8_0::from_float(x, &mut x_q8);
    let mut r = 0usize;
    while r + 4 <= rows {
        let r0 = &blocks[r * bpr..(r + 1) * bpr];
        let r1 = &blocks[(r + 1) * bpr..(r + 2) * bpr];
        let r2 = &blocks[(r + 2) * bpr..(r + 3) * bpr];
        let r3 = &blocks[(r + 3) * bpr..(r + 4) * bpr];
        let (d0, d1, d2, d3) = BlockPQ2_0::vec_dot_4(cols, r0, r1, r2, r3, &x_q8);
        out[r] = d0;
        out[r + 1] = d1;
        out[r + 2] = d2;
        out[r + 3] = d3;
        r += 4;
    }
    while r < rows {
        let row = &blocks[r * bpr..(r + 1) * bpr];
        out[r] = BlockPQ2_0::vec_dot(cols, row, &x_q8);
        r += 1;
    }
}

fn ptq1_matvec_1t(blocks: &[BlockPTQ1_0], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let bpr = cols / QK_PQ2_0;
    let mut x_q8 = vec![BlockQ8_0::zeros(); cols / QK8_0];
    BlockQ8_0::from_float(x, &mut x_q8);
    let mut r = 0usize;
    while r + 4 <= rows {
        let r0 = &blocks[r * bpr..(r + 1) * bpr];
        let r1 = &blocks[(r + 1) * bpr..(r + 2) * bpr];
        let r2 = &blocks[(r + 2) * bpr..(r + 3) * bpr];
        let r3 = &blocks[(r + 3) * bpr..(r + 4) * bpr];
        let (d0, d1, d2, d3) = BlockPTQ1_0::vec_dot_4(cols, r0, r1, r2, r3, &x_q8);
        out[r] = d0;
        out[r + 1] = d1;
        out[r + 2] = d2;
        out[r + 3] = d3;
        r += 4;
    }
    while r < rows {
        let row = &blocks[r * bpr..(r + 1) * bpr];
        out[r] = BlockPTQ1_0::vec_dot(cols, row, &x_q8);
        r += 1;
    }
}

fn bitcos_matvec_1t(bm: &BitcosMatrix, x: &[f32], out: &mut [f32]) {
    let rows = bm.rows();
    let cols = bm.cols();
    let mut x_q8 = vec![BlockQ8_0::zeros(); cols / QK8_0];
    BlockQ8_0::from_float(x, &mut x_q8);
    let mut r = 0usize;
    while r + 4 <= rows {
        let (d0, d1, d2, d3) = bm.row_dot_4_q8_0(r, &x_q8);
        out[r] = d0;
        out[r + 1] = d1;
        out[r + 2] = d2;
        out[r + 3] = d3;
        r += 4;
    }
    while r < rows {
        out[r] = bm.row_dot_q8_0(r, &x_q8);
        r += 1;
    }
}

fn make_matvec_row(
    spec: ShapeSpec,
    zero_density: f64,
    format: &str,
    threads: &str,
    weight_bytes: usize,
    samples_us: &[f64],
    cand: &[f32],
    exact: &[f32],
) -> MatvecBenchRow {
    let elems = (spec.rows * spec.cols) as f64;
    let effective_bpw = (weight_bytes as f64 * 8.0) / elems;
    let (mean_us, stdev_us, cv_pct, p50_us, p95_us) = sample_stats(samples_us);
    let p50_s = (p50_us * 1e-6).max(1e-12);
    let gops = (2.0 * elems) / p50_s / 1e9;
    let effective_gb_s = (weight_bytes as f64) / p50_s / 1e9;
    let (cosine_vs_f32, max_rel_err_vs_f32, _) = accuracy_metrics(cand, exact);
    MatvecBenchRow {
        shape: spec.name.to_string(),
        rows: spec.rows,
        cols: spec.cols,
        zero_density,
        format: format.to_string(),
        threads: threads.to_string(),
        weight_bytes,
        effective_bpw,
        mean_us,
        stdev_us,
        cv_pct,
        p50_us,
        p95_us,
        gops,
        effective_gb_s,
        cosine_vs_f32,
        max_rel_err_vs_f32,
    }
}

/// Multi-format projection storing the exact same ternary weights and per-row f16 scales
/// across FP32, TmacLut, PQ2_0, PTQ1_0, and BitcosMatrix for apples-to-apples E2E decoding.
struct SharedTernaryProj {
    rows: usize,
    cols: usize,
    trits: Vec<i8>,
    row_scales: Vec<f32>,
    tmac: PackedTernaryMatrix,
    pq2_blocks: Vec<BlockPQ2_0>,
    pq2_qmm: QMatMul,
    ptq1_blocks: Vec<BlockPTQ1_0>,
    ptq1_qmm: QMatMul,
    bitcos: BitcosMatrix,
}

#[derive(Clone, Copy)]
enum E2eBackend {
    Fp32Exact,
    Tmac1T,
    Pq2_1T,
    Ptq1_1T,
    Bitcos1T,
    TmacMt,
    Pq2Mt,
    Ptq1Mt,
    BitcosMt,
}

impl E2eBackend {
    fn label(self) -> (&'static str, &'static str) {
        match self {
            Self::Fp32Exact => ("FP32-exact", "1T"),
            Self::Tmac1T => ("TmacLut", "1T"),
            Self::Pq2_1T => ("PQ2_0", "1T"),
            Self::Ptq1_1T => ("PTQ1_0", "1T"),
            Self::Bitcos1T => ("Bitcos(G128)", "1T"),
            Self::TmacMt => ("TmacLut", "MT"),
            Self::Pq2Mt => ("QMatMul(PQ2_0)", "MT"),
            Self::Ptq1Mt => ("QMatMul(PTQ1_0)", "MT"),
            Self::BitcosMt => ("Bitcos(G128)", "MT"),
        }
    }
}

impl SharedTernaryProj {
    fn from_seed(rows: usize, cols: usize, zero_pct: u64, seed: u64) -> Result<Self> {
        let trits = generate_trits(rows * cols, zero_pct, seed);
        let bpr = cols / QK_PQ2_0;
        let row_scales_f16: Vec<f16> = (0..rows)
            .map(|r| {
                let s = 0.03 + (((r * 13 + (seed as usize)) % 23) as f32) * 0.002;
                f16::from_f32(s / (cols as f32).sqrt())
            })
            .collect();
        let row_scales: Vec<f32> = row_scales_f16.iter().map(|d| d.to_f32()).collect();
        let mut block_scales_f16 = Vec::with_capacity(rows * bpr);
        for &d in &row_scales_f16 {
            for _ in 0..bpr {
                block_scales_f16.push(d);
            }
        }
        let tmac = PackedTernaryMatrix::from_trits(&trits, rows, cols, row_scales.clone(), None)?;
        let pq2_blocks = pack_trits_to_pq2_0(&trits, &block_scales_f16)?;
        let ptq1_blocks = pack_trits_to_ptq1_0(&trits, &block_scales_f16)?;
        let bitcos = BitcosMatrix::encode_from_pq2_0(&pq2_blocks, rows, cols)?;
        let pq2_raw = unsafe {
            std::slice::from_raw_parts(
                pq2_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(pq2_blocks.as_slice()),
            )
        };
        let ptq1_raw = unsafe {
            std::slice::from_raw_parts(
                ptq1_blocks.as_ptr().cast::<u8>(),
                std::mem::size_of_val(ptq1_blocks.as_slice()),
            )
        };
        let pq2_qmm = QMatMul::from_qtensor(qtensor_from_ggml(
            GgmlDType::PQ2_0,
            pq2_raw,
            vec![rows, cols],
            &Device::Cpu,
        )?)?;
        let ptq1_qmm = QMatMul::from_qtensor(qtensor_from_ggml(
            GgmlDType::PTQ1_0,
            ptq1_raw,
            vec![rows, cols],
            &Device::Cpu,
        )?)?;
        Ok(Self {
            rows,
            cols,
            trits,
            row_scales,
            tmac,
            pq2_blocks,
            pq2_qmm,
            ptq1_blocks,
            ptq1_qmm,
            bitcos,
        })
    }

    fn elems(&self) -> usize {
        self.rows * self.cols
    }

    fn weight_bytes(&self, backend: E2eBackend) -> usize {
        match backend {
            E2eBackend::Fp32Exact => self.rows * self.cols * 4,
            E2eBackend::Tmac1T | E2eBackend::TmacMt => self.tmac.allocated_bytes(),
            E2eBackend::Pq2_1T | E2eBackend::Pq2Mt => {
                std::mem::size_of_val(self.pq2_blocks.as_slice())
            }
            E2eBackend::Ptq1_1T | E2eBackend::Ptq1Mt => {
                std::mem::size_of_val(self.ptq1_blocks.as_slice())
            }
            E2eBackend::Bitcos1T | E2eBackend::BitcosMt => self.bitcos.total_bytes(),
        }
    }

    fn apply(&self, x: &[f32], backend: E2eBackend, out: &mut [f32]) -> Result<()> {
        match backend {
            E2eBackend::Fp32Exact => {
                exact_f32_matvec(&self.trits, &self.row_scales, self.rows, self.cols, x, out);
                Ok(())
            }
            E2eBackend::Tmac1T => {
                ternary_lut_matvec_f32(&self.tmac, x, TmacKernelBackend::Auto, out)
            }
            E2eBackend::TmacMt => {
                ternary_lut_matvec_f32_parallel(&self.tmac, x, TmacKernelBackend::Auto, out)
            }
            E2eBackend::Pq2_1T => {
                pq2_matvec_1t(&self.pq2_blocks, self.rows, self.cols, x, out);
                Ok(())
            }
            E2eBackend::Pq2Mt => {
                let xt = Tensor::from_slice(x, (1, self.cols), &Device::Cpu)?;
                let y = self.pq2_qmm.forward(&xt)?.flatten_all()?.to_vec1::<f32>()?;
                out.copy_from_slice(&y);
                Ok(())
            }
            E2eBackend::Ptq1_1T => {
                ptq1_matvec_1t(&self.ptq1_blocks, self.rows, self.cols, x, out);
                Ok(())
            }
            E2eBackend::Ptq1Mt => {
                let xt = Tensor::from_slice(x, (1, self.cols), &Device::Cpu)?;
                let y = self
                    .ptq1_qmm
                    .forward(&xt)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                out.copy_from_slice(&y);
                Ok(())
            }
            E2eBackend::Bitcos1T => {
                bitcos_matvec_1t(&self.bitcos, x, out);
                Ok(())
            }
            E2eBackend::BitcosMt => self.bitcos.matvec_q8_0(x, out),
        }
    }
}

struct SharedBlock {
    hidden_dim: usize,
    ffn_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    q: SharedTernaryProj,
    k: SharedTernaryProj,
    v: SharedTernaryProj,
    o: SharedTernaryProj,
    gate: SharedTernaryProj,
    up: SharedTernaryProj,
    down: SharedTernaryProj,
}

struct SharedTransformer {
    profile: E2eProfile,
    embeddings: Vec<f32>,
    blocks: Vec<SharedBlock>,
    final_norm: Vec<f32>,
    lm_head: SharedTernaryProj,
}

fn rms_norm(x: &[f32], w: &[f32], out: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv = 1.0f32 / (mean_sq + 1e-5).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

fn rope_inplace(vec: &mut [f32], num_heads: usize, head_dim: usize, pos: usize) {
    let half = head_dim / 2;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half {
            let theta = (pos as f32) * 10_000.0f32.powf(-2.0 * (i as f32) / (head_dim as f32));
            let (s, c) = theta.sin_cos();
            let x0 = vec[base + i];
            let x1 = vec[base + i + half];
            vec[base + i] = x0 * c - x1 * s;
            vec[base + i + half] = x0 * s + x1 * c;
        }
    }
}

impl SharedTransformer {
    fn build(profile: E2eProfile, zero_pct: u64, seed: u64) -> Result<Self> {
        let head_dim = profile.hidden_dim / profile.num_heads;
        let kv_dim = profile.num_kv_heads * head_dim;
        let embeddings = generate_f32(profile.vocab_size * profile.hidden_dim, 0.25, seed);
        let mut blocks = Vec::with_capacity(profile.num_layers);
        for l in 0..profile.num_layers {
            let s = seed.wrapping_add((l as u64 + 1) * 0x9e37_79b9);
            blocks.push(SharedBlock {
                hidden_dim: profile.hidden_dim,
                ffn_dim: profile.ffn_dim,
                num_heads: profile.num_heads,
                num_kv_heads: profile.num_kv_heads,
                head_dim,
                attn_norm: vec![1.0f32; profile.hidden_dim],
                ffn_norm: vec![1.0f32; profile.hidden_dim],
                q: SharedTernaryProj::from_seed(
                    profile.hidden_dim,
                    profile.hidden_dim,
                    zero_pct,
                    s + 1,
                )?,
                k: SharedTernaryProj::from_seed(kv_dim, profile.hidden_dim, zero_pct, s + 2)?,
                v: SharedTernaryProj::from_seed(kv_dim, profile.hidden_dim, zero_pct, s + 3)?,
                o: SharedTernaryProj::from_seed(
                    profile.hidden_dim,
                    profile.hidden_dim,
                    zero_pct,
                    s + 4,
                )?,
                gate: SharedTernaryProj::from_seed(
                    profile.ffn_dim,
                    profile.hidden_dim,
                    zero_pct,
                    s + 5,
                )?,
                up: SharedTernaryProj::from_seed(
                    profile.ffn_dim,
                    profile.hidden_dim,
                    zero_pct,
                    s + 6,
                )?,
                down: SharedTernaryProj::from_seed(
                    profile.hidden_dim,
                    profile.ffn_dim,
                    zero_pct,
                    s + 7,
                )?,
            });
        }
        let lm_head = SharedTernaryProj::from_seed(
            profile.vocab_size,
            profile.hidden_dim,
            zero_pct,
            seed.wrapping_add(0xDEAD_BEEF),
        )?;
        Ok(Self {
            profile,
            embeddings,
            blocks,
            final_norm: vec![1.0f32; profile.hidden_dim],
            lm_head,
        })
    }

    fn total_linear_elems(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.q.elems()
                    + b.k.elems()
                    + b.v.elems()
                    + b.o.elems()
                    + b.gate.elems()
                    + b.up.elems()
                    + b.down.elems()
            })
            .sum();
        blk + self.lm_head.elems()
    }

    fn linear_weight_bytes(&self, backend: E2eBackend) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.q.weight_bytes(backend)
                    + b.k.weight_bytes(backend)
                    + b.v.weight_bytes(backend)
                    + b.o.weight_bytes(backend)
                    + b.gate.weight_bytes(backend)
                    + b.up.weight_bytes(backend)
                    + b.down.weight_bytes(backend)
            })
            .sum();
        blk + self.lm_head.weight_bytes(backend)
    }

    fn run_20turn_workload(&self, backend: E2eBackend) -> Result<(Vec<usize>, Vec<f32>)> {
        let turns = 20usize;
        let prefill_per_turn = 6usize;
        let decode_per_turn = 4usize;
        let max_seq = turns * (prefill_per_turn + decode_per_turn);
        let head_dim = self.profile.hidden_dim / self.profile.num_heads;
        let kv_dim = self.profile.num_kv_heads * head_dim;
        let mut k_cache = vec![vec![0.0f32; max_seq * kv_dim]; self.profile.num_layers];
        let mut v_cache = vec![vec![0.0f32; max_seq * kv_dim]; self.profile.num_layers];
        let mut seq_len = 0usize;

        let mut x_norm = vec![0.0f32; self.profile.hidden_dim];
        let mut q_buf = vec![0.0f32; self.profile.hidden_dim];
        let mut k_buf = vec![0.0f32; kv_dim];
        let mut v_buf = vec![0.0f32; kv_dim];
        let mut attn_out = vec![0.0f32; self.profile.hidden_dim];
        let mut o_buf = vec![0.0f32; self.profile.hidden_dim];
        let mut gate_buf = vec![0.0f32; self.profile.ffn_dim];
        let mut up_buf = vec![0.0f32; self.profile.ffn_dim];
        let mut down_buf = vec![0.0f32; self.profile.hidden_dim];
        let mut logits = vec![0.0f32; self.profile.vocab_size];

        let mut generated_tokens = Vec::with_capacity(turns * decode_per_turn);

        let mut step_token = |tok: usize,
                              seq_len: &mut usize,
                              k_cache: &mut [Vec<f32>],
                              v_cache: &mut [Vec<f32>],
                              logits_out: &mut [f32]|
         -> Result<()> {
            let pos = *seq_len;
            let h_dim = self.profile.hidden_dim;
            let t_id = tok % self.profile.vocab_size;
            let mut h = self.embeddings[t_id * h_dim..(t_id + 1) * h_dim].to_vec();
            let inv_sqrt_d = 1.0f32 / (head_dim as f32).sqrt();
            let seq_total = pos + 1;

            for (l, blk) in self.blocks.iter().enumerate() {
                rms_norm(&h, &blk.attn_norm, &mut x_norm);
                blk.q.apply(&x_norm, backend, &mut q_buf)?;
                blk.k.apply(&x_norm, backend, &mut k_buf)?;
                blk.v.apply(&x_norm, backend, &mut v_buf)?;
                rope_inplace(&mut q_buf, blk.num_heads, blk.head_dim, pos);
                rope_inplace(&mut k_buf, blk.num_kv_heads, blk.head_dim, pos);

                let kv_off = pos * kv_dim;
                k_cache[l][kv_off..kv_off + kv_dim].copy_from_slice(&k_buf);
                v_cache[l][kv_off..kv_off + kv_dim].copy_from_slice(&v_buf);

                attn_out.fill(0.0);
                let kv_group = blk.num_heads / blk.num_kv_heads;
                let mut scores = vec![0.0f32; seq_total];
                for head in 0..blk.num_heads {
                    let kv_head = head / kv_group;
                    let qs = &q_buf[head * blk.head_dim..(head + 1) * blk.head_dim];
                    let mut max_s = f32::NEG_INFINITY;
                    for t in 0..seq_total {
                        let kb = t * kv_dim + kv_head * blk.head_dim;
                        let ks = &k_cache[l][kb..kb + blk.head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..blk.head_dim {
                            dot += qs[d] * ks[d];
                        }
                        let sc = dot * inv_sqrt_d;
                        scores[t] = sc;
                        if sc > max_s {
                            max_s = sc;
                        }
                    }
                    let mut sum_e = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_e += *s;
                    }
                    let inv_e = 1.0f32 / sum_e;
                    let oh = &mut attn_out[head * blk.head_dim..(head + 1) * blk.head_dim];
                    for t in 0..seq_total {
                        let prob = scores[t] * inv_e;
                        let vb = t * kv_dim + kv_head * blk.head_dim;
                        let vs = &v_cache[l][vb..vb + blk.head_dim];
                        for d in 0..blk.head_dim {
                            oh[d] += prob * vs[d];
                        }
                    }
                }
                blk.o.apply(&attn_out, backend, &mut o_buf)?;
                for i in 0..blk.hidden_dim {
                    h[i] += o_buf[i];
                }

                rms_norm(&h, &blk.ffn_norm, &mut x_norm);
                blk.gate.apply(&x_norm, backend, &mut gate_buf)?;
                blk.up.apply(&x_norm, backend, &mut up_buf)?;
                for i in 0..blk.ffn_dim {
                    let silu = gate_buf[i] / (1.0f32 + (-gate_buf[i]).exp());
                    gate_buf[i] = silu * up_buf[i];
                }
                blk.down.apply(&gate_buf, backend, &mut down_buf)?;
                for i in 0..blk.hidden_dim {
                    h[i] += down_buf[i];
                }
            }
            *seq_len = pos + 1;
            rms_norm(&h, &self.final_norm, &mut x_norm);
            self.lm_head.apply(&x_norm, backend, logits_out)?;
            Ok(())
        };

        let argmax = |slice: &[f32]| -> usize {
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &v) in slice.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi
        };

        for turn in 0..turns {
            for p in 0..prefill_per_turn {
                let tok = (turn * 13 + p * 7 + 3) % self.profile.vocab_size;
                step_token(tok, &mut seq_len, &mut k_cache, &mut v_cache, &mut logits)?;
            }
            let mut next_tok = argmax(&logits);
            generated_tokens.push(next_tok);
            for _ in 1..decode_per_turn {
                step_token(
                    next_tok,
                    &mut seq_len,
                    &mut k_cache,
                    &mut v_cache,
                    &mut logits,
                )?;
                next_tok = argmax(&logits);
                generated_tokens.push(next_tok);
            }
        }
        Ok((generated_tokens, logits))
    }
}

fn main() -> Result<()> {
    let mut json_path: Option<String> = None;
    let args: Vec<String> = std::env::args().collect();
    for i in 1..args.len() {
        if args[i] == "--json" && i + 1 < args.len() {
            json_path = Some(args[i + 1].clone());
        }
    }

    let meta = HardwareMetadata::detect();
    let pool_threads = hanzo_ml::utils::get_num_threads();

    println!(
        "# T-MAC LUT vs Canonical Ternary Formats (PQ2_0 / PTQ1_0 / BitcosMatrix)\n\
         Host: arch={} os={} isa={} barrier_pool_threads={} energy_telemetry={:?}\n",
        meta.cpu_arch, meta.os, meta.detected_isa, pool_threads, meta.energy_telemetry
    );

    // -------------------------------------------------------------------------
    // Part A: Full-Shape Projection Comparison (4B / 9B / 27B attn & ffn)
    // -------------------------------------------------------------------------
    println!(
        "## Part A: Full-Shape Projection Benchmark (Matched 1T and Matched MT barrier_pool)\n"
    );
    println!(
        "| Shape (N x K) | Zero z | Format | Mode | Weight (MB) | BPW | p50 (us) | p95 (us) | Mean +/- SD (us) | CV% | GOPS | Eff GB/s | Cos vs F32 | Max Rel Err |"
    );
    println!(
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    );

    let shapes = [
        ShapeSpec {
            name: "4B-attn",
            rows: 2560,
            cols: 2560,
            iters: 30,
        },
        ShapeSpec {
            name: "4B-ffn",
            rows: 6912,
            cols: 2560,
            iters: 20,
        },
        ShapeSpec {
            name: "9B-attn",
            rows: 4096,
            cols: 4096,
            iters: 20,
        },
        ShapeSpec {
            name: "9B-ffn",
            rows: 11008,
            cols: 4096,
            iters: 15,
        },
        ShapeSpec {
            name: "27B-attn",
            rows: 5120,
            cols: 5120,
            iters: 15,
        },
        ShapeSpec {
            name: "27B-ffn",
            rows: 13824,
            cols: 5120,
            iters: 10,
        },
    ];

    let mut matvec_rows = Vec::new();
    for &zero_pct in &[33u64, 65u64] {
        for (idx, &spec) in shapes.iter().enumerate() {
            let trits = generate_trits(
                spec.rows * spec.cols,
                zero_pct,
                0x2026_1345 ^ (zero_pct << 16) ^ (idx as u64),
            );
            let actual_z =
                trits.iter().filter(|&&t| t == 0).count() as f64 / (spec.rows * spec.cols) as f64;
            let x = generate_f32(spec.cols, 1.0, 0x9e3779b97f4a7c15 ^ (idx as u64));
            let x_tensor = Tensor::from_slice(&x, (1, spec.cols), &Device::Cpu)?;

            let blocks_per_row = spec.cols / QK_PQ2_0;
            let row_scales_f16: Vec<f16> = (0..spec.rows)
                .map(|r| f16::from_f32(0.04 + ((r % 23) as f32) * 0.005))
                .collect();
            let row_scales_f32: Vec<f32> = row_scales_f16.iter().map(|d| d.to_f32()).collect();
            let mut block_scales_f16 = Vec::with_capacity(spec.rows * blocks_per_row);
            for &d in &row_scales_f16 {
                for _ in 0..blocks_per_row {
                    block_scales_f16.push(d);
                }
            }

            let mut y_exact = vec![0.0f32; spec.rows];
            exact_f32_matvec(
                &trits,
                &row_scales_f32,
                spec.rows,
                spec.cols,
                &x,
                &mut y_exact,
            );

            let tmac = PackedTernaryMatrix::from_trits(
                &trits,
                spec.rows,
                spec.cols,
                row_scales_f32,
                None,
            )?;
            let pq2_blocks = pack_trits_to_pq2_0(&trits, &block_scales_f16)?;
            let ptq1_blocks = pack_trits_to_ptq1_0(&trits, &block_scales_f16)?;
            let bitcos = BitcosMatrix::encode_from_pq2_0(&pq2_blocks, spec.rows, spec.cols)?;

            let tmac_bytes = tmac.allocated_bytes();
            let pq2_bytes = std::mem::size_of_val(pq2_blocks.as_slice());
            let ptq1_bytes = std::mem::size_of_val(ptq1_blocks.as_slice());
            let bitcos_bytes = bitcos.total_bytes();

            let pq2_raw =
                unsafe { std::slice::from_raw_parts(pq2_blocks.as_ptr().cast::<u8>(), pq2_bytes) };
            let ptq1_raw = unsafe {
                std::slice::from_raw_parts(ptq1_blocks.as_ptr().cast::<u8>(), ptq1_bytes)
            };
            let pq2_qmm = QMatMul::from_qtensor(qtensor_from_ggml(
                GgmlDType::PQ2_0,
                pq2_raw,
                vec![spec.rows, spec.cols],
                &Device::Cpu,
            )?)?;
            let ptq1_qmm = QMatMul::from_qtensor(qtensor_from_ggml(
                GgmlDType::PTQ1_0,
                ptq1_raw,
                vec![spec.rows, spec.cols],
                &Device::Cpu,
            )?)?;

            let mut out = vec![0.0f32; spec.rows];
            let start_len = matvec_rows.len();

            // 1T matched kernels
            let samples = time_samples_us(3, spec.iters, || {
                ternary_lut_matvec_f32(
                    black_box(&tmac),
                    black_box(&x),
                    TmacKernelBackend::Auto,
                    &mut out,
                )
                .unwrap();
                black_box(&out);
            });
            matvec_rows.push(make_matvec_row(
                spec, actual_z, "TmacLut", "1T", tmac_bytes, &samples, &out, &y_exact,
            ));

            let samples = time_samples_us(3, spec.iters, || {
                pq2_matvec_1t(
                    black_box(&pq2_blocks),
                    spec.rows,
                    spec.cols,
                    black_box(&x),
                    &mut out,
                );
                black_box(&out);
            });
            matvec_rows.push(make_matvec_row(
                spec, actual_z, "PQ2_0", "1T", pq2_bytes, &samples, &out, &y_exact,
            ));

            let samples = time_samples_us(3, spec.iters, || {
                ptq1_matvec_1t(
                    black_box(&ptq1_blocks),
                    spec.rows,
                    spec.cols,
                    black_box(&x),
                    &mut out,
                );
                black_box(&out);
            });
            matvec_rows.push(make_matvec_row(
                spec, actual_z, "PTQ1_0", "1T", ptq1_bytes, &samples, &out, &y_exact,
            ));

            let samples = time_samples_us(3, spec.iters, || {
                bitcos_matvec_1t(black_box(&bitcos), black_box(&x), &mut out);
                black_box(&out);
            });
            matvec_rows.push(make_matvec_row(
                spec,
                actual_z,
                "Bitcos(G128)",
                "1T",
                bitcos_bytes,
                &samples,
                &out,
                &y_exact,
            ));

            // MT matched kernels (all sharing hanzo_ml::utils::barrier_pool())
            let samples = time_samples_us(3, spec.iters, || {
                ternary_lut_matvec_f32_parallel(
                    black_box(&tmac),
                    black_box(&x),
                    TmacKernelBackend::Auto,
                    &mut out,
                )
                .unwrap();
                black_box(&out);
            });
            matvec_rows.push(make_matvec_row(
                spec, actual_z, "TmacLut", "MT", tmac_bytes, &samples, &out, &y_exact,
            ));

            let samples = time_samples_us(3, spec.iters, || {
                let y = pq2_qmm.forward(black_box(&x_tensor)).unwrap();
                black_box(&y);
            });
            let y_qmm = pq2_qmm
                .forward(&x_tensor)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            matvec_rows.push(make_matvec_row(
                spec,
                actual_z,
                "QMatMul(PQ2_0)",
                "MT",
                pq2_bytes,
                &samples,
                &y_qmm,
                &y_exact,
            ));

            let samples = time_samples_us(3, spec.iters, || {
                let y = ptq1_qmm.forward(black_box(&x_tensor)).unwrap();
                black_box(&y);
            });
            let y_qmm = ptq1_qmm
                .forward(&x_tensor)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            matvec_rows.push(make_matvec_row(
                spec,
                actual_z,
                "QMatMul(PTQ1_0)",
                "MT",
                ptq1_bytes,
                &samples,
                &y_qmm,
                &y_exact,
            ));

            let samples = time_samples_us(3, spec.iters, || {
                bitcos.matvec_q8_0(black_box(&x), &mut out).unwrap();
                black_box(&out);
            });
            matvec_rows.push(make_matvec_row(
                spec,
                actual_z,
                "Bitcos(G128)",
                "MT",
                bitcos_bytes,
                &samples,
                &out,
                &y_exact,
            ));

            for r in &matvec_rows[start_len..] {
                println!(
                    "| {} ({}x{}) | {:.2} | {} | {} | {:.2} | {:.3} | {:.1} | {:.1} | {:.1} +/- {:.1} | {:.1}% | {:.2} | {:.2} | {:.6} | {:.2e} |",
                    r.shape,
                    r.rows,
                    r.cols,
                    r.zero_density,
                    r.format,
                    r.threads,
                    r.weight_bytes as f64 / (1024.0 * 1024.0),
                    r.effective_bpw,
                    r.p50_us,
                    r.p95_us,
                    r.mean_us,
                    r.stdev_us,
                    r.cv_pct,
                    r.gops,
                    r.effective_gb_s,
                    r.cosine_vs_f32,
                    r.max_rel_err_vs_f32,
                );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Part B: Identical End-to-End 20-Turn Prompt + KV-Cached Decode Workload
    // -------------------------------------------------------------------------
    println!(
        "\n## Part B: End-to-End 20-Turn Coding-Agent Prefill + KV-Cached Decode (Identical Weights, 200 Tokens/Run, 5 Repeated Runs)\n"
    );
    println!(
        "| Profile | Format | Mode | Weight (MB) | BPW | Wall (ms, mean +/- SD) | Effective Tok/s (mean +/- SD) | CV% | Max Logit Err | Cos vs F32 | Token Match |"
    );
    println!(
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    );

    let e2e_profiles = [
        E2eProfile {
            name: "4B-equiv-4L (h=512,ffn=1536)",
            hidden_dim: 512,
            ffn_dim: 1536,
            num_heads: 8,
            num_kv_heads: 2,
            num_layers: 4,
            vocab_size: 512,
        },
        E2eProfile {
            name: "9B-equiv-2L (h=1024,ffn=2816)",
            hidden_dim: 1024,
            ffn_dim: 2816,
            num_heads: 16,
            num_kv_heads: 2,
            num_layers: 2,
            vocab_size: 512,
        },
    ];

    let runs = 5usize;
    let tokens_per_run = 200usize;
    let mut e2e_rows = Vec::new();

    for (p_idx, &prof) in e2e_profiles.iter().enumerate() {
        let model = SharedTransformer::build(prof, 33, 0x2026_0500 + (p_idx as u64) * 1000)?;
        let total_elems = model.total_linear_elems() as f64;
        let (ref_tokens, ref_logits) = model.run_20turn_workload(E2eBackend::Fp32Exact)?;

        let backends = [
            E2eBackend::Fp32Exact,
            E2eBackend::Tmac1T,
            E2eBackend::Pq2_1T,
            E2eBackend::Ptq1_1T,
            E2eBackend::Bitcos1T,
            E2eBackend::TmacMt,
            E2eBackend::Pq2Mt,
            E2eBackend::Ptq1Mt,
            E2eBackend::BitcosMt,
        ];

        for backend in backends {
            let (fmt_name, mode_name) = backend.label();
            // 1 warmup run
            let (cand_tokens, cand_logits) = model.run_20turn_workload(backend)?;
            let mut wall_samples = Vec::with_capacity(runs);
            let mut toks_samples = Vec::with_capacity(runs);
            for _ in 0..runs {
                let t0 = Instant::now();
                let res = model.run_20turn_workload(backend)?;
                let elapsed_s = t0.elapsed().as_secs_f64();
                black_box(&res);
                wall_samples.push(elapsed_s * 1000.0);
                toks_samples.push((tokens_per_run as f64) / elapsed_s);
            }
            let (wall_mean, wall_sd, _, _, _) = sample_stats(&wall_samples);
            let (toks_mean, toks_sd, toks_cv, _, _) = sample_stats(&toks_samples);
            let (cos, _, max_abs) = accuracy_metrics(&cand_logits, &ref_logits);
            let matches = cand_tokens
                .iter()
                .zip(ref_tokens.iter())
                .filter(|(a, b)| a == b)
                .count();
            let agreement = (matches as f64) / (ref_tokens.len() as f64);
            let w_bytes = model.linear_weight_bytes(backend);
            let bpw = (w_bytes as f64 * 8.0) / total_elems;

            let row = E2eBenchRow {
                profile: prof.name.to_string(),
                format: fmt_name.to_string(),
                threads: mode_name.to_string(),
                runs,
                tokens_per_run,
                weight_bytes: w_bytes,
                effective_bpw: bpw,
                wall_ms_mean: wall_mean,
                wall_ms_stdev: wall_sd,
                tok_s_mean: toks_mean,
                tok_s_stdev: toks_sd,
                tok_s_cv_pct: toks_cv,
                max_logit_err_vs_f32: max_abs,
                cosine_vs_f32: cos,
                token_agreement_vs_f32: agreement,
            };
            println!(
                "| {} | {} | {} | {:.2} | {:.3} | {:.2} +/- {:.2} | {:.1} +/- {:.1} | {:.1}% | {:.2e} | {:.6} | {:.3} |",
                row.profile,
                row.format,
                row.threads,
                row.weight_bytes as f64 / (1024.0 * 1024.0),
                row.effective_bpw,
                row.wall_ms_mean,
                row.wall_ms_stdev,
                row.tok_s_mean,
                row.tok_s_stdev,
                row.tok_s_cv_pct,
                row.max_logit_err_vs_f32,
                row.cosine_vs_f32,
                row.token_agreement_vs_f32,
            );
            e2e_rows.push(row);
        }
    }

    if let Some(path) = json_path {
        let report = BenchReport {
            metadata: meta,
            barrier_pool_threads: pool_threads,
            matvec_rows,
            e2e_rows,
        };
        std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap())
            .map_err(|e| hanzo_ml::Error::Msg(format!("failed to write {path}: {e}")))?;
        eprintln!("Wrote JSON report to {path}");
    }
    Ok(())
}
