use criterion::{criterion_group, BatchSize, Criterion, Throughput};
use half::f16;
use hanzo_ml::quantized::bitcos::{pack_trits_to_pq2_0, pack_trits_to_ptq1_0, BitcosMatrix};
use hanzo_ml::quantized::iq_quants::QK_PQ2_0;
use hanzo_ml::quantized::k_quants::{matmul as q_matmul, BlockQ8_0, GgmlType, QK8_0};
use std::hint::black_box;

const BENCH_ROWS: usize = 1024;
const BENCH_COLS: usize = 4096;
const DENSITIES: [f64; 5] = [0.20, 0.35, 0.50, 0.65, 0.80];

fn lcg_next(st: &mut u64) -> u64 {
    *st = st
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *st
}

fn bench_bitcos_ablation(c: &mut Criterion) {
    let total = BENCH_ROWS * BENCH_COLS;
    let mut st = 0x9988_7766_5544_3322u64;
    let x: Vec<f32> = (0..BENCH_COLS)
        .map(|_| ((lcg_next(&mut st) >> 40) as i32 - 8_388_608) as f32 / 4_194_304.0)
        .collect();
    let mut x_q8 = vec![BlockQ8_0::zeros(); BENCH_COLS / QK8_0];
    BlockQ8_0::from_float(&x, &mut x_q8);

    for &z in &DENSITIES {
        let thresh = (z * (u32::MAX as f64)) as u32;
        let mut trits = Vec::with_capacity(total);
        for _ in 0..total {
            let r = (lcg_next(&mut st) >> 32) as u32;
            if r < thresh {
                trits.push(0i8);
            } else if (r & 1) == 0 {
                trits.push(1i8);
            } else {
                trits.push(-1i8);
            }
        }
        let nb = total / QK_PQ2_0;
        let scales: Vec<f16> = (0..nb)
            .map(|i| f16::from_f32(0.02 * ((i % 19) as f32 + 1.0)))
            .collect();
        let pq2 = pack_trits_to_pq2_0(&trits, &scales).unwrap();
        let ptq1 = pack_trits_to_ptq1_0(&trits, &scales).unwrap();
        let bm = BitcosMatrix::encode_from_pq2_0(&pq2, BENCH_ROWS, BENCH_COLS).unwrap();

        let mut group = c.benchmark_group(format!("ternary_matvec_z{:02}", (z * 100.0) as u32));
        group.throughput(Throughput::Elements(total as u64));

        group.bench_function("pq2_0", |b| {
            b.iter_batched(
                || vec![0.0f32; BENCH_ROWS],
                |mut out| {
                    q_matmul(
                        (1, BENCH_COLS, BENCH_ROWS),
                        black_box(&x),
                        black_box(&pq2),
                        black_box(&mut out),
                    )
                    .unwrap();
                    out
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("ptq1_0", |b| {
            b.iter_batched(
                || vec![0.0f32; BENCH_ROWS],
                |mut out| {
                    q_matmul(
                        (1, BENCH_COLS, BENCH_ROWS),
                        black_box(&x),
                        black_box(&ptq1),
                        black_box(&mut out),
                    )
                    .unwrap();
                    out
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("bitcos_g128", |b| {
            b.iter_batched(
                || vec![0.0f32; BENCH_ROWS],
                |mut out| {
                    black_box(&bm)
                        .matvec_q8_0(black_box(&x), black_box(&mut out))
                        .unwrap();
                    out
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

criterion_group!(benches, bench_bitcos_ablation);
