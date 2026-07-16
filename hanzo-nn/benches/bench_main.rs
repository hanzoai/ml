mod benchmarks;

use criterion::criterion_main;
criterion_main!(
    benchmarks::softmax::benches,
    benchmarks::norm::benches,
    benchmarks::conv::benches
);
