//! Checkpoint Analyzer for Blackwell Pairwise 4:8 Structured Sparsity Coverage.
//!
//! Analyzes all 27B weights of Bonsai 2 (PQ2_0 GGUF) to measure:
//! - Exact pair coverage: non-zeros in best 2 pairs / total non-zeros
//! - Residual ratio: residual non-zeros / total non-zeros
//! - Coverage percentiles (p50, p90, p99) and buckets (100%, >=75%, >=50%)
//! - Breakdown by projection family (q, k, v, out, gate, up, down).

use byteorder::{LittleEndian, ReadBytesExt};
use hanzo_ml::quantized::gguf_file::{Content, TensorInfo};
use hanzo_ml::quantized::GgmlDType;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

struct CoverageStats {
    total_octets: u64,
    total_nonzeros: u64,
    covered_nonzeros: u64,
    residual_nonzeros: u64,
    cov_100_pct_octets: u64,
    cov_ge_75_octets: u64,
    cov_ge_50_octets: u64,
    // Histogram of (S, T) where T in 0..8, S in 0..4
    hist: [[u64; 9]; 5],
}

impl CoverageStats {
    fn new() -> Self {
        Self {
            total_octets: 0,
            total_nonzeros: 0,
            covered_nonzeros: 0,
            residual_nonzeros: 0,
            cov_100_pct_octets: 0,
            cov_ge_75_octets: 0,
            cov_ge_50_octets: 0,
            hist: [[0; 9]; 5],
        }
    }

    #[inline(always)]
    fn record_octet(&mut self, c: [u8; 4]) {
        let mut sorted = c;
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let s = (sorted[0] + sorted[1]) as u64;
        let t = (c[0] + c[1] + c[2] + c[3]) as u64;

        self.total_octets += 1;
        self.total_nonzeros += t;
        self.covered_nonzeros += s;
        self.residual_nonzeros += t - s;

        self.hist[s as usize][t as usize] += 1;

        if t == 0 || s == t {
            self.cov_100_pct_octets += 1;
            self.cov_ge_75_octets += 1;
            self.cov_ge_50_octets += 1;
        } else {
            let cov = (s as f64) / (t as f64);
            if cov >= 0.7499 {
                self.cov_ge_75_octets += 1;
            }
            if cov >= 0.4999 {
                self.cov_ge_50_octets += 1;
            }
        }
    }

    fn merge(&mut self, other: &CoverageStats) {
        self.total_octets += other.total_octets;
        self.total_nonzeros += other.total_nonzeros;
        self.covered_nonzeros += other.covered_nonzeros;
        self.residual_nonzeros += other.residual_nonzeros;
        self.cov_100_pct_octets += other.cov_100_pct_octets;
        self.cov_ge_75_octets += other.cov_ge_75_octets;
        self.cov_ge_50_octets += other.cov_ge_50_octets;
        for s in 0..5 {
            for t in 0..9 {
                self.hist[s][t] += other.hist[s][t];
            }
        }
    }

    fn print_summary(&self, title: &str) {
        let mean_cov = if self.total_nonzeros > 0 {
            (self.covered_nonzeros as f64) / (self.total_nonzeros as f64) * 100.0
        } else {
            100.0
        };
        let res_pct = if self.total_nonzeros > 0 {
            (self.residual_nonzeros as f64) / (self.total_nonzeros as f64) * 100.0
        } else {
            0.0
        };

        let p_100 = (self.cov_100_pct_octets as f64) / (self.total_octets as f64) * 100.0;
        let p_75 = (self.cov_ge_75_octets as f64) / (self.total_octets as f64) * 100.0;
        let p_50 = (self.cov_ge_50_octets as f64) / (self.total_octets as f64) * 100.0;

        println!("=======================================================================================================");
        println!(" {title}");
        println!("=======================================================================================================");
        println!("  Total Octets Analyzed:   {:>15}", self.total_octets);
        println!("  Total Trits Analyzed:    {:>15}", self.total_octets * 8);
        println!("  Total Non-zero Trits:    {:>15} ({:.2}% density)", self.total_nonzeros, (self.total_nonzeros as f64) / (self.total_octets as f64 * 8.0) * 100.0);
        println!("-------------------------------------------------------------------------------------------------------");
        println!("  Mean Structured Coverage: {:>14.2}% (non-zeros handled by Blackwell 4:8 TC)", mean_cov);
        println!("  Residual Non-zero Ratio:  {:>14.2}% (non-zeros needing residual pass / DP4A)", res_pct);
        println!("-------------------------------------------------------------------------------------------------------");
        println!("  Octets with Coverage = 100%: {:>11.2}% (Pass 0 is 100% exact, ZERO residual)", p_100);
        println!("  Octets with Coverage >= 75%: {:>11.2}%", p_75);
        println!("  Octets with Coverage >= 50%: {:>11.2}%", p_50);
        println!("-------------------------------------------------------------------------------------------------------");
        println!("  Non-zero Trit Distribution per 8-element Octet (T: non-zeros, S: top-2 pairs):");
        println!("   T (total nz) | Octets Count  | Pct of Total | Mean S (covered) | Mean Residual");
        for t in 0..=8 {
            let count_t: u64 = (0..5).map(|s| self.hist[s][t]).sum();
            if count_t > 0 {
                let cov_t: u64 = (0..5).map(|s| (s as u64) * self.hist[s][t]).sum();
                let mean_s = (cov_t as f64) / (count_t as f64);
                let mean_res = (t as f64) - mean_s;
                println!(
                    "        {:2}      | {:>12} | {:>10.2}% | {:>16.2} | {:>12.2}",
                    t,
                    count_t,
                    (count_t as f64) / (self.total_octets as f64) * 100.0,
                    mean_s,
                    mean_res
                );
            }
        }
        println!("=======================================================================================================\n");
    }
}

fn analyze_tensor<R: Read + Seek>(
    reader: &mut R,
    info: &TensorInfo,
    tensor_data_offset: u64,
) -> anyhow::Result<CoverageStats> {
    let mut stats = CoverageStats::new();
    let num_elems = info.shape.elem_count();
    let num_blocks = num_elems / 128; // PQ2_0 has 128 elems per block
    let tensor_bytes = num_blocks * 34; // 34 bytes per block (2 bytes d, 32 bytes qs)

    reader.seek(SeekFrom::Start(tensor_data_offset + info.offset))?;

    let mut buf = vec![0u8; 34 * 4096]; // 136 KB chunks

    let mut blocks_left = num_blocks;
    while blocks_left > 0 {
        let blocks_to_read = blocks_left.min(4096);
        let bytes_to_read = blocks_to_read * 34;
        reader.read_exact(&mut buf[..bytes_to_read])?;

        for b in 0..blocks_to_read {
            let block_bytes = &buf[b * 34..(b + 1) * 34];
            let qs = &block_bytes[2..34]; // 32 bytes

            // Each byte has 4 trits (0: -1, 1: 0, 2: +1)
            // 32 bytes = 16 octets of 8 trits
            for oct_idx in 0..16 {
                let b0 = qs[oct_idx * 2];
                let b1 = qs[oct_idx * 2 + 1];

                // Pair 0: trits 0, 1 of b0
                let t0 = ((b0 >> 0) & 3) != 1;
                let t1 = ((b0 >> 2) & 3) != 1;
                let c0 = (t0 as u8) + (t1 as u8);

                // Pair 1: trits 2, 3 of b0
                let t2 = ((b0 >> 4) & 3) != 1;
                let t3 = ((b0 >> 6) & 3) != 1;
                let c1 = (t2 as u8) + (t3 as u8);

                // Pair 2: trits 0, 1 of b1
                let t4 = ((b1 >> 0) & 3) != 1;
                let t5 = ((b1 >> 2) & 3) != 1;
                let c2 = (t4 as u8) + (t5 as u8);

                // Pair 3: trits 2, 3 of b1
                let t6 = ((b1 >> 4) & 3) != 1;
                let t7 = ((b1 >> 6) & 3) != 1;
                let c3 = (t6 as u8) + (t7 as u8);

                stats.record_octet([c0, c1, c2, c3]);
            }
        }
        blocks_left -= blocks_to_read;
    }

    Ok(stats)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "/home/z/work/Bonsai-demo/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    };

    println!("Opening Bonsai-2 GGUF file: {model_path}");
    let file = File::open(model_path)?;
    let mut reader = BufReader::new(file);

    let content = Content::read(&mut reader)?;
    println!(
        "GGUF Header loaded: {} tensors total.",
        content.tensor_infos.len()
    );

    let mut global_stats = CoverageStats::new();
    let mut family_stats: std::collections::HashMap<String, CoverageStats> =
        std::collections::HashMap::new();

    let mut pq2_count = 0;
    for (name, info) in &content.tensor_infos {
        if info.ggml_dtype == GgmlDType::PQ2_0 {
            pq2_count += 1;
            let stats = analyze_tensor(&mut reader, info, content.tensor_data_offset)?;

            let family = if name.contains("attn_q") || name.contains("q_proj") {
                "Attn Q"
            } else if name.contains("attn_k") || name.contains("k_proj") {
                "Attn K"
            } else if name.contains("attn_v") || name.contains("v_proj") {
                "Attn V"
            } else if name.contains("attn_output") || name.contains("o_proj") {
                "Attn Out"
            } else if name.contains("ffn_gate") || name.contains("gate_proj") {
                "MLP Gate"
            } else if name.contains("ffn_up") || name.contains("up_proj") {
                "MLP Up"
            } else if name.contains("ffn_down") || name.contains("down_proj") {
                "MLP Down"
            } else {
                "Other Linear"
            };

            family_stats
                .entry(family.to_string())
                .or_insert_with(CoverageStats::new)
                .merge(&stats);

            global_stats.merge(&stats);
        }
    }

    println!("\nAnalyzed {pq2_count} PQ2_0 tensors across the entire model.\n");

    for (family, stats) in &family_stats {
        stats.print_summary(&format!("PROJECTION FAMILY: {family}"));
    }

    global_stats.print_summary("FULL 27B MODEL GLOBAL AGGREGATE");

    Ok(())
}
