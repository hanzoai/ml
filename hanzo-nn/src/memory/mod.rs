//! Hashed memory: a table of rows addressed by hashed keys and read sparsely.
//!
//! [`hash`] turns tokens ([`hash::Ngram`]) or keys ([`hash::Keyed`]) into table rows; a
//! [`Lookup`] gathers a pass's distinct rows once; [`Table`] serves them from mapped safetensors
//! shards (F32, BF16 or F8E4M3) and [`Rows`] trains them, with AdamW state only for rows a step
//! touched; [`Block`] injects the gathered rows into every residual stream through a gate.

pub mod block;
pub mod hash;
pub mod lookup;
pub mod rows;
pub mod table;

pub use block::Block;
pub use lookup::Lookup;
pub use rows::Rows;
pub use table::Table;

/// A slot that reads no row: it reads the zero row.
pub const NONE: u32 = u32::MAX;

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Device};

    #[test]
    fn a_lookup_uploads_each_distinct_row_once() {
        let l = Lookup::new(&[5, NONE, 5, 9]);
        assert_eq!(l.rows, vec![5, 9]);
        assert_eq!(l.slots, vec![0, NONE, 0, 1]);
    }

    #[test]
    fn a_table_reads_back_what_was_written_across_shards() -> hanzo_ml::Result<()> {
        let (rows, dim) = (10, 4);
        let data: Vec<f32> = (0..rows * dim).map(|i| i as f32 * 0.25 - 3.0).collect();
        // each dtype's relative precision: exact, 8 and 4 significant bits
        for (dtype, rel) in [
            (DType::F32, 0.0),
            (DType::BF16, 1.0 / 128.0),
            (DType::F8E4M3, 1.0 / 16.0),
        ] {
            let dir = std::env::temp_dir().join(format!(
                "hanzo-memory-{}-{}",
                dtype.as_str(),
                std::process::id()
            ));
            let files = Table::write(&dir, "mem", &data, dim, dtype, 3)?;
            let t = Table::open(&files, "mem")?;
            let mut out = vec![0f32; dim];
            for r in 0..rows as u32 {
                t.read(r, &mut out)?;
                let want = &data[r as usize * dim..][..dim];
                for (a, b) in out.iter().zip(want) {
                    assert!(
                        (a - b).abs() <= rel * b.abs() + 1e-6,
                        "{dtype:?} row {r}: {a} vs {b}"
                    );
                }
            }
            let g = t
                .gather(&[7, 2], &Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            assert_eq!((g.len(), g[0].len()), (2, dim));
            std::fs::remove_dir_all(&dir).ok();
        }
        Ok(())
    }

    #[test]
    fn ngram_rows_are_deterministic_and_inside_the_table() -> hanzo_ml::Result<()> {
        let h = hash::Ngram::seeded(7, 0, 3, 2, 1000, 50_000)?;
        let a = h.rows(&[11, 12], &[13, 14, 15]);
        assert_eq!(a, h.rows(&[11, 12], &[13, 14, 15]));
        assert_eq!(a.len(), 3 * h.width());
        assert!(a.iter().all(|&r| u64::from(r) < h.end()));
        Ok(())
    }
}
