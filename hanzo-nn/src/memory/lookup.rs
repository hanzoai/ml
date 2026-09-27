//! What one pass reads from a table: the distinct rows, uploaded once, and each slot's row among
//! them. An embedding `e` is the slots' rows laid end to end; a slot that reads no row reads a
//! zero row, and a pooled slot reads the sum of its rows.

use crate::memory::NONE;
use hanzo_ml::{bail, DType, Result, Tensor};
use std::collections::HashMap;

/// The distinct rows a pass reads and where each read goes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Lookup {
    /// Distinct table rows, in first-read order: what is uploaded.
    pub rows: Vec<u32>,
    /// Each slot's index into `rows`, or [`NONE`] for a zero row.
    pub slots: Vec<u32>,
}

impl Lookup {
    /// The lookup of `reads`, one table row (or [`NONE`]) per slot.
    pub fn new(reads: &[u32]) -> Lookup {
        let mut at: HashMap<u32, u32> = HashMap::with_capacity(reads.len());
        let mut rows = Vec::new();
        let slots = reads
            .iter()
            .map(|&r| {
                if r == NONE {
                    return NONE;
                }
                *at.entry(r).or_insert_with(|| {
                    rows.push(r);
                    rows.len() as u32 - 1
                })
            })
            .collect();
        Lookup { rows, slots }
    }

    /// Add `reads` as more slots, sharing rows already read.
    pub fn extend(&mut self, reads: &[u32]) {
        let mut at: HashMap<u32, u32> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, &r)| (r, i as u32))
            .collect();
        for &r in reads {
            if r == NONE {
                self.slots.push(NONE);
                continue;
            }
            let i = *at.entry(r).or_insert_with(|| {
                self.rows.push(r);
                self.rows.len() as u32 - 1
            });
            self.slots.push(i);
        }
    }

    /// Slot indices with [`NONE`] pointing at the zero row appended after `rows`: `[slots]` u32.
    fn index(&self, dev: &hanzo_ml::Device) -> Result<Tensor> {
        let zero = self.rows.len() as u32;
        let idx: Vec<u32> = self
            .slots
            .iter()
            .map(|&s| if s == NONE { zero } else { s })
            .collect();
        Tensor::from_vec(idx, self.slots.len(), dev)
    }

    /// `u` (`[rows, dim]`, this lookup's rows as uploaded) with a zero row after them.
    fn padded(u: &Tensor) -> Result<Tensor> {
        let dim = u.dim(1)?;
        let zero = Tensor::zeros((1, dim), u.dtype(), u.device())?;
        Tensor::cat(&[u, &zero], 0)
    }

    /// The slots' rows `[slots, dim]` from `u`, the distinct rows `[rows, dim]` (a tensor or a
    /// variable: the gradient of each distinct row is the sum of its slots').
    pub fn embed(&self, u: &Tensor) -> Result<Tensor> {
        if u.dim(0)? != self.rows.len() {
            bail!("{} rows for a lookup of {}", u.dim(0)?, self.rows.len());
        }
        Self::padded(u)?.index_select(&self.index(u.device())?, 0)
    }

    /// Each group of `k` consecutive slots summed: `[slots / k, dim]`. The sum runs in F32 and is
    /// rounded to `u`'s dtype once.
    pub fn pool(&self, u: &Tensor, k: usize) -> Result<Tensor> {
        if k == 0 || self.slots.len() % k != 0 {
            bail!("{} slots in groups of {k}", self.slots.len());
        }
        let e = self.embed(u)?;
        let dim = e.dim(1)?;
        e.to_dtype(DType::F32)?
            .reshape((self.slots.len() / k, k, dim))?
            .sum(1)?
            .to_dtype(u.dtype())
    }

    /// The gradient of every table row this lookup read, from the gradient of its distinct rows
    /// `grad` (`[rows, dim]`): `(row, gradient)` in `rows`' order, as F32.
    pub fn grads(&self, grad: &Tensor) -> Result<Vec<(u32, Vec<f32>)>> {
        let g = grad.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        if g.len() != self.rows.len() {
            bail!(
                "a gradient of {} rows for a lookup of {}",
                g.len(),
                self.rows.len()
            );
        }
        Ok(self.rows.iter().copied().zip(g).collect())
    }
}
