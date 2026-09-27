//! A table in training: F32 master rows on the host, and AdamW state for the rows a step has
//! touched and for no others.
//!
//! A step's gradient reaches only the rows its lookups read ([`super::Lookup::grads`]), so the
//! update is lazy: a row's moments exist from its first gradient on, a row no step touches never
//! moves (no weight decay reaches it either), and the state grows with the rows touched, not the
//! table. Bias correction is the optimizer's global step, as the dense update's.
//!
//! Every row a step changes keeps, until [`Rows::mark`], the value it had before its first
//! change: a round's change to the table is exactly those rows, which is all a cluster sends.

use crate::memory::hash::draw;
use crate::memory::Table;
use crate::optim::Update;
use hanzo_ml::{bail, DType, Device, Result, Tensor};
use std::collections::BTreeMap;

/// A row's AdamW moments: `m` then `v`, `dim` each.
pub type Moments = Box<[f32]>;

pub struct Rows {
    pub dim: usize,
    data: Vec<f32>,
    moments: BTreeMap<u32, Moments>,
    /// Rows changed since the last [`Rows::mark`], as they were before.
    before: BTreeMap<u32, Box<[f32]>>,
}

impl Rows {
    /// Rows from `data` (`count × dim`, row-major).
    pub fn new(data: Vec<f32>, dim: usize) -> Result<Rows> {
        if dim == 0 || data.len() % dim != 0 {
            bail!("{} values in rows of {dim}", data.len());
        }
        Ok(Rows {
            dim,
            data,
            moments: BTreeMap::new(),
            before: BTreeMap::new(),
        })
    }

    /// `count` rows of normal entries of deviation `sd`, drawn from `seed` by position, so the
    /// same seed gives the same table on every machine.
    pub fn seeded(count: usize, dim: usize, seed: u64, sd: f32) -> Result<Rows> {
        let unit = |x: u64| ((x >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let data = (0..(count * dim) as u64)
            .map(|i| {
                // Box-Muller from two draws of the entry's own stream
                let (a, b) = (unit(draw(seed, 2 * i)), unit(draw(seed, 2 * i + 1)));
                ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32 * sd
            })
            .collect();
        Rows::new(data, dim)
    }

    /// A mapped table's rows, read as F32 (an F8E4M3 row times its scale).
    pub fn read(t: &Table) -> Result<Rows> {
        let mut data = vec![0f32; t.rows * t.dim];
        for (r, row) in data.chunks_exact_mut(t.dim).enumerate() {
            t.read(r as u32, row)?;
        }
        Rows::new(data, t.dim)
    }

    pub fn count(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn row(&self, r: u32) -> &[f32] {
        let at = r as usize * self.dim;
        &self.data[at..at + self.dim]
    }

    /// Set row `r`, as a change: it joins the rows since the last mark.
    pub fn set(&mut self, r: u32, values: &[f32]) {
        let at = r as usize * self.dim;
        let row = &mut self.data[at..at + self.dim];
        self.before.entry(r).or_insert_with(|| row.into());
        row.copy_from_slice(values);
    }

    /// Set row `r` without noting it: the table as its owner says it is (a cluster's θ_global).
    pub fn put(&mut self, r: u32, values: &[f32]) {
        let at = r as usize * self.dim;
        self.data[at..at + self.dim].copy_from_slice(values);
    }

    /// `rows` as a `(rows, dim)` tensor in `dtype` on `dev`.
    pub fn gather(&self, rows: &[u32], dtype: DType, dev: &Device) -> Result<Tensor> {
        let mut out = Vec::with_capacity(rows.len() * self.dim);
        for &r in rows {
            if r as usize >= self.count() {
                bail!("row {r} is past the table's {}", self.count());
            }
            out.extend_from_slice(self.row(r));
        }
        Tensor::from_vec(out, (rows.len(), self.dim), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    /// The squared norm of `grads`, each row's gradient.
    pub fn norm(grads: &BTreeMap<u32, Vec<f32>>) -> f64 {
        grads
            .values()
            .flat_map(|g| g.iter())
            .map(|&x| f64::from(x) * f64::from(x))
            .sum()
    }

    /// One AdamW step over the rows `grads` names, at `u` (its weight decay ignored: no row
    /// decays, touched or not). Rows in ascending order; each row's first moment of this call
    /// starts its state at zero.
    pub fn step(&mut self, grads: &BTreeMap<u32, Vec<f32>>, u: &Update) -> Result<()> {
        let dim = self.dim;
        let (b1, b2) = (u.beta1 as f32, u.beta2 as f32);
        let (sm, sv) = (u.scale_m as f32, u.scale_v as f32);
        let (lr, eps, gs) = (u.lr as f32, u.eps as f32, u.grad_scale as f32);
        for (&r, g) in grads {
            if g.len() != dim || r as usize >= self.count() {
                bail!(
                    "a gradient of {} for row {r} of a table of {} rows of {dim}",
                    g.len(),
                    self.count()
                );
            }
            let mv = self
                .moments
                .entry(r)
                .or_insert_with(|| vec![0f32; 2 * dim].into_boxed_slice());
            let at = r as usize * dim;
            let row = &mut self.data[at..at + dim];
            self.before.entry(r).or_insert_with(|| row.to_vec().into());
            let (m, v) = mv.split_at_mut(dim);
            for i in 0..dim {
                let g = g[i] * gs;
                m[i] = m[i] * b1 + g * (1.0 - b1);
                v[i] = v[i] * b2 + g * g * (1.0 - b2);
                row[i] -= lr * (m[i] * sm) / ((v[i] * sv).sqrt() + eps);
            }
        }
        Ok(())
    }

    /// The rows changed since the last mark, each with the value it had before: what this table
    /// moved by since then is `row − before` over exactly these rows. Clears the record.
    pub fn mark(&mut self) -> BTreeMap<u32, Box<[f32]>> {
        std::mem::take(&mut self.before)
    }

    /// The optimizer state: every touched row's moments, rows ascending.
    pub fn moments(&self) -> &BTreeMap<u32, Moments> {
        &self.moments
    }

    /// Replace the optimizer state.
    pub fn load(&mut self, moments: BTreeMap<u32, Moments>) -> Result<()> {
        if moments.values().any(|m| m.len() != 2 * self.dim) {
            bail!("moments for rows of another width");
        }
        self.moments = moments;
        Ok(())
    }

    /// Write the rows as a table under `prefix` into `dir` (see [`Table::write`]).
    pub fn write(
        &self,
        dir: &std::path::Path,
        prefix: &str,
        dtype: DType,
        per: usize,
    ) -> Result<Vec<std::path::PathBuf>> {
        Table::write(dir, prefix, &self.data, self.dim, dtype, per)
    }
}
