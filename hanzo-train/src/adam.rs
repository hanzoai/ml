//! AdamW over named parameters, each at its own base learning rate, with its moments and step
//! count in the open so a checkpoint carries them and a run resumes bit for bit. The update is
//! hanzo-nn's ([`hanzo_nn::optim::adamw`]): on Metal one fused kernel per F32 parameter,
//! elsewhere the same arithmetic in tensor ops.

use anyhow::{bail, ensure, Result};
use hanzo_ml::backprop::GradStore;
use hanzo_ml::{Tensor, Var};
use hanzo_nn::optim::{adamw, grad_norm, Update};
use hanzo_nn::VarMap;
use serde::{Deserialize, Serialize};

/// AdamW's constants, and the global gradient norm a step clips to.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hyper {
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    /// Decoupled weight decay.
    pub decay: f64,
    pub clip: f64,
}

/// AdamW over the parameters in sorted name order, the order [`crate::cluster::Params`]
/// flattens them in.
pub struct Adam {
    vars: Vec<Var>,
    /// First and second moments, one pair per parameter.
    moments: Vec<(Var, Var)>,
    rates: Vec<f64>,
    f: f64,
    hyper: Hyper,
    /// Steps taken: the bias correction's exponent.
    pub t: u64,
}

impl Adam {
    /// `rate` gives each parameter's base learning rate by name; `None` leaves it out.
    pub fn new(vars: &VarMap, rate: impl Fn(&str) -> Option<f64>, hyper: Hyper) -> Result<Adam> {
        let data = vars.data().lock().expect("varmap lock");
        let mut named: Vec<(&String, f64)> = data
            .keys()
            .filter_map(|n| rate(n).map(|r| (n, r)))
            .collect();
        named.sort_by(|a, b| a.0.cmp(b.0));
        let vars: Vec<Var> = named.iter().map(|(n, _)| data[*n].clone()).collect();
        let moments = vars
            .iter()
            .map(|v| {
                let z = || Var::zeros(v.shape(), v.dtype(), v.device());
                Ok((z()?, z()?))
            })
            .collect::<Result<_>>()?;
        Ok(Adam {
            rates: named.iter().map(|(_, r)| *r).collect(),
            vars,
            moments,
            f: 1.0,
            hyper,
            t: 0,
        })
    }

    /// Scale every learning rate by `f` (warmup, decay).
    pub fn schedule(&mut self, f: f64) {
        self.f = f;
    }

    /// Clip the gradients to the global norm `clip`, then step. Returns the norm before
    /// clipping; a norm that is not finite is an error, and nothing moves.
    pub fn step(&mut self, grads: &GradStore) -> Result<f32> {
        let h = self.hyper;
        let norm = grad_norm(grads, &self.vars)?;
        if !norm.is_finite() {
            bail!("gradient norm is {norm}");
        }
        let scale = if norm > h.clip { h.clip / norm } else { 1.0 };
        self.t += 1;
        let (scale_m, scale_v) = (
            1.0 / (1.0 - h.beta1.powi(self.t as i32)),
            1.0 / (1.0 - h.beta2.powi(self.t as i32)),
        );
        for ((theta, (m, v)), rate) in self.vars.iter().zip(&self.moments).zip(&self.rates) {
            let Some(g) = grads.get(theta.as_tensor()) else {
                continue;
            };
            let u = Update {
                lr: self.f * rate,
                beta1: h.beta1,
                beta2: h.beta2,
                eps: h.eps,
                weight_decay: h.decay,
                scale_m,
                scale_v,
                grad_scale: scale,
            };
            adamw(theta, g, m, v, &u)?;
        }
        Ok(norm as f32)
    }

    pub fn vars(&self) -> &[Var] {
        &self.vars
    }

    /// The moments, flattened in parameter order: every first moment, then every second.
    pub fn moments(&self) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        for pick in [0, 1] {
            for (m, v) in &self.moments {
                let t = if pick == 0 { m } else { v };
                out.extend(t.as_tensor().flatten_all()?.to_vec1::<f32>()?);
            }
        }
        Ok(out)
    }

    /// Set the moments from [`Adam::moments`]'s layout and the step count to `t`.
    pub fn load(&mut self, flat: &[f32], t: u64) -> Result<()> {
        let n: usize = self.vars.iter().map(|v| v.elem_count()).sum();
        ensure!(
            flat.len() == 2 * n,
            "moments of {} values, expected {}",
            flat.len(),
            2 * n
        );
        let mut at = 0;
        for pick in [0, 1] {
            for (m, v) in &self.moments {
                let x = if pick == 0 { m } else { v };
                let k = x.elem_count();
                x.set(&Tensor::from_slice(
                    &flat[at..at + k],
                    x.shape(),
                    x.device(),
                )?)?;
                at += k;
            }
        }
        self.t = t;
        Ok(())
    }
}

/// Add `more` into `acc`: gradients accumulated across the micro-batches of one batch.
pub fn accumulate(acc: &mut Option<GradStore>, more: GradStore, vars: &[Var]) -> Result<()> {
    match acc {
        None => *acc = Some(more),
        Some(a) => {
            for v in vars {
                if let Some(g) = more.get(v.as_tensor()) {
                    let sum = match a.remove(v.as_tensor()) {
                        Some(prev) => (prev + g)?,
                        None => g.clone(),
                    };
                    a.insert(v.as_tensor(), sum);
                }
            }
        }
    }
    Ok(())
}
