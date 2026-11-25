//! Conditional low-rank attention updates in one encoder pass. Frozen projections are shared;
//! an unselected row executes the original projection without computing or adding a zero delta.
use super::*;

/// `scale * B A x`; scale is explicit (use alpha/rank for conventional LoRA).
#[derive(Clone)]
pub struct Lora {
    a: Linear,
    b: Linear,
    input: usize,
    output: usize,
    scale: f64,
}

impl Lora {
    /// A is `[rank, input]`, B `[output, rank]`. Factors may be trainable tensors.
    pub fn new(a: Tensor, b: Tensor, scale: f64) -> Result<Self> {
        let (rank, input) = a.dims2()?;
        let (output, br) = b.dims2()?;
        if rank == 0
            || input == 0
            || output == 0
            || rank != br
            || !scale.is_finite()
            || a.dtype() != b.dtype()
            || !a.device().same_device(b.device())
        {
            hanzo_ml::bail!("invalid low-rank attention factors");
        }
        Ok(Self {
            a: Linear::new(a, None),
            b: Linear::new(b, None),
            input,
            output,
            scale,
        })
    }

    /// Factor storage only; no frozen base weights are owned by this update.
    pub fn bytes(&self) -> usize {
        [self.a.weight(), self.b.weight()]
            .iter()
            .map(|t| t.elem_count() * t.dtype().size_in_bytes())
            .sum()
    }

    /// The low-rank delta alone, useful for training and isolated systems measurements.
    pub fn delta(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.a)?.apply(&self.b)? * self.scale
    }
}

/// Optional updates to the attention input and output projections of one layer.
#[derive(Clone, Default)]
pub struct Layer {
    pub qkv: Option<Lora>,
    pub output: Option<Lora>,
}

/// Updates to consecutive upper layers; every layer below `first` is untouched.
#[derive(Clone)]
pub struct Adapter {
    pub first: usize,
    pub layers: Vec<Layer>,
}

impl Adapter {
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| [&l.qkv, &l.output])
            .flatten()
            .map(Lora::bytes)
            .sum()
    }

    fn check(&self, first: usize, count: usize, width: usize) -> Result<()> {
        if self.first < first
            || self.layers.is_empty()
            || self
                .first
                .checked_add(self.layers.len())
                .is_none_or(|n| n > count)
        {
            hanzo_ml::bail!("adapter layers outside the routed upper stack");
        }
        for l in &self.layers {
            for (p, output) in [(&l.qkv, 3 * width), (&l.output, width)] {
                if let Some(p) = p {
                    if p.input != width || p.output != output {
                        hanzo_ml::bail!("adapter projection shape does not match backbone");
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) struct Routing<'a> {
    bank: &'a [Adapter],
    selected: &'a [Option<usize>],
    lens: &'a [usize],
    packed: bool,
    layer: usize,
}

/// Project the whole batch by the original kernel, then update selected sequences only.
/// In particular, no multiply or addition touches the other sequences' projection values.
pub(super) fn project(
    p: &Proj,
    x: &Tensor,
    route: Option<&Routing<'_>>,
    qkv: bool,
) -> Result<Tensor> {
    let base = x.apply(p)?;
    let Some(route) = route else { return Ok(base) };
    let updates: Vec<Option<&Lora>> = route
        .selected
        .iter()
        .map(|selected| {
            let a = &route.bank[(*selected)?];
            let l = a.layers.get(route.layer.checked_sub(a.first)?)?;
            if qkv {
                l.qkv.as_ref()
            } else {
                l.output.as_ref()
            }
        })
        .collect();
    if updates.iter().all(Option::is_none) {
        return Ok(base);
    }
    let mut parts = Vec::with_capacity(updates.len());
    let mut offset = 0;
    for (row, update) in updates.into_iter().enumerate() {
        let (at, len) = if route.packed {
            (offset, route.lens[row])
        } else {
            (row, 1)
        };
        offset += route.lens[row];
        let old = base.narrow(0, at, len)?;
        parts.push(match update {
            Some(update) if len > 0 => (old + update.delta(&x.narrow(0, at, len)?)?)?,
            _ => old,
        });
    }
    Tensor::cat(&parts, 0)
}

/// Mean real-token residual per sequence, before the adapted layers. Empty padding rows are zero.
fn means(h: &Tensor, lens: &[usize], packed: bool) -> Result<Tensor> {
    let d = h.dim(D::Minus1)?;
    let mut at = 0;
    let mut means = Vec::with_capacity(lens.len());
    for (i, &n) in lens.iter().enumerate() {
        let x = if packed {
            h.narrow(0, at, n)?
        } else {
            h.get(i)?.narrow(0, 0, n)?
        };
        means.push(if n == 0 {
            Tensor::zeros(d, h.dtype(), h.device())?
        } else {
            (Tensor::ones((1, n), h.dtype(), h.device())?
                .matmul(&x)?
                .squeeze(0)?
                / n as f64)?
        });
        at += n;
    }
    Tensor::stack(&means, 0)
}

impl ModernBert {
    /// One pass through the frozen lower stack, call `route` exactly once on its `[B,d]` mean
    /// residuals, then continue the same pass through the upper stack. `None` selects the
    /// original operations. The router returns one optional bank index per sequence.
    ///
    /// The bank owns only low-rank factors. No checkpoint or upper-stack copy is instantiated.
    /// Right-padded inputs only, as in the packed inference path. An empty bank skips the router
    /// and calls the old forward directly. The ordinary `forward` method is unaffected.
    pub fn forward_routed(
        &self,
        ids: &Tensor,
        mask: &Tensor,
        first: usize,
        bank: &[Adapter],
        mut route: impl FnMut(&Tensor) -> Result<Vec<Option<usize>>>,
    ) -> Result<Tensor> {
        if bank.is_empty() {
            return self.forward(ids, mask);
        }
        let n_layers = self.stack.layers.len();
        if first >= n_layers {
            hanzo_ml::bail!("router boundary outside encoder");
        }
        let (b, _l) = ids.dims2()?;
        if mask.dims() != ids.dims() {
            hanzo_ml::bail!("routing mask shape");
        }
        let d = self.word_embeddings.embeddings().dim(1)?;
        for a in bank {
            a.check(first, n_layers, d)?;
        }
        let masks = mask.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let lens: Vec<usize> = masks
            .iter()
            .map(|r| r.iter().take_while(|&&x| x != 0).count())
            .collect();
        if masks
            .iter()
            .zip(&lens)
            .any(|(r, &n)| r[n..].iter().any(|&x| x != 0))
        {
            hanzo_ml::bail!("routed encoder requires right padding");
        }

        // Run the frozen lower stack through `first`, preserving packed/padded/YaRN/D-LSA semantics.
        let mut s = self.stack.to(self.embed(ids, mask)?, first)?;
        let packed = matches!(s.form, Form::Packed { .. });
        let selected = route(&means(&s.h, &s.lens, packed)?)?;
        if selected.len() != b || selected.iter().flatten().any(|&i| i >= bank.len()) {
            hanzo_ml::bail!("router returned invalid adapter indices");
        }
        let any_selected = selected.iter().any(Option::is_some);
        if any_selected {
            for (i, layer) in self.stack.layers[first..].iter().enumerate() {
                let layer_idx = first + i;
                let routing = Routing {
                    bank,
                    selected: &selected,
                    lens: &s.lens,
                    packed,
                    layer: layer_idx,
                };
                let _span =
                    tracing::span!(tracing::Level::TRACE, "layer", at = layer_idx).entered();
                s.h = match &s.form {
                    Form::Packed { .. } => {
                        layer.packed_routed(&s.h, &s.lens, self.stack.window, Some(&routing))?
                    }
                    Form::Padded { masks, long } => layer.forward_routed(
                        &s.h,
                        masks,
                        match long {
                            Some(p) => Place::Long(p),
                            None => Place::Index,
                        },
                        Some(&routing),
                    )?,
                };
            }
            s.at = self.stack.at + n_layers;
        }
        self.stack.finish(s)
    }
}
