//! The gated injection: gathered rows `e` into a delta for every residual stream.
//!
//! Per stream, a key projection of `e` meets the stream's own state (both RMS-normed) in a
//! scaled dot product; the gate `σ(sign(s)·√max(|s|, 10⁻⁶))` scales a value projection of `e`
//! shared by the streams. The gated value, RMS-normed, runs through a causal depthwise conv
//! whose taps sit `dilation` positions apart (the n-gram order), then a SiLU, and joins the gated
//! value: `Δ = gv + silu(conv(norm(gv)))`. An optional output gate, one weight per channel,
//! scales Δ; at zero the block adds exactly nothing, so a model given a fresh block computes as
//! it did without one.
//!
//! Numerics are Qwen3.8-Flash-Next's served kernels (vLLM `ple_layer.py:730-850`, inductor's
//! fused gate kernel): key and value are GEMMs in the streams' dtype; both norms, their dot
//! product and the gate stay F32; `gate · value` is stored (and normed from its unrounded square
//! sum); the norm is stored; the conv accumulates in F32 and is stored, then its SiLU. With F32
//! streams every rounding is a no-op.

use crate::ops::{sigmoid, silu};
use crate::{Init, VarBuilder};
use hanzo_ml::{bail, DType, IndexOp, Module, Result, Tensor, D};
use std::sync::Arc;

/// A projection of `e`: a dense or quantized linear map.
pub type Map = Arc<dyn Module + Send + Sync>;

/// Stages a probed forward notes, by name.
pub type Probe = Vec<(&'static str, Tensor)>;

pub struct Block {
    /// `e` to one key per stream, `(streams · hidden)` wide.
    pub key: Map,
    /// `e` to one value the streams share, `hidden` wide.
    pub value: Map,
    /// Per-stream RMS weights, `(streams · hidden)`, as multiplied (a Gemma `1 + w` folded in).
    pub norm_query: Tensor,
    pub norm_key: Tensor,
    pub norm_conv: Tensor,
    /// Depthwise conv taps, `(streams · hidden, kernel)`: tap `j` reads `j · dilation` back.
    pub conv: Tensor,
    pub dilation: usize,
    pub eps: f64,
    /// Output gate, `(streams · hidden)`; `None` adds Δ as it is.
    pub gate: Option<Tensor>,
}

/// RMS norm over each stream of `x` (.., streams, hidden) with that stream's slice of `w`
/// (streams · hidden), in F32 and not rounded (the served gate kernel keeps it in registers).
fn norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let (n, h) = (x.dim(D::Minus2)?, x.dim(D::Minus1)?);
    let f = x.to_dtype(DType::F32)?;
    let rstd = (f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    f.broadcast_mul(&rstd)?
        .broadcast_mul(&w.to_dtype(DType::F32)?.reshape((n, h))?)
}

impl Block {
    /// A fresh block at `vb` for `streams` streams of `hidden` from embeddings `width` wide:
    /// `key` and `value` (no bias), the three norms at one, conv taps uniform in ±1/√kernel,
    /// and, with `gate`, the output gate at zero.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: VarBuilder,
        width: usize,
        streams: usize,
        hidden: usize,
        kernel: usize,
        dilation: usize,
        eps: f64,
        gate: bool,
    ) -> Result<Block> {
        let c = streams * hidden;
        let bound = 1.0 / (kernel as f64).sqrt();
        Ok(Block {
            key: Arc::new(crate::linear_no_bias(width, c, vb.pp("key"))?),
            value: Arc::new(crate::linear_no_bias(width, hidden, vb.pp("value"))?),
            norm_query: vb.get_with_hints(c, "norm_query", Init::Const(1.0))?,
            norm_key: vb.get_with_hints(c, "norm_key", Init::Const(1.0))?,
            norm_conv: vb.get_with_hints(c, "norm_conv", Init::Const(1.0))?,
            conv: vb.get_with_hints(
                (c, kernel),
                "conv",
                Init::Uniform {
                    lo: -bound,
                    up: bound,
                },
            )?,
            dilation,
            eps,
            gate: if gate {
                Some(vb.get_with_hints(c, "gate", Init::Const(0.0))?)
            } else {
                None
            },
        })
    }

    /// The conv history a sequence carries: (channels, rows), rows = (kernel − 1) · dilation.
    pub fn history(&self) -> Result<(usize, usize)> {
        let (channels, kernel) = self.conv.dims2()?;
        Ok((channels, (kernel - 1) * self.dilation))
    }

    /// Δ for streams `x` `(batch, seq, streams, hidden)` from embeddings `e` `(batch, seq,
    /// width)`, after the conv history `state` `(batch, channels, rows)` (zeros: a new sequence).
    /// Returns Δ in F32 (the served graph adds it to the streams inside one kernel and rounds
    /// only the sum) and the history the next chunk continues from.
    pub fn forward(&self, x: &Tensor, e: &Tensor, state: &Tensor) -> Result<(Tensor, Tensor)> {
        self.probed(x, e, state, None)
    }

    /// [`Block::forward`], noting each stored stage in `probe` when given: `key`, `value`, `gv`,
    /// `nrm`, `conv`.
    pub fn probed(
        &self,
        x: &Tensor,
        e: &Tensor,
        state: &Tensor,
        mut probe: Option<&mut Probe>,
    ) -> Result<(Tensor, Tensor)> {
        let mut note = |name: &'static str, t: &Tensor| {
            if let Some(p) = probe.as_mut() {
                p.push((name, t.clone()));
            }
        };
        let (b, s, n, h) = x.dims4()?;
        let dtype = x.dtype();
        let store = |t: &Tensor| -> Result<Tensor> { t.to_dtype(dtype)?.to_dtype(DType::F32) };
        let e = e.to_dtype(dtype)?;
        let key = store(&self.key.forward(&e)?)?.reshape((b, s, n, h))?;
        let value = store(&self.value.forward(&e)?)?;
        note("key", &key);
        note("value", &value);
        let kn = norm(&key, &self.norm_key, self.eps)?;
        let qn = norm(&x.to_dtype(DType::F32)?, &self.norm_query, self.eps)?;
        let scale = (1.0 / (h as f64).sqrt()) as f32;
        let score = (kn * qn)?
            .sum_keepdim(D::Minus1)?
            .affine(f64::from(scale), 0.0)?;
        let gate = sigmoid(&(score.sign()? * score.abs()?.maximum(1e-6)?.sqrt()?)?)?;
        let gv_f = gate.broadcast_mul(&value.unsqueeze(2)?)?;
        let gv = store(&gv_f)?;
        note("gv", &gv);
        let ssq = gv_f.sqr()?.sum_keepdim(D::Minus1)?;
        let rstd = ((ssq / h as f64)? + self.eps)?.sqrt()?.recip()?;
        let u = gv
            .broadcast_mul(&rstd)?
            .broadcast_mul(&self.norm_conv.to_dtype(DType::F32)?.reshape((n, h))?)?
            .to_dtype(dtype)?
            .reshape((b, s, n * h))?;
        note("nrm", &u);
        let (c, next) = self.conv(&u, state)?;
        let c = c.to_dtype(DType::F32)?;
        note("conv", &c);
        let delta = (gv.reshape((b, s, n * h))? + c)?;
        let delta = match &self.gate {
            Some(g) => delta.broadcast_mul(&g.to_dtype(DType::F32)?)?,
            None => delta,
        };
        Ok((delta.reshape((b, s, n, h))?, next))
    }

    /// silu of the causal depthwise conv over `u` `(batch, seq, channels)` with taps at t, t−d,
    /// …, t−(k−1)·d, after the history `state` `(batch, channels, rows)`; also the last (k−1)·d
    /// inputs, `(batch, channels, rows)` in `u`'s dtype, which the next chunk reads as its
    /// history. The conv accumulates in F32 and is rounded to `u`'s dtype, then the silu is
    /// rounded again, as the eager bf16 `F.conv1d` then `F.silu` store them.
    pub fn conv(&self, u: &Tensor, state: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, s, _) = u.dims3()?;
        let (c, rows) = self.history()?;
        let (k, d) = (self.conv.dim(1)?, self.dilation);
        if state.dims3()? != (b, c, rows) {
            bail!(
                "conv history is {:?}, expected ({b}, {c}, {rows})",
                state.dims()
            );
        }
        // (batch, rows + seq, channels): the carried history, then this chunk.
        let history = Tensor::cat(&[&state.to_dtype(u.dtype())?.transpose(1, 2)?, u], 1)?;
        let next = history.narrow(1, s, rows)?.transpose(1, 2)?.contiguous()?;
        // out_t = Σ_j w_j ⊙ history[t + j·d] (ple_layer.py:820-826).
        let history = history.to_dtype(DType::F32)?;
        let w = self.conv.to_dtype(DType::F32)?;
        let mut out = history.narrow(1, 0, s)?.broadcast_mul(&w.i((.., 0))?)?;
        for j in 1..k {
            out = (out + history.narrow(1, j * d, s)?.broadcast_mul(&w.i((.., j))?)?)?;
        }
        let out = out.to_dtype(u.dtype())?.to_dtype(DType::F32)?;
        Ok((silu(&out)?.to_dtype(u.dtype())?, next))
    }
}
