//! Compute policy — the one place a model decides how its activations flow.
//!
//! `hanzo-ml` departs from candle's "dtype is a runtime property checked (and
//! bailed on) at every op" model. That design makes every binary op a *partial
//! morphism*: defined only when operands agree, otherwise a runtime crash deep in
//! a kernel. A single model forward threads dtype through dozens of ops; one wrong
//! (or missing) cast is a production crash, not a compile error.
//!
//! The fix is categorical: **factor the failure-causing properties (compute dtype,
//! accumulation dtype, KV-cache dtype, device) out of the value and into one
//! threaded context, so the ops a model composes become *total*.**
//!
//! Three structures, each a named concept rather than scattered `to_dtype` calls:
//!
//! 1. [`ComputeCtx`] — the single policy value. The *only* place dtype/device are
//!    decided for a model; every consumer reads it, none re-decides. (A model that
//!    threads `ctx` cannot reproduce the BF16-cache-vs-F32-activation class of bug.)
//!
//! 2. [`ComputeCtx::precise`] — the **upcast ⊣ downcast adjunction**. Precision-
//!    sensitive ops (RMS-norm, RoPE, softmax, MoE gate) are `downcast ∘ op_accum ∘
//!    upcast`. One combinator instead of the same three-line dance hand-rolled at
//!    every call site.
//!
//! 3. [`promote`] / [`promoted`] — the dtype **join-semilattice** (NumPy/JAX-style
//!    promotion). The canonical *minimal* cast for a binary op: lift both operands
//!    to `a ⊔ b`, never a surprise downcast, and never cast the operand that already
//!    sits at the join. This is how you avoid the unnecessary casts.

use crate::{DType, Device, Result, Tensor};

/// The compute policy for one model's forward pass: a single value, threaded to
/// every consumer (carrier, attention, KV cache, MoE). The one place dtype and
/// device are decided.
///
/// `compute` is the activation dtype (carrier / attention / matmuls). `kv` is the
/// KV-cache storage dtype (often `== compute`; pin to `F32` when decode stability
/// matters more than cache footprint). `accum` is what precision-sensitive ops
/// upcast to internally (default `F32`). Picking these per model is a *policy
/// choice* (see [`ComputeCtx::ds4_parity`] vs [`ComputeCtx::serving`]), not a
/// hardcode buried in the model.
#[derive(Clone, Debug)]
pub struct ComputeCtx {
    /// Dtype activations flow in (carrier, attention, projections).
    pub compute: DType,
    /// Dtype the KV cache is stored at.
    pub kv: DType,
    /// Dtype precision-sensitive ops accumulate in internally.
    pub accum: DType,
    /// Device the model runs on.
    pub device: Device,
}

impl ComputeCtx {
    /// Default policy: activations + KV at `compute`, accumulate in F32.
    pub fn new(compute: DType, device: Device) -> Self {
        Self {
            compute,
            kv: compute,
            accum: DType::F32,
            device,
        }
    }

    /// Numeric-parity policy (e.g. matching antirez/ds4): everything F32. Slower,
    /// bit-faithful to an all-F32 reference. Weights stay quantized regardless.
    pub fn ds4_parity(device: Device) -> Self {
        Self {
            compute: DType::F32,
            kv: DType::F32,
            accum: DType::F32,
            device,
        }
    }

    /// Serving policy: BF16 activations (bandwidth), F32 KV cache (decode
    /// stability), F32 accumulation. The production default on accelerators.
    pub fn serving(device: Device) -> Self {
        Self {
            compute: DType::BF16,
            kv: DType::F32,
            accum: DType::F32,
            device,
        }
    }

    /// Override the KV-cache dtype.
    pub fn with_kv(mut self, kv: DType) -> Self {
        self.kv = kv;
        self
    }

    /// Override the accumulation dtype.
    pub fn with_accum(mut self, accum: DType) -> Self {
        self.accum = accum;
        self
    }

    /// The **upcast ⊣ downcast adjunction**. Run `f` in the accumulation dtype and
    /// return the input's dtype — `t.dtype() → accum → t.dtype()`. Precision-
    /// sensitive ops (RMS-norm, RoPE, softmax, gate) factor through this instead of
    /// hand-rolling the cast dance. No-op when `t` is already at `accum` (so the
    /// all-F32 policy pays nothing).
    pub fn precise<F>(&self, t: &Tensor, f: F) -> Result<Tensor>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        let orig = t.dtype();
        if orig == self.accum {
            return f(t);
        }
        let up = t.to_dtype(self.accum)?;
        f(&up)?.to_dtype(orig)
    }

    /// Cast a tensor to the compute dtype (the carrier-entry boundary), no-op if
    /// already there.
    pub fn to_compute(&self, t: &Tensor) -> Result<Tensor> {
        cast_if(t, self.compute)
    }

    /// Cast a tensor to the KV-cache dtype (the cache-append boundary), no-op if
    /// already there. This is the single boundary that prevents the cache-vs-
    /// activation dtype split.
    pub fn to_kv(&self, t: &Tensor) -> Result<Tensor> {
        cast_if(t, self.kv)
    }
}

/// Cast only if needed — the minimal cast (idempotent identity at the target).
pub fn cast_if(t: &Tensor, dt: DType) -> Result<Tensor> {
    if t.dtype() == dt {
        Ok(t.clone())
    } else {
        t.to_dtype(dt)
    }
}

/// Float promotion rank for the join-semilattice. `None` for non-float dtypes
/// (integers / block-quants do not participate in float promotion).
fn float_rank(d: DType) -> Option<u8> {
    match d {
        DType::F64 => Some(4),
        DType::F32 => Some(3),
        // F16 and BF16 share width but are *incomparable* (different exponent/
        // mantissa splits): their join is F32, never one or the other.
        DType::F16 | DType::BF16 => Some(2),
        _ => None,
    }
}

/// Dtype **join** (least upper bound) in the promotion semilattice. Commutative,
/// associative, idempotent. `F32 ⊔ BF16 = F32`, `BF16 ⊔ F16 = F32` (incomparable
/// same-width → promote to the safe common float), `BF16 ⊔ BF16 = BF16`. For non-
/// float / mixed pairs there is no float join, so the left operand is returned
/// (callers that mix a quant with a float should `dequantize` first — an explicit
/// boundary, not an implicit promotion).
pub fn promote(a: DType, b: DType) -> DType {
    if a == b {
        return a;
    }
    match (float_rank(a), float_rank(b)) {
        (Some(ra), Some(rb)) => {
            if ra == rb {
                // same width, incomparable (F16 vs BF16) → F32
                DType::F32
            } else if ra > rb {
                a
            } else {
                b
            }
        }
        _ => a,
    }
}

/// Bring two tensors to their promoted dtype — the canonical minimal cast for a
/// binary op. Only the operand below the join is cast; an operand already at the
/// join is returned untouched (no unnecessary cast).
pub fn promoted(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let p = promote(a.dtype(), b.dtype());
    Ok((cast_if(a, p)?, cast_if(b, p)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promote_is_a_semilattice() {
        use DType::*;
        // idempotent
        for d in [F32, F16, BF16, F64] {
            assert_eq!(promote(d, d), d);
        }
        // commutative
        assert_eq!(promote(BF16, F32), promote(F32, BF16));
        assert_eq!(promote(F16, BF16), promote(BF16, F16));
        // joins
        assert_eq!(promote(BF16, F32), F32);
        assert_eq!(promote(F16, F32), F32);
        assert_eq!(promote(F16, BF16), F32); // incomparable same-width -> F32
        assert_eq!(promote(F32, F64), F64);
        // associative on the float sub-lattice
        assert_eq!(
            promote(promote(BF16, F16), F32),
            promote(BF16, promote(F16, F32))
        );
    }

    #[test]
    fn promoted_only_casts_the_lower_operand() -> Result<()> {
        let dev = Device::Cpu;
        let a = Tensor::zeros((2, 2), DType::BF16, &dev)?;
        let b = Tensor::zeros((2, 2), DType::F32, &dev)?;
        let (a2, b2) = promoted(&a, &b)?;
        assert_eq!(a2.dtype(), DType::F32); // bf16 lifted
        assert_eq!(b2.dtype(), DType::F32); // f32 untouched (same handle dtype)
        Ok(())
    }

    #[test]
    fn precise_runs_in_accum_and_restores_dtype() -> Result<()> {
        let dev = Device::Cpu;
        let ctx = ComputeCtx::new(DType::BF16, dev.clone());
        let t = Tensor::zeros((4,), DType::BF16, &dev)?;
        // The closure observes F32 (accum); the result is restored to BF16.
        let out = ctx.precise(&t, |x| {
            assert_eq!(x.dtype(), DType::F32);
            x.affine(1.0, 1.0)
        })?;
        assert_eq!(out.dtype(), DType::BF16);
        Ok(())
    }

    #[test]
    fn precise_is_noop_cast_when_already_accum() -> Result<()> {
        let dev = Device::Cpu;
        let ctx = ComputeCtx::ds4_parity(dev.clone()); // accum == compute == F32
        let t = Tensor::zeros((4,), DType::F32, &dev)?;
        let out = ctx.precise(&t, |x| {
            assert_eq!(x.dtype(), DType::F32);
            Ok(x.clone())
        })?;
        assert_eq!(out.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn policies_set_expected_dtypes() {
        let dev = Device::Cpu;
        let s = ComputeCtx::serving(dev.clone());
        assert_eq!((s.compute, s.kv, s.accum), (DType::BF16, DType::F32, DType::F32));
        let p = ComputeCtx::ds4_parity(dev);
        assert_eq!((p.compute, p.kv, p.accum), (DType::F32, DType::F32, DType::F32));
    }
}
