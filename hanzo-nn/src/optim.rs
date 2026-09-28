//! Various optimization algorithms.
use hanzo_ml::{Result, Tensor, Var};

/// The interface optimizers should implement.
pub trait Optimizer: Sized {
    type Config: Sized;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self>;

    fn step(&mut self, grads: &hanzo_ml::backprop::GradStore) -> Result<()>;

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, lr: f64);

    fn empty(config: Self::Config) -> Result<Self> {
        Self::new(vec![], config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(vars: &[&Var], config: Self::Config) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, config)
    }
}

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SGD {
    type Config = f64;

    fn new(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self {
            vars,
            learning_rate,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn step(&mut self, grads: &hanzo_ml::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}

#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type Config = ParamsAdamW;

    fn new(vars: Vec<Var>, params: ParamsAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr
    }

    fn step(&mut self, grads: &hanzo_ml::backprop::GradStore) -> Result<()> {
        self.step_scaled(grads, 1.0)
    }
}

impl AdamW {
    /// One step with every gradient multiplied by `grad_scale` first.
    pub fn step_scaled(
        &mut self,
        grads: &hanzo_ml::backprop::GradStore,
        grad_scale: f64,
    ) -> Result<()> {
        self.step_t += 1;
        let p = &self.params;
        let u = Update {
            lr: p.lr,
            beta1: p.beta1,
            beta2: p.beta2,
            eps: p.eps,
            weight_decay: p.weight_decay,
            scale_m: 1f64 / (1f64 - p.beta1.powi(self.step_t as i32)),
            scale_v: 1f64 / (1f64 - p.beta2.powi(self.step_t as i32)),
            grad_scale,
        };
        for var in self.vars.iter() {
            if let Some(g) = grads.get(&var.var) {
                adamw(&var.var, g, &var.first_moment, &var.second_moment, &u)?;
            }
        }
        Ok(())
    }

    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW {
            lr: learning_rate,
            ..ParamsAdamW::default()
        };
        Self::new(vars, params)
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

/// One AdamW update's constants: the learning rate, betas, epsilon and decoupled weight decay,
/// the bias corrections `1 / (1 − βᵗ)` of step `t`, and the factor every gradient is multiplied
/// by first (clipping, loss scaling).
#[derive(Clone, Copy, Debug)]
pub struct Update {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub scale_m: f64,
    pub scale_v: f64,
    pub grad_scale: f64,
}

/// One AdamW update of `theta` from its gradient `g` and moments `m`, `v`:
///
///   m ← β₁m + (1 − β₁)g,  v ← β₂v + (1 − β₂)g²,
///   θ ← θ(1 − lr·λ) − lr · m·scale_m / (√(v·scale_v) + ε).
///
/// One kernel for a contiguous F32 parameter on Metal or CUDA, tensor ops elsewhere.
pub fn adamw(theta: &Var, g: &Tensor, m: &Var, v: &Var, u: &Update) -> Result<()> {
    #[cfg(any(feature = "metal", feature = "cuda"))]
    if fused::fits(theta, g) {
        let step = fused::Step {
            lr: u.lr as f32,
            beta1: u.beta1 as f32,
            beta2: u.beta2 as f32,
            eps: u.eps as f32,
            weight_decay: u.weight_decay as f32,
            scale_m: u.scale_m as f32,
            scale_v: u.scale_v as f32,
            grad_scale: u.grad_scale as f32,
        };
        return theta
            .as_tensor()
            .inplace_op([g, m.as_tensor(), v.as_tensor()], &fused::AdamW(step));
    }
    let g = if u.grad_scale == 1.0 {
        g.clone()
    } else {
        (g * u.grad_scale)?
    };
    let next_m = ((m.as_tensor() * u.beta1)? + (&g * (1.0 - u.beta1))?)?;
    let next_v = ((v.as_tensor() * u.beta2)? + (g.sqr()? * (1.0 - u.beta2))?)?;
    let m_hat = (&next_m * u.scale_m)?;
    let v_hat = (&next_v * u.scale_v)?;
    let next_theta = (theta.as_tensor() * (1f64 - u.lr * u.weight_decay))?;
    let adjusted_grad = (m_hat / (v_hat.sqrt()? + u.eps)?)?;
    let next_theta = (next_theta - (adjusted_grad * u.lr)?)?;
    m.set(&next_m)?;
    v.set(&next_v)?;
    theta.set(&next_theta)
}

/// The global L2 norm of the gradients of `vars`, `sqrt(Σ‖g‖²)`: one host read.
pub fn grad_norm(grads: &hanzo_ml::backprop::GradStore, vars: &[Var]) -> Result<f64> {
    let present: Vec<&Tensor> = vars
        .iter()
        .filter_map(|v| grads.get(v.as_tensor()))
        .collect();
    let Some(first) = present.first() else {
        return Ok(0.0);
    };
    #[cfg(any(feature = "metal", feature = "cuda"))]
    if fused::on(first.device()) {
        let acc = Tensor::zeros(1, hanzo_ml::DType::F32, first.device())?;
        let mut rest = Vec::new();
        for g in &present {
            if fused::fits(g, g) {
                acc.inplace_op([*g], &fused::SumSq)?;
            } else {
                rest.push(g.to_dtype(hanzo_ml::DType::F32)?.sqr()?.sum_all()?);
            }
        }
        let mut total = acc.sum_all()?;
        if !rest.is_empty() {
            total = (total + Tensor::stack(&rest, 0)?.sum_all()?)?;
        }
        return Ok((total.to_scalar::<f32>()? as f64).sqrt());
    }
    let _ = first;
    let sq = present
        .iter()
        .map(|g| g.to_dtype(hanzo_ml::DType::F32)?.sqr()?.sum_all())
        .collect::<Result<Vec<_>>>()?;
    Ok((Tensor::stack(&sq, 0)?.sum_all()?.to_scalar::<f32>()? as f64).sqrt())
}

#[cfg(any(feature = "metal", feature = "cuda"))]
mod fused {
    use hanzo_ml::{DType, Device, InplaceOpN, Layout, Result, Tensor};

    #[derive(Clone, Copy)]
    pub struct Step {
        pub lr: f32,
        pub beta1: f32,
        pub beta2: f32,
        pub eps: f32,
        pub weight_decay: f32,
        pub scale_m: f32,
        pub scale_v: f32,
        pub grad_scale: f32,
    }

    pub fn on(dev: &Device) -> bool {
        dev.is_metal() || dev.is_cuda()
    }

    pub fn fits(theta: &Tensor, g: &Tensor) -> bool {
        on(theta.device())
            && theta.dtype() == DType::F32
            && g.dtype() == DType::F32
            && theta.is_contiguous()
            && g.is_contiguous()
    }

    #[cfg(feature = "metal")]
    fn at<'a>(s: &'a hanzo_ml::MetalStorage, l: &Layout) -> hanzo_metal_kernels::BufferOffset<'a> {
        hanzo_metal_kernels::BufferOffset {
            buffer: s.buffer(),
            offset_in_bytes: l.start_offset() * DType::F32.size_in_bytes(),
        }
    }

    pub struct AdamW(pub Step);

    impl InplaceOpN<3> for AdamW {
        fn name(&self) -> &'static str {
            "adamw"
        }

        #[cfg(feature = "metal")]
        fn metal_fwd(
            &self,
            w: &mut hanzo_ml::MetalStorage,
            wl: &Layout,
            [(g, gl), (m, ml), (v, vl)]: [(&hanzo_ml::MetalStorage, &Layout); 3],
        ) -> Result<()> {
            use hanzo_ml::backend::BackendStorage;
            let device = w.device().clone();
            let encoder = device.command_encoder()?;
            encoder.set_label("adamw");
            let s = self.0;
            let step = hanzo_metal_kernels::AdamWStep {
                lr: s.lr,
                beta1: s.beta1,
                beta2: s.beta2,
                eps: s.eps,
                weight_decay: s.weight_decay,
                scale_m: s.scale_m,
                scale_v: s.scale_v,
                grad_scale: s.grad_scale,
            };
            hanzo_metal_kernels::call_adamw(
                device.metal_device(),
                &encoder,
                device.kernels(),
                wl.shape().elem_count(),
                step,
                at(w, wl),
                at(g, gl),
                at(m, ml),
                at(v, vl),
            )
            .map_err(hanzo_ml::Error::wrap)
        }

        #[cfg(feature = "cuda")]
        fn cuda_fwd(
            &self,
            w: &mut hanzo_ml::CudaStorage,
            wl: &Layout,
            [(g, gl), (m, ml), (v, vl)]: [(&hanzo_ml::CudaStorage, &Layout); 3],
        ) -> Result<()> {
            use hanzo_ml::backend::BackendStorage;
            use hanzo_ml::cuda_backend::cudarc::driver::PushKernelArg;
            use hanzo_ml::cuda_backend::{kernels, WrapErr};
            let dev = w.device().clone();
            let n = wl.shape().elem_count();
            let s = self.0;
            let func = dev.get_or_load_func("adamw_f32", &kernels::OPTIM)?;
            let mut w = w.as_cuda_slice_mut::<f32>()?.slice_mut(wl.start_offset()..);
            let g = g.as_cuda_slice::<f32>()?.slice(gl.start_offset()..);
            let m = m.as_cuda_slice::<f32>()?.slice(ml.start_offset()..);
            let v = v.as_cuda_slice::<f32>()?.slice(vl.start_offset()..);
            let mut b = func.builder();
            hanzo_ml::builder_arg!(
                b,
                n as u32,
                s.lr,
                s.beta1,
                s.beta2,
                s.eps,
                s.weight_decay,
                s.scale_m,
                s.scale_v,
                s.grad_scale
            );
            b.arg(&mut w);
            b.arg(&g);
            b.arg(&m);
            b.arg(&v);
            // SAFETY: ffi.
            unsafe { b.launch(super::launch(n)) }.w()?;
            Ok(())
        }
    }

    pub struct SumSq;

    impl InplaceOpN<1> for SumSq {
        fn name(&self) -> &'static str {
            "sumsq"
        }

        #[cfg(feature = "metal")]
        fn metal_fwd(
            &self,
            acc: &mut hanzo_ml::MetalStorage,
            _acc_l: &Layout,
            [(x, xl)]: [(&hanzo_ml::MetalStorage, &Layout); 1],
        ) -> Result<()> {
            use hanzo_ml::backend::BackendStorage;
            let device = acc.device().clone();
            let encoder = device.command_encoder()?;
            encoder.set_label("sumsq");
            hanzo_metal_kernels::call_sumsq(
                device.metal_device(),
                &encoder,
                device.kernels(),
                xl.shape().elem_count(),
                at(x, xl),
                acc.buffer(),
            )
            .map_err(hanzo_ml::Error::wrap)
        }

        #[cfg(feature = "cuda")]
        fn cuda_fwd(
            &self,
            acc: &mut hanzo_ml::CudaStorage,
            acc_l: &Layout,
            [(x, xl)]: [(&hanzo_ml::CudaStorage, &Layout); 1],
        ) -> Result<()> {
            use hanzo_ml::backend::BackendStorage;
            use hanzo_ml::cuda_backend::cudarc::driver::PushKernelArg;
            use hanzo_ml::cuda_backend::{kernels, WrapErr};
            let dev = acc.device().clone();
            let n = xl.shape().elem_count();
            let func = dev.get_or_load_func("sumsq_f32", &kernels::OPTIM)?;
            let mut acc = acc
                .as_cuda_slice_mut::<f32>()?
                .slice_mut(acc_l.start_offset()..);
            let x = x.as_cuda_slice::<f32>()?.slice(xl.start_offset()..);
            let mut b = func.builder();
            hanzo_ml::builder_arg!(b, n as u32);
            b.arg(&x);
            b.arg(&mut acc);
            // SAFETY: ffi.
            unsafe { b.launch(super::launch(n)) }.w()?;
            Ok(())
        }
    }
}

#[cfg(feature = "cuda")]
fn launch(n: usize) -> hanzo_ml::cuda_backend::cudarc::driver::LaunchConfig {
    hanzo_ml::cuda_backend::cudarc::driver::LaunchConfig {
        grid_dim: (n.div_ceil(256).clamp(1, 4096) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Configuration parameters for the Muon (Momentum Orthogonalized Update via Newton-Schulz) optimizer.
#[derive(Clone, Debug)]
pub struct ParamsMuon {
    pub lr: f64,
    pub momentum: f64,
    pub n_iterations: usize,
    pub weight_decay: f64,
}

impl Default for ParamsMuon {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            n_iterations: 5,
            weight_decay: 0.01,
        }
    }
}

/// Computes the orthogonalized matrix update via quintic Newton-Schulz iteration.
///
/// Solves $X_{k+1} = a X + b (X X^T) X + c (X X^T)^2 X$ using optimal convergence
/// coefficients $(a=3.4445, b=-4.7750, c=2.0315)$.
pub fn newton_schulz(g: &Tensor, steps: usize) -> Result<Tensor> {
    let shape = g.shape();
    let dims = shape.dims();
    if dims.len() < 2 {
        return Ok(g.clone());
    }
    let d0 = dims[0];
    let d1: usize = dims[1..].iter().product();
    let g_flat = g.reshape((d0, d1))?;

    let transposed = d0 > d1;
    let x = if transposed { g_flat.t()? } else { g_flat };

    let norm = x
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_dtype(hanzo_ml::DType::F64)?
        .to_scalar::<f64>()?;
    let mut x = x.affine(1.0 / (norm + 1e-7), 0.0)?;

    let a = 3.4445f64;
    let b = -4.7750f64;
    let c = 2.0315f64;

    for _ in 0..steps {
        let xt = x.t()?;
        let a_mat = x.matmul(&xt)?;
        let a_sq = a_mat.matmul(&a_mat)?;
        let b_part = a_mat.affine(b, 0.0)?;
        let c_part = a_sq.affine(c, 0.0)?;
        let b_mat = b_part.add(&c_part)?;
        let bx = b_mat.matmul(&x)?;
        let ax = x.affine(a, 0.0)?;
        x = ax.add(&bx)?;
    }

    let res = if transposed { x.t()? } else { x };
    res.reshape(shape)
}

#[derive(Debug)]
struct VarMuon {
    var: Var,
    momentum: Var,
}

/// Muon for 2-D weight matrices: updates orthogonalized by Newton-Schulz iterations.
#[derive(Debug)]
pub struct Muon {
    vars: Vec<VarMuon>,
    params: ParamsMuon,
}

impl Optimizer for Muon {
    type Config = ParamsMuon;

    fn new(vars: Vec<Var>, params: ParamsMuon) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float() && var.shape().dims().len() >= 2)
            .map(|var| {
                let shape = var.shape();
                let dtype = var.dtype();
                let device = var.device();
                let momentum = Var::zeros(shape, dtype, device)?;
                Ok(VarMuon { var, momentum })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { vars, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &hanzo_ml::backprop::GradStore) -> Result<()> {
        let lr = self.params.lr;
        let beta = self.params.momentum;
        let lambda = self.params.weight_decay;
        let steps = self.params.n_iterations;

        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.momentum;
            if let Some(g) = grads.get(theta) {
                let m_decayed = m.as_tensor().affine(beta, 0.0)?;
                let g_scaled = g.affine(1.0 - beta, 0.0)?;
                let next_m = m_decayed.add(&g_scaled)?;
                let ortho_update = newton_schulz(&next_m, steps)?;
                let next_theta = theta.as_tensor().affine(1.0 - lr * lambda, 0.0)?;
                let step_update = ortho_update.affine(-lr, 0.0)?;
                let next_theta = next_theta.add(&step_update)?;
                m.set(&next_m)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

/// Dual optimizer combining Muon for 2D+ matrix projections and AdamW for vectors/biases/embeddings.
#[derive(Debug)]
pub struct HybridMuonAdamW {
    muon: Muon,
    adamw: AdamW,
}

impl HybridMuonAdamW {
    pub fn new(vars: Vec<Var>, muon_params: ParamsMuon, adamw_params: ParamsAdamW) -> Result<Self> {
        let mut matrix_vars = Vec::new();
        let mut vector_vars = Vec::new();

        for var in vars {
            if var.shape().dims().len() >= 2 {
                matrix_vars.push(var);
            } else {
                vector_vars.push(var);
            }
        }

        let muon = Muon::new(matrix_vars, muon_params)?;
        let adamw = AdamW::new(vector_vars, adamw_params)?;
        Ok(Self { muon, adamw })
    }

    pub fn step(&mut self, grads: &hanzo_ml::backprop::GradStore) -> Result<()> {
        self.muon.step(grads)?;
        self.adamw.step(grads)?;
        Ok(())
    }

    pub fn set_learning_rates(&mut self, muon_lr: f64, adamw_lr: f64) {
        self.muon.set_learning_rate(muon_lr);
        self.adamw.set_learning_rate(adamw_lr);
    }
}

/// Configuration parameters for MuonClip (Moonshot AI Kimi K2/K3 matrix optimizer).
///
/// Extends Muon with:
/// 1. Consistent RMS matching across layer types to prevent gradient magnitude mismatch with AdamW.
/// 2. QK-Clip: Clamping projection gradient norm to prevent softmax saturation and loss spikes in long-context attention.
#[derive(Clone, Debug)]
pub struct ParamsMuonClip {
    pub lr: f64,
    pub momentum: f64,
    pub n_iterations: usize,
    pub weight_decay: f64,
    pub qk_clip_threshold: Option<f64>,
    pub rms_match: bool,
}

impl Default for ParamsMuonClip {
    fn default() -> Self {
        Self {
            lr: 0.02,
            momentum: 0.95,
            n_iterations: 5,
            weight_decay: 0.01,
            qk_clip_threshold: Some(30.0),
            rms_match: true,
        }
    }
}

/// MuonClip optimizer based on Moonshot AI Kimi K2/K3 frontier training recipes.
///
/// Combines momentum orthogonalization (Newton-Schulz) with consistent RMS matching and
/// QK-clipping to eliminate loss spikes and maintain training stability over long sequences.
#[derive(Debug)]
pub struct MuonClip {
    vars: Vec<VarMuon>,
    params: ParamsMuonClip,
}

impl Optimizer for MuonClip {
    type Config = ParamsMuonClip;

    fn new(vars: Vec<Var>, params: ParamsMuonClip) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float() && var.shape().dims().len() >= 2)
            .map(|var| {
                let shape = var.shape();
                let dtype = var.dtype();
                let device = var.device();
                let momentum = Var::zeros(shape, dtype, device)?;
                Ok(VarMuon { var, momentum })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { vars, params })
    }

    fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    fn step(&mut self, grads: &hanzo_ml::backprop::GradStore) -> Result<()> {
        let lr = self.params.lr;
        let beta = self.params.momentum;
        let lambda = self.params.weight_decay;
        let steps = self.params.n_iterations;

        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.momentum;
            if let Some(g) = grads.get(theta) {
                let m_decayed = m.as_tensor().affine(beta, 0.0)?;
                let g_scaled = g.affine(1.0 - beta, 0.0)?;
                let next_m = m_decayed.add(&g_scaled)?;
                let mut ortho_update = newton_schulz(&next_m, steps)?;

                // 1. Consistent RMS matching
                if self.params.rms_match {
                    let dims = theta.shape().dims();
                    let d0 = dims[0];
                    let d1: usize = dims[1..].iter().product();
                    let numel = (d0 * d1) as f64;
                    let target_rms = 1.0 / (d0.max(d1) as f64).sqrt();

                    let sum_sq = ortho_update
                        .sqr()?
                        .sum_all()?
                        .to_dtype(hanzo_ml::DType::F64)?
                        .to_scalar::<f64>()?;
                    let current_rms = (sum_sq / numel).sqrt();
                    let scale = target_rms / (current_rms + 1e-7);
                    ortho_update = ortho_update.affine(scale, 0.0)?;
                }

                // 2. QK-Clip: prevent loss spikes and softmax saturation
                if let Some(clip_threshold) = self.params.qk_clip_threshold {
                    let frob_norm = ortho_update
                        .sqr()?
                        .sum_all()?
                        .sqrt()?
                        .to_dtype(hanzo_ml::DType::F64)?
                        .to_scalar::<f64>()?;
                    if frob_norm > clip_threshold {
                        let scale = clip_threshold / (frob_norm + 1e-7);
                        ortho_update = ortho_update.affine(scale, 0.0)?;
                    }
                }

                // 3. Update weights with decoupled decay
                let next_theta = theta.as_tensor().affine(1.0 - lr * lambda, 0.0)?;
                let step_update = ortho_update.affine(-lr, 0.0)?;
                let next_theta = next_theta.add(&step_update)?;
                m.set(&next_m)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    #[test]
    fn test_newton_schulz_orthogonalization() -> Result<()> {
        let dev = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 0.0, 0.5, 0.0, 1.0, 0.0, -0.5, 0.0, 1.0];
        let t = Tensor::from_vec(data, (3, 3), &dev)?;
        let ortho = newton_schulz(&t, 5)?;
        let prod = ortho.matmul(&ortho.t()?)?;
        let mat = prod.to_vec2::<f32>()?;
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert!(
                        mat[i][j].abs() < 1e-4,
                        "Off-diag not close to 0: {}",
                        mat[i][j]
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_muon_step() -> Result<()> {
        let dev = Device::Cpu;
        let w = Var::from_tensor(&Tensor::randn(0.0f32, 1.0f32, (4, 4), &dev)?)?;
        let mut muon = Muon::new(vec![w.clone()], ParamsMuon::default())?;

        let loss = w.as_tensor().sqr()?.sum_all()?;
        let grads = loss.backward()?;
        muon.step(&grads)?;
        assert_eq!(muon.learning_rate(), 0.02);
        Ok(())
    }
}
