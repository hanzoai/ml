//! Pin candle's `AdamW` step rule against a manual `torch.optim.AdamW`
//! reference implementation.
//!
//! We use a tiny 16-d quadratic loss `L(θ) = ½‖θ - target‖²` (so the
//! gradient is exactly `θ - target`), run 100 steps of both, and assert
//! the final parameter vectors agree to 1e-5.
//!
//! The PyTorch reference rule (decoupled weight decay, no AMSGrad):
//!
//! ```text
//! m_t = β1·m_{t-1} + (1-β1)·g
//! v_t = β2·v_{t-1} + (1-β2)·g²
//! m̂  = m_t / (1 - β1^t)
//! v̂  = v_t / (1 - β2^t)
//! θ_t = (1 - lr·wd) · θ_{t-1} - lr · m̂ / (√v̂ + ε)
//! ```

use candle::{DType, Device, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

const DIM: usize = 16;
const STEPS: usize = 100;
const LR: f64 = 1e-2;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const WD: f64 = 0.01;
const TOL: f64 = 1e-5;

fn reference_adamw(theta_init: &[f64], target: &[f64]) -> Vec<f64> {
    let mut theta: Vec<f64> = theta_init.to_vec();
    let mut m: Vec<f64> = vec![0.0; DIM];
    let mut v: Vec<f64> = vec![0.0; DIM];
    for step in 1..=STEPS {
        // Gradient of ½‖θ - target‖² is (θ - target).
        let g: Vec<f64> = theta.iter().zip(target).map(|(a, b)| a - b).collect();
        let bc1 = 1.0 - BETA1.powi(step as i32);
        let bc2 = 1.0 - BETA2.powi(step as i32);
        for i in 0..DIM {
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * g[i];
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * g[i] * g[i];
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            theta[i] = (1.0 - LR * WD) * theta[i] - LR * m_hat / (v_hat.sqrt() + EPS);
        }
    }
    theta
}

#[test]
fn candle_adamw_matches_torch_reference() -> candle::Result<()> {
    let device = Device::Cpu;
    // Deterministic initialisation: theta_i = (i+1)/10, target_i = -(i+1)/10
    let theta_init: Vec<f64> = (0..DIM).map(|i| (i + 1) as f64 / 10.0).collect();
    let target_vec: Vec<f64> = (0..DIM).map(|i| -((i + 1) as f64 / 10.0)).collect();

    // candle path (using f64 to match the reference exactly)
    let theta = Var::from_slice(&theta_init, DIM, &device)?;
    let target = Tensor::from_slice(&target_vec, DIM, &device)?;
    let mut opt = AdamW::new(
        vec![theta.clone()],
        ParamsAdamW {
            lr: LR,
            beta1: BETA1,
            beta2: BETA2,
            eps: EPS,
            weight_decay: WD,
        },
    )?;
    for _ in 0..STEPS {
        let loss = theta
            .as_tensor()
            .sub(&target)?
            .sqr()?
            .sum_all()?
            .affine(0.5, 0.0)?;
        opt.backward_step(&loss)?;
    }

    let got: Vec<f64> = theta.as_tensor().to_dtype(DType::F64)?.to_vec1()?;
    let want = reference_adamw(&theta_init, &target_vec);

    let mut max_abs = 0.0;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    eprintln!(
        "AdamW max abs diff vs torch reference after {} steps: {:.3e}",
        STEPS, max_abs
    );
    assert!(
        max_abs < TOL,
        "max abs diff {max_abs} exceeded tolerance {TOL}"
    );
    Ok(())
}
