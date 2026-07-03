// Foundational probe: do cmma tensor cores work on GB10 (sm_121 Blackwell) through the DSL?
// Out = A @ B^T (16x16x16, f16 in -> f32 acc) via cmma, checked vs CPU reference.
use hanzo_kernel::prelude::*;
use hanzo_kernel::cubecl::prelude::cmma;
use half::f16;

#[kernel(targets(cuda))]
fn cmma_mm(lhs: &Array<f16>, rhs: &Array<f16>, out: &mut Array<f32>) {
    let a = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &lhs.to_slice(), 16);
    let b = cmma::Matrix::<f16>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &rhs.to_slice(), 16);
    let c = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);
    cmma::store(&mut out.to_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
}

fn main() {
    use hanzo_kernel::cubecl::cuda::{CudaDevice, CudaRuntime};
    let client = CudaRuntime::client(&CudaDevice::default());
    let n = 16usize;
    let lhs: Vec<f16> = (0..n*n).map(|i| f16::from_f32((i%7) as f32 * 0.1)).collect();
    let rhs: Vec<f16> = (0..n*n).map(|i| f16::from_f32((i%5) as f32 * 0.1)).collect();
    let lh = client.create_from_slice(f16::as_bytes(&lhs));
    let rh = client.create_from_slice(f16::as_bytes(&rhs));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0f32; n*n]));
    unsafe { cmma_mm::launch::<CudaRuntime>(&client, Grid::Static(1,1,1), Block::new_1d(32),
        ArrayArg::from_raw_parts(lh.clone(), n*n),
        ArrayArg::from_raw_parts(rh.clone(), n*n),
        ArrayArg::from_raw_parts(oh.clone(), n*n)); }
    let obytes = client.read_one_unchecked(oh); let got = f32::from_bytes(&obytes);
    let mut refv = vec![0f32; n*n]; let mut maxrel = 0f32;
    for i in 0..n { for j in 0..n { let mut s=0f32; for k in 0..n { s += lhs[i*n+k].to_f32()*rhs[j*n+k].to_f32(); } refv[i*n+j]=s; }}
    for i in 0..n*n { let d=(got[i]-refv[i]).abs()/refv[i].abs().max(1e-3); maxrel=maxrel.max(d); }
    println!("cmma GB10: got[0..4]={:?} ref[0..4]={:?} max_rel={:.2e} {}", &got[0..4], &refv[0..4], maxrel, if maxrel<2e-2 {"MATCH -- tensor cores WORK via DSL"} else {"MISMATCH"});
}
