fn load_morph(
    vb: &ShardedVarBuilder,
) -> Result<((Vec<u8>, usize, usize), (Vec<u8>, usize, usize), (Vec<u8>, usize, usize))> {
    let to_u8 = |name: &str, h: usize, w: usize| -> Result<(Vec<u8>, usize, usize)> {
        let t = vb.get((h, w), name)?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let v: Vec<f32> = t.flatten_all()?.to_vec1::<f32>()?;
        Ok((v.iter().map(|&x| u8::from(x != 0.0)).collect(), h, w))
    };
    let cone = to_u8("morph.cone_kernel", 33, 33)?;
    let cheek = to_u8("morph.cheek_kernel", 3, 35)?;
    let mask = to_u8("morph.cheek_mask", 512, 512)?;
    Ok((cone, cheek, mask))
}

fn iou_u8(a: &[u8], b: &[u8]) -> f64 {
    let (mut inter, mut union) = (0u64, 0u64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (xb, yb) = (x >= 128, y >= 128);
        if xb && yb {
            inter += 1;
        }
        if xb || yb {
            union += 1;
        }
    }
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

fn psnr_ssim_u8(a: &[u8], b: &[u8], h: usize, w: usize) -> (f64, f64) {
    let mut mse = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x as f64 - y as f64;
        mse += d * d;
    }
    mse /= a.len() as f64;
    let psnr = if mse <= 1e-9 {
        120.0
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    };
    let ch = a.len() / (h * w);
    let to_luma = |buf: &[u8]| -> Vec<f64> {
        (0..h * w)
            .map(|i| {
                if ch == 3 {
                    0.114 * buf[i * 3] as f64 + 0.587 * buf[i * 3 + 1] as f64 + 0.299 * buf[i * 3 + 2] as f64
                } else {
                    buf[i] as f64
                }
            })
            .collect()
    };
    let (la, lb) = (to_luma(a), to_luma(b));
    let n = (h * w) as f64;
    let ma = la.iter().sum::<f64>() / n;
    let mb = lb.iter().sum::<f64>() / n;
    let (mut va, mut vb_, mut cov) = (0f64, 0f64, 0f64);
    for i in 0..h * w {
        va += (la[i] - ma).powi(2);
        vb_ += (lb[i] - mb).powi(2);
        cov += (la[i] - ma) * (lb[i] - mb);
    }
    va /= n - 1.0;
    vb_ /= n - 1.0;
    cov /= n - 1.0;
    let c1 = (0.01 * 255.0_f64).powi(2);
    let c2 = (0.03 * 255.0_f64).powi(2);
    let ssim = ((2.0 * ma * mb + c1) * (2.0 * cov + c2))
        / ((ma * ma + mb * mb + c1) * (va + vb_ + c2));
    (psnr, ssim)
}

fn meta_arr(meta: &str, key: &str) -> Vec<i64> {
    let p = meta.find(key).unwrap();
    let rest = &meta[p + key.len()..];
    let lb = rest.find('[').unwrap();
    let rb = rest.find(']').unwrap();
    rest[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect()
}

fn meta_int(meta: &str, key: &str) -> usize {
    let p = meta.find(key).unwrap();
    let rest = &meta[p + key.len()..];
    let c = rest.find(':').unwrap();
    let end = rest[c + 1..].find([',', '}']).unwrap();
    rest[c + 1..c + 1 + end].trim().parse().unwrap()
}

/// Verify the Rust BiSeNet face-parse + feathered blend against the PyTorch reference dumps
/// (blendref/). Reports per-frame mask IoU (network parse + jaw morphology) and composite
/// PSNR/SSIM vs the proven Python `get_image` blend, plus writes the Rust composite PNGs.
fn run_face_blend() -> Result<()> {
    use musetalk::blend;
    use musetalk::face_parse::BiSeNet;

    let refdir = std::env::var("MUSETALK_BLENDREF")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/blendref".to_string());
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let outdir = std::env::var("MUSETALK_BLENDOUT")
        .unwrap_or_else(|_| format!("{refdir}/rust"));
    std::fs::create_dir_all(&outdir).ok();
    let dev = match std::env::var("MUSETALK_DEV").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let dtype = pick_dtype();
    let idxs: Vec<usize> = std::env::var("MUSETALK_BLEND_IDXS")
        .unwrap_or_else(|_| "0,5,10,20,40".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let vb = real_vb(&format!("{wdir}/bisenet.safetensors"), dtype, &dev)?;
    let net = BiSeNet::new(vb.clone())?;
    let (cone, cheek, cheek_mask) = load_morph(&vb)?;
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    println!(
        "\n==== Rust BiSeNet face-parse + blend  dev={:?} dtype={:?} vs PyTorch ====",
        dev.location(),
        dtype
    );
    println!("(eliminates the Python BiSeNet dependency; CPU morphology+blur+composite mirror cv2/PIL)\n");

    let (mut net_ious, mut full_ious) = (Vec::new(), Vec::new());
    let (mut comp_psnrs, mut comp_ssims, mut alpha_psnrs) = (Vec::new(), Vec::new(), Vec::new());

    for &i in &idxs {
        // (A) network-isolation: feed the EXACT PyTorch-normalized 512 input
        let bin = Tensor::read_npy(format!("{refdir}/bisenet_in_{i}.npy"))?
            .to_device(&dev)?
            .to_dtype(dtype)?;
        let parse_a: Vec<i64> = net.parse(&bin)?.flatten_all()?.to_vec1::<i64>()?;
        let jaw_a = blend::jaw_mask(
            &parse_a, 512, 512, &cone.0, cone.1, cone.2, &cheek.0, cheek.1, cheek.2, &cheek_mask.0,
        );
        let jaw_ref: Vec<u8> = Tensor::read_npy(format!("{refdir}/jawmask_{i}.npy"))?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let iou_net = iou_u8(&jaw_a, &jaw_ref);
        net_ious.push(iou_net);

        // (B) full path: facelarge -> resize512 -> normalize -> net -> jaw
        let fl = Tensor::read_npy(format!("{refdir}/facelarge_{i}.npy"))?; // [oh,ow,3] u8 BGR
        let (oh, ow) = (fl.dim(0)?, fl.dim(1)?);
        let fl_v: Vec<u8> = fl.flatten_all()?.to_vec1::<u8>()?;
        let mut chan = vec![vec![0u8; oh * ow]; 3]; // RGB planar
        for p in 0..oh * ow {
            chan[0][p] = fl_v[p * 3 + 2];
            chan[1][p] = fl_v[p * 3 + 1];
            chan[2][p] = fl_v[p * 3];
        }
        let mut inp = vec![0f32; 3 * 512 * 512];
        for c in 0..3 {
            let r = blend::resize_bilinear_u8(&chan[c], oh, ow, 512, 512);
            for p in 0..512 * 512 {
                inp[c * 512 * 512 + p] = (r[p] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
        let inp_t = Tensor::from_vec(inp, (1usize, 3, 512, 512), &dev)?.to_dtype(dtype)?;
        let parse_b: Vec<i64> = net.parse(&inp_t)?.flatten_all()?.to_vec1::<i64>()?;
        let jaw_b = blend::jaw_mask(
            &parse_b, 512, 512, &cone.0, cone.1, cone.2, &cheek.0, cheek.1, cheek.2, &cheek_mask.0,
        );
        full_ious.push(iou_u8(&jaw_b, &jaw_ref));

        // (C) feathered alpha + composite
        let meta = std::fs::read_to_string(format!("{refdir}/meta_{i}.json"))?;
        let bx = meta_arr(&meta, "\"box\"");
        let cb = meta_arr(&meta, "\"crop_box\"");
        let blur_k = meta_int(&meta, "\"blur_kernel\"");
        let face_box = [bx[0], bx[1], bx[2], bx[3]];
        let crop_box = [cb[0], cb[1], cb[2], cb[3]];

        let alpha = blend::feather_alpha(&jaw_b, 512, 512, oh, ow, face_box, crop_box, 0.5, blur_k);
        let alpha_ref: Vec<u8> = Tensor::read_npy(format!("{refdir}/alpha_{i}.npy"))?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let (ap, _) = psnr_ssim_u8(&alpha, &alpha_ref, oh, ow);
        alpha_psnrs.push(ap);

        let rf = Tensor::read_npy(format!("{refdir}/rf_{i}.npy"))?; // [fbh,fbw,3] u8 BGR
        let (fbh, fbw) = (rf.dim(0)?, rf.dim(1)?);
        let rf_v: Vec<u8> = rf.flatten_all()?.to_vec1::<u8>()?;

        let comp_ref = Tensor::read_npy(format!("{refdir}/composite_{i}.npy"))?; // [H,W,3] u8 BGR
        let (ch_, cw_) = (comp_ref.dim(0)?, comp_ref.dim(1)?);
        let comp_ref_v: Vec<u8> = comp_ref.flatten_all()?.to_vec1::<u8>()?;

        let body_src: Vec<u8> = Tensor::read_npy(format!("{refdir}/frame_{i}.npy"))?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let mut body = body_src.clone();
        blend::composite(&mut body, ch_, cw_, &rf_v, fbh, fbw, face_box, crop_box, &alpha, oh, ow);
        let (cp, cs) = psnr_ssim_u8(&body, &comp_ref_v, ch_, cw_);
        comp_psnrs.push(cp);
        comp_ssims.push(cs);

        // write Rust composite (BGR HxWx3) as npy for offline PNG/visual diff
        Tensor::from_vec(body, (ch_, cw_, 3), &Device::Cpu)?
            .write_npy(format!("{outdir}/composite_{i}.npy"))?;
        Tensor::from_vec(jaw_b.clone(), (512usize, 512), &Device::Cpu)?
            .write_npy(format!("{outdir}/jawmask_{i}.npy"))?;

        println!(
            "frame {i:>3}: mask-IoU net={:.5} full={:.5} | alpha PSNR {:6.2} dB | composite PSNR {:6.2} dB  SSIM {:.5}",
            iou_net, full_ious.last().unwrap(), ap, cp, cs
        );
    }

    let mean_f = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!("\n---- summary (mean over {} frames) ----", idxs.len());
    println!("mask IoU (network-isolated input): {:.5}", mean_f(&net_ious));
    println!("mask IoU (full Rust resize+net):   {:.5}", mean_f(&full_ious));
    println!("feathered alpha PSNR:              {:6.2} dB", mean_f(&alpha_psnrs));
    println!("composite PSNR:                    {:6.2} dB", mean_f(&comp_psnrs));
    println!("composite SSIM:                    {:.5}", mean_f(&comp_ssims));
    println!("(mask IoU>=0.99 and composite PSNR>=35 dB / SSIM>=0.99 = blend matches Python)");
    Ok(())
}
