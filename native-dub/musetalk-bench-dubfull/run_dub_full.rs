fn run_dub_full() -> Result<()> {
    use musetalk::blend;
    use musetalk::face_parse::BiSeNet;

    let frames_dir = std::env::var("MUSETALK_FRAMES").expect("set MUSETALK_FRAMES");
    let audio = std::env::var("MUSETALK_AUDIO").expect("set MUSETALK_AUDIO");
    let outframes = std::env::var("MUSETALK_OUTFRAMES").expect("set MUSETALK_OUTFRAMES");
    let wdir = std::env::var("MUSETALK_WDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/rustweights".to_string());
    let facedir = std::env::var("MUSETALK_FACEDIR")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/facedump".to_string());
    let fps: usize = std::env::var("MUSETALK_FPS").ok().and_then(|s| s.parse().ok()).unwrap_or(25);
    let bsz: usize = std::env::var("MUSETALK_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let extra_margin: i64 = std::env::var("MUSETALK_EXTRA_MARGIN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let dev = match std::env::var("MUSETALK_DEV").as_deref() {
        Ok("cuda") => Device::new_cuda(0)?,
        _ => Device::Cpu,
    };
    let dtype = pick_dtype();
    std::fs::create_dir_all(&outframes).ok();

    // ---- load all models (MuseTalk, whisper feats, SFD+FAN, BiSeNet) ----
    let cfg = MuseTalkConfig::default();
    let sz = cfg.resized_img; // 256
    let model = MuseTalk::new(
        cfg,
        real_vb(&format!("{wdir}/vae.safetensors"), dtype, &dev)?,
        real_vb(&format!("{wdir}/unet.safetensors"), dtype, &dev)?,
        &dev,
        dtype,
    )?;
    let fx = build_audio_feats(&wdir, &dev, dtype)?;
    let s3fd_vb = unsafe {
        hanzo_nn::VarBuilder::from_mmaped_safetensors(
            &[std::path::PathBuf::from(format!("{facedir}/s3fd.safetensors"))], DType::F32, &dev)?
    };
    let fan_vb = unsafe {
        hanzo_nn::VarBuilder::from_mmaped_safetensors(
            &[std::path::PathBuf::from(format!("{facedir}/fan2d.safetensors"))], DType::F32, &dev)?
    };
    let fa = face_detection::FaceAlignment::new(s3fd_vb, fan_vb, dev.clone())?;
    let bvb = real_vb(&format!("{wdir}/bisenet.safetensors"), dtype, &dev)?;
    let bisenet = BiSeNet::new(bvb.clone())?;
    let (cone, cheek, cheek_mask) = load_morph(&bvb)?;
    let imnet_mean = [0.485f32, 0.456, 0.406];
    let imnet_std = [0.229f32, 0.224, 0.225];

    // ---- load source frames (BGR u8, like cv2.imread) ----
    let mut paths: Vec<std::path::PathBuf> = std::fs::read_dir(&frames_dir)
        .map_err(hanzo_ml::Error::wrap)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    paths.sort();
    let mut frames: Vec<(Vec<u8>, usize, usize)> = Vec::with_capacity(paths.len());
    for p in &paths {
        let img = image::open(p).map_err(|e| hanzo_ml::Error::msg(format!("open {p:?}: {e}")))?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = img.into_raw(); // HWC RGB u8
        let mut bgr = vec![0u8; rgb.len()];
        for i in 0..w * h {
            bgr[i * 3] = rgb[i * 3 + 2];
            bgr[i * 3 + 1] = rgb[i * 3 + 1];
            bgr[i * 3 + 2] = rgb[i * 3];
        }
        frames.push((bgr, h, w));
    }
    let nframes_src = frames.len();
    if nframes_src == 0 {
        hanzo_ml::bail!("no source frames in {frames_dir}");
    }
    println!("==== FULL-NATIVE dub  dev={:?} dtype={:?} frames={} fps={} ====", dev.location(), dtype, nframes_src, fps);

    // ---- per-frame face bbox (SFD+FAN), with last-good fallback (mirrors python) ----
    println!("[native] SFD+FAN bboxes...");
    let mut coords: Vec<(i64, i64, i64, i64)> = Vec::with_capacity(nframes_src);
    let mut last: Option<(i64, i64, i64, i64)> = None;
    for (bgr, h, w) in &frames {
        let mut rgb = vec![0u8; bgr.len()];
        for i in 0..w * h {
            rgb[i * 3] = bgr[i * 3 + 2];
            rgb[i * 3 + 1] = bgr[i * 3 + 1];
            rgb[i * 3 + 2] = bgr[i * 3];
        }
        let bb = match fa.get_landmarks(&rgb, *h, *w)? {
            Some((lm, _sfd)) => face_detection::musetalk_bbox(&lm, *h),
            None => None,
        };
        let cur = bb.or(last).unwrap_or((0, 0, 0, 0));
        last = Some(cur);
        coords.push(cur);
    }

    // ---- audio -> per-frame post-PE whisper features [1,50,384] ----
    println!("[native] whisper-tiny features...");
    let pcm = read_wav_16k_mono(&audio)?;
    let chunks = fx.whisper_chunks(&pcm, fps, 2, 2)?; // [video_num,50,384]
    let video_num = chunks.dim(0)?;
    println!("[native] {video_num} audio frames ({:.2}s)", pcm.len() as f64 / 16000.0);

    // cycle frames+coords (forward then reverse) to cover audio length (mirrors python)
    let cyc_len = nframes_src * 2;
    let cyc = |i: usize| -> usize {
        let j = i % cyc_len;
        if j < nframes_src { j } else { cyc_len - 1 - j }
    };

    let _transform_mean = 0.5f32; // MuseTalk face normalize mean=std=0.5

    // normalized model-input crop [1,3,256,256] from a frame+box (matches python norm_crop)
    let norm_crop = |bgr: &[u8], h: usize, w: usize, box_: (i64, i64, i64, i64)| -> Result<Tensor> {
        let (x1, y1, x2, y2) = box_;
        let y2m = ((y2 + extra_margin) as usize).min(h);
        let (x1u, y1u, x2u) = (x1.max(0) as usize, y1.max(0) as usize, (x2 as usize).min(w));
        let (cw, ch) = (x2u.saturating_sub(x1u).max(1), y2m.saturating_sub(y1u).max(1));
        // crop region (BGR) -> planar RGB u8 -> resize 256 -> normalize
        let mut chans = vec![vec![0u8; cw * ch]; 3]; // R,G,B planar
        for yy in 0..ch {
            for xx in 0..cw {
                let sp = ((y1u + yy) * w + (x1u + xx)) * 3;
                let dp = yy * cw + xx;
                chans[0][dp] = bgr[sp + 2];
                chans[1][dp] = bgr[sp + 1];
                chans[2][dp] = bgr[sp];
            }
        }
        // forward_batched normalizes internally (mean=std=0.5), so feed RAW [0,1] here.
        let mut data = vec![0f32; 3 * sz * sz];
        for c in 0..3 {
            let r = blend::resize_bilinear_u8(&chans[c], ch, cw, sz, sz);
            for p in 0..sz * sz {
                data[c * sz * sz + p] = r[p] as f32 / 255.0;
            }
        }
        Tensor::from_vec(data, (1usize, 3, sz, sz), &dev)
    };

    // ---- main render loop, batched ----
    println!("[native] MuseTalk + BiSeNet blend...");
    let t0 = Instant::now();
    let mut done = 0usize;
    while done < video_num {
        let n = bsz.min(video_num - done);
        let mut faces = Vec::with_capacity(n);
        let mut auds = Vec::with_capacity(n);
        for k in 0..n {
            let i = done + k;
            let fi = cyc(i);
            let (bgr, h, w) = &frames[fi];
            faces.push(norm_crop(bgr, *h, *w, coords[fi])?);
            let chunk = chunks.narrow(0, i, 1)?; // [1,50,384]
            auds.push(fx.positional_encoding(&chunk)?);
        }
        let faces_t = Tensor::cat(&faces, 0)?.to_dtype(dtype)?;
        let auds_t = Tensor::cat(&auds, 0)?.to_dtype(dtype)?;
        let mouths = model.forward_batched(&faces_t, &auds_t)?; // [n,3,256,256] RGB [0,1]
        let mouths = mouths.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        for k in 0..n {
            let i = done + k;
            let fi = cyc(i);
            let (bgr, h, w) = &frames[fi];
            let (x1, y1, x2, y2) = coords[fi];
            let y2m = ((y2 + extra_margin).min(*h as i64)) as i64;
            let face_box = [x1, y1, x2, y2m];
            let (bw, bh) = ((x2 - x1).max(1) as usize, (y2m - y1).max(1) as usize);

            // mouth tensor [3,256,256] RGB[0,1] -> BGR u8 [bh,bw,3] (resize to face box)
            let m = mouths.narrow(0, k, 1)?.squeeze(0)?; // [3,256,256]
            let mv: Vec<f32> = m.flatten_all()?.to_vec1::<f32>()?;
            let mut mouth_rgb_planar = vec![vec![0u8; sz * sz]; 3];
            for c in 0..3 {
                for p in 0..sz * sz {
                    mouth_rgb_planar[c][p] = (mv[c * sz * sz + p].clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
            // resize each plane to (bh,bw), pack BGR HWC
            let rp: Vec<Vec<u8>> = (0..3).map(|c| blend::resize_bilinear_u8(&mouth_rgb_planar[c], sz, sz, bh, bw)).collect();
            let mut face_bgr = vec![0u8; bh * bw * 3];
            for p in 0..bh * bw {
                face_bgr[p * 3] = rp[2][p];     // B
                face_bgr[p * 3 + 1] = rp[1][p]; // G
                face_bgr[p * 3 + 2] = rp[0][p]; // R
            }

            // crop_box + face_large geometry (expand=1.5)
            let (crop_box_arr, s) = blend::get_crop_box(face_box, 1.5);
            let oh = (crop_box_arr[3] - crop_box_arr[1]).max(1) as usize;
            let ow = (crop_box_arr[2] - crop_box_arr[0]).max(1) as usize;
            let _ = s;

            // BiSeNet parse of the face crop (resize face box region of frame to 512, imagenet norm)
            // python parses face_large region; we parse the same crop_box sub-image of the frame.
            let mut crop_chan = vec![vec![0u8; oh * ow]; 3]; // RGB planar from frame crop_box
            for yy in 0..oh {
                for xx in 0..ow {
                    let sy = crop_box_arr[1] + yy as i64;
                    let sx = crop_box_arr[0] + xx as i64;
                    let dp = yy * ow + xx;
                    if sy >= 0 && sy < *h as i64 && sx >= 0 && sx < *w as i64 {
                        let sp = (sy as usize * *w + sx as usize) * 3;
                        crop_chan[0][dp] = bgr[sp + 2];
                        crop_chan[1][dp] = bgr[sp + 1];
                        crop_chan[2][dp] = bgr[sp];
                    }
                }
            }
            let mut binp = vec![0f32; 3 * 512 * 512];
            for c in 0..3 {
                let r = blend::resize_bilinear_u8(&crop_chan[c], oh, ow, 512, 512);
                for p in 0..512 * 512 {
                    binp[c * 512 * 512 + p] = (r[p] as f32 / 255.0 - imnet_mean[c]) / imnet_std[c];
                }
            }
            let binp_t = Tensor::from_vec(binp, (1usize, 3, 512, 512), &dev)?.to_dtype(dtype)?;
            let parse: Vec<i64> = bisenet.parse(&binp_t)?.flatten_all()?.to_vec1::<i64>()?;
            let jaw = blend::jaw_mask(&parse, 512, 512, &cone.0, cone.1, cone.2, &cheek.0, cheek.1, cheek.2, &cheek_mask.0);

            let blur_k = ((0.05 * oh as f64) as usize / 2) * 2 + 1;
            let alpha = blend::feather_alpha(&jaw, 512, 512, oh, ow, face_box, crop_box_arr, 0.5, blur_k);

            // composite mouth onto a copy of the original frame
            let mut body = bgr.clone();
            blend::composite(&mut body, *h, *w, &face_bgr, bh, bw, face_box, crop_box_arr, &alpha, oh, ow);

            // body is BGR -> write PNG as RGB
            let mut out_rgb = vec![0u8; body.len()];
            for p in 0..*h * *w {
                out_rgb[p * 3] = body[p * 3 + 2];
                out_rgb[p * 3 + 1] = body[p * 3 + 1];
                out_rgb[p * 3 + 2] = body[p * 3];
            }
            let imgbuf = image::RgbImage::from_raw(*w as u32, *h as u32, out_rgb)
                .ok_or_else(|| hanzo_ml::Error::msg("RgbImage::from_raw failed"))?;
            imgbuf.save(format!("{outframes}/{i:08}.png"))
                .map_err(|e| hanzo_ml::Error::msg(format!("save frame {i}: {e}")))?;
        }
        done += n;
        if done % 32 == 0 || done == video_num {
            println!("  {done}/{video_num}");
        }
    }
    dev.synchronize()?;
    let dt = t0.elapsed().as_secs_f64();
    println!("[native] DONE: {video_num} frames in {dt:.2}s ({:.2} fps)", video_num as f64 / dt);
    Ok(())
}
