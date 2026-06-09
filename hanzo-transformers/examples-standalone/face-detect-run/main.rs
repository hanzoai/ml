// Standalone runner that exercises the real face_detection module
// (hanzo-transformers/src/models/face_detection/*) without compiling the rest
// of hanzo-transformers (which has unrelated pre-existing build breakage at this
// commit). Detects the MuseTalk bbox per frame and compares to the Python
// face_alignment reference (IoU).

#[path = "../../../hanzo-transformers/src/models/face_detection/mod.rs"]
mod face_detection;

use anyhow::{Context, Result};
use hanzo_ml::{DType, Device};
use hanzo_nn::VarBuilder;
use serde::Deserialize;
use std::path::PathBuf;

use face_detection::{iou, musetalk_bbox, FaceAlignment};

#[derive(Deserialize)]
struct RefRecord {
    frame: usize,
    ok: bool,
    sfd_box: Option<Vec<f64>>,
    musetalk_bbox: Vec<i64>,
}
#[derive(Deserialize)]
struct RefFile {
    w: usize,
    h: usize,
    n: usize,
    records: Vec<RefRecord>,
}

fn main() -> Result<()> {
    let dump = std::env::var("DUMP")
        .unwrap_or_else(|_| "/home/z/work/zen-dub-run/facedump".to_string());
    let dump = PathBuf::from(dump);
    let frames_dir = dump.join("frames"); // already extracted by the python ref step
    let s3fd_st = dump.join("s3fd.safetensors");
    let fan_st = dump.join("fan2d.safetensors");
    let ref_json = dump.join("ref_face.json");

    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0).context("cuda device")?
    } else {
        Device::Cpu
    };
    println!("device = {:?}", device);

    let refdata: RefFile =
        serde_json::from_reader(std::fs::File::open(&ref_json).context("open ref json")?)?;
    println!(
        "reference: {}x{} n={} records={}",
        refdata.w,
        refdata.h,
        refdata.n,
        refdata.records.len()
    );

    let s3fd_vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[s3fd_st], DType::F32, &device)? };
    let fan_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[fan_st], DType::F32, &device)? };
    let fa = FaceAlignment::new(s3fd_vb, fan_vb, device.clone())?;
    println!("loaded S3FD + 2DFAN-4");

    // Which frames to evaluate. Default: a representative spread; FULL=1 -> all.
    let full = std::env::var("FULL").map(|v| v == "1").unwrap_or(false);
    let sample: Vec<usize> = if full {
        (0..refdata.records.len()).collect()
    } else {
        // first 5, then every 50th, plus last
        let mut v: Vec<usize> = (0..5).collect();
        let mut k = 25;
        while k < refdata.records.len() {
            v.push(k);
            k += 25;
        }
        v.push(refdata.records.len() - 1);
        v.dedup();
        v
    };

    let mut ious = Vec::new();
    let mut max_coord_err = 0i64;
    let mut sum_coord_err = 0i64;
    let mut n_coord = 0i64;
    let mut detected = 0usize;
    let mut sfd_iou_sum = 0.0f64;
    let mut sfd_n = 0usize;

    println!("\nframe |  rust bbox (x1,y1,x2,y2)  |  py bbox                 |  IoU   | max|dcoord|");
    println!("------+---------------------------+--------------------------+--------+-----------");
    for &idx in &sample {
        let rec = &refdata.records[idx];
        debug_assert_eq!(rec.frame, idx, "frame index mismatch");
        let path = frames_dir.join(format!("{idx:08}.png"));
        let img = image::open(&path)
            .with_context(|| format!("open frame {}", path.display()))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = img.into_raw(); // HWC RGB u8

        let res = fa.get_landmarks(&rgb, h, w)?;
        let (rust_bbox, rust_sfd) = match &res {
            Some((lm, sfd)) => {
                let bb = musetalk_bbox(lm, h);
                (bb, Some(*sfd))
            }
            None => (None, None),
        };

        // Compare SFD raw box (sanity for stage 1)
        if let (Some(rs), Some(ps)) = (&rust_sfd, &rec.sfd_box) {
            let pb = (
                ps[0] as i64,
                ps[1] as i64,
                ps[2] as i64,
                ps[3] as i64,
            );
            let rb = (rs.x1 as i64, rs.y1 as i64, rs.x2 as i64, rs.y2 as i64);
            sfd_iou_sum += iou(rb, pb);
            sfd_n += 1;
        }

        match rust_bbox {
            Some(rb) => {
                detected += 1;
                let pb = (
                    rec.musetalk_bbox[0],
                    rec.musetalk_bbox[1],
                    rec.musetalk_bbox[2],
                    rec.musetalk_bbox[3],
                );
                let i = iou(rb, pb);
                ious.push(i);
                let d = [
                    (rb.0 - pb.0).abs(),
                    (rb.1 - pb.1).abs(),
                    (rb.2 - pb.2).abs(),
                    (rb.3 - pb.3).abs(),
                ];
                let dmax = *d.iter().max().unwrap();
                max_coord_err = max_coord_err.max(dmax);
                for &dd in &d {
                    sum_coord_err += dd;
                    n_coord += 1;
                }
                println!(
                    "{:5} | ({:4},{:4},{:4},{:4})       | ({:4},{:4},{:4},{:4})      | {:.4} | {}",
                    idx, rb.0, rb.1, rb.2, rb.3, pb.0, pb.1, pb.2, pb.3, i, dmax
                );
            }
            None => {
                println!(
                    "{:5} | NO FACE (rust)            | ({:4},{:4},{:4},{:4}) ok={} |   --   |  --",
                    idx,
                    rec.musetalk_bbox[0],
                    rec.musetalk_bbox[1],
                    rec.musetalk_bbox[2],
                    rec.musetalk_bbox[3],
                    rec.ok
                );
            }
        }
    }

    let n = ious.len() as f64;
    let mean_iou = if n > 0.0 { ious.iter().sum::<f64>() / n } else { 0.0 };
    let min_iou = ious.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean_coord = if n_coord > 0 {
        sum_coord_err as f64 / n_coord as f64
    } else {
        0.0
    };
    let pct_ge_095 = if n > 0.0 {
        100.0 * ious.iter().filter(|&&v| v >= 0.95).count() as f64 / n
    } else {
        0.0
    };
    println!("\n==================== SUMMARY ====================");
    println!("frames evaluated : {}", sample.len());
    println!("faces detected   : {}", detected);
    println!(
        "SFD-box mean IoU : {:.4}  (over {} frames, stage-1 sanity)",
        if sfd_n > 0 { sfd_iou_sum / sfd_n as f64 } else { 0.0 },
        sfd_n
    );
    println!("MuseTalk bbox    : mean IoU = {:.4}", mean_iou);
    println!("                   min  IoU = {:.4}", min_iou);
    println!("                   IoU>=0.95: {:.1}%", pct_ge_095);
    println!("                   mean |coord err| = {:.3} px", mean_coord);
    println!("                   max  |coord err| = {} px", max_coord_err);
    println!("================================================");
    Ok(())
}
