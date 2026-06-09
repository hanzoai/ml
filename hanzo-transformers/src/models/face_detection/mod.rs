//! Face detection + 68-landmark localization, a Rust port of the
//! `face_alignment` (SFD detector + 2D-FAN) pipeline used by MuseTalk to derive
//! the nose-centered lip-sync crop box.
//!
//! Two stages:
//!   1. [`s3fd::S3fd`] — VGG16 single-shot face detector -> face box.
//!   2. [`fan::Fan`]   — 4-stack hourglass -> 68 landmark heatmaps -> points.
//!
//! Then [`musetalk_bbox`] reproduces MuseTalk's box: symmetric around the nose
//! (landmark 29), chin (max-y) at the bottom.
//!
//! This mirrors `face_alignment/api.py::get_landmarks_from_image` and
//! `face_alignment/utils.py` (crop/transform/get_preds_fromhm) so the resulting
//! bbox matches the Python reference closely (IoU comparison in the binary).

pub mod fan;
pub mod s3fd;

pub use s3fd::Bbox;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_nn::VarBuilder;

// Constants from face_alignment/utils.py
const REFERENCE_SCALE: f32 = 195.0;
const SCALE_FACTOR: f32 = 200.0;
const CENTER_Y_OFFSET: f32 = 0.12;
const CROP_RESOLUTION: usize = 256;
const NUM_LANDMARKS: usize = 68;

/// The complete face-alignment model: detector + landmark net.
pub struct FaceAlignment {
    pub detector: s3fd::S3fd,
    pub fan: fan::Fan,
    device: Device,
}

impl FaceAlignment {
    /// Load from two safetensors VarBuilders. `num_modules` is 4 for 2DFAN-4.
    pub fn new(s3fd_vb: VarBuilder, fan_vb: VarBuilder, device: Device) -> Result<Self> {
        let detector = s3fd::S3fd::new(s3fd_vb)?;
        let fan = fan::Fan::new(4, fan_vb)?;
        Ok(Self {
            detector,
            fan,
            device,
        })
    }

    /// Detect faces and return their boxes (no landmark stage).
    pub fn detect_faces(&self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<Bbox>> {
        s3fd::detect(&self.detector, rgb, h, w, 0.5, &self.device)
    }

    /// Full pipeline: detect first face, predict its 68 landmarks in image coords.
    /// Returns `(landmarks[68][2], face_box)` or `None` if no face.
    pub fn get_landmarks(
        &self,
        rgb: &[u8],
        h: usize,
        w: usize,
    ) -> Result<Option<(Vec<[f32; 2]>, Bbox)>> {
        let faces = self.detect_faces(rgb, h, w)?;
        if faces.is_empty() {
            return Ok(None);
        }
        let d = faces[0];
        // center / scale (api.py)
        let cx = d.x2 - (d.x2 - d.x1) / 2.0;
        let mut cy = d.y2 - (d.y2 - d.y1) / 2.0;
        cy -= (d.y2 - d.y1) * CENTER_Y_OFFSET;
        let center = [cx, cy];
        let scale = (d.x2 - d.x1 + d.y2 - d.y1) / REFERENCE_SCALE;

        // crop -> [256,256,3] u8 RGB, then /255, CHW, [1,3,256,256]
        let crop = crop_image(rgb, h, w, center, scale, CROP_RESOLUTION);
        let inp = Tensor::from_vec(
            crop.iter().map(|&v| v as f32 / 255.0).collect::<Vec<f32>>(),
            (CROP_RESOLUTION, CROP_RESOLUTION, 3),
            &self.device,
        )?
        .permute((2, 0, 1))?
        .unsqueeze(0)?;

        // FAN forward (flip_input=False in the dub) -> [1,68,64,64]
        let out = self.fan.forward(&inp)?;
        let out = out.to_dtype(hanzo_ml::DType::F32)?;
        let hm: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let (_b, c, hh, ww) = out.dims4()?;
        let lm = get_preds_fromhm(&hm, c, hh, ww, center, scale);
        Ok(Some((lm, d)))
    }
}

/// Affine transform identical to utils.py::transform with `.int()` truncation.
/// Returns integer (x, y).
fn transform_int(point: [f32; 2], center: [f32; 2], scale: f32, resolution: f32, invert: bool) -> (i64, i64) {
    let h = SCALE_FACTOR * scale;
    // Forward matrix t:
    //   [ res/h     0      res*(-cx/h + 0.5) ]
    //   [ 0       res/h    res*(-cy/h + 0.5) ]
    //   [ 0         0              1         ]
    let a = resolution / h;
    let tx = resolution * (-center[0] / h + 0.5);
    let ty = resolution * (-center[1] / h + 0.5);
    if !invert {
        let nx = a * point[0] + tx;
        let ny = a * point[1] + ty;
        (nx as i64, ny as i64) // .int() truncates toward zero
    } else {
        // inverse of an affine scale+translate: x = (p - t)/a
        let nx = (point[0] - tx) / a;
        let ny = (point[1] - ty) / a;
        (nx as i64, ny as i64)
    }
}

/// cv2-compatible bilinear resize (INTER_LINEAR) from a src RGB buffer to
/// `(dst_h, dst_w)`. cv2 maps dst->src as `s = (d + 0.5)*scale - 0.5`.
fn resize_bilinear(
    src: &[u8],
    sh: usize,
    sw: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; dst_h * dst_w * 3];
    if sh == 0 || sw == 0 {
        return out;
    }
    let scale_y = sh as f32 / dst_h as f32;
    let scale_x = sw as f32 / dst_w as f32;
    for dy in 0..dst_h {
        let fy = (dy as f32 + 0.5) * scale_y - 0.5;
        let sy0 = fy.floor();
        let wy1 = fy - sy0;
        let wy0 = 1.0 - wy1;
        let y0 = sy0 as i64;
        let y0c = y0.clamp(0, sh as i64 - 1) as usize;
        let y1c = (y0 + 1).clamp(0, sh as i64 - 1) as usize;
        for dx in 0..dst_w {
            let fx = (dx as f32 + 0.5) * scale_x - 0.5;
            let sx0 = fx.floor();
            let wx1 = fx - sx0;
            let wx0 = 1.0 - wx1;
            let x0 = sx0 as i64;
            let x0c = x0.clamp(0, sw as i64 - 1) as usize;
            let x1c = (x0 + 1).clamp(0, sw as i64 - 1) as usize;
            for ch in 0..3 {
                let p00 = src[(y0c * sw + x0c) * 3 + ch] as f32;
                let p01 = src[(y0c * sw + x1c) * 3 + ch] as f32;
                let p10 = src[(y1c * sw + x0c) * 3 + ch] as f32;
                let p11 = src[(y1c * sw + x1c) * 3 + ch] as f32;
                let top = p00 * wx0 + p01 * wx1;
                let bot = p10 * wx0 + p11 * wx1;
                let v = top * wy0 + bot * wy1;
                out[(dy * dst_w + dx) * 3 + ch] = (v + 0.5).floor().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Port of utils.py::crop. Extracts a center/scale crop with zero padding, then
/// bilinearly resizes to `resolution`. Input/output are RGB HWC u8.
fn crop_image(
    img: &[u8],
    ht: usize,
    wd: usize,
    center: [f32; 2],
    scale: f32,
    resolution: usize,
) -> Vec<u8> {
    let res_f = resolution as f32;
    let ul = transform_int([1.0, 1.0], center, scale, res_f, true);
    let br = transform_int([res_f, res_f], center, scale, res_f, true);
    let new_w = (br.0 - ul.0) as i64;
    let new_h = (br.1 - ul.1) as i64;
    if new_w <= 0 || new_h <= 0 {
        return vec![0u8; resolution * resolution * 3];
    }
    let new_w = new_w as usize;
    let new_h = new_h as usize;
    let mut new_img = vec![0u8; new_h * new_w * 3];

    // index ranges (1-based in python; here we compute 0-based slices)
    // newX = [max(1, -ul0+1), min(br0, wd) - ul0]   (1-based, inclusive)
    // oldX = [max(1, ul0+1),  min(br0, wd)]
    let newx0 = (1.max(-ul.0 + 1)) as i64;
    let newx1 = (br.0.min(wd as i64) - ul.0) as i64;
    let newy0 = (1.max(-ul.1 + 1)) as i64;
    let newy1 = (br.1.min(ht as i64) - ul.1) as i64;
    let oldx0 = (1.max(ul.0 + 1)) as i64;
    let oldx1 = (br.0.min(wd as i64)) as i64;
    let oldy0 = (1.max(ul.1 + 1)) as i64;
    let oldy1 = (br.1.min(ht as i64)) as i64;

    // new_img[newY0-1:newY1, newX0-1:newX1] = img[oldY0-1:oldY1, oldX0-1:oldX1]
    let copy_h = (newy1 - (newy0 - 1)).min(oldy1 - (oldy0 - 1));
    let copy_w = (newx1 - (newx0 - 1)).min(oldx1 - (oldx0 - 1));
    if copy_h > 0 && copy_w > 0 {
        for r in 0..copy_h {
            let ny = (newy0 - 1 + r) as usize;
            let oy = (oldy0 - 1 + r) as usize;
            for cc in 0..copy_w {
                let nx = (newx0 - 1 + cc) as usize;
                let ox = (oldx0 - 1 + cc) as usize;
                if ny < new_h && nx < new_w && oy < ht && ox < wd {
                    for ch in 0..3 {
                        new_img[(ny * new_w + nx) * 3 + ch] = img[(oy * wd + ox) * 3 + ch];
                    }
                }
            }
        }
    }
    resize_bilinear(&new_img, new_h, new_w, resolution, resolution)
}

/// Port of utils.py::get_preds_fromhm (+_get_preds_fromhm). hm is flat
/// `[C*H*W]`. Returns 68 landmark points in original image coords.
fn get_preds_fromhm(
    hm: &[f32],
    c: usize,
    h: usize,
    w: usize,
    center: [f32; 2],
    scale: f32,
) -> Vec<[f32; 2]> {
    let hw = h * w;
    let mut preds = vec![[0f32; 2]; c];
    for j in 0..c {
        // argmax over HxW
        let base = j * hw;
        let mut best = f32::NEG_INFINITY;
        let mut bi = 0usize;
        for k in 0..hw {
            let v = hm[base + k];
            if v > best {
                best = v;
                bi = k;
            }
        }
        // idx (0-based) -> python idx is +1; preds x = (idx-1)%W +1 ; y = floor((idx-1)/H)+1
        // With 0-based bi: px0 = bi % W ; py0 = bi / W  (== python px-1, py-1)
        let mut px = (bi % w) as f32 + 1.0; // python 1-based px
        let mut py = (bi / w) as f32 + 1.0; // python 1-based py
        // subpixel: use 0-based neighbor lookups
        let pxi = px as i64 - 1; // 0-based
        let pyi = py as i64 - 1;
        if pxi > 0 && pxi < (w as i64 - 1) && pyi > 0 && pyi < (h as i64 - 1) {
            let pxi = pxi as usize;
            let pyi = pyi as usize;
            let dx = hm[base + pyi * w + (pxi + 1)] - hm[base + pyi * w + (pxi - 1)];
            let dy = hm[base + (pyi + 1) * w + pxi] - hm[base + (pyi - 1) * w + pxi];
            px += dx.signum() * 0.25;
            py += dy.signum() * 0.25;
        }
        px -= 0.5;
        py -= 0.5;
        // transform back to original coords (invert=True, resolution=H=64)
        let (ox, oy) = transform_int([px, py], center, scale, h as f32, true);
        preds[j] = [ox as f32, oy as f32];
    }
    let _ = NUM_LANDMARKS;
    preds
}

/// MuseTalk bbox derivation from 68 iBUG landmarks, identical to musetalk_dub.py.
/// Returns `(x1, y1, x2, y2)` in pixel coords, or `None` if degenerate.
pub fn musetalk_bbox(lm: &[[f32; 2]], frame_h: usize) -> Option<(i64, i64, i64, i64)> {
    if lm.len() < 68 {
        return None;
    }
    let to_i = |f: f32| f as i64; // np.int32 truncation toward zero
    let half_y = to_i(lm[29][1]);
    let chin_y = lm.iter().map(|p| to_i(p[1])).max().unwrap();
    let half_dist = chin_y - half_y;
    let upper = (half_y - half_dist).max(0);
    let x1 = lm.iter().map(|p| to_i(p[0])).min().unwrap();
    let x2 = lm.iter().map(|p| to_i(p[0])).max().unwrap();
    let y1 = upper;
    let y2 = chin_y;
    let _ = frame_h;
    if x2 - x1 <= 0 || y2 - y1 <= 0 || x1 < 0 {
        None
    } else {
        Some((x1, y1, x2, y2))
    }
}

/// IoU between two integer boxes `(x1,y1,x2,y2)`.
pub fn iou(a: (i64, i64, i64, i64), b: (i64, i64, i64, i64)) -> f64 {
    let ix1 = a.0.max(b.0);
    let iy1 = a.1.max(b.1);
    let ix2 = a.2.min(b.2);
    let iy2 = a.3.min(b.3);
    let iw = (ix2 - ix1).max(0);
    let ih = (iy2 - iy1).max(0);
    let inter = (iw * ih) as f64;
    let area_a = ((a.2 - a.0).max(0) * (a.3 - a.1).max(0)) as f64;
    let area_b = ((b.2 - b.0).max(0) * (b.3 - b.1).max(0)) as f64;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}
