//! MuseTalk `get_image` feathered face-paste, ported to Rust (CPU image ops).
//!
//! Pipeline (verbatim from `musetalk/utils/blending.py::get_image`, mode="jaw"):
//!   1. crop_box = expand(face_box, 1.5) ; face_large = body.crop(crop_box)
//!   2. jaw mask  = BiSeNet(face_large@512) argmax -> jaw morphology  (face_parse.rs + this)
//!   3. mask resized to face_large size, cropped to the face_box sub-rect, pasted into a
//!      black canvas, top `upper_boundary_ratio` zeroed, GaussianBlur(k = 0.05*W rounded odd)
//!   4. face_large.paste(rendered_face) ; body.paste(face_large, crop_box, mask=alpha)
//!
//! Every op mirrors cv2 / PIL semantics so the composite matches the proven Python blend.
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::too_many_arguments)]

// ---- cv2-compatible binary morphology -------------------------------------

/// Binary dilation: out[y,x] = max over kernel hits (anchor = kernel center).
/// cv2 default border for dilate is an implicit -inf (constant 0 here, since input is {0,255}
/// and 0 is the identity for max), so out-of-bounds contributes nothing. Matches cv2.dilate.
pub fn dilate(src: &[u8], h: usize, w: usize, k: &[u8], kh: usize, kw: usize) -> Vec<u8> {
    let ay = kh / 2;
    let ax = kw / 2;
    let mut out = vec![0u8; h * w];
    // precompute kernel offsets that are set
    let mut offs = Vec::new();
    for ky in 0..kh {
        for kx in 0..kw {
            if k[ky * kw + kx] != 0 {
                offs.push((ky as isize - ay as isize, kx as isize - ax as isize));
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut v = 0u8;
            for &(dy, dx) in &offs {
                let sy = y as isize + dy;
                let sx = x as isize + dx;
                if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                    let s = src[sy as usize * w + sx as usize];
                    if s > v {
                        v = s;
                    }
                }
            }
            out[y * w + x] = v;
        }
    }
    out
}

/// Binary erosion: out[y,x] = min over kernel hits. cv2 erode default border replicates a
/// max value (255) at out-of-bounds so the image border is NOT eroded; we replicate that.
pub fn erode(src: &[u8], h: usize, w: usize, k: &[u8], kh: usize, kw: usize) -> Vec<u8> {
    let ay = kh / 2;
    let ax = kw / 2;
    let mut out = vec![0u8; h * w];
    let mut offs = Vec::new();
    for ky in 0..kh {
        for kx in 0..kw {
            if k[ky * kw + kx] != 0 {
                offs.push((ky as isize - ay as isize, kx as isize - ax as isize));
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut v = 255u8;
            for &(dy, dx) in &offs {
                let sy = y as isize + dy;
                let sx = x as isize + dx;
                let s = if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                    src[sy as usize * w + sx as usize]
                } else {
                    255 // cv2 BORDER replicate-max for erode
                };
                if s < v {
                    v = s;
                }
            }
            out[y * w + x] = v;
        }
    }
    out
}

pub fn erode_iter(src: &[u8], h: usize, w: usize, k: &[u8], kh: usize, kw: usize, iters: usize) -> Vec<u8> {
    let mut cur = src.to_vec();
    for _ in 0..iters {
        cur = erode(&cur, h, w, k, kh, kw);
    }
    cur
}

/// jaw-mode mask assembly on the 512x512 argmax parse map (FaceParsing.__call__ mode="jaw").
///   face_region = (parsing==1)*255
///   dil = dilate(face_region, cone_kernel)
///   ero = erode(dil, cheek_kernel, iters=2)
///   face_region = (ero & cheek_mask) | (dil & ~cheek_mask)
///   out = 255 where (face_region==255 & parsing!=10) OR parsing in {11,12,13}, else 0
pub fn jaw_mask(
    parsing: &[i64],
    h: usize,
    w: usize,
    cone_k: &[u8],
    cone_kh: usize,
    cone_kw: usize,
    cheek_k: &[u8],
    cheek_kh: usize,
    cheek_kw: usize,
    cheek_mask: &[u8],
) -> Vec<u8> {
    let n = h * w;
    let mut face_region = vec![0u8; n];
    for i in 0..n {
        if parsing[i] == 1 {
            face_region[i] = 255;
        }
    }
    let dil = dilate(&face_region, h, w, cone_k, cone_kh, cone_kw);
    let ero = erode_iter(&dil, h, w, cheek_k, cheek_kh, cheek_kw, 2);
    let mut fr = vec![0u8; n];
    for i in 0..n {
        let a = if cheek_mask[i] != 0 { ero[i] } else { 0 }; // ero & cheek_mask
        let b = if cheek_mask[i] == 0 { dil[i] } else { 0 }; // dil & ~cheek_mask
        fr[i] = a.max(b); // bitwise_or of two disjoint-region {0,255} maps
    }
    let mut out = vec![0u8; n];
    for i in 0..n {
        let p = parsing[i];
        let cond = (fr[i] == 255 && p != 10) || p == 11 || p == 12 || p == 13;
        out[i] = if cond { 255 } else { 0 };
    }
    out
}

// ---- resize (bilinear, PIL/cv2 INTER_LINEAR half-pixel convention) ---------

/// Bilinear resize of a single-channel u8 image, cv2 INTER_LINEAR semantics
/// (half-pixel centers: src = (dst+0.5)*scale - 0.5). Used for jawmask(512)->face_large size.
pub fn resize_bilinear_u8(src: &[u8], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<u8> {
    let mut out = vec![0u8; dh * dw];
    let sy_scale = sh as f32 / dh as f32;
    let sx_scale = sw as f32 / dw as f32;
    for dy in 0..dh {
        let fy = ((dy as f32 + 0.5) * sy_scale - 0.5).max(0.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = fy - y0 as f32;
        for dx in 0..dw {
            let fx = ((dx as f32 + 0.5) * sx_scale - 0.5).max(0.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = fx - x0 as f32;
            let v00 = src[y0 * sw + x0] as f32;
            let v01 = src[y0 * sw + x1] as f32;
            let v10 = src[y1 * sw + x0] as f32;
            let v11 = src[y1 * sw + x1] as f32;
            let top = v00 * (1.0 - wx) + v01 * wx;
            let bot = v10 * (1.0 - wx) + v11 * wx;
            let v = top * (1.0 - wy) + bot * wy;
            out[dy * dw + dx] = (v + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---- cv2.GaussianBlur (sigma derived from ksize when sigma==0) -------------

fn gaussian_kernel_1d(ksize: usize) -> Vec<f32> {
    // cv2: sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 when sigma<=0
    let sigma = 0.3 * (((ksize - 1) as f32) * 0.5 - 1.0) + 0.8;
    let r = (ksize / 2) as isize;
    let mut k = vec![0f32; ksize];
    let mut sum = 0f32;
    let s2 = 2.0 * sigma * sigma;
    for i in 0..ksize {
        let x = i as isize - r;
        let v = (-(x * x) as f32 / s2).exp();
        k[i] = v;
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// cv2.GaussianBlur on a single-channel u8 image with separable kernel and BORDER_REFLECT_101
/// (cv2 default), sigma auto-derived from ksize. Returns u8 (rounded).
pub fn gaussian_blur_u8(src: &[u8], h: usize, w: usize, ksize: usize) -> Vec<u8> {
    if ksize <= 1 {
        return src.to_vec();
    }
    let k = gaussian_kernel_1d(ksize);
    let r = (ksize / 2) as isize;
    let reflect = |i: isize, n: isize| -> usize {
        // BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba  (reflect without edge repeat)
        let mut i = i;
        if n == 1 {
            return 0;
        }
        loop {
            if i < 0 {
                i = -i;
            } else if i >= n {
                i = 2 * (n - 1) - i;
            } else {
                break;
            }
        }
        i as usize
    };
    // horizontal pass -> f32
    let mut tmp = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sx = reflect(x as isize + ki as isize - r, w as isize);
                acc += kv * src[y * w + sx] as f32;
            }
            tmp[y * w + x] = acc;
        }
    }
    // vertical pass -> u8
    let mut out = vec![0u8; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0f32;
            for (ki, &kv) in k.iter().enumerate() {
                let sy = reflect(y as isize + ki as isize - r, h as isize);
                acc += kv * tmp[sy * w + x];
            }
            out[y * w + x] = (acc + 0.5).clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---- coordinate helpers ----------------------------------------------------

/// MuseTalk get_crop_box: expand the face box about its centre by `expand`.
/// s = int(max(w,h)//2 * expand) ; crop = [xc-s, yc-s, xc+s, yc+s].
pub fn get_crop_box(box_: [i64; 4], expand: f64) -> ([i64; 4], i64) {
    let [x, y, x1, y1] = box_;
    let xc = (x + x1) / 2;
    let yc = (y + y1) / 2;
    let bw = x1 - x;
    let bh = y1 - y;
    let s = ((bw.max(bh) / 2) as f64 * expand) as i64;
    ([xc - s, yc - s, xc + s, yc + s], s)
}

// ---- final composite -------------------------------------------------------

/// Replicate `get_image`'s alpha-feathered paste, given:
///   - `body_bgr`: full frame, BGR HxWx3 (modified copy returned)
///   - `face_bgr`: rendered face crop, BGR (h_box x w_box x3), already resized to (x1-x, y1-y)
///   - `face_box` = [x, y, x1, y1]; `crop_box` = [x_s, y_s, x_e, y_e]
///   - `alpha`: feathered mask at face_large size (ori_h x ori_w), u8
/// face_large is the crop_box sub-image of body; we paste face into it at (x-x_s, y-y_s),
/// then alpha-composite face_large back into body over crop_box[:2].
pub fn composite(
    body_bgr: &mut [u8],
    bh: usize,
    bw: usize,
    face_bgr: &[u8],
    fbh: usize,
    fbw: usize,
    face_box: [i64; 4],
    crop_box: [i64; 4],
    alpha: &[u8],
    oh: usize,
    ow: usize,
) {
    let [x, y, _x1, _y1] = face_box;
    let [x_s, y_s, _xe, _ye] = crop_box;
    let (x_s, y_s) = (x_s as isize, y_s as isize); // composite in isize to match loop indices
    let off_x = (x as isize - x_s) as isize; // paste position of face inside face_large
    let off_y = (y as isize - y_s) as isize;
    // For each pixel of face_large (size oh x ow at body offset (x_s,y_s)), composite into body.
    for ly in 0..oh as isize {
        let by = y_s + ly; // body row
        if by < 0 || by >= bh as isize {
            continue;
        }
        for lx in 0..ow as isize {
            let bx = x_s + lx; // body col
            if bx < 0 || bx >= bw as isize {
                continue;
            }
            let a = alpha[(ly as usize) * ow + lx as usize] as f32 / 255.0;
            if a <= 0.0 {
                continue; // body unchanged where alpha==0
            }
            // face_large pixel = pasted face (if inside the face sub-rect) else original body crop.
            let fy = ly - off_y;
            let fx = lx - off_x;
            let src = if fy >= 0 && fy < fbh as isize && fx >= 0 && fx < fbw as isize {
                let fi = (fy as usize * fbw + fx as usize) * 3;
                [face_bgr[fi], face_bgr[fi + 1], face_bgr[fi + 2]]
            } else {
                let bi = (by as usize * bw + bx as usize) * 3;
                [body_bgr[bi], body_bgr[bi + 1], body_bgr[bi + 2]]
            };
            let bi = (by as usize * bw + bx as usize) * 3;
            for c in 0..3 {
                let dst = body_bgr[bi + c] as f32;
                let s = src[c] as f32;
                // PIL Image.paste with L mask: out = dst*(1-a) + src*a, rounded.
                body_bgr[bi + c] = (dst * (1.0 - a) + s * a + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Build the feathered alpha (steps 2b-3 of get_image) from the 512x512 jaw mask.
///   jaw512 -> resize to (ow,oh)=face_large size -> crop the face sub-rect, paste into black,
///   zero the top `upper_boundary_ratio` rows, GaussianBlur(blur_k).
pub fn feather_alpha(
    jaw512: &[u8],
    parse_h: usize,
    parse_w: usize,
    oh: usize,
    ow: usize,
    face_box: [i64; 4],
    crop_box: [i64; 4],
    upper_boundary_ratio: f64,
    blur_k: usize,
) -> Vec<u8> {
    // resize jaw mask to face_large size
    let mask_full = resize_bilinear_u8(jaw512, parse_h, parse_w, oh, ow);
    let [x, y, x1, y1] = face_box;
    let [x_s, y_s, _xe, _ye] = crop_box;
    // mask_small = mask_full.crop((x-x_s, y-y_s, x1-x_s, y1-y_s)) pasted into black canvas at same pos.
    // Net effect: keep mask_full only inside the face sub-rect, zero elsewhere.
    let cx0 = (x - x_s).max(0);
    let cy0 = (y - y_s).max(0);
    let cx1 = (x1 - x_s).min(ow as i64);
    let cy1 = (y1 - y_s).min(oh as i64);
    let mut mask = vec![0u8; oh * ow];
    for yy in cy0..cy1 {
        for xx in cx0..cx1 {
            let idx = yy as usize * ow + xx as usize;
            mask[idx] = mask_full[idx];
        }
    }
    // keep bottom: zero rows [0, top_boundary)
    let top_boundary = (oh as f64 * upper_boundary_ratio) as usize;
    for yy in 0..top_boundary.min(oh) {
        for xx in 0..ow {
            mask[yy * ow + xx] = 0;
        }
    }
    gaussian_blur_u8(&mask, oh, ow, blur_k)
}
