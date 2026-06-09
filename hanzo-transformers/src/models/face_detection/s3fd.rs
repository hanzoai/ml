//! S3FD (Single Shot Scale-invariant Face Detector), the VGG16-based single-shot
//! detector used by `face_alignment`. Rust port of
//! `face_alignment/detection/sfd/net_s3fd.py` + `detect.py` + `bbox.py`.
//!
//! Produces face bounding boxes `[x1, y1, x2, y2, score]` in image pixel coords,
//! byte-for-byte matching the PyTorch reference (modulo f32 rounding).

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor};
use hanzo_nn::{
    conv2d, ops::softmax, Conv2d, Conv2dConfig, Module, VarBuilder,
};

/// L2Norm channel normalization layer (SSD-style), with a per-channel learnable scale.
#[derive(Debug)]
struct L2Norm {
    weight: Tensor, // [C]
    eps: f64,
}

impl L2Norm {
    fn new(n_channels: usize, vb: VarBuilder) -> Result<Self> {
        // The .pth stores it under `<name>.weight`.
        let weight = vb.get(n_channels, "weight")?;
        Ok(Self { weight, eps: 1e-10 })
    }
}

impl Module for L2Norm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // norm = sqrt(sum_c x^2) + eps ; x = x / norm * weight[1,C,1,1]
        let norm = (x.sqr()?.sum_keepdim(1)?.sqrt()? + self.eps)?;
        let x = x.broadcast_div(&norm)?;
        let w = self.weight.reshape((1, self.weight.dim(0)?, 1, 1))?;
        x.broadcast_mul(&w)
    }
}

fn c(i: usize, o: usize, k: usize, pad: usize, stride: usize, vb: VarBuilder) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding: pad,
        stride,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    conv2d(i, o, k, cfg, vb)
}

#[derive(Debug)]
pub struct S3fd {
    conv1_1: Conv2d,
    conv1_2: Conv2d,
    conv2_1: Conv2d,
    conv2_2: Conv2d,
    conv3_1: Conv2d,
    conv3_2: Conv2d,
    conv3_3: Conv2d,
    conv4_1: Conv2d,
    conv4_2: Conv2d,
    conv4_3: Conv2d,
    conv5_1: Conv2d,
    conv5_2: Conv2d,
    conv5_3: Conv2d,
    fc6: Conv2d,
    fc7: Conv2d,
    conv6_1: Conv2d,
    conv6_2: Conv2d,
    conv7_1: Conv2d,
    conv7_2: Conv2d,
    conv3_3_norm: L2Norm,
    conv4_3_norm: L2Norm,
    conv5_3_norm: L2Norm,
    conv3_3_norm_mbox_conf: Conv2d,
    conv3_3_norm_mbox_loc: Conv2d,
    conv4_3_norm_mbox_conf: Conv2d,
    conv4_3_norm_mbox_loc: Conv2d,
    conv5_3_norm_mbox_conf: Conv2d,
    conv5_3_norm_mbox_loc: Conv2d,
    fc7_mbox_conf: Conv2d,
    fc7_mbox_loc: Conv2d,
    conv6_2_mbox_conf: Conv2d,
    conv6_2_mbox_loc: Conv2d,
    conv7_2_mbox_conf: Conv2d,
    conv7_2_mbox_loc: Conv2d,
}

impl S3fd {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv1_1: c(3, 64, 3, 1, 1, vb.pp("conv1_1"))?,
            conv1_2: c(64, 64, 3, 1, 1, vb.pp("conv1_2"))?,
            conv2_1: c(64, 128, 3, 1, 1, vb.pp("conv2_1"))?,
            conv2_2: c(128, 128, 3, 1, 1, vb.pp("conv2_2"))?,
            conv3_1: c(128, 256, 3, 1, 1, vb.pp("conv3_1"))?,
            conv3_2: c(256, 256, 3, 1, 1, vb.pp("conv3_2"))?,
            conv3_3: c(256, 256, 3, 1, 1, vb.pp("conv3_3"))?,
            conv4_1: c(256, 512, 3, 1, 1, vb.pp("conv4_1"))?,
            conv4_2: c(512, 512, 3, 1, 1, vb.pp("conv4_2"))?,
            conv4_3: c(512, 512, 3, 1, 1, vb.pp("conv4_3"))?,
            conv5_1: c(512, 512, 3, 1, 1, vb.pp("conv5_1"))?,
            conv5_2: c(512, 512, 3, 1, 1, vb.pp("conv5_2"))?,
            conv5_3: c(512, 512, 3, 1, 1, vb.pp("conv5_3"))?,
            fc6: c(512, 1024, 3, 3, 1, vb.pp("fc6"))?,
            fc7: c(1024, 1024, 1, 0, 1, vb.pp("fc7"))?,
            conv6_1: c(1024, 256, 1, 0, 1, vb.pp("conv6_1"))?,
            conv6_2: c(256, 512, 3, 1, 2, vb.pp("conv6_2"))?,
            conv7_1: c(512, 128, 1, 0, 1, vb.pp("conv7_1"))?,
            conv7_2: c(128, 256, 3, 1, 2, vb.pp("conv7_2"))?,
            conv3_3_norm: L2Norm::new(256, vb.pp("conv3_3_norm"))?,
            conv4_3_norm: L2Norm::new(512, vb.pp("conv4_3_norm"))?,
            conv5_3_norm: L2Norm::new(512, vb.pp("conv5_3_norm"))?,
            conv3_3_norm_mbox_conf: c(256, 4, 3, 1, 1, vb.pp("conv3_3_norm_mbox_conf"))?,
            conv3_3_norm_mbox_loc: c(256, 4, 3, 1, 1, vb.pp("conv3_3_norm_mbox_loc"))?,
            conv4_3_norm_mbox_conf: c(512, 2, 3, 1, 1, vb.pp("conv4_3_norm_mbox_conf"))?,
            conv4_3_norm_mbox_loc: c(512, 4, 3, 1, 1, vb.pp("conv4_3_norm_mbox_loc"))?,
            conv5_3_norm_mbox_conf: c(512, 2, 3, 1, 1, vb.pp("conv5_3_norm_mbox_conf"))?,
            conv5_3_norm_mbox_loc: c(512, 4, 3, 1, 1, vb.pp("conv5_3_norm_mbox_loc"))?,
            fc7_mbox_conf: c(1024, 2, 3, 1, 1, vb.pp("fc7_mbox_conf"))?,
            fc7_mbox_loc: c(1024, 4, 3, 1, 1, vb.pp("fc7_mbox_loc"))?,
            conv6_2_mbox_conf: c(512, 2, 3, 1, 1, vb.pp("conv6_2_mbox_conf"))?,
            conv6_2_mbox_loc: c(512, 4, 3, 1, 1, vb.pp("conv6_2_mbox_loc"))?,
            conv7_2_mbox_conf: c(256, 2, 3, 1, 1, vb.pp("conv7_2_mbox_conf"))?,
            conv7_2_mbox_loc: c(256, 4, 3, 1, 1, vb.pp("conv7_2_mbox_loc"))?,
        })
    }

    fn maxpool(x: &Tensor) -> Result<Tensor> {
        // kernel 2, stride 2 (PyTorch F.max_pool2d(h, 2, 2)); ceil_mode=False (default).
        x.max_pool2d_with_stride(2, 2)
    }

    /// Forward pass. `x` is `[1, 3, H, W]` already in BGR with mean subtracted.
    /// Returns the 12 output heads `[cls1, reg1, cls2, reg2, ...]`.
    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let h = self.conv1_1.forward(x)?.relu()?;
        let h = self.conv1_2.forward(&h)?.relu()?;
        let h = Self::maxpool(&h)?;

        let h = self.conv2_1.forward(&h)?.relu()?;
        let h = self.conv2_2.forward(&h)?.relu()?;
        let h = Self::maxpool(&h)?;

        let h = self.conv3_1.forward(&h)?.relu()?;
        let h = self.conv3_2.forward(&h)?.relu()?;
        let h = self.conv3_3.forward(&h)?.relu()?;
        let f3_3 = h.clone();
        let h = Self::maxpool(&h)?;

        let h = self.conv4_1.forward(&h)?.relu()?;
        let h = self.conv4_2.forward(&h)?.relu()?;
        let h = self.conv4_3.forward(&h)?.relu()?;
        let f4_3 = h.clone();
        let h = Self::maxpool(&h)?;

        let h = self.conv5_1.forward(&h)?.relu()?;
        let h = self.conv5_2.forward(&h)?.relu()?;
        let h = self.conv5_3.forward(&h)?.relu()?;
        let f5_3 = h.clone();
        let h = Self::maxpool(&h)?;

        let h = self.fc6.forward(&h)?.relu()?;
        let h = self.fc7.forward(&h)?.relu()?;
        let ffc7 = h.clone();
        let h = self.conv6_1.forward(&h)?.relu()?;
        let h = self.conv6_2.forward(&h)?.relu()?;
        let f6_2 = h.clone();
        let h = self.conv7_1.forward(&h)?.relu()?;
        let f7_2 = self.conv7_2.forward(&h)?.relu()?;

        let f3_3 = self.conv3_3_norm.forward(&f3_3)?;
        let f4_3 = self.conv4_3_norm.forward(&f4_3)?;
        let f5_3 = self.conv5_3_norm.forward(&f5_3)?;

        let cls1 = self.conv3_3_norm_mbox_conf.forward(&f3_3)?;
        let reg1 = self.conv3_3_norm_mbox_loc.forward(&f3_3)?;
        let cls2 = self.conv4_3_norm_mbox_conf.forward(&f4_3)?;
        let reg2 = self.conv4_3_norm_mbox_loc.forward(&f4_3)?;
        let cls3 = self.conv5_3_norm_mbox_conf.forward(&f5_3)?;
        let reg3 = self.conv5_3_norm_mbox_loc.forward(&f5_3)?;
        let cls4 = self.fc7_mbox_conf.forward(&ffc7)?;
        let reg4 = self.fc7_mbox_loc.forward(&ffc7)?;
        let cls5 = self.conv6_2_mbox_conf.forward(&f6_2)?;
        let reg5 = self.conv6_2_mbox_loc.forward(&f6_2)?;
        let cls6 = self.conv7_2_mbox_conf.forward(&f7_2)?;
        let reg6 = self.conv7_2_mbox_loc.forward(&f7_2)?;

        // max-out background label on the first conf head:
        // chunk into 4 along channel; bmax = max(max(c0,c1),c2); cls1 = cat([bmax, c3])
        let c0 = cls1.i((.., 0..1))?;
        let c1 = cls1.i((.., 1..2))?;
        let c2 = cls1.i((.., 2..3))?;
        let c3 = cls1.i((.., 3..4))?;
        let bmax = c0.maximum(&c1)?.maximum(&c2)?;
        let cls1 = Tensor::cat(&[&bmax, &c3], 1)?;

        Ok(vec![
            cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6,
        ])
    }
}

/// A detected box `[x1, y1, x2, y2, score]`.
#[derive(Debug, Clone, Copy)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// Decode + collect candidate boxes from the network outputs, mirroring
/// `get_predictions` in detect.py. Softmax is applied to the conf heads first.
fn get_predictions(olist: &[Tensor]) -> Result<Vec<Bbox>> {
    let variances = [0.1f32, 0.2f32];
    let mut out = Vec::new();
    let n = olist.len() / 2;
    for i in 0..n {
        // softmax over channel dim on the conf head
        let ocls = softmax(&olist[i * 2], 1)?;
        let oreg = &olist[i * 2 + 1];
        let stride = 2f32.powi((i + 2) as i32); // 4,8,16,32,64,128

        // ocls[:,1,:,:] > 0.05 → gather positions. Batch is always 1.
        let (_b, _c, hh, ww) = ocls.dims4()?;
        let cls1 = ocls.i((0, 1))?.to_dtype(DType::F32)?; // [H,W]
        let cls1v: Vec<f32> = cls1.flatten_all()?.to_vec1()?;
        // reg: [1,4,H,W] → [4,H,W]
        let reg = oreg.i(0)?.to_dtype(DType::F32)?;
        let regv: Vec<f32> = reg.flatten_all()?.to_vec1()?; // index = ch*H*W + y*W + x
        let hw = hh * ww;
        for hindex in 0..hh {
            for windex in 0..ww {
                let p = cls1v[hindex * ww + windex];
                if p > 0.05 {
                    let axc = stride / 2.0 + windex as f32 * stride;
                    let ayc = stride / 2.0 + hindex as f32 * stride;
                    let prior = [axc, ayc, stride * 4.0, stride * 4.0];
                    let loc = [
                        regv[hindex * ww + windex],          // ch 0
                        regv[hw + hindex * ww + windex],     // ch 1
                        regv[2 * hw + hindex * ww + windex], // ch 2
                        regv[3 * hw + hindex * ww + windex], // ch 3
                    ];
                    // decode (bbox.py): center-form -> point-form
                    let cx = prior[0] + loc[0] * variances[0] * prior[2];
                    let cy = prior[1] + loc[1] * variances[0] * prior[3];
                    let w = prior[2] * (loc[2] * variances[1]).exp();
                    let h = prior[3] * (loc[3] * variances[1]).exp();
                    let x1 = cx - w / 2.0;
                    let y1 = cy - h / 2.0;
                    let x2 = x1 + w;
                    let y2 = y1 + h;
                    out.push(Bbox {
                        x1,
                        y1,
                        x2,
                        y2,
                        score: p,
                    });
                }
            }
        }
    }
    Ok(out)
}

/// Non-maximum suppression, mirroring bbox.py `nms` (note the +1 on areas).
pub fn nms(dets: &[Bbox], thresh: f32) -> Vec<usize> {
    if dets.is_empty() {
        return vec![];
    }
    let areas: Vec<f32> = dets
        .iter()
        .map(|d| (d.x2 - d.x1 + 1.0) * (d.y2 - d.y1 + 1.0))
        .collect();
    let mut order: Vec<usize> = (0..dets.len()).collect();
    // descending by score (argsort()[::-1]); stable to match numpy semantics closely
    order.sort_by(|&a, &b| dets[b].score.partial_cmp(&dets[a].score).unwrap());
    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        let mut new_order = Vec::new();
        for &j in &order[1..] {
            let xx1 = dets[i].x1.max(dets[j].x1);
            let yy1 = dets[i].y1.max(dets[j].y1);
            let xx2 = dets[i].x2.min(dets[j].x2);
            let yy2 = dets[i].y2.min(dets[j].y2);
            let w = (xx2 - xx1 + 1.0).max(0.0);
            let h = (yy2 - yy1 + 1.0).max(0.0);
            let inter = w * h;
            let ovr = inter / (areas[i] + areas[j] - inter);
            if ovr <= thresh {
                new_order.push(j);
            }
        }
        order = new_order;
    }
    keep
}

/// Build the input tensor for S3FD from an RGB image (H,W,3 u8) given as a flat
/// `[H*W*3]` row-major RGB byte buffer, replicating detect.py preprocessing:
///   img(HWC RGB) -> CHW -> flip RGB->BGR -> subtract [104,117,123].
pub fn preprocess_rgb(rgb: &[u8], h: usize, w: usize, device: &Device) -> Result<Tensor> {
    // HWC RGB f32
    let t = Tensor::from_vec(
        rgb.iter().map(|&v| v as f32).collect::<Vec<f32>>(),
        (h, w, 3),
        device,
    )?;
    // -> CHW
    let t = t.permute((2, 0, 1))?; // [3,H,W] RGB
                                   // flip channel RGB->BGR
    let b = t.i(2)?;
    let g = t.i(1)?;
    let r = t.i(0)?;
    let bgr = Tensor::stack(&[&b, &g, &r], 0)?; // [3,H,W] BGR
                                                // subtract mean
    let mean = Tensor::from_vec(vec![104.0f32, 117.0, 123.0], (3, 1, 1), device)?;
    let bgr = bgr.broadcast_sub(&mean)?;
    bgr.unsqueeze(0) // [1,3,H,W]
}

/// Full detect: run the net, decode, NMS@0.3, filter by `filter_threshold` (default 0.5),
/// matching SFDDetector.detect_from_image + _filter_bboxes.
pub fn detect(
    net: &S3fd,
    rgb: &[u8],
    h: usize,
    w: usize,
    filter_threshold: f32,
    device: &Device,
) -> Result<Vec<Bbox>> {
    let x = preprocess_rgb(rgb, h, w, device)?;
    let olist = net.forward(&x)?;
    let cands = get_predictions(&olist)?;
    let keep = nms(&cands, 0.3);
    let mut filtered: Vec<Bbox> = keep
        .into_iter()
        .map(|i| cands[i])
        .filter(|b| b.score > filter_threshold)
        .collect();
    // detect_from_image returns boxes in NMS/score order already (kept order).
    // Keep as-is.
    filtered.shrink_to_fit();
    Ok(filtered)
}
