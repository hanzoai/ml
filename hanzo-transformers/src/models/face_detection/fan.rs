//! 2D-FAN-4 (Face Alignment Network), the 4-stack hourglass that predicts 68
//! landmark heatmaps. Rust port of `face_alignment/models/fan.py`.
//!
//! Input is a 256x256 face crop (RGB, /255), output is the last stack's
//! `[1, 68, 64, 64]` heatmaps.

use hanzo_ml::{Result, Tensor};
use hanzo_nn::{batch_norm, conv2d, BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT, VarBuilder};

fn conv3x3(i: usize, o: usize, vb: VarBuilder) -> Result<Conv2d> {
    // bias=False in conv3x3
    let cfg = Conv2dConfig {
        padding: 1,
        stride: 1,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    hanzo_nn::conv2d_no_bias(i, o, 3, cfg, vb)
}

fn bn(c: usize, vb: VarBuilder) -> Result<BatchNorm> {
    batch_norm(c, BatchNormConfigDefault::cfg(), vb)
}

struct BatchNormConfigDefault;
impl BatchNormConfigDefault {
    fn cfg() -> hanzo_nn::BatchNormConfig {
        hanzo_nn::BatchNormConfig::default() // eps 1e-5, affine, momentum 0.1
    }
}

/// ConvBlock: bottleneck residual with concatenation of three conv outputs.
#[derive(Debug)]
struct ConvBlock {
    bn1: BatchNorm,
    conv1: Conv2d,
    bn2: BatchNorm,
    conv2: Conv2d,
    bn3: BatchNorm,
    conv3: Conv2d,
    downsample: Option<(BatchNorm, Conv2d)>, // (bn, conv1x1) — Sequential[0]=bn,[2]=conv
}

impl ConvBlock {
    fn new(in_planes: usize, out_planes: usize, vb: VarBuilder) -> Result<Self> {
        let h2 = out_planes / 2;
        let h4 = out_planes / 4;
        let bn1 = bn(in_planes, vb.pp("bn1"))?;
        let conv1 = conv3x3(in_planes, h2, vb.pp("conv1"))?;
        let bn2 = bn(h2, vb.pp("bn2"))?;
        let conv2 = conv3x3(h2, h4, vb.pp("conv2"))?;
        let bn3 = bn(h4, vb.pp("bn3"))?;
        let conv3 = conv3x3(h4, h4, vb.pp("conv3"))?;
        let downsample = if in_planes != out_planes {
            // nn.Sequential(BatchNorm2d, ReLU, Conv2d(1x1,bias=False))
            let dbn = bn(in_planes, vb.pp("downsample").pp("0"))?;
            let cfg = Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            let dconv = hanzo_nn::conv2d_no_bias(in_planes, out_planes, 1, cfg, vb.pp("downsample").pp("2"))?;
            Some((dbn, dconv))
        } else {
            None
        };
        Ok(Self {
            bn1,
            conv1,
            bn2,
            conv2,
            bn3,
            conv3,
            downsample,
        })
    }
}

impl Module for ConvBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let out1 = self.bn1.forward_t(x, false)?.relu()?;
        let out1 = self.conv1.forward(&out1)?;
        let out2 = self.bn2.forward_t(&out1, false)?.relu()?;
        let out2 = self.conv2.forward(&out2)?;
        let out3 = self.bn3.forward_t(&out2, false)?.relu()?;
        let out3 = self.conv3.forward(&out3)?;
        let cat = Tensor::cat(&[&out1, &out2, &out3], 1)?;
        let residual = if let Some((dbn, dconv)) = &self.downsample {
            let r = dbn.forward_t(residual, false)?.relu()?;
            dconv.forward(&r)?
        } else {
            residual.clone()
        };
        cat + residual
    }
}

/// HourGlass module, recursive over `depth`. ConvBlocks named b1_<L>, b2_<L>,
/// b3_<L>, plus b2_plus_<1> at the bottom.
#[derive(Debug)]
struct HourGlass {
    depth: usize,
    // per level (1..=depth): b1, b2, b3 ; plus b2_plus at level 1
    b1: Vec<ConvBlock>,      // index by (level-1)
    b2: Vec<ConvBlock>,      // index by (level-1)
    b3: Vec<ConvBlock>,      // index by (level-1)
    b2_plus: ConvBlock,      // level 1 only
    features: usize,
}

impl HourGlass {
    fn new(depth: usize, num_features: usize, vb: VarBuilder) -> Result<Self> {
        let mut b1 = Vec::with_capacity(depth);
        let mut b2 = Vec::with_capacity(depth);
        let mut b3 = Vec::with_capacity(depth);
        // _generate_network adds for each level from `depth` down to 1
        for level in 1..=depth {
            b1.push(ConvBlock::new(
                num_features,
                num_features,
                vb.pp(format!("b1_{level}")),
            )?);
            b2.push(ConvBlock::new(
                num_features,
                num_features,
                vb.pp(format!("b2_{level}")),
            )?);
            b3.push(ConvBlock::new(
                num_features,
                num_features,
                vb.pp(format!("b3_{level}")),
            )?);
        }
        let b2_plus = ConvBlock::new(num_features, num_features, vb.pp("b2_plus_1"))?;
        Ok(Self {
            depth,
            b1,
            b2,
            b3,
            b2_plus,
            features: num_features,
        })
    }

    fn forward_level(&self, level: usize, inp: &Tensor) -> Result<Tensor> {
        let idx = level - 1;
        // up1 branch
        let up1 = self.b1[idx].forward(inp)?;
        // low branch
        let low1 = inp.avg_pool2d_with_stride(2, 2)?;
        let low1 = self.b2[idx].forward(&low1)?;
        let low2 = if level > 1 {
            self.forward_level(level - 1, &low1)?
        } else {
            self.b2_plus.forward(&low1)?
        };
        let low3 = self.b3[idx].forward(&low2)?;
        // upsample nearest x2
        let (_, _, h, w) = low3.dims4()?;
        let up2 = low3.upsample_nearest2d(h * 2, w * 2)?;
        up1 + up2
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _ = self.features;
        self.forward_level(self.depth, x)
    }
}

#[derive(Debug)]
pub struct Fan {
    num_modules: usize,
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: ConvBlock,
    conv3: ConvBlock,
    conv4: ConvBlock,
    m: Vec<HourGlass>,
    top_m: Vec<ConvBlock>,
    conv_last: Vec<Conv2d>,
    bn_end: Vec<BatchNorm>,
    l: Vec<Conv2d>,
    bl: Vec<Conv2d>,
    al: Vec<Conv2d>,
}

impl Fan {
    pub fn new(num_modules: usize, vb: VarBuilder) -> Result<Self> {
        // conv1: 3->64, k7 s2 p3, bias=True (default Conv2d has bias)
        let cfg7 = Conv2dConfig {
            padding: 3,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv2d(3, 64, 7, cfg7, vb.pp("conv1"))?;
        let bn1 = bn(64, vb.pp("bn1"))?;
        let conv2 = ConvBlock::new(64, 128, vb.pp("conv2"))?;
        let conv3 = ConvBlock::new(128, 128, vb.pp("conv3"))?;
        let conv4 = ConvBlock::new(128, 256, vb.pp("conv4"))?;

        let cfg1 = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let mut m = Vec::new();
        let mut top_m = Vec::new();
        let mut conv_last = Vec::new();
        let mut bn_end = Vec::new();
        let mut l = Vec::new();
        let mut bl = Vec::new();
        let mut al = Vec::new();
        for hg in 0..num_modules {
            m.push(HourGlass::new(4, 256, vb.pp(format!("m{hg}")))?);
            top_m.push(ConvBlock::new(256, 256, vb.pp(format!("top_m_{hg}")))?);
            conv_last.push(conv2d(256, 256, 1, cfg1, vb.pp(format!("conv_last{hg}")))?);
            bn_end.push(bn(256, vb.pp(format!("bn_end{hg}")))?);
            l.push(conv2d(256, 68, 1, cfg1, vb.pp(format!("l{hg}")))?);
            if hg < num_modules - 1 {
                bl.push(conv2d(256, 256, 1, cfg1, vb.pp(format!("bl{hg}")))?);
                al.push(conv2d(68, 256, 1, cfg1, vb.pp(format!("al{hg}")))?);
            }
        }
        Ok(Self {
            num_modules,
            conv1,
            bn1,
            conv2,
            conv3,
            conv4,
            m,
            top_m,
            conv_last,
            bn_end,
            l,
            bl,
            al,
        })
    }

    /// Forward, returns the heatmaps from the LAST stack: `[B, 68, 64, 64]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.bn1.forward_t(&self.conv1.forward(x)?, false)?.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.avg_pool2d_with_stride(2, 2)?;
        let x = self.conv3.forward(&x)?;
        let x = self.conv4.forward(&x)?;

        let mut previous = x;
        let mut last_out = None;
        for i in 0..self.num_modules {
            let hg = self.m[i].forward(&previous)?;
            let ll = self.top_m[i].forward(&hg)?;
            let ll = self.bn_end[i]
                .forward_t(&self.conv_last[i].forward(&ll)?, false)?
                .relu()?;
            let tmp_out = self.l[i].forward(&ll)?;
            last_out = Some(tmp_out.clone());
            if i < self.num_modules - 1 {
                let ll2 = self.bl[i].forward(&ll)?;
                let tmp_out_ = self.al[i].forward(&tmp_out)?;
                previous = ((previous + ll2)? + tmp_out_)?;
            }
        }
        last_out.ok_or_else(|| hanzo_ml::Error::Msg("FAN produced no output".into()))
    }
}
