//! BiSeNet face-parsing network (MuseTalk `utils/face_parsing`) ported to hanzo-ml.
//!
//! Architecture (verbatim from the PyTorch reference):
//!   ContextPath: Resnet18 backbone (feat8/16/32) + ARM16/ARM32 + global-avg branch +
//!                conv_head16/32 (nearest-upsample fusion)
//!   FeatureFusionModule(256,256): cat(res8-as-spatial, cp8) -> conv1x1 -> SE-style attention
//!   BiSeNetOutput seg heads (conv_out / conv_out16 / conv_out32) -> bilinear upsample to HxW
//!
//! Only `conv_out` (the 19-class logits at full res) is used at inference; the 16/32 aux heads
//! exist in the checkpoint but their argmax is identical contribution-wise — we still build them
//! so the VarBuilder consumes every tensor (and to mirror the reference forward exactly).
//!
//! Weight names match the converted `bisenet.safetensors` (PyTorch module paths verbatim).
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Result, Tensor};
use hanzo_nn::{BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT};
use hanzo_quant::{Convolution, ShardedVarBuilder};

use crate::layers::{batch_norm, conv2d_no_bias};

const BN_EPS: f64 = 1e-5;
pub const N_CLASSES: usize = 19;

// ---------------------------------------------------------------------------
// ConvBNReLU: Conv2d(bias=False) -> BatchNorm2d -> ReLU
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct ConvBnReLU {
    conv: Conv2d,
    bn: BatchNorm,
}

impl ConvBnReLU {
    fn new(
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        padding: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        };
        let conv = conv2d_no_bias(in_c, out_c, ks, cfg, vb.pp("conv"))?;
        let bn = batch_norm(out_c, BN_EPS, vb.pp("bn"))?;
        Ok(Self { conv, bn })
    }
}

impl Module for ConvBnReLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = Convolution.forward_2d(&self.conv, x)?;
        let x = self.bn.forward_t(&x, false)?;
        x.relu()
    }
}

// ---------------------------------------------------------------------------
// ResNet-18 backbone (BasicBlock, stride-2 downsampling, returns feat8/16/32)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct BasicBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    downsample: Option<(Conv2d, BatchNorm)>,
}

impl BasicBlock {
    fn new(in_c: usize, out_c: usize, stride: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let cfg3 = Conv2dConfig {
            padding: 1,
            stride,
            ..Default::default()
        };
        let cfg3b = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(in_c, out_c, 3, cfg3, vb.pp("conv1"))?;
        let bn1 = batch_norm(out_c, BN_EPS, vb.pp("bn1"))?;
        let conv2 = conv2d_no_bias(out_c, out_c, 3, cfg3b, vb.pp("conv2"))?;
        let bn2 = batch_norm(out_c, BN_EPS, vb.pp("bn2"))?;
        let downsample = if in_c != out_c || stride != 1 {
            let dcfg = Conv2dConfig {
                stride,
                ..Default::default()
            };
            let dconv = conv2d_no_bias(in_c, out_c, 1, dcfg, vb.pp("downsample").pp(0))?;
            let dbn = batch_norm(out_c, BN_EPS, vb.pp("downsample").pp(1))?;
            Some((dconv, dbn))
        } else {
            None
        };
        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
        })
    }
}

impl Module for BasicBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut residual = Convolution.forward_2d(&self.conv1, x)?;
        residual = self.bn1.forward_t(&residual, false)?.relu()?;
        residual = Convolution.forward_2d(&self.conv2, &residual)?;
        residual = self.bn2.forward_t(&residual, false)?;
        let shortcut = match &self.downsample {
            None => x.clone(),
            Some((c, b)) => {
                let s = Convolution.forward_2d(c, x)?;
                b.forward_t(&s, false)?
            }
        };
        (shortcut + residual)?.relu()
    }
}

fn make_layer(
    in_c: usize,
    out_c: usize,
    bnum: usize,
    stride: usize,
    vb: ShardedVarBuilder,
) -> Result<Vec<BasicBlock>> {
    let mut blocks = Vec::with_capacity(bnum);
    blocks.push(BasicBlock::new(in_c, out_c, stride, vb.pp(0))?);
    for i in 1..bnum {
        blocks.push(BasicBlock::new(out_c, out_c, 1, vb.pp(i))?);
    }
    Ok(blocks)
}

#[derive(Debug, Clone)]
struct Resnet18 {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
}

impl Resnet18 {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 3,
            stride: 2,
            ..Default::default()
        };
        let conv1 = conv2d_no_bias(3, 64, 7, cfg, vb.pp("conv1"))?;
        let bn1 = batch_norm(64, BN_EPS, vb.pp("bn1"))?;
        let layer1 = make_layer(64, 64, 2, 1, vb.pp("layer1"))?;
        let layer2 = make_layer(64, 128, 2, 2, vb.pp("layer2"))?;
        let layer3 = make_layer(128, 256, 2, 2, vb.pp("layer3"))?;
        let layer4 = make_layer(256, 512, 2, 2, vb.pp("layer4"))?;
        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer4,
        })
    }

    /// Returns (feat8, feat16, feat32) at 1/8, 1/16, 1/32 resolution.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let mut x = Convolution.forward_2d(&self.conv1, x)?;
        x = self.bn1.forward_t(&x, false)?.relu()?;
        // maxpool k=3 s=2 p=1
        x = x.pad_with_zeros(2, 1, 1)?.pad_with_zeros(3, 1, 1)?;
        x = x.max_pool2d_with_stride(3, 2)?;
        for b in &self.layer1 {
            x = b.forward(&x)?;
        }
        for b in &self.layer2 {
            x = b.forward(&x)?;
        }
        let feat8 = x.clone();
        for b in &self.layer3 {
            x = b.forward(&x)?;
        }
        let feat16 = x.clone();
        for b in &self.layer4 {
            x = b.forward(&x)?;
        }
        let feat32 = x;
        Ok((feat8, feat16, feat32))
    }
}

// ---------------------------------------------------------------------------
// AttentionRefinementModule
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct AttentionRefinement {
    conv: ConvBnReLU,
    conv_atten: Conv2d,
    bn_atten: BatchNorm,
}

impl AttentionRefinement {
    fn new(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = ConvBnReLU::new(in_c, out_c, 3, 1, 1, vb.pp("conv"))?;
        let conv_atten =
            conv2d_no_bias(out_c, out_c, 1, Conv2dConfig::default(), vb.pp("conv_atten"))?;
        let bn_atten = batch_norm(out_c, BN_EPS, vb.pp("bn_atten"))?;
        Ok(Self {
            conv,
            conv_atten,
            bn_atten,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let feat = self.conv.forward(x)?;
        // atten = sigmoid(bn(conv1x1(global_avg_pool(feat))))
        let (_n, _c, h, w) = feat.dims4()?;
        let atten = feat.avg_pool2d((h, w))?; // [n,c,1,1]
        let atten = Convolution.forward_2d(&self.conv_atten, &atten)?;
        let atten = self.bn_atten.forward_t(&atten, false)?;
        let atten = hanzo_nn::ops::sigmoid(&atten)?;
        feat.broadcast_mul(&atten)
    }
}

// ---------------------------------------------------------------------------
// ContextPath
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct ContextPath {
    resnet: Resnet18,
    arm16: AttentionRefinement,
    arm32: AttentionRefinement,
    conv_head32: ConvBnReLU,
    conv_head16: ConvBnReLU,
    conv_avg: ConvBnReLU,
}

impl ContextPath {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let resnet = Resnet18::new(vb.pp("resnet"))?;
        let arm16 = AttentionRefinement::new(256, 128, vb.pp("arm16"))?;
        let arm32 = AttentionRefinement::new(512, 128, vb.pp("arm32"))?;
        let conv_head32 = ConvBnReLU::new(128, 128, 3, 1, 1, vb.pp("conv_head32"))?;
        let conv_head16 = ConvBnReLU::new(128, 128, 3, 1, 1, vb.pp("conv_head16"))?;
        let conv_avg = ConvBnReLU::new(512, 128, 1, 1, 0, vb.pp("conv_avg"))?;
        Ok(Self {
            resnet,
            arm16,
            arm32,
            conv_head32,
            conv_head16,
            conv_avg,
        })
    }

    /// Returns (feat8, feat16_up, feat32_up) — feat8 at x8, feat16_up at x8, feat32_up at x16.
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (feat8, feat16, feat32) = self.resnet.forward(x)?;
        let (_n, _c, h8, w8) = feat8.dims4()?;
        let (_, _, h16, w16) = feat16.dims4()?;
        let (_, _, h32, w32) = feat32.dims4()?;

        // global average over feat32 -> conv_avg -> nearest-up to (h32,w32)
        let avg = feat32.avg_pool2d((h32, w32))?;
        let avg = self.conv_avg.forward(&avg)?;
        let avg_up = avg.upsample_nearest2d(h32, w32)?;

        let feat32_arm = self.arm32.forward(&feat32)?;
        let feat32_sum = feat32_arm.broadcast_add(&avg_up)?;
        let feat32_up = feat32_sum.upsample_nearest2d(h16, w16)?;
        let feat32_up = self.conv_head32.forward(&feat32_up)?;

        let feat16_arm = self.arm16.forward(&feat16)?;
        let feat16_sum = (feat16_arm + &feat32_up)?;
        let feat16_up = feat16_sum.upsample_nearest2d(h8, w8)?;
        let feat16_up = self.conv_head16.forward(&feat16_up)?;

        Ok((feat8, feat16_up, feat32_up))
    }
}

// ---------------------------------------------------------------------------
// FeatureFusionModule
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct FeatureFusion {
    convblk: ConvBnReLU,
    conv1: Conv2d,
    conv2: Conv2d,
}

impl FeatureFusion {
    fn new(in_c: usize, out_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let convblk = ConvBnReLU::new(in_c, out_c, 1, 1, 0, vb.pp("convblk"))?;
        let conv1 = conv2d_no_bias(out_c, out_c / 4, 1, Conv2dConfig::default(), vb.pp("conv1"))?;
        let conv2 = conv2d_no_bias(out_c / 4, out_c, 1, Conv2dConfig::default(), vb.pp("conv2"))?;
        Ok(Self {
            convblk,
            conv1,
            conv2,
        })
    }

    fn forward(&self, fsp: &Tensor, fcp: &Tensor) -> Result<Tensor> {
        let fcat = Tensor::cat(&[fsp, fcp], 1)?;
        let feat = self.convblk.forward(&fcat)?;
        let (_n, _c, h, w) = feat.dims4()?;
        let atten = feat.avg_pool2d((h, w))?;
        let atten = Convolution.forward_2d(&self.conv1, &atten)?.relu()?;
        let atten = Convolution.forward_2d(&self.conv2, &atten)?;
        let atten = hanzo_nn::ops::sigmoid(&atten)?;
        let feat_atten = feat.broadcast_mul(&atten)?;
        feat_atten + &feat
    }
}

// ---------------------------------------------------------------------------
// BiSeNetOutput seg head: ConvBNReLU(in,mid) -> Conv2d(mid,n_classes,1,bias=False)
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct BiSeNetOutput {
    conv: ConvBnReLU,
    conv_out: Conv2d,
}

impl BiSeNetOutput {
    fn new(in_c: usize, mid_c: usize, n_classes: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = ConvBnReLU::new(in_c, mid_c, 3, 1, 1, vb.pp("conv"))?;
        let conv_out =
            conv2d_no_bias(mid_c, n_classes, 1, Conv2dConfig::default(), vb.pp("conv_out"))?;
        Ok(Self { conv, conv_out })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        Convolution.forward_2d(&self.conv_out, &x)
    }
}

// ---------------------------------------------------------------------------
// BiSeNet
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct BiSeNet {
    cp: ContextPath,
    ffm: FeatureFusion,
    conv_out: BiSeNetOutput,
    conv_out16: BiSeNetOutput,
    conv_out32: BiSeNetOutput,
}

impl BiSeNet {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let cp = ContextPath::new(vb.pp("cp"))?;
        let ffm = FeatureFusion::new(256, 256, vb.pp("ffm"))?;
        let conv_out = BiSeNetOutput::new(256, 256, N_CLASSES, vb.pp("conv_out"))?;
        let conv_out16 = BiSeNetOutput::new(128, 64, N_CLASSES, vb.pp("conv_out16"))?;
        let conv_out32 = BiSeNetOutput::new(128, 64, N_CLASSES, vb.pp("conv_out32"))?;
        Ok(Self {
            cp,
            ffm,
            conv_out,
            conv_out16,
            conv_out32,
        })
    }

    /// Full forward -> 19-class logits upsampled to the input HxW (`feat_out`, the head used for
    /// argmax). align_corners=true matches PyTorch `F.interpolate(mode='bilinear', align_corners=True)`.
    pub fn forward_logits(&self, x: &Tensor) -> Result<Tensor> {
        let (_n, _c, h, w) = x.dims4()?;
        let (feat_res8, feat_cp8, _feat_cp16) = self.cp.forward(x)?;
        let feat_fuse = self.ffm.forward(&feat_res8, &feat_cp8)?;
        let feat_out = self.conv_out.forward(&feat_fuse)?;
        feat_out.upsample_bilinear2d(h, w, true)
    }

    /// Argmax 19-class label map at full input resolution, as i64 [H,W] on CPU.
    pub fn parse(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(x)?; // [1,19,H,W]
        let parsing = logits.squeeze(0)?.argmax(0)?; // [H,W]
        parsing.to_dtype(hanzo_ml::DType::I64)
    }
}
