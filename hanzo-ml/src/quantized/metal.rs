use super::{GgmlDType, QStorage};
use crate::backend::BackendStorage;
use crate::{DType, MetalDevice, MetalStorage, Result, Shape, D};
use hanzo_metal_kernels::metal::Buffer;
use std::sync::Arc;

pub struct QMetalStorage {
    dtype: GgmlDType,
    device: MetalDevice,
    buffer: Arc<Buffer>,
}

impl QMetalStorage {
    pub fn zeros(device: &MetalDevice, elem_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size = elem_count * dtype.type_size() / dtype.block_size();
        let buffer = device.allocate_zeros(size)?;
        Ok(Self {
            buffer,
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &MetalDevice {
        &self.device
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<MetalStorage> {
        use crate::quantized::k_quants::GgmlType;

        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => {
                let vec: Vec<f32> = read_to_vec(&buffer, block_len);
                f32::to_float(&vec, &mut out);
            }
            GgmlDType::F16 => {
                let vec: Vec<half::f16> = read_to_vec(&buffer, block_len);
                half::f16::to_float(&vec, &mut out);
            }
            GgmlDType::BF16 => {
                let vec: Vec<half::bf16> = read_to_vec(&buffer, block_len);
                half::bf16::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_0 => {
                let vec: Vec<crate::quantized::BlockQ4_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q4_1 => {
                let vec: Vec<crate::quantized::BlockQ4_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_0 => {
                let vec: Vec<crate::quantized::BlockQ5_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q5_1 => {
                let vec: Vec<crate::quantized::BlockQ5_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_0 => {
                let vec: Vec<crate::quantized::BlockQ8_0> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_0::to_float(&vec, &mut out);
            }
            GgmlDType::Q8_1 => {
                let vec: Vec<crate::quantized::BlockQ8_1> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8_1::to_float(&vec, &mut out);
            }
            GgmlDType::Q2K => {
                let vec: Vec<crate::quantized::BlockQ2K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ2K::to_float(&vec, &mut out);
            }
            GgmlDType::Q3K => {
                let vec: Vec<crate::quantized::BlockQ3K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ3K::to_float(&vec, &mut out);
            }
            GgmlDType::Q4K => {
                let vec: Vec<crate::quantized::BlockQ4K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ4K::to_float(&vec, &mut out);
            }
            GgmlDType::Q5K => {
                let vec: Vec<crate::quantized::BlockQ5K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ5K::to_float(&vec, &mut out);
            }
            GgmlDType::Q6K => {
                let vec: Vec<crate::quantized::BlockQ6K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ6K::to_float(&vec, &mut out);
            }
            GgmlDType::Q8K => {
                let vec: Vec<crate::quantized::BlockQ8K> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockQ8K::to_float(&vec, &mut out);
            }
            GgmlDType::IQ4_NL => {
                let vec: Vec<crate::quantized::BlockIQ4nl> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ4nl::to_float(&vec, &mut out);
            }
            GgmlDType::IQ4_XS => {
                let vec: Vec<crate::quantized::BlockIQ4xs> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ4xs::to_float(&vec, &mut out);
            }
            GgmlDType::MXFP4 => {
                let vec: Vec<crate::quantized::BlockMXFP4> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockMXFP4::to_float(&vec, &mut out);
            }
            // i-quant codebook family: CPU codebook decode of the resident block bytes (the exact
            // `to_float` the MSL matvec kernels reconstruct), backing ISQ / dequant-to-f32. The native
            // matvec/matmul path (fwd) keeps the weights quantized and never hits this.
            GgmlDType::IQ2_XXS => {
                let vec: Vec<crate::quantized::BlockIQ2xxs> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ2xxs::to_float(&vec, &mut out);
            }
            GgmlDType::IQ2_XS => {
                let vec: Vec<crate::quantized::BlockIQ2xs> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ2xs::to_float(&vec, &mut out);
            }
            GgmlDType::IQ2_S => {
                let vec: Vec<crate::quantized::BlockIQ2s> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ2s::to_float(&vec, &mut out);
            }
            GgmlDType::IQ3_XXS => {
                let vec: Vec<crate::quantized::BlockIQ3xxs> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ3xxs::to_float(&vec, &mut out);
            }
            GgmlDType::IQ3_S => {
                let vec: Vec<crate::quantized::BlockIQ3s> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ3s::to_float(&vec, &mut out);
            }
            GgmlDType::IQ1_S => {
                let vec: Vec<crate::quantized::BlockIQ1s> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ1s::to_float(&vec, &mut out);
            }
            GgmlDType::IQ1_M => {
                let vec: Vec<crate::quantized::BlockIQ1m> = read_to_vec(&buffer, block_len);
                crate::quantized::BlockIQ1m::to_float(&vec, &mut out);
            }
            // dbc-validation: dequant-to-float not wired for these newer ternary/FP4
            // codecs on the Metal readback path (GAP in source). Bail honestly rather
            // than silently mis-decode. Q8_0/Q4_K (the validated models) have arms above.
            other => crate::bail!(
                "dequantize-to-float on Metal not implemented for {:?}",
                other
            ),
        }

        let buffer = self.device.new_buffer_with_data(&out)?;
        Ok(MetalStorage::new(
            buffer,
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    pub fn quantize(&mut self, src: &MetalStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &MetalStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let src = src.to_cpu::<f32>()?;
        let elem_count = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Quantization only happens on CPU for now.
        let elem_count = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(elem_count, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let buffer = self.device.new_buffer_with_data(&qcpu_storage.data()?)?;
        self.buffer = buffer;
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.buffer.length()
    }

    fn fwd_mv(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let (n, k) = self_shape.dims2()?;
        let mut dst_shape = src_shape.dims().to_vec();

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in hanzo-metal-kernels doesn't handle batch
        // properly.
        let m = match dst_shape.len() {
            3 => dst_shape[0] * dst_shape[1],
            2 => dst_shape[0],
            n => crate::bail!("Invalid rank {n} for quantized matmul metal"),
        };
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let kdtype: hanzo_metal_kernels::GgmlDType = self.dtype.try_into()?;
        // bf16-native decode: when the activation arrives in bf16 and the weight is a K-quant with a
        // bf16 matvec, read/write bf16 directly (f32 accumulation) so the bf16<->f32 round-trip that
        // GgufMatMul wraps around the projection disappears. Any other activation stays f32.
        let src1_bf16 = storage.dtype() == DType::BF16
            && matches!(
                kdtype,
                hanzo_metal_kernels::GgmlDType::Q4K | hanzo_metal_kernels::GgmlDType::Q6K
            );
        let out_dtype = if src1_bf16 { DType::BF16 } else { DType::F32 };
        let dst = device.new_buffer(dst_shape.elem_count(), out_dtype, "qmatmul")?;
        let encoder = device.command_encoder()?;
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for batch_id in 0..m {
            hanzo_metal_kernels::call_quantized_matmul_mv_t(
                device.device(),
                &encoder,
                device.kernels(),
                kdtype,
                (1, 1, n, k),
                storage.buffer(),
                (layout.start_offset() + batch_id * k) * storage.dtype().size_in_bytes(),
                &self.buffer,
                batch_id * n * out_dtype.size_in_bytes(),
                &dst,
                src1_bf16,
            )
            .map_err(MetalError::from)?;
        }
        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), out_dtype);
        Ok((dst_storage, dst_shape))
    }

    pub fn fwd(
        &self,
        self_shape: &Shape,
        storage: &MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use crate::MetalError;

        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let n = self_shape.dim(D::Minus2)?;
        let k = self_shape.dim(D::Minus1)?;
        let mut dst_shape = src_shape.dims().to_vec();

        if src_shape.rank() < self_shape.rank() {
            crate::bail!(
                "input rank ({}) must be >= weight rank ({})",
                src_shape.rank(),
                self_shape.rank()
            )
        }

        if src_shape.dim(D::Minus2)? == 1 {
            return self.fwd_mv(self_shape, storage, layout);
        }

        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self_shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        let device = storage.device().clone();
        let kdtype: hanzo_metal_kernels::GgmlDType = self.dtype.try_into()?;
        // bf16-native prefill: a bf16 activation against a K-quant with a bf16 mm (Q4_K/Q6_K) reads
        // src1 and writes dst in bf16 directly, so the bf16<->f32 round-trip GgufMatMul wraps around
        // the projection disappears -- the multi-token twin of the bf16 decode matvec in `fwd_mv`.
        // The simdgroup math is identical to the f32 mm (src1 is widened to f32 on stage); only the
        // I/O dtype changes, so f32 models and unsupported quants are unaffected.
        let src1_bf16 = storage.dtype() == DType::BF16
            && matches!(
                kdtype,
                hanzo_metal_kernels::GgmlDType::Q4K | hanzo_metal_kernels::GgmlDType::Q6K
            );
        let out_dtype = if src1_bf16 { DType::BF16 } else { DType::F32 };
        let dst = device.new_buffer(dst_shape.elem_count(), out_dtype, "qmatmul")?;
        let encoder = device.command_encoder()?;

        if !src1_bf16 {
            assert_eq!(storage.dtype(), DType::F32);
        }

        if self_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", self_shape.rank())
        }
        let src0_l = crate::Layout::contiguous(
            [vec![1; 4 - self_shape.rank()], self_shape.dims().to_vec()].concat(),
        );
        let src0_stride = src0_l
            .stride()
            .iter()
            .map(|x| {
                (*x as f32 * (self.dtype.type_size() as f32 / self.dtype.block_size() as f32))
                    as usize
            })
            .collect::<Vec<_>>();

        if src_shape.rank() > 4 {
            crate::bail!("weight rank ({}) must be <= 4", src_shape.rank())
        }
        let src1_l = crate::Layout::contiguous(
            [vec![1; 4 - src_shape.rank()], src_shape.dims().to_vec()].concat(),
        );

        hanzo_metal_kernels::call_quantized_matmul_mm_t(
            device.device(),
            &encoder,
            device.kernels(),
            kdtype,
            src0_l.dims(),
            &src0_stride,
            &self.buffer,
            src1_l.dims(),
            &src1_l
                .stride()
                .iter()
                .map(|x| x * storage.dtype().size_in_bytes())
                .collect::<Vec<_>>(),
            storage.buffer(),
            src1_l.start_offset() * storage.dtype().size_in_bytes(),
            dst_shape.dims(),
            0,
            &dst,
            src1_bf16,
        )
        .map_err(MetalError::from)?;

        let dst_storage =
            crate::MetalStorage::new(dst, device.clone(), dst_shape.elem_count(), out_dtype);
        Ok((dst_storage, dst_shape))
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let buffer = self.device.allocate_buffer(self.buffer.length())?;
        {
            let mut blit = self.device.blit_command_encoder()?;
            blit.set_label("blit_to_cpu");
            blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        }
        self.device.flush_and_wait_current()?;
        Ok(read_to_vec::<u8>(&buffer, self.storage_size_in_bytes()))
    }

    /// One expert's quantized weights `[n, k]` times `x` `[m, k]` (contiguous f32), read straight out
    /// of the resident `[E, n, k]` bank at byte offset `weight_offset` -- no dequant, no copy.
    /// Returns `[m, n]` f32. Decode (m==1) uses the matvec kernel (same as `fwd_mv`); prefill (m>1)
    /// uses the matmul kernel (same as `fwd`), each just offset into the bank by `weight_offset`.
    fn moe_expert_matmul(
        &self,
        x: &MetalStorage,
        weight_offset: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<MetalStorage> {
        use crate::MetalError;
        let device = self.device.clone();
        let dst = device.new_buffer(m * n, DType::F32, "moe_expert_matmul")?;
        let dtype: hanzo_metal_kernels::GgmlDType = self.dtype.try_into()?;
        let encoder = device.command_encoder()?;
        if m == 1 {
            hanzo_metal_kernels::call_quantized_matmul_mv_t_offset(
                device.device(),
                &encoder,
                device.kernels(),
                dtype,
                (1, 1, n, k),
                x.buffer(),
                0,
                &self.buffer,
                weight_offset,
                0,
                &dst,
                false,
            )
            .map_err(MetalError::from)?;
        } else {
            // src0 = weight [n, k], src1 = x [m, k] (both 4D-padded, contiguous), as in `fwd`.
            let bs = self.dtype.block_size() as f32;
            let ts = self.dtype.type_size() as f32;
            let w_l = crate::Layout::contiguous(&[1, 1, n, k]);
            let w_stride = w_l
                .stride()
                .iter()
                .map(|x| (*x as f32 * (ts / bs)) as usize)
                .collect::<Vec<_>>();
            let x_l = crate::Layout::contiguous(&[1, 1, m, k]);
            let x_stride = x_l
                .stride()
                .iter()
                .map(|x| x * DType::F32.size_in_bytes())
                .collect::<Vec<_>>();
            hanzo_metal_kernels::call_quantized_matmul_mm_t_offset(
                device.device(),
                &encoder,
                device.kernels(),
                dtype,
                w_l.dims(),
                &w_stride,
                &self.buffer,
                weight_offset,
                x_l.dims(),
                &x_stride,
                x.buffer(),
                0,
                &[1, 1, m, n],
                0,
                &dst,
                false,
            )
            .map_err(MetalError::from)?;
        }
        drop(encoder); // dbc-validation: release CommandsGuard borrow of `device` before move
        Ok(MetalStorage::new(dst, device, m * n, DType::F32))
    }

    /// Keep-quantized indexed MoE forward, mirroring the CUDA fused path but on Metal's unified
    /// memory: the `[E, n, k]` GGUF expert bank stays quantized in `self.buffer` (no whole-bank
    /// dequant, the multi-GB f32 OOM), and each routed expert's matvec reads its slice by byte
    /// offset via the native quant matvec kernel. Slots are grouped by expert host-side (router ids
    /// are tiny), gathered, run, and scattered back -- cost scales with active experts, not E.
    pub fn indexed_moe_forward(
        &self,
        self_shape: &Shape,   // [num_experts, n, k]
        input: &MetalStorage, // [t, 1 (gate/up) or topk (down), k]
        input_l: &crate::Layout,
        ids: &MetalStorage, // [t, topk]
        ids_l: &crate::Layout,
    ) -> Result<(MetalStorage, Shape)> {
        // Fused ggml `mul_mv_id` (decode) / `mul_mm_id` (prefill): ONE dispatch computes every routed
        // (token, expert-slot) product, reading the expert id per row from the on-device `ids` buffer.
        // Replaces the per-expert host loop -- which did an `ids.to_vec1` device->host sync plus a
        // gather/matmul/scatter per expert, per projection, per layer. The [E, n, k] GGUF bank stays
        // quantized and resident in `self.buffer`; each expert block is `expert_bytes` apart.
        let device = self.device.clone();
        let (e_cnt, n, k) = self_shape.dims3()?;
        let (t, topk) = ids_l.shape().dims2()?;
        let s = input_l.shape().dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)

        // Mirrors the CUDA indexed-MoE contract: inputs arrive contiguous at offset 0.
        if !input_l.is_contiguous() || input_l.start_offset() != 0 {
            crate::bail!("indexed_moe_forward: input must be contiguous at offset 0");
        }
        if !ids_l.is_contiguous() || ids_l.start_offset() != 0 {
            crate::bail!("indexed_moe_forward: ids must be contiguous at offset 0");
        }

        let block_size = self.dtype.block_size();
        let type_size = self.dtype.type_size();
        if !k.is_multiple_of(block_size) {
            crate::bail!("indexed_moe_forward: k {k} not a multiple of block size {block_size}");
        }
        let expert_bytes = n * (k / block_size) * type_size;
        // Metal setBuffer:offset: needs a 4-byte-aligned offset; holds for every GGML block size at
        // even n, which real weight dims always are. Fail loudly otherwise.
        if !expert_bytes.is_multiple_of(4) {
            crate::bail!(
                "indexed_moe_forward: expert stride {expert_bytes} bytes not 4-byte aligned"
            );
        }

        let none = crate::op::BackpropOp::none();
        // src1: [t, s, k] contiguous f32, resident on the Metal device.
        let input_t = crate::tensor::from_storage(
            crate::Storage::Metal(input.clone()),
            input_l.shape().clone(),
            none.clone(),
            false,
        );
        let x_f32 = input_t.to_dtype(DType::F32)?.contiguous()?;

        // ids: [t, topk] contiguous u32, resident on the Metal device (read on-GPU by the kernel).
        let ids_t = crate::tensor::from_storage(
            crate::Storage::Metal(ids.clone()),
            ids_l.shape().clone(),
            none,
            false,
        );
        let ids_u32 = ids_t.to_dtype(DType::U32)?.contiguous()?;

        let dst = device.new_buffer(t * topk * n, DType::F32, "moe_mv_id")?;
        let kdtype: hanzo_metal_kernels::GgmlDType = self.dtype.try_into()?;

        let (x_store, _) = x_f32.storage_and_layout();
        let x_metal = match &*x_store {
            crate::Storage::Metal(st) => st,
            _ => crate::bail!("indexed_moe_forward: x not on metal after contiguous()"),
        };
        let (ids_store, _) = ids_u32.storage_and_layout();
        let ids_metal = match &*ids_store {
            crate::Storage::Metal(st) => st,
            _ => crate::bail!("indexed_moe_forward: ids not on metal after contiguous()"),
        };

        {
            let encoder = device.command_encoder()?;
            // Prefill (t>1) rides the fused expert-grouped GEMM (mul_mm_id): each expert's weight is
            // read once and amortized over its routed tokens. Decode (t==1) keeps the per-slot matvec
            // (mul_mv_id). Identical arg block and [t,topk,n] output, so only the kernel differs.
            if t > 1 {
                hanzo_metal_kernels::call_mul_mm_id(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    kdtype,
                    e_cnt,
                    n,
                    k,
                    t,
                    topk,
                    s,
                    expert_bytes,
                    &self.buffer,
                    0,
                    x_metal.buffer(),
                    0,
                    ids_metal.buffer(),
                    0,
                    &dst,
                    0,
                )
            } else {
                hanzo_metal_kernels::call_mul_mv_id(
                    device.device(),
                    &encoder,
                    device.kernels(),
                    kdtype,
                    e_cnt,
                    n,
                    k,
                    t,
                    topk,
                    s,
                    expert_bytes,
                    &self.buffer,
                    0,
                    x_metal.buffer(),
                    0,
                    ids_metal.buffer(),
                    0,
                    &dst,
                    0,
                )
            }
            .map_err(crate::MetalError::from)?;
        }

        let out_storage = MetalStorage::new(dst, device, t * topk * n, DType::F32);
        Ok((out_storage, (t, topk, n).into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &MetalDevice,
    data: &[T],
) -> Result<QStorage> {
    let buffer = device.new_buffer_with_data(data)?;
    let device = device.clone();
    Ok(QStorage::Metal(QMetalStorage {
        dtype: T::DTYPE,
        device,
        buffer,
    }))
}

fn read_to_vec<T: Clone>(buffer: &Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

// Fallible: any ggml dtype without a Metal kernel returns an error rather than panicking. A
// panicking `From` is a footgun -- a caller that hits an unmapped quant (e.g. a future MXFP4/IQ4
// bank, or ISQ with an unexpected type) should surface a clean error, not abort the process.
impl TryFrom<GgmlDType> for hanzo_metal_kernels::GgmlDType {
    type Error = crate::Error;

    fn try_from(value: GgmlDType) -> Result<Self> {
        let dt = match value {
            GgmlDType::Q4_0 => hanzo_metal_kernels::GgmlDType::Q4_0,
            GgmlDType::Q4_1 => hanzo_metal_kernels::GgmlDType::Q4_1,
            GgmlDType::Q5_0 => hanzo_metal_kernels::GgmlDType::Q5_0,
            GgmlDType::Q5_1 => hanzo_metal_kernels::GgmlDType::Q5_1,
            GgmlDType::Q8_0 => hanzo_metal_kernels::GgmlDType::Q8_0,
            GgmlDType::Q8_1 => hanzo_metal_kernels::GgmlDType::Q8_1,
            GgmlDType::Q2K => hanzo_metal_kernels::GgmlDType::Q2K,
            GgmlDType::Q3K => hanzo_metal_kernels::GgmlDType::Q3K,
            GgmlDType::Q4K => hanzo_metal_kernels::GgmlDType::Q4K,
            GgmlDType::Q5K => hanzo_metal_kernels::GgmlDType::Q5K,
            GgmlDType::Q6K => hanzo_metal_kernels::GgmlDType::Q6K,
            GgmlDType::Q8K => hanzo_metal_kernels::GgmlDType::Q8K,
            GgmlDType::F16 => hanzo_metal_kernels::GgmlDType::F16,
            GgmlDType::F32 => hanzo_metal_kernels::GgmlDType::F32,
            GgmlDType::BF16 => hanzo_metal_kernels::GgmlDType::BF16,
            // i-quant codebook family -- native MSL matvec/matmul/mul_mv_id kernels.
            GgmlDType::IQ2_XXS => hanzo_metal_kernels::GgmlDType::IQ2_XXS,
            GgmlDType::IQ2_XS => hanzo_metal_kernels::GgmlDType::IQ2_XS,
            GgmlDType::IQ2_S => hanzo_metal_kernels::GgmlDType::IQ2_S,
            GgmlDType::IQ3_XXS => hanzo_metal_kernels::GgmlDType::IQ3_XXS,
            GgmlDType::IQ3_S => hanzo_metal_kernels::GgmlDType::IQ3_S,
            GgmlDType::IQ1_S => hanzo_metal_kernels::GgmlDType::IQ1_S,
            GgmlDType::IQ1_M => hanzo_metal_kernels::GgmlDType::IQ1_M,
            GgmlDType::IQ4_NL => hanzo_metal_kernels::GgmlDType::IQ4_NL,
            GgmlDType::IQ4_XS => hanzo_metal_kernels::GgmlDType::IQ4_XS,
            #[allow(unreachable_patterns)]
            other => crate::bail!("no Metal quantized kernel for dtype {other:?}"),
        };
        Ok(dt)
    }
}
