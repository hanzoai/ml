//! CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
//! Handles batch > 8 (complement to fast_mmvq which handles batch 1-8).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::cuda::{QCudaStorage, MATRIX_ROW_PADDING};
use super::GgmlDType;
use crate::cuda_backend::DeviceId;
use crate::{backend::BackendStorage, CudaDevice, CudaStorage, DType, Result, Shape};

use cudarc::driver::{CudaSlice, CudaView, DevicePtr};

const QK8_1: usize = 32;
const BLOCK_Q8_1_MMQ_SIZE: usize = 4 * QK8_1 + 4 * 4; // 128 qs + 16 scale bytes = 144

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

/// Quant types with an int8-WMMA MMQ prefill kernel: the 10 K-quants + the 5 IQ2/IQ3 codebooks whose
/// `load_tiles_iq*` read the standard ggml codebook grids (filled from iq_grids.rs in mmq_common.cuh).
/// IQ1_S/IQ1_M (distinct GPU-packed iq1s_grid_gpu) + IQ1_M (no kernel) keep dequant prefill -- they
/// still have native dp4a DECODE.
fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::IQ2_XXS
            | GgmlDType::IQ2_XS
            | GgmlDType::IQ2_S
            | GgmlDType::IQ3_XXS
            | GgmlDType::IQ3_S
            | GgmlDType::IQ4_XS
    )
}

/// qk (block quantization size) per dtype.
fn qk_for(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            32
        }
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S
        | GgmlDType::IQ4_XS => 256,
        _ => unreachable!(),
    }
}

// ds_layout mapping: which Q8_1_mmq scale layout to use per weight type.
enum DsLayout {
    D4,
    DS4,
    D2S6,
}

fn ds_layout_for(dtype: GgmlDType) -> DsLayout {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 => DsLayout::DS4,
        GgmlDType::Q5_0 => DsLayout::D4,
        GgmlDType::Q5_1 => DsLayout::DS4,
        GgmlDType::Q8_0 => DsLayout::D4,
        GgmlDType::Q2K => DsLayout::D2S6,
        GgmlDType::Q3K => DsLayout::D4,
        GgmlDType::Q4K | GgmlDType::Q5K => DsLayout::DS4,
        GgmlDType::Q6K => DsLayout::D4,
        // i-quant codebooks (mmq_get_q8_1_ds_layout): IQ2/IQ3 symmetric -> D4.
        GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S
        | GgmlDType::IQ4_XS => DsLayout::D4,
        _ => unreachable!(),
    }
}

type QuantizeLauncher = unsafe extern "C" fn(
    x: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut std::ffi::c_void,
);

fn quantize_launcher(layout: DsLayout) -> QuantizeLauncher {
    use hanzo_kernels::ffi;
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

type MmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    stream: *mut std::ffi::c_void,
);

fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
    use hanzo_kernels::ffi;
    let f: MmqLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k,
        GgmlDType::IQ2_XXS => ffi::launch_mmq_gguf_iq2_xxs,
        GgmlDType::IQ2_XS => ffi::launch_mmq_gguf_iq2_xs,
        GgmlDType::IQ2_S => ffi::launch_mmq_gguf_iq2_s,
        GgmlDType::IQ3_XXS => ffi::launch_mmq_gguf_iq3_xxs,
        GgmlDType::IQ3_S => ffi::launch_mmq_gguf_iq3_s,
        GgmlDType::IQ4_XS => ffi::launch_mmq_gguf_iq4_xs,
        _ => return None,
    };
    Some(f)
}

// ---------------------------------------------------------------------------
// Per-device workspaces (grows-only, reused across calls).
// ---------------------------------------------------------------------------

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

type WsMap = Mutex<HashMap<DeviceId, &'static Mutex<WorkspaceSlot>>>;

static MMQ_WORKSPACE: OnceLock<WsMap> = OnceLock::new();
static FIXUP_WORKSPACE: OnceLock<WsMap> = OnceLock::new();

fn workspace_ensure(
    ws: &'static OnceLock<WsMap>,
    dev: &CudaDevice,
    bytes: usize,
) -> Result<(u64, std::sync::MutexGuard<'static, WorkspaceSlot>)> {
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot {
                    slice,
                    cap: bytes.max(1),
                })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < bytes {
        slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.cap = bytes;
    }
    let ptr = slot.slice.device_ptr(slot.slice.stream()).0;
    Ok((ptr, slot))
}

// ---------------------------------------------------------------------------
// Per-device info cache (compute capability, SM count, etc.)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct DeviceInfo {
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

static DEVICE_INFO: OnceLock<Mutex<HashMap<DeviceId, DeviceInfo>>> = OnceLock::new();

fn get_device_info(dev: &CudaDevice) -> DeviceInfo {
    use cudarc::driver::{result, sys};
    let map = DEVICE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    let key = dev.id();
    let mut guard = map.lock().unwrap();
    if let Some(info) = guard.get(&key) {
        return *info;
    }
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    .unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
    }
    .unwrap_or(32);
    let info = DeviceInfo {
        cc: major * 100 + minor * 10,
        nsm,
        smpbo: smpbo as i64,
        warp_size,
    };
    guard.insert(key, info);
    info
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Try the fast MMQ path. Returns `Ok(None)` when the fast path is not applicable:
/// - unsupported quant dtype
/// - batch size 0
/// - non-BF16/F16/F32 input
/// - non-contiguous input
pub fn try_fwd(
    qstorage: &QCudaStorage,
    self_shape: &Shape,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
) -> Result<Option<(CudaStorage, Shape)>> {
    let w_dtype = qstorage.dtype();
    if !supports(w_dtype) {
        return Ok(None);
    }
    let input_dtype = rhs.dtype();
    if !matches!(input_dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }

    let (nrows, ncols) = self_shape.dims2()?;

    let (b_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => return Ok(None),
    };
    if ncols != k {
        return Ok(None);
    }
    if b_size == 0 {
        return Ok(None);
    }

    let qk = qk_for(w_dtype);
    if k % qk != 0 {
        return Ok(None);
    }

    let (o1, o2) = match rhs_l.contiguous_offsets() {
        Some(offsets) => offsets,
        None => return Ok(None),
    };

    let dev = qstorage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;

    // MMQ quantize expects f32 input. Convert if needed.
    let rhs_f32_storage: Option<CudaStorage> = if input_dtype != DType::F32 {
        let layout_for_cast =
            crate::Layout::contiguous_with_offset(rhs_l.shape(), rhs_l.start_offset());
        Some(rhs.to_dtype(&layout_for_cast, DType::F32)?)
    } else {
        None
    };

    // Get the f32 slice (either original or converted)
    let rhs_f32_ref = rhs_f32_storage.as_ref().unwrap_or(rhs);
    let rhs_f32_slice = rhs_f32_ref.as_cuda_slice::<f32>()?;
    // For the converted case, offset is 0 (to_dtype returns contiguous from 0).
    // For the original f32 case, apply the original offsets.
    let rhs_f32_slice = if rhs_f32_storage.is_some() {
        rhs_f32_slice.slice(..b_size * k)
    } else {
        rhs_f32_slice.slice(o1..o2)
    };
    let rhs_ptr = rhs_f32_slice.device_ptr(&stream).0 as *const std::ffi::c_void;

    // Compute padded dimensions
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Must also be multiple of 4*QK8_1 = 128 for block_q8_1_mmq
    let k_padded = pad(k_padded, 4 * QK8_1);

    // Workspace for block_q8_1_mmq quantized activations
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = b_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    // Extra padding for mmq_x_max (128 for MMA path)
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;

    let (scratch_ptr, _workspace_guard) = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    // Stream-k fixup workspace
    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let (fixup_ptr, _fixup_guard) = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes)?;
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let weight_ptr = qstorage.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let di = get_device_info(dev);

    let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    let stride_col_dst = nrows as i64;

    let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

    unsafe {
        let quantize = quantize_launcher(ds_layout_for(w_dtype));
        quantize(
            rhs_ptr,
            std::ptr::null(),
            scratch_ptr,
            0,
            k as i64,
            k as i64,
            0,
            0,
            k_padded as i64,
            b_size as i64,
            1,
            1,
            stream_ptr,
        );

        let launcher = mmq_launcher(w_dtype).expect("supports() checked");
        launcher(
            fixup_ptr,
            weight_ptr,
            scratch_ptr as *const std::ffi::c_void,
            out_ptr,
            k as i64,
            nrows as i64,
            b_size as i64,
            stride_row_x,
            stride_col_dst,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warp_size,
            stream_ptr,
        );
    }

    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);

    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());

    if input_dtype == DType::F32 {
        Ok(Some((out_storage, out_shape.into())))
    } else {
        // Cast f32 output back to input dtype
        let out_layout = crate::Layout::contiguous(Shape::from(out_shape.clone()));
        let cast_storage = out_storage.to_dtype(&out_layout, input_dtype)?;
        Ok(Some((cast_storage, out_shape.into())))
    }
}

// ---------------------------------------------------------------------------
// Expert-grouped MoE prefill GEMM (llama's mul_mat_id).
//
// The decode/single-token MoE keeps the per-slot matvec in `cuda.rs` (bandwidth-bound, fine). This is
// the prefill (rows>1) lever: instead of one matvec per (token, expert-slot) -- which re-streams the
// whole expert weight for every routed token and runs no matrix cores -- we group the routed tokens by
// expert (host-side counting sort over the router ids) and run ONE int8 MMQ where each expert's weight
// is staged once and amortized over all its tokens via the tensor cores. Same `mul_mat_q` core as the
// dense path; experts ride the channel dim (see DEFINE_MMQ_GGUF_MOE_LAUNCHER).
// ---------------------------------------------------------------------------

type MoeMmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    ids_dst: *const std::ffi::c_void,
    expert_bounds: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    ncols_max: i64,
    stride_row_x: i64,
    stride_channel_x: i64,
    n_experts: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    stream: *mut std::ffi::c_void,
);

fn moe_mmq_launcher(dtype: GgmlDType) -> Option<MoeMmqLauncher> {
    use hanzo_kernels::ffi;
    let f: MoeMmqLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_moe_q4_0,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_moe_q4_1,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_moe_q5_0,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_moe_q5_1,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_moe_q8_0,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_moe_q2_k,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_moe_q3_k,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_moe_q4_k,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_moe_q5_k,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_moe_q6_k,
        GgmlDType::IQ2_XXS => ffi::launch_mmq_gguf_moe_iq2_xxs,
        GgmlDType::IQ2_XS => ffi::launch_mmq_gguf_moe_iq2_xs,
        GgmlDType::IQ2_S => ffi::launch_mmq_gguf_moe_iq2_s,
        GgmlDType::IQ3_XXS => ffi::launch_mmq_gguf_moe_iq3_xxs,
        GgmlDType::IQ3_S => ffi::launch_mmq_gguf_moe_iq3_s,
        GgmlDType::IQ4_XS => ffi::launch_mmq_gguf_moe_iq4_xs,
        _ => return None,
    };
    Some(f)
}

/// Expert-grouped MoE prefill GEMM. Mirrors the shape contract of
/// `cuda::indexed_moe_forward_fused_q8_1_input` but runs ONE int8 MMQ (llama mul_mat_id) instead of the
/// per-slot matvec. Returns `Ok(None)` when the weight dtype has no MMQ kernel (caller falls back to the
/// per-slot path). `input` is read from offset 0 -- identical bytes to the per-slot path, so the A/B is
/// apples-to-apples. Output is f32 `[batch, topk, n]`.
pub(crate) fn indexed_moe_grouped(
    weight: &CudaView<u8>,
    w_shape: &Shape, // [E, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &Shape, // [batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &Shape, // [batch, topk] or flat [batch*topk]
    dev: &CudaDevice,
) -> Result<Option<(CudaStorage, Shape)>> {
    if !supports(w_dtype) {
        return Ok(None);
    }
    let (e_cnt, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];
    let nslots = idx_shape.elem_count();
    if nslots == 0 || batch == 0 || nslots % batch != 0 {
        return Ok(None);
    }
    let topk = nslots / batch;

    let qk = qk_for(w_dtype);
    if k % qk != 0 {
        return Ok(None);
    }

    // Host-side counting sort of the routed slots by expert (llama mul_mat_id). Prefill is never
    // graph-captured, so the ids DtoH + the metadata HtoD below are free.
    let ids_host: Vec<u32> = dev.clone_dtoh(ids)?;
    if ids_host.len() != nslots {
        return Ok(None);
    }
    let mut counts = vec![0i32; e_cnt];
    for &e in &ids_host {
        let e = e as usize;
        if e >= e_cnt {
            crate::bail!("indexed_moe_grouped: expert id {e} >= num_experts {e_cnt}");
        }
        counts[e] += 1;
    }
    // expert_bounds[e] = prefix sums over counts [e_cnt+1]; expert e owns sorted columns [eb[e],eb[e+1]).
    let mut expert_bounds = vec![0i32; e_cnt + 1];
    for e in 0..e_cnt {
        expert_bounds[e + 1] = expert_bounds[e] + counts[e];
    }
    // For each sorted column p: ids_dst[p] = original output row = the routed slot s (= token*topk+slot,
    // exactly the row of the [batch,topk,n] output); quantize_ids[p] = input row to gather (the token
    // when the input is broadcast across topk, i.e. input_dim1==1; else the slot itself).
    let mut ids_dst = vec![0i32; nslots];
    let mut quantize_ids = vec![0i32; nslots];
    let mut cursor = expert_bounds.clone();
    for (s, &e) in ids_host.iter().enumerate() {
        let e = e as usize;
        let p = cursor[e] as usize;
        ids_dst[p] = s as i32;
        quantize_ids[p] = if input_dim1 == 1 { (s / topk) as i32 } else { s as i32 };
        cursor[e] += 1;
    }
    let ncols_max = counts.iter().copied().max().unwrap_or(0) as i64;

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;

    // Padded k for the q8_1_mmq activation (matches the dense MMQ path exactly).
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);
    let blocks_per_row = k_padded / (4 * QK8_1);

    // Workspace for the expert-sorted q8_1_mmq activations (one column per routed slot).
    let workspace_main = nslots * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let (scratch_ptr, _ws_guard) =
        workspace_ensure(&MMQ_WORKSPACE, dev, workspace_main + workspace_extra)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    // Stream-k fixup workspace (same sizing as the dense path).
    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let (fixup_ptr, _fixup_guard) = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes)?;
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    // Upload routing metadata (kept alive until the launches are enqueued on the stream).
    let eb_dev = dev.clone_htod(&expert_bounds)?;
    let iddst_dev = dev.clone_htod(&ids_dst)?;
    let qids_dev = dev.clone_htod(&quantize_ids)?;

    let input_ptr = input.device_ptr(&stream).0 as *const std::ffi::c_void;
    let eb_ptr = eb_dev.device_ptr(&stream).0 as *const std::ffi::c_void;
    let iddst_ptr = iddst_dev.device_ptr(&stream).0 as *const std::ffi::c_void;
    let qids_ptr = qids_dev.device_ptr(&stream).0 as *const i32;
    let weight_ptr = weight.device_ptr(&stream).0 as *const std::ffi::c_void;

    let out = dev.alloc_zeros::<f32>(nslots * n)?;
    let out_ptr = out.device_ptr(&stream).0 as *mut std::ffi::c_void;

    let di = get_device_info(dev);
    let stride_row_x = (k / qk) as i64;
    let stride_channel_x = (n * (k / qk)) as i64;

    unsafe {
        // Quantize the activation into expert-sorted q8_1_mmq columns (gather input row by quantize_ids).
        let quantize = quantize_launcher(ds_layout_for(w_dtype));
        quantize(
            input_ptr,
            qids_ptr,
            scratch_ptr,
            0,
            k as i64,        // ne00 (real row length)
            k as i64,        // s01  (input row stride, elements)
            0,
            0,
            k_padded as i64, // ne0  (padded loop bound)
            nslots as i64,   // ne1  (number of sorted columns)
            1,
            1,
            stream_ptr,
        );

        let launcher = moe_mmq_launcher(w_dtype).expect("supports() checked");
        launcher(
            fixup_ptr,
            weight_ptr,
            scratch_ptr as *const std::ffi::c_void,
            iddst_ptr,
            eb_ptr,
            out_ptr,
            k as i64,      // ncols_x
            n as i64,      // nrows_x
            nslots as i64, // ncols_dst (total routed columns)
            ncols_max,     // ncols_max (largest per-expert column count)
            stride_row_x,
            stride_channel_x,
            e_cnt as i64,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warp_size,
            stream_ptr,
        );
    }

    // eb_dev/iddst_dev/qids_dev free stream-ordered on drop here (after the launches are enqueued).
    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok(Some((out_storage, out_shape.into())))
}
