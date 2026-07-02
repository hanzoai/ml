use core::ffi::c_void;
#[allow(dead_code)]
#[allow(improper_ctypes)]
extern "C" {
    // Fused F32 head_dim-512 online-softmax flash-decode attention (from fattn_ds4.cu).
    // q/out are [n_head, 512]; k/v are [n_kv_head, kv_len, 512] (n_kv_head == 1 => plain
    // [kv_len, 512], the V4 MQA case). `sinks` ([n_head]) may be null. `window` == 0
    // disables the sliding window; otherwise the query (newest token) attends the last
    // `window` KV rows. `scale` is applied to the QK dot (e.g. 1/sqrt(512)).
    pub fn hanzo_fattn_decode_f32_hd512(
        stream: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        sinks: *const f32,
        out: *mut f32,
        n_head: i32,
        n_kv_head: i32,
        kv_len: i32,
        window: i32,
        scale: f32,
    );

    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5  (for weights)
        stream: i64,
    );

    // ============== Dense GGUF MMVQ launchers (from mmvq_gguf.cu) ==============

    // BF16 output launchers
    pub fn launch_mmvq_gguf_q4_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // F32 output launchers
    pub fn launch_mmvq_gguf_q4_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    pub fn launch_mmvq_gguf_q4_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    // Quantize launchers (activation → Q8_1)
    pub fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    // ============== Dense GGUF MMQ launchers (from mmq_gguf/) ==============

    // MMQ quantize launchers (f32 -> block_q8_1_mmq, 3 scale layouts)
    pub fn launch_mmq_quantize_q8_1_D4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_DS4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_D2S6(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );

    // MMQ matmul launchers (one per quant type)
    pub fn launch_mmq_gguf_q4_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q8_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q2_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q3_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q6_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    // i-quant codebook dense MMQ prefill launchers (int8 WMMA via load_tiles_iq*). Same signature as
    // the K-quant launchers above; defined by the weak launch_mmq_gguf_<iq*> in each mmq_instance_iq*.cu.
    pub fn launch_mmq_gguf_iq2_xxs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_iq2_xs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_iq2_s(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_iq3_xxs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_iq3_s(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_iq4_xs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    // IQ1_S (1-bit grid + delta): DS4 ds-layout (d + delta*sum) -- distinct GPU-packed iq1s_grid_gpu.
    pub fn launch_mmq_gguf_iq1_s(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_y: i64, stride_row_x: i64, stride_col_dst: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );

    // Expert-grouped MoE prefill launchers (llama mul_mat_id). Same int8 MMQ core as the dense
    // launchers above; experts ride the channel dim. `ids_dst[col]` scatters sorted column `col` to its
    // original token row, `expert_bounds[e]` (len num_experts+1) bound each expert's sorted columns, and
    // `stride_channel_x` steps one expert's weight bank. Output dst is [ncols_dst, nrows_x]. Defined by
    // DEFINE_MMQ_GGUF_MOE_LAUNCHER in each mmq_instance_*.cu.
    pub fn launch_mmq_gguf_moe_q4_0(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q4_1(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q5_0(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q5_1(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q8_0(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q2_k(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q3_k(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q4_k(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q5_k(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_q6_k(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    // i-quant codebook MoE MMQ prefill launchers (DEFINE_MMQ_GGUF_MOE_LAUNCHER in mmq_instance_iq*.cu).
    pub fn launch_mmq_gguf_moe_iq2_xxs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_iq2_xs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_iq2_s(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_iq3_xxs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_iq3_s(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_iq4_xs(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_moe_iq1_s(
        tmp_fixup: *mut c_void, x: *const c_void, y: *const c_void,
        ids_dst: *const c_void, expert_bounds: *const c_void, dst: *mut c_void,
        ncols_x: i64, nrows_x: i64, ncols_dst: i64, ncols_max: i64,
        stride_row_x: i64, stride_channel_x: i64, n_experts: i64,
        cc: i32, nsm: i32, smpbo: i64, warp_size: i32, stream: *mut c_void,
    );
}
