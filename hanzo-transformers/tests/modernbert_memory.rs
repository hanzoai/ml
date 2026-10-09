//! A long row's padded pass holds no `[L, L]` tensor of any type, and answers as the packed pass,
//! whose attention is another algorithm (`hanzo_nn::attention::packed`). Its own binary, so the
//! allocator counts this pass alone.

use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{VarBuilder, VarMap};
use hanzo_transformers::models::modernbert::{Config, ModernBert};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// The system allocator, counting the bytes held, their peak and the largest block.
struct Counting;

static HELD: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static LARGEST: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let held = HELD.fetch_add(l.size(), Relaxed) + l.size();
            PEAK.fetch_max(held, Relaxed);
            LARGEST.fetch_max(l.size(), Relaxed);
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        HELD.fetch_sub(l.size(), Relaxed);
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

#[test]
fn a_long_row_holds_no_square_tensor() -> Result<()> {
    let dev = Device::Cpu;
    // two heads, 8,192 trained positions stretched to 32,768: a row past 5,792 tokens tiles
    let cfg = Config {
        vocab_size: 50,
        hidden_size: 64,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        intermediate_size: 64,
        max_position_embeddings: 8192,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160000.,
        local_attention: 8,
        local_rope_theta: 10000.,
        yarn: Some(4.0),
        classifier_config: None,
    };
    // weights apart from their variables, as inference reads them: no pass keeps its graph
    let vm = VarMap::new();
    ModernBert::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), &cfg)?;
    let weights = vm
        .data()
        .lock()
        .expect("varmap lock")
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().detach()))
        .collect();
    let bert = ModernBert::new(VarBuilder::from_tensors(weights, DType::F32, &dev), &cfg)?;
    let l = 20_480;
    let lens = [l, 9_000];
    let ids: Vec<u32> = lens
        .iter()
        .enumerate()
        .flat_map(|(r, &n)| (0..l).map(move |t| if t < n { 1 + ((r * 13 + t * 7) % 49) as u32 } else { 0 }))
        .collect();
    let mask: Vec<u32> = lens.iter().flat_map(|&n| (0..l).map(move |t| u32::from(t < n))).collect();
    let ids = Tensor::from_vec(ids, (2, l), &dev)?;
    let mask = Tensor::from_vec(mask, (2, l), &dev)?;

    let base = HELD.load(Relaxed);
    PEAK.store(base, Relaxed);
    LARGEST.store(0, Relaxed);
    let want = bert.forward(&ids, &mask)?;
    let (peak, largest) = (PEAK.load(Relaxed) - base, LARGEST.load(Relaxed));
    // whole, a global layer's scores alone are [2, 2, L, L] in F32, 6.7 GB
    assert!(largest < l * l, "a block of {largest} bytes, {l} x {l} is {}", l * l);
    assert!(peak < 4 * l * l, "{peak} bytes held at once");

    let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let want = want.broadcast_mul(&keep)?;
    let mut packs = bert.clone();
    packs.quantize(GgmlDType::F32)?;
    assert!(packs.packs());
    let got = packs.forward(&ids, &mask)?;
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
    let gap = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
    // over 16,384 keys the two algorithms' float sums part by 1.6e-5 of the scale, the padded
    // pass's scores whole or tiled alike (whole and tiled part by 5e-7)
    assert!(gap <= 5e-5 * scale, "packed {gap} against {scale}");
    Ok(())
}
