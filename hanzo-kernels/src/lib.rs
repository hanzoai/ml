mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Attention,
    Backward,
    Binary,
    Cast,
    Conv,
    Fill,
    Glu,
    Indexing,
    IquantMmvq,
    Optim,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 16] = [
    Id::Affine,
    Id::Attention,
    Id::Backward,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::Fill,
    Id::Glu,
    Id::Indexing,
    Id::IquantMmvq,
    Id::Optim,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(ATTENTION, Attention);
mdl!(BACKWARD, Backward);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(FILL, Fill);
mdl!(GLU, Glu);
mdl!(INDEXING, Indexing);
mdl!(IQUANT_MMVQ, IquantMmvq);
mdl!(OPTIM, Optim);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);

pub mod ffi;
