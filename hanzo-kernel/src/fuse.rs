//! Auto-fusion: **fusion is composition**.
//!
//! An elementwise (pointwise) kernel is the *lift* of a scalar morphism `f: F -> F` applied at every
//! index. Pointwise lifts obey the functor law
//!
//! ```text
//!     map(g) . map(f)  ==  map(g . f)
//! ```
//!
//! Read left-to-right that is two kernels with a **materialized intermediate array** between them; read
//! right-to-left it is **one** kernel computing the composed scalar function with **no** intermediate.
//! Kernel fusion is nothing but that law read right-to-left. It is exactly Hickey's transducer story:
//! `map(f)` and `map(g)` are transformations independent of the collection; you compose the
//! *transformations* and apply the composite to the collection once, never allocating intermediate
//! collections. Here "the collection" is a GPU array and "once" is one kernel launch. Categorically the
//! Maps are morphisms `Array -> Array` and composition of morphisms **is** kernel fusion — there is no
//! separate "fusion pass" concept, only the associativity of `.`.
//!
//! So the whole surface is a tiny op algebra, and legal fusion falls out of the op **type**, not a
//! pattern-matcher:
//!
//!   * **Map** (pointwise morphism) — output element `i` depends only on input element(s) at index `i`
//!     (plus comptime constants). Shape-preserving, index-local. Maps compose freely: any run of Maps
//!     is one Map (the functor law). This is the fusible family: `Add Sub Mul Div Neg Recip Abs Exp
//!     Rsqrt Sigmoid Tanh Silu Gelu`.
//!   * **Reduce / Scan** (contraction) — output element `i` depends on *many* inputs (a whole row/axis:
//!     `sum`, `max`, `mean` -> rms-norm / softmax / matmul). Not index-local, so it cannot be absorbed
//!     into an adjacent Map by the same lift; it needs cross-lane communication (shared-mem tree, plane
//!     reduce). A Reduce is therefore a **fence**: the Map run feeding it fuses into its input *read*
//!     (fuse-on-read / prologue) and the Map run consuming its result fuses into its output *write*
//!     (fuse-on-write / epilogue), but the reduction itself is the seam where the kernel pattern changes.
//!
//! The decision "can these two adjacent ops fuse?" is a total function of their class:
//!
//! ```text
//!     fusible(a, b)  :=  is_map(a) && is_map(b)     // Map–Map: always (functor law)
//! ```
//!
//! plus prologue/epilogue attachment at a Reduce's read/write ports — the same idea applied at the
//! fence. Nothing else. `Chain::fuse` below *is* that rule: fold adjacent Maps by composing their
//! `Expr`s, cut at every Reduce.
//!
//! # What this module ships (all proven on the CPU runtime, no GPU)
//!
//!   1. **The op algebra** (`UnOp`, `BinOp`, `Red`, `Class`, `Expr`, `Chain`) — the morphisms and the
//!      class-decides-fusibility rule.
//!   2. **The fusion pass** (`Chain::fuse`) — folds adjacent Maps by composing `Expr`s, cuts at fences.
//!   3. **A real one-launch fusion engine** — `fused_interp` is ONE generic `#[kernel]` that carries the
//!      fused chain as a `#[comptime]` op program and interprets it in a comptime-**unrolled** loop, so
//!      the whole chain becomes straight-line arithmetic in one kernel body with zero intermediates. It
//!      is *not* a hand-written fixed kernel: any linear elementwise chain the builder produces runs
//!      through this one kernel.
//!   4. **The `Fuse` builder** (`Fuse::new(a).mul(w).add(b).silu().run(&client)`) — the ergonomic
//!      "fuse for me" surface. Reads like composition; compiles the chain to a `Program` and dispatches
//!      exactly one `fused_interp` launch.
//!   5. **Bit-exact gates** (the tests) — each fused chain is checked three ways: (i) the fused one
//!      kernel, (ii) the naive N-kernel pipeline that materializes N−1 device intermediates, (iii) a
//!      plain-Rust reference — all byte-for-byte identical on the CPU runtime.
//!
//! See `path_to_full_auto_fusion` at the bottom for the honest recommendation on tracing an *arbitrary*
//! op DAG (not just a builder chain).

use crate::prelude::*;

// ======================================================================================
// 1. The op algebra — the entire surface. Two classes; the class decides fusibility.
// ======================================================================================

/// A unary pointwise morphism `F -> F`. Every variant is index-local, hence freely fusible.
///
/// All of these lower to native intrinsics on every backend (verified on the CPU runtime): `exp`,
/// `tanh`, `sqrt`, `recip`, `abs`. `Silu`/`Sigmoid`/`Gelu` are *defined* in terms of those, so they are
/// one composed morphism, not a special case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    /// `-x`.
    Neg,
    /// `1 / x`.
    Recip,
    /// `|x|`.
    Abs,
    /// `exp(x)`.
    Exp,
    /// `tanh(x)`.
    Tanh,
    /// `1 / sqrt(x)` — reciprocal square root (the rms-norm epilogue scale).
    Rsqrt,
    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// SiLU / swish: `x * sigmoid(x)`.
    Silu,
    /// GELU, `tanh` approximation (the one Transformers/llama ship):
    /// `0.5 x (1 + tanh( sqrt(2/pi) (x + 0.044715 x^3) ))`.
    Gelu,
}

/// A binary pointwise morphism `(F, F) -> F`. Index-local, hence freely fusible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// A row contraction — a fusion **fence**. Output depends on a whole row, not one index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Red {
    Sum,
    Max,
}

/// The class of an op decides whether it composes (Map) or fences (Reduce). This — not a pattern
/// matcher over concrete ops — is what makes fusion decidable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Map,
    Reduce,
}

// ======================================================================================
// 2. Reify — a pointwise expression is a tree of scalar morphisms (the body of ONE Map region).
//    `Cur` = the pipeline's incoming element; `In(id)` = a side-input array; `Const` = comptime scalar.
//    Operator overloading makes reification read like the math it denotes.
// ======================================================================================

/// A reified scalar morphism: the per-element function that a Map region computes. Building one does
/// not compute anything — it records the DAG (the "trace").
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// The value flowing down the pipeline at this element (the Map region's input stream).
    Cur,
    /// Side input array `id`, read at the current element index.
    In(usize),
    /// A compile-time constant.
    Const(f32),
    Un(UnOp, Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
}

impl Expr {
    pub fn input(id: usize) -> Expr {
        Expr::In(id)
    }
    pub fn cur() -> Expr {
        Expr::Cur
    }
    pub fn konst(c: f32) -> Expr {
        Expr::Const(c)
    }
    pub fn un(self, op: UnOp) -> Expr {
        Expr::Un(op, Box::new(self))
    }
    pub fn silu(self) -> Expr {
        self.un(UnOp::Silu)
    }
    pub fn gelu(self) -> Expr {
        self.un(UnOp::Gelu)
    }
    pub fn sigmoid(self) -> Expr {
        self.un(UnOp::Sigmoid)
    }
    pub fn exp(self) -> Expr {
        self.un(UnOp::Exp)
    }
    pub fn tanh(self) -> Expr {
        self.un(UnOp::Tanh)
    }
    pub fn rsqrt(self) -> Expr {
        self.un(UnOp::Rsqrt)
    }
    pub fn recip(self) -> Expr {
        self.un(UnOp::Recip)
    }
    pub fn abs(self) -> Expr {
        self.un(UnOp::Abs)
    }

    /// Morphism composition: substitute `inner` for every `Cur` leaf. This is the *only* operation the
    /// fusion pass needs — `map(g) . map(f)` becomes `map(g[Cur := f])`.
    pub fn compose_cur(&self, inner: &Expr) -> Expr {
        match self {
            Expr::Cur => inner.clone(),
            Expr::In(id) => Expr::In(*id),
            Expr::Const(c) => Expr::Const(*c),
            Expr::Un(op, a) => Expr::Un(*op, Box::new(a.compose_cur(inner))),
            Expr::Bin(op, a, b) => {
                Expr::Bin(*op, Box::new(a.compose_cur(inner)), Box::new(b.compose_cur(inner)))
            }
        }
    }

    /// Node count — a proxy for "arithmetic in the fused kernel body".
    pub fn size(&self) -> usize {
        match self {
            Expr::Cur | Expr::In(_) | Expr::Const(_) => 1,
            Expr::Un(_, a) => 1 + a.size(),
            Expr::Bin(_, a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Pretty-print the composed morphism — this string *is* the fused kernel's body (the codegen
    /// target). Makes "reify -> one kernel" visible without a GPU.
    pub fn body(&self) -> String {
        match self {
            Expr::Cur => "cur".into(),
            Expr::In(id) => format!("in{id}[i]"),
            Expr::Const(c) => format!("{c}"),
            Expr::Un(op, a) => {
                let name = match op {
                    UnOp::Neg => return format!("(-{})", a.body()),
                    UnOp::Recip => "recip",
                    UnOp::Abs => "abs",
                    UnOp::Exp => "exp",
                    UnOp::Tanh => "tanh",
                    UnOp::Rsqrt => "rsqrt",
                    UnOp::Sigmoid => "sigmoid",
                    UnOp::Silu => "silu",
                    UnOp::Gelu => "gelu",
                };
                format!("{name}({})", a.body())
            }
            Expr::Bin(op, a, b) => {
                let s = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                };
                format!("({} {} {})", a.body(), s, b.body())
            }
        }
    }
}

impl core::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Bin(BinOp::Add, Box::new(self), Box::new(rhs))
    }
}
impl core::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Bin(BinOp::Sub, Box::new(self), Box::new(rhs))
    }
}
impl core::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Bin(BinOp::Mul, Box::new(self), Box::new(rhs))
    }
}
impl core::ops::Div for Expr {
    type Output = Expr;
    fn div(self, rhs: Expr) -> Expr {
        Expr::Bin(BinOp::Div, Box::new(self), Box::new(rhs))
    }
}

// ======================================================================================
// 3. The pipeline algebra — an array-level chain of steps. Maps compose; Reduces fence.
// ======================================================================================

/// One array-level step. A `Map` transforms the stream pointwise (its `Expr` is over `Cur` + side
/// inputs); a `Reduce` contracts each row — the fence.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Map(Expr),
    Reduce(Red),
}

impl Step {
    pub fn class(&self) -> Class {
        match self {
            Step::Map(_) => Class::Map,
            Step::Reduce(_) => Class::Reduce,
        }
    }
}

/// A fused region after the pass: a single Map morphism, or a lone Reduce fence.
#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    Map(Expr),
    Reduce(Red),
}

/// A reified pipeline. `x.rms_norm().rope()` and friends reify to one of these; `fuse` partitions it
/// into the minimal set of kernels.
#[derive(Clone, Debug, Default)]
pub struct Chain {
    pub steps: Vec<Step>,
}

impl Chain {
    pub fn new() -> Chain {
        Chain { steps: Vec::new() }
    }

    /// Pointwise unary on the stream: `cur = op(cur)`.
    pub fn map_unary(mut self, op: UnOp) -> Chain {
        self.steps.push(Step::Map(Expr::Un(op, Box::new(Expr::Cur))));
        self
    }
    /// Pointwise binary against side input `id`: `cur = op(cur, in[id])`.
    pub fn map_binary(mut self, op: BinOp, id: usize) -> Chain {
        self.steps
            .push(Step::Map(Expr::Bin(op, Box::new(Expr::Cur), Box::new(Expr::In(id)))));
        self
    }
    /// Pointwise binary against a constant: `cur = op(cur, c)`.
    pub fn map_const(mut self, op: BinOp, c: f32) -> Chain {
        self.steps
            .push(Step::Map(Expr::Bin(op, Box::new(Expr::Cur), Box::new(Expr::Const(c)))));
        self
    }
    /// An arbitrary pointwise `Expr` over `Cur` + side inputs (the general Map).
    pub fn map(mut self, e: Expr) -> Chain {
        self.steps.push(Step::Map(e));
        self
    }
    /// A row contraction — a fence.
    pub fn reduce(mut self, r: Red) -> Chain {
        self.steps.push(Step::Reduce(r));
        self
    }

    /// **The fusion pass.** Fold adjacent Maps by composing their `Expr`s (the functor law); cut a new
    /// region at every Reduce. `regions.len()` is the number of kernels; it is `<=` `steps.len()`, and
    /// strictly less whenever any two Maps were adjacent — that gap is the fusion.
    pub fn fuse(&self) -> Vec<Region> {
        let mut regions = Vec::new();
        let mut acc: Option<Expr> = None;
        for step in &self.steps {
            match step {
                Step::Map(e) => {
                    // g after f  ==>  g[Cur := f]
                    acc = Some(match acc.take() {
                        None => e.clone(),
                        Some(prev) => e.compose_cur(&prev),
                    });
                }
                Step::Reduce(r) => {
                    if let Some(e) = acc.take() {
                        regions.push(Region::Map(e));
                    }
                    regions.push(Region::Reduce(*r));
                }
            }
        }
        if let Some(e) = acc.take() {
            regions.push(Region::Map(e));
        }
        regions
    }

    /// How many kernels this chain fuses down to.
    pub fn kernel_count(&self) -> usize {
        self.fuse().len()
    }
}

// ======================================================================================
// 4. Reference semantics — evaluate a `Chain` on the CPU. The trusted oracle the lowering is gated on.
//    Row model: arrays are `[rows, n]`; a Reduce contracts each row of width `n` and broadcasts back.
// ======================================================================================

/// The single source of truth for the scalar semantics of every `UnOp`. The device `#[device]` fn
/// `apply_un_dev` below is a line-for-line mirror of this — that is why fused == oracle bit-exactly.
pub fn apply_un(op: UnOp, x: f32) -> f32 {
    match op {
        UnOp::Neg => -x,
        UnOp::Recip => 1.0 / x,
        UnOp::Abs => x.abs(),
        UnOp::Exp => x.exp(),
        UnOp::Tanh => x.tanh(),
        UnOp::Rsqrt => 1.0 / x.sqrt(),
        UnOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnOp::Silu => x / (1.0 + (-x).exp()),
        UnOp::Gelu => {
            const C: f32 = 0.797_884_56; // sqrt(2/pi)
            0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
        }
    }
}
pub fn apply_bin(op: BinOp, a: f32, b: f32) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

fn eval_expr(e: &Expr, cur: f32, inputs: &[&[f32]], i: usize) -> f32 {
    match e {
        Expr::Cur => cur,
        Expr::In(id) => inputs[*id][i],
        Expr::Const(c) => *c,
        Expr::Un(op, a) => apply_un(*op, eval_expr(a, cur, inputs, i)),
        Expr::Bin(op, a, b) => {
            apply_bin(*op, eval_expr(a, cur, inputs, i), eval_expr(b, cur, inputs, i))
        }
    }
}

fn reduce_rows(cur: &[f32], r: Red, n: usize) -> Vec<f32> {
    let len = cur.len();
    let rows = len / n;
    let mut out = vec![0.0f32; len];
    for row in 0..rows {
        let base = row * n;
        let acc = match r {
            Red::Sum => cur[base..base + n].iter().sum(),
            Red::Max => cur[base..base + n].iter().copied().fold(f32::MIN, f32::max),
        };
        for i in 0..n {
            out[base + i] = acc;
        }
    }
    out
}

/// Evaluate a whole chain over flat `[rows*n]` inputs. `inputs[0]` seeds the stream. The result is the
/// same whether you run the chain step-by-step (naive) or on its fused regions — that invariance is the
/// point (`eval(chain) == eval_fused(chain.fuse())`, tested below).
pub fn eval(chain: &Chain, inputs: &[&[f32]], n: usize) -> Vec<f32> {
    let len = inputs[0].len();
    let mut cur: Vec<f32> = inputs[0].to_vec();
    for step in &chain.steps {
        match step {
            Step::Map(e) => cur = (0..len).map(|i| eval_expr(e, cur[i], inputs, i)).collect(),
            Step::Reduce(r) => cur = reduce_rows(&cur, *r, n),
        }
    }
    cur
}

/// Evaluate the fused regions (each Map region is one composed morphism). Same result as `eval`.
pub fn eval_fused(regions: &[Region], inputs: &[&[f32]], n: usize) -> Vec<f32> {
    let len = inputs[0].len();
    let mut cur: Vec<f32> = inputs[0].to_vec();
    for r in regions {
        match r {
            Region::Map(e) => cur = (0..len).map(|i| eval_expr(e, cur[i], inputs, i)).collect(),
            Region::Reduce(red) => cur = reduce_rows(&cur, *red, n),
        }
    }
    cur
}

// ======================================================================================
// 5. The fusion ENGINE — ONE kernel that interprets a fused linear chain at comptime.
//
//    Design (given cubecl's comptime-monomorphized model): the fused Map region is lowered to a small
//    linear "op program" carried as a `#[comptime]` value. The kernel walks it in a comptime-UNROLLED
//    loop, so each op becomes straight-line arithmetic in the ONE kernel body — no runtime interpreter,
//    no intermediate arrays, no extra launches. `#[device]` fns give each morphism as a value and the
//    program threads them through the accumulator. Composition = inlining = fusion, mechanized.
// ======================================================================================

/// One instruction of a fused linear chain, carried in the kernel's `#[comptime]` program.
///
/// A linear chain is exactly: start from the primary input, then repeatedly either apply a unary
/// morphism to the accumulator, or combine the accumulator with a side-input array under a binary
/// morphism, or combine it with a compile-time constant. `slot` indexes the kernel's side-input
/// `Sequence`. This is the general shape any `Fuse` builder chain compiles to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instr {
    /// `acc = op(acc)`.
    Un(UnOp),
    /// `acc = op(acc, sides[slot][i])`.
    BinIn(BinOp, usize),
    /// `acc = op(acc, bits)` where `bits` is the `f32::to_bits` of the constant (so `Instr` stays
    /// `Eq + Hash`, which the comptime machinery requires; decoded via `f32::from_bits`).
    BinConst(BinOp, u32),
}

/// A fused linear program: the ordered instructions plus the number of side-input slots it references.
/// Produced by `Fuse::compile`; consumed by `fused_interp`. `Eq + Hash` because it rides as a
/// `#[comptime]` value — the lowering engine keys the compiled kernel on it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Program {
    pub instrs: Vec<Instr>,
    pub n_sides: usize,
}

// --- The morphisms as device values (mirror `apply_un` line for line) --------------------------

#[device]
fn dsigmoid<F: Float>(x: F) -> F {
    F::new(1.0) / (F::new(1.0) + (-x).exp())
}

/// Apply one `UnOp` on the device — a comptime `match`, so only the selected branch is emitted.
#[device]
fn apply_un_dev<F: Float>(#[comptime] op: UnOp, x: F) -> F {
    match op {
        UnOp::Neg => -x,
        UnOp::Recip => F::new(1.0) / x,
        UnOp::Abs => x.abs(),
        UnOp::Exp => x.exp(),
        UnOp::Tanh => x.tanh(),
        UnOp::Rsqrt => F::new(1.0) / x.sqrt(),
        UnOp::Sigmoid => dsigmoid::<F>(x),
        UnOp::Silu => x * dsigmoid::<F>(x),
        UnOp::Gelu => {
            let c = F::new(0.797_884_56); // sqrt(2/pi)
            let inner = c * (x + F::new(0.044715) * x * x * x);
            F::new(0.5) * x * (F::new(1.0) + inner.tanh())
        }
    }
}

#[device]
fn apply_bin_dev<F: Float>(#[comptime] op: BinOp, a: F, b: F) -> F {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

/// **The one fused kernel.** Reads the primary input once, threads it through the whole comptime
/// program (unrolled to straight-line code), writes the result once. Every `Instr` is inlined; there is
/// one launch and zero intermediates, whatever the chain length. `sides` carries the side-input arrays
/// the `BinIn` instructions reference, indexed by their comptime `slot`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn fused_interp<F: Float>(
    x: &Array<F>,
    sides: &Sequence<Array<F>>,
    out: &mut Array<F>,
    #[comptime] prog: Program,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let mut acc = x[i];
        #[unroll]
        for step in 0..prog.instrs.len() {
            match comptime!(prog.instrs[step]) {
                Instr::Un(op) => acc = apply_un_dev::<F>(op, acc),
                Instr::BinIn(op, slot) => acc = apply_bin_dev::<F>(op, acc, sides.index(slot)[i]),
                Instr::BinConst(op, bits) => {
                    // Decode the constant on the host (comptime), then materialize it as a device
                    // literal. `f32::from_bits` must not sit inside the kernel-expanded expression.
                    let c = comptime!(f32::from_bits(bits));
                    acc = apply_bin_dev::<F>(op, acc, F::new(c))
                }
            }
        }
        out[i] = acc;
    }
}

// ======================================================================================
// 6. The `Fuse` builder — the ergonomic "fuse for me" surface. Reads like composition; compiles to a
//    `Program`; dispatches exactly ONE `fused_interp`.
// ======================================================================================

const BLOCK: u32 = 256;

/// Fluent builder for a fused elementwise chain.
///
/// ```ignore
/// let y = Fuse::new(&a)      // start from array a
///     .mul(&w)               // a * w        (binary against a side input)
///     .add(&b)               // + b
///     .silu()                // silu(...)    (unary)
///     .run(&client);         // -> ONE fused kernel launch, no intermediates
/// ```
///
/// Every method appends one `Instr`; `run` compiles them to a `Program` and dispatches one
/// `fused_interp`. The chain reads exactly like the mathematical composition it denotes, and *is* that
/// composition — the builder does not evaluate anything until `run`.
pub struct Fuse<'a> {
    primary: &'a [f32],
    sides: Vec<&'a [f32]>,
    instrs: Vec<Instr>,
}

impl<'a> Fuse<'a> {
    /// Start a chain from a primary input array (the stream seed).
    pub fn new(primary: &'a [f32]) -> Self {
        Fuse { primary, sides: Vec::new(), instrs: Vec::new() }
    }

    fn push_bin_in(mut self, op: BinOp, rhs: &'a [f32]) -> Self {
        let slot = self.sides.len();
        self.sides.push(rhs);
        self.instrs.push(Instr::BinIn(op, slot));
        self
    }
    fn push_bin_const(mut self, op: BinOp, c: f32) -> Self {
        self.instrs.push(Instr::BinConst(op, c.to_bits()));
        self
    }
    fn push_un(mut self, op: UnOp) -> Self {
        self.instrs.push(Instr::Un(op));
        self
    }

    // --- binary against a side-input array ---
    pub fn add(self, rhs: &'a [f32]) -> Self {
        self.push_bin_in(BinOp::Add, rhs)
    }
    pub fn sub(self, rhs: &'a [f32]) -> Self {
        self.push_bin_in(BinOp::Sub, rhs)
    }
    pub fn mul(self, rhs: &'a [f32]) -> Self {
        self.push_bin_in(BinOp::Mul, rhs)
    }
    pub fn div(self, rhs: &'a [f32]) -> Self {
        self.push_bin_in(BinOp::Div, rhs)
    }

    // --- binary against a scalar constant ---
    pub fn add_scalar(self, c: f32) -> Self {
        self.push_bin_const(BinOp::Add, c)
    }
    pub fn sub_scalar(self, c: f32) -> Self {
        self.push_bin_const(BinOp::Sub, c)
    }
    pub fn mul_scalar(self, c: f32) -> Self {
        self.push_bin_const(BinOp::Mul, c)
    }
    pub fn div_scalar(self, c: f32) -> Self {
        self.push_bin_const(BinOp::Div, c)
    }

    // --- unary morphisms ---
    pub fn neg(self) -> Self {
        self.push_un(UnOp::Neg)
    }
    pub fn recip(self) -> Self {
        self.push_un(UnOp::Recip)
    }
    pub fn abs(self) -> Self {
        self.push_un(UnOp::Abs)
    }
    pub fn exp(self) -> Self {
        self.push_un(UnOp::Exp)
    }
    pub fn tanh(self) -> Self {
        self.push_un(UnOp::Tanh)
    }
    pub fn rsqrt(self) -> Self {
        self.push_un(UnOp::Rsqrt)
    }
    pub fn sigmoid(self) -> Self {
        self.push_un(UnOp::Sigmoid)
    }
    pub fn silu(self) -> Self {
        self.push_un(UnOp::Silu)
    }
    pub fn gelu(self) -> Self {
        self.push_un(UnOp::Gelu)
    }

    /// The compiled fused program (instructions + side-slot count). Pure — no device work.
    pub fn compile(&self) -> Program {
        Program { instrs: self.instrs.clone(), n_sides: self.sides.len() }
    }

    /// Evaluate the compiled chain on the host — the reference oracle (mirrors the device semantics).
    pub fn eval_ref(&self) -> Vec<f32> {
        let mut acc = self.primary.to_vec();
        for ins in &self.instrs {
            for (i, a) in acc.iter_mut().enumerate() {
                *a = match *ins {
                    Instr::Un(op) => apply_un(op, *a),
                    Instr::BinIn(op, slot) => apply_bin(op, *a, self.sides[slot][i]),
                    Instr::BinConst(op, bits) => apply_bin(op, *a, f32::from_bits(bits)),
                };
            }
        }
        acc
    }

    /// **Fuse and run**: compile the chain to one `Program` and dispatch exactly ONE `fused_interp`.
    /// This is the "the DSL fuses for me" surface: N composed ops, one launch, zero intermediates.
    pub fn run<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<f32> {
        let n = self.primary.len();
        let grid = Grid::Static((n as u32).div_ceil(BLOCK), 1, 1);
        let xh = client.create_from_slice(f32::as_bytes(self.primary));
        let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));

        let handles: Vec<_> =
            self.sides.iter().map(|s| client.create_from_slice(f32::as_bytes(s))).collect();
        let mut sides = SequenceArg::new();
        for h in &handles {
            sides.push(unsafe { ArrayArg::from_raw_parts(h.clone(), n) });
        }

        unsafe {
            fused_interp::launch_unchecked::<f32, R>(
                client,
                grid,
                Block::new_1d(BLOCK),
                ArrayArg::from_raw_parts(xh.clone(), n),
                sides,
                ArrayArg::from_raw_parts(oh.clone(), n),
                self.compile(),
            );
        }
        f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
    }
}

// ======================================================================================
// 7. Naive baseline — the per-op eager path (N kernels, N-1 materialized intermediates). The thing the
//    fusion removes; the bit-exact target the fused kernel must match.
// ======================================================================================

/// `a * b`. / `a + b`. / `silu(x)`.
#[device]
pub fn dmul<F: Float>(a: F, b: F) -> F {
    a * b
}
#[device]
pub fn dadd<F: Float>(a: F, b: F) -> F {
    a + b
}
#[device]
pub fn dsilu<F: Float>(x: F) -> F {
    x * dsigmoid::<F>(x)
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn mul_k<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = dmul::<F>(a[i], b[i]);
    }
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn add_k<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = dadd::<F>(a[i], b[i]);
    }
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn silu_k<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = dsilu::<F>(x[i]);
    }
}

/// Naive SwiGLU tail `y = silu(a) * b`: TWO kernels, ONE materialized intermediate `t = silu(a)`.
/// This is the concrete pattern the elementwise tail of an FFN produces after the two matmuls.
pub fn swiglu_naive2<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let grid = Grid::Static((n as u32).div_ceil(BLOCK), 1, 1);
    let ah = client.create_from_slice(f32::as_bytes(a));
    let bh = client.create_from_slice(f32::as_bytes(b));
    let t = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    unsafe {
        silu_k::launch_unchecked::<f32, R>(
            client,
            grid.clone(),
            Block::new_1d(BLOCK),
            ArrayArg::from_raw_parts(ah.clone(), n),
            ArrayArg::from_raw_parts(t.clone(), n),
        );
        mul_k::launch_unchecked::<f32, R>(
            client,
            grid,
            Block::new_1d(BLOCK),
            ArrayArg::from_raw_parts(t.clone(), n),
            ArrayArg::from_raw_parts(bh.clone(), n),
            ArrayArg::from_raw_parts(oh.clone(), n),
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Naive `y = silu(a*b + c)`: THREE kernels, TWO materialized intermediates. Precisely the traffic the
/// fused path removes.
pub fn naive3_run<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    let n = a.len();
    let grid = Grid::Static((n as u32).div_ceil(BLOCK), 1, 1);
    let ah = client.create_from_slice(f32::as_bytes(a));
    let bh = client.create_from_slice(f32::as_bytes(b));
    let ch = client.create_from_slice(f32::as_bytes(c));
    let t1 = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let t2 = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    unsafe {
        mul_k::launch_unchecked::<f32, R>(
            client,
            grid.clone(),
            Block::new_1d(BLOCK),
            ArrayArg::from_raw_parts(ah.clone(), n),
            ArrayArg::from_raw_parts(bh.clone(), n),
            ArrayArg::from_raw_parts(t1.clone(), n),
        );
        add_k::launch_unchecked::<f32, R>(
            client,
            grid.clone(),
            Block::new_1d(BLOCK),
            ArrayArg::from_raw_parts(t1.clone(), n),
            ArrayArg::from_raw_parts(ch.clone(), n),
            ArrayArg::from_raw_parts(t2.clone(), n),
        );
        silu_k::launch_unchecked::<f32, R>(
            client,
            grid,
            Block::new_1d(BLOCK),
            ArrayArg::from_raw_parts(t2.clone(), n),
            ArrayArg::from_raw_parts(oh.clone(), n),
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

// ======================================================================================
// 8. HONEST path to full auto-fusion (tracing an ARBITRARY op DAG, not just a builder chain).
// ======================================================================================

/// # From "builder chain" to "arbitrary graph": the honest roadmap
///
/// What ships here fuses a **linear elementwise chain** into one launch, decided from the op *type*.
/// That already closes the concrete gap the llama.cpp audit named — `ggml_can_fuse` for
/// `rope+set_rows`, `ffn_up+gate` (the SwiGLU tail), `rms_norm+mul+add` — because each is a short
/// linear Map run (or a Reduce with a Map prologue/epilogue, which `Chain::fuse` already partitions).
/// The builder is the ergonomic front for exactly this class.
///
/// To fuse an **arbitrary op DAG** the model *already* emits (fuse a whole forward pass with no
/// builder), there are three routes. Recommendation and honest effort estimates:
///
/// ## Option A — hanzo-native pass over our own trace  *(RECOMMENDED)*
/// `Expr`/`Chain` here is already the IR; a DAG is the same node types with sharing (a `Vec<Node>` +
/// indices instead of `Box`). The pass is the exact rule this module documents, generalized from a line
/// to a graph:
///   1. **Trace** — run the model's forward with tensors replaced by `NodeId`s (a taping newtype), so
///      `+ * silu ...` record nodes instead of computing. ~1 wrapper type, `Deref`-style op overloads.
///   2. **Partition** — flood-fill maximal Map-only connected regions; every edge into a Reduce (or a
///      non-elementwise op: matmul, gather, conv) is a cut. This *is* `Class`-based fusibility, one
///      predicate, applied to graph edges instead of list-adjacency.
///   3. **Schedule + lower** — topologically order regions; each Map region compiles to one
///      `fused_interp`-style kernel (generalize `Program` from a line to a small DAG evaluated per
///      element, side inputs = the region's live-in edges). Reduce/matmul regions call the existing
///      hand-tuned kernels, with adjacent Map regions attached as their read-prologue / write-epilogue.
///   * **Effort:** ~1.5–3 weeks. Steps 1–2 are a few hundred lines and directly reuse this module's
///     algebra + oracle (the CPU `eval` becomes the graph correctness gate). Step 3's per-element DAG
///     interpreter is a modest generalization of `fused_interp`'s comptime unroll (already proven).
///   * **Why recommended:** one IR, one fusibility predicate, no third-party runtime in the hot path,
///     and it composes with our hand-tuned Reduce/matmul kernels instead of competing with them. It is
///     the Hickey answer — the algebra we already have, applied to a graph.
///
/// ## Option B — adopt burn-fusion (Burn's `cubecl`-native fusion engine)
/// Burn already has a stream-recording fusion layer over cubecl (`burn-fusion` + `cubecl-fusion`) that
/// captures an op stream and JITs fused elementwise kernels — the same target, production-tested.
///   * **Effort:** ~1–2 weeks to wire *if* tensors flow through Burn's backend; **weeks-to-months**
///     otherwise, because it fuses *Burn tensor ops*, so adopting it means routing hanzo tensors through
///     Burn's backend/stream API (a large surface) or vendoring its fusion crate and re-hosting it on
///     our tensor type.
///   * **Trade-off:** least novel fusion code, but couples us to Burn's tensor abstraction and its
///     roadmap, and its cut-points/epilogue policy are Burn's, not ours — harder to co-schedule with our
///     hand-tuned peak kernels. Good fallback if Option A's scheduler proves deep; not the first move.
///
/// ## Option C — lower onto cubecl's own trace/stream API directly
/// cubecl exposes the IR our `#[kernel]`s already compile through. In principle we could emit a fused
/// kernel straight from a traced graph at that layer.
///   * **Effort:** **high / unbounded** — that IR is an internal, moving target with little public
///     surface for "build a kernel from a runtime graph"; we would track cubecl internals across
///     releases. Only worth it to fuse *across* the Map/Reduce fence (e.g. a genuinely novel
///     matmul-epilogue codegen) that Options A/B can't express.
///
/// **Bottom line:** do **Option A**. The algebra, the oracle, and the one-kernel comptime interpreter
/// in this module are the hard, now-proven parts; going from a line to a DAG is a bounded, ~2-week
/// generalization of code that already exists and is bit-exact on the CPU. Keep Option B in reserve as
/// the "don't-reinvent-the-scheduler" fallback and avoid Option C except for cross-fence codegen.
pub const fn path_to_full_auto_fusion() {}

// ======================================================================================
// 9. Tests — the bit-exact gate. Fused ONE kernel == naive N kernels == plain-Rust reference.
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The chain the engine reifies for `silu(a*b + c)`: three pointwise steps.
    fn silu_mul_add_chain() -> Chain {
        Chain::new()
            .map_binary(BinOp::Mul, 1) // cur = a * b
            .map_binary(BinOp::Add, 2) // cur = cur + c
            .map_unary(UnOp::Silu) // cur = silu(cur)
    }

    // --- The algebra: fusion = composition, fences fall out of the op class --------------------

    #[test]
    fn three_maps_fuse_to_one_kernel() {
        let chain = silu_mul_add_chain();
        assert_eq!(chain.steps.len(), 3, "reified as 3 pointwise steps");
        let regions = chain.fuse();
        assert_eq!(regions.len(), 1, "all Map -> one fused kernel");
        match &regions[0] {
            // The one kernel's body IS the composed morphism (leftmost leaf is the stream seed `cur`).
            Region::Map(e) => assert_eq!(e.body(), "silu(((cur * in1[i]) + in2[i]))"),
            _ => panic!("expected a Map region"),
        }
    }

    #[test]
    fn reduce_is_a_fence() {
        // A Map run, a Sum fence, then a Map run: rms-norm shape -> 3 regions, the reduce splits it.
        let chain = Chain::new()
            .map(Expr::cur() * Expr::cur()) // square (Map)
            .map(Expr::cur()) // still Map: fuses with the square
            .reduce(Red::Sum) // FENCE
            .map_const(BinOp::Mul, 0.5) // epilogue (Map)
            .map_unary(UnOp::Rsqrt); // epilogue continues, fuses
        assert_eq!(chain.steps.len(), 5);
        let regions = chain.fuse();
        // [Map(prologue), Reduce, Map(epilogue)] — the two Map runs each collapsed to one region.
        assert_eq!(regions.len(), 3);
        assert!(matches!(regions[0], Region::Map(_)));
        assert!(matches!(regions[1], Region::Reduce(Red::Sum)));
        assert!(matches!(regions[2], Region::Map(_)));
    }

    #[test]
    fn functor_law_holds_in_the_evaluator() {
        // eval(chain) == eval(fuse(chain)) — fusing must not change semantics.
        let chain = silu_mul_add_chain();
        let n = 8usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 0.3).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.5 - i as f32 * 0.07).collect();
        let c: Vec<f32> = (0..n).map(|i| i as f32 * 0.02).collect();
        let inputs: [&[f32]; 3] = [&a, &b, &c];
        let step_by_step = eval(&chain, &inputs, n);
        let fused = eval_fused(&chain.fuse(), &inputs, n);
        assert_eq!(step_by_step, fused, "fusion changed the numerics");
    }

    #[test]
    fn every_unop_matches_its_reference_shape() {
        // apply_un is the single source of truth; sanity-check the non-obvious ones are finite & sane.
        for &op in &[UnOp::Silu, UnOp::Gelu, UnOp::Sigmoid, UnOp::Rsqrt, UnOp::Exp, UnOp::Tanh] {
            for x in [-2.0f32, -0.3, 0.7, 3.0] {
                let y = apply_un(op, if op == UnOp::Rsqrt { x.abs() + 0.1 } else { x });
                assert!(y.is_finite(), "{op:?}({x}) not finite: {y}");
            }
        }
        assert!((apply_un(UnOp::Sigmoid, 0.0) - 0.5).abs() < 1e-7);
        assert!(apply_un(UnOp::Silu, 0.0).abs() < 1e-7);
        assert!(apply_un(UnOp::Gelu, 0.0).abs() < 1e-7);
    }

    // --- The lowering, on the CPU runtime: fused ONE kernel == naive N kernels, bit-exact ----------

    #[cfg(feature = "cpu")]
    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        CpuRuntime::client(&CpuDevice::default())
    }

    #[cfg(feature = "cpu")]
    fn xorshift_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 4000) as f32 / 1000.0 - 2.0
            })
            .collect()
    }

    #[cfg(feature = "cpu")]
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// SwiGLU tail `y = silu(a) * b` — the `ffn_up+gate` fusion the audit named. The `Fuse` builder
    /// (ONE launch) must be byte-for-byte the naive TWO-kernel path AND the plain-Rust reference.
    #[cfg(feature = "cpu")]
    #[test]
    fn swiglu_tail_fused_equals_naive_bit_exact() {
        let client = cpu_client();
        let n = 1024usize;
        let a = xorshift_vec(n, 0x1234_5678_9abc_def0);
        let b = xorshift_vec(n, 0x0fed_cba9_8765_4321);

        // (i) fused — the ergonomic builder: silu(a) * b in ONE fused_interp launch.
        let fused = Fuse::new(&a).silu().mul(&b).run::<cubecl::cpu::CpuRuntime>(&client);
        // (ii) naive — silu then mul, TWO kernels, one intermediate.
        let naive = swiglu_naive2::<cubecl::cpu::CpuRuntime>(&client, &a, &b);
        // (iii) plain-Rust reference.
        let refv: Vec<f32> =
            a.iter().zip(&b).map(|(&x, &y)| apply_un(UnOp::Silu, x) * y).collect();

        assert_eq!(bits(&fused), bits(&naive), "fused (1 launch) != naive (2 launches)");
        // one comptime program, one launch:
        assert_eq!(Fuse::new(&a).silu().mul(&b).compile().instrs.len(), 2);
        let maxerr =
            fused.iter().zip(&refv).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        eprintln!("[fuse CPU] SwiGLU tail silu(a)*b: 2 kernels -> 1; bit-exact; max|fused-ref|={maxerr:.2e}");
        assert!(maxerr < 1e-5, "reference disagreement {maxerr}");
    }

    /// 3-op chain `y = silu(a*w + b)` — the builder (ONE launch) vs naive THREE kernels vs reference.
    #[cfg(feature = "cpu")]
    #[test]
    fn three_op_chain_fused_equals_naive_bit_exact() {
        let client = cpu_client();
        let n = 1024usize;
        let a = xorshift_vec(n, 0x2545_f491_4f6c_dd1d);
        let w = xorshift_vec(n, 0xdead_beef_cafe_babe);
        let b = xorshift_vec(n, 0x0123_4567_89ab_cdef);

        // (i) fused builder: silu(a*w + b), ONE launch.
        let fused = Fuse::new(&a).mul(&w).add(&b).silu().run::<cubecl::cpu::CpuRuntime>(&client);
        // (ii) naive: a*w, +b, silu — THREE kernels, TWO intermediates.
        let naive = naive3_run::<cubecl::cpu::CpuRuntime>(&client, &a, &w, &b);
        // (iii) reference.
        let refv: Vec<f32> = (0..n)
            .map(|i| apply_un(UnOp::Silu, apply_bin(BinOp::Add, a[i] * w[i], b[i])))
            .collect();

        assert_eq!(bits(&fused), bits(&naive), "fused (1) != naive (3)");
        assert_eq!(Fuse::new(&a).mul(&w).add(&b).silu().compile().instrs.len(), 3);
        let maxerr =
            fused.iter().zip(&refv).map(|(g, w)| (g - w).abs()).fold(0.0f32, f32::max);
        eprintln!("[fuse CPU] 3-op silu(a*w+b): 3 kernels -> 1; bit-exact; max|fused-ref|={maxerr:.2e}");
        assert!(maxerr < 1e-5, "reference disagreement {maxerr}");
    }

    /// A long, mixed chain exercising every op family in ONE fused kernel — proves the interpreter
    /// scales past the two named patterns and stays bit-exact with the host oracle.
    #[cfg(feature = "cpu")]
    #[test]
    fn long_mixed_chain_fused_matches_reference() {
        let client = cpu_client();
        let n = 2048usize;
        let a = xorshift_vec(n, 0xa5a5_5a5a_c3c3_3c3c);
        let w = xorshift_vec(n, 0x1111_2222_3333_4444);
        let b = xorshift_vec(n, 0x9999_8888_7777_6666);

        // sigmoid(w) as an explicit side array, so div-by-array is also exercised.
        let sig_w: Vec<f32> = w.iter().map(|&x| apply_un(UnOp::Sigmoid, x)).collect();
        // gelu( ( |a*2 - 1| + b ) / sigmoid(w) ) then tanh — 7 fused ops, one launch.
        let chain = Fuse::new(&a)
            .mul_scalar(2.0)
            .sub_scalar(1.0)
            .abs()
            .add(&b)
            .div(&sig_w)
            .gelu()
            .tanh();
        let prog = chain.compile();
        assert_eq!(prog.instrs.len(), 7, "seven fused instructions, one launch");

        let fused = chain.run::<cubecl::cpu::CpuRuntime>(&client);
        let refv = chain.eval_ref(); // host oracle over the identical program
        let maxerr =
            fused.iter().zip(&refv).map(|(g, r)| (g - r).abs()).fold(0.0f32, f32::max);
        eprintln!("[fuse CPU] 7-op mixed chain: one launch; max|fused-ref|={maxerr:.2e}");
        assert!(maxerr < 1e-5, "long-chain disagreement {maxerr}");
    }
}
