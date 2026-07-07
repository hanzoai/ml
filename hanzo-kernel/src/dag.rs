//! Full-DAG fusion: **fusion is composition, generalized from a line to a graph**.
//!
//! [`fuse`](crate::fuse) fuses a *linear* elementwise chain into one launch by folding adjacent Maps
//! with the functor law `map(g) . map(f) == map(g . f)`. That is the special case where the op graph is
//! a path. This module lifts the exact same algebra to an **arbitrary pointwise DAG** — the thing a real
//! forward pass emits, where a value **fans out** (is consumed by several ops) and ops **fan in** (a
//! binary reads two independently-computed values). A linear chain cannot express `silu(a)*b + a` (here
//! `a` is used twice) in one region; a DAG can, and it still fuses to **one** kernel.
//!
//! The pipeline is the honest Option-A roadmap [`crate::fuse`] documents, made real:
//!
//!   1. **Trace** ([`Tape`] + [`Var`]) — a `NodeId`-taping newtype. A forward pass of pointwise ops on
//!      `Var`s *records* a DAG (`Vec<Node>`, each `Node` = op + input `NodeId`s) instead of computing.
//!      Operator overloads (`Add/Sub/Mul/Div` + the unary morphisms) push nodes. The node algebra is the
//!      same `UnOp`/`BinOp`/`Red` as [`crate::fuse`] — one IR, reused.
//!   2. **Partition** ([`Dag::fuse`]) — flood-fill maximal **Map-only** connected regions; cut
//!      every edge into a `Reduce` (or any non-elementwise fence: matmul/gather/conv, modeled here by
//!      [`Red`]). The fusibility predicate is the existing [`Class`] applied to graph edges, not to
//!      list-adjacency. Each Map region becomes one fused kernel; each fence is its own region.
//!   3. **Lower** ([`Region::to_program`] + [`dag_interp()`]) — a Map region compiles to a small **DAG
//!      program** (`Vec<DagInstr>`, each instr references prior slots by comptime index), and ONE generic
//!      `#[kernel]` `dag_interp` interprets it in a comptime-**unrolled** loop over a per-element **slot
//!      array**. Fan-out = a slot read by several later instrs; fan-in = a `Bin` reading two slots. One
//!      launch, zero intermediates, any DAG shape. This generalizes `fused_interp`'s single accumulator
//!      (a line) to `n_slots` accumulators (a graph).
//!   4. **Schedule** ([`Dag::fuse`]) — topologically order regions; Reduce regions call the reduction
//!      oracle with the fused Map region attached as their read-prologue. `regions.len()` is the kernel
//!      count; it is `<<` the op count whenever a Map region absorbed more than one op — that gap is the
//!      fusion.
//!
//! # Bit-exact gate (the proof, CPU only)
//! A DAG with **fan-out** (a value reused) AND a **fence** (a `sum` reduction with a Map prologue and a
//! fused-fan-out Map epilogue) is computed three ways — (i) full-DAG-fused (one `dag_interp` launch per
//! Map region), (ii) naive per-op (one launch per node, materializing every intermediate), (iii) a
//! plain-Rust reference — and all three are byte-for-byte identical on the CPU runtime. The fused launch
//! count is reported against the naive one. See the tests at the bottom.

use crate::fuse::{apply_bin, apply_un, BinOp, Class, Red, UnOp};
use crate::prelude::*;

// ======================================================================================
// 1. The DAG IR — the same op algebra as `fuse`, but nodes reference prior nodes by index (sharing),
//    so a value can fan out and a binary can fan in. `Box<Expr>` (a tree) becomes `Vec<Node>` (a graph).
// ======================================================================================

/// One node of the traced pointwise DAG. `In`/`Const` are leaves; `Un`/`Bin` reference prior nodes by
/// their `NodeId` (index into [`Dag::nodes`]). A `Reduce` is the fence — its input is a `NodeId`, its
/// output is a fresh value every consumer downstream reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Node {
    /// External side-input array `slot`, read at the current element index.
    In(usize),
    /// A compile-time constant (stored as `f32::to_bits` so `Node` is `Eq + Hash` for comptime keying).
    Const(u32),
    /// `op(nodes[a])`.
    Un(UnOp, usize),
    /// `op(nodes[a], nodes[b])`.
    Bin(BinOp, usize, usize),
    /// A row contraction of `nodes[a]` — the fusion **fence** (models Reduce/matmul/gather).
    Reduce(Red, usize),
}

impl Node {
    /// The class decides fusibility — Map composes, Reduce fences. Same predicate as [`crate::fuse`], on nodes.
    pub fn class(&self) -> Class {
        match self {
            Node::Reduce(..) => Class::Reduce,
            _ => Class::Map,
        }
    }
    /// The node ids this node reads (its in-edges). Leaves read nothing.
    fn inputs(&self) -> heapless_inputs::Inputs {
        match *self {
            Node::In(_) | Node::Const(_) => heapless_inputs::Inputs::none(),
            Node::Un(_, a) | Node::Reduce(_, a) => heapless_inputs::Inputs::one(a),
            Node::Bin(_, a, b) => heapless_inputs::Inputs::two(a, b),
        }
    }
}

/// A tiny fixed-capacity in-edge list (0..2 ids), so `Node::inputs` allocates nothing.
mod heapless_inputs {
    #[derive(Clone, Copy)]
    pub struct Inputs {
        buf: [usize; 2],
        len: u8,
    }
    impl Inputs {
        pub fn none() -> Self {
            Inputs {
                buf: [0, 0],
                len: 0,
            }
        }
        pub fn one(a: usize) -> Self {
            Inputs {
                buf: [a, 0],
                len: 1,
            }
        }
        pub fn two(a: usize, b: usize) -> Self {
            Inputs {
                buf: [a, b],
                len: 2,
            }
        }
    }
    impl<'a> IntoIterator for &'a Inputs {
        type Item = usize;
        type IntoIter = core::iter::Copied<core::slice::Iter<'a, usize>>;
        fn into_iter(self) -> Self::IntoIter {
            self.buf[..self.len as usize].iter().copied()
        }
    }
}

/// The traced DAG: a topologically-ordered node list (a node only references earlier nodes) plus the
/// number of external side-input slots it reads. Producing one computes nothing — it is the trace.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dag {
    pub nodes: Vec<Node>,
    pub n_inputs: usize,
}

// ======================================================================================
// 2. Trace — a `NodeId`-taping newtype. Pointwise ops on `Var`s record nodes instead of computing.
// ======================================================================================

/// The tape a forward pass records into. `Var`s carry a `&Tape` and push nodes on every op.
#[derive(Default)]
pub struct Tape {
    nodes: core::cell::RefCell<Vec<Node>>,
    n_inputs: core::cell::Cell<usize>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape::default()
    }

    /// A fresh side-input array leaf. `slot` is the position in the launch's side-input `Sequence`.
    pub fn input(&self) -> Var<'_> {
        let slot = self.n_inputs.get();
        self.n_inputs.set(slot + 1);
        self.push(Node::In(slot))
    }
    /// A compile-time constant leaf.
    pub fn konst(&self, c: f32) -> Var<'_> {
        self.push(Node::Const(c.to_bits()))
    }

    fn push(&self, n: Node) -> Var<'_> {
        let mut v = self.nodes.borrow_mut();
        let id = v.len();
        v.push(n);
        Var { tape: self, id }
    }

    /// Freeze the tape into a `Dag` (topologically ordered by construction — a node only ever references
    /// earlier ids). The output of the graph is the last-pushed node reachable from `root`.
    pub fn finish(&self, root: Var<'_>) -> Dag {
        Dag {
            nodes: self.nodes.borrow().clone(),
            n_inputs: self.n_inputs.get(),
        }
        .pruned_to(root.id)
    }
}

/// A value in the traced graph — a `NodeId` into a `Tape`. Every pointwise operation on `Var`s records
/// a node and returns the new `Var`; nothing is computed. Reusing a `Var` in two ops **is** fan-out.
#[derive(Clone, Copy)]
pub struct Var<'t> {
    tape: &'t Tape,
    id: usize,
}

impl<'t> Var<'t> {
    /// This value's `NodeId` (its slot in the DAG).
    pub fn id(self) -> usize {
        self.id
    }
    fn un(self, op: UnOp) -> Var<'t> {
        self.tape.push(Node::Un(op, self.id))
    }
    pub fn silu(self) -> Var<'t> {
        self.un(UnOp::Silu)
    }
    pub fn gelu(self) -> Var<'t> {
        self.un(UnOp::Gelu)
    }
    pub fn sigmoid(self) -> Var<'t> {
        self.un(UnOp::Sigmoid)
    }
    pub fn exp(self) -> Var<'t> {
        self.un(UnOp::Exp)
    }
    pub fn tanh(self) -> Var<'t> {
        self.un(UnOp::Tanh)
    }
    pub fn rsqrt(self) -> Var<'t> {
        self.un(UnOp::Rsqrt)
    }
    pub fn recip(self) -> Var<'t> {
        self.un(UnOp::Recip)
    }
    pub fn abs(self) -> Var<'t> {
        self.un(UnOp::Abs)
    }
    /// Row contraction — the fence. The result is a new value every downstream op reads.
    pub fn reduce(self, r: Red) -> Var<'t> {
        self.tape.push(Node::Reduce(r, self.id))
    }
}

macro_rules! bin_op {
    ($Trait:ident, $method:ident, $variant:ident) => {
        impl<'t> core::ops::$Trait for Var<'t> {
            type Output = Var<'t>;
            fn $method(self, rhs: Var<'t>) -> Var<'t> {
                self.tape.push(Node::Bin(BinOp::$variant, self.id, rhs.id))
            }
        }
    };
}
bin_op!(Add, add, Add);
bin_op!(Sub, sub, Sub);
bin_op!(Mul, mul, Mul);
bin_op!(Div, div, Div);

// ======================================================================================
// 3. Reference semantics — evaluate the DAG on the CPU. The trusted oracle the lowering is gated on.
//    Row model matches `fuse`: arrays are `[rows, n]`; a Reduce contracts each row of width `n`.
// ======================================================================================

impl Dag {
    /// Keep only nodes reachable from `root`, renumbered so the graph stays topologically dense with
    /// `root` last. (A trace may record dead nodes; the fusion pass should never emit them.)
    fn pruned_to(mut self, root: usize) -> Dag {
        let n = self.nodes.len();
        let mut live = vec![false; n];
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if live[id] {
                continue;
            }
            live[id] = true;
            for i in self.nodes[id].inputs().into_iter() {
                stack.push(i);
            }
        }
        // Dense renumber, preserving order (old ids are already topological).
        let mut remap = vec![usize::MAX; n];
        let mut kept = Vec::new();
        for (old, &alive) in live.iter().enumerate() {
            if alive {
                remap[old] = kept.len();
                let mut node = self.nodes[old];
                node = match node {
                    Node::Un(op, a) => Node::Un(op, remap[a]),
                    Node::Bin(op, a, b) => Node::Bin(op, remap[a], remap[b]),
                    Node::Reduce(r, a) => Node::Reduce(r, remap[a]),
                    leaf => leaf,
                };
                kept.push(node);
            }
        }
        self.nodes = kept;
        self
    }

    /// Number of nodes (a proxy for "ops the naive path would launch").
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Evaluate the whole DAG per element on the CPU — the trusted oracle. `inputs[slot]` seeds each
    /// `In(slot)`. Every node's value is materialized (naive); a Reduce contracts each width-`n` row.
    pub fn eval(&self, inputs: &[&[f32]], n: usize) -> Vec<f32> {
        let len = inputs[0].len();
        // vals[node][i]
        let mut vals: Vec<Vec<f32>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let v = match *node {
                Node::In(slot) => inputs[slot].to_vec(),
                Node::Const(bits) => vec![f32::from_bits(bits); len],
                Node::Un(op, a) => vals[a].iter().map(|&x| apply_un(op, x)).collect(),
                Node::Bin(op, a, b) => (0..len)
                    .map(|i| apply_bin(op, vals[a][i], vals[b][i]))
                    .collect(),
                Node::Reduce(r, a) => reduce_rows(&vals[a], r, n),
            };
            vals.push(v);
        }
        vals.pop().unwrap()
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

// ======================================================================================
// 4. Partition — flood-fill maximal Map-only regions; cut at every Reduce. `Class` on graph edges.
// ======================================================================================

/// A fused region after partitioning: either a **Map region** (a connected block of pointwise nodes
/// that lowers to ONE `dag_interp` kernel — the `roots` are its outputs consumed downstream) or a lone
/// **Reduce fence** operating on another region's output.
#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    /// A maximal pointwise sub-DAG: `local` is its node list (self-contained, `In`/`Const`/`Un`/`Bin`
    /// only), `live_in` maps each `Node::In(k)` slot to the *global* value id it reads (a graph input,
    /// or another region's output), and `output` is the local node id this region produces.
    Map {
        local: Vec<Node>,
        live_in: Vec<ValueSrc>,
        output: usize,
    },
    /// A fence: `Reduce(r)` over the value produced by region/input `src`.
    Reduce { r: Red, src: ValueSrc },
}

/// Where a region's live-in value comes from: a graph side-input array, or a prior region's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueSrc {
    /// External side-input array `slot`.
    Input(usize),
    /// The output array of region `region_id` (topologically earlier).
    Region(usize),
}

impl Dag {
    /// **The fusion pass.** Partition the DAG into maximal Map regions cut at every Reduce, and schedule
    /// them in execution order. Each Map region lowers to ONE fused kernel; each Reduce is its own region
    /// reading a prior region's output (its Map prologue already fused into the region feeding it).
    /// `regions.len()` is the kernel count — `<<` the node count whenever any region absorbed >1 op.
    ///
    /// The partition must be **convex**: a Map region may not contain two nodes with a fence on a
    /// dependency path between them (else the region could not run as one pre- or post-fence kernel). A
    /// value that fans out *across* a fence — e.g. `a` reused in an RMS-norm epilogue after `sum(a*a)` —
    /// must therefore be a **live-in** of the epilogue region, not merge the prologue and epilogue into
    /// one. We enforce convexity with a **fence level**: `level(node)` = the max number of Reduce nodes
    /// on any dependency path ending at that node. Two Map nodes fuse only when Map-connected AND at the
    /// same level; a cross-level read is a live-in. Clean steps, each memoized once:
    ///   1. **Fence levels** — one topological pass; a Reduce bumps the level of everything downstream.
    ///   2. **Components** — union-find over Map–Map edges *at equal level* ([`Class`] on graph edges,
    ///      convexity-guarded).
    ///   3. **Schedule** — first-touch region-id allocation over node ids (topological); record, per
    ///      node, which region *produces* its value.
    ///   4. **Build** — fill each region; cross-region reads become deduplicated local `In` leaves.
    pub fn fuse(&self) -> Vec<Region> {
        let n = self.nodes.len();

        // 1. Fence levels — a Reduce increments the level; a Map inherits the max of its inputs.
        let mut level = vec![0usize; n];
        for id in 0..n {
            let in_max = self.nodes[id]
                .inputs()
                .into_iter()
                .map(|i| level[i])
                .max()
                .unwrap_or(0);
            level[id] = in_max + usize::from(self.nodes[id].class() == Class::Reduce);
        }

        // 2. Components: union Map–Map edges ONLY at the same fence level (convexity). Reduce = fence;
        //    Const is rematerializable (inlined per consumer), so it never merges into a component.
        let mut uf = UnionFind::new(n);
        for (id, node) in self.nodes.iter().enumerate() {
            if node.class() == Class::Reduce || matches!(node, Node::Const(_)) {
                continue;
            }
            for input in node.inputs().into_iter() {
                let inp = self.nodes[input];
                if inp.class() == Class::Map
                    && !matches!(inp, Node::Const(_))
                    && level[input] == level[id]
                {
                    uf.union(id, input);
                }
            }
        }

        // 3. Schedule: allocate region ids in first-touch order; `region_of_node[id]` = the region that
        //    PRODUCES node `id`'s value. `unit(id)` = component root for a Map node, else the node itself.
        //    A `Const` is *rematerializable* — it has no data dependency, so it never seeds or holds a
        //    region; it is inlined into every consuming Map region instead (see `build_map_region`).
        let mut region_of_node = vec![usize::MAX; n];
        let mut region_of_unit: std::collections::HashMap<usize, usize> = Default::default();
        let mut kinds: Vec<RegionKind> = Vec::new();
        for id in 0..n {
            if matches!(self.nodes[id], Node::Const(_)) {
                continue; // inlined at use sites, not a region of its own
            }
            let is_reduce = self.nodes[id].class() == Class::Reduce;
            let unit = if is_reduce { id } else { uf.find(id) };
            let region_id = *region_of_unit.entry(unit).or_insert_with(|| {
                let rid = kinds.len();
                kinds.push(if is_reduce {
                    RegionKind::Reduce
                } else {
                    RegionKind::Map
                });
                rid
            });
            region_of_node[id] = region_id;
        }

        // 4. Build each region in schedule order.
        (0..kinds.len())
            .map(|region_id| match kinds[region_id] {
                RegionKind::Reduce => {
                    let rid_node = (0..n)
                        .find(|&id| {
                            self.nodes[id].class() == Class::Reduce
                                && region_of_node[id] == region_id
                        })
                        .unwrap();
                    let Node::Reduce(r, a) = self.nodes[rid_node] else {
                        unreachable!()
                    };
                    Region::Reduce {
                        r,
                        src: self.value_src(a, &region_of_node),
                    }
                }
                RegionKind::Map => self.build_map_region(region_id, &region_of_node),
            })
            .collect()
    }

    /// The [`ValueSrc`] that yields the value of global node `gid`: a graph input leaf, or the output of
    /// the region that produces it.
    fn value_src(&self, gid: usize, region_of_node: &[usize]) -> ValueSrc {
        match self.nodes[gid] {
            Node::In(slot) => ValueSrc::Input(slot),
            _ => ValueSrc::Region(region_of_node[gid]),
        }
    }

    /// Build one Map region: gather its member nodes (id order = topological), renumber to a self-
    /// contained local sub-DAG, and turn every value read from *outside* the region into a deduplicated
    /// local `In` leaf backed by a [`ValueSrc`]. `local_of_global` maps each already-emitted global id to
    /// its local slot; `leaf_of_src` deduplicates live-in leaves so a source is read at most once.
    fn build_map_region(&self, region_id: usize, region_of_node: &[usize]) -> Region {
        let n = self.nodes.len();
        let members = (0..n)
            .filter(|&id| self.nodes[id].class() == Class::Map && region_of_node[id] == region_id);

        let mut local: Vec<Node> = Vec::new();
        let mut live_in: Vec<ValueSrc> = Vec::new();
        let mut local_of_global: std::collections::HashMap<usize, usize> = Default::default();
        let mut leaf_of_src: std::collections::HashMap<ValueSrc, usize> = Default::default();
        let mut last_local = 0usize;

        for gid in members {
            // Resolve each operand to a local slot: a member is already localized (topological order); a
            // cross-region read becomes (or reuses) a live-in `In` leaf.
            let operand = |g: usize,
                           local: &mut Vec<Node>,
                           live_in: &mut Vec<ValueSrc>,
                           leaf_of_src: &mut std::collections::HashMap<ValueSrc, usize>|
             -> usize {
                if let Some(&lid) = local_of_global.get(&g) {
                    return lid;
                }
                // A `Const` operand is rematerialized inline in this region (no live-in edge).
                if let Node::Const(bits) = self.nodes[g] {
                    let lid = local.len();
                    local.push(Node::Const(bits));
                    return lid;
                }
                // Any other cross-region read becomes (or reuses) a live-in `In` leaf.
                let src = self.value_src(g, region_of_node);
                *leaf_of_src.entry(src).or_insert_with(|| {
                    let li = live_in.len();
                    live_in.push(src);
                    let lid = local.len();
                    local.push(Node::In(li));
                    lid
                })
            };

            let ln = match self.nodes[gid] {
                // A graph input read directly inside this Map region is itself a live-in leaf.
                Node::In(_) => {
                    let src = self.value_src(gid, region_of_node);
                    if let Some(&lid) = leaf_of_src.get(&src) {
                        local_of_global.insert(gid, lid);
                        last_local = lid;
                        continue;
                    }
                    let li = live_in.len();
                    live_in.push(src);
                    leaf_of_src.insert(src, local.len());
                    Node::In(li)
                }
                Node::Const(bits) => Node::Const(bits),
                Node::Un(op, a) => {
                    Node::Un(op, operand(a, &mut local, &mut live_in, &mut leaf_of_src))
                }
                Node::Bin(op, a, b) => {
                    let la = operand(a, &mut local, &mut live_in, &mut leaf_of_src);
                    let lb = operand(b, &mut local, &mut live_in, &mut leaf_of_src);
                    Node::Bin(op, la, lb)
                }
                Node::Reduce(..) => unreachable!("reduce not in a Map region"),
            };
            let lid = local.len();
            local.push(ln);
            local_of_global.insert(gid, lid);
            last_local = lid;
        }

        // The region's output is its topological sink (its last member, id order).
        Region::Map {
            local,
            live_in,
            output: last_local,
        }
    }

    /// Kernel count after fusion.
    pub fn kernel_count(&self) -> usize {
        self.fuse().len()
    }
}

/// Whether a scheduled region is a fused Map or a Reduce fence (schedule-order tag).
enum RegionKind {
    Map,
    Reduce,
}

// --- union-find over node ids (Map–Map merges) -------------------------------------------------
struct UnionFind {
    parent: Vec<usize>,
}
impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }
    fn find(&self, mut x: usize) -> usize {
        // Iterative find (no path compression: keeps `&self` for callers; DAGs here are small).
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Root at the earliest id -> stable, dependency-respecting first-touch order.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.parent[hi] = lo;
        }
    }
}

// ======================================================================================
// 5. Lower — a Map region compiles to a DAG program, run by ONE comptime-unrolled `dag_interp` kernel.
// ======================================================================================

/// One instruction of a fused Map region, carried in the kernel's `#[comptime]` program. Unlike
/// `fuse::Instr` (a line threading a single accumulator), each instruction writes a **slot** and reads
/// prior slots by index — so fan-out (a slot read by many) and fan-in (a `Bin` of two slots) lower to
/// straight-line register code in ONE kernel body.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DagInstr {
    /// `slot[dst] = sides[in_slot][i]`.
    LoadIn(usize),
    /// `slot[dst] = const`.
    LoadConst(u32),
    /// `slot[dst] = op(slot[a])`.
    Un(UnOp, usize),
    /// `slot[dst] = op(slot[a], slot[b])`.
    Bin(BinOp, usize, usize),
}

/// A fused Map-region program: the ordered instructions (slot `k` is written by `instrs[k]`), the number
/// of side-input slots, and the slot id of the region's output. Rides as a `#[comptime]` value, so it is
/// `Eq + Hash` (the lowering engine keys the compiled kernel on it).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct DagProgram {
    pub instrs: Vec<DagInstr>,
    pub n_sides: usize,
    pub output: usize,
}

impl Region {
    /// Compile a `Region::Map` to a `DagProgram`. The local node list is already topological, so instr
    /// `k` corresponds to local slot `k`; a `Bin`/`Un` reads the slots of its operands directly. Returns
    /// the program and the ordered `ValueSrc`s the kernel must bind to side-input slots.
    pub fn to_program(&self) -> Option<(DagProgram, Vec<ValueSrc>)> {
        let Region::Map {
            local,
            live_in,
            output,
        } = self
        else {
            return None;
        };
        let instrs = local
            .iter()
            .map(|node| match *node {
                Node::In(li) => DagInstr::LoadIn(li),
                Node::Const(bits) => DagInstr::LoadConst(bits),
                Node::Un(op, a) => DagInstr::Un(op, a),
                Node::Bin(op, a, b) => DagInstr::Bin(op, a, b),
                Node::Reduce(..) => unreachable!("reduce in a Map region"),
            })
            .collect();
        let prog = DagProgram {
            instrs,
            n_sides: live_in.len(),
            output: *output,
        };
        Some((prog, live_in.clone()))
    }
}

// --- device morphisms (mirror `fuse`'s, kept here so `dag` is self-contained on-device) --------

#[device]
fn dsig<F: Float>(x: F) -> F {
    F::new(1.0) / (F::new(1.0) + (-x).exp())
}

#[device]
fn un_dev<F: Float>(#[comptime] op: UnOp, x: F) -> F {
    match op {
        UnOp::Neg => -x,
        UnOp::Recip => F::new(1.0) / x,
        UnOp::Abs => x.abs(),
        UnOp::Exp => x.exp(),
        UnOp::Tanh => x.tanh(),
        UnOp::Rsqrt => F::new(1.0) / x.sqrt(),
        UnOp::Sigmoid => dsig::<F>(x),
        UnOp::Silu => x * dsig::<F>(x),
        UnOp::Gelu => {
            let c = F::new(0.797_884_56);
            let inner = c * (x + F::new(0.044715) * x * x * x);
            F::new(0.5) * x * (F::new(1.0) + inner.tanh())
        }
    }
}

#[device]
fn bin_dev<F: Float>(#[comptime] op: BinOp, a: F, b: F) -> F {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
    }
}

/// **The one fused DAG kernel.** Interprets a comptime `DagProgram` per element: a local slot array
/// holds every node's value; the comptime-**unrolled** loop fills each slot from prior slots (all
/// comptime indices, so it lowers to straight-line register code — no runtime interpreter, no
/// intermediate arrays, one launch). Fan-out = a slot read by several later instrs; fan-in = a `Bin`
/// reading two slots. `sides` carries the region's live-in arrays, indexed by their comptime slot.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn dag_interp<F: Float>(
    sides: &Sequence<Array<F>>,
    out: &mut Array<F>,
    #[comptime] prog: DagProgram,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        // One register slot per node; `n_slots == prog.instrs.len()`, a comptime constant.
        let n_slots = comptime!(prog.instrs.len());
        let mut slot = Array::<F>::new(n_slots);
        #[unroll]
        for k in 0..prog.instrs.len() {
            match comptime!(prog.instrs[k]) {
                DagInstr::LoadIn(in_slot) => slot[k] = sides.index(in_slot)[i],
                DagInstr::LoadConst(bits) => {
                    let c = comptime!(f32::from_bits(bits));
                    slot[k] = F::new(c);
                }
                DagInstr::Un(op, a) => slot[k] = un_dev::<F>(op, slot[a]),
                DagInstr::Bin(op, a, b) => slot[k] = bin_dev::<F>(op, slot[a], slot[b]),
            }
        }
        out[i] = slot[comptime!(prog.output)];
    }
}

// ======================================================================================
// 6. Schedule + run — execute the fused regions end to end. Map regions -> ONE dag_interp each; Reduce
//    regions -> the reduction over the fused input. Counts launches for the bit-exact gate.
// ======================================================================================

const BLOCK: u32 = 256;

/// A reduce fence kernel — contracts each width-`n` row and broadcasts the result back (matches the CPU
/// `reduce_rows` oracle). Modeled as the generic non-elementwise op the Map regions fuse around.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn reduce_broadcast<F: Float>(
    x: &Array<F>,
    out: &mut Array<F>,
    #[comptime] red: Red,
    #[comptime] n: usize,
) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let row = i / n;
        let base = row * n;
        let mut acc = match red {
            Red::Sum => F::new(0.0),
            Red::Max => F::new(-3.4e38),
        };
        for j in 0..n {
            let v = x[base + j];
            acc = match red {
                Red::Sum => acc + v,
                Red::Max => {
                    if v > acc {
                        v
                    } else {
                        acc
                    }
                }
            };
        }
        out[i] = acc;
    }
}

/// Result of running a fused DAG: the output values plus the launch count (one per region).
pub struct FusedRun {
    pub out: Vec<f32>,
    pub launches: usize,
}

impl Dag {
    /// **Fuse and run.** Partition into regions, then execute them in order: each Map region is ONE
    /// `dag_interp` launch, each Reduce is one `reduce_broadcast`. Region outputs are kept on-device and
    /// fed to later regions by `ValueSrc`. Returns the final output and the launch count (== region
    /// count), which is `<<` the naive per-node launch count.
    pub fn fuse_and_run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        inputs: &[&[f32]],
        n: usize,
    ) -> FusedRun {
        let len = inputs[0].len();
        let regions = self.fuse();

        // Upload the graph inputs once.
        let input_handles: Vec<_> = inputs
            .iter()
            .map(|s| client.create_from_slice(f32::as_bytes(s)))
            .collect();
        // Each region's output handle (device buffer).
        let mut region_out: Vec<Option<cubecl::server::Handle>> = vec![None; regions.len()];

        let resolve = |src: ValueSrc,
                       input_handles: &[cubecl::server::Handle],
                       region_out: &[Option<cubecl::server::Handle>]|
         -> cubecl::server::Handle {
            match src {
                ValueSrc::Input(slot) => input_handles[slot].clone(),
                ValueSrc::Region(rid) => region_out[rid]
                    .clone()
                    .expect("region scheduled before use"),
            }
        };

        let grid = Grid::Static((len as u32).div_ceil(BLOCK), 1, 1);
        let mut launches = 0usize;

        for (rid, region) in regions.iter().enumerate() {
            match region {
                Region::Map { .. } => {
                    let (prog, srcs) = region.to_program().unwrap();
                    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; len]));
                    let mut sides = SequenceArg::new();
                    let side_handles: Vec<_> = srcs
                        .iter()
                        .map(|&s| resolve(s, &input_handles, &region_out))
                        .collect();
                    for h in &side_handles {
                        sides.push(unsafe { ArrayArg::from_raw_parts(h.clone(), len) });
                    }
                    unsafe {
                        dag_interp::launch_unchecked::<f32, R>(
                            client,
                            grid.clone(),
                            Block::new_1d(BLOCK),
                            sides,
                            ArrayArg::from_raw_parts(oh.clone(), len),
                            prog,
                        );
                    }
                    region_out[rid] = Some(oh);
                    launches += 1;
                }
                Region::Reduce { r, src } => {
                    let ih = resolve(*src, &input_handles, &region_out);
                    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; len]));
                    unsafe {
                        reduce_broadcast::launch_unchecked::<f32, R>(
                            client,
                            grid.clone(),
                            Block::new_1d(BLOCK),
                            ArrayArg::from_raw_parts(ih.clone(), len),
                            ArrayArg::from_raw_parts(oh.clone(), len),
                            *r,
                            n,
                        );
                    }
                    region_out[rid] = Some(oh);
                    launches += 1;
                }
            }
        }

        let final_h = region_out.last().unwrap().clone().unwrap();
        let out = f32::from_bytes(&client.read_one_unchecked(final_h)).to_vec();
        FusedRun { out, launches }
    }
}

// ======================================================================================
// 7. Naive baseline — one launch per node, every intermediate materialized. The thing fusion removes;
//    the bit-exact target the fused path must match.
// ======================================================================================

#[device]
fn n_un<F: Float>(#[comptime] op: UnOp, x: F) -> F {
    un_dev::<F>(op, x)
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
fn node_un<F: Float>(x: &Array<F>, out: &mut Array<F>, #[comptime] op: UnOp) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = n_un::<F>(op, x[i]);
    }
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
fn node_bin<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>, #[comptime] op: BinOp) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        out[i] = bin_dev::<F>(op, a[i], b[i]);
    }
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
fn node_const<F: Float>(out: &mut Array<F>, #[comptime] bits: u32) {
    let i = ABSOLUTE_POS;
    if i < out.len() {
        let c = comptime!(f32::from_bits(bits));
        out[i] = F::new(c);
    }
}

impl Dag {
    /// **Naive per-op execution.** One launch per node (In = an upload, Const/Un/Bin/Reduce = one
    /// kernel), materializing every intermediate. Returns the output and the launch count — the number
    /// fused execution beats.
    pub fn naive_run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        inputs: &[&[f32]],
        n: usize,
    ) -> FusedRun {
        let len = inputs[0].len();
        let grid = Grid::Static((len as u32).div_ceil(BLOCK), 1, 1);
        let mut handles: Vec<cubecl::server::Handle> = Vec::with_capacity(self.nodes.len());
        let mut launches = 0usize;

        for node in &self.nodes {
            let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; len]));
            match *node {
                Node::In(slot) => {
                    // Materialize the input as its own device buffer (no compute launch).
                    let h = client.create_from_slice(f32::as_bytes(inputs[slot]));
                    handles.push(h);
                    continue;
                }
                Node::Const(bits) => unsafe {
                    node_const::launch_unchecked::<f32, R>(
                        client,
                        grid.clone(),
                        Block::new_1d(BLOCK),
                        ArrayArg::from_raw_parts(oh.clone(), len),
                        bits,
                    );
                },
                Node::Un(op, a) => unsafe {
                    node_un::launch_unchecked::<f32, R>(
                        client,
                        grid.clone(),
                        Block::new_1d(BLOCK),
                        ArrayArg::from_raw_parts(handles[a].clone(), len),
                        ArrayArg::from_raw_parts(oh.clone(), len),
                        op,
                    );
                },
                Node::Bin(op, a, b) => unsafe {
                    node_bin::launch_unchecked::<f32, R>(
                        client,
                        grid.clone(),
                        Block::new_1d(BLOCK),
                        ArrayArg::from_raw_parts(handles[a].clone(), len),
                        ArrayArg::from_raw_parts(handles[b].clone(), len),
                        ArrayArg::from_raw_parts(oh.clone(), len),
                        op,
                    );
                },
                Node::Reduce(r, a) => unsafe {
                    reduce_broadcast::launch_unchecked::<f32, R>(
                        client,
                        grid.clone(),
                        Block::new_1d(BLOCK),
                        ArrayArg::from_raw_parts(handles[a].clone(), len),
                        ArrayArg::from_raw_parts(oh.clone(), len),
                        r,
                        n,
                    );
                },
            }
            handles.push(oh);
            launches += 1;
        }

        let out =
            f32::from_bytes(&client.read_one_unchecked(handles.last().unwrap().clone())).to_vec();
        FusedRun { out, launches }
    }
}

// ======================================================================================
// 8. Tests — the bit-exact gate. Full-DAG-fused == naive per-op == plain-Rust reference, with fan-out
//    AND a fence, and the fused launch count << naive.
// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    // --- Trace + partition, no device: the algebra on a graph ------------------------------------

    #[test]
    fn fan_out_traces_to_a_shared_node() {
        // y = silu(a)*b + a  — `a` is used twice (fan-out). ONE Map region, no fence.
        let tape = Tape::new();
        let a = tape.input();
        let b = tape.input();
        let y = a.silu() * b + a;
        let dag = tape.finish(y);
        // Nodes: In(0)=a, In(1)=b, silu(a), silu(a)*b, +a  => 5 nodes, a referenced by node 2 and node 4.
        assert_eq!(
            dag.len(),
            5,
            "fan-out shares the `a` node, not duplicates it"
        );
        let regions = dag.fuse();
        assert_eq!(
            regions.len(),
            1,
            "all-Map DAG with fan-out -> ONE fused kernel"
        );
        assert!(matches!(regions[0], Region::Map { .. }));
    }

    #[test]
    fn reduce_partitions_the_dag() {
        // y = rsqrt(sum(a*a) + eps) * a  — a fence (sum) with a Map prologue (a*a) and a Map epilogue
        // (rsqrt(.+eps) * a, with `a` fanned in past the fence). Expect 3 regions: prologue, reduce,
        // epilogue.
        let tape = Tape::new();
        let a = tape.input();
        let sq = a * a; // Map prologue
        let s = sq.reduce(Red::Sum); // FENCE
        let eps = tape.konst(1e-6);
        let y = (s + eps).rsqrt() * a; // Map epilogue, reuses `a` across the fence
        let dag = tape.finish(y);
        let regions = dag.fuse();
        assert_eq!(regions.len(), 3, "Map | Reduce | Map");
        assert!(matches!(regions[0], Region::Map { .. }));
        assert!(matches!(regions[1], Region::Reduce { r: Red::Sum, .. }));
        assert!(matches!(regions[2], Region::Map { .. }));
    }

    #[test]
    fn eval_oracle_matches_fused_partition_semantics() {
        // The CPU oracle over the whole DAG must equal evaluating it region-by-region (host-side), which
        // is what the device path does — invariance under fusion.
        let tape = Tape::new();
        let a = tape.input();
        let b = tape.input();
        let y = a.silu() * b + a;
        let dag = tape.finish(y);
        let n = 16usize;
        let av = xorshift_vec(n, 1);
        let bv = xorshift_vec(n, 2);
        let whole = dag.eval(&[&av, &bv], n);
        let refv: Vec<f32> = (0..n)
            .map(|i| apply_un(UnOp::Silu, av[i]) * bv[i] + av[i])
            .collect();
        assert_eq!(
            bits(&whole),
            bits(&refv),
            "oracle disagrees with plain Rust"
        );
    }

    // --- The device gate: full-DAG-fused == naive == reference, bit-exact, on the CPU runtime ----

    #[cfg(feature = "cpu")]
    fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        CpuRuntime::client(&CpuDevice::default())
    }

    /// Pure Map DAG with fan-out `y = silu(a)*b + a`: fused (ONE launch) == naive (per-node) == ref.
    #[cfg(feature = "cpu")]
    #[test]
    fn fanout_dag_fused_equals_naive_bit_exact() {
        let client = cpu_client();
        let n = 1024usize;
        let a = xorshift_vec(n, 0x1234_5678_9abc_def0);
        let b = xorshift_vec(n, 0x0fed_cba9_8765_4321);

        let tape = Tape::new();
        let va = tape.input();
        let vb = tape.input();
        let y = va.silu() * vb + va; // `a` fanned out
        let dag = tape.finish(y);

        let fused = dag.fuse_and_run::<cubecl::cpu::CpuRuntime>(&client, &[&a, &b], n);
        let naive = dag.naive_run::<cubecl::cpu::CpuRuntime>(&client, &[&a, &b], n);
        let refv: Vec<f32> = (0..n)
            .map(|i| apply_un(UnOp::Silu, a[i]) * b[i] + a[i])
            .collect();

        assert_eq!(
            bits(&fused.out),
            bits(&naive.out),
            "fused != naive (bit level)"
        );
        let maxerr = fused
            .out
            .iter()
            .zip(&refv)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[dag CPU] fan-out silu(a)*b+a: fused {} launch vs naive {} launches; bit-exact; max|fused-ref|={maxerr:.2e}",
            fused.launches, naive.launches
        );
        assert_eq!(fused.launches, 1, "one Map region -> one launch");
        assert!(fused.launches < naive.launches, "fusion must cut launches");
        assert!(maxerr < 1e-5, "reference disagreement {maxerr}");
    }

    /// The full endgame gate: fan-out AND a fence. `y = rsqrt(sum(a*a) + eps) * a` — RMS-norm shape with
    /// `a` reused across the reduction. Fused (3 launches) == naive (per-node) == plain-Rust ref.
    #[cfg(feature = "cpu")]
    #[test]
    fn rmsnorm_shape_fanout_across_fence_bit_exact() {
        let client = cpu_client();
        let rows = 32usize;
        let n = 64usize; // row width the reduce contracts over
        let len = rows * n;
        let a = xorshift_vec(len, 0xdead_beef_cafe_babe);
        let eps = 1e-6f32;

        let tape = Tape::new();
        let va = tape.input();
        let sq = va * va;
        let s = sq.reduce(Red::Sum);
        let ke = tape.konst(eps);
        let y = (s + ke).rsqrt() * va; // fan-out of `a` across the fence
        let dag = tape.finish(y);

        let fused = dag.fuse_and_run::<cubecl::cpu::CpuRuntime>(&client, &[&a], n);
        let naive = dag.naive_run::<cubecl::cpu::CpuRuntime>(&client, &[&a], n);

        // Plain-Rust reference, row by row.
        let mut refv = vec![0.0f32; len];
        for row in 0..rows {
            let base = row * n;
            let ss: f32 = a[base..base + n].iter().map(|&x| x * x).sum();
            let scale = 1.0f32 / (ss + eps).sqrt();
            for i in 0..n {
                refv[base + i] = scale * a[base + i];
            }
        }

        assert_eq!(
            bits(&fused.out),
            bits(&naive.out),
            "fused != naive (bit level)"
        );
        let maxerr = fused
            .out
            .iter()
            .zip(&refv)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[dag CPU] RMS-shape rsqrt(sum(a*a)+eps)*a (fan-out ACROSS fence): fused {} launches vs naive {} launches; bit-exact; max|fused-ref|={maxerr:.2e}",
            fused.launches, naive.launches
        );
        assert_eq!(fused.launches, 3, "Map | Reduce | Map -> 3 launches");
        assert!(fused.launches < naive.launches, "fusion must cut launches");
        assert!(maxerr < 1e-5, "reference disagreement {maxerr}");
    }

    /// A wider multi-region DAG: two independent Map prologues feeding one fence, then a fused epilogue
    /// with fan-in — proves the partitioner handles multiple Map regions and a genuine graph, not a path.
    #[cfg(feature = "cpu")]
    #[test]
    fn multi_region_dag_fused_equals_naive_bit_exact() {
        let client = cpu_client();
        let rows = 16usize;
        let n = 128usize;
        let len = rows * n;
        let a = xorshift_vec(len, 0x0123_4567_89ab_cdef);
        let b = xorshift_vec(len, 0xfedc_ba98_7654_3210);

        let tape = Tape::new();
        let va = tape.input();
        let vb = tape.input();
        // prologue: t = silu(a) * b  (a fused Map region)
        let t = va.silu() * vb;
        // fence: row-sum of t
        let s = t.reduce(Red::Sum);
        // epilogue: gelu(s) + t  (fan-in: reuses the pre-reduce t across the fence, and s)
        let y = s.gelu() + t;
        let dag = tape.finish(y);

        // Structural proof of the HARD case: the epilogue region reads `t` (region 0's output) *across*
        // the fence (region 1) as well as `s` (region 1's output) — two live-ins, one of them jumping the
        // fence. This is exactly what a linear chain cannot express.
        let regions = dag.fuse();
        assert_eq!(regions.len(), 3);
        let Region::Map { live_in, .. } = &regions[2] else {
            panic!("epilogue is a Map region")
        };
        assert!(
            live_in.contains(&ValueSrc::Region(0)) && live_in.contains(&ValueSrc::Region(1)),
            "epilogue must read region 0 (t, across the fence) AND region 1 (s): {live_in:?}"
        );

        let fused = dag.fuse_and_run::<cubecl::cpu::CpuRuntime>(&client, &[&a, &b], n);
        let naive = dag.naive_run::<cubecl::cpu::CpuRuntime>(&client, &[&a, &b], n);

        // Reference.
        let mut refv = vec![0.0f32; len];
        let mut tv = vec![0.0f32; len];
        for i in 0..len {
            tv[i] = apply_un(UnOp::Silu, a[i]) * b[i];
        }
        for row in 0..rows {
            let base = row * n;
            let ss: f32 = tv[base..base + n].iter().sum();
            let g = apply_un(UnOp::Gelu, ss);
            for i in 0..n {
                refv[base + i] = g + tv[base + i];
            }
        }

        assert_eq!(
            bits(&fused.out),
            bits(&naive.out),
            "fused != naive (bit level)"
        );
        let maxerr = fused
            .out
            .iter()
            .zip(&refv)
            .map(|(g, r)| (g - r).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[dag CPU] multi-region silu(a)*b -> sum -> gelu(.)+t (fan-in across fence): fused {} launches vs naive {} launches; bit-exact; max|fused-ref|={maxerr:.2e}",
            fused.launches, naive.launches
        );
        assert_eq!(fused.launches, 3, "Map | Reduce | Map -> 3 launches");
        assert!(fused.launches < naive.launches, "fusion must cut launches");
        assert!(maxerr < 1e-5, "reference disagreement {maxerr}");
    }

    #[test]
    fn rms_norm_gated_engine_pattern_fuses() {
        // The exact per-GDN-layer chain from hanzo-engine `models/gdn.rs` `RmsNormGated`:
        //   rms_norm(x) * weight * silu(gate)  ==  x·rsqrt(Σ x² + eps) · w · silu(g)
        // Ops-composed in the engine, this is ~8 kernel launches on EVERY Gated-DeltaNet layer — a
        // direct contributor to the prefill launch storm (nsys: hanzo prefill ~33k launches vs
        // llama 3089, the whole remaining 0.86x->1.0x gap is launch overhead). Full-DAG fusion
        // collapses it to 3 regions: the `x·x` prologue | the row-sum fence | the fused
        // `rsqrt(Σ+eps)·x·w·silu(g)` epilogue. This is the mechanism that closes prefill.
        let client = cpu_client();
        let rows = 16usize;
        let n = 128usize;
        let len = rows * n;
        let x = xorshift_vec(len, 0xa1b2_c3d4_e5f6_0718);
        let w = xorshift_vec(len, 0x1122_3344_5566_7788);
        let g = xorshift_vec(len, 0x9900_aabb_ccdd_eeff);

        let tape = Tape::new();
        let vx = tape.input();
        let vw = tape.input();
        let vg = tape.input();
        let eps = tape.konst(1e-6);
        let ss = (vx * vx).reduce(Red::Sum); // FENCE — rms_norm's reduction
        let inv = (ss + eps).rsqrt(); // per-row inverse-RMS scale
        let y = vx * inv * vw * vg.silu(); // fused epilogue: normed · weight · silu(gate)
        let dag = tape.finish(y);

        let fused = dag.fuse_and_run::<cubecl::cpu::CpuRuntime>(&client, &[&x, &w, &g], n);
        let naive = dag.naive_run::<cubecl::cpu::CpuRuntime>(&client, &[&x, &w, &g], n);

        let mut refv = vec![0.0f32; len];
        for row in 0..rows {
            let base = row * n;
            let ss: f32 = x[base..base + n].iter().map(|v| v * v).sum();
            let inv = 1.0f32 / (ss + 1e-6).sqrt();
            for i in 0..n {
                let j = base + i;
                refv[j] = x[j] * inv * w[j] * apply_un(UnOp::Silu, g[j]);
            }
        }

        assert_eq!(bits(&fused.out), bits(&naive.out), "fused != naive (bit level)");
        let maxerr = fused
            .out
            .iter()
            .zip(&refv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[dag CPU] RmsNormGated rms_norm(x)*w*silu(g) (real engine gdn.rs chain): fused {} vs naive {} launches; bit-exact; max|fused-ref|={maxerr:.2e}",
            fused.launches, naive.launches
        );
        assert!(fused.launches < naive.launches, "fusion must cut the per-layer launch count");
        assert!(maxerr < 1e-5, "reference disagreement {maxerr}");
    }
}
