//! Purpose (HIP-1334 §3) and the lineage walk promotion reads.
//!
//! Every item and artifact carries one [`Purpose`], set when written. A derived artifact takes
//! its purpose from its parents ([`derive`]); nothing derives from evaluation data. Promotion
//! walks the artifact graph by id and parent edges ([`verify_promotion_eligibility`]) and
//! passes only when every check holds.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// What an item or artifact may be used for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    /// Gradient steps, selection, derivation, teacher queries.
    Train,
    /// Selection, early stopping, calibration; never a gradient step.
    Dev,
    /// Evaluation and observation; nothing is derived from it.
    EvalOnly,
    /// Scored by the server only; its gold never leaves the store.
    SealedEval,
    /// Sent to a teacher; trains only through what is derived from the answers.
    TeacherMining,
}

/// How a run reads a dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Gradient steps.
    Train,
    /// Selection, early stopping, calibration.
    Select,
    /// An evaluation scores it.
    Evaluate,
    /// A teacher or another system is asked about it.
    Query,
}

impl Purpose {
    /// Whether a run may read data of this purpose in `role` (HIP-1334 §3.2).
    pub fn admits(self, role: Role) -> bool {
        use Purpose::*;
        match role {
            Role::Train => self == Train,
            Role::Select => matches!(self, Train | Dev),
            Role::Evaluate => matches!(self, Train | Dev | EvalOnly | SealedEval),
            Role::Query => matches!(self, Train | Dev | EvalOnly | TeacherMining),
        }
    }

    /// Whether anything may be derived from data of this purpose.
    pub fn derivation(self) -> bool {
        matches!(self, Purpose::Train | Purpose::Dev | Purpose::TeacherMining)
    }

    /// The wire name.
    pub fn name(self) -> &'static str {
        match self {
            Purpose::Train => "train",
            Purpose::Dev => "dev",
            Purpose::EvalOnly => "eval_only",
            Purpose::SealedEval => "sealed_eval",
            Purpose::TeacherMining => "teacher_mining",
        }
    }
}

/// The purpose of an artifact derived from `parents` (HIP-1334 §3.3): refused when any parent
/// forbids derivation, `dev` when any is `dev`, else `train`.
pub fn derive(parents: &[Purpose]) -> Result<Purpose, Purpose> {
    if let Some(&p) = parents.iter().find(|p| !p.derivation()) {
        return Err(p);
    }
    Ok(if parents.contains(&Purpose::Dev) {
        Purpose::Dev
    } else {
        Purpose::Train
    })
}

/// One artifact as the lineage walk reads it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub sha256: String,
    pub kind: String,
    /// Absent on an artifact recorded before purpose existed.
    pub purpose: Option<Purpose>,
    pub parents: Vec<String>,
    pub git_sha: String,
    pub git_dirty: bool,
}

/// Where the walk reads nodes: the server's artifact store, or nodes it answered.
pub trait Store {
    /// The node addressed by `sha256`, or `None` when the store holds none.
    fn node(&mut self, sha256: &str) -> crate::Result<Option<Node>>;
}

impl Store for BTreeMap<String, Node> {
    fn node(&mut self, sha256: &str) -> crate::Result<Option<Node>> {
        Ok(self.get(sha256).cloned())
    }
}

/// One check of a lineage, with what it found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Check {
    pub name: String,
    pub pass: bool,
    pub detail: String,
}

/// A walk's nodes and every check over them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub sha256: String,
    pub nodes: Vec<Node>,
    pub checks: Vec<Check>,
}

impl Report {
    /// Every check ran and passed.
    pub fn pass(&self) -> bool {
        self.checks.len() == CHECKS.len() && self.checks.iter().all(|c| c.pass)
    }
}

/// The checks every walk reports, in order.
pub const CHECKS: [&str; 8] = [
    "bounded",
    "present",
    "acyclic",
    "lineage",
    "purpose",
    "derivation",
    "eval",
    "clean",
];

/// The most nodes one walk reads.
pub const MAX_NODES: usize = 10_000;

/// An iterative depth-first walk: `stack` holds (node, next parent index), `open` the nodes on
/// the current path, so a parent already open closes a cycle.
struct Walk {
    nodes: BTreeMap<String, Node>,
    missing: BTreeSet<String>,
    cycles: BTreeSet<String>,
    open: BTreeSet<String>,
    done: BTreeSet<String>,
    stack: Vec<(String, usize)>,
    bounded: bool,
}

impl Default for Walk {
    fn default() -> Walk {
        Walk {
            nodes: BTreeMap::new(),
            missing: BTreeSet::new(),
            cycles: BTreeSet::new(),
            open: BTreeSet::new(),
            done: BTreeSet::new(),
            stack: Vec::new(),
            bounded: true,
        }
    }
}

impl Walk {
    fn visit(&mut self, store: &mut impl Store, id: &str) -> crate::Result<()> {
        if self.open.contains(id) {
            self.cycles.insert(id.to_string());
        } else if !self.done.contains(id) && !self.missing.contains(id) {
            match store.node(id)? {
                Some(n) if n.sha256 == id => {
                    self.nodes.insert(id.to_string(), n);
                    self.open.insert(id.to_string());
                    self.stack.push((id.to_string(), 0));
                }
                _ => {
                    self.missing.insert(id.to_string());
                }
            }
        }
        Ok(())
    }
}

/// Walk `sha256`'s lineage in `store` and check it for promotion:
/// - `bounded`: at most [`MAX_NODES`] nodes;
/// - `present`: the subject and every ancestor it names are in the store;
/// - `acyclic`: no ancestor is its own ancestor;
/// - `lineage`: the subject names at least one parent;
/// - `purpose`: every node carries a purpose, and the subject's is `train`;
/// - `derivation`: every node with parents has the purpose [`derive`] gives them;
/// - `eval`: no node is `eval_only` or `sealed_eval`, at any depth;
/// - `clean`: every node names its commit, built from a clean tree.
///
/// The report is the verdict; [`Report::pass`] is true only when all hold.
pub fn verify_promotion_eligibility(store: &mut impl Store, sha256: &str) -> crate::Result<Report> {
    let mut w = Walk::default();
    w.visit(store, sha256)?;
    while let Some((id, i)) = w.stack.pop() {
        let parents = &w.nodes[&id].parents;
        if i < parents.len() {
            let next = parents[i].clone();
            w.stack.push((id, i + 1));
            if w.nodes.len() >= MAX_NODES && !w.nodes.contains_key(&next) {
                w.bounded = false;
                break;
            }
            w.visit(store, &next)?;
        } else {
            w.open.remove(&id);
            w.done.insert(id);
        }
    }
    let Walk {
        nodes,
        missing,
        cycles,
        bounded,
        ..
    } = w;

    let subject = nodes.get(sha256);
    let mut checks = Vec::new();
    let mut check = |name: &str, bad: Vec<String>| {
        checks.push(Check {
            name: name.into(),
            pass: bad.is_empty(),
            detail: if bad.is_empty() {
                "ok".into()
            } else {
                bad.join("; ")
            },
        });
    };
    check(
        "bounded",
        if bounded {
            vec![]
        } else {
            vec![format!("more than {MAX_NODES} nodes")]
        },
    );
    check(
        "present",
        missing.iter().map(|m| format!("{m} missing")).collect(),
    );
    check(
        "acyclic",
        cycles
            .iter()
            .map(|c| format!("{c} is its own ancestor"))
            .collect(),
    );
    check(
        "lineage",
        match subject {
            Some(s) if !s.parents.is_empty() => vec![],
            Some(_) => vec![format!("{sha256} names no parent")],
            None => vec![format!("{sha256} missing")],
        },
    );
    let mut purpose: Vec<String> = nodes
        .values()
        .filter(|n| n.purpose.is_none())
        .map(|n| format!("{} has no purpose", n.sha256))
        .collect();
    if let Some(p) = subject
        .and_then(|s| s.purpose)
        .filter(|&p| p != Purpose::Train)
    {
        purpose.push(format!("{sha256} is {}, not train", p.name()));
    }
    check("purpose", purpose);
    let derivation = nodes
        .values()
        .filter(|n| !n.parents.is_empty())
        .filter_map(|n| {
            let parents: Option<Vec<Purpose>> = n
                .parents
                .iter()
                .map(|p| nodes.get(p).and_then(|p| p.purpose))
                .collect();
            let parents = parents?;
            match (derive(&parents), n.purpose) {
                (Err(p), _) => Some(format!("{} derives from {} data", n.sha256, p.name())),
                (Ok(d), Some(have)) if d != have => Some(format!(
                    "{} is {}, its parents give {}",
                    n.sha256,
                    have.name(),
                    d.name()
                )),
                _ => None,
            }
        })
        .collect();
    check("derivation", derivation);
    check(
        "eval",
        nodes
            .values()
            .filter(|n| matches!(n.purpose, Some(Purpose::EvalOnly | Purpose::SealedEval)))
            .map(|n| format!("{} is {}", n.sha256, n.purpose.map_or("", Purpose::name)))
            .collect(),
    );
    check(
        "clean",
        nodes
            .values()
            .filter(|n| n.git_dirty || n.git_sha.is_empty())
            .map(|n| {
                if n.git_sha.is_empty() {
                    format!("{} names no commit", n.sha256)
                } else {
                    format!("{} built from a dirty tree", n.sha256)
                }
            })
            .collect(),
    );
    Ok(Report {
        sha256: sha256.to_string(),
        nodes: nodes.into_values().collect(),
        checks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, purpose: Option<Purpose>, parents: &[&str]) -> Node {
        Node {
            sha256: id.into(),
            kind: "checkpoint".into(),
            purpose,
            parents: parents.iter().map(|p| p.to_string()).collect(),
            git_sha: "a".repeat(40),
            git_dirty: false,
        }
    }

    fn store(nodes: Vec<Node>) -> BTreeMap<String, Node> {
        nodes.into_iter().map(|n| (n.sha256.clone(), n)).collect()
    }

    fn failed(r: &Report) -> Vec<&str> {
        r.checks
            .iter()
            .filter(|c| !c.pass)
            .map(|c| c.name.as_str())
            .collect()
    }

    #[test]
    fn matrix() {
        use Purpose::*;
        let yes = |p: Purpose| {
            [Role::Train, Role::Select, Role::Evaluate, Role::Query].map(|r| p.admits(r))
        };
        assert_eq!(yes(Train), [true, true, true, true]);
        assert_eq!(yes(Dev), [false, true, true, true]);
        assert_eq!(yes(EvalOnly), [false, false, true, true]);
        assert_eq!(yes(SealedEval), [false, false, true, false]);
        assert_eq!(yes(TeacherMining), [false, false, false, true]);
        assert_eq!(
            [Train, Dev, EvalOnly, SealedEval, TeacherMining].map(Purpose::derivation),
            [true, true, false, false, true]
        );
    }

    #[test]
    fn derivation() {
        use Purpose::*;
        assert_eq!(derive(&[]), Ok(Train));
        assert_eq!(derive(&[TeacherMining, Train]), Ok(Train));
        assert_eq!(derive(&[Train, Dev]), Ok(Dev));
        assert_eq!(derive(&[Train, EvalOnly]), Err(EvalOnly));
        assert_eq!(derive(&[Dev, SealedEval]), Err(SealedEval));
    }

    #[test]
    fn clean_lineage_passes() {
        use Purpose::*;
        let mut s = store(vec![
            node("data", Some(Train), &[]),
            node("base", Some(Train), &[]),
            node("ckpt", Some(Train), &["base", "data"]),
        ]);
        let r = verify_promotion_eligibility(&mut s, "ckpt").unwrap();
        assert!(r.pass(), "{:?}", r.checks);
        assert_eq!(
            r.checks.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(),
            CHECKS
        );
        assert_eq!(r.nodes.len(), 3);
    }

    /// An eval_only grandparent laundered through a train child into a train grandchild.
    #[test]
    fn eval_grandparent_is_refused() {
        use Purpose::*;
        let mut s = store(vec![
            node("eval", Some(EvalOnly), &[]),
            node("child", Some(Train), &["eval"]),
            node("grandchild", Some(Train), &["child"]),
        ]);
        let r = verify_promotion_eligibility(&mut s, "grandchild").unwrap();
        assert!(!r.pass());
        assert_eq!(failed(&r), ["derivation", "eval"]);
        let mut s = store(vec![
            node("sealed", Some(SealedEval), &[]),
            node("a", Some(Train), &["sealed"]),
            node("b", Some(Train), &["a"]),
            node("c", Some(Train), &["b"]),
        ]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "c").unwrap()),
            ["derivation", "eval"]
        );
    }

    #[test]
    fn missing_ancestor_is_refused() {
        use Purpose::*;
        let mut s = store(vec![
            node("a", Some(Train), &["gone"]),
            node("b", Some(Train), &["a"]),
        ]);
        let r = verify_promotion_eligibility(&mut s, "b").unwrap();
        assert_eq!(failed(&r), ["present"]);
        let mut s = store(vec![]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "x").unwrap()),
            ["present", "lineage"]
        );
    }

    #[test]
    fn cycle_is_refused() {
        use Purpose::*;
        let mut s = store(vec![
            node("a", Some(Train), &["b"]),
            node("b", Some(Train), &["c"]),
            node("c", Some(Train), &["a"]),
        ]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "a").unwrap()),
            ["acyclic"]
        );
        let mut s = store(vec![node("self", Some(Train), &["self"])]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "self").unwrap()),
            ["acyclic"]
        );
    }

    #[test]
    fn empty_lineage_is_refused() {
        let mut s = store(vec![node("root", Some(Purpose::Train), &[])]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "root").unwrap()),
            ["lineage"]
        );
    }

    #[test]
    fn unknown_purpose_dirty_and_dev_are_refused() {
        use Purpose::*;
        let mut s = store(vec![
            node("old", None, &[]),
            node("ckpt", Some(Train), &["old"]),
        ]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "ckpt").unwrap()),
            ["purpose"]
        );
        let mut dirty = node("ckpt", Some(Train), &["data"]);
        dirty.git_dirty = true;
        let mut s = store(vec![node("data", Some(Train), &[]), dirty]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "ckpt").unwrap()),
            ["clean"]
        );
        let mut s = store(vec![
            node("val", Some(Dev), &[]),
            node("ckpt", Some(Dev), &["val"]),
        ]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "ckpt").unwrap()),
            ["purpose"]
        );
        let mut s = store(vec![
            node("val", Some(Dev), &[]),
            node("ckpt", Some(Train), &["val"]),
        ]);
        assert_eq!(
            failed(&verify_promotion_eligibility(&mut s, "ckpt").unwrap()),
            ["derivation"]
        );
    }
}
