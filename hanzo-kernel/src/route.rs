//! Routing: the third class, **placed**.
//!
//! [`Class::Route`](crate::fuse::Class::Route) is the class of *applying a harvested expert*. Like
//! [`Reduce`](crate::fuse::Class::Reduce) it is a fusion fence — a Map run cannot be folded across it —
//! but where a Reduce fences *within* a device (its lanes must talk), a Route fences *across* devices: an
//! expert's weights live on the one device the placement map π gives them ([`crate::place`]), so a
//! request reaches an expert by moving to `π(expert)`. Whether that move costs anything is a property of
//! π and nothing else — two experts π co-locates cost no crossing though each is a Route; two experts π
//! splits cost one.
//!
//! **The class is decided by composition, the fence by placement, and neither ever reads what the expert
//! computes.** That is why a diffusion expert and a text expert route the same way: the partitioner sees
//! "a Route", never "a *diffusion* Route" (`muen.tex`; README §"The unbinding is free"). This module is
//! the smallest thing that makes it concrete — classify an expert application (always [`Class::Route`],
//! never a function of *which* expert), and, given a router's plan and a π, read off the crossings.

use crate::fuse::Class;
use crate::place::{Device, Expert, Place};

/// The class of applying an expert. A constant function of *nothing* about the expert — not its id, not
/// its modality, not whether it denoises or decodes. That constancy is the unbinding itself: the
/// partitioner is handed "a Route" and can never privilege one process over another, because there is no
/// per-expert information in the answer to privilege on.
pub fn expert_class() -> Class {
    Class::Route
}

/// A router's emission: the ordered experts one request visits. Placement turns it into a path over the
/// fleet, and the crossings fall out of π — the plan itself names no devices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    hops: Vec<Expert>,
}

impl FromIterator<Expert> for Plan {
    fn from_iter<I: IntoIterator<Item = Expert>>(hops: I) -> Plan {
        Plan { hops: hops.into_iter().collect() }
    }
}

impl Plan {
    /// The experts this plan visits, in order.
    pub fn hops(&self) -> &[Expert] {
        &self.hops
    }

    /// The device each hop runs on under π, or `None` at the first hop π does not place: an expert with
    /// no address cannot be routed to, and the plan fails closed rather than guess a device.
    pub fn devices(&self, pi: &Place) -> Option<Vec<Device>> {
        self.hops.iter().map(|&e| pi.device(e)).collect()
    }

    /// The hop indices at which the request **crosses to another device** under π — the Route fences that
    /// actually move data. A plan whose experts π co-locates has none, though every hop is still a Route:
    /// the class says a Route *may* cross, π says which ones *do*. `None` if any hop is unplaced.
    ///
    /// Index `k` in the result means the crossing happens *entering* hop `k` (hop `k` sits on a different
    /// device than hop `k-1`).
    pub fn crossings(&self, pi: &Place) -> Option<Vec<usize>> {
        let devices = self.devices(pi)?;
        Some(
            devices
                .windows(2)
                .enumerate()
                .filter(|(_, w)| w[0] != w[1])
                .map(|(i, _)| i + 1)
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The unbinding, stated as a test: an expert application classifies as a Route *regardless of the
    /// expert*, and a Route fences while a Map composes. Same class for every expert is what makes the
    /// partitioner blind to modality/tier/process.
    #[test]
    fn applying_any_expert_is_the_same_class_a_fence_that_is_not_a_map() {
        for id in [0usize, 1, 7, 4096] {
            let _ = Expert(id); // the id never reaches the class
            assert_eq!(expert_class(), Class::Route, "the class must not depend on which expert");
        }
        assert!(Class::Route.fences(), "a Route is a fence");
        assert!(!Class::Route.is_map(), "a Route is not a Map");
        assert!(Class::Map.is_map() && !Class::Map.fences(), "a Map still composes");
    }

    /// The load-bearing claim: **the crossings are a property of π, not of the plan.** One plan, three
    /// placements — co-located (no crossings), split at one seam, and ping-ponged — and the plan is never
    /// touched between them. Every fence is π's doing.
    #[test]
    fn crossings_come_from_pi_not_from_the_plan() {
        let plan: Plan = [Expert(0), Expert(1), Expert(2)].into_iter().collect();

        // π co-locates all three: every hop is a Route, but none crosses a device.
        let together: Place =
            [(Expert(0), Device(0)), (Expert(1), Device(0)), (Expert(2), Device(0))]
                .into_iter()
                .collect();
        assert_eq!(plan.devices(&together), Some(vec![Device(0), Device(0), Device(0)]));
        assert_eq!(plan.crossings(&together), Some(vec![]), "co-located Routes cross nothing");

        // π splits after the first hop: one crossing, entering hop 1.
        let split: Place =
            [(Expert(0), Device(0)), (Expert(1), Device(1)), (Expert(2), Device(1))]
                .into_iter()
                .collect();
        assert_eq!(plan.crossings(&split), Some(vec![1]));

        // π ping-pongs D0→D1→D0: two crossings — and the plan is byte-for-byte the same object.
        let pingpong: Place =
            [(Expert(0), Device(0)), (Expert(1), Device(1)), (Expert(2), Device(0))]
                .into_iter()
                .collect();
        assert_eq!(plan.crossings(&pingpong), Some(vec![1, 2]));
    }

    /// Fail closed: a plan hop π does not place has no device, so the whole plan has no path. Routing to
    /// an unplaced expert is refused, never guessed.
    #[test]
    fn an_unplaced_hop_leaves_the_plan_without_a_path() {
        let plan: Plan = [Expert(0), Expert(1)].into_iter().collect();
        let partial: Place = [(Expert(0), Device(0))].into_iter().collect();
        assert_eq!(plan.devices(&partial), None, "Expert(1) is unplaced");
        assert_eq!(plan.crossings(&partial), None);
    }

    /// A single-hop plan (or an empty one) crosses nothing — there is no pair of hops to cross between.
    #[test]
    fn a_lone_hop_has_no_crossing() {
        let one: Plan = [Expert(3)].into_iter().collect();
        let pi: Place = [(Expert(3), Device(2))].into_iter().collect();
        assert_eq!(one.crossings(&pi), Some(vec![]));

        let none: Plan = core::iter::empty().collect();
        assert_eq!(none.crossings(&pi), Some(vec![]));
    }
}
