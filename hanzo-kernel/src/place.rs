//! Placement: **π is a value, not a schedule**.
//!
//! A harvested expert is served frozen, and that one fact collapses distribution into a lookup
//! (`mode.tex` §5.1). A frozen expert has no gradient and no optimizer state, so nothing about it ever
//! travels between devices; its residency is paid once at extraction and never reloaded; it is a pure
//! function of its input. What is left of "where does this expert run?" is a static map
//!
//! ```text
//!     π : expert -> device
//! ```
//!
//! decided once and shared read-only. A router that emits an expert id has, **by composition with π**,
//! emitted an address — there is no placement *policy* at runtime to get wrong, because there is no
//! decision left to make.
//!
//! That is what lets [`Class::Route`](crate::fuse::Class::Route) fence on placement without the fuser
//! ever learning what a device is. The fuser cuts a trace at its fences reading the class and nothing
//! else; π addresses the cuts afterwards. The two never change each other's rules: another expert family
//! adds rows to π and nothing to the fuser, another fusable op adds a variant to `UnOp` and nothing to π.
//!
//! [`Place::pin`] is the one solver that builds a π — largest expert first, onto the device with the
//! most room that still fits it, the rule the zen5 lab scheduler uses. It is a pure function of the pool
//! and the fleet: a fleet whose nodes disagreed about where an expert lives could not route to it at all.

use core::cmp::Reverse;
use core::fmt;
use std::collections::HashMap;

/// A frozen expert, harvested from a source model and served as-is.
///
/// An id, not a name: the node algebra is `Copy`, and how the pool names its experts is the caller's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Expert(pub usize);

/// A device in the fleet — the thing an expert is resident *on*.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Device(pub usize);

/// π : expert → device — which device owns each expert.
///
/// Frozen weights make this a value: build it once, share it read-only, and routing is an address
/// lookup. Many experts to one device is the normal case — a device hosts a shard of the pool.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Place {
    at: HashMap<Expert, Device>,
}

/// Declare a π directly. The lab's hand-written pins are exactly this, and [`Place::pin`] packs the rest
/// of the pool around them.
impl FromIterator<(Expert, Device)> for Place {
    fn from_iter<I: IntoIterator<Item = (Expert, Device)>>(pins: I) -> Place {
        Place { at: pins.into_iter().collect() }
    }
}

impl Place {
    /// π(e) — the device that owns `e`, or `None` if `e` is not in the pool.
    pub fn device(&self, e: Expert) -> Option<Device> {
        self.at.get(&e).copied()
    }

    /// Extend π by pinning every expert it does not already place: **largest first, onto the device with
    /// the most room that still fits it**, so the pool spreads across the fleet instead of filling one
    /// device. Pins already in `self` are hard constraints — their memory is charged to their device
    /// before anything else is packed, so a declared pin is honored or the whole plan is refused. (A
    /// declared pin naming a device outside `devices` is left as declared.)
    ///
    /// `experts` and `devices` carry (id, memory) in whatever unit the caller keeps; the solver only
    /// compares them. π is a function of the pool and the fleet *as sets*: experts are ordered by
    /// decreasing memory and then by id, devices are considered in the order given, and a tie takes the
    /// first. Nothing else can move an expert.
    pub fn pin(
        mut self,
        experts: &[(Expert, u64)],
        devices: &[(Device, u64)],
    ) -> Result<Place, Error> {
        let size: HashMap<Expert, u64> = experts.iter().copied().collect();

        // Declared pins are constraints, not candidates: charge each device its declared load up front.
        let mut free: Vec<(Device, u64)> = Vec::with_capacity(devices.len());
        for &(device, have) in devices {
            let need: u64 = self
                .at
                .iter()
                .filter(|(_, &owner)| owner == device)
                .filter_map(|(e, _)| size.get(e))
                .sum();
            if need > have {
                return Err(Error::Over { device, need, have });
            }
            free.push((device, have - need));
        }

        // Largest first — place the experts that are hardest to fit while the fleet is still empty.
        // Iterate the deduped `size` map, not the raw slice: the pool is a set, so a repeated id is
        // charged and placed exactly once (a raw-slice iteration would place each occurrence, phantom-
        // reserving room on the abandoned device). The (need desc, id) sort makes the map order irrelevant.
        let mut order: Vec<(Expert, u64)> = size
            .iter()
            .map(|(&e, &need)| (e, need))
            .filter(|(e, _)| !self.at.contains_key(e))
            .collect();
        order.sort_by_key(|&(e, need)| (Reverse(need), e));

        for (e, need) in order {
            // `min_by_key(Reverse)` is the most room that still fits, first-listed device winning a tie.
            let room = free.iter_mut().filter(|(_, left)| *left >= need).min_by_key(|(_, left)| Reverse(*left));
            let Some(slot) = room else {
                return Err(Error::Unplaceable { expert: e, need });
            };
            slot.1 -= need;
            self.at.insert(e, slot.0);
        }
        Ok(self)
    }
}

/// Why no π exists for a pool and a fleet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A device is declared more than it holds.
    Over { device: Device, need: u64, have: u64 },
    /// An expert fits on no device with room left.
    Unplaceable { expert: Expert, need: u64 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::Over { device, need, have } => write!(
                f,
                "device {} is declared {need} but holds {have} — split the model into smaller slabs, or declare the pin on a larger device",
                device.0
            ),
            Error::Unplaceable { expert, need } => write!(
                f,
                "expert {} needs {need}; no device has that much room left",
                expert.0
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fleet shaped like the zen5 lab: two big boxes and a small one.
    const FLEET: [(Device, u64); 3] = [(Device(0), 128), (Device(1), 128), (Device(2), 64)];
    /// One expert too big to share, two that pair up, one that fills a gap.
    const POOL: [(Expert, u64); 4] =
        [(Expert(0), 100), (Expert(1), 40), (Expert(2), 40), (Expert(3), 10)];

    #[test]
    fn pin_places_the_largest_expert_first_where_there_is_most_room() {
        let pi = Place::default().pin(&POOL, &FLEET).unwrap();
        // 100 goes first and only D0/D1 can hold it; D0 wins the tie by being listed first.
        assert_eq!(pi.device(Expert(0)), Some(Device(0)));
        // Both 40s then prefer D1 (128, then 88) over D2 (64) — the most room that fits.
        assert_eq!(pi.device(Expert(1)), Some(Device(1)));
        assert_eq!(pi.device(Expert(2)), Some(Device(1)));
        // 10 fits everywhere, so it lands on whatever is emptiest by then: D2 (64) beats D1 (48), D0 (28).
        assert_eq!(pi.device(Expert(3)), Some(Device(2)));
    }

    #[test]
    fn a_declared_pin_is_honored_and_charged_to_its_device() {
        // The lab declares E1 (40) on D2 (64). Charging that 40 up front is what stops E4 (30) from
        // being packed on top of a device that cannot hold both.
        let declared: Place = [(Expert(1), Device(2))].into_iter().collect();
        let pool = [(Expert(1), 40), (Expert(4), 30)];
        assert_eq!(
            declared.clone().pin(&pool, &[(Device(2), 64)]),
            Err(Error::Unplaceable { expert: Expert(4), need: 30 }),
            "the declared pin's 40 was not charged to D2"
        );

        // Given somewhere else to put it, the declaration survives the packing untouched.
        let pi = declared.pin(&pool, &[(Device(2), 64), (Device(3), 64)]).unwrap();
        assert_eq!(pi.device(Expert(1)), Some(Device(2)), "declared pin moved");
        assert_eq!(pi.device(Expert(4)), Some(Device(3)));
    }

    #[test]
    fn a_pool_that_does_not_fit_is_refused_not_rounded() {
        assert_eq!(
            Place::default().pin(&[(Expert(0), 200)], &FLEET),
            Err(Error::Unplaceable { expert: Expert(0), need: 200 })
        );
        let declared: Place = [(Expert(0), Device(2))].into_iter().collect();
        assert_eq!(
            declared.pin(&[(Expert(0), 100)], &[(Device(2), 64)]),
            Err(Error::Over { device: Device(2), need: 100, have: 64 })
        );
    }

    #[test]
    fn a_repeated_expert_id_is_charged_and_placed_once() {
        // The pool is a set: listing an expert twice must not double-charge a device. E0(100) and E1(100)
        // fit as a set on two 120-capacity devices; a per-occurrence charge would phantom-reserve 100 on a
        // second device and spuriously refuse E1.
        let pool = [(Expert(0), 100), (Expert(0), 100), (Expert(1), 100)];
        let fleet = [(Device(0), 120), (Device(1), 120)];
        let pi = Place::default().pin(&pool, &fleet).unwrap();
        assert_eq!(pi.device(Expert(0)), Some(Device(0)));
        assert_eq!(pi.device(Expert(1)), Some(Device(1)));
        // One entry per distinct expert, and the same pool listed in another order is identical.
        assert_eq!(pi.at.len(), 2);
        let reordered = [(Expert(1), 100), (Expert(0), 100), (Expert(0), 100)];
        assert_eq!(Place::default().pin(&reordered, &fleet).unwrap(), pi);
    }

    #[test]
    fn pi_resolves_an_expert_to_a_device_deterministically() {
        // π must be a function of the pool and the fleet ALONE: every node works it out independently,
        // and two nodes that disagreed about where an expert lives could not route to it at all.
        let pi = Place::default().pin(&POOL, &FLEET).unwrap();
        for _ in 0..64 {
            assert_eq!(
                Place::default().pin(&POOL, &FLEET).unwrap(),
                pi,
                "π is not a function of its inputs"
            );
        }
        // The pool is a set, not a sequence — listing it in another order is the same pool.
        let shuffled = [(Expert(3), 10), (Expert(2), 40), (Expert(0), 100), (Expert(1), 40)];
        assert_eq!(
            Place::default().pin(&shuffled, &FLEET).unwrap(),
            pi,
            "π depends on the order the pool was listed in"
        );
        // And π is a function: one expert, one device, every call.
        assert_eq!(pi.device(Expert(0)), pi.device(Expert(0)));
        assert_eq!(pi.device(Expert(9)), None, "an expert outside the pool has no address");
    }
}
