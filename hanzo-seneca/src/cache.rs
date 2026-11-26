//! Bounded multi-stage, multi-tenant LRU cache for training data pipelines.
//!
//! Enforces three hard safety invariants:
//! 1. **Bounded memory**: `total_used_bytes() <= total_budget_bytes()` holds at
//!    every instant, including during dynamic partition rebalancing (shrink-first,
//!    expand-second).
//! 2. **Strict tenant/job isolation**: `Private` dataset entries are keyed by
//!    `(tenant_id, job_id, dataset_digest, sample_id, stage, transform_profile_hash)`
//!    and verify caller `(tenant_id, job_id)` on every lookup (`0` cross-tenant leaks).
//! 3. **Safe content-addressed sharing**: ONLY explicitly `SharedPublic` datasets
//!    are shareable across concurrent jobs, keyed by `(dataset_digest, sample_id,
//!    stage, transform_profile_hash)`.

use std::collections::{BTreeMap, HashMap};

/// The three cacheable representations in a training data-ingestion pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CacheStage {
    /// Compressed bytes as fetched from S3 / VFS object storage.
    Encoded = 0,
    /// Decompressed / decoded in-memory tensor or token buffer.
    Decoded = 1,
    /// Post-augmentation / post-tokenization model-ready tensor.
    /// MUST NOT be cached across epochs when a job uses stochastic augmentations.
    Transformed = 2,
}

impl CacheStage {
    pub const ALL: [CacheStage; 3] = [
        CacheStage::Encoded,
        CacheStage::Decoded,
        CacheStage::Transformed,
    ];

    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn as_str(self) -> &'static str {
        match self {
            CacheStage::Encoded => "encoded",
            CacheStage::Decoded => "decoded",
            CacheStage::Transformed => "transformed",
        }
    }
}

/// Isolation boundary for a dataset in the shared host cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DatasetVisibility {
    /// Strictly private to a single `(tenant_id, job_id)` pair. Never visible to
    /// any other tenant or job, even if payload bytes happen to coincide.
    Private { tenant_id: u64, job_id: u64 },
    /// Explicitly declared public dataset shareable across concurrent jobs via
    /// content-addressed `dataset_digest`.
    SharedPublic { dataset_digest: u128 },
}

/// Cache partition ownership scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScopeId {
    Private { tenant_id: u64, job_id: u64 },
    SharedPublic { dataset_digest: u128 },
}

impl ScopeId {
    #[inline]
    pub const fn from_visibility(vis: DatasetVisibility) -> Self {
        match vis {
            DatasetVisibility::Private { tenant_id, job_id } => {
                ScopeId::Private { tenant_id, job_id }
            }
            DatasetVisibility::SharedPublic { dataset_digest } => {
                ScopeId::SharedPublic { dataset_digest }
            }
        }
    }

    #[inline]
    pub const fn is_shared_public(self) -> bool {
        matches!(self, ScopeId::SharedPublic { .. })
    }
}

/// Content-addressed, tenant-scoped cache key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey {
    pub visibility: DatasetVisibility,
    pub dataset_digest: u128,
    pub sample_id: u64,
    pub stage: CacheStage,
    /// Stage configuration hash:
    /// - `0` for `Encoded` entries (raw compressed storage bytes depend only on
    ///   `(dataset_digest, sample_id)`),
    /// - decoder configuration hash (e.g. modality + `decode_compute_passes`)
    ///   for `Decoded` entries,
    /// - full decode + transform profile hash for deterministic `Transformed` entries.
    pub transform_profile_hash: u64,
}

impl CacheKey {
    /// Constructs a cache key for a given visibility and stage.
    ///
    /// Preserves `stage_profile_hash` for both [`CacheStage::Decoded`] and
    /// [`CacheStage::Transformed`] so jobs with different decoder or transform
    /// configurations never share incompatible cached tensors.
    #[inline]
    pub fn new(
        visibility: DatasetVisibility,
        dataset_digest: u128,
        sample_id: u64,
        stage: CacheStage,
        stage_profile_hash: u64,
    ) -> Self {
        let profile = match stage {
            CacheStage::Encoded => 0,
            CacheStage::Decoded | CacheStage::Transformed => stage_profile_hash,
        };
        Self {
            visibility,
            dataset_digest,
            sample_id,
            stage,
            transform_profile_hash: profile,
        }
    }

    /// Returns `true` iff `(caller_tenant, caller_job)` is authorized to read this key.
    #[inline]
    pub fn is_accessible_by(&self, caller_tenant: u64, caller_job: u64) -> bool {
        match self.visibility {
            DatasetVisibility::Private { tenant_id, job_id } => {
                tenant_id == caller_tenant && job_id == caller_job
            }
            DatasetVisibility::SharedPublic { dataset_digest } => {
                dataset_digest == self.dataset_digest
            }
        }
    }

    #[inline]
    pub const fn scope(&self) -> ScopeId {
        ScopeId::from_visibility(self.visibility)
    }
}

/// Stored cache payload with checksum and epoch provenance.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub payload: Vec<u8>,
    pub crc32: u32,
    pub epoch_written: u32,
}

impl CacheEntry {
    #[inline]
    pub fn new(payload: Vec<u8>, crc32: u32, epoch_written: u32) -> Self {
        Self {
            payload,
            crc32,
            epoch_written,
        }
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.payload.len()
    }
}

/// Per-stage aggregate counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct StageStats {
    pub hits: u64,
    pub misses: u64,
    pub admissions: u64,
    pub evictions: u64,
    pub bypassed_inserts: u64,
    pub bytes_served: u64,
}

impl StageStats {
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Exact byte-bounded LRU partition using monotonic timestamps (`O(log N)` eviction, zero `unsafe`).
#[derive(Debug, Default)]
pub struct LruPartition {
    budget_bytes: usize,
    used_bytes: usize,
    clock: u64,
    entries: HashMap<CacheKey, (u64, CacheEntry)>,
    order: BTreeMap<u64, CacheKey>,
    evictions: u64,
}

impl LruPartition {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            ..Self::default()
        }
    }

    #[inline]
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    #[inline]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline]
    pub fn contains_key(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    fn next_tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    /// Updates partition budget, immediately evicting LRU entries if shrinking.
    pub fn set_budget(&mut self, new_budget: usize) -> u64 {
        self.budget_bytes = new_budget;
        let mut evicted = 0u64;
        while self.used_bytes > self.budget_bytes {
            if self.evict_oldest() {
                evicted += 1;
            } else {
                break;
            }
        }
        debug_assert!(self.used_bytes <= self.budget_bytes);
        evicted
    }

    fn evict_oldest(&mut self) -> bool {
        let oldest_tick = match self.order.keys().next().copied() {
            Some(t) => t,
            None => return false,
        };
        if let Some(key) = self.order.remove(&oldest_tick) {
            if let Some((_, entry)) = self.entries.remove(&key) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.byte_len());
                self.evictions += 1;
                return true;
            }
        }
        false
    }

    /// Looks up a key, updating LRU recency on hit.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        if !self.entries.contains_key(key) {
            return None;
        }
        let new_tick = self.next_tick();
        if let Some((old_tick, _)) = self.entries.get_mut(key) {
            let prev = *old_tick;
            *old_tick = new_tick;
            self.order.remove(&prev);
            self.order.insert(new_tick, *key);
        }
        self.entries.get(key).map(|(_, entry)| entry)
    }

    /// Inserts an entry if it fits within `budget_bytes`, evicting LRU entries as needed.
    /// Returns `(admitted, evictions_performed)`.
    pub fn insert(&mut self, key: CacheKey, entry: CacheEntry) -> (bool, u64) {
        let item_bytes = entry.byte_len();
        if item_bytes == 0 || item_bytes > self.budget_bytes {
            return (false, 0);
        }

        // Remove existing entry for the same key if replacing.
        if let Some((old_tick, old_entry)) = self.entries.remove(&key) {
            self.order.remove(&old_tick);
            self.used_bytes = self.used_bytes.saturating_sub(old_entry.byte_len());
        }

        let mut evicted = 0u64;
        while self.used_bytes + item_bytes > self.budget_bytes {
            if self.evict_oldest() {
                evicted += 1;
            } else {
                break;
            }
        }

        if self.used_bytes + item_bytes > self.budget_bytes {
            return (false, evicted);
        }

        let tick = self.next_tick();
        self.used_bytes += item_bytes;
        self.entries.insert(key, (tick, entry));
        self.order.insert(tick, key);
        debug_assert!(self.used_bytes <= self.budget_bytes);
        (true, evicted)
    }
}

/// Global multi-stage, multi-tenant cache enforcing `total_used_bytes <= total_budget_bytes`.
#[derive(Debug)]
pub struct MultiStageCache {
    total_budget_bytes: usize,
    partitions: HashMap<(ScopeId, CacheStage), LruPartition>,
    stage_stats: [StageStats; 3],
    peak_used_bytes: usize,
    utilization_sum_bytes: u128,
    utilization_samples: u64,
    unauthorized_probes_blocked: u64,
    cross_tenant_leaks: u64,
}

impl MultiStageCache {
    /// Creates a new bounded multi-stage cache with the given global byte budget.
    pub fn new(total_budget_bytes: usize) -> Self {
        Self {
            total_budget_bytes,
            partitions: HashMap::new(),
            stage_stats: [StageStats::default(); 3],
            peak_used_bytes: 0,
            utilization_sum_bytes: 0,
            utilization_samples: 0,
            unauthorized_probes_blocked: 0,
            cross_tenant_leaks: 0,
        }
    }

    #[inline]
    pub fn total_budget_bytes(&self) -> usize {
        self.total_budget_bytes
    }

    /// Returns the exact sum of bytes currently stored across all partitions.
    pub fn total_used_bytes(&self) -> usize {
        self.partitions.values().map(LruPartition::used_bytes).sum()
    }

    /// Returns the exact sum of allocated partition budgets.
    pub fn total_allocated_budget_bytes(&self) -> usize {
        self.partitions
            .values()
            .map(LruPartition::budget_bytes)
            .sum()
    }

    #[inline]
    pub fn peak_used_bytes(&self) -> usize {
        self.peak_used_bytes
    }

    #[inline]
    pub fn avg_utilization_pct(&self) -> f64 {
        if self.total_budget_bytes == 0 || self.utilization_samples == 0 {
            return 0.0;
        }
        let avg_bytes = self.utilization_sum_bytes as f64 / self.utilization_samples as f64;
        (100.0 * avg_bytes / self.total_budget_bytes as f64).min(100.0)
    }

    #[inline]
    pub fn peak_utilization_pct(&self) -> f64 {
        if self.total_budget_bytes == 0 {
            return 0.0;
        }
        (100.0 * self.peak_used_bytes as f64 / self.total_budget_bytes as f64).min(100.0)
    }

    #[inline]
    pub fn stage_stats(&self, stage: CacheStage) -> StageStats {
        self.stage_stats[stage.index()]
    }

    #[inline]
    pub fn unauthorized_probes_blocked(&self) -> u64 {
        self.unauthorized_probes_blocked
    }

    #[inline]
    pub fn cross_tenant_leaks(&self) -> u64 {
        self.cross_tenant_leaks
    }

    /// Returns the current budget of a specific `(scope, stage)` partition.
    pub fn partition_budget(&self, scope: ScopeId, stage: CacheStage) -> usize {
        self.partitions
            .get(&(scope, stage))
            .map_or(0, LruPartition::budget_bytes)
    }

    /// Returns the current used bytes of a specific `(scope, stage)` partition.
    pub fn partition_used(&self, scope: ScopeId, stage: CacheStage) -> usize {
        self.partitions
            .get(&(scope, stage))
            .map_or(0, LruPartition::used_bytes)
    }

    /// Applies a new global budget map using **shrink-first, expand-second** ordering
    /// so `total_used_bytes() <= total_budget_bytes` holds at every intermediate step.
    pub fn apply_budgets(&mut self, requested: &HashMap<(ScopeId, CacheStage), usize>) {
        // Clamp requested budgets proportionally if the caller ever exceeds global cap.
        let req_sum: usize = requested.values().copied().sum();
        let normalized: HashMap<(ScopeId, CacheStage), usize> =
            if req_sum > self.total_budget_bytes && req_sum > 0 {
                let scale = self.total_budget_bytes as f64 / req_sum as f64;
                let mut out = HashMap::with_capacity(requested.len());
                let mut running = 0usize;
                let mut keys: Vec<_> = requested.keys().copied().collect();
                keys.sort();
                for k in keys {
                    let v = requested[&k];
                    let scaled = ((v as f64) * scale).floor() as usize;
                    let capped = scaled.min(self.total_budget_bytes.saturating_sub(running));
                    running += capped;
                    out.insert(k, capped);
                }
                out
            } else {
                requested.clone()
            };

        // Pass 1: Shrink existing partitions whose target budget decreased (or disappeared).
        let existing_keys: Vec<(ScopeId, CacheStage)> = self.partitions.keys().copied().collect();
        for key in existing_keys {
            let target = normalized.get(&key).copied().unwrap_or(0);
            if let Some(part) = self.partitions.get_mut(&key) {
                if target < part.budget_bytes() {
                    let evicted = part.set_budget(target);
                    self.stage_stats[key.1.index()].evictions += evicted;
                }
            }
        }

        // Pass 2: Expand or create partitions whose target budget increased.
        for (&key, &target) in &normalized {
            let part = self
                .partitions
                .entry(key)
                .or_insert_with(|| LruPartition::new(0));
            if target > part.budget_bytes() {
                part.set_budget(target);
            }
        }

        let used = self.total_used_bytes();
        assert!(
            used <= self.total_budget_bytes,
            "cache memory invariant violated: used={used} > budget={}",
            self.total_budget_bytes
        );
        self.record_utilization_sample(used);
    }

    /// Non-mutating check whether a key is currently cached and accessible by caller.
    /// Used by Seneca's opportunistic cache-aware sampler (ODS).
    pub fn contains_accessible(&self, key: &CacheKey, caller_tenant: u64, caller_job: u64) -> bool {
        if !key.is_accessible_by(caller_tenant, caller_job) {
            return false;
        }
        let part_key = (key.scope(), key.stage);
        self.partitions
            .get(&part_key)
            .is_some_and(|part| part.contains_key(key))
    }

    /// Looks up a key on behalf of `(caller_tenant, caller_job)` with strict
    /// tenant-isolation enforcement.
    pub fn get(
        &mut self,
        key: &CacheKey,
        caller_tenant: u64,
        caller_job: u64,
    ) -> Option<&CacheEntry> {
        if !key.is_accessible_by(caller_tenant, caller_job) {
            self.unauthorized_probes_blocked += 1;
            return None;
        }
        let stage_idx = key.stage.index();
        let part_key = (key.scope(), key.stage);
        let found = self
            .partitions
            .get_mut(&part_key)
            .and_then(|part| part.get(key));
        match found {
            Some(entry) => {
                self.stage_stats[stage_idx].hits += 1;
                self.stage_stats[stage_idx].bytes_served += entry.byte_len() as u64;
                Some(entry)
            }
            None => {
                self.stage_stats[stage_idx].misses += 1;
                None
            }
        }
    }

    /// Explicitly probes a victim key from a different `(attacker_tenant, attacker_job)`
    /// to audit tenant isolation without polluting normal hit/miss stats.
    pub fn probe_cross_tenant(
        &mut self,
        victim_key: &CacheKey,
        attacker_tenant: u64,
        attacker_job: u64,
    ) -> Option<&CacheEntry> {
        if !victim_key.is_accessible_by(attacker_tenant, attacker_job) {
            self.unauthorized_probes_blocked += 1;
            return None;
        }
        // Only reachable for SharedPublic keys or same-owner keys.
        if let DatasetVisibility::Private { tenant_id, job_id } = victim_key.visibility {
            if tenant_id != attacker_tenant || job_id != attacker_job {
                self.cross_tenant_leaks += 1;
                return None;
            }
        }
        let part_key = (victim_key.scope(), victim_key.stage);
        self.partitions
            .get_mut(&part_key)
            .and_then(|part| part.get(victim_key))
    }

    /// Inserts an entry into the partition for `(key.scope(), key.stage)`.
    /// Respects `admit_enabled` (streaming bypass) and enforces global memory bounds.
    pub fn insert(
        &mut self,
        key: CacheKey,
        entry: CacheEntry,
        caller_tenant: u64,
        caller_job: u64,
        admit_enabled: bool,
    ) -> bool {
        let stage_idx = key.stage.index();
        if !admit_enabled {
            self.stage_stats[stage_idx].bypassed_inserts += 1;
            return false;
        }
        if !key.is_accessible_by(caller_tenant, caller_job) {
            self.unauthorized_probes_blocked += 1;
            return false;
        }

        let part_key = (key.scope(), key.stage);
        let Some(part) = self.partitions.get_mut(&part_key) else {
            self.stage_stats[stage_idx].bypassed_inserts += 1;
            return false;
        };
        if part.budget_bytes() == 0 {
            self.stage_stats[stage_idx].bypassed_inserts += 1;
            return false;
        }

        let (admitted, evicted) = part.insert(key, entry);
        self.stage_stats[stage_idx].evictions += evicted;
        if admitted {
            self.stage_stats[stage_idx].admissions += 1;
        } else {
            self.stage_stats[stage_idx].bypassed_inserts += 1;
        }

        let used = self.total_used_bytes();
        debug_assert!(used <= self.total_budget_bytes);
        self.record_utilization_sample(used);
        admitted
    }

    fn record_utilization_sample(&mut self, used: usize) {
        if used > self.peak_used_bytes {
            self.peak_used_bytes = used;
        }
        self.utilization_sum_bytes += used as u128;
        self.utilization_samples += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::crc32;

    #[test]
    fn bounded_memory_invariant_holds_across_inserts_and_rebalances() {
        let budget = 1024;
        let mut cache = MultiStageCache::new(budget);
        let scope_a = ScopeId::Private {
            tenant_id: 1,
            job_id: 10,
        };
        let scope_b = ScopeId::Private {
            tenant_id: 2,
            job_id: 20,
        };

        let mut b = HashMap::new();
        b.insert((scope_a, CacheStage::Encoded), 768);
        b.insert((scope_b, CacheStage::Decoded), 256);
        cache.apply_budgets(&b);

        for sample_id in 0..20u64 {
            let payload = vec![sample_id as u8; 128];
            let c = crc32(&payload);
            let k = CacheKey::new(
                DatasetVisibility::Private {
                    tenant_id: 1,
                    job_id: 10,
                },
                0xABCD,
                sample_id,
                CacheStage::Encoded,
                0,
            );
            assert!(cache.insert(k, CacheEntry::new(payload, c, 0), 1, 10, true));
            assert!(cache.total_used_bytes() <= budget);
        }
        assert_eq!(cache.partition_used(scope_a, CacheStage::Encoded), 768);

        // Rebalance heavily from scope_a to scope_b: shrink-first must keep total <= budget.
        let mut b2 = HashMap::new();
        b2.insert((scope_a, CacheStage::Encoded), 256);
        b2.insert((scope_b, CacheStage::Decoded), 768);
        cache.apply_budgets(&b2);
        assert!(cache.total_used_bytes() <= budget);
        assert!(cache.partition_used(scope_a, CacheStage::Encoded) <= 256);
    }

    #[test]
    fn strict_tenant_isolation_blocks_cross_tenant_reads() {
        let mut cache = MultiStageCache::new(4096);
        let vis_a = DatasetVisibility::Private {
            tenant_id: 100,
            job_id: 1,
        };
        let mut b = HashMap::new();
        b.insert((ScopeId::from_visibility(vis_a), CacheStage::Decoded), 2048);
        cache.apply_budgets(&b);

        let payload = b"confidential-tenant-100-training-sample".to_vec();
        let k = CacheKey::new(vis_a, 0x9999, 7, CacheStage::Decoded, 0);
        assert!(cache.insert(
            k,
            CacheEntry::new(payload.clone(), crc32(&payload), 0),
            100,
            1,
            true
        ));

        // Legitimate owner can read.
        let hit = cache.get(&k, 100, 1);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().payload, payload);

        // Different tenant or different job MUST be blocked.
        assert!(cache.get(&k, 200, 2).is_none());
        assert!(cache.get(&k, 100, 2).is_none());
        assert!(cache.probe_cross_tenant(&k, 200, 2).is_none());
        assert_eq!(cache.cross_tenant_leaks(), 0);
        assert!(cache.unauthorized_probes_blocked() >= 3);
    }

    #[test]
    fn shared_public_allows_content_addressed_sharing_and_separates_transforms() {
        let mut cache = MultiStageCache::new(8192);
        let digest = 0xDEAD_BEEF_1234_5678u128;
        let vis = DatasetVisibility::SharedPublic {
            dataset_digest: digest,
        };
        let scope = ScopeId::from_visibility(vis);
        let mut b = HashMap::new();
        b.insert((scope, CacheStage::Decoded), 4096);
        b.insert((scope, CacheStage::Transformed), 4096);
        cache.apply_budgets(&b);

        // Tenant 1 writes decoded public sample -> Tenant 2 reads it!
        let k_dec = CacheKey::new(vis, digest, 42, CacheStage::Decoded, 0);
        let dec_bytes = b"shared-decoded-tensor".to_vec();
        assert!(cache.insert(
            k_dec,
            CacheEntry::new(dec_bytes.clone(), crc32(&dec_bytes), 0),
            1,
            1,
            true
        ));
        assert_eq!(cache.get(&k_dec, 2, 2).unwrap().payload, dec_bytes);

        // Transformed entries with different transform_profile_hash NEVER collide!
        let k_t1 = CacheKey::new(vis, digest, 42, CacheStage::Transformed, 0x1111);
        let k_t2 = CacheKey::new(vis, digest, 42, CacheStage::Transformed, 0x2222);
        assert!(cache.insert(
            k_t1,
            CacheEntry::new(b"profile-A".to_vec(), crc32(b"profile-A"), 0),
            1,
            1,
            true
        ));
        assert!(cache.get(&k_t2, 2, 2).is_none());
        assert_eq!(cache.get(&k_t1, 2, 2).unwrap().payload, b"profile-A");
    }

    #[test]
    fn decoded_entries_with_different_decoder_profiles_never_collide() {
        let mut cache = MultiStageCache::new(8192);
        let digest = 0xCAFE_BABE_1349u128;
        let vis = DatasetVisibility::SharedPublic {
            dataset_digest: digest,
        };
        let scope = ScopeId::from_visibility(vis);
        let mut b = HashMap::new();
        b.insert((scope, CacheStage::Encoded), 2048);
        b.insert((scope, CacheStage::Decoded), 4096);
        cache.apply_budgets(&b);

        // Encoded keys normalize profile hash to 0 regardless of caller tag.
        let k_e1 = CacheKey::new(vis, digest, 7, CacheStage::Encoded, 0x1111);
        let k_e2 = CacheKey::new(vis, digest, 7, CacheStage::Encoded, 0x2222);
        assert_eq!(k_e1, k_e2);
        assert_eq!(k_e1.transform_profile_hash, 0);

        // Decoded keys preserve decoder configuration hash and never collide.
        let k_d1 = CacheKey::new(vis, digest, 7, CacheStage::Decoded, 0xAAAA);
        let k_d2 = CacheKey::new(vis, digest, 7, CacheStage::Decoded, 0xBBBB);
        assert_ne!(k_d1, k_d2);
        assert_eq!(k_d1.transform_profile_hash, 0xAAAA);
        assert_eq!(k_d2.transform_profile_hash, 0xBBBB);

        assert!(cache.insert(
            k_d1,
            CacheEntry::new(b"decode-1pass".to_vec(), crc32(b"decode-1pass"), 0),
            1,
            1,
            true
        ));
        assert!(cache.get(&k_d2, 2, 2).is_none());
        assert!(cache.insert(
            k_d2,
            CacheEntry::new(b"decode-2pass".to_vec(), crc32(b"decode-2pass"), 0),
            2,
            2,
            true
        ));
        assert_eq!(cache.get(&k_d1, 2, 2).unwrap().payload, b"decode-1pass");
        assert_eq!(cache.get(&k_d2, 1, 1).unwrap().payload, b"decode-2pass");
    }
}
