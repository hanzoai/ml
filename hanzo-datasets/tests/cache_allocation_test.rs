use hanzo_datasets::cache_manager::{DynamicCacheManager, CacheStage};

#[test]
fn test_dynamic_partitioning_logic() {
    let manager = DynamicCacheManager::new(1.0); // 1GB
    
    // Initial state
    let (e, d, t) = manager.get_target_ratio();
    assert_eq!(e, 0.33);

    // Simulate GPU Starvation
    for _ in 0..10 {
        manager.report_gpu_starvation();
    }

    // Check if ratios adjusted to prioritize Transformed data (Seneca logic)
    let (_, _, t_new) = manager.get_target_ratio();
    assert!(t_new > t);
}

#[test]
fn test_memory_tracking() {
    let manager = DynamicCacheManager::new(1.0);
    manager.update_usage(CacheStage::Encoded, 1024);
    assert!(manager.available_capacity() < 1024 * 1024 * 1024);
    
    manager.update_usage(CacheStage::Encoded, -1024);
    // Should return to full capacity (approx)
    assert!(manager.available_capacity() > 1000000);
}
