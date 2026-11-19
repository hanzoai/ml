use std::time::{Instant, Duration};

/// AdaCheck Controller: Decides when to checkpoint based on training dynamics
pub struct AdaCheckController {
    last_checkpoint_time: Instant,
    last_loss: f32,
    threshold: f32,
    min_interval: Duration,
}

impl AdaCheckController {
    pub fn new(threshold: f32, min_interval_secs: u64) -> Self {
        Self {
            last_checkpoint_time: Instant::now(),
            last_loss: f32::MAX,
            threshold,
            min_interval: Duration::from_secs(min_interval_secs),
        }
    }

    /// AdaCheck logic: Checkpoint if loss variance is high or time interval exceeded
    pub fn should_checkpoint(&mut self, current_loss: f32) -> bool {
        let now = Instant::now();
        let time_passed = now.duration_since(self.last_checkpoint_time);

        if time_passed < self.min_interval {
            return false;
        }

        let loss_diff = (self.last_loss - current_loss).abs();
        let significant_change = loss_diff > self.threshold;

        if significant_change || time_passed > self.min_interval * 4 {
            self.last_checkpoint_time = now;
            self.last_loss = current_loss;
            true
        } else {
            false
        }
    }
}
