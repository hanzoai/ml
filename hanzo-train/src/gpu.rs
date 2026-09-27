//! GPU utilization as the driver reports it (`ioreg`, IOAccelerator "Device Utilization %"),
//! sampled every half second on a background thread; macOS only. It counts every client of the
//! GPU, not only this process. And a floor on free memory: GPU buffers are unified memory, so a
//! runaway allocation starves the whole machine, not just this process.

#[cfg(target_os = "macos")]
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub struct Meter {
    samples: Arc<Mutex<Vec<f32>>>,
    stop: Arc<AtomicBool>,
}

#[cfg(not(target_os = "macos"))]
fn read() -> Option<f32> {
    None
}

#[cfg(target_os = "macos")]
fn read() -> Option<f32> {
    let out = Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let key = "\"Device Utilization %\"=";
    let at = text.find(key)? + key.len();
    let digits: String = text[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Free memory in percent: the kernel's memory-status level on macOS.
#[cfg(target_os = "macos")]
pub fn free() -> Option<u32> {
    let out = Command::new("sysctl")
        .args(["-n", "kern.memorystatus_level"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// Free memory in percent: `MemAvailable` over `MemTotal` from `/proc/meminfo` elsewhere.
#[cfg(not(target_os = "macos"))]
pub fn free() -> Option<u32> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let field = |key: &str| -> Option<u64> {
        let rest = text.lines().find_map(|l| l.strip_prefix(key))?;
        rest.trim().trim_end_matches("kB").trim().parse().ok()
    };
    Some((field("MemAvailable:")? * 100 / field("MemTotal:")?.max(1)) as u32)
}

/// This process's resident memory in bytes, as `ps` reports it.
pub fn resident() -> Option<u64> {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    let kb: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(kb * 1024)
}

/// Exit the process when free memory falls below `floor` percent, polled every quarter second.
pub fn guard(floor: u32) {
    std::thread::spawn(move || loop {
        if let Some(f) = free() {
            if f < floor {
                eprintln!("memory: {f}% free, under the {floor}% floor; exiting");
                std::process::exit(137);
            }
        }
        std::thread::sleep(Duration::from_millis(250));
    });
}

impl Meter {
    pub fn start() -> Meter {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (s, st) = (samples.clone(), stop.clone());
        std::thread::spawn(move || {
            while !st.load(Ordering::Relaxed) {
                if let Some(u) = read() {
                    s.lock().expect("meter lock").push(u);
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });
        Meter { samples, stop }
    }

    /// Mean utilization since the last call, in percent.
    pub fn utilization(&self) -> Option<f32> {
        let mut s = self.samples.lock().expect("meter lock");
        if s.is_empty() {
            return None;
        }
        let m = s.iter().sum::<f32>() / s.len() as f32;
        s.clear();
        Some(m)
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}
