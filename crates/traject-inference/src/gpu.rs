//! CUDA / GPU capability probe for Traject inference backends.

/// Snapshot of host GPU readiness (Phase 1: env / nvidia-smi based).
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    pub cuda_visible: bool,
    pub device_count: u32,
    pub nvidia_smi_ok: bool,
}

impl GpuCapabilities {
    pub fn probe() -> Self {
        let cuda_visible = std::env::var_os("CUDA_VISIBLE_DEVICES").is_some()
            || std::path::Path::new("/usr/local/cuda").exists()
            || std::path::Path::new("/dev/nvidia0").exists();
        let nvidia_smi_ok = std::process::Command::new("nvidia-smi")
            .arg("-L")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        let device_count = if nvidia_smi_ok {
            std::process::Command::new("nvidia-smi")
                .arg("-L")
                .output()
                .ok()
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .filter(|l| l.starts_with("GPU "))
                        .count() as u32
                })
                .unwrap_or(0)
        } else {
            0
        };
        Self {
            cuda_visible,
            device_count,
            nvidia_smi_ok,
        }
    }

    pub fn ready_for_local_engine(&self) -> bool {
        self.nvidia_smi_ok && self.device_count > 0
    }
}
