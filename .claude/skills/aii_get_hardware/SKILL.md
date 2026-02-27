---
name: aii_get_hardware
description: Detect available hardware (CPUs, RAM, GPU/VRAM, disk, OS). Run before writing performance-sensitive code.
---

**Step 1** — Run `bash scripts/get_hardware.sh` (relative to this skill's directory).

**Step 2** — Set Python constants from results:
```python
import os, math, torch, psutil
from pathlib import Path

def _cgroup_cpus() -> int | None:
    """Read actual CPU allocation from cgroup limits (containers/pods)."""
    try:  # cgroups v2
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError): pass
    try:  # cgroups v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            return math.ceil(q / p)
    except (FileNotFoundError, ValueError): pass
    return None

def _container_ram_gb() -> float | None:
    """Read RAM limit from cgroup (containers/pods)."""
    for p in ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]:
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError): pass
    return None

NUM_CPUS = _cgroup_cpus() or os.cpu_count() or 1
HAS_GPU = torch.cuda.is_available()
VRAM_GB = torch.cuda.get_device_properties(0).total_mem / 1e9 if HAS_GPU else 0
DEVICE = torch.device("cuda" if HAS_GPU else "cpu")
TOTAL_RAM_GB = _container_ram_gb() or psutil.virtual_memory().total / 1e9
AVAILABLE_RAM_GB = min(psutil.virtual_memory().available / 1e9, TOTAL_RAM_GB)
```

## Hardware Use

- Keep these results in mind for ALL subsequent tasks — don't assume more than detected
- Use the best hardware for each task: GPU if available and parallelizable, multiprocessing if multiple CPUs
- Push available resources to their full potential where appropriate— don't leave hardware idle
