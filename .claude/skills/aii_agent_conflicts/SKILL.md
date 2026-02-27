---
name: aii_agent_conflicts
description: Isolation rules for agents running code in parallel on the same machine. Covers process safety (PID-only management), GPU sharing, RAM awareness, and file isolation. Use when multiple agents execute artifacts concurrently.
---

Other agents are running code on this machine AT THE SAME TIME as you. They may run scripts with the same names (method.py, eval.py, etc.) — these are NOT zombies.

**Philosophy: maximize utilization of what's CURRENTLY available. Check real-time usage before launching work, adapt to what other processes are already consuming.**

## Processes — Only touch your own PIDs

- Track every process you launch by PID: `uv run method.py & PID=$!`
- NEVER `killall`, `pkill -f`, or `ps aux | grep ... | kill` — this kills other agents' work
- Only `kill $PID` for PIDs you started yourself

## Resource Detection — Check current usage, then use what's free

- **CPU**: check `psutil.cpu_percent(interval=1)` — scale `NUM_WORKERS` proportionally to available capacity (e.g. 30% used → use ~70% of cores)
- **GPU**: check `nvidia-smi` for current VRAM usage, size batches to fit in free VRAM
- **RAM**: check `psutil.virtual_memory().available`, size buffers to what's actually free
- **If resource unavailable**: wait up to ~5min polling every 10-15s to see if it frees up. If still blocked, fall back to an alternative that fits (CPU instead of GPU, smaller workers, etc.)
- **On OOM / contention**: reduce batch size or workers and retry. Don't kill other processes.
- **Report degradation**: if you had to use a suboptimal path (e.g. CPU fallback, reduced batch size) because resources were occupied, mention it in your final output to the user.

## Files — Stay in your workspace

- Write all files inside your workspace directory only
- Temp files in `./tmp/` inside your workspace, not `/tmp/`
