#!/bin/bash
echo "=== OS ===" && uname -sr && \
grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null && \
echo "=== CPU ===" && \
(cpus=""; \
cg2=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null) && \
[[ "$cg2" != "max"* ]] && q=${cg2%% *} && p=${cg2##* } && \
cpus=$(( (q + p - 1) / p )); \
if [ -z "$cpus" ]; then \
cg1q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) && \
cg1p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null) && \
[[ "$cg1q" -gt 0 ]] && cpus=$(( (cg1q + cg1p - 1) / cg1p )); fi; \
echo "${cpus:-$(nproc)} CPUs") && \
lscpu | grep 'Model name' && \
echo "=== RAM ===" && \
(memlimit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || \
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null); \
if [ -n "$memlimit" ] && [ "$memlimit" != "max" ] && \
[ "$memlimit" -lt 1000000000000 ] 2>/dev/null; then \
echo "$((memlimit / 1073741824)) GB (container limit)"; \
else free -h | awk '/Mem:/{print $2" total, "$7" available"}'; fi) && \
echo "=== DISK ===" && df -h . | awk 'NR==2{print $2" total, "$4" free"}' && \
echo "=== GPU ===" && \
(gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
--format=csv,noheader 2>/dev/null) && \
echo "$gpu" | awk -F', ' '{print $1", VRAM: "$2" ("$3" free), Util: "$4}' || \
echo "No GPU")
