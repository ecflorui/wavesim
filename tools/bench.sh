#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'Usage: %s <config.cfg> (from the repository root)\n' "$0" >&2
    exit 2
fi
if [[ ! -f "$1" ]]; then
    printf 'Config not found: %s\n' "$1" >&2
    exit 2
fi
if [[ ! -x ./wavesim ]]; then
    printf 'Build ./wavesim with make gpu first.\n' >&2
    exit 2
fi

run_backend() {
    local output status=0 line dir=""
    output=$("$@") || status=$?
    printf '%s\n' "$output" >&2
    if (( status != 0 )); then
        return "$status"
    fi

    while IFS= read -r line; do
        if [[ $line == run_dir=* ]]; then
            dir=${line#run_dir=}
        fi
    done <<< "$output"
    if [[ -z $dir ]]; then
        printf 'WaveSim did not report its run directory. Rebuild with make gpu.\n' >&2
        return 1
    fi
    printf '%s\n' "$dir"
}

cpu_id=$(python3 -c 'import os; print(min(os.sched_getaffinity(0)))')
printf 'CPU baseline pinned to logical CPU %s\n' "$cpu_id"
cpu_dir=$(run_backend taskset -c "$cpu_id" ./wavesim "$1" --backend cpu)
gpu_dir=$(run_backend ./wavesim "$1" --backend gpu)

python3 - "$cpu_dir" "$gpu_dir" <<'PY'
import math
from pathlib import Path
import sys

def read_timing(directory):
    path = Path(directory) / "timing.txt"
    try:
        values = [float(token) for token in path.read_text().split()]
        if len(values) != 4 or any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("expected four finite, nonnegative times")
        if values[-1] <= 0:
            raise ValueError("total must be positive")
        return values
    except (OSError, ValueError) as error:
        sys.exit(f"{path}: {error}")

cpu = read_timing(sys.argv[1])
gpu = read_timing(sys.argv[2])
print(f"{'Backend':<8} {'Init ms':>12} {'Steps ms':>12} {'Copy ms':>12} {'Total ms':>12}")
for name, values in (("CPU", cpu), ("GPU", gpu)):
    print(f"{name:<8} " + " ".join(f"{value:12.6f}" for value in values))
print(f"Speedup (CPU total / GPU total): {cpu[-1] / gpu[-1]:.3f}x")
PY

python3 tools/compare.py "$cpu_dir" "$gpu_dir"
