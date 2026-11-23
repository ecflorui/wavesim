#!/usr/bin/env bash
# ============================================================================
# tools/check_cpu.sh  --  end-to-end CPU baseline check (owner: Krish)
#
#   make check            # or: tools/check_cpu.sh   (from the repo root)
#
# Runs every reference config through the real program (./wavesim), the same
# way a user would, and checks:
#   * max |p| after the last step matches the reference table (4 digits)
#   * final.bin has exactly nx*ny*nz float32 values
#   * snapshot files appear when snapshot_every > 0
#   * compare.py passes the run against itself
#   * bad_cfl.cfg is rejected with an error naming the line
# Prints one row per config with the CPU timing. Exit code 0 = all passed.
# ============================================================================
set -uo pipefail

if [[ ! -x ./wavesim ]]; then
    echo "Build first: make cpu" >&2
    exit 2
fi

fails=0
pass() { printf '  [PASS] %s\n' "$1"; }
fail() { printf '  [FAIL] %s\n' "$1"; fails=$((fails + 1)); }

# config  expected_max_abs
refs=(
    "tiny_33 0.04977"
    "one_source_64 0.02026"
    "two_source_64 0.03303"
    "bench_128 0.02129"
)

printf '%-15s %-10s %-10s %-8s %10s\n' "config" "expected" "max|p|" "result" "total ms"
for entry in "${refs[@]}"; do
    read -r name expect <<< "$entry"
    cfg="configs/${name}.cfg"
    if ! out=$(./wavesim "$cfg" --backend cpu 2>&1); then
        printf '%-15s %-10s %-10s %-8s\n' "$name" "$expect" "-" "FAIL"
        echo "$out" | sed 's/^/    /'
        fails=$((fails + 1))
        continue
    fi
    got=$(sed -n 's/.*max |p| = \([^ ]*\).*/\1/p' <<< "$out")
    dir=$(sed -n 's/^run_dir=//p' <<< "$out")
    total=$(awk '{print $4}' "$dir/timing.txt")
    ok=$(python3 -c "import sys; g,e=float(sys.argv[1]),float(sys.argv[2]); print(abs(g-e)/e < 5e-4)" "$got" "$expect")
    verdict=$([[ $ok == True ]] && echo PASS || echo FAIL)
    [[ $verdict == FAIL ]] && fails=$((fails + 1))
    printf '%-15s %-10s %-10s %-8s %10.1f\n' "$name" "$expect" "$got" "$verdict" "$total"
done

echo
echo "Output files:"
read -r nx ny nz < <(awk '/^n[xyz] /{printf "%s ", $2}' out/one_source_64/cpu/meta.txt)
want=$((nx * ny * nz * 4))
size=$(wc -c < out/one_source_64/cpu/final.bin)
[[ $size -eq $want ]] && pass "one_source_64 final.bin is $size bytes (${nx}x${ny}x${nz} float32)" \
                      || fail "one_source_64 final.bin is $size bytes, expected $want"
slices=$(ls out/one_source_64/cpu/slice_z_*.bin 2>/dev/null | wc -l)
[[ $slices -eq 20 ]] && pass "one_source_64 wrote 20 slices (every 50 of 1000 steps)" \
                     || fail "one_source_64 wrote $slices slices, expected 20"
if python3 tools/compare.py out/one_source_64/cpu out/one_source_64/cpu > /dev/null; then
    pass "compare.py: one_source_64 against itself"
else
    fail "compare.py: one_source_64 against itself"
fi

echo
echo "Config validation:"
if err=$(./wavesim configs/bad_cfl.cfg --backend cpu 2>&1); then
    fail "bad_cfl.cfg was accepted"
else
    grep -q 'bad_cfl.cfg:[0-9]*: cfl' <<< "$err" && pass "bad_cfl.cfg rejected: ${err#error: }" \
                                                 || fail "bad_cfl.cfg rejected without a line number: $err"
fi

echo
if (( fails == 0 )); then
    echo "CPU BASELINE: ALL CHECKS PASSED"
else
    echo "CPU BASELINE: $fails CHECK(S) FAILED"
fi
exit $(( fails > 0 ))
