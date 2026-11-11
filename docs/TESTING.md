# WaveSim testing

Run the commands below from the repository root in Bash. For local testing on
Windows, use WSL.

## CPU sanity checks

Build the CPU executable and run the solver's sanity checks:

```bash
make cpu test-cpu
```

Expect `ALL CHECKS PASSED (0 failures)`. The checks cover a zero field with no
sources, symmetry, stability at CFL 0.5, instability at CFL 0.6, and the
`tiny_33` reference peak.

## CPU runner and output smoke test

While the config loader is still a stub, it ignores the config path and uses
the built-in `tiny_33` case. After the real parser lands, these commands
require `configs/tiny_33.cfg`.

```bash
./wavesim configs/tiny_33.cfg --backend cpu --steps 10
grep '^steps 10$' out/tiny_33/cpu/meta.txt

./wavesim configs/tiny_33.cfg --backend cpu
grep '^steps 200$' out/tiny_33/cpu/meta.txt
wc -c < out/tiny_33/cpu/final.bin
cat out/tiny_33/cpu/timing.txt

python3 tools/compare.py out/tiny_33/cpu out/tiny_33/cpu
```

The default run should report a peak absolute pressure near `0.04977`. Its
`final.bin` should contain `143748` bytes for the 33 x 33 x 33 float32 field.
`timing.txt` contains four values in milliseconds: initialization, stepping,
final host copy, and total. The self-comparison should print
`PASS relative_l2=0 max_abs_diff=0`.

## Snapshots with a real config

Once the real configs land, run a CPU case with `snapshot_every` set to a
positive interval. Check for `slice_<axis>_<step>.bin` at each exact multiple
of the interval through the final step. `final.bin` holds the last step.
Intermediate snapshot copies and file writes are outside the timing totals.

## CPU-GPU integration on Lonestar6

Once the CUDA solver and configs are available, run on a Lonestar6 GPU compute
node with CUDA loaded:

```bash
(
    set -e
    make gpu
    for case_name in tiny_33 one_source_64 two_source_64 bench_128; do
        tools/bench.sh "configs/${case_name}.cfg"
    done
)
```

For each config, the script runs the CPU baseline pinned to one logical CPU
and the GPU solver on the same node. It prints both timing rows, the speedup
(CPU total divided by GPU total), and the field comparison.

The comparator uses the CPU field as the reference. A comparison passes when
the grid shapes match, both fields contain finite values, the CPU L2 norm is
nonzero, and the relative L2 error is strictly below `0.01`. The comparator
also reports the maximum absolute difference and exits with status 0 on PASS.
