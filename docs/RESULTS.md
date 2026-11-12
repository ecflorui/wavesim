# WaveSim checkout results

## CPU validation

The CPU sanity tests passed on commit `4d346f1`. The built-in `tiny_33` case
completed 200 steps and reached a peak absolute pressure of `0.0497707`.
Its `final.bin` contained 143,748 bytes, as expected for a 33 x 33 x 33
float32 field. Comparing the CPU field against itself returned zero relative
L2 error and zero maximum absolute difference.

## Lonestar6 CPU-GPU results

Run the CPU baseline and GPU backend on the same GPU compute node using
`tools/bench.sh`. Record both total times in milliseconds and calculate
speedup as CPU total divided by GPU total. See [TESTING.md](TESTING.md).

| Config | CPU-GPU verdict | Relative L2 error | CPU total (ms) | GPU total (ms) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `tiny_33` | Pending | - | - | - | - |
| `one_source_64` | Pending | - | - | - | - |
| `two_source_64` | Pending | - | - | - | - |
| `bench_128` | Pending | - | - | - | - |
