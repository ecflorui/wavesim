# Wave Sim — ECE 379K

GPU-accelerated 3D wave simulator. See the week plan for the full contract.

| Piece | Owner | Files |
|---|---|---|
| CPU reference | Krish | `include/physics.hpp`, `src/cpu_solver.cpp`, `tools/test_cpu.cpp`, `tools/check_cpu.sh`, `src/live_api.cpp`, `tools/live_gui.*`, `docs/PHYSICS.md`, `docs/CPU.md` |
| Config | Rishik | `include/config.hpp`, `src/config.cpp`, `configs/`, `tools/plot_slice.py` |
| GPU | Connor | `src/gpu_solver.cu` (shares `include/solver.hpp` with Krish) |
| Runner + tests | Venthan | `include/output.hpp`, `src/main.cpp`, `src/output.cpp`, `tools/compare.py`, `tools/bench.sh` |

## Quick start (Linux, WSL or Lonestar6)

```bash
make cpu                                          # build ./wavesim
./wavesim configs/one_source_64.cfg --backend cpu # run one config, writes out/one_source_64/cpu/
make check                                        # full CPU baseline check
make gui && python3 tools/live_gui.py             # live view at http://localhost:8000
```

Example configs live in `configs/`: `tiny_33`, `one_source_64`, `two_source_64`,
`bench_128`, `sine_64`, and `bad_cfl` (deliberately rejected).

Status of each piece: [docs/CPU.md](docs/CPU.md) (CPU baseline),
[docs/TESTING.md](docs/TESTING.md) and [docs/RESULTS.md](docs/RESULTS.md) (runner, benchmark).
`src/gpu_stub.cpp` is a placeholder until the CUDA solver lands.
