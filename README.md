# Wave Sim — ECE 379K

GPU-accelerated 3D wave simulator. See the week plan for the full contract.

| Piece | Owner | Files |
|---|---|---|
| CPU reference | Krish | `include/physics.hpp`, `src/cpu_solver.cpp`, `tools/test_cpu.cpp`, `docs/PHYSICS.md` |
| Config | Rishik | `include/config.hpp`, `src/config.cpp`, `configs/`, `tools/plot_slice.py` |
| GPU | Connor | `src/gpu_solver.cu` (shares `include/solver.hpp` with Krish) |
| Runner + tests | Venthan | `include/output.hpp`, `src/main.cpp`, `src/output.cpp`, `tools/compare.py`, `tools/bench.sh` |

```
make cpu && ./wavesim configs/tiny_33.cfg --backend cpu
make test-cpu            # CPU sanity checks
./test_cpu --all-refs    # + full reference table
```

`src/config.cpp`, `src/main.cpp`, `src/output.cpp`, `src/gpu_stub.cpp` are currently
placeholder stubs so the CPU solver can build and run; their owners replace them.

## SigMF capture analysis (OAI 5G uplink)

`sigmf_analysis/` holds the offline Python pipeline for the OAI T-Tracer SigMF captures: loader, quality report,
operating point, channel features, plots, Python reference receiver and training-data packer. Start with
`sigmf_analysis/SigMF_walkthrough.ipynb`; the brief is `sigmf_analysis/docs/SigMF_Task_Brief.pdf`.
