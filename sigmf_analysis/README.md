# SigMF capture analysis — OAI 5G NR uplink (T-Tracer + NI data recorder)

Offline Python tooling to load, check, understand and package the uplink PUSCH captures recorded at
the gNB and UE of the OAI testbed, as the input pipeline for a neural (AI-native) receiver.

**Start with [`SigMF_walkthrough.ipynb`](SigMF_walkthrough.ipynb)**. It explains the data from scratch and
runs every task. A short brief is in [`docs/SigMF_Task_Brief.pdf`](docs/SigMF_Task_Brief.pdf).

```bash
pip install -r requirements.txt
python demo_loader.py      data/v3_g16_mcs9_classical.zip            # Task 1
python quality_report.py   data/v3_g16_mcs9_classical.zip            # Task 2
python operating_point.py  data/v3_g16_mcs9_classical.zip            # Task 3
python channel_features.py data/v3_g16_mcs9_classical.zip            # Task 4
python plots.py            data/v3_g16_mcs9_classical.zip            # Task 5
python classical_rx.py     data/v3_g16_mcs9_classical.zip            # Task 6
python pack_dataset.py     data/run_a data/run_b --test-runs run_b   # Task 7
```

Every script takes one or more **run folders or `.zip` files** and writes to `results/` (`--out` to change).
Nothing about the run is hard-coded: shapes, DMRS positions, modulation etc. come from each `.sigmf-meta`.

## Files

| file | task | what it does |
|---|---|---|
| `sigmf_loader.py` | 1 | `load_run(path) -> list[dict]`. Handles the format quirks below, checks frame/slot/RNTI across streams, verifies SHA-512, flags missing/truncated streams |
| `demo_loader.py` | 1 | prints the first record's shapes and metadata |
| `quality_report.py` | 2 | completeness, payload BER, perfect-decode count, ID mismatches, metadata constancy → CSV + TXT |
| `operating_point.py` | 3 | classical vs AI-native receiver (with evidence), SNR per record (3 estimators), histogram → Markdown |
| `channel_features.py` | 4 | SNR, time correlation / Doppler, CFO, timing offset, frequency correlation, delay spread, \|H\| variation → CSV, stability table, plots |
| `plots.py` | 5 | constellation, channel response, resource-grid heatmap per record |
| `classical_rx.py` | 6 | Python receiver from `rx-fd-data` + `raw-ce-fd-data` only → coded BER, compared with the gNB's interpolation and a genie channel |
| `pack_dataset.py` | 7 | training NPZ (inputs, labels, masks, metadata), train/test split **by run**; schema in [`DATASET_SCHEMA.md`](DATASET_SCHEMA.md) |
| `dsp.py` | — | shared DSP: 38.211 QAM mapper, LLRs, equalization, SNR estimators, channel-feature primitives |
| `data/` | — | the capture `v3_g16_mcs9_classical.zip` (60 records) |
| `results/` | — | all outputs for this run (CSVs, PNGs, Markdown, packed NPZ) |

## Format quirks of the recorded files (all handled by the loader)

1. **Symbol-major storage.** `serialization:counts = [72, 13]` (subcarriers, symbols), but the samples run through
   symbol 0's subcarriers first → reshape with `order="F"`.
2. **Data REs packed in DMRS symbols.** In symbols 0/5/10 the 36 data REs (odd subcarriers) are stored in rows
   0–35. `rx_grid`/`ce_interp` are returned unpacked; the stored versions are kept as `*_stored`.
3. **Bits are LSB-first per byte.** `ri8_le`, one bit per byte, with every group of 8 reversed. Without the fix the
   coded BER is 0.5; with it, 0. Disable with `load_run(..., bit_order="as_stored")`.
4. **`tx-pilots-fd-data` is the transmitted DMRS** (±23170 ± j23170, int16 full scale), not the received pilots.
   Received pilot REs are not recorded; `raw-ce-fd-data` (= y_p/x_p) carries that information. OAI holds each
   raw CE value over 2 adjacent DMRS subcarriers.

The channel-model fields in the metadata (`tdl_d`, 5 m/s, 50 ns, `snr_esn0_db` 20) are the emulator
**configuration** from the recorder config, not measurements. They do not match the measured SNR (26 dB).
`payload_bit_pattern = "zeros"` is also wrong (the payload is random). Treat both as untrusted.

## Results for `v3_g16_mcs9_classical`

| | |
|---|---|
| Records | 60/60 complete, hashes OK, IDs consistent gNB↔UE |
| Parameters (constant) | MCS 9, QPSK, 6 PRB from 0, 13 symbols, DMRS 0/5/10 (type 1), 1 Rx ant, 1128 payload / 1656 coded bits |
| Payload BER | 0 / 67 680 (< 4.4e-5 @ 95 %) — 60/60 perfect |
| Receiver | classical (interp. CE present & non-zero in 100 %, mean \|H\| 57.6) |
| Channel SNR (data-aided) | 26.2 ± 0.14 dB · blind pilot estimate 25.3 ± 0.7 dB |
| Post-eq SNR, gNB CE | 20.8 ± 0.6 dB → ~5 dB lost in OAI's estimate (held constant over symbols 1–4 / 6–8 / 9–12) |
| Python RX (Task 6) | 0 coded-bit errors / 99 360; post-eq SNR 24.9 dB (linear time interp), 24.0 (hold), 21.9 (slot mean) |
| Doppler | not measurable: raw \|ρ\| 0.9987 is explained by CE noise; corrected 4 ± 8 Hz |
| Delay spread | not resolvable: apparent 247 ns = flat-channel floor 249 ns (resolution 463 ns at 6 PRB) |
| CFO / timing | −73 ± 6 Hz residual CFO; −107 ± 4 ns timing offset |
| Noise-augmented sweep | genie-CSI receiver on real channels matches QPSK theory; Python RX ≈ 1.5–2 dB behind |

This is **one operating point** (one MCS, a strong static link, no bit errors). It is for building and validating
the pipeline, not for training or drawing conclusions about receiver performance.
