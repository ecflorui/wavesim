# Packed dataset schema (v1) — `pack_dataset.py`

One compressed NPZ per packing call, plus `<name>_info.json`. Load with `np.load(path)`.
N = number of samples (one per uplink slot), S × T = largest grid among the packed runs
(subcarriers × OFDM symbols; smaller allocations are zero-padded, see the masks), Qm = largest modulation order,
B = largest number of coded bits.

| key | shape | dtype | content |
|---|---|---|---|
| `inputs` | (N, S, T, 4) | float32 | channels: `[Re y, Im y, Re Ĥp, Im Ĥp]` |
| `labels_grid` | (N, S, T, Qm) | uint8 | coded (scrambled) bits on their RE; 0 where `data_mask` is False or beyond the sample's Qm |
| `labels_flat` | (N, B) | uint8 | the same bits in 3GPP order (symbol by symbol, subcarrier ascending); valid length `n_coded_bits` |
| `n_coded_bits` | (N,) | int32 | valid length of `labels_flat` |
| `data_mask` | (N, S, T) | bool | RE carries PUSCH data — **mask the loss with this** |
| `pilot_mask` | (N, S, T) | bool | RE carries DMRS (where `Ĥp` is defined) |
| `y_scale`, `h_scale` | (N,) | float32 | per-sample RMS used to normalize `y` (over data REs) and `Ĥp` (over pilot REs); multiply back for fixed-point units |
| `run_id` | (N,) | str | run folder name |
| `split` | (N,) | str | `"train"` / `"test"` — assigned **per run** via `--test-runs` |
| `meta_record`, `meta_frame`, `meta_slot`, `meta_rnti`, `meta_mcs_index`, `meta_qam_mod_order`, `meta_rb_size`, `meta_rb_start`, `meta_nr_of_symbols`, `meta_number_of_bits`, `meta_number_of_coded_bits`, `meta_num_rx_antennas` | (N,) | int64 | per-sample metadata (−1 if missing) |
| `info` | () | str | JSON copy of `<name>_info.json` |

## Definitions

* **y** — `rx-fd-data`, the received frequency-domain grid with data REs moved back to their true subcarriers.
  DMRS REs are **zero**: OAI's recorder does not store the received pilots.
* **Ĥp** — `raw-ce-fd-data`, OAI's least-squares channel estimate at the DMRS REs (= received pilot / known pilot),
  zero elsewhere. This is the pilot information the network gets. OAI holds each value over 2 adjacent DMRS subcarriers.
* **labels** — `tx-scrambled-bits` from the UE: the bits after CRC, LDPC, rate matching and scrambling, i.e. exactly
  the bits mapped onto the QPSK/QAM symbols. Bit `j` of RE `(k, l)` in `labels_grid[..., j]` is the j-th bit of that
  symbol's 38.211 mapping (bit 0 → I sign, bit 1 → Q sign for QPSK).
* LLR convention used by the baselines: LLR > 0 ⇔ bit = 0.

## Intended use

A neural receiver maps `inputs` → per-RE LLRs of shape (S, T, Qm), trained with binary cross-entropy against
`labels_grid`, masked by `data_mask` (and by `j < meta_qam_mod_order` when runs with different Qm are mixed).
`labels_flat` lets the LLRs go straight into a descrambler + LDPC decoder to measure BLER.

Excluded from packing: records missing `rx-fd-data`, `raw-ce-fd-data` or `tx-scrambled-bits`, or with a frame/slot/RNTI
mismatch (listed under `skipped` in the info JSON). `raw-inter-ce-fd-data` is *not* required, so AI-native runs pack too.

## Split

Pass whole run names to `--test-runs`. Records of one run are strongly correlated (same channel realization, CFO,
gain), so a per-record split would leak. With a single run (as now) everything is `train`.
