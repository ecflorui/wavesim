"""Task 7 — pack one or more runs into a training-ready NPZ (schema: DATASET_SCHEMA.md).

    python pack_dataset.py data/run_a.zip data/run_b.zip ... --test-runs run_b --out results/dataset.npz

The split is BY RUN: every record of a run listed in --test-runs goes to the
test set, everything else to train.  Runs with different allocation sizes are
zero-padded to the largest grid; the masks say which REs are real.
"""
import argparse
import json
import os

import numpy as np

from sigmf_loader import has_streams, load_run, run_name

META_FIELDS = ["record", "frame", "slot", "rnti", "mcs_index", "qam_mod_order", "rb_size",
               "rb_start", "nr_of_symbols", "number_of_bits", "number_of_coded_bits",
               "num_rx_antennas"]
SCHEMA_VERSION = 1


def _norm(x: np.ndarray, mask: np.ndarray) -> float:
    v = x[mask]
    return float(np.sqrt(np.mean(np.abs(v) ** 2))) if v.size else 1.0


def pack(runs, test_runs=(), normalize=True):
    samples, skipped = [], []
    for path in runs:
        name = run_name(path)
        for r in load_run(path):
            # only the streams the sample needs; raw-inter-ce is absent under the AI-native RX
            if not has_streams(r) or r["id_mismatch"]:
                skipped.append((name, r["record"], r["missing_streams"] or "id mismatch"))
                continue
            samples.append((name, r))
    if not samples:
        raise SystemExit("no complete records")
    n_sc = max(r["rx_grid"].shape[0] for _, r in samples)
    n_sym = max(r["rx_grid"].shape[1] for _, r in samples)
    qm_max = max(r["qam_mod_order"] for _, r in samples)
    nb_max = max(r["tx_coded"].size for _, r in samples)
    N = len(samples)

    X = np.zeros((N, n_sc, n_sym, 4), np.float32)
    Yg = np.zeros((N, n_sc, n_sym, qm_max), np.uint8)
    Yf = np.zeros((N, nb_max), np.uint8)
    nbits = np.zeros(N, np.int32)
    dmask = np.zeros((N, n_sc, n_sym), bool)
    pmask = np.zeros((N, n_sc, n_sym), bool)
    y_scale = np.ones(N, np.float32)
    h_scale = np.ones(N, np.float32)
    meta = {k: np.zeros(N, np.int64) for k in META_FIELDS}
    run_ids = np.empty(N, dtype=object)

    for i, (name, r) in enumerate(samples):
        S, T = r["rx_grid"].shape
        dm, pm = r["data_mask"], r["pilot_mask"]
        y = r["rx_grid"].astype(np.complex64)
        hp = np.where(pm, r["ce_pilots"], 0).astype(np.complex64)
        if normalize:
            y_scale[i] = _norm(y, dm)
            h_scale[i] = _norm(hp, pm)
            y, hp = y / y_scale[i], hp / h_scale[i]
        X[i, :S, :T, 0], X[i, :S, :T, 1] = y.real, y.imag
        X[i, :S, :T, 2], X[i, :S, :T, 3] = hp.real, hp.imag
        dmask[i, :S, :T], pmask[i, :S, :T] = dm, pm
        qm = r["qam_mod_order"]
        b = r["tx_coded"].reshape(-1, qm)  # 3GPP order: symbol-major, subcarrier ascending
        lab = np.zeros((S, T, qm), np.uint8)
        lab.transpose(1, 0, 2)[dm.T] = b
        Yg[i, :S, :T, :qm] = lab
        Yf[i, :r["tx_coded"].size] = r["tx_coded"]
        nbits[i] = r["tx_coded"].size
        for k in META_FIELDS:
            meta[k][i] = r.get(k) if r.get(k) is not None else -1
        run_ids[i] = name

    split = np.where(np.isin(run_ids, list(test_runs)), "test", "train")
    out = dict(inputs=X, labels_grid=Yg, labels_flat=Yf, n_coded_bits=nbits,
               data_mask=dmask, pilot_mask=pmask, y_scale=y_scale, h_scale=h_scale,
               run_id=run_ids.astype(str), split=split,
               **{f"meta_{k}": v for k, v in meta.items()})
    info = {"schema_version": SCHEMA_VERSION, "n_samples": N, "grid": [n_sc, n_sym],
            "runs": sorted(set(run_ids)), "test_runs": sorted(test_runs),
            "n_train": int((split == "train").sum()), "n_test": int((split == "test").sum()),
            "normalized": normalize, "skipped": [list(map(str, s)) for s in skipped],
            "input_channels": ["rx_re", "rx_im", "ce_pilot_re", "ce_pilot_im"]}
    return out, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--test-runs", nargs="*", default=[], help="run names held out as test")
    ap.add_argument("--out", default="results/dataset.npz")
    ap.add_argument("--no-normalize", action="store_true")
    a = ap.parse_args()
    names = [run_name(p) for p in a.runs]
    bad = set(a.test_runs) - set(names)
    if bad:
        raise SystemExit(f"--test-runs not among the given runs: {bad}  (available: {names})")
    arrays, info = pack(a.runs, a.test_runs, not a.no_normalize)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, info=json.dumps(info), **arrays)
    with open(os.path.splitext(a.out)[0] + "_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    for k, v in arrays.items():
        print(f"  {k:<22} {str(v.shape):<18} {v.dtype}")
    if not a.test_runs:
        print("NOTE: no --test-runs given -> every sample is in 'train'. Hold out whole runs once "
              "more than one run exists.")
