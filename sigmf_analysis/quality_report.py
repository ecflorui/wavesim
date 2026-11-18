"""Task 2 — data-quality report for one or more run folders (or .zip files).

    python quality_report.py data/v3_g16_mcs9_classical.zip --out results

Prints the summary and writes  <out>/<run>_quality_records.csv  and
<out>/<run>_quality_summary.txt.
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from sigmf_loader import load_run, records_to_frame, run_name

CONST_FIELDS = ["mcs_index", "modulation", "qam_mod_order", "rb_size", "rb_start",
                "nr_of_symbols", "ul_dmrs_symb_pos", "number_of_bits",
                "number_of_coded_bits", "num_rx_antennas", "rnti", "receiver_type"]


def bit_errors(a, b):
    if a is None or b is None:
        return np.nan, np.nan
    if a.size != b.size:
        return np.nan, max(a.size, b.size)
    return int(np.sum(a != b)), a.size


def report(run: str, out: str) -> pd.DataFrame:
    recs = load_run(run)
    name = run_name(run)
    df = records_to_frame(recs)
    errs, nbits = zip(*(bit_errors(r["tx_bits"], r["rx_bits"]) for r in recs))
    df["payload_bit_errors"] = errs
    df["payload_bits_compared"] = nbits
    df["ber"] = df["payload_bit_errors"] / df["payload_bits_compared"]
    # gNB vs UE identity check (frame, slot, rnti per side)
    gnb_ue = []
    for r in recs:
        ids = r["stream_ids"]
        g = {v for k, v in ids.items() if not k.startswith("tx-payload") and not k.startswith("tx-scrambled")}
        u = {v for k, v in ids.items() if k.startswith("tx-payload") or k.startswith("tx-scrambled")}
        gnb_ue.append(bool(g and u and g != u))
    df["gnb_ue_mismatch"] = gnb_ue

    lines = [f"Run: {name}"]
    lines.append(f"1. Records found: {len(df)};  complete (all 7 streams): {int(df.complete.sum())}")
    inc = df[~df.complete]
    for _, row in inc.iterrows():
        lines.append(f"     record {row.record}: missing {row.missing_streams}")
    if not df.hash_ok.all():
        lines.append(f"   SHA-512 mismatch in records: {list(df.record[~df.hash_ok])}")
    else:
        lines.append("   SHA-512 of every .sigmf-data matches its .sigmf-meta")
    tot_e = np.nansum(df.payload_bit_errors)
    tot_b = np.nansum(df.payload_bits_compared)
    lines.append(f"2. Payload BER overall: {tot_e:.0f} / {tot_b:.0f} = {tot_e / tot_b:.3e}"
                 + ("  (zero errors -> BER < %.1e at 95%% conf.)" % (3 / tot_b) if tot_e == 0 else ""))
    lines.append(f"   per-record BER: min {df.ber.min():.3e}  max {df.ber.max():.3e}")
    n_na = int(df.payload_bit_errors.isna().sum())
    lines.append(f"3. Records decoded perfectly: {int((df.payload_bit_errors == 0).sum())} / {len(df)}"
                 + (f"  ({n_na} not evaluable: payload stream missing)" if n_na else ""))
    mm = df[df.id_mismatch | df.gnb_ue_mismatch]
    lines.append(f"4. frame/slot/RNTI mismatches: {len(mm)}"
                 + ("" if mm.empty else f" -> records {list(mm.record)}"))
    dup = df.duplicated(subset=["frame", "slot"], keep=False)
    if dup.any():
        lines.append(f"   duplicate (frame, slot) pairs in records {list(df.record[dup])}")
    lines.append(f"   frames {df.frame.min()}..{df.frame.max()}, slots used {sorted(int(s) for s in df.slot.unique())}")
    lines.append("5. Metadata distribution:")
    for f in CONST_FIELDS:
        c = Counter(df[f].astype(str))
        mode = c.most_common(1)[0][0]
        odd = list(df.record[df[f].astype(str) != mode])
        lines.append(f"   {f:<22} {dict(c)}" + ("  CONSTANT" if not odd else f"  DIFFERS in records {odd}"))
    # notes on metadata that does not describe the data
    pat = {r.get("payload_bit_pattern") for r in recs}
    if "zeros" in pat and any(r["tx_bits"] is not None and r["tx_bits"].any() for r in recs):
        lines.append("   NOTE: metadata says payload_bit_pattern='zeros' but the payload bits are not all zero")
    text = "\n".join(lines)
    print(text)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{name}_quality_records.csv"), index=False)
    with open(os.path.join(out, f"{name}_quality_summary.txt"), "w") as f:
        f.write(text + "\n")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folder(s) or .zip")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    for run in a.runs:
        report(run, a.out)
        print()
