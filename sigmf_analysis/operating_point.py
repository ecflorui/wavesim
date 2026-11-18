"""Task 3 — identify the operating point of a run from the data itself.

    python operating_point.py data/v3_g16_mcs9_classical.zip --out results

3a  Which receiver ran: classical (raw-inter-ce-fd-data populated) or AI-native (absent/zero).
3b  Link quality: per-record SNR (three estimators), mean/std, histogram, metadata cross-check.
Writes <out>/<run>_operating_point.md and <run>_snr_hist.png.
"""
import argparse
import os

import numpy as np
import pandas as pd

from channel_features import compute_features, snr_histogram
from sigmf_loader import has_streams, load_run, run_name


def receiver_evidence(recs) -> dict:
    present, nonzero, mags = 0, 0, []
    for r in recs:
        h = r["ce_interp"]
        if h is None:
            continue
        present += 1
        m = np.abs(h)
        if m.max() > 0:
            nonzero += 1
            mags.append(m[m > 0].mean())
    n = len(recs)
    rx_types = {r.get("receiver_type") for r in recs}
    verdict = "classical" if nonzero == n else ("AI-native" if nonzero == 0 else "MIXED")
    return {"n": n, "present": present, "nonzero": nonzero,
            "frac_populated": nonzero / n if n else np.nan,
            "mean_mag": float(np.mean(mags)) if mags else 0.0,
            "metadata_receiver_type": sorted(map(str, rx_types)), "verdict": verdict}


def run(path: str, out: str) -> str:
    name = run_name(path)
    recs = load_run(path)
    ev = receiver_evidence(recs)
    df = pd.DataFrame([compute_features(r) for r in recs if has_streams(r)])
    os.makedirs(out, exist_ok=True)
    snr_histogram(df, os.path.join(out, f"{name}_snr_hist.png"), name)
    r0 = recs[0]
    cols = ["snr_data_aided_db", "snr_pilot_db", "snr_post_eq_db"]
    s = df[cols].agg(["mean", "std", "min", "max"]).T
    md = [f"# Operating point — `{name}`", "",
          "## 3a. Receiver",
          f"**Verdict: {ev['verdict']} receiver.**", "",
          f"- `raw-inter-ce-fd-data` present in {ev['present']}/{ev['n']} records, "
          f"non-zero in {ev['nonzero']}/{ev['n']} ({100 * ev['frac_populated']:.0f} %)",
          f"- mean |H| of the interpolated estimate over non-zero REs: {ev['mean_mag']:.1f} (fixed-point units)",
          f"- metadata `phy_freq_domain_receiver_type`: {ev['metadata_receiver_type']} "
          + ("(consistent with the data)" if all(("classical" in m) == (ev["verdict"] == "classical")
                                                for m in ev["metadata_receiver_type"])
             else "(**INCONSISTENT with the data — check the metadata**)"), "",
          "## 3b. Link quality (per record = one slot)", "",
          "| estimator | what it measures | mean [dB] | std [dB] | min | max |",
          "|---|---|---|---|---|---|"]
    desc = {"snr_data_aided_db": "channel SNR, known tx symbols + smoothed y/x (genie)",
            "snr_pilot_db": "raw DMRS estimate vs. its smoothed version (blind)",
            "snr_post_eq_db": "after equalizing with the gNB interpolated CE (1/EVM²)"}
    for c in cols:
        if s.loc[c].isna().all():
            md.append(f"| `{c}` | {desc[c]} | n/a (no gNB interpolated CE in this run) | | | |")
            continue
        md.append(f"| `{c}` | {desc[c]} | {s.loc[c, 'mean']:.2f} | {s.loc[c, 'std']:.2f} | "
                  f"{s.loc[c, 'min']:.2f} | {s.loc[c, 'max']:.2f} |")
    md += ["",
           f"Metadata cross-check: `snr_esn0_db` = {r0.get('cfg_snr_db')} dB, channel model "
           f"`{r0.get('cfg_channel_model')}`, speed {r0.get('cfg_speed_mps')} m/s, delay spread "
           f"{(r0.get('cfg_delay_spread_s') or 0) * 1e9:.0f} ns.  These are the *channel-emulator "
           "configuration* copied from the recorder config, not measurements.",
           "",
           f"- Measured channel SNR ({s.loc['snr_data_aided_db', 'mean']:.1f} dB) differs from the "
           f"configured {r0.get('cfg_snr_db')} dB → the metadata SNR field should not be used as a label "
           "until the setup (emulator vs. cabled attenuator) is confirmed.",
           (f"- The SNR the demapper actually sees with the gNB's classical CE is "
            f"{s.loc['snr_post_eq_db', 'mean']:.1f} dB, i.e. "
            f"{s.loc['snr_data_aided_db', 'mean'] - s.loc['snr_post_eq_db', 'mean']:.1f} dB of channel-estimation loss."
            if ev["verdict"] == "classical" else "- No gNB interpolated CE: post-equalization SNR of the gNB path not available."),
           f"- Zero coded/payload bit errors are expected at this SNR for QPSK (uncoded BER ≈ Q(√SNR) < 1e-20).",
           "", f"![SNR histogram]({name}_snr_hist.png)", ""]
    text = "\n".join(md)
    with open(os.path.join(out, f"{name}_operating_point.md"), "w") as f:
        f.write(text)
    print(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    for p in a.runs:
        run(p, a.out)
