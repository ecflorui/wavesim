"""Task 6 — classical reference receiver in Python.

Uses only ``rx-fd-data`` and ``raw-ce-fd-data`` (never the gNB's interpolated
estimate or decoded bits):

1. Channel interpolation of the raw DMRS estimate
   - frequency: unique pilot values (OAI repeats each over 2 subcarriers) ->
     linear phase removed -> linear interpolation onto every subcarrier
   - time: ``--time linear`` (between DMRS symbols, held at the edges),
     ``hold`` (nearest DMRS symbol) or ``mean`` (one estimate for the slot)
2. ZF equalization, QPSK LLRs (scale from the blind pilot SNR, see ``demodulate``),
   hard decisions (3GPP RE order)
3. Coded BER against ``tx-scrambled-bits``

It also runs the same demapper on the gNB's ``raw-inter-ce-fd-data`` and on a
genie channel (smoothed y/x with the known tx symbols) for comparison.

    python classical_rx.py data/v3_g16_mcs9_classical.zip --out results
"""
import argparse
import os

import numpy as np
import pandas as pd

import dsp
from sigmf_loader import has_gnb_interp, has_streams, load_run, run_name


def interpolate_ce(r: dict, time_mode: str = "linear") -> np.ndarray:
    ce, pm = r["ce_pilots"], r["pilot_mask"]
    n_sc, n_sym = ce.shape
    sc = np.arange(n_sc)
    dmrs = np.where(pm.any(axis=0))[0]
    Hf = np.zeros((n_sc, len(dmrs)), complex)
    for j, s in enumerate(dmrs):
        k = np.where(pm[:, s])[0]
        v = ce[k, s].astype(complex)
        keep = np.r_[True, np.abs(np.diff(v)) > 1e-6]
        if (~keep).sum() > 0.4 * keep.size:
            # value held over a pair of pilots -> place it at the pair centre
            kk = k[keep] + (k[1] - k[0]) / 2
            vv = v[keep]
        else:
            kk, vv = k, v
        c = np.sum(np.conj(vv[:-1]) * vv[1:])
        slope = np.angle(c) / np.mean(np.diff(kk))
        rot_in, rot_out = np.exp(1j * slope * kk), np.exp(1j * slope * sc)
        d = vv / rot_in
        Hf[:, j] = (np.interp(sc, kk, d.real) + 1j * np.interp(sc, kk, d.imag)) * rot_out
    H = np.zeros((n_sc, n_sym), complex)
    t = np.arange(n_sym)
    if time_mode == "mean":
        H[:] = Hf.mean(axis=1, keepdims=True)
    elif time_mode == "hold":
        idx = np.abs(t[:, None] - dmrs[None, :]).argmin(axis=1)
        H = Hf[:, idx]
    else:  # linear between DMRS symbols, flat outside
        for i in range(n_sc):
            H[i] = np.interp(t, dmrs, Hf[i].real) + 1j * np.interp(t, dmrs, Hf[i].imag)
    return H


def demodulate(r: dict, H: np.ndarray, noise_var: float | None = None):
    """Equalize + QPSK LLRs. Returns (llr, hard bits, post-eq SNR dB).

    LLRs use the matched-filter form u = y conj(H) / mean|H|^2 (equivalent to ZF with
    per-RE noise N/|H|^2, without dividing by a possibly tiny, noisy estimate).  The LLR
    scale comes from the blind pilot-domain SNR (``dsp.snr_pilot``) inflated by the
    channel-estimation error.  Moment-based estimators (M2M4, decision-directed EVM) were
    tried and found to collapse at <= 2 dB with a noisy channel estimate.
    Hard decisions are identical to ZF.
    """
    if r["qam_mod_order"] != 2:
        raise NotImplementedError("demapper implemented for QPSK only (this dataset)")
    m = r["data_mask"]
    y, h = dsp.data_res(r["rx_grid"], m), dsp.data_res(H, m)
    u = y * np.conj(h) / np.mean(np.abs(h) ** 2)
    m2 = float(np.mean(np.abs(u) ** 2))
    if noise_var is None:
        # blind SINR: per-RE SNR from the DMRS estimate, plus the CE error itself
        # (OAI pair-averages pilots -> CE noise ~ half the RE noise)
        snr = 10 ** (dsp.snr_pilot(r) / 10)
        sinr = 1 / (1 / snr + 1 / (2 * snr))
    else:
        sinr = 1 / noise_var
    sig = m2 * sinr / (1 + sinr)
    nv = m2 - sig
    llr = dsp.qpsk_llr(u / np.sqrt(sig), nv / sig)
    return llr, dsp.hard_bits(llr), dsp.snr_post_eq(r, H)


def nmse_db(a, b, mask):
    a, b = dsp.data_res(a, mask), dsp.data_res(b, mask)
    g = np.vdot(a, b) / np.vdot(a, a)  # remove the arbitrary fixed-point scale
    return float(10 * np.log10(np.mean(np.abs(b - g * a) ** 2) / np.mean(np.abs(b) ** 2)))


def run(path: str, out: str, time_mode: str = "linear") -> pd.DataFrame:
    name = run_name(path)
    rows, llrs = [], []
    for r in load_run(path):
        if not has_streams(r):
            continue
        lab = r["tx_coded"]
        row = {"record": r["record"]}
        genie = dsp.snr_data_aided(r)["h_smooth"]
        paths = [("py", interpolate_ce(r, time_mode)), ("genie", genie)]
        if has_gnb_interp(r):
            paths.insert(1, ("gnb", r["ce_interp"]))
        for tag, H in paths:
            llr, hb, snr = demodulate(r, H)
            row[f"{tag}_bit_errors"] = int(np.sum(hb != lab))
            row[f"{tag}_post_eq_snr_db"] = snr
            row[f"{tag}_min_abs_llr"] = float(np.min(np.abs(llr)))
            if tag == "py":
                llrs.append(llr)
        row["n_coded_bits"] = lab.size
        row["nmse_py_vs_genie_db"] = nmse_db(interpolate_ce(r, time_mode), genie, r["data_mask"])
        if has_gnb_interp(r):
            row["nmse_py_vs_gnb_db"] = nmse_db(interpolate_ce(r, time_mode), r["ce_interp"], r["data_mask"])
            row["nmse_gnb_vs_genie_db"] = nmse_db(r["ce_interp"], genie, r["data_mask"])
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{name}_classical_rx_{time_mode}.csv"), index=False)
    nb = df.n_coded_bits.sum()
    print(f"Run {name}  ({len(df)} records, {nb} coded bits, time interpolation = {time_mode})")
    for tag, lbl in [("py", "Python RX (raw CE + own interp)"), ("gnb", "gNB interpolated CE"),
                     ("genie", "genie smoothed channel")]:
        if f"{tag}_bit_errors" not in df:
            print(f"  {lbl:<34} not available in this run")
            continue
        e = df[f"{tag}_bit_errors"].sum()
        ber = e / nb
        bound = f"  (< {3 / nb:.1e} @95%)" if e == 0 else ""
        print(f"  {lbl:<34} coded BER {ber:.2e}{bound}   post-eq SNR "
              f"{df[f'{tag}_post_eq_snr_db'].mean():5.2f} ± {df[f'{tag}_post_eq_snr_db'].std():.2f} dB")
    for c, lbl in [("nmse_py_vs_gnb_db", "py vs gNB CE"), ("nmse_py_vs_genie_db", "py vs genie"),
                   ("nmse_gnb_vs_genie_db", "gNB vs genie")]:
        if c in df:
            print(f"  NMSE  {lbl:<14} {df[c].mean():6.2f} dB")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results")
    ap.add_argument("--time", default="linear", choices=["linear", "hold", "mean"])
    a = ap.parse_args()
    for p in a.runs:
        run(p, a.out, a.time)
