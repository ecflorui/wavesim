"""Task 3b + Task 4 — per-record link-quality and channel features.

    python channel_features.py data/v3_g16_mcs9_classical.zip --out results

Writes <out>/<run>_channel_features.csv, <run>_feature_stability.csv and
<run>_features.png.  ``compute_features`` is importable for the notebook.

Feature definitions (all per record = one slot)
-----------------------------------------------
snr_data_aided_db  channel SNR from known tx symbols + smoothed y/x (genie, label-aided)
snr_pilot_db       SNR from the raw DMRS channel estimate vs. its smoothed version (blind)
snr_post_eq_db     SNR after equalizing with the gNB's interpolated CE (1/EVM^2)
rho_t_a_b          |corr| of raw pilot CE between DMRS symbols a and b (time selectivity)
rho_t_mean_corr    same, averaged and corrected for estimation noise
doppler_hz         Jakes inversion of rho_t_mean_corr  (0 when indistinguishable from 1)
cfo_hz             common phase drift between DMRS symbols -> residual frequency offset
timing_offset_ns   linear phase slope of H(f) -> residual timing offset
rho_f_lag1         |corr| of H between adjacent subcarriers (frequency selectivity)
delay_spread_ns    RMS width of the PDP (taps < -20 dB ignored), *apparent*
ds_floor_ns        the same metric on a perfectly flat channel (window/resolution floor)
mag_var_norm       var(|H|)/mean(|H|)^2 across the slot (on the smoothed channel)
"""
import argparse
import os

import numpy as np
import pandas as pd

import dsp
from sigmf_loader import has_gnb_interp, has_streams, load_run, run_name


def compute_features(r: dict, thresh_db: float = 20.0) -> dict:
    scs = r["scs_hz"]
    f = {"record": r["record"], "frame": r["frame"], "slot": r["slot"]}
    da = dsp.snr_data_aided(r)
    f["snr_data_aided_db"] = da["snr_db"]
    f["snr_pilot_db"] = dsp.snr_pilot(r)
    gi = has_gnb_interp(r)
    f["snr_post_eq_db"] = dsp.snr_post_eq(r) if gi else np.nan
    f["snr_post_eq_blind_db"] = dsp.snr_post_eq(r, aided=False) if gi else np.nan
    f["cfg_snr_db"] = r.get("cfg_snr_db")

    # ---- time selectivity
    tc = dsp.time_correlation(r)
    tsym = dsp.symbol_duration(scs)
    rhos, dts, phases = [], [], []
    for (a, b), rho in tc.items():
        f[f"rho_t_{a}_{b}"] = abs(rho)
        rhos.append(abs(rho)); dts.append((b - a) * tsym); phases.append(np.angle(rho))
    # CE noise: per raw-CE value SNR = pilot SNR (+3 dB pair averaging, see dsp.snr_pilot)
    snr_ce = 10 ** ((f["snr_pilot_db"] + 3) / 10)
    rho_c = np.mean(rhos) * (1 + 1 / snr_ce)
    f["rho_t_mean"] = float(np.mean(rhos))
    f["rho_t_mean_corr"] = float(rho_c)
    f["doppler_hz"] = dsp.doppler_from_corr(min(rho_c, 1.0), float(np.mean(dts)))
    f["doppler_hz_uncorrected"] = dsp.doppler_from_corr(float(np.mean(rhos)), float(np.mean(dts)))
    f["cfo_hz"] = float(np.mean(np.array(phases) / (2 * np.pi * np.array(dts))))

    # ---- frequency selectivity (gNB interpolated CE on symbols with all subcarriers;
    #      the genie-smoothed channel if the gNB did not produce one)
    H = (r["ce_interp"] if gi else da["h_smooth"])[:, dsp.full_symbols(r)]
    c1 = dsp.freq_correlation(H, 1)
    f["rho_f_lag1"] = float(abs(c1))
    f["timing_offset_ns"] = float(-np.angle(c1) / (2 * np.pi * scs) * 1e9)
    ds = dsp.rms_delay_spread(H, scs, thresh_db)
    f["delay_spread_ns"] = ds["rms_s"] * 1e9
    flat = dsp.rms_delay_spread(np.ones((H.shape[0], 1)), scs, thresh_db)
    f["ds_floor_ns"] = flat["rms_s"] * 1e9
    f["ds_excess_ns"] = float(np.sqrt(max(f["delay_spread_ns"] ** 2 - f["ds_floor_ns"] ** 2, 0)))
    f["ds_resolution_ns"] = ds["resolution_s"] * 1e9

    # ---- magnitude variation
    hs = dsp.data_res(da["h_smooth"], r["data_mask"])
    f["mag_var_norm"] = float(np.var(np.abs(hs)) / np.mean(np.abs(hs)) ** 2)
    if gi:
        hi = dsp.data_res(r["ce_interp"], r["data_mask"])
        f["mag_var_norm_gnb_ce"] = float(np.var(np.abs(hi)) / np.mean(np.abs(hi)) ** 2)
    return f


def stability(df: pd.DataFrame, rel_tol: float = 0.10) -> pd.DataFrame:
    """Spread of each feature and records needed for the mean to be within
    +-tol (0.1 dB for SNR, 1e-4 for correlations, 5 Hz / 5 ns, else 10 % relative) at 95 % confidence:
    N = (1.96 * std / tol)^2."""
    tol_abs = {"doppler_hz": 5.0, "doppler_hz_uncorrected": 5.0, "cfo_hz": 5.0,
               "timing_offset_ns": 5.0, "delay_spread_ns": 5.0, "ds_excess_ns": 5.0}
    rows = []
    for c in df.columns:
        if c in ("record", "frame", "slot", "cfg_snr_db", "ds_floor_ns", "ds_resolution_ns") \
                or not np.issubdtype(df[c].dtype, np.number):
            continue
        if c.startswith("snr_"):
            tol_abs[c] = 0.1  # dB
        elif c.startswith("rho_"):
            tol_abs[c] = 1e-4  # correlation sits near 1: absolute tolerance
        v = df[c].dropna()
        if len(v) < 2:
            continue
        tol = tol_abs.get(c, rel_tol * abs(v.mean()) if v.mean() != 0 else np.nan)
        n = (1.96 * v.std(ddof=1) / tol) ** 2 if tol else np.nan
        rows.append({"feature": c, "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(),
                     "max": v.max(), "tolerance": tol, "records_needed_95pct": int(np.ceil(n))})
    return pd.DataFrame(rows)


def plot_features(df: pd.DataFrame, path: str, title: str):
    import matplotlib.pyplot as plt
    panels = [
        (["snr_data_aided_db", "snr_pilot_db", "snr_post_eq_db"], "SNR [dB]"),
        (["rho_t_0_5", "rho_t_5_10", "rho_t_mean_corr"], "|time corr| between DMRS"),
        (["doppler_hz", "doppler_hz_uncorrected", "cfo_hz"], "Hz"),
        (["delay_spread_ns", "ds_floor_ns", "timing_offset_ns"], "ns"),
        (["rho_f_lag1"], "|freq corr| lag 1"),
        (["mag_var_norm", "mag_var_norm_gnb_ce"], "var|H| / mean|H|^2"),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for ax, (cols, lab) in zip(axs.ravel(), panels):
        for c in cols:
            if c in df and df[c].notna().any():
                ax.plot(df["record"], df[c], ".-", lw=1, ms=4, label=c)
        ax.set_ylabel(lab); ax.grid(alpha=.3); ax.legend(fontsize=7, loc="best")
    for ax in axs[-1]:
        ax.set_xlabel("record")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=120); plt.close(fig)


def snr_histogram(df: pd.DataFrame, path: str, title: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for c in ["snr_data_aided_db", "snr_pilot_db", "snr_post_eq_db"]:
        if df[c].isna().all():
            continue
        ax.hist(df[c], bins=15, alpha=.6, label=f"{c}: {df[c].mean():.1f} ± {df[c].std():.2f} dB")
    if df["cfg_snr_db"].notna().any():
        ax.axvline(df["cfg_snr_db"].iloc[0], color="k", ls="--", label="metadata snr_esn0_db (config)")
    ax.set_xlabel("SNR [dB]"); ax.set_ylabel("records"); ax.legend(fontsize=8); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def run(path: str, out: str) -> pd.DataFrame:
    name = run_name(path)
    recs = [r for r in load_run(path) if has_streams(r)]
    df = pd.DataFrame([compute_features(r) for r in recs])
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{name}_channel_features.csv"), index=False)
    st = stability(df)
    st.to_csv(os.path.join(out, f"{name}_feature_stability.csv"), index=False)
    plot_features(df, os.path.join(out, f"{name}_features.png"), name)
    snr_histogram(df, os.path.join(out, f"{name}_snr_hist.png"), name)
    with pd.option_context("display.width", 160, "display.float_format", "{:.4g}".format):
        print(st.to_string(index=False))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    for p in a.runs:
        run(p, a.out)
