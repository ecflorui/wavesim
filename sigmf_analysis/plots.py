"""Task 5 — visualizations for a few representative records.

    python plots.py data/v3_g16_mcs9_classical.zip --records 0 30 59 --out results/plots

Per record: constellation, channel response per DMRS symbol, resource-grid heatmap
(stored vs. unpacked).  Also a run-level overlay of all constellations.
"""
import argparse
import os

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dsp  # noqa: E402
from sigmf_loader import has_gnb_interp, has_streams, load_run, run_name  # noqa: E402


def constellation(ax, r, H=None, title="", color_by_symbol=True):
    z, _ = dsp.equalize(r, H)
    sym = np.where(r["data_mask"].T)[0]  # OFDM symbol index of every data RE (3GPP order)
    sc = ax.scatter(z.real, z.imag, s=6, c=sym if color_by_symbol else None, cmap="viridis")
    a = 1 / np.sqrt(2)
    ax.scatter([a, -a, a, -a], [a, a, -a, -a], marker="+", c="r", s=120, lw=2)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_title(title or f"rec {r['record']}: SNR_eq {dsp.snr_post_eq(r, H):.1f} dB", fontsize=9)
    return sc


def channel_response(axm, axp, r):
    ce, pm, hi = r["ce_pilots"], r["pilot_mask"], r["ce_interp"]
    for s in np.where(pm.any(axis=0))[0]:
        k = np.where(pm[:, s])[0]
        l, = axm.plot(k, 20 * np.log10(np.abs(ce[k, s])), "o", ms=3, label=f"raw CE, DMRS sym {s}")
        axp.plot(k, np.unwrap(np.angle(ce[k, s])), "o", ms=3, color=l.get_color())
    for s in (dsp.full_symbols(r)[[0, -1]] if np.any(hi) else []):
        axm.plot(20 * np.log10(np.abs(hi[:, s])), "-", lw=1, label=f"gNB interp CE, sym {s}")
        axp.plot(np.unwrap(np.angle(hi[:, s])), "-", lw=1)
    axm.set_ylabel("|H| [dB, fixed-point units]"); axp.set_ylabel("phase [rad]")
    axp.set_xlabel("subcarrier"); axm.legend(fontsize=7); axm.grid(alpha=.3); axp.grid(alpha=.3)
    axm.set_title(f"rec {r['record']}: channel response", fontsize=9)


def grid_heatmap(ax, g, r, title):
    p = 20 * np.log10(np.abs(g) + 1e-3)
    im = ax.imshow(p, aspect="auto", origin="lower", cmap="magma", vmin=np.percentile(p[p > -50], 1) - 3)
    for s in r["ul_dmrs_symb_pos"]:
        ax.axvline(s, color="cyan", lw=1, ls="--")
    ax.set_xlabel("OFDM symbol (dashed = DMRS)"); ax.set_ylabel("subcarrier"); ax.set_title(title, fontsize=9)
    return im


def plot_record(r, out):
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4)
    from classical_rx import interpolate_ce
    if has_gnb_interp(r):
        sc = constellation(fig.add_subplot(gs[0, 0]), r, None, f"rec {r['record']}: gNB interp CE "
                           f"({dsp.snr_post_eq(r):.1f} dB)")
    else:
        g = dsp.snr_data_aided(r)["h_smooth"]
        sc = constellation(fig.add_subplot(gs[0, 0]), r, g, "genie channel (no gNB interp CE in run)")
    from classical_rx import interpolate_ce
    Hp = interpolate_ce(r)
    constellation(fig.add_subplot(gs[1, 0]), r, Hp, f"Python RX interp CE ({dsp.snr_post_eq(r, Hp):.1f} dB)")
    fig.colorbar(sc, ax=fig.axes[0], label="OFDM symbol")
    channel_response(fig.add_subplot(gs[0, 1:3]), fig.add_subplot(gs[1, 1:3]), r)
    grid_heatmap(fig.add_subplot(gs[0, 3]), r["rx_grid_stored"], r, "rx-fd-data AS STORED (packed)")
    im = grid_heatmap(fig.add_subplot(gs[1, 3]), r["rx_grid"], r, "rx_grid unpacked (true REs)")
    fig.colorbar(im, ax=fig.axes[-1], label="dB")
    fig.suptitle(f"frame {r['frame']} slot {r['slot']} rnti {r['rnti']} | MCS {r['mcs_index']} "
                 f"{r['modulation']} {r['rb_size']} PRB")
    fig.tight_layout()
    path = os.path.join(out, f"record_{r['record']:03d}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_run_constellation(recs, out, name):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    from classical_rx import interpolate_ce
    for r in recs:
        if not has_gnb_interp(r):
            continue
        z, _ = dsp.equalize(r); axs[0].plot(z.real, z.imag, ".", ms=1, alpha=.3, c="C0")
        z, _ = dsp.equalize(r, interpolate_ce(r)); axs[1].plot(z.real, z.imag, ".", ms=1, alpha=.3, c="C1")
    for ax, t in zip(axs, ["gNB interpolated CE", "Python RX (raw CE, linear interp)"]):
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_title(f"{t}: all {len(recs)} records", fontsize=10)
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    fig.tight_layout(); p = os.path.join(out, f"{name}_constellation_all.png"); fig.savefig(p, dpi=110)
    plt.close(fig); return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--records", type=int, nargs="*", help="record numbers (default: first, middle, last)")
    ap.add_argument("--out", default="results/plots")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = [r for r in load_run(a.run) if has_streams(r)]
    by_id = {r["record"]: r for r in recs}
    ids = a.records or [recs[0]["record"], recs[len(recs) // 2]["record"], recs[-1]["record"]]
    for i in ids:
        print(plot_record(by_id[i], a.out))
    print(plot_run_constellation(recs, a.out, run_name(a.run)))
