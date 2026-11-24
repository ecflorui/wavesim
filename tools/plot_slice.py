#!/usr/bin/env python3
"""
tools/plot_slice.py -- turn saved slices into pictures and a GIF

    python3 tools/plot_slice.py out/one_source_64/cpu            # strip of 6 frames -> slices.png
    python3 tools/plot_slice.py out/one_source_64/cpu --all      # + one PNG per slice and wave.gif

Reads meta.txt for the grid size and slice axis, loads every
slice_<axis>_<step>.bin (raw float32), and draws them with a blue-white-red
colour map. All frames share one colour range (+/- the largest |p| over all
slices), so 0 is always white and brightness can be compared between frames.
Needs numpy and matplotlib (pip3 install --user numpy matplotlib).

Placeholder for Rishik's plot_slice.py from the week plan; same interface.
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir):
    meta = {}
    with open(os.path.join(run_dir, "meta.txt")) as f:
        for line in f:
            if line.strip():
                k, v = line.split(maxsplit=1)
                meta[k] = v.strip()
    nx, ny, nz = (int(meta[k]) for k in ("nx", "ny", "nz"))
    axis = meta["snapshot_axis"]
    rows, cols = {"z": (ny, nx), "y": (nz, nx), "x": (nz, ny)}[axis]
    files = sorted(glob.glob(os.path.join(run_dir, f"slice_{axis}_*.bin")))
    if not files:
        sys.exit(f"no slice files in {run_dir}: set snapshot_every > 0 in the config and run again")
    steps = [int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in files]
    slices = [np.fromfile(f, np.float32).reshape(rows, cols) for f in files]
    dt = float(meta.get("dt", "nan"))
    return meta, axis, steps, slices, dt


def draw(ax, s, v, title):
    im = ax.imshow(s, cmap="seismic", vmin=-v, vmax=v, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="e.g. out/one_source_64/cpu")
    ap.add_argument("--all", action="store_true", help="also write one PNG per slice and wave.gif")
    ap.add_argument("--frames", type=int, default=6, help="frames in the overview strip (default 6)")
    ap.add_argument("--out", help="output folder (default: <run_dir>/plots)")
    args = ap.parse_args()

    meta, axis, steps, slices, dt = load(args.run_dir)
    out = args.out or os.path.join(args.run_dir, "plots")
    os.makedirs(out, exist_ok=True)
    v = max(float(np.abs(s).max()) for s in slices) or 1.0
    label = lambda st: f"step {st}" + ("" if np.isnan(dt) else f"  ({st * dt * 1e3:.2f} ms)")

    # Overview strip: evenly spaced frames.
    pick = np.unique(np.linspace(0, len(slices) - 1, min(args.frames, len(slices))).round().astype(int))
    fig, axes = plt.subplots(1, len(pick), figsize=(2.6 * len(pick), 3.0), squeeze=False)
    for ax, p in zip(axes[0], pick):
        im = draw(ax, slices[p], v, label(steps[p]))
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="pressure")
    fig.suptitle(f"{meta['nx']}×{meta['ny']}×{meta['nz']}, {axis}-slice at index "
                 f"{meta.get('snapshot_index', '?')}, {meta.get('backend', '')}", fontsize=11)
    strip = os.path.join(out, "slices.png")
    fig.savefig(strip, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {strip}  ({len(pick)} frames, colour range ±{v:.3g})")

    if args.all:
        for st, s in zip(steps, slices):
            fig, ax = plt.subplots(figsize=(4, 4))
            draw(ax, s, v, label(st))
            fig.savefig(os.path.join(out, f"slice_{axis}_{st:06d}.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, ax = plt.subplots(figsize=(4, 4))
        im = draw(ax, slices[0], v, label(steps[0]))

        def frame(n):
            im.set_data(slices[n])
            ax.set_title(label(steps[n]), fontsize=10)
            return (im,)

        anim = FuncAnimation(fig, frame, frames=len(slices), blit=False)
        gif = os.path.join(out, "wave.gif")
        anim.save(gif, writer=PillowWriter(fps=6))
        plt.close(fig)
        print(f"wrote {len(slices)} PNGs and {gif}")


if __name__ == "__main__":
    main()
