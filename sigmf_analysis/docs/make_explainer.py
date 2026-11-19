"""Builds docs/Beginners_Guide.pdf (concept figures + text).  Run from sigmf_analysis/:
    python docs/make_explainer.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT); sys.path.insert(0, ROOT)
import dsp  # noqa: E402
from sigmf_loader import load_run  # noqa: E402

FIG = os.path.join(HERE, "explainer_figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": False, "savefig.bbox": "tight", "savefig.dpi": 150})
BLUE, ORANGE, GREEN, GREY, RED = "#2f6db5", "#e8873a", "#3a9a5b", "#9aa3ab", "#c0392b"
recs = load_run("data/v3_g16_mcs9_classical.zip")
r = recs[30]


def box(ax, x, y, w, h, text, fc="#eef3f8", ec=BLUE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1, c="#555555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, color=c, lw=1.2))


# ---------------------------------------------------------------- 1. big picture
def fig_big_picture():
    fig, ax = plt.subplots(figsize=(10, 3.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")
    box(ax, 0.1, 1.2, 1.8, 1.0, "Phone (UE)\nsends bits", "#fff4e8", ORANGE, bold=True)
    box(ax, 2.6, 1.2, 1.6, 1.0, "Radio channel\n(air / cable)\nnoise, distortion", "#f3f3f3", GREY)
    box(ax, 4.9, 0.9, 5.0, 1.6, "", "#eef3f8", BLUE)
    ax.text(7.4, 2.32, "Base station (gNB) receiver", ha="center", fontsize=9.5, fontweight="bold")
    box(ax, 5.05, 1.05, 1.35, 1.0, "estimate\nchannel", "#ffffff", BLUE, fs=8.5)
    box(ax, 6.6, 1.05, 1.35, 1.0, "undo channel\n+ guess bits\n(LLRs)", "#ffffff", BLUE, fs=8.5)
    box(ax, 8.15, 1.05, 1.6, 1.0, "error\ncorrection\n(LDPC)", "#ffffff", BLUE, fs=8.5)
    for a, b in [(1.9, 2.6), (4.2, 5.05), (6.4, 6.6), (7.95, 8.15)]:
        arrow(ax, a, 1.7, b, 1.7)
    ax.add_patch(FancyBboxPatch((5.0, 0.95), 3.0, 1.2, boxstyle="round,pad=0.02", fc="none", ec=RED, lw=2, ls="--"))
    ax.text(6.5, 0.55, "THE GOAL: replace this part with a neural network\n(an 'AI-native receiver')",
            ha="center", color=RED, fontsize=9, fontweight="bold")
    fig.savefig(f"{FIG}/big_picture.png"); plt.close(fig)


# ---------------------------------------------------------------- 2. resource grid
def fig_grid():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    g = np.zeros((72, 13)); g[r["data_mask"]] = 1; g[r["pilot_mask"]] = 2
    cmap = matplotlib.colors.ListedColormap(["white", "#a9c7ea", ORANGE])
    axs[0].imshow(g, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2)
    axs[0].set_xlabel("OFDM symbol (time) →"); axs[0].set_ylabel("subcarrier (frequency) →")
    axs[0].set_title("One slot = 72 × 13 grid of 'resource elements'", fontsize=9.5)
    axs[0].legend(handles=[matplotlib.patches.Patch(color="#a9c7ea", label="data RE (carries 2 bits)"),
                           matplotlib.patches.Patch(color=ORANGE, label="pilot / DMRS RE (known)")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    p = 20 * np.log10(np.abs(r["rx_grid"]) + 1e-3)
    im = axs[1].imshow(p, aspect="auto", origin="lower", cmap="magma", vmin=30)
    axs[1].set_xlabel("OFDM symbol"); axs[1].set_title("What the gNB actually recorded (rx-fd-data, power in dB)",
                                                      fontsize=9.5)
    fig.colorbar(im, ax=axs[1], fraction=.04)
    fig.tight_layout(); fig.savefig(f"{FIG}/grid.png"); plt.close(fig)


# ---------------------------------------------------------------- 3. QPSK + noise
def fig_qpsk():
    rng = np.random.default_rng(1)
    pts = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)
    labels = ["00", "10", "11", "01"]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, snr, t in [(axs[0], None, "QPSK: 2 bits → 1 point"), (axs[1], 20, "received at high SNR (20 dB)"),
                       (axs[2], 3, "received at low SNR (3 dB)")]:
        if snr is not None:
            x = pts[rng.integers(0, 4, 800)]
            n = (rng.standard_normal(800) + 1j * rng.standard_normal(800)) * np.sqrt(10 ** (-snr / 10) / 2)
            ax.plot((x + n).real, (x + n).imag, ".", ms=2, alpha=.5, color=BLUE)
        ax.plot(pts.real, pts.imag, "o", color=RED, ms=7)
        if snr is None:
            for p_, l in zip(pts, labels):
                ax.text(p_.real * 1.25, p_.imag * 1.25, l, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_aspect("equal"); ax.set_title(t, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    axs[0].set_xlabel("1st bit: sign of the real (I) part\n2nd bit: sign of the imaginary (Q) part", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/qpsk.png"); plt.close(fig)


# ---------------------------------------------------------------- 4. channel + equalization (real data)
def fig_channel():
    m = r["data_mask"]
    y = dsp.data_res(r["rx_grid"], m); y = y / np.sqrt(np.mean(np.abs(y) ** 2))
    z, _ = dsp.equalize(r)
    from classical_rx import interpolate_ce
    zp, _ = dsp.equalize(r, interpolate_ce(r))
    sym = np.where(m.T)[0]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, v, t in [(axs[0], y, "1. raw received REs\n(channel rotated everything)"),
                     (axs[1], z, "2. after undoing the channel\nwith OAI's estimate"),
                     (axs[2], zp, "3. with a better estimate\n(our Python receiver)")]:
        sc = ax.scatter(v.real, v.imag, c=sym, s=5, cmap="viridis")
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal"); ax.set_title(t, fontsize=9)
        ax.axhline(0, color="grey", lw=.5); ax.axvline(0, color="grey", lw=.5); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=axs, fraction=.02, label="OFDM symbol (time)")
    fig.savefig(f"{FIG}/channel.png"); plt.close(fig)


# ---------------------------------------------------------------- 5. LLR
def fig_llr():
    x = np.linspace(-1.5, 1.5, 300)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for snr, c in [(3, ORANGE), (10, BLUE)]:
        nv = 10 ** (-snr / 10)
        ax.plot(x, 2 * np.sqrt(2) * x / nv, color=c, label=f"SNR {snr} dB")
    ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6); ax.set_ylim(-30, 30)
    ax.set_xlabel("received real part (I)"); ax.set_ylabel("LLR")
    ax.text(0.95, -9, "right of 0 → bit = 0\n(steeper = more sure)", fontsize=8, ha="center")
    ax.text(-0.85, 16, "left of 0 → bit = 1", fontsize=8, ha="center")
    ax.annotate("near 0 = unsure", xy=(0, 0), xytext=(0.35, 8), fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.legend(fontsize=8, loc="lower right"); ax.set_title("LLR = signed confidence for one bit", fontsize=9.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/llr.png"); plt.close(fig)


# ---------------------------------------------------------------- 6. files on the chain
def fig_files():
    fig, ax = plt.subplots(figsize=(10, 4.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(0.05, 3.75, "UE (phone)", fontweight="bold", color=ORANGE)
    ue = ["payload\nbits", "add CRC +\nLDPC code", "scramble", "QPSK +\nmap to grid\n(+ pilots)"]
    for i, t in enumerate(ue):
        box(ax, 0.1 + i * 1.6, 2.6, 1.35, 0.95, t, "#fff4e8", ORANGE, fs=8)
        if i: arrow(ax, 0.1 + i * 1.6 - 0.25, 3.07, 0.1 + i * 1.6, 3.07)
    ax.text(6.6, 3.07, "→  over the air  →", fontsize=9, va="center", color="grey")
    ax.text(0.05, 1.75, "gNB (base station)", fontweight="bold", color=BLUE)
    gn = ["FFT →\nreceived grid", "channel est.\nat pilots", "interpolate\nto all REs", "equalize +\nLLRs",
          "descramble\n+ LDPC decode"]
    for i, t in enumerate(gn):
        box(ax, 0.1 + i * 1.6, 0.6, 1.35, 0.95, t, "#eef3f8", BLUE, fs=8)
        if i: arrow(ax, 0.1 + i * 1.6 - 0.25, 1.07, 0.1 + i * 1.6, 1.07)
    box(ax, 8.1, 0.6, 1.35, 0.95, "decoded\npayload", "#eef3f8", BLUE, fs=8)
    arrow(ax, 7.85, 1.07, 8.1, 1.07)
    tags = [(0.78, 2.45, "tx-payload-bits", ORANGE, "top"), (3.98, 2.45, "tx-scrambled-bits\n= LABEL", RED, "top"),
            (5.58, 2.45, "tx-pilots-fd-data", ORANGE, "top"),
            (0.78, 0.45, "rx-fd-data\n= INPUT", RED, "top"), (2.38, 0.45, "raw-ce-fd-data\n= INPUT", RED, "top"),
            (3.98, 0.45, "raw-inter-ce-fd-data\n(baseline)", BLUE, "top"),
            (8.78, 0.45, "rx-payload-bits", BLUE, "top")]
    for x, y, t, c, va in tags:
        ax.text(x, y, t, ha="center", va=va, fontsize=7.5, color=c, fontweight="bold",
                family="monospace")
    fig.savefig(f"{FIG}/files.png"); plt.close(fig)


# ---------------------------------------------------------------- 7. training loop
def fig_training():
    fig, ax = plt.subplots(figsize=(10, 2.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 2.9); ax.axis("off")
    box(ax, 0.1, 1.0, 2.0, 1.2, "INPUT per slot\n72×13 grid × 4 numbers\n(received signal +\npilot channel estimate)",
        "#eef3f8", BLUE, fs=8)
    box(ax, 2.8, 1.0, 1.7, 1.2, "Neural\nnetwork\n(e.g. small CNN)", "#f4ecfa", "#7d3c98", fs=9, bold=True)
    box(ax, 5.2, 1.0, 1.7, 1.2, "OUTPUT\n2 LLRs per\ndata RE", "#eef3f8", BLUE, fs=8.5)
    box(ax, 7.6, 1.0, 2.2, 1.2, "LABEL\nthe true bits the\nphone sent\n(tx-scrambled-bits)", "#fff4e8", ORANGE, fs=8)
    arrow(ax, 2.1, 1.6, 2.8, 1.6); arrow(ax, 4.5, 1.6, 5.2, 1.6)
    ax.text(7.25, 1.6, "vs", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(7.6, 0.62, "compare → loss (binary cross-entropy: how wrong & how confident)", ha="center", fontsize=8.5)
    ax.annotate("", xy=(3.65, 1.0), xytext=(6.0, 0.45), arrowprops=dict(arrowstyle="-|>", color="#7d3c98",
                                                                         connectionstyle="arc3,rad=-0.25"))
    ax.text(2.2, 0.12, "adjust the network's weights to lower the loss; repeat over thousands of slots",
            fontsize=8.5, color="#7d3c98")
    fig.savefig(f"{FIG}/training.png"); plt.close(fig)


# ---------------------------------------------------------------- PDF
def build_pdf():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    F, B, M = "Helvetica", "Helvetica-Bold", "Courier"
    try:
        d = "/usr/share/fonts/truetype/dejavu/"
        for n, f in [("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
            pdfmetrics.registerFont(TTFont(n, d + f))
        F, B, M = "DV", "DVB", "DVM"
    except Exception:
        pass
    A = colors.HexColor("#1f4e79")
    ss = getSampleStyleSheet()
    T = ParagraphStyle("T", fontName=B, fontSize=20, textColor=A, leading=24, spaceAfter=4)
    H = ParagraphStyle("H", fontName=B, fontSize=13.5, textColor=A, spaceBefore=8, spaceAfter=5, leading=17)
    H3 = ParagraphStyle("H3", fontName=B, fontSize=10.5, textColor=colors.HexColor("#333333"), spaceBefore=6,
                        spaceAfter=2)
    P = ParagraphStyle("P", parent=ss["BodyText"], fontName=F, fontSize=9.6, leading=13.2)
    S = ParagraphStyle("S", parent=P, fontSize=8.6, leading=11.2)
    CAP = ParagraphStyle("C", parent=P, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    KEY = ParagraphStyle("K", parent=P, backColor=colors.HexColor("#fff7e6"), borderColor=colors.HexColor("#f0c27b"),
                         borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=8)
    W = A4[0] - 3.6 * cm

    def img(name, w=W):
        p = f"{FIG}/{name}" if not name.startswith("results") else name
        iw, ih = ImageReader(p).getSize()
        return Image(p, width=w, height=w * ih / iw)

    def tbl(rows, widths, size=8.4):
        st = ParagraphStyle("t", parent=P, fontSize=size, leading=size * 1.28)
        sb = ParagraphStyle("tb", parent=st, fontName=B, textColor=colors.white)
        t = Table([[Paragraph(str(c), sb if i == 0 else st) for c in row] for i, row in enumerate(rows)],
                  colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), A), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#bbbbbb")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                               ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        return t

    def bullets(items, style=P):
        out = []
        for it in items:
            out += [Paragraph("•&nbsp;&nbsp;" + it, style), Spacer(1, 2)]
        return out

    s = []
    # ---------------- page 1
    s += [Paragraph("A Beginner's Guide to the SigMF Project", T),
          Paragraph("The task, the wireless and ML ideas behind it, and how the code fits together",
                    ParagraphStyle("sub", parent=P, fontSize=10.5, textColor=colors.HexColor("#555555"))),
          Spacer(1, 8),
          Paragraph("1. The project in one paragraph", H),
          Paragraph("When your phone sends data to a cell tower, the signal gets distorted and noisy on the way. The "
                    "tower's <b>receiver</b> has to figure out which bits were sent anyway. Today that is done with "
                    "hand-designed math (the <b>classical receiver</b>). The research idea is to replace part of it with "
                    "a <b>neural network that learns to do it better</b> from examples. To learn, the network needs "
                    "many examples of <i>what the tower received</i> paired with <i>what the phone actually sent</i>. "
                    "That is exactly what this data is, and our job is to turn it into training data.", P),
          Spacer(1, 6), img("big_picture.png"),
          Paragraph("Where the data comes from: a small real 5G testbed (open-source OpenAirInterface, 'OAI') with one "
                    "computer acting as the phone (UE) and one as the tower (gNB). A recording tool saved, for every "
                    "transmission, what both sides saw, in a file format called <b>SigMF</b>.", P),
          Paragraph("<b>Your job in one line:</b> load these recordings correctly, check they are good, understand what "
                    "they contain, build a non-AI baseline to compare against, and package them so a neural network can "
                    "be trained on them later, when the bigger dataset arrives.", KEY),
          Paragraph("2. Glossary you'll keep running into", H),
          tbl([["term", "plain meaning"],
               ["UE / gNB", "the phone / the base station (5G names)"],
               ["uplink", "phone → tower direction (all of this data is uplink)"],
               ["slot", "one short transmission (0.5 ms here); one slot = one record = one training example"],
               ["IQ / complex number", "radio samples are complex numbers: a size (amplitude) and an angle (phase)"],
               ["SNR (dB)", "signal-to-noise ratio. Higher = cleaner. +10 dB = 10× more signal than before"],
               ["BER / BLER", "bit error rate / block error rate: fraction of bits / packets received wrong"],
               ["MCS", "modulation-and-coding scheme: how aggressively data is packed (here MCS 9 = QPSK)"],
               ["PRB", "physical resource block = 12 subcarriers. Here 6 PRB = 72 subcarriers"],
               ["T-Tracer / SigMF", "OAI's logging tool / an open file format for radio recordings (JSON + binary)"]],
              [3.4 * cm, W - 3.4 * cm])]
    s.append(PageBreak())

    # ---------------- page 2: wireless basics
    s += [Paragraph("3. The wireless ideas you need (and only those)", H),
          Paragraph("3.1 The resource grid", H3),
          Paragraph("5G (using OFDM) splits a transmission into a grid: <b>subcarriers</b> (frequency, rows) × <b>OFDM "
                    "symbols</b> (time, columns). Each cell is a <b>resource element (RE)</b> and carries one complex "
                    "number. Most REs carry data. A few, called <b>pilots</b> or <b>DMRS</b>, carry a sequence both "
                    "sides already know, so the receiver can measure the distortion.", P),
          img("grid.png", W * 0.82),
          Paragraph("3.2 Bits become points: QPSK", H3),
          Paragraph("Each data RE carries 2 bits by placing a point in one of 4 corners. Noise scatters the points. If "
                    "noise pushes a point across an axis, a bit comes out wrong. That's a bit error.", P),
          img("qpsk.png", W * 0.78),
          Paragraph("3.3 The channel, channel estimation and equalization", H3),
          Paragraph("On the way, the signal on each RE gets multiplied by an unknown complex number <b>h</b>, the "
                    "<b>channel</b>, which scales and rotates it. The receiver <b>estimates</b> h at the pilots (known "
                    "value in, measured value out), <b>interpolates</b> that estimate to the data REs, then "
                    "<b>equalizes</b> by dividing it out. A worse estimate means fuzzier points and more errors. Below "
                    "is real data from record 30:", P),
          img("channel.png", W * 0.84),
          Paragraph("Panel 2 shows clusters smeared by symbol colour: OAI's estimate doesn't track a slow phase drift "
                    "over time. Panel 3 is tighter. This 'room for improvement' is the opening for a neural receiver.",
                    CAP)]
    s.append(PageBreak())

    # ---------------- page 3: LLR, coding, then ML
    s += [Paragraph("3.4 LLRs: the receiver's confidence", H3),
          Paragraph("Instead of a hard 0/1, the receiver outputs a <b>log-likelihood ratio (LLR)</b> per bit: the sign "
                    "says which bit is more likely, and the size says how sure it is. The error-correction decoder "
                    "uses these confidences. <b>Producing good LLRs is exactly what the neural network will "
                    "learn.</b>", P),
          img("llr.png", W * 0.62),
          Paragraph("3.5 Coding and scrambling (just know they exist)", H3),
          Paragraph("Before sending, the phone adds a checksum (CRC), adds redundancy with an LDPC code so errors can "
                    "be fixed, and scrambles the bits. Those coded, scrambled bits are what actually ride on the "
                    "QPSK points. That makes them the right <b>labels</b> for training: one known label per bit on "
                    "every RE.", P),
          Paragraph("4. The ML ideas you need", H),
          Paragraph("<b>Supervised learning</b> = show the model an input, compare its output with the right answer "
                    "(the <b>label</b>), measure how wrong it was (the <b>loss</b>), nudge its weights, and repeat many "
                    "times.", P),
          img("training.png"),
          tbl([["idea", "what it means here"],
               ["input", "per slot: the received grid + the pilot channel estimate → a 72×13×4 array of numbers"],
               ["label", "the coded bits the phone sent (2 per data RE), from the UE recording"],
               ["loss", "binary cross-entropy (BCE): punishes wrong bits, and especially <i>confidently</i> wrong "
                        "ones"],
               ["mask", "only data REs have labels, so pilot/empty REs are ignored in the loss"],
               ["train/test split", "hold out entire <b>runs</b> (recording sessions), not random slots. Slots in one "
                                    "run look alike, so mixing them would let the model 'cheat'"],
               ["baseline", "a non-AI receiver on the same data. The model is only interesting if it beats it"],
               ["augmentation", "adding artificial noise to real recordings to create harder examples"]],
              [3.0 * cm, W - 3.0 * cm])]
    s.append(PageBreak())

    # ---------------- page 4: data
    s += [Paragraph("5. What is in the recording", H),
          Paragraph("Each of the 60 records is one slot, saved as <b>7 streams</b>. Each stream is a SigMF pair: a "
                    "<font face='%s'>.sigmf-meta</font> (JSON describing shape and settings) and a "
                    "<font face='%s'>.sigmf-data</font> (raw numbers). The diagram shows where each one is captured:"
                    % (M, M), P),
          img("files.png"),
          tbl([["stream", "what it is", "used as"],
               ["rx-fd-data", "the received grid at the tower (the noisy, distorted signal)", "<b>model input</b>"],
               ["raw-ce-fd-data", "OAI's channel estimate at the pilot REs", "<b>model input</b>"],
               ["raw-inter-ce-fd-data", "OAI's estimate spread to all REs (only the classical receiver makes it)",
                "baseline"],
               ["tx-pilots-fd-data", "the pilot values the phone was told to send (known)", "pilot positions"],
               ["tx-scrambled-bits", "the bits actually on the air", "<b>label</b>"],
               ["tx-payload-bits / rx-payload-bits", "original data sent / what the tower decoded", "error check"]],
              [4.2 * cm, 8.6 * cm, W - 12.8 * cm]),
          Paragraph("Four surprises in the files (now handled by the loader)", H3),
          Paragraph("Each of these makes the data look like garbage if you read it naively. Figuring them out was a big "
                    "part of the work, and the notebook proves each one:", P)]
    s += bullets([
        "<b>Order:</b> the numbers are stored time-column by time-column, not the way the metadata implies.",
        "<b>Packing:</b> in pilot columns, the data values are squeezed to the top rows and must be moved back.",
        "<b>Bit order:</b> every group of 8 bits is reversed. Without the fix, 50% of bits 'mismatch'; with it, 0%.",
        "<b>Pilots:</b> <font face='%s'>tx-pilots</font> is what was <i>sent</i>; the received pilots aren't saved, "
        "but the channel estimate contains the same information." % M], S)
    s.append(PageBreak())

    # ---------------- page 5: what was built
    s += [Paragraph("6. What was built, and how it fits together", H),
          Paragraph("Everything lives in <font face='%s'>sigmf_analysis/</font>. Each script answers one question from "
                    "the task list and works on any recording folder, so it can be rerun on future data unchanged." % M,
                    P), Spacer(1, 4),
          tbl([["#", "file", "the question it answers", "answer for this recording"],
               ["1", "sigmf_loader.py", "Can we read the files correctly?",
                "Yes: 60 records, all checks pass, the 4 quirks are fixed"],
               ["2", "quality_report.py", "Is the recording complete and trustworthy?",
                "All 60 complete, phone and tower agree, 0 bit errors"],
               ["3", "operating_point.py", "Which receiver ran? How clean is the link?",
                "Classical receiver; channel SNR ≈ 26 dB, but only ≈ 21 dB after OAI's estimate"],
               ["4", "channel_features.py", "What is the channel like (speed, echoes, frequency offset)?",
                "Nearly flat; small frequency offset (−73 Hz); speed and echoes not measurable here"],
               ["5", "plots.py", "Does it look right?", "Clean QPSK; the time-drift problem is visible"],
               ["6", "classical_rx.py", "How good is a simple non-AI receiver?",
                "0 errors; ≈ 25 dB, better than OAI's own ≈ 21 dB"],
               ["7", "pack_dataset.py", "Can a model train on it?",
                "Yes: inputs, labels, masks in one file (format in DATASET_SCHEMA.md)"]],
              [0.8 * cm, 3.2 * cm, 5.6 * cm, W - 9.6 * cm], size=8),
          Spacer(1, 6),
          Paragraph("The notebook <font face='%s'>SigMF_walkthrough.ipynb</font> runs all of these in order with "
                    "explanations. <font face='%s'>dsp.py</font> holds the shared math. Results (CSVs, plots) are in "
                    "<font face='%s'>results/</font>." % (M, M, M), P),
          Paragraph("7. The main takeaways", H)]
    s += bullets([
        "<b>The pipeline works and the data is clean.</b> Inputs and labels line up bit-for-bit.",
        "<b>OAI's classical channel estimate loses about 5 dB.</b> It doesn't follow a slow phase drift over time. A "
        "simple fix in Python recovers ~4 dB. This is evidence that a learned receiver has room to help.",
        "<b>This recording alone can't train a model.</b> It is one setting with a strong signal and zero errors, so "
        "there is nothing to learn yet. It is for building and testing the pipeline.",
        "<b>Some metadata is just configuration.</b> The channel fields describe the emulator settings, not what was "
        "measured, and don't match the measured SNR. Ask the testbed team before using them as labels.",
        "<b>Noise augmentation works.</b> Adding noise to real recordings gives realistic error curves. With a perfect "
        "channel estimate they match textbook theory, which validates the method:"])
    s += [KeepTogether([img("results/plots/augmented_ber_bce.png", W * 0.85),
                        Paragraph("Lower is better. Left: bit error rate vs. SNR. Right: BCE, the training loss. "
                                  "The gap between the two curves is roughly what a neural receiver could win.", CAP)])]
    s.append(PageBreak())

    # ---------------- page 6: next steps
    s += [Paragraph("8. What happens next", H)]
    s += bullets([
        "<b>Now:</b> get comfortable with the notebook. Rerun it and make sure each section makes sense using this "
        "guide.",
        "<b>Soon:</b> train a first small neural receiver on noise-augmented versions of this recording, so the "
        "training code is ready. 'Success' = its curve lands between our Python receiver and the perfect-channel curve.",
        "<b>Then:</b> add error-correction decoding (descramble + LDPC, available in open-source libraries) so we can "
        "report block error rate, the number papers usually show.",
        "<b>When the big dataset arrives</b> (a channel-emulator sweep over SNR, speed, echo, MCS): run the quality and "
        "operating-point scripts on each folder, pack everything with whole runs held out for testing, and train for "
        "real.",
        "<b>Questions for the testbed team:</b> was this cabled or emulated? Can future runs go to lower SNR (so errors "
        "happen), and can they also record the received pilots and the tower's LLRs?"])
    s += [Paragraph("9. How to read a result quickly", H),
          tbl([["if you see…", "it means…"],
               ["SNR in dB", "every +3 dB doubles signal vs noise; ~10 dB QPSK already has few errors"],
               ["post-eq SNR lower than channel SNR", "the receiver's own channel estimate is adding error"],
               ["BER = 0 on 99,360 bits", "fewer than about 3 in 100,000 bits are wrong (95% confidence), not 'perfect'"],
               ["BCE close to 0", "LLRs are confident and correct; close to 1 = coin-flipping"],
               ["'genie' / 'perfect CSI'", "a cheating reference that uses the true answer; an upper bound on performance"],
               ["records_needed_95pct", "how many slots you'd need to average for a stable measurement"]],
              [5.2 * cm, W - 5.2 * cm]),
          Spacer(1, 8),
          Paragraph("<b>Where to look:</b> SigMF_walkthrough.ipynb (the full story) · README.md (how to run) · "
                    "DATASET_SCHEMA.md (training file format) · docs/SigMF_Task_Brief.pdf (short technical summary) · "
                    "results/ (all outputs).", KEY)]

    def footer(c, doc):
        c.saveState(); c.setFont(F, 7.5); c.setFillColor(colors.grey)
        c.drawString(1.8 * cm, 1.1 * cm, "Beginner's guide — SigMF capture analysis (OAI 5G uplink)")
        c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
        c.restoreState()

    out = os.path.join(HERE, "Beginners_Guide.pdf")
    SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.7 * cm, title="Beginner's Guide to the SigMF Project").build(
        s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    for f in [fig_big_picture, fig_grid, fig_qpsk, fig_channel, fig_llr, fig_files, fig_training]:
        f()
    build_pdf()
