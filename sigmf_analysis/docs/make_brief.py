"""Builds docs/SigMF_Task_Brief.pdf.  Run from sigmf_analysis/:  python docs/make_brief.py"""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FONT, BOLD, MONO = "Helvetica", "Helvetica-Bold", "Courier"
try:
    d = "/usr/share/fonts/truetype/dejavu/"
    pdfmetrics.registerFont(TTFont("DV", d + "DejaVuSans.ttf"))
    pdfmetrics.registerFont(TTFont("DVB", d + "DejaVuSans-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("DVM", d + "DejaVuSansMono.ttf"))
    FONT, BOLD, MONO = "DV", "DVB", "DVM"
except Exception:
    pass

ACCENT = colors.HexColor("#1f4e79")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName=BOLD, fontSize=17, textColor=ACCENT, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName=BOLD, fontSize=12.5, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("P", parent=ss["BodyText"], fontName=FONT, fontSize=9.4, leading=12.6, alignment=TA_LEFT)
SM = ParagraphStyle("SM", parent=P, fontSize=8.2, leading=10.4)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
SUB = ParagraphStyle("SUB", parent=P, fontSize=10, textColor=colors.HexColor("#444444"))
CODE = ParagraphStyle("CODE", fontName=MONO, fontSize=6.6, leading=8.4, backColor=colors.HexColor("#f4f6f8"),
                      borderPadding=4)


def table(rows, widths, head=True, size=8.2):
    st = ParagraphStyle("t", parent=P, fontSize=size, leading=size * 1.25)
    sb = ParagraphStyle("tb", parent=st, fontName=BOLD, textColor=colors.white)
    data = [[Paragraph(str(c), sb if (head and i == 0) else st) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bbbbbb")),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head:
        style += [("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")])]
    t.setStyle(TableStyle(style))
    return t


def img(path, width):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    return Image(path, width=width, height=width * h / w)


def footer(c, doc):
    c.saveState(); c.setFont(FONT, 7.5); c.setFillColor(colors.grey)
    c.drawString(1.8 * cm, 1.1 * cm, "SigMF capture analysis — OAI 5G NR uplink — task brief")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
    c.restoreState()


def build(out="docs/SigMF_Task_Brief.pdf"):
    W = A4[0] - 3.6 * cm
    s = []
    s += [Paragraph("SigMF Capture Analysis — Task Brief", H1),
          Paragraph("Turning OAI testbed uplink captures into training data for a neural receiver", SUB),
          Paragraph("Dataset <b>v3_g16_mcs9_classical</b> · 60 uplink slots · captured 18 Nov 2026 · "
                    "code: <font face='%s'>sigmf_analysis/</font> in the repository" % MONO, CAP),
          Spacer(1, 8)]

    s += [Paragraph("1. What the task is", H2),
          Paragraph(
              "The OAI 5G SA testbed (2× Jetson AGX Orin, 2× USRP B210) records every uplink PUSCH slot "
              "<b>simultaneously at the base station (gNB) and the phone (UE)</b>, using the OAI T-Tracer and the NI "
              "data-recording app. The files are in SigMF format: a JSON <i>.sigmf-meta</i> that describes the samples "
              "and a raw binary <i>.sigmf-data</i>. Because the gNB records what it <i>received</i> and the UE records "
              "what it <i>sent</i>, every slot comes with its own ground truth, which is what you need to train a "
              "receiver with supervised learning.", P), Spacer(1, 4),
          Paragraph(
              "The goal is an offline Python pipeline that (1) loads the captures correctly, (2) checks their quality, "
              "(3) works out the operating point from the data, (4) extracts channel features, (5) visualizes them, "
              "(6) gives a classical baseline receiver, and (7) packages everything as training data. The pipeline must "
              "run unchanged on the coming channel-emulator sweep (many SNR / Doppler / delay-spread / MCS runs).", P)]

    s += [Paragraph("2. What is in each record", H2),
          table([["stream", "side", "what it is", "role for ML"],
                 ["rx-fd-data", "gNB", "received resource grid after the FFT (data REs only)", "<b>input</b>"],
                 ["raw-ce-fd-data", "gNB", "channel estimate at DMRS pilots (received ÷ known pilot)", "<b>input</b>"],
                 ["raw-inter-ce-fd-data", "gNB", "classical estimate interpolated to all REs", "baseline (absent with AI receiver)"],
                 ["tx-pilots-fd-data", "gNB", "the DMRS sequence that was sent (reference, not received)", "pilot positions"],
                 ["tx-scrambled-bits", "UE", "coded + scrambled bits = exactly what is on the air", "<b>label</b> (per-RE bits)"],
                 ["tx-payload-bits", "UE", "transport block before coding", "end-to-end ground truth"],
                 ["rx-payload-bits", "gNB", "what the gNB decoded", "BER / BLER check"]],
                [3.4 * cm, 1.2 * cm, 8.0 * cm, W - 12.6 * cm]),
          Spacer(1, 5),
          Preformatted(
              "UE : payload -> CRC -> LDPC -> rate-match -> scramble -> QPSK -> RE map (+DMRS) -> OFDM -> RF\n"
              "      [tx-payload]                       [tx-scrambled]            [tx-pilots]\n"
              "gNB: RF -> FFT -> channel est. -> interpolate -> equalize -> LLR -> descramble -> LDPC -> payload\n"
              "           [rx-fd]  [raw-ce]      [raw-inter-ce]                 ^ a neural receiver replaces   [rx-payload]\n"
              "                                                                   this block (grid -> LLRs)", CODE),
          Spacer(1, 4),
          Paragraph("This run: MCS 9 (QPSK, rate ≈ 0.66), 6 PRB = 72 subcarriers × 13 OFDM symbols, DMRS type 1 in "
                    "symbols 0/5/10, 1 Rx antenna. 828 data REs → 1656 coded bits per slot; 1128-bit transport block "
                    "(matches the 3GPP TS 38.214 TBS calculation).", P)]

    s += [Paragraph("3. Four things that are not obvious from the files", H2),
          Paragraph("Read naively, the grid comes out scrambled and the coded BER is 50 %. The loader fixes all four "
                    "issues, and the notebook proves each one on the data:", P),
          table([["#", "quirk", "evidence", "fix"],
                 ["1", "IQ grids are stored symbol-major, although counts say [72, 13]",
                  "DMRS lands on 3 symbols only with Fortran order", "reshape(order='F')"],
                 ["2", "In DMRS symbols the 36 data REs are <i>packed</i> into rows 0–35",
                  "packed CE equals neighbouring symbol's odd subcarrier, ratio 1.000", "unpack to odd subcarriers"],
                 ["3", "Bit streams: 1 bit per byte, but LSB-first in every group of 8",
                  "coded BER 0.50 as stored → 0.00 after reversing", "reverse each 8-bit group"],
                 ["4", "tx-pilots is the <i>sent</i> DMRS; received pilots are not stored",
                  "values ±23170 ± j23170 (ideal); zeros in rx-fd-data", "use raw-ce as pilot input"]],
                [0.5 * cm, 5.6 * cm, 6.4 * cm, W - 12.5 * cm])]
    s.append(PageBreak())

    s += [Paragraph("4. The seven tasks and what was delivered", H2),
          table([["task", "deliverable", "result on this run"],
                 ["1 Loader", "sigmf_loader.py, demo_loader.py",
                  "60 records; shapes from metadata; SHA-512 verified; missing or truncated streams flagged (tested)"],
                 ["2 Quality", "quality_report.py → CSV/TXT",
                  "60/60 complete; payload BER 0/67 680 (< 4.4e-5 @95 %); 0 ID mismatches; all parameters constant"],
                 ["3 Operating point", "operating_point.py → Markdown",
                  "<b>classical</b> receiver (interp. CE in 100 % of records). Channel SNR 26.2 dB, post-eq 20.8 dB"],
                 ["4 Channel features", "channel_features.py → CSV, plots, stability table",
                  "flat channel; CFO −73 Hz; timing −107 ns; Doppler and delay spread <b>not measurable</b> here"],
                 ["5 Visualization", "plots.py → PNGs", "clean QPSK; clusters rotate with symbol under gNB CE"],
                 ["6 Python receiver", "classical_rx.py → coded BER",
                  "0 / 99 360 coded-bit errors; 24.9 dB post-eq SNR, ≈4 dB better than the gNB's interpolation"],
                 ["7 Packaging", "pack_dataset.py, DATASET_SCHEMA.md → NPZ",
                  "inputs (N,72,13,4), labels (N,72,13,2), masks, metadata; split <b>by run</b>"]],
                [2.6 * cm, 4.6 * cm, W - 7.2 * cm]),
          Spacer(1, 6)]

    s += [Paragraph("5. Key findings", H2)]
    pts = [
        "<b>The data is clean and aligned.</b> Every gNB/UE pair agrees on frame, slot and RNTI, and the labels line up "
        "with the received constellation bit-for-bit.",
        "<b>OAI's classical channel estimate costs ~5 dB.</b> The channel itself is at 26 dB SNR, but after the gNB's "
        "estimate the demapper sees only ~21 dB. The estimate is held constant over symbols 1–4, 6–8 and 9–12, so a "
        "−73 Hz residual frequency offset turns the constellation within each segment. Linear interpolation in time "
        "recovers ~4 dB, which shows concretely how much room a learned receiver has.",
        "<b>Metadata channel fields are configuration, not measurement.</b> They say TDL-D, 5 m/s, 50 ns, 20 dB, but the "
        "task sheet says cabled + attenuator, and 26 dB is measured. Do not use them as labels until the setup is confirmed.",
        "<b>Some features cannot be measured at 6 PRB / one slot.</b> Delay spread: the resolution is 463 ns and the "
        "measured value equals a flat channel's. Doppler: the time decorrelation is explained by estimation noise. "
        "Both need wider allocations, more slots per point, or emulator labels.",
        "<b>Real channels can be noise-augmented.</b> Adding noise to the measured grid gives BER curves; with perfect "
        "CSI they match QPSK theory exactly, and the Python receiver is ~1.5–2 dB behind (figure below).",
    ]
    for p in pts:
        s.append(Paragraph("• " + p, P)); s.append(Spacer(1, 2.5))

    s += [KeepTogether([Spacer(1, 4), img("results/plots/record_030.png", W * 0.8),
                        Paragraph("Record 30. Left: constellations with the gNB estimate (top, coloured by OFDM symbol, "
                                  "rotation visible) and the Python receiver (bottom). Middle: |H| and phase across "
                                  "subcarriers. Right: the grid as stored (packed) vs. unpacked.", CAP)])]
    s.append(PageBreak())

    s += [Paragraph("6. How this data trains a receiver", H2),
          Paragraph(
              "Each slot becomes one sample. The network input is the received grid <i>y</i> and the pilot channel "
              "estimate (zero elsewhere), as 4 real channels on the 72 × 13 grid. The target is the coded, scrambled bit "
              "of every data RE (2 per RE for QPSK). A convolutional/ResNet receiver outputs one LLR per bit and is "
              "trained with <b>binary cross-entropy masked to data REs</b>. Because the labels are the bits on the air, "
              "the trained model drops in where the classical equalizer + demapper sit, and OAI's descrambler and LDPC "
              "decoder stay unchanged. Models are compared on BCE and BER/BLER, and the split is <b>by run</b> so that "
              "whole operating points can be held out.", P),
          Spacer(1, 5),
          KeepTogether([img("results/plots/augmented_ber_bce.png", W * 0.92),
                        Paragraph("Semi-synthetic sweep: noise added to the real captures (y and pilot estimate). "
                                  "Left: coded-bit BER; right: BCE of the LLRs, the training loss. The genie (perfect-CSI) "
                                  "curve sits on QPSK theory; the gap to the Python receiver is the headroom.", CAP)])]

    s += [Paragraph("7. Limits of this capture and next steps", H2),
          Paragraph("This run is <b>one operating point</b>: a single MCS, a strong static link, 60 slots and no bit "
                    "errors. It validates the pipeline but cannot train or rank models. For the next captures:", P)]
    for p in [
        "Confirm the setup (emulator vs. cable + attenuator) and record the <i>actual</i> channel parameters per run.",
        "Sweep SNR into the waterfall (≈ −2 … 10 dB for QPSK) so real decoding errors occur; add more MCS values.",
        "Use a wider allocation (≥ 25–50 PRB) if delay spread matters; record ≥ 25–50 slots per point for Doppler.",
        "If possible, also record the received DMRS REs and the gNB's LLRs (for a direct neural-vs-OAI comparison).",
        "Fix the <i>payload_bit_pattern</i> metadata ('zeros', but the payload is random). Keep one run per folder.",
        "Then: run every script on each new folder unchanged, pack with <font face='%s'>--test-runs</font>, and train "
        "a first neural receiver against the Python and OAI baselines." % MONO,
    ]:
        s.append(Paragraph("• " + p, P)); s.append(Spacer(1, 2))
    s += [Spacer(1, 6), Paragraph("8. How to run", H2),
          Preformatted("cd sigmf_analysis && pip install -r requirements.txt\n"
                       "jupyter notebook SigMF_walkthrough.ipynb          # guided tour, all tasks\n"
                       "python quality_report.py  <run folder or .zip>    # each task is also a script\n"
                       "python pack_dataset.py run_a run_b --test-runs run_b --out results/dataset.npz", CODE)]

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.7 * cm, title="SigMF Capture Analysis — Task Brief")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", out)


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    build()
