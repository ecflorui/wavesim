"""SigMF loader for OAI T-Tracer / NI data-recording-app captures.

Usage
-----
    from sigmf_loader import load_run
    records = load_run("data/v3_g16_mcs9_classical")      # folder
    records = load_run("data/v3_g16_mcs9_classical.zip")  # or the zip directly

Each record is one uplink PUSCH slot captured at the gNB and the UE.  Every
shape and parameter is read from the ``.sigmf-meta`` files; nothing about this
particular run (72 subcarriers, 13 symbols, QPSK, ...) is hard-coded.

Format facts established from the data (see README / notebook for the proof)
-------------------------------------------------------------------------
1. IQ grids are stored *symbol-major*: the first ``n_subcarriers`` samples are
   OFDM symbol 0, the next block is symbol 1, ...  ``serialization:counts`` is
   ``[subcarriers, ofdm_symbols]``, so the correct reshape is Fortran order.
2. In DMRS symbols the gNB *packs* the data REs (the subcarriers not used by the
   DMRS) into the first rows of the column and leaves the rest zero.  The loader
   moves them back to their true subcarriers (``rx_grid``) and keeps the stored
   version as ``rx_grid_stored``.  Same for ``raw-inter-ce-fd-data``.
3. The bit streams (``ri8_le``) hold one bit per byte, but each group of 8 is
   in LSB-first order.  Reversing every 8-bit group makes the gNB's equalized
   constellation match ``tx-scrambled-bits`` with zero errors.  The loader does
   this by default (``bit_order="lsb"``); pass ``bit_order="as_stored"`` to
   disable it.
4. ``tx-pilots-fd-data`` is the *transmitted* DMRS sequence (fixed-point
   amplitude 23170 = 2^15/sqrt 2), not the received pilots.  The received
   pilot REs themselves are not recorded; ``raw-ce-fd-data`` (= y_p / x_p)
   carries that information.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import tempfile
import warnings
import zipfile
from typing import Optional

import numpy as np

# stream prefix -> key in the record dict
STREAMS = {
    "rx-fd-data": "rx_grid",
    "tx-pilots-fd-data": "pilots",
    "raw-ce-fd-data": "ce_pilots",
    "raw-inter-ce-fd-data": "ce_interp",
    "rx-payload-bits": "rx_bits",
    "tx-payload-bits": "tx_bits",
    "tx-scrambled-bits": "tx_coded",
}
GNB_STREAMS = ["rx-fd-data", "tx-pilots-fd-data", "raw-ce-fd-data",
               "raw-inter-ce-fd-data", "rx-payload-bits"]
UE_STREAMS = ["tx-payload-bits", "tx-scrambled-bits"]
IQ_STREAMS = ["rx-fd-data", "tx-pilots-fd-data", "raw-ce-fd-data", "raw-inter-ce-fd-data"]
BIT_STREAMS = ["rx-payload-bits", "tx-payload-bits", "tx-scrambled-bits"]

# SigMF core datatypes -> numpy
_DTYPES = {
    "cf32_le": "<c8", "cf64_le": "<c16",
    "rf32_le": "<f4", "rf64_le": "<f8",
    "ri8_le": "i1", "ri8": "i1", "ru8_le": "u1", "ru8": "u1",
    "ri16_le": "<i2", "ru16_le": "<u2", "ri32_le": "<i4",
    "ci16_le": "ci16",  # handled specially
}
QAM_ORDER = {"pi/2-bpsk": 1, "bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6, "256qam": 8,
             "16-qam": 4, "64-qam": 6, "256-qam": 8}

_FNAME = re.compile(r"^(?P<stream>[a-z\-]+?)-rec-(?P<rec>\d+)-(?P<ts>.+)\.sigmf-meta$")


# --------------------------------------------------------------------------- helpers
def resolve_run(path: str) -> str:
    """Accept a run folder or a .zip of one; return a folder containing .sigmf-meta files."""
    if os.path.isfile(path) and path.endswith(".zip"):
        out = os.path.join(tempfile.gettempdir(), "sigmf_runs",
                           os.path.splitext(os.path.basename(path))[0])
        if not glob.glob(os.path.join(out, "**", "*.sigmf-meta"), recursive=True):
            os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(path) as z:
                for m in z.namelist():  # refuse path traversal
                    if m.startswith("/") or ".." in m.split("/"):
                        raise ValueError(f"unsafe path in zip: {m}")
                z.extractall(out)
        path = out
    metas = glob.glob(os.path.join(path, "**", "*.sigmf-meta"), recursive=True)
    if not metas:
        raise FileNotFoundError(f"no .sigmf-meta files under {path}")
    # the folder that actually holds the files
    return os.path.dirname(sorted(metas)[0])


def run_name(path: str) -> str:
    return os.path.basename(os.path.normpath(resolve_run(path)))


def _read_data(meta: dict, data_path: str) -> np.ndarray:
    dt = meta["global"]["core:datatype"]
    if dt not in _DTYPES:
        raise ValueError(f"unsupported SigMF datatype {dt}")
    if dt == "ci16_le":
        raw = np.fromfile(data_path, dtype="<i2").astype(np.float32)
        return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    return np.fromfile(data_path, dtype=_DTYPES[dt])


def _shape_grid(x: np.ndarray, capture: dict) -> np.ndarray:
    counts = capture.get("serialization:counts")
    if counts and int(np.prod(counts)) == x.size:
        return x.reshape(counts, order="F")  # symbol-major storage (fact 1)
    return x


def _fix_bits(b: np.ndarray, bit_order: str) -> np.ndarray:
    b = b.astype(np.uint8).ravel()
    if bit_order == "lsb":
        if b.size % 8:
            warnings.warn(f"bit stream length {b.size} not a multiple of 8; left as stored")
            return b
        return b.reshape(-1, 8)[:, ::-1].ravel()
    return b


def _nr_detail(meta: dict) -> dict:
    try:
        ann = meta["annotations"][0]
        return ann["system_components:transmitter"][0]["signal:detail"]["5gnr"]
    except (KeyError, IndexError):
        return {}


def _annotation(meta: dict) -> dict:
    return (meta.get("annotations") or [{}])[0]


def unpack_dmrs_symbols(stored: np.ndarray, dmrs_mask: np.ndarray) -> np.ndarray:
    """Move packed data REs of DMRS symbols back to their true subcarriers (fact 2).

    stored    : (n_sc, n_sym) grid as recorded
    dmrs_mask : (n_sc, n_sym) bool, True on DMRS REs
    """
    out = stored.copy()
    for s in np.where(dmrs_mask.any(axis=0))[0]:
        data_sc = np.where(~dmrs_mask[:, s])[0]
        col = stored[:, s]
        n = data_sc.size
        if np.any(col[n:] != 0):
            # not packed after all -> leave as is
            continue
        out[:, s] = 0
        out[data_sc, s] = col[:n]
    return out


# --------------------------------------------------------------------------- main API
def load_record_files(folder: str) -> dict[int, dict[str, str]]:
    """Map record number -> {stream prefix: path to .sigmf-meta}."""
    recs: dict[int, dict[str, str]] = {}
    for p in glob.glob(os.path.join(folder, "*.sigmf-meta")):
        m = _FNAME.match(os.path.basename(p))
        if not m or m["stream"] not in STREAMS:
            continue
        recs.setdefault(int(m["rec"]), {})[m["stream"]] = p
    return recs


def load_run(folder: str, bit_order: str = "lsb", verify_hash: bool = True,
             unpack_dmrs: bool = True) -> list[dict]:
    """Load every record of a run. Returns one dict per record, sorted by record number."""
    folder = resolve_run(folder)
    files = load_record_files(folder)
    out = []
    for rec in sorted(files):
        out.append(_load_record(rec, files[rec], bit_order, verify_hash, unpack_dmrs))
    return out


def _load_record(rec: int, streams: dict[str, str], bit_order: str,
                 verify_hash: bool, unpack_dmrs: bool) -> dict:
    r: dict = {"record": rec, "missing_streams": [], "warnings": [], "hash_ok": True}
    ids: dict[str, tuple] = {}
    metas: dict[str, dict] = {}

    for prefix, key in STREAMS.items():
        r[key] = None
        mp = streams.get(prefix)
        dp = mp[:-len(".sigmf-meta")] + ".sigmf-data" if mp else None
        if mp is None or not os.path.exists(dp):
            r["missing_streams"].append(prefix)
            continue
        try:
            with open(mp) as f:
                meta = json.load(f)
            x = _read_data(meta, dp)
        except Exception as e:  # corrupt stream: mark, don't crash
            r["missing_streams"].append(prefix)
            r["warnings"].append(f"{prefix}: unreadable ({e})")
            continue
        metas[prefix] = meta
        sha = meta["global"].get("core:sha512")
        if verify_hash and sha:
            with open(dp, "rb") as f:
                if hashlib.sha512(f.read()).hexdigest() != sha:
                    r["hash_ok"] = False
                    r["warnings"].append(f"{prefix}: sha512 mismatch")
        cap = (meta.get("captures") or [{}])[0]
        n_exp = (_annotation(meta).get("core:sample_count")
                 or (int(np.prod(cap["serialization:counts"])) if prefix in IQ_STREAMS
                     and cap.get("serialization:counts") else None))
        if n_exp is not None and x.size != n_exp:
            r["missing_streams"].append(prefix)
            r["warnings"].append(f"{prefix}: {x.size} samples on disk, metadata says {n_exp}")
            continue
        if prefix in BIT_STREAMS:
            r[key] = _fix_bits(x, bit_order)
        else:
            r[key] = _shape_grid(x.astype(np.complex64), cap)
        d = _nr_detail(meta)
        ids[prefix] = (d.get("frame"), d.get("slot"), d.get("rnti"))
        r.setdefault("timestamps", {})[prefix] = cap.get("core:datetime")

    r["complete"] = not r["missing_streams"]

    # ---- frame / slot / rnti consistency across all streams
    uniq = set(ids.values())
    r["id_mismatch"] = len(uniq) > 1
    if r["id_mismatch"]:
        msg = f"record {rec}: frame/slot/rnti disagree across streams: {ids}"
        r["warnings"].append(msg)
        warnings.warn(msg)
    r["stream_ids"] = ids

    # ---- metadata (prefer a gNB stream, else any)
    ref = next((p for p in GNB_STREAMS + UE_STREAMS if p in metas), None)
    if ref is None:
        return r
    meta = metas[ref]
    d = _nr_detail(meta)
    ann = _annotation(meta)
    chan = (ann.get("system_components:channel") or [{}])[0]
    rx = ann.get("system_components:receiver") or {}
    cap = (meta.get("captures") or [{}])[0]
    mod = str(d.get("pusch:modulation", "")).lower()
    r.update(
        frame=d.get("frame"), slot=d.get("slot"), rnti=d.get("rnti"),
        mcs_index=d.get("pusch:mcs"), mcs_table=d.get("pusch:mcs_table_index"),
        modulation=mod, qam_mod_order=QAM_ORDER.get(mod),
        rb_size=d.get("pusch:num_prb"), rb_start=d.get("pusch:start_prb"),
        nr_of_symbols=d.get("pusch:num_ofdm_symbols"),
        start_symbol=d.get("pusch:start_ofdm_symbol"),
        ul_dmrs_symb_pos=list(d.get("pusch_dmrs:ofdm_symbol_idx", [])),
        dmrs_config=d.get("pusch_dmrs:resource_map_config"),
        dmrs_add_pos=d.get("pusch_dmrs:num_add_positions"),
        num_layers=d.get("pusch:num_layer"),
        num_rx_antennas=rx.get("num_rx_antennas"),
        receiver_type=rx.get("phy_freq_domain_receiver_type"),
        scs_hz=d.get("subcarrier_spacing"),
        carrier_hz=cap.get("core:frequency"),
        sample_rate=meta["global"].get("core:sample_rate"),
        timestamp=cap.get("core:datetime"),
        # channel-emulator *configuration* fields (not measurements!)
        cfg_channel_model=chan.get("channel_model"),
        cfg_snr_db=chan.get("snr_esn0_db"),
        cfg_speed_mps=chan.get("speed"),
        cfg_delay_spread_s=chan.get("delay_spread"),
        payload_bit_pattern=d.get("pusch:payload_bit_pattern"),
    )
    if "tx-payload-bits" in metas or "rx-payload-bits" in metas:
        m = metas.get("tx-payload-bits") or metas["rx-payload-bits"]
        r["number_of_bits"] = _annotation(m).get("core:sample_count")
    if "tx-scrambled-bits" in metas:
        r["number_of_coded_bits"] = _annotation(metas["tx-scrambled-bits"]).get("core:sample_count")

    # ---- resource-element masks and DMRS unpacking
    if r["pilots"] is not None:
        r["pilot_mask"] = np.abs(r["pilots"]) > 0
    elif r["ce_pilots"] is not None:
        r["pilot_mask"] = np.abs(r["ce_pilots"]) > 0
    else:
        r["pilot_mask"] = None
    r["data_mask"] = None
    pm = r["pilot_mask"]
    for key in ("rx_grid", "ce_interp"):
        g = r[key]
        r[key + "_stored"] = g
        if g is not None and g.ndim != 2:
            r[key] = None
            continue
        if g is not None and pm is not None and g.shape == pm.shape and unpack_dmrs:
            r[key] = unpack_dmrs_symbols(g, pm)
    if r["rx_grid"] is not None and pm is not None and r["rx_grid"].shape == pm.shape:
        n_sym = r.get("nr_of_symbols") or r["rx_grid"].shape[1]
        s0 = r.get("start_symbol") or 0
        mask = np.zeros_like(pm)
        mask[:, s0:s0 + n_sym] = True
        r["data_mask"] = mask & ~pm  # all allocated REs that are not DMRS
        nz = np.abs(r["rx_grid"]) > 0
        if (nz != r["data_mask"]).any():
            r["warnings"].append("rx_grid non-zero pattern differs from expected data-RE mask")
    return r


def has_streams(r: dict, keys=("rx_grid", "ce_pilots", "tx_coded")) -> bool:
    """True if the record has the given arrays (default: what training/analysis needs)."""
    return all(r.get(k) is not None for k in keys) and r.get("data_mask") is not None


def has_gnb_interp(r: dict) -> bool:
    """Classical-path interpolated CE present and non-zero (absent under the AI-native RX)."""
    h = r.get("ce_interp")
    return h is not None and bool(np.any(h))


def records_to_frame(records: list[dict]):
    """Flat per-record metadata table (pandas DataFrame)."""
    import pandas as pd
    cols = ["record", "frame", "slot", "rnti", "timestamp", "mcs_index", "modulation",
            "qam_mod_order", "rb_size", "rb_start", "nr_of_symbols", "number_of_bits",
            "number_of_coded_bits", "num_rx_antennas", "receiver_type", "complete",
            "id_mismatch", "hash_ok"]
    rows = []
    for r in records:
        row = {c: r.get(c) for c in cols}
        row["ul_dmrs_symb_pos"] = ",".join(map(str, r.get("ul_dmrs_symb_pos") or []))
        row["missing_streams"] = ";".join(r["missing_streams"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(r: dict) -> str:
    lines = [f"record {r['record']}  complete={r['complete']}  missing={r['missing_streams']}"]
    for k in ["rx_grid", "ce_pilots", "ce_interp", "pilots", "tx_bits", "rx_bits", "tx_coded",
              "data_mask", "pilot_mask"]:
        v = r.get(k)
        lines.append(f"  {k:<11} " + ("None" if v is None else f"{str(v.shape):<10} {v.dtype}"))
    for k in ["frame", "slot", "rnti", "timestamp", "mcs_index", "modulation", "qam_mod_order",
              "rb_size", "rb_start", "nr_of_symbols", "start_symbol", "ul_dmrs_symb_pos",
              "dmrs_config", "num_rx_antennas", "number_of_bits", "number_of_coded_bits",
              "receiver_type", "cfg_channel_model", "cfg_snr_db", "cfg_speed_mps",
              "cfg_delay_spread_s"]:
        lines.append(f"  {k:<20} {r.get(k)}")
    if r["warnings"]:
        lines.append("  warnings: " + "; ".join(r["warnings"]))
    return "\n".join(lines)
