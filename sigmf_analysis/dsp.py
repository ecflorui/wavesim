"""Shared signal-processing helpers: QAM mapping, equalization, SNR, channel features.

All functions work on one record dict produced by ``sigmf_loader.load_run``.
Grids are (n_subcarriers, n_symbols).  Bit order inside the coded-bit stream is
3GPP order: OFDM symbol by symbol, subcarrier ascending, data REs only.
"""
from __future__ import annotations

import numpy as np

NR_CP_FACTOR = 1 + 144 / 2048  # normal-CP symbol length relative to useful symbol (non-first)


# --------------------------------------------------------------------------- mapping
def qam_map(bits: np.ndarray, qm: int) -> np.ndarray:
    """3GPP TS 38.211 sec. 5.1 modulation mapper (unit average energy), qm in {2,4,6,8}."""
    s = 1 - 2 * np.asarray(bits, dtype=np.float64).reshape(-1, qm)  # bit 0 -> +1
    if qm == 2:
        return (s[:, 0] + 1j * s[:, 1]) / np.sqrt(2)
    norm = {4: np.sqrt(10), 6: np.sqrt(42), 8: np.sqrt(170)}[qm]
    return (_gray_pam(s[:, 0::2]) + 1j * _gray_pam(s[:, 1::2])) / norm


def _gray_pam(s: np.ndarray) -> np.ndarray:
    """Per-axis amplitude, e.g. 64QAM: s0 * (4 - s2 * (2 - s4))."""
    k = s.shape[1]
    inner = 2 - s[:, k - 1]
    for i in range(k - 2, 0, -1):
        inner = 2 ** (k - i) - s[:, i] * inner
    return s[:, 0] * inner


def qpsk_llr(z: np.ndarray, noise_var: np.ndarray | float) -> np.ndarray:
    """Max-log = exact LLRs for Gray QPSK, sign convention LLR>0 <=> bit 0.

    z: equalized symbols (unit-energy constellation), noise_var: post-eq noise variance.
    Returns (N*2,) interleaved [b0, b1, b0, b1, ...].
    """
    l0 = 2 * np.sqrt(2) * z.real / noise_var
    l1 = 2 * np.sqrt(2) * z.imag / noise_var
    return np.stack([l0, l1], axis=1).ravel()


def hard_bits(llr: np.ndarray) -> np.ndarray:
    return (llr < 0).astype(np.uint8)


# --------------------------------------------------------------------------- RE helpers
def data_res(grid: np.ndarray, data_mask: np.ndarray) -> np.ndarray:
    """Data REs in 3GPP order (symbol by symbol, subcarrier ascending)."""
    return grid.T[data_mask.T]


def tx_symbols(r: dict) -> np.ndarray:
    """Known transmitted data symbols reconstructed from tx_coded (label-aided)."""
    return qam_map(r["tx_coded"], r["qam_mod_order"])


def to_grid(values: np.ndarray, mask: np.ndarray, dtype=np.complex64) -> np.ndarray:
    g = np.zeros(mask.shape, dtype=dtype)
    g.T[mask.T] = values
    return g


# --------------------------------------------------------------------------- equalization
def equalize(r: dict, h: np.ndarray | None = None):
    """ZF-equalize data REs with channel grid h (default: gNB interpolated CE).

    The gNB works in fixed point, so y = c * h * x with an unknown common scale c.
    We return z = y/h normalised so the constellation has unit average power,
    plus the complex scale used.
    """
    h = r["ce_interp"] if h is None else h
    m = r["data_mask"]
    y, hh = data_res(r["rx_grid"], m), data_res(h, m)
    z = y / hh
    scale = np.sqrt(np.mean(np.abs(z) ** 2))
    return z / scale, scale


# --------------------------------------------------------------------------- SNR
def snr_post_eq(r: dict, h: np.ndarray | None = None, aided: bool = True) -> float:
    """Post-equalization SNR (dB) = 1 / EVM^2.

    aided=True : reference = true tx symbols (needs tx_coded label)
    aided=False: reference = hard decisions (blind, decision-directed)
    Includes channel-estimation error, so it is what the demapper actually sees.
    """
    z, _ = equalize(r, h)
    if aided:
        x = tx_symbols(r)
    else:
        x = (np.sign(z.real) + 1j * np.sign(z.imag)) / np.sqrt(2)  # QPSK only
    a = np.vdot(x, z) / np.vdot(x, x)  # residual complex gain
    e = z - a * x
    return float(10 * np.log10(np.abs(a) ** 2 / np.mean(np.abs(e) ** 2)))


def _smooth_fit(k: np.ndarray, v: np.ndarray, n_sc: int, deg: int | None = None):
    """Smooth H(f) on subcarriers k: remove the linear phase (timing offset), then fit a
    low-order complex polynomial.  Returns (fitted values, number of free parameters).

    Integer-bin DFT truncation is NOT used: a fractional timing offset leaks into
    every delay bin and gets mistaken for noise (verified on this data).
    """
    if deg is None:
        deg = max(2, n_sc // 24)  # ~one degree per 2 PRB; 3 for 6 PRB
    dk = np.diff(k)
    c = np.sum(np.conj(v[:-1]) * v[1:] * (dk == dk.min()))
    slope = np.angle(c) / dk.min()  # rad per subcarrier
    rot = np.exp(1j * slope * k)
    u = (k - k.mean()) / max(1, np.ptp(k))
    V = np.vander(u, deg + 1)
    coef, *_ = np.linalg.lstsq(V, v / rot, rcond=None)
    return (V @ coef) * rot, deg + 1 + 0.5  # +0.5: the slope is one extra real parameter


def snr_data_aided(r: dict, deg: int | None = None) -> dict:
    """Channel SNR (dB) independent of the gNB channel estimator.

    Uses the known tx symbols: h_hat = y / x on every data RE, smooths h_hat
    across frequency in each OFDM symbol (``_smooth_fit``), and treats the
    residual y - h_smooth * x as noise (corrected for the fitted degrees of freedom).
    """
    m = r["data_mask"]
    x = tx_symbols(r)
    y = data_res(r["rx_grid"], m)
    hraw = to_grid(y / x, m)
    hs = np.zeros_like(hraw)
    dofs = []
    for s in range(m.shape[1]):
        k = np.where(m[:, s])[0]
        if k.size < 8:
            continue
        fit, p = _smooth_fit(k, hraw[k, s], m.shape[0], deg)
        hs[k, s] = fit
        dofs.append(1 - p / k.size)
    hsd = data_res(hs, m)
    e = y - hsd * x
    nv = np.mean(np.abs(e) ** 2) / np.mean(dofs)
    sp = np.mean(np.abs(hsd * x) ** 2)
    return {"snr_db": float(10 * np.log10(sp / nv)), "noise_var": float(nv),
            "signal_power": float(sp), "h_smooth": hs}


def snr_pilot(r: dict, deg: int | None = None) -> float:
    """Pilot-domain SNR (dB): power of the raw CE at DMRS vs. its residual after smoothing.

    Caveat: OAI's raw CE already averages pairs of adjacent DMRS (each value is
    repeated on 2 subcarriers), so its noise is ~3 dB lower than per-RE noise.
    Only the unique values are used, and 3 dB is subtracted to refer the result
    to a per-RE SNR comparable with ``snr_data_aided``.
    """
    ce, pm = r["ce_pilots"], r["pilot_mask"]
    sig, noi = [], []
    for s in np.where(pm.any(axis=0))[0]:
        k = np.where(pm[:, s])[0]
        v = ce[k, s]
        keep = np.r_[True, np.abs(np.diff(v)) > 1e-6]
        pair = (~keep).sum() > 0.4 * keep.size  # values held over pairs
        k, v = k[keep], v[keep]
        fit, p = _smooth_fit(k, v, ce.shape[0], deg)
        sig.append(np.mean(np.abs(fit) ** 2))
        noi.append(np.mean(np.abs(v - fit) ** 2) / (1 - p / k.size) * (2 if pair else 1))
    return float(10 * np.log10(np.mean(sig) / np.mean(noi)))


# --------------------------------------------------------------------------- channel features
def symbol_duration(scs_hz: float) -> float:
    return NR_CP_FACTOR / scs_hz


def time_correlation(r: dict) -> dict:
    """Correlation of the raw pilot CE between DMRS symbols (0 vs 5, 5 vs 10, ...).

    |rho| -> time selectivity (Doppler);  angle(rho) -> common phase drift (CFO).
    Also returns the noise-corrected |rho| using the pilot SNR.
    """
    ce, pm = r["ce_pilots"], r["pilot_mask"]
    syms = list(np.where(pm.any(axis=0))[0])
    out = {}
    for a, b in zip(syms[:-1], syms[1:]):
        k = np.where(pm[:, a] & pm[:, b])[0]
        ha, hb = ce[k, a], ce[k, b]
        rho = np.vdot(ha, hb) / np.sqrt(np.vdot(ha, ha).real * np.vdot(hb, hb).real)
        out[(a, b)] = rho
    return out


def bessel_j0(x):
    """J0 by power series (accurate for |x| < ~10; we only need x < 2.405)."""
    x = np.asarray(x, dtype=float)
    term, total = np.ones_like(x), np.ones_like(x)
    for k in range(1, 30):
        term = term * -(x / 2) ** 2 / k ** 2
        total = total + term
    return total


def doppler_from_corr(rho_abs: float, dt: float) -> float:
    """Invert the Jakes model rho = J0(2 pi f_D dt) for f_D (Hz). 0 if rho >= 1."""
    if rho_abs >= 1:
        return 0.0
    if not rho_abs > 0:
        return float("nan")
    lo, hi = 0.0, 2.4048 / (2 * np.pi * dt)  # J0 decreasing on [0, first zero]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bessel_j0(2 * np.pi * mid * dt) > rho_abs:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def power_delay_profile(h_f: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    """|IFFT|^2 of H(f) (columns = OFDM symbols, averaged). Zero-padded for display."""
    h_f = np.atleast_2d(h_f.T).T
    p = np.abs(np.fft.ifft(h_f, n=n_fft, axis=0)) ** 2
    return p.mean(axis=1)


def rms_delay_spread(h_f: np.ndarray, scs_hz: float, thresh_db: float = 20.0,
                     n_fft: int = 1024) -> dict:
    """RMS delay spread from the PDP, ignoring taps more than thresh_db below the peak.

    Resolution is 1/(N_sc * scs); anything below that is not physically resolved.
    """
    n_sc = h_f.shape[0]
    pdp = power_delay_profile(h_f, n_fft)
    pk = np.argmax(pdp)
    pdp = np.roll(pdp, n_fft // 2 - pk)  # centre the peak (removes timing offset)
    tau = (np.arange(n_fft) - n_fft // 2) / (n_fft * scs_hz)
    keep = pdp >= pdp.max() * 10 ** (-thresh_db / 10)
    # only the contiguous lobe region around the peak
    p, t = pdp * keep, tau
    mu = np.sum(p * t) / np.sum(p)
    rms = np.sqrt(np.sum(p * (t - mu) ** 2) / np.sum(p))
    return {"rms_s": float(rms), "resolution_s": 1 / (n_sc * scs_hz), "pdp": pdp, "tau": tau}


def freq_correlation(h_f: np.ndarray, lag: int = 1) -> complex:
    """Normalised correlation between subcarriers k and k+lag, averaged over symbols."""
    a, b = h_f[:-lag], h_f[lag:]
    return np.sum(np.conj(a) * b) / np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))


def full_symbols(r: dict) -> np.ndarray:
    """Indices of OFDM symbols where ce_interp covers every subcarrier."""
    return np.where(r["data_mask"].all(axis=0))[0]
