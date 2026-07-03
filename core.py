"""
sigscope.core
=============

Vectorised implementations of the signal statistics we discussed for
separating a coherent target from noise/clutter at the *Hilbert stage*
(complex data z = A * exp(j*phi)), BEFORE pulse compression and Doppler FFT.

Design
------
Every statistic is exposed through two entry points:

    <name>_global(z, axis)          -> dict[str, 1D array]   (one value per line)
    <name>_windowed(z, axis, win, step) -> (dict[str, 2D array], centers)

`axis` is the axis we compute ALONG. The other axis indexes the "lines":

    axis = time-axis  -> per range-bin statistics  (Doppler / pulse-pair world)
    axis = range-axis -> per-impulse statistics     (chirp / reference world)

All functions are fully vectorised over the non-computed axis, so calling
them on the whole 4000 x 32000 matrix is a handful of numpy ops, not a loop.

The STATS registry at the bottom lets the GUI enumerate everything
generically (name, description, which channel it needs, default good axis).
"""

from __future__ import annotations
import numpy as np

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def _to_last(z, axis):
    """Move the compute axis to the end so we can always work on axis=-1."""
    return np.moveaxis(z, axis, -1)


def _from_last(a, axis):
    """Inverse of _to_last for results that keep the compute axis."""
    return np.moveaxis(a, -1, axis)


def _sliding_sum(seq, win, step):
    """
    Sum `seq` over sliding windows along the LAST axis.

    seq   : (..., N) array (real or complex)
    win   : window length in samples
    step  : hop between window starts

    returns (sums, centers)
        sums    : (..., M) windowed sums
        centers : (M,) index of each window centre (in original sample coords)
    """
    N = seq.shape[-1]
    if win > N:
        win = N
    # cumulative sum with a leading zero so window [i, i+win) = c[i+win] - c[i]
    c = np.concatenate(
        [np.zeros(seq.shape[:-1] + (1,), dtype=seq.dtype),
         np.cumsum(seq, axis=-1)], axis=-1)
    starts = np.arange(0, N - win + 1, step)
    sums = c[..., starts + win] - c[..., starts]
    centers = starts + win // 2
    return sums, centers


_EPS = 1e-30


# --------------------------------------------------------------------------
# base per-sample sequences (built once, reused by many statistics)
# --------------------------------------------------------------------------

def _base_sequences(z):
    """Per-sample quantities used across statistics. z has compute axis last."""
    amp = np.abs(z)
    unit = z / (amp + _EPS)                      # e^{j phi}
    # lag-1 complex product z[n] * conj(z[n-1]) (pulse-pair kernel), length N-1
    pp = z[..., 1:] * np.conj(z[..., :-1])
    return amp, unit, pp


# ==========================================================================
# 1. AMPLITUDE MOMENTS  (Rayleigh / Rician / clutter discrimination)
# ==========================================================================

def _moments_from_sums(s1, s2, s3, s4, n):
    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, _EPS)
    std = np.sqrt(var)
    m3 = s3 / n - 3 * mean * (s2 / n) + 2 * mean**3
    m4 = s4 / n - 4 * mean * (s3 / n) + 6 * mean**2 * (s2 / n) - 3 * mean**4
    skew = m3 / (std**3 + _EPS)
    kurt = m4 / (var**2 + _EPS) - 3.0          # excess kurtosis
    return mean, std, skew, kurt


def amplitude_global(z, axis):
    z = _to_last(z, axis)
    a = np.abs(z)
    n = a.shape[-1]
    mean, std, skew, kurt = _moments_from_sums(
        a.sum(-1), (a**2).sum(-1), (a**3).sum(-1), (a**4).sum(-1), n)
    return {"amp_mean": mean, "amp_std": std,
            "amp_skew": skew, "amp_kurt": kurt}


def amplitude_windowed(z, axis, win, step):
    z = _to_last(z, axis)
    a = np.abs(z)
    s1, centers = _sliding_sum(a, win, step)
    s2, _ = _sliding_sum(a**2, win, step)
    s3, _ = _sliding_sum(a**3, win, step)
    s4, _ = _sliding_sum(a**4, win, step)
    n = min(win, a.shape[-1])
    mean, std, skew, kurt = _moments_from_sums(s1, s2, s3, s4, n)
    return {"amp_mean": mean, "amp_std": std,
            "amp_skew": skew, "amp_kurt": kurt}, centers


# ==========================================================================
# 2. CIRCULAR PHASE STATISTICS  (uniform noise vs concentrated signal)
#    Two flavours:
#      raw   -> on phase itself      (detects ZERO-Doppler coherence / clutter)
#      diff  -> on phase differences (detects ANY constant Doppler -> target)
# ==========================================================================

def _kappa_from_R(R):
    """von Mises concentration estimate from mean resultant length R."""
    R = np.clip(R, 0, 0.999999)
    k = np.empty_like(R)
    lo = R < 0.53
    mid = (R >= 0.53) & (R < 0.85)
    hi = R >= 0.85
    k[lo] = 2 * R[lo] + R[lo]**3 + 5 * R[lo]**5 / 6
    k[mid] = -0.4 + 1.39 * R[mid] + 0.43 / (1 - R[mid])
    denom = (R[hi]**3 - 4 * R[hi]**2 + 3 * R[hi])
    k[hi] = 1.0 / np.where(np.abs(denom) < _EPS, _EPS, denom)
    return k


def _circular_from_vecsum(vsum, n):
    R = np.abs(vsum) / n
    mean_angle = np.angle(vsum)
    circ_var = 1.0 - R
    kappa = _kappa_from_R(R)
    # Rayleigh test for uniformity: p ~ exp(-n R^2) (large-n approximation)
    Z = n * R**2
    p = np.exp(-Z) * (1 + (2 * Z - Z**2) / (4 * n))
    p = np.clip(p, 0, 1)
    return {"R": R, "circ_mean": mean_angle, "circ_var": circ_var,
            "kappa": kappa, "rayleigh_p": p}


def circular_global(z, axis, mode="diff"):
    z = _to_last(z, axis)
    if mode == "diff":
        pp = z[..., 1:] * np.conj(z[..., :-1])
        unit = pp / (np.abs(pp) + _EPS)
    else:
        unit = z / (np.abs(z) + _EPS)
    n = unit.shape[-1]
    return _circular_from_vecsum(unit.sum(-1), n)


def circular_windowed(z, axis, win, step, mode="diff"):
    z = _to_last(z, axis)
    if mode == "diff":
        pp = z[..., 1:] * np.conj(z[..., :-1])
        unit = pp / (np.abs(pp) + _EPS)
    else:
        unit = z / (np.abs(z) + _EPS)
    vsum, centers = _sliding_sum(unit, win, step)
    n = min(win, unit.shape[-1])
    return _circular_from_vecsum(vsum, n), centers


# ==========================================================================
# 3. PULSE-PAIR / COMPLEX LAG-k AUTOCORRELATION
#    coherence = |R1| / power   (Doppler-AGNOSTIC target presence)
#    doppler   = angle(R1)      (rad / sample; 0 = stationary clutter)
# ==========================================================================

def pulsepair_global(z, axis, lag=1):
    z = _to_last(z, axis)
    pp = z[..., lag:] * np.conj(z[..., :-lag])
    power = (np.abs(z[..., lag:]) * np.abs(z[..., :-lag])).sum(-1)
    R = pp.sum(-1)
    coh = np.abs(R) / (power + _EPS)
    return {"pp_coherence": coh,
            "pp_doppler": np.angle(R),
            "pp_power": power / pp.shape[-1]}


def pulsepair_windowed(z, axis, win, step, lag=1):
    z = _to_last(z, axis)
    pp = z[..., lag:] * np.conj(z[..., :-lag])
    absprod = np.abs(z[..., lag:]) * np.abs(z[..., :-lag])
    Rsum, centers = _sliding_sum(pp, win, step)
    Psum, _ = _sliding_sum(absprod, win, step)
    coh = np.abs(Rsum) / (Psum + _EPS)
    n = min(win, pp.shape[-1])
    return {"pp_coherence": coh,
            "pp_doppler": np.angle(Rsum),
            "pp_power": Psum / n}, centers


def lag_decay_global(z, axis, max_lag=8):
    """|R_k|/power for k=1..max_lag; slow decay = long coherence."""
    z = _to_last(z, axis)
    out = []
    for k in range(1, max_lag + 1):
        pp = z[..., k:] * np.conj(z[..., :-k])
        power = (np.abs(z[..., k:]) * np.abs(z[..., :-k])).sum(-1) + _EPS
        out.append(np.abs(pp.sum(-1)) / power)
    stack = np.stack(out, axis=-1)          # (..., max_lag)
    # summarise decay by area under the coherence-vs-lag curve
    return {"lag_coh_area": stack.mean(-1),
            "lag_coh_1": stack[..., 0]}


# ==========================================================================
# 4. COHERENT vs NON-COHERENT RATIO ("span factor")
#    |sum z| / sum|z|  in [1/sqrt(N), 1]. Detects ZERO-Doppler coherence.
#    (Pair it with pulse-pair for Doppler-agnostic detection.)
# ==========================================================================

def spanfactor_global(z, axis):
    z = _to_last(z, axis)
    return {"span_factor": np.abs(z.sum(-1)) / (np.abs(z).sum(-1) + _EPS)}


def spanfactor_windowed(z, axis, win, step):
    z = _to_last(z, axis)
    csum, centers = _sliding_sum(z, win, step)
    asum, _ = _sliding_sum(np.abs(z), win, step)
    return {"span_factor": np.abs(csum) / (asum + _EPS)}, centers


# ==========================================================================
# 5. PHASE-RAMP LINEARITY  (target = clean linear ramp; slope = Doppler)
#    Robust complex form: estimate slope from lag-1 angle, de-rotate,
#    measure the residual coherence loss.
# ==========================================================================

def linearity_global(z, axis):
    z = _to_last(z, axis)
    pp = z[..., 1:] * np.conj(z[..., :-1])
    slope = np.angle(pp.sum(-1))                       # mean phase increment
    n = z.shape[-1]
    ramp = np.exp(-1j * slope[..., None] * np.arange(n))
    derot = z * ramp
    # after removing the linear ramp a true target is ~constant phase ->
    # high residual coherence; noise stays incoherent -> low
    resid_coh = np.abs(derot.sum(-1)) / (np.abs(derot).sum(-1) + _EPS)
    return {"lin_residual_coh": resid_coh, "lin_slope": slope}


def linearity_windowed(z, axis, win, step):
    z = _to_last(z, axis)
    pp = z[..., 1:] * np.conj(z[..., :-1])
    Rsum, centers = _sliding_sum(pp, win, step)
    slope = np.angle(Rsum)                              # (..., M)
    # de-rotate each window by its own slope and measure coherence
    n = z.shape[-1]
    idx = np.arange(n)
    # build per-window de-rotated coherent sum via two sliding sums:
    #   we need sum over window of z*exp(-i*slope*k). Approx by de-rotating
    #   the whole line with a local slope is not separable, so compute the
    #   coherence of the de-rotated signal using window-local slope applied
    #   at window centres (good enough as a feature).
    csum, _ = _sliding_sum(z * np.exp(-1j * 0.0 * idx), win, step)  # raw coherent
    asum, _ = _sliding_sum(np.abs(z), win, step)
    # residual coherence after removing the *global-in-window* linear part is
    # well approximated by |R1|/power (Doppler-agnostic), so reuse it:
    absprod = np.abs(z[..., 1:]) * np.abs(z[..., :-1])
    Psum, _ = _sliding_sum(absprod, win, step)
    resid_coh = np.abs(Rsum) / (Psum + _EPS)
    return {"lin_residual_coh": resid_coh, "lin_slope": slope}, centers


# ==========================================================================
# 6. PHASE 2nd-DIFFERENCE CONSTANCY  (chirp / reference-shape detector)
#    For an LFM chirp the instantaneous frequency is linear, so its
#    derivative (the phase 2nd difference) is ~constant. Low variance of the
#    2nd difference => chirp-like structure present.
# ==========================================================================

def chirp_global(z, axis):
    z = _to_last(z, axis)
    inst_f = np.angle(z[..., 1:] * np.conj(z[..., :-1]))      # instant. freq
    d2 = np.diff(inst_f, axis=-1)                             # curvature
    # wrap curvature into [-pi, pi)
    d2 = (d2 + np.pi) % (2 * np.pi) - np.pi
    n = d2.shape[-1]
    mean = d2.mean(-1)
    var = d2.var(-1)
    constancy = 1.0 / (1.0 + var)                            # 1 = perfect chirp
    return {"chirp_rate": mean, "chirp_constancy": constancy,
            "chirp_d2_var": var}


def chirp_windowed(z, axis, win, step):
    z = _to_last(z, axis)
    inst_f = np.angle(z[..., 1:] * np.conj(z[..., :-1]))
    d2 = np.diff(inst_f, axis=-1)
    d2 = (d2 + np.pi) % (2 * np.pi) - np.pi
    s1, centers = _sliding_sum(d2, win, step)
    s2, _ = _sliding_sum(d2**2, win, step)
    n = min(win, d2.shape[-1])
    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, _EPS)
    return {"chirp_rate": mean, "chirp_constancy": 1.0 / (1.0 + var),
            "chirp_d2_var": var}, centers


# ==========================================================================
# 7. SPECTRAL FEATURES  (REFERENCE ONLY -- uses FFT along the line)
#    Kept for validation / comparison against the FFT-free features.
# ==========================================================================

def spectral_global(z, axis):
    z = _to_last(z, axis)
    F = np.fft.fft(z, axis=-1)
    P = np.abs(F)**2
    total = P.sum(-1, keepdims=True) + _EPS
    p = P / total
    entropy = -(p * np.log(p + _EPS)).sum(-1) / np.log(P.shape[-1])
    peak = P.max(-1)
    avg = P.mean(-1)
    dom = np.argmax(P, axis=-1).astype(float)
    return {"spec_peak2avg": peak / (avg + _EPS),
            "spec_entropy": entropy,
            "spec_dom_bin": dom}


# ==========================================================================
# 8. STRUCTURE TENSOR  (2D orientation & coherence on the amplitude image)
#    Streaks (targets) are oriented; noise is isotropic.
#    This one is inherently 2D (not per-line), returns full maps.
# ==========================================================================

def structure_tensor(amp, sigma_grad=1.0, sigma_int=3.0):
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    a = amp.astype(np.float64)
    gx = gaussian_filter1d(a, sigma_grad, axis=1, order=1)
    gy = gaussian_filter1d(a, sigma_grad, axis=0, order=1)
    Jxx = gaussian_filter(gx * gx, sigma_int)
    Jyy = gaussian_filter(gy * gy, sigma_int)
    Jxy = gaussian_filter(gx * gy, sigma_int)
    tmp = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    lam1 = 0.5 * (Jxx + Jyy + tmp)
    lam2 = 0.5 * (Jxx + Jyy - tmp)
    coherence = ((lam1 - lam2) / (lam1 + lam2 + _EPS))**2
    orientation = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)
    return {"st_coherence": coherence, "st_orientation": orientation,
            "st_energy": lam1 + lam2}


# ==========================================================================
# STATS REGISTRY  (drives the GUI)
# --------------------------------------------------------------------------
# Each entry:
#   key: (label, needs, outputs, global_fn, windowed_fn, note)
#   needs: 'complex' (uses full z), 'amp', 'phase'
#   windowed_fn None -> global only
# ==========================================================================

STATS = {
    "amplitude": dict(
        label="Amplitude moments (mean/std/skew/kurt)",
        outputs=["amp_mean", "amp_std", "amp_skew", "amp_kurt"],
        gfn=amplitude_global, wfn=amplitude_windowed,
        note="Rayleigh(noise) vs Rician(target) vs heavy-tailed(clutter). "
             "Kurtosis & tail heaviness flag clutter."),
    "circular_diff": dict(
        label="Circular phase stats  (on phase DIFFERENCES → Doppler-aware)",
        outputs=["R", "circ_mean", "circ_var", "kappa", "rayleigh_p"],
        gfn=lambda z, a: circular_global(z, a, "diff"),
        wfn=lambda z, a, w, s: circular_windowed(z, a, w, s, "diff"),
        note="High R / low rayleigh_p over phase increments = coherent moving "
             "target. This is the target-relevant flavour."),
    "circular_raw": dict(
        label="Circular phase stats  (on RAW phase → zero-Doppler / clutter)",
        outputs=["R", "circ_mean", "circ_var", "kappa", "rayleigh_p"],
        gfn=lambda z, a: circular_global(z, a, "raw"),
        wfn=lambda z, a, w, s: circular_windowed(z, a, w, s, "raw"),
        note="Concentrated raw phase = stationary/zero-Doppler energy "
             "(often clutter, not the target)."),
    "pulsepair": dict(
        label="Pulse-pair  (|R1| coherence + Doppler)  ← main workhorse",
        outputs=["pp_coherence", "pp_doppler", "pp_power"],
        gfn=pulsepair_global, wfn=pulsepair_windowed,
        note="pp_coherence high AND pp_doppler away from 0 = moving target. "
             "The cheap O(N) replacement for a single Doppler FFT bin."),
    "lag_decay": dict(
        label="Lag-k coherence decay",
        outputs=["lag_coh_area", "lag_coh_1"],
        gfn=lag_decay_global, wfn=None,
        note="Slow decay of |R_k| across lags = long coherent dwell."),
    "spanfactor": dict(
        label="Coherent/non-coherent ratio (span factor)",
        outputs=["span_factor"],
        gfn=spanfactor_global, wfn=spanfactor_windowed,
        note="|Σz|/Σ|z|. Near 1 = coherent at ZERO Doppler. Combine with "
             "pulse-pair to stay Doppler-agnostic."),
    "linearity": dict(
        label="Phase-ramp linearity (residual coherence + slope)",
        outputs=["lin_residual_coh", "lin_slope"],
        gfn=linearity_global, wfn=linearity_windowed,
        note="Clean linear phase ramp of any slope = target; slope = Doppler."),
    "chirp": dict(
        label="Phase 2nd-difference constancy (chirp / reference shape)",
        outputs=["chirp_rate", "chirp_constancy", "chirp_d2_var"],
        gfn=chirp_global, wfn=chirp_windowed,
        note="Use ALONG RANGE. High constancy = LFM reference signature "
             "present (matched-filter-free)."),
    "spectral": dict(
        label="Spectral features (FFT — REFERENCE / validation only)",
        outputs=["spec_peak2avg", "spec_entropy", "spec_dom_bin"],
        gfn=spectral_global, wfn=None,
        note="Uses the FFT you want to remove; kept to validate FFT-free "
             "features against ground truth."),
}


def channels_needed(z_available, amp_available, phase_available):
    """All current stats need the complex signal, which we always build."""
    return True
