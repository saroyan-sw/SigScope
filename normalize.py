"""
sigscope.normalize
===================

Normalisation / background-equalisation tools. These flatten the
range-dependent noise floor (clearly visible in the raw amplitude image)
so that a downstream feature or threshold is not just learning the
range-power gradient.

Every routine takes explicit, data-dependent parameters that the GUI
exposes as live controls ("regularise within the app").
"""

from __future__ import annotations
import numpy as np

_EPS = 1e-30


# --------------------------------------------------------------------------
# 1. RANGE-PROFILE NORMALISATION  (the "range-based normalisation")
#    Estimate a background level per range bin (a robust average across the
#    time axis) and divide it out. Removes the vertical floor gradient.
# --------------------------------------------------------------------------

def range_profile_normalize(mat, range_axis=0, method="median",
                            percentile=50.0, mode="divide"):
    """
    mat        : 2D real array (usually amplitude)
    range_axis : axis that indexes range bins
    method     : 'median' | 'mean' | 'percentile'
    mode       : 'divide' (linear) | 'db' (subtract in dB)

    The background profile has one value per range bin, broadcast across
    the other axis.
    """
    other = 1 - range_axis
    if method == "median":
        prof = np.median(mat, axis=other, keepdims=True)
    elif method == "mean":
        prof = np.mean(mat, axis=other, keepdims=True)
    else:
        prof = np.percentile(mat, percentile, axis=other, keepdims=True)

    if mode == "db":
        out = 20 * np.log10(np.abs(mat) + _EPS) - 20 * np.log10(prof + _EPS)
    else:
        out = mat / (prof + _EPS)
    return out, np.squeeze(prof)


# --------------------------------------------------------------------------
# 2. CA-CFAR  (Cell-Averaging Constant False Alarm Rate), 1D along an axis
#    For each cell, estimate the noise from TRAINING cells on both sides,
#    skipping GUARD cells next to the cell under test. Output = cell / noise.
#    This is both a normaliser and, thresholded, a detector statistic.
# --------------------------------------------------------------------------

def ca_cfar_1d(mat, axis, guard=2, train=8, scale=1.0):
    """
    mat   : 2D real power-like array (use amplitude**2 for a true CFAR)
    axis  : axis along which the CFAR window slides
    guard : guard cells on EACH side
    train : training cells on EACH side
    scale : multiply the noise estimate (raises/lowers the effective threshold)

    returns ratio = cell / (scale * noise_estimate). Values >> 1 are detections.
    """
    x = np.moveaxis(mat, axis, -1).astype(np.float64)
    N = x.shape[-1]
    half = guard + train
    # cumulative sum with leading zero for O(N) window sums
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, -1)], -1)

    def wsum(lo, hi):
        """sum of x[..., lo:hi] with clipping, vectorised over start index."""
        lo = np.clip(lo, 0, N)
        hi = np.clip(hi, 0, N)
        return c[..., hi] - c[..., lo]

    idx = np.arange(N)
    left_lo = idx - half
    left_hi = idx - guard
    right_lo = idx + guard + 1
    right_hi = idx + half + 1
    left_sum = wsum(left_lo, left_hi)
    right_sum = wsum(right_lo, right_hi)
    left_cnt = np.clip(left_hi, 0, N) - np.clip(left_lo, 0, N)
    right_cnt = np.clip(right_hi, 0, N) - np.clip(right_lo, 0, N)
    noise = (left_sum + right_sum) / np.maximum(left_cnt + right_cnt, 1)
    ratio = x / (scale * noise + _EPS)
    return np.moveaxis(ratio, -1, axis)


# --------------------------------------------------------------------------
# 3. LOCAL RANK / PERCENTILE NORMALISATION
#    Replace each value by where it sits (0..1) within a local window.
#    Distribution-free; very robust to a varying floor. Heavier to compute,
#    so it uses a strided/looped 1D pass along one axis.
# --------------------------------------------------------------------------

def local_rank_1d(mat, axis, win=64):
    """
    Approximate local rank along `axis`: for each cell, fraction of the
    surrounding `win` cells it exceeds. Implemented with a sliding compare
    on a coarse grid for speed, then interpolated.
    """
    x = np.moveaxis(mat, axis, -1).astype(np.float64)
    N = x.shape[-1]
    half = win // 2
    pad = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(half, half)], mode="edge")
    out = np.empty_like(x)
    # vectorised over all lines, loop only over sample positions
    for i in range(N):
        window = pad[..., i:i + win]
        out[..., i] = (window < x[..., i:i + 1]).mean(-1)
    return np.moveaxis(out, -1, axis)


# --------------------------------------------------------------------------
# 4. DISPLAY SCALING  (dB + robust percentile clip) -- for the image view
# --------------------------------------------------------------------------

def to_db(mat):
    return 20 * np.log10(np.abs(mat) + _EPS)


def clip_levels(mat, lo_pct=2.0, hi_pct=99.5):
    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        return 0.0, 1.0
    return (float(np.percentile(finite, lo_pct)),
            float(np.percentile(finite, hi_pct)))


NORMALIZERS = {
    "none": "No normalisation",
    "range_profile": "Range-profile (divide by per-range background)",
    "range_profile_db": "Range-profile in dB (subtract per-range background)",
    "ca_cfar": "CA-CFAR ratio (cell / local training average)",
    "local_rank": "Local rank / percentile (distribution-free)",
}
