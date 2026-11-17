"""
Recognition metrics.

``compute_eer`` mirrors the official ``getEER.py``:

    scores = genuine/impostor matching distances
    if genuine.mean() < impostor.mean(): negate both  (use similarity)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(...)
    eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0, 1)

Both the fitted EER (primary, as in the paper) and the "mean EER"
(``eer_1/2`` in the official script) are returned.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn import metrics


def compute_eer(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
) -> Dict[str, float]:
    """Calculate EER / AUC from matching distances (smaller = more similar).

    Args:
        genuine_scores: distances of same-class pairs.
        impostor_scores: distances of different-class pairs.

    Returns:
        Dict with ``eer``, ``eer_percent``, ``eer_mean``, thresholds, ``auc``
        and score statistics. Private keys (``_fpr``, ``_tpr``,
        ``_thresholds``) hold the ROC arrays for plotting and are removed
        before JSON serialization.
    """
    g = np.asarray(genuine_scores, dtype=np.float64).ravel()
    i = np.asarray(impostor_scores, dtype=np.float64).ravel()
    if g.size == 0 or i.size == 0:
        raise ValueError("Both genuine and impostor scores are required.")

    genuine_mean = float(g.mean())
    impostor_mean = float(i.mean())

    # Official getEER.py: roc_curve expects higher score = more genuine.
    flipped = genuine_mean < impostor_mean
    if flipped:
        g, i = -g, -i

    y = np.concatenate([np.ones(g.size, dtype=np.int8),
                        np.zeros(i.size, dtype=np.int8)])
    scores = np.concatenate([g, i])

    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = float(metrics.auc(fpr, tpr))

    tpr_fn = interp1d(fpr, tpr)
    eer = float(brentq(lambda x: 1.0 - x - tpr_fn(x), 0.0, 1.0))
    thresh = float(interp1d(fpr, thresholds)(eer))

    # "mean EER" from the official script (eer_1/2)
    diff = np.abs(fpr - (1.0 - tpr))
    idx = int(np.argmin(diff))
    eer_mean = float((fpr[idx] + (1.0 - tpr[idx])) / 2.0)
    thresh_mean = float(thresholds[idx])

    if flipped:
        thresh = -thresh
        thresh_mean = -thresh_mean
        fpr, tpr, thresholds = fpr, tpr, -thresholds

    return {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_mean": eer_mean,
        "eer_mean_percent": eer_mean * 100.0,
        "threshold": thresh,
        "threshold_mean": thresh_mean,
        "auc": roc_auc,
        "num_genuine": int(g.size),
        "num_impostor": int(i.size),
        "genuine_mean": genuine_mean,
        "impostor_mean": impostor_mean,
        "flipped_to_similarity": bool(flipped),
        # private: used by plotting, stripped before JSON
        "_fpr": fpr,
        "_tpr": tpr,
        "_thresholds": thresholds,
    }


def score_statistics(scores: np.ndarray) -> Dict[str, float]:
    """min / max / mean / std of a score array (official getGI style)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "count": int(s.size),
    }


def gi_histogram(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    samples: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Genuine/impostor histogram exactly as official ``getGI.py``.

    Genuine and impostor scores are min-max scaled independently, rounded to
    ``samples + 1`` bins, and normalized to percentages.
    """
    g = np.asarray(genuine_scores, dtype=np.float64).ravel()
    o = np.asarray(impostor_scores, dtype=np.float64).ravel()

    g_min, g_max = float(g.min()), float(g.max())
    o_min, o_max = float(o.min()), float(o.max())
    g_scaled = (g - g_min) / (g_max - g_min) * samples
    o_scaled = (o - o_min) / (o_max - o_min) * samples

    hist_g = np.zeros(samples + 1, dtype=np.float64)
    hist_o = np.zeros(samples + 1, dtype=np.float64)
    for v in g_scaled:
        hist_g[int(round(v))] += 1
    for v in o_scaled:
        hist_o[int(round(v))] += 1

    hist_g = hist_g / hist_g.sum() * 100.0
    hist_o = hist_o / hist_o.sum() * 100.0

    x_g = np.linspace(0, 1, samples + 1) * (g_max - g_min) + g_min
    x_o = np.linspace(0, 1, samples + 1) * (o_max - o_min) + o_min
    return x_g, hist_g, x_o, hist_o


def tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    far: float,
) -> float:
    """True accept rate at a given false accept rate (distance scores)."""
    g = np.asarray(genuine_scores, dtype=np.float64).ravel()
    i = np.asarray(impostor_scores, dtype=np.float64).ravel()
    threshold = float(np.quantile(i, far))
    return float((g <= threshold).mean())
