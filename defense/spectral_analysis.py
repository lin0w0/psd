"""Statistical analysis, plotting and reporting for spectral consistency.

The core quantity is the cosine similarity between the CompNet feature of an
image and the feature of its Gaussian low-pass filtered version::

    C_r(x) = cos(f(x), f(G_r(x)))

Features are L2-normalized by ``CompNet.getFeatureCode``, so the cosine is a
plain dot product.  All model inputs go through the **official**
``dataset.dataset.NormSingleROI`` transform exactly once.

This module is deliberately free of experiment-orchestration logic so that
``scripts/evaluate_spectral_consistency.py`` (full run) and
``scripts/generate_spectral_report.py`` (regenerate from saved outputs) can
share the same statistics and figure code.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from dataset.dataset import NormSingleROI
from defense.spectral_filter import gaussian_lowpass

# ---------------------------------------------------------------------------
# Constants / labels
# ---------------------------------------------------------------------------

MAIN_ATTACKS = ("pgd", "apgd", "cw")
ATTACK_LABELS = {
    "clean": "Clean",
    "pgd": "PGD",
    "apgd": "APGD",
    "cw": "CW",
    "patch": "Patch",
}
MODE_LABELS = {
    "full_frame": "Full-frame LPF",
    "roi_aware": "ROI-aware LPF",
}
SESSION_LABELS = {
    "session1": "Session 1 (calibration)",
    "session2": "Session 2 (validation)",
}
CONDITION_STYLE = {
    "clean": dict(color="#000000", marker="o", linestyle="-"),
    "pgd": dict(color="#0072B2", marker="s", linestyle="-"),
    "apgd": dict(color="#E69F00", marker="^", linestyle="-"),
    "cw": dict(color="#009E73", marker="D", linestyle="-"),
    "patch": dict(color="#56B4E9", marker="X", linestyle="-"),
}
ACCURACY_STYLE = {
    "clean": dict(color="#000000", marker="o"),
    "clean_lpf": dict(color="#0072B2", marker="s"),
    "adv": dict(color="#D55E00", marker="^"),
    "adv_lpf": dict(color="#009E73", marker="D"),
}


def attack_label(name: str) -> str:
    return ATTACK_LABELS.get(name, name)


def mode_label(mode: str) -> str:
    return MODE_LABELS.get(mode, mode)


def epsilon_tag(eps: Optional[float]) -> str:
    """Stable string tag for an epsilon (``None`` -> ``"none"``)."""
    if eps is None:
        return "none"
    return f"{float(eps):.10f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Official preprocessing + feature extraction
# ---------------------------------------------------------------------------

_NORM = NormSingleROI()


def official_normalize_batch(pixels: torch.Tensor) -> torch.Tensor:
    """Apply the official ``NormSingleROI`` to a batch, one ROI at a time.

    ``NormSingleROI`` expects ``[1, H, W]`` and mutates its input in place, so
    each image is cloned and normalized individually.  This is the exact
    transform used by ``PalmROIDataset`` at inference time.
    """
    if pixels.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(pixels.shape)}")
    out = torch.empty_like(pixels)
    for i in range(pixels.shape[0]):
        out[i : i + 1] = _NORM(pixels[i].clone())
    return out


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    device: torch.device | str,
    batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return L2-normalized feature codes and ArcFace logits for pixel inputs.

    ``NormSingleROI`` is applied exactly once.  In ``eval`` mode CompNet's
    dropout is the identity and ``ArcMarginProduct.forward`` normalizes its
    input, so ``arclayer(getFeatureCode(x))`` is numerically equivalent to
    ``model(x)`` while sharing a single convolutional forward pass.
    """
    device = torch.device(device)
    model.eval()
    codes: List[torch.Tensor] = []
    logits: List[torch.Tensor] = []
    n = int(pixels.shape[0])
    for start in range(0, n, batch_size):
        batch = pixels[start : start + batch_size].to(device)
        x = official_normalize_batch(batch)
        c = model.getFeatureCode(x)
        logits.append(model.arclayer(c).float().cpu())
        codes.append(c.float().cpu())
    return torch.cat(codes, 0), torch.cat(logits, 0)


@torch.no_grad()
def filtered_similarity(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    base_codes: torch.Tensor,
    labels: np.ndarray,
    radius: float,
    mode: str,
    roi_mask: Optional[torch.Tensor],
    device: torch.device | str,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """Cosine consistency and predictions after low-pass filtering.

    Args:
        pixels: ``[N, 1, H, W]`` pixel tensors in ``[0, 1]``.
        base_codes: ``[N, 512]`` codes of the *unfiltered* image.
        labels: ``[N]`` ground-truth labels.
        radius: FFT-bin Gaussian radius.
        mode: ``"full_frame"`` or ``"roi_aware"``.
        roi_mask: Clean-image ROI mask used by the ROI-aware mode.

    Returns:
        ``{"cosine", "prediction", "top1", "top5"}`` as numpy arrays.
    """
    device = torch.device(device)
    model.eval()
    base_codes = base_codes.to(device)
    n = int(pixels.shape[0])
    cosines: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    top1: List[np.ndarray] = []
    top5: List[np.ndarray] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = pixels[start:end].to(device)
        m = None if roi_mask is None else roi_mask[start:end].to(device)
        x_low = gaussian_lowpass(x, radius, mode=mode, mask=m)
        codes, logits = encode_pixels(
            model, x_low, device, batch_size=int(x_low.shape[0])
        )
        cos = (base_codes[start:end] * codes.to(device)).sum(dim=1)
        cosines.append(cos.float().cpu().numpy())

        top = logits.topk(min(5, logits.shape[1]), dim=1).indices.numpy()
        pred = top[:, 0]
        y = labels[start:end]
        predictions.append(pred.astype(np.int64))
        top1.append(pred == y)
        top5.append((top == y[:, None]).any(axis=1))

    return {
        "cosine": np.concatenate(cosines).astype(np.float64),
        "prediction": np.concatenate(predictions),
        "top1": np.concatenate(top1).astype(bool),
        "top5": np.concatenate(top5).astype(bool),
    }


def predictions_from_logits(
    logits: torch.Tensor, labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """Top-1/top-5 predictions and correctness flags."""
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices.numpy()
    pred = top[:, 0].astype(np.int64)
    y = np.asarray(labels, dtype=np.int64)
    return {
        "prediction": pred,
        "top1": pred == y,
        "top5": (top == y[:, None]).any(axis=1),
    }


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------

def _safe_auc(y: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return None


def _pooled_cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if len(a) < 2 or len(b) < 2:
        return None
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = math.sqrt(max((var_a + var_b) / 2.0, 0.0))
    if pooled <= 0.0:
        return None
    return float((np.mean(b) - np.mean(a)) / pooled)


def detection_metrics(
    clean_scores: np.ndarray,
    adv_scores: np.ndarray,
    fpr_target: float = 0.01,
    threshold: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Detection metrics for ``score = 1 - cosine`` (larger = adversarial).

    Args:
        clean_scores: Scores of clean samples (negative class).
        adv_scores: Scores of adversarial samples (positive class).
        fpr_target: Target false-positive rate for threshold calibration.
        threshold: Fixed threshold.  If ``None``, calibrated on
            ``clean_scores`` at ``fpr_target`` (``np.quantile`` with
            ``method="higher"``).

    Returns:
        AUC, PR-AUC, Cohen's d, KS statistic/p-value, TPR/FPR, threshold.
    """
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    adv_scores = np.asarray(adv_scores, dtype=np.float64)
    out: Dict[str, Optional[float]] = {
        "n_clean": int(clean_scores.size),
        "n_adv": int(adv_scores.size),
        "auc": None,
        "pr_auc": None,
        "threshold": None,
        "tpr": None,
        "fpr": None,
        "cohens_d": None,
        "ks_statistic": None,
        "ks_pvalue": None,
    }
    if clean_scores.size == 0 or adv_scores.size == 0:
        return out

    y = np.concatenate(
        [np.zeros(clean_scores.size, dtype=np.int64), np.ones(adv_scores.size, dtype=np.int64)]
    )
    scores = np.concatenate([clean_scores, adv_scores])
    out["auc"] = _safe_auc(y, scores)
    if len(np.unique(y)) == 2:
        try:
            out["pr_auc"] = float(average_precision_score(y, scores))
        except ValueError:
            out["pr_auc"] = None

    if threshold is None:
        threshold = float(
            np.quantile(clean_scores, 1.0 - fpr_target, method="higher")
        )
    out["threshold"] = float(threshold)
    # Strictly greater keeps the empirical FPR <= target when the threshold
    # falls on a clean score (the usual case for 1% of 1200 samples).
    out["tpr"] = float(np.mean(adv_scores > threshold))
    out["fpr"] = float(np.mean(clean_scores > threshold))
    out["cohens_d"] = _pooled_cohens_d(clean_scores, adv_scores)

    ks = scipy_stats.ks_2samp(clean_scores, adv_scores)
    out["ks_statistic"] = float(ks.statistic)
    out["ks_pvalue"] = float(ks.pvalue)
    return out


def paired_delta_stats(
    clean_values: np.ndarray, adv_values: np.ndarray
) -> Dict[str, Optional[float]]:
    """Paired comparison of per-image cosine similarities.

    ``delta_i = C_clean(x_i) - C_adv(x_i)``.  Reports the mean, a 95% CI,
    a paired t-test, a Wilcoxon signed-rank test and Cohen's ``dz``.
    """
    d = np.asarray(clean_values, dtype=np.float64) - np.asarray(
        adv_values, dtype=np.float64
    )
    n = int(d.size)
    out: Dict[str, Optional[float]] = {
        "delta_mean": None,
        "delta_std": None,
        "delta_ci95_low": None,
        "delta_ci95_high": None,
        "paired_t_statistic": None,
        "paired_t_pvalue": None,
        "paired_wilcoxon_pvalue": None,
        "paired_cohens_dz": None,
        "n_pairs": n,
    }
    if n == 0:
        return out

    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if n > 1 else 0.0
    out["delta_mean"] = mean
    out["delta_std"] = std
    if n > 1:
        se = std / math.sqrt(n)
        if se > 0:
            crit = float(scipy_stats.t.ppf(0.975, df=n - 1))
            out["delta_ci95_low"] = mean - crit * se
            out["delta_ci95_high"] = mean + crit * se
            out["paired_cohens_dz"] = mean / std if std > 0 else None
            try:
                t = scipy_stats.ttest_1samp(d, 0.0)
                out["paired_t_statistic"] = float(t.statistic)
                out["paired_t_pvalue"] = float(t.pvalue)
            except ValueError:
                pass
            if np.any(d != 0):
                try:
                    w = scipy_stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
                    out["paired_wilcoxon_pvalue"] = float(w.pvalue)
                except ValueError:
                    pass
            else:
                out["paired_wilcoxon_pvalue"] = 1.0
        else:
            out["delta_ci95_low"] = mean
            out["delta_ci95_high"] = mean
            out["paired_cohens_dz"] = 0.0
            out["paired_t_pvalue"] = 1.0
            out["paired_wilcoxon_pvalue"] = 1.0
    return out


def distribution_stats(values: np.ndarray) -> Dict[str, float]:
    """Mean / std / median / 25% / 75% quantiles (sample std, ddof=1)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def compare_attacks(
    deltas: Dict[str, np.ndarray],
) -> Dict[str, object]:
    """Friedman + pairwise Wilcoxon comparison of paired cosine deltas."""
    names = [k for k, v in deltas.items() if v is not None and len(v) > 0]
    out: Dict[str, object] = {
        "attacks": names,
        "friedman_statistic": None,
        "friedman_pvalue": None,
        "pairwise": [],
    }
    if len(names) < 2:
        return out
    lengths = {len(deltas[n]) for n in names}
    if len(lengths) != 1:
        return out
    matrix = np.column_stack([deltas[n] for n in names])
    # Drop rows where all deltas are identical (tied ranks are fine, but an
    # all-zero row adds no information and can make the test singular).
    if len(names) >= 3:
        try:
            fried = scipy_stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
            out["friedman_statistic"] = float(fried.statistic)
            out["friedman_pvalue"] = float(fried.pvalue)
        except ValueError:
            pass

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = matrix[:, i]
            b = matrix[:, j]
            diff = a - b
            pvalue = None
            if np.any(diff != 0):
                try:
                    pvalue = float(
                        scipy_stats.wilcoxon(
                            a, b, zero_method="wilcox", alternative="two-sided"
                        ).pvalue
                    )
                except ValueError:
                    pvalue = None
            else:
                pvalue = 1.0
            pairs.append(
                {
                    "attack_a": names[i],
                    "attack_b": names[j],
                    "mean_delta_a": float(np.mean(a)),
                    "mean_delta_b": float(np.mean(b)),
                    "wilcoxon_pvalue": pvalue,
                }
            )
    raw = [p["wilcoxon_pvalue"] for p in pairs]
    valid = [p for p in raw if p is not None]
    if len(valid) == len(raw):
        adjusted = holm_adjust(raw)  # type: ignore[arg-type]
        for record, adj in zip(pairs, adjusted):
            record["wilcoxon_pvalue_holm"] = float(adj)
    out["pairwise"] = pairs
    return out


def roc_arrays(
    clean_similarity: np.ndarray,
    adv_similarity: np.ndarray,
    fpr_target: float = 0.01,
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """ROC curve points and metrics for ``score = 1 - similarity``."""
    clean_scores = 1.0 - np.asarray(clean_similarity, dtype=np.float64)
    adv_scores = 1.0 - np.asarray(adv_similarity, dtype=np.float64)
    metrics = detection_metrics(
        clean_scores, adv_scores, fpr_target=fpr_target, threshold=threshold
    )
    y = np.concatenate(
        [np.zeros(clean_scores.size, dtype=np.int64), np.ones(adv_scores.size, dtype=np.int64)]
    )
    scores = np.concatenate([clean_scores, adv_scores])
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, scores)
    else:
        fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return {"fpr": fpr, "tpr": tpr, "metrics": metrics}


def compute_radius_transfer(
    sample_scores: Dict[Tuple, Dict[str, np.ndarray]],
    calibration_session: str,
    validation_session: str,
    eps_tag: str,
    attacks: Sequence[str],
    modes: Sequence[str],
    radii: Sequence[float],
    fpr_target: float = 0.01,
) -> Dict[str, List[Dict[str, Optional[float]]]]:
    """Fixed-threshold transfer across radii for one epsilon.

    For every radius, calibrate the FPR=``fpr_target`` threshold on the
    calibration session's clean scores, then evaluate on the validation
    session.  This separates *ranking* performance (AUC, threshold-free)
    from *operating-point transfer* (FPR/TPR), which is the key practical
    issue for test-time detection.

    Returns:
        ``{mode: [{"radius", "threshold", "validation_auc_mean",
        "validation_fpr", "validation_tpr_mean"}, ...]}``
    """
    out: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for mode in modes:
        entries: List[Dict[str, Optional[float]]] = []
        for radius in radii:
            cal_clean = sample_scores[
                sample_score_key(
                    calibration_session, "clean", "clean", mode, float(radius)
                )
            ]["clean"]
            val_clean = sample_scores[
                sample_score_key(
                    validation_session, "clean", "clean", mode, float(radius)
                )
            ]["clean"]
            cal_scores = 1.0 - cal_clean
            val_scores = 1.0 - val_clean
            threshold = float(
                np.quantile(cal_scores, 1.0 - fpr_target, method="higher")
            )
            fpr = float(np.mean(val_scores > threshold))
            aucs: List[float] = []
            tprs: List[float] = []
            for attack in attacks:
                val_pair = sample_scores[
                    sample_score_key(
                        validation_session, attack, eps_tag, mode, float(radius)
                    )
                ]
                auc = _safe_auc(
                    np.r_[
                        np.zeros(val_pair["clean"].size),
                        np.ones(val_pair["adv"].size),
                    ],
                    np.r_[1.0 - val_pair["clean"], 1.0 - val_pair["adv"]],
                )
                if auc is not None:
                    aucs.append(float(auc))
                tprs.append(float(np.mean(1.0 - val_pair["adv"] > threshold)))
            entries.append(
                {
                    "radius": float(radius),
                    "threshold": threshold,
                    "validation_fpr": fpr,
                    "validation_auc_mean": float(np.mean(aucs)) if aucs else None,
                    "validation_tpr_mean": float(np.mean(tprs)) if tprs else None,
                }
            )
        out[mode] = entries
    return out


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _records_for(
    records: Sequence[Dict],
    session: str,
    attack: str,
    eps_tag: str,
    mode: str,
) -> List[Dict]:
    rows = [
        r
        for r in records
        if r.get("session") == session
        and r.get("attack") == attack
        and (attack == "clean" or r.get("epsilon_tag") == eps_tag)
        and r.get("filter_mode") == mode
    ]
    return sorted(rows, key=lambda r: float(r["filter_radius"]))


def _clean_record(
    records: Sequence[Dict],
    session: str,
    eps_tag: str,
    mode: str,
    radius: Optional[float] = None,
) -> Optional[Dict]:
    rows = _records_for(records, session, "clean", eps_tag, mode)
    if radius is not None:
        rows = [r for r in rows if int(r["filter_radius"]) == int(radius)]
    return rows[0] if rows else None


def _make_figure(nrows: int, ncols: int, **kwargs):
    _set_plot_style()
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    if nrows * ncols == 1:
        axes = np.asarray([[axes]])
    elif nrows == 1:
        axes = np.asarray(axes).reshape(1, ncols)
    elif ncols == 1:
        axes = np.asarray(axes).reshape(nrows, 1)
    return fig, axes


def plot_consistency_vs_radius(
    records: Sequence[Dict],
    attacks: Sequence[str],
    eps_tag: str,
    sessions: Sequence[str],
    modes: Sequence[str],
    path: Path,
) -> None:
    """Figure 1: mean +/- std cosine consistency vs radius."""
    fig, axes = _make_figure(
        len(modes), len(sessions), figsize=(5.2 * len(sessions), 3.8 * len(modes)),
        sharex=True, sharey=True, squeeze=False,
    )
    conditions = ["clean", *attacks]
    for i, mode in enumerate(modes):
        for j, session in enumerate(sessions):
            ax = axes[i, j]
            for name in conditions:
                rows = _records_for(records, session, name, eps_tag, mode)
                if not rows:
                    continue
                x = [r["filter_radius"] for r in rows]
                if name == "clean":
                    y = [r["clean_cos_mean"] for r in rows]
                    e = [r["clean_cos_std"] for r in rows]
                else:
                    y = [r["adv_cos_mean"] for r in rows]
                    e = [r["adv_cos_std"] for r in rows]
                style = CONDITION_STYLE.get(name, {})
                ax.errorbar(
                    x, y, yerr=e, capsize=2.0, elinewidth=0.8,
                    label=attack_label(name), markersize=4.0, linewidth=1.3,
                    alpha=0.95, **style,
                )
            ax.set_title(f"{mode_label(mode)} | {SESSION_LABELS.get(session, session)}")
            ax.set_xlabel("Gaussian LPF radius $r$ (FFT bins)")
            if j == 0:
                ax.set_ylabel(r"Cosine similarity $C_r(x)$")
            ax.set_xticks(list(sorted({r["filter_radius"] for r in records})))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(conditions), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def sample_score_key(
    session: str, attack: str, eps_tag: str, mode: str, radius: float
) -> Tuple[str, str, str, str, float]:
    """Key into ``sample_scores``; clean rows ignore the epsilon tag."""
    tag = "clean" if attack == "clean" else eps_tag
    return (session, attack, tag, mode, float(radius))


def plot_distributions(
    records: Sequence[Dict],
    sample_scores: Dict[Tuple[str, str, str, str, float], Dict[str, np.ndarray]],
    attacks: Sequence[str],
    eps_tag: str,
    sessions: Sequence[str],
    modes: Sequence[str],
    best_radius: Dict[str, int],
    thresholds: Dict[str, float],
    path: Path,
) -> None:
    """Figure 2: cosine distributions at the selected radius (violin)."""
    conditions = ["clean", *attacks]
    fig, axes = _make_figure(
        len(modes), len(sessions), figsize=(5.2 * len(sessions), 3.8 * len(modes)),
        sharey=True, squeeze=False,
    )
    for i, mode in enumerate(modes):
        radius = float(best_radius[mode])
        for j, session in enumerate(sessions):
            ax = axes[i, j]
            data = []
            for name in conditions:
                key = sample_score_key(session, name, eps_tag, mode, radius)
                pair = sample_scores.get(key)
                if pair is None:
                    data.append(np.array([]))
                else:
                    data.append(pair["clean"] if name == "clean" else pair["adv"])
            positions = np.arange(1, len(conditions) + 1)
            nonempty = [d for d in data if d.size > 0]
            vp = ax.violinplot(
                nonempty, positions=[p for p, d in zip(positions, data) if d.size > 0],
                widths=0.8, showmeans=False, showmedians=True, showextrema=False,
            )
            for body, name in zip(vp["bodies"], [n for n, d in zip(conditions, data) if d.size > 0]):
                body.set_facecolor(CONDITION_STYLE.get(name, {}).get("color", "#999999"))
                body.set_edgecolor("black")
                body.set_alpha(0.55)
                body.set_linewidth(0.6)
            if "cmedians" in vp:
                vp["cmedians"].set_color("black")
                vp["cmedians"].set_linewidth(1.0)
            # Mean marker.
            for p, d in zip(positions, data):
                if d.size:
                    ax.plot([p], [float(np.mean(d))], marker="o", color="white",
                            markeredgecolor="black", markersize=3.5, zorder=5)
            ax.set_xticks(positions)
            ax.set_xticklabels([attack_label(n) for n in conditions], rotation=20, ha="right")
            ax.set_title(f"{mode_label(mode)} | {SESSION_LABELS.get(session, session)} "
                         f"($r={int(radius)}$)")
            ax.set_ylabel(r"Cosine similarity $C_r(x)$")
            thr = thresholds.get(mode)
            if thr is not None:
                ax.axhline(1.0 - thr, color="#B2182B", linestyle="--", linewidth=1.0,
                           label="FPR=1% threshold")
            ax.grid(axis="x", visible=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_delta_consistency(
    records: Sequence[Dict],
    attacks: Sequence[str],
    eps_tag: str,
    sessions: Sequence[str],
    modes: Sequence[str],
    path: Path,
) -> None:
    """Figure 3: Delta_C(r) = mean(C_clean) - mean(C_adv)."""
    fig, axes = _make_figure(
        len(modes), len(sessions), figsize=(5.2 * len(sessions), 3.8 * len(modes)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, mode in enumerate(modes):
        for j, session in enumerate(sessions):
            ax = axes[i, j]
            for name in attacks:
                rows = _records_for(records, session, name, eps_tag, mode)
                if not rows:
                    continue
                x = [r["filter_radius"] for r in rows]
                y = [r["delta_c"] for r in rows]
                style = CONDITION_STYLE.get(name, {})
                ax.plot(x, y, label=attack_label(name), markersize=4.5,
                        linewidth=1.4, **style)
            ax.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
            ax.set_title(f"{mode_label(mode)} | {SESSION_LABELS.get(session, session)}")
            ax.set_xlabel("Gaussian LPF radius $r$ (FFT bins)")
            if j == 0:
                ax.set_ylabel(r"$\Delta_C = \overline{C}_{clean} - \overline{C}_{adv}$")
            ax.set_xticks(list(sorted({r["filter_radius"] for r in records})))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(attacks), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_roc_curves(
    records: Sequence[Dict],
    sample_scores: Dict[Tuple[str, str, str, str, float], Dict[str, np.ndarray]],
    attacks: Sequence[str],
    eps_tag: str,
    session: str,
    modes: Sequence[str],
    best_radius: Dict[str, int],
    path: Path,
    fpr_target: float = 0.01,
) -> None:
    """Figure 4: ROC curves across radii (validation session)."""
    radii = sorted({r["filter_radius"] for r in records})
    norm = Normalize(vmin=min(radii), vmax=max(radii))
    cmap = plt.get_cmap("viridis")
    fig, axes = _make_figure(
        len(modes), len(attacks), figsize=(3.1 * len(attacks), 3.4 * len(modes)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, mode in enumerate(modes):
        for j, attack in enumerate(attacks):
            ax = axes[i, j]
            for radius in radii:
                pair = sample_scores.get(
                    sample_score_key(session, attack, eps_tag, mode, float(radius))
                )
                if pair is None:
                    continue
                roc = roc_arrays(pair["clean"], pair["adv"], fpr_target=fpr_target)
                auc = roc["metrics"].get("auc")
                is_best = int(radius) == int(best_radius.get(mode, -1))
                ax.plot(
                    roc["fpr"], roc["tpr"],
                    color=cmap(norm(radius)),
                    linewidth=2.6 if is_best else 1.0,
                    alpha=1.0 if is_best else 0.65,
                    label=f"$r={int(radius)}$" if is_best else None,
                )
            ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlim(-0.01, 1.01)
            ax.set_ylim(-0.01, 1.01)
            ax.set_title(f"{mode_label(mode)} | {attack_label(attack)}")
            if i == len(modes) - 1:
                ax.set_xlabel("False positive rate")
            if j == 0:
                ax.set_ylabel("True positive rate")
            best = int(best_radius.get(mode, -1))
            pair = sample_scores.get(
                sample_score_key(session, attack, eps_tag, mode, float(best))
            )
            if pair is not None:
                auc = roc_arrays(pair["clean"], pair["adv"], fpr_target=fpr_target)[
                    "metrics"
                ].get("auc")
                ax.text(0.98, 0.04, f"best $r={best}$, AUC={auc:.3f}",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5)
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Gaussian LPF radius $r$ (FFT bins)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.9, 1.0))
    fig.savefig(path)
    plt.close(fig)


def plot_classification_accuracy(
    records: Sequence[Dict],
    attacks: Sequence[str],
    eps_tag: str,
    session: str,
    modes: Sequence[str],
    path: Path,
) -> None:
    """Figure 5: Top-1/Top-5 accuracy for the four input conditions."""
    conditions = [
        ("clean", "Top-1", "clean_top1", "-"),
        ("clean", "Top-5", "clean_top5", "--"),
        ("clean_lpf", "Top-1", "clean_lpf_top1", "-"),
        ("clean_lpf", "Top-5", "clean_lpf_top5", "--"),
        ("adv", "Top-1", "adv_top1", "-"),
        ("adv", "Top-5", "adv_top5", "--"),
        ("adv_lpf", "Top-1", "adv_lpf_top1", "-"),
        ("adv_lpf", "Top-5", "adv_lpf_top5", "--"),
    ]
    fig, axes = _make_figure(
        len(modes), len(attacks), figsize=(3.1 * len(attacks), 3.4 * len(modes)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, mode in enumerate(modes):
        for j, attack in enumerate(attacks):
            ax = axes[i, j]
            rows = _records_for(records, session, attack, eps_tag, mode)
            if not rows:
                continue
            x = [r["filter_radius"] for r in rows]
            for condition, label, field, linestyle in conditions:
                style = ACCURACY_STYLE[condition]
                ax.plot(
                    x, [r[field] for r in rows], linestyle=linestyle,
                    marker=style["marker"], color=style["color"],
                    markersize=3.2, linewidth=1.2, label=label,
                )
            ax.set_title(f"{mode_label(mode)} | {attack_label(attack)}")
            ax.set_ylim(0, 105)
            if i == len(modes) - 1:
                ax.set_xlabel("Gaussian LPF radius $r$ (FFT bins)")
            if j == 0:
                ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x)
    # Compact two-part legend: condition colour + line style.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ACCURACY_STYLE[k]["color"], marker=ACCURACY_STYLE[k]["marker"],
               label=lab)
        for k, lab in [("clean", "Clean"), ("clean_lpf", "Clean-LPF"),
                       ("adv", "Adversarial"), ("adv_lpf", "Adversarial-LPF")]
    ] + [
        Line2D([0], [0], color="black", linestyle="-", label="Top-1"),
        Line2D([0], [0], color="black", linestyle="--", label="Top-5"),
    ]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def plot_auc_vs_radius(
    records: Sequence[Dict],
    attacks: Sequence[str],
    eps_tag: str,
    sessions: Sequence[str],
    modes: Sequence[str],
    best_radius: Dict[str, int],
    path: Path,
) -> None:
    """Supplementary: ROC-AUC vs radius (supports the radius choice)."""
    fig, axes = _make_figure(
        len(modes), len(sessions), figsize=(5.2 * len(sessions), 3.8 * len(modes)),
        sharex=True, sharey=True, squeeze=False,
    )
    for i, mode in enumerate(modes):
        for j, session in enumerate(sessions):
            ax = axes[i, j]
            for name in attacks:
                rows = _records_for(records, session, name, eps_tag, mode)
                if not rows:
                    continue
                x = [r["filter_radius"] for r in rows]
                y = [r["auc"] for r in rows]
                style = CONDITION_STYLE.get(name, {})
                ax.plot(x, y, label=attack_label(name), markersize=4.2,
                        linewidth=1.3, **style)
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.axvline(best_radius.get(mode, 0), color="#B2182B", linestyle="--",
                       linewidth=0.9, alpha=0.8)
            ax.set_title(f"{mode_label(mode)} | {SESSION_LABELS.get(session, session)}")
            ax.set_xlabel("Gaussian LPF radius $r$ (FFT bins)")
            if j == 0:
                ax.set_ylabel("ROC-AUC")
            ax.set_xticks(list(sorted({r["filter_radius"] for r in records})))
            ax.set_ylim(0.4, 1.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(attacks), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def read_summary_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}%"


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "—"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4g}"


def _spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(x) < 3:
        return None, None
    rho, p = scipy_stats.spearmanr(x, y)
    return float(rho), float(p)


def write_report(
    path: Path,
    meta: Dict,
    selection: Dict,
    validation: Dict,
    records: Sequence[Dict],
    attack_comparison: Dict[str, Dict],
    roi_stats: Dict,
    figures: Dict[str, str],
    radius_transfer: Optional[Dict[str, List[Dict[str, Optional[float]]]]] = None,
) -> None:
    """Write a Markdown report driven entirely by computed results."""
    primary_eps = meta.get("primary_epsilon")
    eps_tag_value = epsilon_tag(primary_eps)
    attacks = selection.get("main_attacks", list(MAIN_ATTACKS))
    modes = meta.get("filter_modes", ["full_frame", "roi_aware"])
    sessions = meta.get("sessions", ["session1", "session2"])
    cal = meta.get("calibration_session", "session1")
    val = meta.get("validation_session", "session2")
    fpr_target = float(selection.get("fpr_target", 0.01))
    best_radius = {m: selection["by_mode"][m]["best_radius"] for m in modes}
    thresholds = {m: selection["by_mode"][m]["threshold"] for m in modes}
    primary_mode = selection.get("primary_mode", modes[0])

    lines: List[str] = []
    lines.append("# CompNet 频谱一致性预实验报告")
    lines.append("")
    lines.append(
        "本报告由 `scripts/evaluate_spectral_consistency.py` 根据实际运行结果自动生成；"
        "所有数值均来自 `summary.json` / `summary.csv` / `sample_level.csv`，不存在写死的结果。"
    )
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append(f"- 模型：CompNet（Tongji，600 类，128×128 单通道，512 维 L2 归一化特征）")
    lines.append(f"- 预处理：官方 `NormSingleROI`（每个输入仅执行一次）")
    lines.append(f"- 干净样本：`{cal}` 与 `{val}` 各 {meta.get('sample_n')} 张（每类 {meta.get('per_class')} 张，seed={meta.get('seed')}）")
    lines.append(f"- 攻击：{', '.join(attack_label(a) for a in attacks)}；主 epsilon = {primary_eps!r}（像素域 [0,1] L∞）")
    extra_attacks = meta.get("extra_attacks") or []
    if extra_attacks:
        lines.append(
            "- 补充攻击（不参与半径选择/主图）："
            + ", ".join(attack_label(a) for a in extra_attacks)
            + "；其逐样本结果保存在 `summary.csv` / `sample_level.csv`。"
        )
    lines.append(f"- 滤波：FFT 高斯低通，r ∈ {meta.get('radii')}（频率 bin 单位），模式：{' / '.join(mode_label(m) for m in modes)}")
    lines.append(f"- 检测分数：S(x) = 1 − C_r(x)；FPR 目标 = {fpr_target:.0%}")
    lines.append(f"- 半径选择：在 `{cal}` 上按主攻击的平均 ROC-AUC 选择；阈值由 `{cal}` 干净样本的 99% 分位数校准")
    lines.append(f"- 独立验证：`{val}` 使用 `{cal}` 选定的半径与阈值")
    if meta.get("device"):
        lines.append(f"- 设备：{meta.get('device')}；运行时间：{meta.get('elapsed_seconds', 0):.1f} s")
    lines.append("")

    # ROI finding
    lines.append("## 2. ROI 背景与两种滤波模式")
    lines.append("")
    zero_frac = roi_stats.get("images_with_zero_pixels_fraction")
    mean_mask = roi_stats.get("mean_roi_fraction")
    eq_diff = roi_stats.get("mode_equivalence_max_abs_diff")
    lines.append(
        f"- 干净图像中检测到零像素（黑色背景）的比例：{_fmt_pct(100.0 * zero_frac if zero_frac is not None else None, 2)}"
    )
    lines.append(f"- ROI 掩码 M 的平均面积占比：{_fmt(mean_mask, 4)}")
    lines.append(
        f"- 同一半径下 Full-frame 与 ROI-aware 输出的最大绝对差：{eq_diff if eq_diff is not None else '—'}"
    )
    if zero_frac is not None and zero_frac < 1e-6:
        lines.append(
            "- **结论：** 本数据集 ROI 图像没有黑色背景，M 恒为 1，ROI-aware 归一化卷积在数学上"
            "退化为 Full-frame 滤波；两种模式的统计结果相同（差异仅为数值噪声）。因此后续讨论"
            "以 Full-frame 为主，ROI-aware 结果作为等价性验证。"
        )
    else:
        lines.append(
            "- **结论：** 两种模式在具有背景的样本上会产生差异，后续以 Full-frame 为基线并单独报告 ROI-aware 结果。"
        )
    lines.append("")

    # Selection
    lines.append("## 3. 滤波半径选择（Session 1）")
    lines.append("")
    lines.append("| 模式 | 最佳 r | 平均 AUC | " + " | ".join(attack_label(a) for a in attacks) + " | 阈值 |")
    lines.append("|---|---|---|" + "---|" * len(attacks) + "---|")
    for mode in modes:
        entry = selection["by_mode"].get(mode)
        if not entry:
            continue
        per_attack = entry.get("per_attack_auc", {})
        lines.append(
            f"| {mode_label(mode)} | {entry['best_radius']} | {_fmt(entry['mean_auc'])} | "
            + " | ".join(_fmt(per_attack.get(a)) for a in attacks)
            + f" | {_fmt(entry.get('threshold'))} |"
        )
    lines.append("")
    lines.append(f"主模式（按平均 AUC 选取）：**{mode_label(primary_mode)}**。")
    lines.append("")

    # Attack-effectiveness check on the calibration session.  A zero attack
    # success rate means the "adversarial" samples were not actually moved
    # across the decision boundary; their consistency statistics must not be
    # read as detection of a successful attack.
    cal_asr = {}
    for attack in attacks:
        rows = _records_for(records, cal, attack, eps_tag_value, primary_mode)
        row = next(
            (x for x in rows
             if int(x["filter_radius"]) == int(best_radius[primary_mode])),
            None,
        )
        if row is not None:
            cal_asr[attack] = float(row.get("attack_success_rate") or 0.0)
    zero_asr = [a for a in attacks if cal_asr.get(a, 0.0) <= 0.0]
    if zero_asr:
        lines.append("### 3.1 攻击有效性检查")
        lines.append("")
        for attack in zero_asr:
            vrow = (
                validation.get("by_mode", {})
                .get(primary_mode, {})
                .get("by_attack", {})
                .get(attack)
            )
            n_ok = vrow.get("n_adv_success") if vrow else None
            n_bad = vrow.get("n_adv_fail") if vrow else None
            val_asr = (
                100.0 * n_ok / (n_ok + n_bad)
                if n_ok is not None and n_bad is not None and (n_ok + n_bad) > 0
                else None
            )
            lines.append(
                f"- {attack_label(attack)} 在 Session 1 上的攻击成功率为 0.00%，"
                "说明该设置下扰动未能改变任何样本的预测；其 Session 1 校准指标不能"
                "代表对成功攻击的检测能力。"
                f"同一攻击在 Session 2 上的成功率为 {_fmt_pct(val_asr)}，"
                "应以 Session 2 结果为准。"
            )
        valid = [a for a in attacks if cal_asr.get(a, 0.0) > 0.0]
        if valid:
            radius_aucs = {}
            for entry in selection["by_mode"][primary_mode].get("by_radius", []):
                aucs = [
                    entry["per_attack_auc"].get(a) for a in valid
                ]
                aucs = [float(x) for x in aucs if x is not None]
                if aucs:
                    radius_aucs[float(entry["radius"])] = float(np.mean(aucs))
            if radius_aucs:
                robust_radius = max(
                    radius_aucs, key=lambda r: (radius_aucs[r], -float(r))
                )
                if int(robust_radius) == int(best_radius[primary_mode]):
                    lines.append(
                        "排除上述无效攻击后，按剩余攻击重新计算的平均 AUC 最优半径仍为 "
                        f"r={int(robust_radius)}（平均 AUC={_fmt(radius_aucs[robust_radius])}）；"
                        "因此半径选择不依赖于该无效攻击。"
                    )
                else:
                    lines.append(
                        "排除上述无效攻击后，平均 AUC 最优半径变为 "
                        f"r={int(robust_radius)}（平均 AUC={_fmt(radius_aucs[robust_radius])}）；"
                        "建议结合这一结果重新审视半径选择。"
                    )
        margin_stats = meta.get("clean_margin_stats") or {}
        if zero_asr and margin_stats:
            s1 = margin_stats.get(cal) or {}
            s2 = margin_stats.get(val) or {}
            if s1 and s2:
                lines.append(
                    "- 诊断：Session 1 干净样本的 top1−top2 ArcFace logit margin 均值为 "
                    f"{_fmt(s1.get('mean'), 2)}（最小值 {_fmt(s1.get('min'), 2)}），"
                    f"Session 2 为 {_fmt(s2.get('mean'), 2)}（最小值 "
                    f"{_fmt(s2.get('min'), 2)}）。Session 1 的干净 CE 更饱和，"
                    "梯度攻击的相对有效步长更小；这与该攻击在 Session 1 上失效、"
                    "在 Session 2 上有效的现象一致（诊断性解释，不是因果证明）。"
                )
        lines.append("")

    # Main table at selected radius
    lines.append("## 4. 选定半径下的主要结果")
    lines.append("")
    lines.append(
        "下表同时给出 Session 1（校准）与 Session 2（用 Session 1 阈值独立验证）的结果。"
        "ΔC = mean(C_clean) − mean(C_adv)；AUC 为干净 vs 对抗的检测 AUC；"
        "TPR@1%FPR 使用校准集阈值。"
    )
    lines.append("")
    for mode in modes:
        r = best_radius[mode]
        lines.append(f"### {mode_label(mode)}（r = {r}）")
        lines.append("")
        lines.append(
            "| Session | 攻击 | C_clean | C_adv | ΔC | 配对 Δ 95% CI | 配对 p | AUC | TPR@1%FPR | Cohen's d | KS | 攻击成功率 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for session in sessions:
            for attack in attacks:
                rows = _records_for(records, session, attack, eps_tag_value, mode)
                row = next((x for x in rows if int(x["filter_radius"]) == int(r)), None)
                if row is None:
                    continue
                ci = (
                    f"[{_fmt(row.get('paired_delta_ci95_low'))}, {_fmt(row.get('paired_delta_ci95_high'))}]"
                    if row.get("paired_delta_ci95_low") is not None
                    else "—"
                )
                p = row.get("paired_t_pvalue")
                if p is None:
                    p = row.get("paired_wilcoxon_pvalue")
                lines.append(
                    f"| {session} | {attack_label(attack)} | "
                    f"{_fmt(row.get('clean_cos_mean'))} ± {_fmt(row.get('clean_cos_std'))} | "
                    f"{_fmt(row.get('adv_cos_mean'))} ± {_fmt(row.get('adv_cos_std'))} | "
                    f"{_fmt(row.get('delta_c'))} | {ci} | {_fmt_p(p)} | "
                    f"{_fmt(row.get('auc'))} | {_fmt_pct(100.0 * row['tpr_at_fpr_1pct_insession'] if row.get('tpr_at_fpr_1pct_insession') is not None else None)} | "
                    f"{_fmt(row.get('cohens_d_scores'))} | {_fmt(row.get('ks_statistic'))} | "
                    f"{_fmt_pct(row.get('attack_success_rate'))} |"
                )
        lines.append("")
    lines.append(
        "注：上表 TPR 使用“同一 session 干净分数”校准的阈值（探索性）。"
        "严格的独立验证见下一节：Session 2 使用 Session 1 的阈值。"
    )
    lines.append("")
    lines.append(
        "**重要区分：** AUC 是尺度无关的排序指标，与 ΔC 的绝对大小并不等价。"
        "当组内方差很小时，即使 ΔC 很小也可能得到高 AUC；因此不能把高 AUC 直接"
        "解释为“频谱一致性差异很大”。本报告同时给出 ΔC（效应量）与 AUC（可分性）。"
    )
    lines.append("")

    # Validation
    lines.append("## 5. 独立验证（Session 2，固定 Session 1 的 r 与阈值）")
    lines.append("")
    lines.append(
        "`TPR@S1阈值` 是使用 Session 1 校准阈值的固定操作点；`验证FPR` 是同一阈值在"
        " Session 2 干净样本上的实际 FPR；`同session重校准TPR@1%` 是在 Session 2 干净"
        "样本上重新取 99% 分位数的 oracle 操作点（仅用于诊断阈值迁移，不是独立验证结果）。"
    )
    lines.append("")
    lines.append(
        "| 模式 | 攻击 | AUC | PR-AUC | TPR@S1阈值 | 验证FPR | 同session重校准TPR@1% | Cohen's d | KS | 成功数/失败数 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for mode in modes:
        entry = validation.get("by_mode", {}).get(mode, {})
        radius = float(entry.get("best_radius", 0))
        for attack in attacks:
            row = entry.get("by_attack", {}).get(attack)
            if not row:
                continue
            oracle_rows = _records_for(records, val, attack, eps_tag_value, mode)
            oracle = next(
                (x for x in oracle_rows if int(x["filter_radius"]) == int(radius)),
                None,
            )
            lines.append(
                f"| {mode_label(mode)} | {attack_label(attack)} | {_fmt(row.get('auc'))} | "
                f"{_fmt(row.get('pr_auc'))} | {_fmt_pct(100.0 * row['tpr'] if row.get('tpr') is not None else None)} | "
                f"{_fmt_pct(100.0 * row['fpr'] if row.get('fpr') is not None else None)} | "
                f"{_fmt_pct(100.0 * oracle['tpr_at_fpr_1pct_insession'] if oracle and oracle.get('tpr_at_fpr_1pct_insession') is not None else None)} | "
                f"{_fmt(row.get('cohens_d'))} | {_fmt(row.get('ks_statistic'))} | "
                f"{row.get('n_adv_success', '—')} / {row.get('n_adv_fail', '—')} |"
            )
    lines.append("")
    lines.append("### 攻击成功 / 失败样本的检测结果")
    lines.append("")
    lines.append("| 模式 | 攻击 | AUC（全部） | AUC（成功） | AUC（失败） | TPR@1%（成功） | TPR@1%（失败） |")
    lines.append("|---|---|---|---|---|---|---|")
    for mode in modes:
        entry = validation.get("by_mode", {}).get(mode, {})
        for attack in attacks:
            row = entry.get("by_attack", {}).get(attack)
            if not row:
                continue
            lines.append(
                f"| {mode_label(mode)} | {attack_label(attack)} | {_fmt(row.get('auc'))} | "
                f"{_fmt(row.get('auc_success'))} | {_fmt(row.get('auc_fail'))} | "
                f"{_fmt_pct(100.0 * row['tpr_success'] if row.get('tpr_success') is not None else None)} | "
                f"{_fmt_pct(100.0 * row['tpr_fail'] if row.get('tpr_fail') is not None else None)} |"
            )
    lines.append("")

    # Fixed-threshold transfer across radii
    if radius_transfer:
        lines.append("### 5.1 阈值迁移随半径的变化")
        lines.append("")
        lines.append(
            "每个半径先在 Session 1 干净样本上校准 FPR 阈值，再在 Session 2 上评估。"
            "AUC 与阈值无关；FPR/TPR 反映固定操作点能否迁移。"
        )
        lines.append("")
        lines.append("| 模式 | r | 验证 AUC 均值 | 固定阈值验证 FPR | 固定阈值 TPR 均值 | 阈值 |")
        lines.append("|---|---|---|---|---|---|")
        for mode in modes:
            for entry in radius_transfer.get(mode, []):
                lines.append(
                    f"| {mode_label(mode)} | {int(entry['radius'])} | "
                    f"{_fmt(entry.get('validation_auc_mean'))} | "
                    f"{_fmt_pct(100.0 * entry['validation_fpr'] if entry.get('validation_fpr') is not None else None)} | "
                    f"{_fmt_pct(100.0 * entry['validation_tpr_mean'] if entry.get('validation_tpr_mean') is not None else None)} | "
                    f"{_fmt(entry.get('threshold'), 6)} |"
                )
        lines.append("")
        lines.append(
            "该表使用 Session 2，属于**诊断性分析**；不能反过来用 Session 2 的 FPR "
            "选择半径，否则会破坏独立验证。若以 FPR 迁移为首要目标，应只在 Session 1 "
            "内部做嵌套校准/验证，或采用目标域无标签校准。"
        )
        lines.append("")

    # Attack comparison
    lines.append("## 6. 不同攻击之间的一致性变化是否相似")
    lines.append("")
    for mode in modes:
        comp = attack_comparison.get(mode)
        if not comp:
            continue
        r = best_radius[mode]
        lines.append(
            f"- **{mode_label(mode)}（r={r}, Session 1）**：Friedman 检验 "
            f"χ²={_fmt(comp.get('friedman_statistic'))}, p={_fmt_p(comp.get('friedman_pvalue'))}。"
        )
        pairwise = comp.get("pairwise", [])
        if not pairwise:
            lines.append("  - 仅一种攻击，无法进行攻击间两两比较。")
            continue
        if comp.get("friedman_pvalue") is not None and comp["friedman_pvalue"] < 0.05:
            lines.append(
                "  - Friedman 检验拒绝“各攻击 ΔC 分布相同”的原假设，说明不同攻击的"
                "一致性下降幅度存在统计显著差异。"
            )
        elif comp.get("friedman_pvalue") is not None:
            lines.append(
                "  - Friedman 检验未拒绝原假设，说明在所测攻击之间，ΔC 的分布形态大体相似。"
            )
        significant = [
            p for p in pairwise
            if p.get("wilcoxon_pvalue_holm") is not None and p["wilcoxon_pvalue_holm"] < 0.05
        ]
        if significant:
            pairs = ", ".join(
                f"{attack_label(p['attack_a'])} vs {attack_label(p['attack_b'])}"
                f" (Holm p={_fmt_p(p['wilcoxon_pvalue_holm'])})"
                for p in significant
            )
            lines.append(f"  - Holm 校正后显著的攻击对：{pairs}。")
        else:
            lines.append("  - 所有攻击两两之间在 Holm 校正后均无显著差异。")
    if zero_asr:
        lines.append(
            "注意：上述 Friedman/两两检验包含在 Session 1 上 ASR=0% 的攻击；"
            "其 ΔC 分布较弱，可能影响检验结果。攻击间的实用比较应以 Session 2 "
            "验证 AUC 为准（见 Q2 与 3.1 节）。"
        )
    lines.append("")

    # Classification
    lines.append("## 7. 低通滤波对分类准确率的影响")
    lines.append("")
    lines.append(
        "下表为选定半径下 Session 2 的 Top-1 / Top-5 准确率。"
        "`Clean-LPF` 衡量滤波本身是否破坏干净掌纹的身份信息。"
    )
    lines.append("")
    lines.append("| 模式 | 攻击 | Clean | Clean-LPF | Adv | Adv-LPF |")
    lines.append("|---|---|---|---|---|---|")
    for mode in modes:
        r = best_radius[mode]
        for attack in attacks:
            rows = _records_for(records, val, attack, eps_tag_value, mode)
            row = next((x for x in rows if int(x["filter_radius"]) == int(r)), None)
            if row is None:
                continue
            lines.append(
                f"| {mode_label(mode)} | {attack_label(attack)} | "
                f"{_fmt_pct(row.get('clean_top1'))} / {_fmt_pct(row.get('clean_top5'))} | "
                f"{_fmt_pct(row.get('clean_lpf_top1'))} / {_fmt_pct(row.get('clean_lpf_top5'))} | "
                f"{_fmt_pct(row.get('adv_top1'))} / {_fmt_pct(row.get('adv_top5'))} | "
                f"{_fmt_pct(row.get('adv_lpf_top1'))} / {_fmt_pct(row.get('adv_lpf_top5'))} |"
            )
    lines.append("")
    clean_row = _clean_record(
        records, val, eps_tag_value, primary_mode, best_radius[primary_mode]
    )
    if clean_row is not None:
        drop = clean_row.get("clean_top1") - clean_row.get("clean_lpf_top1", 0.0)
        lines.append(
            f"在 {mode_label(primary_mode)}、r={best_radius[primary_mode]} 下，"
            f"干净样本 Top-1 从 {_fmt_pct(clean_row.get('clean_top1'))} 变为 "
            f"{_fmt_pct(clean_row.get('clean_lpf_top1'))}（下降 {_fmt(drop, 3)} 个百分点）。"
        )
    lines.append("")

    # Figures
    lines.append("## 8. 图表")
    lines.append("")
    for key, rel in figures.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append(f"![{key}]({rel})")
        lines.append("")

    # Answers
    lines.append("## 9. 科研问题回答")
    lines.append("")
    lines.extend(_answers(selection, validation, records, attack_comparison, roi_stats,
                          meta, primary_eps, eps_tag_value, attacks, modes, best_radius,
                          thresholds, primary_mode, cal, val, fpr_target,
                          radius_transfer))

    lines.append("## 10. 局限与后续建议")
    lines.append("")
    lines.append(
        "- 本实验是检测可行性预实验，没有训练任何检测器；AUC/TPR 仅说明单次观测分数的"
        "可分性，实际部署还需要在独立攻击、更多 session 和自适应攻击下验证。"
    )
    n_cal = int(meta.get("sample_n") or 0)
    n_fp = int(round(float(fpr_target) * n_cal)) if n_cal else 0
    lines.append(
        f"- 阈值仅由有限数量的干净样本校准（每 session {n_cal} 张），"
        f"FPR={fpr_target:.0%} 对应约 {n_fp} 个假阳性，阈值本身存在抽样误差。"
    )
    lines.append(
        "- 固定阈值从 Session 1 迁移到 Session 2 时可能出现明显的 FPR 漂移（见第 5 节"
        "的验证 FPR 列）；这说明部署时需要目标域校准、分布无关的 p-value，或把检测分数"
        "与其他信号联合使用，而不能直接沿用源域阈值。"
    )
    lines.append(
        "- 攻击均为像素域 L∞（除另行指定的 Patch）；攻击者可以针对低通滤波设计"
        "扰动，因此本结果不能直接外推为对自适应攻击的鲁棒性。"
    )
    lines.append(
        "- 结论仅适用于 CompNet + Tongji ROI；跨数据集、跨模型的可迁移性需要额外实验。"
    )
    lines.append("")
    lines.append("## 11. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(meta.get("reproduce_command", ""))
    lines.append("```")
    lines.append("")

    # Appendix: supplementary epsilons, using the primary-epsilon radius and
    # threshold (a fixed-parameter sensitivity check, not a re-selection).
    primary_tag_value = meta.get("primary_epsilon_tag")
    by_epsilon = validation.get("by_epsilon", {})
    extra_tags = [tag for tag in by_epsilon if tag != primary_tag_value]
    if extra_tags:
        lines.append("## 附录 A：其他 epsilon 的独立验证")
        lines.append("")
        lines.append(
            "半径与阈值仍固定在主 epsilon（"
            f"{_fmt(float(meta.get('primary_epsilon', 0.0)) * 255.0, 1)}/255）"
            "选出的值上；本表用于观察扰动强度变化时的检测趋势。"
        )
        lines.append("")
        lines.append("| 模式 | epsilon | 攻击 | AUC | TPR@S1阈值 | 验证集 FPR | 攻击成功率 |")
        lines.append("|---|---|---|---|---|---|---|")
        for tag in extra_tags:
            eps_value = float(tag)
            eps_display = f"{eps_value:.6f} ({eps_value * 255.0:.0f}/255)"
            entry = by_epsilon[tag]
            for mode in modes:
                mode_entry = entry.get("by_mode", {}).get(mode)
                if not mode_entry:
                    continue
                for attack in attacks:
                    row = mode_entry.get("by_attack", {}).get(attack)
                    if not row:
                        continue
                    n_ok = row.get("n_adv_success") or 0
                    n_bad = row.get("n_adv_fail") or 0
                    success_rate = (
                        100.0 * n_ok / (n_ok + n_bad) if (n_ok + n_bad) else None
                    )
                    lines.append(
                        f"| {mode_label(mode)} | {eps_display} | {attack_label(attack)} | "
                        f"{_fmt(row.get('auc'))} | "
                        f"{_fmt_pct(100.0 * row['tpr'] if row.get('tpr') is not None else None)} | "
                        f"{_fmt_pct(100.0 * row['fpr'] if row.get('fpr') is not None else None)} | "
                        f"{_fmt_pct(success_rate)} |"
                    )
        lines.append("")
        lines.append(
            "若 AUC 随 epsilon 减小而下降，说明弱扰动下的频谱一致性差异更弱；"
            "若 AUC 基本稳定，则说明该指标对扰动强度不敏感。"
        )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _answers(
    selection: Dict,
    validation: Dict,
    records: Sequence[Dict],
    attack_comparison: Dict[str, Dict],
    roi_stats: Dict,
    meta: Dict,
    primary_eps: float,
    eps_tag_value: str,
    attacks: Sequence[str],
    modes: Sequence[str],
    best_radius: Dict[str, int],
    thresholds: Dict[str, float],
    primary_mode: str,
    cal: str,
    val: str,
    fpr_target: float = 0.01,
    radius_transfer: Optional[Dict[str, List[Dict[str, Optional[float]]]]] = None,
) -> List[str]:
    """Generate answers to the eight research questions from the results."""
    lines: List[str] = []
    cal_rows = {}
    val_rows = {}
    for mode in modes:
        r = best_radius[mode]
        for attack in attacks:
            rows = _records_for(records, cal, attack, eps_tag_value, mode)
            cal_rows[(mode, attack)] = next(
                (x for x in rows if int(x["filter_radius"]) == int(r)), None
            )
            vrows = validation.get("by_mode", {}).get(mode, {}).get("by_attack", {})
            val_rows[(mode, attack)] = vrows.get(attack)

    # Q1
    supported = []
    partial = []
    unsupported = []
    for attack in attacks:
        row = cal_rows.get((primary_mode, attack))
        vrow = val_rows.get((primary_mode, attack))
        if row is None:
            continue
        delta_positive = row.get("delta_c") is not None and row["delta_c"] > 0
        p = row.get("paired_t_pvalue")
        auc = vrow.get("auc") if vrow else row.get("auc")
        if delta_positive and p is not None and p < 0.05 and auc is not None and auc >= 0.6:
            supported.append(attack)
        elif delta_positive and auc is not None and auc > 0.5:
            partial.append(attack)
        else:
            unsupported.append(attack)
    lines.append("### Q1. CompNet 是否具有与 CLIP 类似的频谱一致性差异？")
    lines.append("")
    if len(supported) == len(attacks) and attacks:
        lines.append(
            f"**支持。** 在 {mode_label(primary_mode)}、r={best_radius[primary_mode]} 下，"
            f"全部 {len(attacks)} 种攻击都满足 ΔC > 0、配对检验 p < 0.05，且 Session 2 "
            f"独立验证 AUC ≥ 0.6。"
        )
    elif supported:
        lines.append(
            f"**部分支持。** {', '.join(attack_label(a) for a in supported)} 满足 ΔC > 0、"
            f"配对检验显著且验证 AUC ≥ 0.6；"
            + (f"{', '.join(attack_label(a) for a in partial)} 方向一致但证据较弱；" if partial else "")
            + (f"{', '.join(attack_label(a) for a in unsupported)} 不支持该假设。" if unsupported else "")
        )
    elif partial:
        lines.append(
            f"**证据有限。** {', '.join(attack_label(a) for a in partial)} 的 ΔC 方向一致，"
            "但未同时满足显著性与 AUC 门槛；其余攻击不支持该假设。"
        )
    else:
        lines.append(
            "**不支持。** 在所测攻击中，干净样本的低通特征一致性没有稳定地高于对抗样本"
            "（ΔC 未稳定为正，或验证 AUC 未超过随机水平）。"
        )
    lines.append(
        "注意：这里的“支持/部分支持”指统计显著且独立验证 AUC 达到门槛，不等同于效应量"
        "很大；应结合 ΔC、Cohen's d 与原始分布重叠程度一起解读。"
    )
    zero_cal = [
        a for a in attacks
        if (cal_rows.get((primary_mode, a)) or {}).get("attack_success_rate", 1.0) <= 0.0
    ]
    if zero_cal:
        lines.append(
            "此外，"
            + "、".join(attack_label(a) for a in zero_cal)
            + " 在 Session 1 上的攻击成功率为 0%，其校准 ΔC 不代表对成功攻击的检测；"
            "Q1 的判断主要依赖 Session 2 的独立验证结果（见 3.1 节）。"
        )
    lines.append("")

    # Q2
    lines.append("### Q2. 哪些攻击方法具有明显的特征一致性变化？")
    lines.append("")
    ranking = []
    for attack in attacks:
        row = cal_rows.get((primary_mode, attack))
        vrow = val_rows.get((primary_mode, attack))
        if row is None:
            continue
        delta = row.get("delta_c")
        auc = vrow.get("auc") if vrow else None
        ranking.append((
            float(delta) if delta is not None else -np.inf,
            attack,
            auc,
        ))
    ranking.sort(
        key=lambda item: (
            item[0],
            item[2] if item[2] is not None else -np.inf,
        ),
        reverse=True,
    )
    lines.append(
        "按 Session 1 的 ΔC 排序："
        + "; ".join(
            f"{attack_label(a)}（ΔC={_fmt(d)}, 验证 AUC={_fmt(au)}）"
            for d, a, au in ranking
        )
        + "。"
    )
    auc_ranking = []
    for attack in attacks:
        vrow = val_rows.get((primary_mode, attack))
        if vrow and vrow.get("auc") is not None:
            auc_ranking.append((float(vrow["auc"]), attack))
    auc_ranking.sort(reverse=True)
    if auc_ranking:
        lines.append(
            "按 Session 2 独立验证 AUC 排序："
            + "; ".join(
                f"{attack_label(a)}（AUC={_fmt(v)}）" for v, a in auc_ranking
            )
            + "。"
        )
    zero_cal_q2 = [
        a for a in attacks
        if (cal_rows.get((primary_mode, a)) or {}).get("attack_success_rate", 1.0) <= 0.0
    ]
    if zero_cal_q2:
        lines.append(
            "其中 "
            + "、".join(attack_label(a) for a in zero_cal_q2)
            + " 的 Session 1 ΔC 来自 ASR=0% 的弱攻击，其 ΔC 排序不具解释力；"
            "攻击间的比较应以 Session 2 验证 AUC 为准。"
        )
    comp = attack_comparison.get(primary_mode, {})
    if comp.get("friedman_pvalue") is not None:
        if comp["friedman_pvalue"] < 0.05:
            lines.append(
                f"Friedman 检验 p={_fmt_p(comp['friedman_pvalue'])}，说明各攻击的 ΔC 分布"
                "并非完全相同；具体差异见第 6 节的两两比较。"
            )
        else:
            lines.append(
                f"Friedman 检验 p={_fmt_p(comp['friedman_pvalue'])}，未发现各攻击 ΔC 分布"
                "存在显著差异，即变化规律具有相似性。"
            )
    if ranking and all(d > 0 for d, _, _ in ranking):
        lines.append(
            "所有被评估攻击的 ΔC 均为正，说明变化方向一致（干净样本的低通一致性更高）；"
            "攻击间差异主要体现在幅度与可分性上。"
        )
    lines.append("")

    # Q3
    lines.append("### Q3. 哪些攻击方法无法通过当前指标有效区分？")
    lines.append("")
    weak = []
    for attack in attacks:
        vrow = val_rows.get((primary_mode, attack))
        if not vrow:
            continue
        auc = vrow.get("auc")
        tpr = vrow.get("tpr")
        if (auc is not None and auc < 0.6) or (tpr is not None and tpr < 0.1):
            weak.append((attack, auc, tpr))
    if weak:
        lines.append(
            "以下攻击在独立验证中较弱（AUC < 0.6 或 TPR@1%FPR < 10%）："
            + "; ".join(
                f"{attack_label(a)}（AUC={_fmt(au)}, TPR@1%={_fmt_pct(100.0 * tp if tp is not None else None)}）"
                for a, au, tp in weak
            )
            + "。"
        )
    else:
        lines.append(
            "在本次测试的攻击与 epsilon 下，没有攻击同时落到 AUC < 0.6 与 TPR@1%FPR < 10% "
            "的弱检测区间；但这不等于对自适应攻击也有效。"
        )
    auc_pairs = []
    for attack in attacks:
        vrow = val_rows.get((primary_mode, attack))
        if vrow and vrow.get("auc") is not None:
            auc_pairs.append((float(vrow["auc"]), attack, vrow))
    if auc_pairs:
        weakest_auc, weakest_attack, weakest_row = min(
            auc_pairs, key=lambda x: x[0]
        )
        lines.append(
            f"相对而言，验证 AUC 最低的是 {attack_label(weakest_attack)}"
            f"（AUC={_fmt(weakest_auc)}）。"
        )
        if weakest_row.get("auc_fail") is not None:
            lines.append(
                f"其攻击失败样本的 AUC 为 {_fmt(weakest_row.get('auc_fail'))}"
                f"（固定阈值 TPR={_fmt_pct(100.0 * weakest_row['tpr_fail'] if weakest_row.get('tpr_fail') is not None else None)}）；"
                "若接近 0.5，说明失败攻击基本无法由该指标识别。"
            )
    lines.append("")

    # Q4
    lines.append("### Q4. 不同滤波半径对干净样本和对抗样本的影响是什么？")
    lines.append("")
    for mode in modes:
        clean_rows = _records_for(records, cal, "clean", eps_tag_value, mode)
        if clean_rows:
            rx = [r["filter_radius"] for r in clean_rows]
            cy = [r["clean_cos_mean"] for r in clean_rows]
            rho_c, p_c = _spearman(rx, cy)
            clean_rows_val = _records_for(records, val, "clean", eps_tag_value, mode)
            cyv = [r["clean_cos_mean"] for r in clean_rows_val]
            rho_cv, _ = _spearman(rx, cyv)
            lines.append(
                f"- {mode_label(mode)}：干净样本的 C_clean 随 r 的 Spearman ρ = "
                f"{_fmt(rho_c, 3)}（p={_fmt_p(p_c)}，Session 1），Session 2 为 ρ = {_fmt(rho_cv, 3)}。"
            )
        for attack in attacks:
            rows = _records_for(records, cal, attack, eps_tag_value, mode)
            if not rows:
                continue
            rx = [r["filter_radius"] for r in rows]
            ay = [r["adv_cos_mean"] for r in rows]
            dy = [r["delta_c"] for r in rows]
            rho_a, _ = _spearman(rx, ay)
            rho_d, _ = _spearman(rx, dy)
            lines.append(
                f"  - {attack_label(attack)}：C_adv 与 r 的 ρ = {_fmt(rho_a, 3)}，"
                f"ΔC 与 r 的 ρ = {_fmt(rho_d, 3)}。"
            )
    all_radii = sorted({float(r["filter_radius"]) for r in records})
    peaks = []
    for mode in modes:
        for attack in attacks:
            rows = _records_for(records, cal, attack, eps_tag_value, mode)
            if not rows:
                continue
            best = max(rows, key=lambda x: float(x["delta_c"]))
            peaks.append(
                (mode, attack, float(best["filter_radius"]), float(best["delta_c"]))
            )
    if peaks:
        lines.append(
            "Session 1 中各攻击的 ΔC 峰值："
            + "; ".join(
                f"{mode_label(m)}/{attack_label(a)} r={int(r)} (ΔC={_fmt(d)})"
                for m, a, r, d in peaks
            )
            + "。"
        )
        interior = [
            p for p in peaks
            if all_radii and p[2] not in (all_radii[0], all_radii[-1])
        ]
        if interior:
            lines.append(
                "部分攻击在测试半径区间的内部达到峰值，说明在这些攻击上存在一个"
                "可分性相对更优的中间半径。"
            )
        else:
            lines.append(
                "所有攻击的 ΔC 峰值都位于测试半径区间的端点；在本次测试范围内没有"
                "观察到内部峰值，ΔC 随 r 单调变化。"
            )
    lines.append("")

    # Q5
    lines.append("### Q5. 是否存在适用于 CompNet 的最佳滤波半径？")
    lines.append("")
    all_radii_q5 = sorted({float(r["filter_radius"]) for r in records})
    validation_mean_aucs: List[float] = []
    for mode in modes:
        entry = selection["by_mode"].get(mode, {})
        vrows = validation.get("by_mode", {}).get(mode, {})
        v_aucs = [vrows.get("by_attack", {}).get(a, {}).get("auc") for a in attacks]
        v_aucs = [x for x in v_aucs if x is not None]
        mean_v = float(np.mean(v_aucs)) if v_aucs else None
        if mean_v is not None:
            validation_mean_aucs.append(mean_v)
        lines.append(
            f"- {mode_label(mode)}：Session 1 选择 r={entry.get('best_radius')}，"
            f"校准平均 AUC={_fmt(entry.get('mean_auc'))}；Session 2 独立验证平均 AUC="
            f"{_fmt(mean_v)}。"
        )
        if all_radii_q5 and int(entry.get("best_radius", -1)) in (
            int(all_radii_q5[0]), int(all_radii_q5[-1])
        ):
            lines.append(
                "  - 该半径位于本次测试网格的边界，不能排除网格外（"
                f"r<{int(all_radii_q5[0])} 或 r>{int(all_radii_q5[-1])}）存在更优值。"
            )
        by_radius = entry.get("by_radius", [])
        delta_entries = [
            e for e in by_radius if e.get("mean_delta_c") is not None
        ]
        if delta_entries:
            delta_best = max(delta_entries, key=lambda e: float(e["mean_delta_c"]))
            same = int(delta_best["radius"]) == int(entry.get("best_radius", -1))
            lines.append(
                f"  - 若改用“平均 ΔC 最大化”而不是 AUC，最优半径为 "
                f"r={int(delta_best['radius'])}（平均 ΔC={_fmt(delta_best['mean_delta_c'])}）"
                + ("，与 AUC 选出的半径一致。" if same else
                   "，与 AUC 选出的半径不同；这说明效应量与可分性是两个不同的优化目标。")
            )
        if radius_transfer:
            compliant = [
                e for e in radius_transfer.get(mode, [])
                if e.get("validation_fpr") is not None
                and e["validation_fpr"] <= 2.0 * fpr_target
                and e.get("validation_auc_mean") is not None
            ]
            if compliant:
                best_ok = max(
                    compliant, key=lambda e: float(e["validation_auc_mean"])
                )
                lines.append(
                    f"  - 若要求 Session 2 的固定阈值 FPR ≤ {2.0 * fpr_target:.0%}，"
                    f"满足条件且验证 AUC 最高的半径是 r={int(best_ok['radius'])}"
                    f"（AUC={_fmt(best_ok['validation_auc_mean'])}, "
                    f"FPR={_fmt_pct(100.0 * best_ok['validation_fpr'])}）。"
                )
            else:
                lines.append(
                    f"  - 在本次测试的半径中，没有半径能把 Session 2 的固定阈值 FPR "
                    f"控制在 {2.0 * fpr_target:.0%} 以内。"
                )
    if not validation_mean_aucs or max(validation_mean_aucs) < 0.6:
        lines.append(
            "验证集平均 AUC 未达到 0.6，因此在本实验条件下不能确认存在有实用价值的"
            "“最佳半径”；Session 1 的 argmax 可能只是噪声。"
        )
    lines.append(
        "注意：本节的选择只优化阈值无关的 AUC；固定阈值在 Session 2 上的实际 FPR "
        "见第 5 节。如果阈值迁移失败，则“最佳半径”的实用性需要结合阈值校准策略"
        "重新评估。上述 FPR 约束下的半径比较同样来自 Session 2 的诊断性分析，"
        "只用于说明 trade-off，不构成独立验证中的选择。"
    )
    lines.append("")

    # Q6
    lines.append("### Q6. 低通滤波是否破坏了干净掌纹的身份判别特征？")
    lines.append("")
    primary_row = _clean_record(
        records, val, eps_tag_value, primary_mode, best_radius[primary_mode]
    )
    if primary_row is not None:
        r = best_radius[primary_mode]
        drop1 = primary_row["clean_top1"] - primary_row["clean_lpf_top1"]
        drop5 = primary_row["clean_top5"] - primary_row["clean_lpf_top5"]
        drop1_text = (
            f"下降 {drop1:.3f} pp" if drop1 >= 0 else f"上升 {-drop1:.3f} pp"
        )
        drop5_text = (
            f"下降 {drop5:.3f} pp" if drop5 >= 0 else f"上升 {-drop5:.3f} pp"
        )
        lines.append(
            f"在 {mode_label(primary_mode)}、r={r} 下，Session 2 干净样本 Top-1 从 "
            f"{_fmt_pct(primary_row['clean_top1'])} 变为 {_fmt_pct(primary_row['clean_lpf_top1'])}"
            f"（{drop1_text}），Top-5 从 {_fmt_pct(primary_row['clean_top5'])} 变为 "
            f"{_fmt_pct(primary_row['clean_lpf_top5'])}（{drop5_text}）。"
        )
        if max(abs(drop1), abs(drop5)) <= 0.5:
            lines.append("可以认为该半径下的低通滤波基本没有破坏干净掌纹的身份判别信息。")
        elif max(abs(drop1), abs(drop5)) <= 2.0:
            lines.append("低通滤波对身份判别信息有轻微影响，但仍处于可接受范围。")
        else:
            lines.append(
                "低通滤波明显损害了干净掌纹的分类性能，该半径不宜直接用于后续防御。"
            )
    all_radii_q6 = sorted({float(x["filter_radius"]) for x in records})
    if all_radii_q6:
        small_r = float(all_radii_q6[0])
        if primary_row is not None and int(small_r) != int(best_radius[primary_mode]):
            row_small = _clean_record(
                records, val, eps_tag_value, primary_mode, small_r
            )
            if row_small is not None:
                drop_small = (
                    row_small["clean_top1"] - row_small["clean_lpf_top1"]
                )
                drop_small_text = (
                    f"下降 {drop_small:.3f} pp" if drop_small >= 0
                    else f"上升 {-drop_small:.3f} pp"
                )
                if abs(drop_small) <= 0.5:
                    assessment = "基本无损"
                elif abs(drop_small) <= 2.0:
                    assessment = "损失有限"
                else:
                    assessment = "损失明显"
                lines.append(
                    f"作为对比，在 ΔC 最大的 r={int(small_r)} 下，Clean-LPF Top-1 为 "
                    f"{_fmt_pct(row_small['clean_lpf_top1'])}（{drop_small_text}），"
                    f"身份信息{assessment}。"
                )
    lines.append("")

    # Q7
    lines.append("### Q7. 频谱一致性指标能否作为独立的对抗样本检测依据？")
    lines.append("")
    if val_rows:
        rows = [val_rows.get((primary_mode, attack)) for attack in attacks]
        rows = [v for v in rows if v]
        aucs = [v.get("auc") for v in rows if v.get("auc") is not None]
        tprs = [v.get("tpr") for v in rows if v.get("tpr") is not None]
        fprs = [v.get("fpr") for v in rows if v.get("fpr") is not None]
        mean_auc = float(np.mean(aucs)) if aucs else None
        mean_tpr = float(np.mean(tprs)) if tprs else None
        mean_fpr = float(np.mean(fprs)) if fprs else None
        oracle_by_attack = {}
        for attack in attacks:
            oracle_rows = _records_for(records, val, attack, eps_tag_value, primary_mode)
            oracle = next(
                (x for x in oracle_rows
                 if int(x["filter_radius"]) == int(best_radius[primary_mode])),
                None,
            )
            if oracle and oracle.get("tpr_at_fpr_1pct_insession") is not None:
                oracle_by_attack[attack] = float(
                    oracle["tpr_at_fpr_1pct_insession"]
                )
        oracle_tprs = list(oracle_by_attack.values())
        mean_oracle_tpr = float(np.mean(oracle_tprs)) if oracle_tprs else None
        lines.append(
            f"Session 2 独立验证的平均 AUC = {_fmt(mean_auc)}；使用 Session 1 固定阈值时，"
            f"平均 TPR = {_fmt_pct(100.0 * mean_tpr if mean_tpr is not None else None)}，"
            f"但平均验证集 FPR = {_fmt_pct(100.0 * mean_fpr if mean_fpr is not None else None)}"
            f"（目标 {100.0 * fpr_target:.0f}%）。"
        )
        if mean_oracle_tpr is not None:
            lines.append(
                f"若在 Session 2 干净样本上重新校准 99% 分位数阈值，平均 TPR@1%FPR = "
                f"{_fmt_pct(100.0 * mean_oracle_tpr)}；这是 oracle 诊断值，不能作为独立"
                "验证结果。"
            )
            per_attack_text = "; ".join(
                f"{attack_label(a)}={_fmt_pct(100.0 * v)}"
                for a, v in oracle_by_attack.items()
            )
            lines.append(f"分攻击看，同 session 重校准后的 TPR@1%FPR 为：{per_attack_text}。")
            weakest_oracle = min(
                oracle_by_attack.items(), key=lambda item: item[1]
            )
            if weakest_oracle[1] < 0.3:
                lines.append(
                    f"其中 {attack_label(weakest_oracle[0])} 最低"
                    f"（{_fmt_pct(100.0 * weakest_oracle[1])}）；其 AUC 虽高，"
                    "但真正 1% FPR 操作点仍然较弱。"
                )
        if (
            mean_auc is not None and mean_auc >= 0.8
            and mean_tpr is not None and mean_tpr >= 0.5
            and mean_fpr is not None and mean_fpr <= 2.0 * fpr_target
        ):
            lines.append(
                "在本次实验范围内，AUC 与固定阈值下的 TPR/FPR 都满足可用条件，该指标"
                "可以作为独立的检测分数；但仍需针对自适应攻击和跨数据集验证后再部署。"
            )
        elif mean_auc is not None and mean_auc >= 0.6:
            if mean_fpr is not None and mean_fpr > 2.0 * fpr_target:
                clean_row_q7 = _clean_record(
                    records, val, eps_tag_value, primary_mode,
                    best_radius[primary_mode],
                )
                clean_std_q7 = (
                    clean_row_q7.get("clean_cos_std") if clean_row_q7 else None
                )
                lines.append(
                    "AUC 表明两类分数具有排序可分性，但固定阈值迁移到 Session 2 后 FPR "
                    "明显高于目标，说明**阈值校准/域偏移是当前的主要瓶颈**。"
                    + (
                        f"所选半径下干净 C 的标准差仅为 {_fmt(clean_std_q7, 5)}，"
                        "99% 分位数阈值位于分布极端尾部，微小的 session 漂移即可放大 FPR。"
                        if clean_std_q7 is not None else ""
                    )
                    + "因此该指标目前更适合作为多信号检测器中的辅助特征，或在目标 session "
                    "内用干净参考样本重新校准阈值后再评估，而不能直接作为固定阈值的独立检测器。"
                )
            else:
                lines.append(
                    "该指标具有一定的检测能力，但单独使用时 TPR/FPR 权衡仍不理想，更适合"
                    "作为多信号检测器中的一个辅助特征。"
                )
        else:
            lines.append(
                "该指标在独立验证中的可分性不足，不能单独作为对抗样本检测依据。"
            )
    lines.append("")

    # Q8
    lines.append("### Q8. 是否值得进一步开展基于 Contrastive Spectral Rectification 的掌纹测试时防御研究？")
    lines.append("")
    q1_supported = any(
        (cal_rows.get((primary_mode, a)) or {}).get("delta_c", 0) > 0
        and (cal_rows.get((primary_mode, a)) or {}).get("paired_t_pvalue") is not None
        and (cal_rows.get((primary_mode, a)) or {}).get("paired_t_pvalue") < 0.05
        and val_rows.get((primary_mode, a)) is not None
        and (val_rows.get((primary_mode, a)) or {}).get("auc", 0) >= 0.6
        for a in attacks
    )
    mean_v = None
    mean_fpr_v = None
    vrows = validation.get("by_mode", {}).get(primary_mode, {}).get("by_attack", {})
    if vrows:
        vals = [v.get("auc") for v in vrows.values() if v.get("auc") is not None]
        mean_v = float(np.mean(vals)) if vals else None
        fprs = [v.get("fpr") for v in vrows.values() if v.get("fpr") is not None]
        mean_fpr_v = float(np.mean(fprs)) if fprs else None
    threshold_ok = mean_fpr_v is not None and mean_fpr_v <= 2.0 * fpr_target
    if q1_supported and mean_v is not None and mean_v >= 0.7 and threshold_ok:
        lines.append(
            "**值得继续。** 频谱一致性差异在至少一种攻击上通过了校准与独立验证，"
            "平均 AUC 与固定阈值下的 FPR 都支持存在可用的检测信号。建议下一步："
            "（1）在 Session 1 上训练/校准轻量检测器并冻结；（2）加入自适应攻击"
            "（针对 LPF 的 EOT/高频惩罚）检验鲁棒性；（3）再考虑把频谱一致性作为 "
            "P-CSR 的辅助约束，而不是直接假设它足够。"
        )
    elif q1_supported and mean_v is not None and mean_v >= 0.7:
        lines.append(
            "**谨慎推进。** AUC 表明频谱一致性差异具有排序可分性，但固定阈值迁移到 "
            f"Session 2 后的平均 FPR 为 {_fmt_pct(100.0 * mean_fpr_v if mean_fpr_v is not None else None)}"
            f"（目标 {100.0 * fpr_target:.0f}%），说明域偏移下的阈值校准尚未解决。"
            "建议先把检测器限制为辅助模块，并在目标 session 内用干净参考样本重新校准"
            "阈值（或采用分布无关的 p-value）；在解决这个问题之前，不建议直接投入完整"
            "的 P-CSR 防御训练。"
        )
    elif q1_supported:
        lines.append(
            "**谨慎推进。** 存在方向正确但强度有限的信号。建议先把检测器限制为辅助模块，"
            "并在更多攻击/数据集上验证；不建议直接投入完整的 P-CSR 防御训练。"
        )
    else:
        lines.append(
            "**暂不建议直接投入完整 P-CSR。** 当前结果不足以证明频谱一致性差异可以作为"
            "可靠的检测/校正信号。建议先扩大攻击类型与 epsilon 范围、检查是否有攻击专门"
            "规避低通滤波，并考虑其他频域统计量（如径向功率谱、相位一致性）后再决定。"
        )
    lines.append("")
    return lines
