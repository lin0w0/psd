"""Reporting and plotting utilities for CompNet reproduction results."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .evaluator import strip_private
from .metrics import gi_histogram

_DISPLAY_NAMES = {
    "verification": "Verification (session1 gallery vs session2 probe)",
    "aggregated_verification": "Aggregated verification (per-class min)",
    "dataset_eer": "Dataset EER (all pairs in session2)",
    "rank1": "Rank-1 identification",
    "classification": "Closed-set classification (session2 probe)",
}


def _metric_rows(results: Dict) -> List[List[str]]:
    rows: List[List[str]] = []

    for key in ("verification", "aggregated_verification", "dataset_eer"):
        if key not in results:
            continue
        r = results[key]
        name = _DISPLAY_NAMES[key]
        rows.append([name, "EER (fitted, official getEER)", f"{r['eer_percent']:.6f}", "%"])
        rows.append([name, "EER (mean, eer_1/2)", f"{r['eer_mean_percent']:.6f}", "%"])
        rows.append([name, "AUC", f"{r['auc']:.8f}", ""])
        if "rank1_accuracy" in r:
            rows.append([name, "Rank-1", f"{r['rank1_accuracy']:.4f}", "%"])
        for far in (1e-2, 1e-3, 1e-4):
            k = f"tar_at_far_{far:g}"
            if k in r:
                rows.append([name, f"TAR @ FAR={far:g}", f"{r[k] * 100:.4f}", "%"])
        rows.append([name, "Genuine pairs", str(r["num_genuine"]), ""])
        rows.append([name, "Impostor pairs", str(r["num_impostor"]), ""])

    if "rank1" in results:
        r = results["rank1"]
        rows.append([
            _DISPLAY_NAMES["rank1"],
            "Rank-1 accuracy",
            f"{r['accuracy']:.4f}",
            "%",
        ])
        rows.append([
            _DISPLAY_NAMES["rank1"],
            "Correct / probes",
            f"{r['num_correct']} / {r['num_probe']}",
            "",
        ])

    if "classification" in results:
        r = results["classification"]
        name = _DISPLAY_NAMES["classification"]
        rows.append([name, "Top-1 accuracy", f"{r['top1_accuracy']:.4f}", "%"])
        rows.append([name, "Top-5 accuracy", f"{r['top5_accuracy']:.4f}", "%"])
        rows.append([name, "Macro F1", f"{r['macro_f1']:.4f}", "%"])
        rows.append([name, "Weighted F1", f"{r['weighted_f1']:.4f}", "%"])
        rows.append([
            name,
            "Per-class acc (mean / min)",
            f"{r['per_class_accuracy_mean']:.4f} / {r['per_class_accuracy_min']:.4f}",
            "%",
        ])
        rows.append([
            name,
            "Correct / probes",
            f"{r['num_correct']} / {r['num_samples']}",
            "",
        ])

    return rows


def print_summary(results: Dict, reference=None) -> None:
    """Print a fixed-width summary table."""
    rows = _metric_rows(results)
    header = ["Protocol", "Metric", "Value", "Unit"]
    widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(4)]

    def fmt(row):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row))

    print("\n" + "=" * (sum(widths) + 6))
    print(fmt(header))
    print("-" * (sum(widths) + 6))
    for row in rows:
        print(fmt(row))
    print("=" * (sum(widths) + 6))

    if reference is not None and "verification" in results:
        print("\nPaper reference (Tongji, d=9, m=0.5):")
        print(f"  Rank-1 = {reference.rank1:.2f}%   EER = {reference.eer:.3f}%")
        r = results["verification"]
        print(f"  Reproduced: Rank-1 = {r['rank1_accuracy']:.4f}%   "
              f"EER = {r['eer_percent']:.6f}%")
        print(f"  Source: {reference.source}")
        print()


def write_results(
    results: Dict,
    output_dir: str | Path,
    meta: Optional[Dict] = None,
    reference=None,
) -> Dict[str, str]:
    """Write report.json, summary.csv and a config snapshot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            **(meta or {}),
        },
        "reference": (
            {
                "rank1_percent": reference.rank1,
                "eer_percent": reference.eer,
                "source": reference.source,
            }
            if reference is not None else None
        ),
        "results": strip_private(results),
    }

    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "summary.csv"
    rows = _metric_rows(results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protocol", "metric", "value", "unit"])
        writer.writerows(rows)

    written = {"report_json": str(json_path), "summary_csv": str(csv_path)}
    return written


# ======================== Plotting ========================

def _official_gi_plot(genuine, impostor, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_g, h_g, x_o, h_o = gi_histogram(genuine, impostor, samples=100)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_o, h_o, "r-", label="Impostor")
    ax.plot(x_g, h_g, "b-", label="Genuine")
    ax.set_xlabel("Matching distance")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 1.2 * max(h_g.max(), h_o.max()))
    ax.set_title(f"{title} (official getGI style: per-distribution min-max)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _raw_score_hist(genuine, impostor, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = np.asarray(genuine, dtype=np.float64)
    o = np.asarray(impostor, dtype=np.float64)
    lo = min(g.min(), o.min())
    hi = max(g.max(), o.max())
    bins = np.linspace(lo, hi, 121)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(o, bins=bins, density=True, alpha=0.45, color="tab:red",
            label=f"Impostor (n={o.size:,})")
    ax.hist(g, bins=bins, density=True, alpha=0.55, color="tab:blue",
            label=f"Genuine (n={g.size:,})")
    ax.set_xlabel("Matching distance")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _roc_det_plots(result: Dict, title: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr = np.asarray(result["_fpr"], dtype=np.float64) * 100.0
    tpr = np.asarray(result["_tpr"], dtype=np.float64) * 100.0

    # ROC
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, "b-", label="ROC")
    ax.plot([0, 100], [100, 0], "k--", lw=0.8, label="EER")
    ax.set_xlim(0, 5)
    ax.set_ylim(90, 100)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("GAR (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc.png", dpi=150)
    plt.close(fig)

    # DET
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, 100.0 - tpr, "b-", label="DET")
    ax.plot([0, 5], [0, 5], "k--", lw=0.8, label="EER")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "det.png", dpi=150)
    plt.close(fig)


def _per_class_accuracy_plot(result: Dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = np.asarray(result["per_class_accuracy"], dtype=np.float64)
    support = np.asarray(result.get("per_class_support", []), dtype=np.int64)
    if support.size == acc.size:
        # Plot only classes present in the evaluated split; absent classes are
        # stored as 0% in the full-length array and would dilute the mean.
        acc = acc[support > 0]
    bins = np.linspace(0.0, 100.0, 41)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(acc, bins=bins, color="tab:blue", alpha=0.75,
            label=f"classes={acc.size}, mean={acc.mean():.2f}%")
    ax.set_xlabel("Per-class top-1 accuracy (%)")
    ax.set_ylabel("Number of classes")
    ax.set_title(
        f"Closed-set classification: top-1={result['top1_accuracy']:.2f}% "
        f"top-5={result['top5_accuracy']:.2f}%"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_results(results: Dict, output_dir: str | Path) -> List[str]:
    """Generate GI / score-histogram / ROC / DET figures.

    Requires the private score arrays produced by ``run_protocols``.
    """
    output_dir = Path(output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    specs = [
        ("verification", "_verification_genuine", "_verification_impostor",
         "Verification", "gi_verification.png"),
        ("aggregated_verification", "_aggregated_genuine", "_aggregated_impostor",
         "Aggregated verification", "gi_aggregated_verification.png"),
        ("dataset_eer", "_dataset_genuine", "_dataset_impostor",
         "Dataset (session2)", "gi_dataset_eer.png"),
    ]

    for result_key, g_key, o_key, title, gi_name in specs:
        if g_key not in results or o_key not in results:
            continue
        genuine = results[g_key]
        impostor = results[o_key]

        gi_path = output_dir / gi_name
        _official_gi_plot(genuine, impostor, title, gi_path)
        written.append(str(gi_path))

        hist_path = output_dir / f"scores_{result_key}.png"
        _raw_score_hist(genuine, impostor, title, hist_path)
        written.append(str(hist_path))

    if "verification" in results and "_fpr" in results["verification"]:
        _roc_det_plots(results["verification"], "Verification", output_dir)
        written.extend([
            str(output_dir / "roc.png"),
            str(output_dir / "det.png"),
        ])

    if "classification" in results:
        path = output_dir / "classification_per_class_accuracy.png"
        _per_class_accuracy_plot(results["classification"], path)
        written.append(str(path))

    return written
