#!/usr/bin/env python
"""Regenerate the spectral-consistency figures and Markdown report.

This script does **not** touch the model, data or attacks.  It reads the
saved ``summary.json`` (selection, validation, per-radius statistics) and
``sample_level.csv`` (per-image similarities) and re-runs the shared plotting
and reporting code from :mod:`defense.spectral_analysis`.

Usage::

    python scripts/generate_spectral_report.py \
        --output-dir results/spectral_consistency
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from defense.spectral_analysis import (  # noqa: E402
    MAIN_ATTACKS,
    compute_radius_transfer,
    plot_auc_vs_radius,
    plot_classification_accuracy,
    plot_consistency_vs_radius,
    plot_delta_consistency,
    plot_distributions,
    plot_roc_curves,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=str,
                   default=str(PROJECT_DIR / "results" / "spectral_consistency"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def load_sample_scores(path: Path) -> Dict[Tuple, Dict[str, np.ndarray]]:
    """Build ``{(session, attack, eps_tag, mode, radius): arrays}`` from CSV."""
    store: Dict[Tuple, Dict[str, list]] = {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "session", "attack", "epsilon_tag", "filter_mode", "filter_radius",
            "clean_similarity", "adv_similarity", "attack_success",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            key = (
                row["session"],
                row["attack"],
                row["epsilon_tag"],
                row["filter_mode"],
                float(row["filter_radius"]),
            )
            entry = store.get(key)
            if entry is None:
                entry = {"clean": [], "adv": [], "adv_success": []}
                store[key] = entry
            entry["clean"].append(float(row["clean_similarity"]))
            entry["adv"].append(float(row["adv_similarity"]))
            entry["adv_success"].append(
                False if row["attack"] == "clean"
                else bool(int(row["attack_success"]))
            )

    output: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for key, entry in store.items():
        output[key] = {
            "clean": np.asarray(entry["clean"], dtype=np.float64),
            "adv": np.asarray(entry["adv"], dtype=np.float64),
            "adv_success": np.asarray(entry["adv_success"], dtype=bool),
        }
    return output


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary_path = output_dir / "summary.json"
    samples_path = output_dir / "sample_level.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not samples_path.exists():
        raise FileNotFoundError(samples_path)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    meta = summary["meta"]
    records = summary["per_radius"]
    selection = summary["selection"]
    validation = summary["validation"]
    attack_comparison = summary["attack_comparison"]
    roi_stats = summary["roi_stats"]

    attacks = meta.get("main_attacks") or list(MAIN_ATTACKS)
    sessions = meta["sessions"]
    modes = meta["filter_modes"]
    primary_tag = meta["primary_epsilon_tag"]
    validation_session = meta["validation_session"]

    print(f"[Load] {samples_path} ...")
    sample_scores = load_sample_scores(samples_path)
    print(f"[Load] {len(sample_scores)} score groups")

    best_radius = {
        mode: selection["by_mode"][mode]["best_radius"] for mode in modes
    }
    thresholds = {
        mode: selection["by_mode"][mode]["threshold"] for mode in modes
    }
    radii = sorted({float(r["filter_radius"]) for r in records})
    radius_transfer = compute_radius_transfer(
        sample_scores,
        meta.get("calibration_session", "session1"),
        meta.get("validation_session", "session2"),
        primary_tag, attacks, modes, radii,
        float(meta.get("fpr_target", 0.01)),
    )

    figures: Dict[str, str] = {}
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        plot_consistency_vs_radius(
            records, attacks, primary_tag, sessions, modes,
            plots_dir / "consistency_vs_radius.png",
        )
        figures["Figure 1: consistency vs radius"] = "plots/consistency_vs_radius.png"

        plot_distributions(
            records, sample_scores, attacks, primary_tag, sessions, modes,
            best_radius, thresholds, plots_dir / "consistency_distribution.png",
        )
        figures["Figure 2: distributions at selected radius"] = (
            "plots/consistency_distribution.png"
        )

        plot_delta_consistency(
            records, attacks, primary_tag, sessions, modes,
            plots_dir / "delta_consistency.png",
        )
        figures["Figure 3: Delta_C vs radius"] = "plots/delta_consistency.png"

        plot_roc_curves(
            records, sample_scores, attacks, primary_tag, validation_session,
            modes, best_radius, plots_dir / "roc_curves.png",
            fpr_target=float(meta.get("fpr_target", 0.01)),
        )
        figures["Figure 4: ROC curves across radii (validation session)"] = (
            "plots/roc_curves.png"
        )

        plot_classification_accuracy(
            records, attacks, primary_tag, validation_session, modes,
            plots_dir / "classification_accuracy.png",
        )
        figures["Figure 5: low-pass filtering vs classification accuracy"] = (
            "plots/classification_accuracy.png"
        )

        plot_auc_vs_radius(
            records, attacks, primary_tag, sessions, modes, best_radius,
            plots_dir / "auc_vs_radius.png",
        )
        figures["Supplementary: AUC vs radius"] = "plots/auc_vs_radius.png"
        print(f"[Plots] {plots_dir}")

    write_report(
        output_dir / "REPORT.md", meta, selection, validation, records,
        attack_comparison, roi_stats, figures, radius_transfer,
    )
    print(f"[Report] {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
