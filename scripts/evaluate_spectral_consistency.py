#!/usr/bin/env python
"""Spectral-consistency pre-experiment for CompNet palmprint verification.

Hypothesis
----------
Clean palmprints keep a stable CompNet feature under Gaussian low-pass
filtering, while adversarial palmprints change more::

    C_r(x) = cos(f(x), f(G_r(x)))
    C_clean(r) > C_adv(r)

This script measures that effect for PGD / APGD / CW, with two
low-pass modes (full-frame and ROI-aware), selects a radius and an FPR=1%
threshold on Session 1, and reports independent validation on Session 2.

Pipeline (no model retraining, no attack changes)::

    ROI -> Resize -> ToTensor (pixels [0,1])
        -> attack (existing attack/ modules, pixel-space L-inf)
        -> Gaussian LPF (FFT) *before* NormSingleROI
        -> official NormSingleROI (exactly once)
        -> frozen CompNet -> 512-d L2-normalized code / 600-d logits

Outputs (default ``results/spectral_consistency/``)::

    config.json  summary.json  summary.csv  sample_level.csv
    REPORT.md
    plots/{consistency_vs_radius,consistency_distribution,delta_consistency,
           roc_curves,classification_accuracy,auc_vs_radius}.png
    logs/run.log
    cache/<run-hash>/...

Every block is cached, so an interrupted run resumes without recomputation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torchvision import transforms as T

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from attack import CompNetAttacker  # noqa: E402
from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from dataset.dataset import PalmROIDataset  # noqa: E402
from defense.spectral_analysis import (  # noqa: E402
    MAIN_ATTACKS,
    compare_attacks,
    compute_radius_transfer,
    detection_metrics,
    distribution_stats,
    encode_pixels,
    epsilon_tag,
    filtered_similarity,
    paired_delta_stats,
    plot_auc_vs_radius,
    plot_classification_accuracy,
    plot_consistency_vs_radius,
    plot_delta_consistency,
    plot_distributions,
    plot_roc_curves,
    predictions_from_logits,
    sample_score_key,
    write_csv,
    write_report,
)
from defense.spectral_filter import (  # noqa: E402
    FILTER_MODES,
    gaussian_lowpass,
    roi_mask,
)
from model.loader import load_compnet, resolve_device  # noqa: E402
from scripts.generate_adv import (  # noqa: E402
    ATTACK_NAMES,
    NORM_BOUNDED,
    build_attack,
    ensure_manifests,
)

DEFAULT_RADII = [8, 12, 16, 20, 24, 28, 32, 40]
DEFAULT_EPS = [4.0 / 255.0]
SESSION_MANIFEST = {"session1": "train_manifest", "session2": "test_manifest"}

SUMMARY_FIELDS = [
    "session", "attack", "epsilon", "epsilon_tag", "filter_mode", "filter_radius",
    "n_samples", "n_clean_correct", "n_adv_success", "n_adv_fail",
    "attack_success_rate",
    "clean_cos_mean", "clean_cos_std", "clean_cos_median", "clean_cos_q25",
    "clean_cos_q75",
    "adv_cos_mean", "adv_cos_std", "adv_cos_median", "adv_cos_q25", "adv_cos_q75",
    "delta_c",
    "paired_delta_mean", "paired_delta_std", "paired_delta_ci95_low",
    "paired_delta_ci95_high", "paired_t_statistic", "paired_t_pvalue",
    "paired_wilcoxon_pvalue", "paired_cohens_dz",
    "auc", "pr_auc", "threshold_insession", "tpr_at_fpr_1pct_insession",
    "fpr_insession", "cohens_d_scores", "ks_statistic", "ks_pvalue",
    "auc_success", "auc_fail", "tpr_success", "tpr_fail",
    "clean_top1", "clean_top5", "clean_lpf_top1", "clean_lpf_top5",
    "adv_top1", "adv_top5", "adv_lpf_top1", "adv_lpf_top5",
]

SAMPLE_FIELDS = [
    "image_id", "index", "label", "session", "attack", "epsilon", "epsilon_tag",
    "attack_success", "filter_mode", "filter_radius",
    "clean_similarity", "adv_similarity",
    "clean_prediction", "adv_prediction",
    "clean_lpf_prediction", "adv_lpf_prediction",
    "clean_top1", "clean_top5", "adv_top1", "adv_top5",
    "clean_lpf_top1", "clean_lpf_top5", "adv_lpf_top1", "adv_lpf_top5",
]


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--sessions", nargs="+", choices=sorted(SESSION_MANIFEST),
                   default=["session1", "session2"])
    p.add_argument("--calibration-session", choices=sorted(SESSION_MANIFEST),
                   default="session1")
    p.add_argument("--validation-session", choices=sorted(SESSION_MANIFEST),
                   default="session2")
    p.add_argument("--attacks", nargs="+", choices=ATTACK_NAMES,
                   default=list(MAIN_ATTACKS),
                   help="Attacks to run. Default: PGD/APGD/CW.")
    p.add_argument("--eps", nargs="+", type=float, default=DEFAULT_EPS,
                   help="L-inf budgets in pixel [0,1] for bounded attacks "
                        "(default: 4/255).")
    p.add_argument("--primary-eps", type=float, default=4.0 / 255.0,
                   help="Epsilon used for radius selection and main figures.")
    p.add_argument("--radii", nargs="+", type=float, default=DEFAULT_RADII,
                   help="Gaussian LPF radii in FFT-bin units.")
    p.add_argument("--filter-modes", nargs="+", choices=list(FILTER_MODES),
                   default=list(FILTER_MODES))
    p.add_argument("--sample-n", type=int, default=1200,
                   help="Total selected images per session (0 = all classes x "
                        "per-class). Default 1200 = 600 classes x 2.")
    p.add_argument("--per-class", type=int, default=2,
                   help="Images sampled per class (default 2).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None,
                   help="auto | cuda | cuda:0 | cpu (default: config).")
    p.add_argument("--feature-batch-size", type=int, default=128)
    p.add_argument("--attack-batch-size", type=int, default=None,
                   help="Default: config attack.batch_size.")
    p.add_argument("--steps", type=int, default=None,
                   help="Override iterations for pgd/apgd/cw.")
    p.add_argument("--fpr-target", type=float, default=0.01)
    p.add_argument("--output-dir", type=str,
                   default=str(PROJECT_DIR / "results" / "spectral_consistency"))
    p.add_argument("--fresh", action="store_true",
                   help="Ignore cached tensors/blocks and recompute.")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


# ===========================================================================
# Small utilities
# ===========================================================================

class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")
        self._file.write(f"\n===== run {datetime.now().isoformat()} =====\n")
        self._file.flush()

    def __call__(self, message: str = "") -> None:
        print(message, flush=True)
        self._file.write(str(message) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def jsonify(value):
    """Recursively convert numpy scalars/arrays to JSON-serializable objects."""
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _evenly_spaced(count: int, total: int) -> List[int]:
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    return np.unique(np.linspace(0, total - 1, count).round().astype(int)).tolist()


def select_stratified_indices(
    labels: Sequence[int],
    num_classes: int,
    per_class: int,
    sample_n: int,
    seed: int,
) -> np.ndarray:
    """Deterministic class-stratified selection (same set for every attack)."""
    labels = np.asarray(labels, dtype=np.int64)
    by_class = [np.flatnonzero(labels == c) for c in range(num_classes)]
    if sample_n <= 0 or sample_n >= num_classes * per_class:
        classes = list(range(num_classes))
    else:
        n_classes = max(1, min(num_classes, sample_n // max(1, per_class)))
        classes = _evenly_spaced(n_classes, num_classes)
    rng = np.random.default_rng(seed)
    indices: List[int] = []
    for c in classes:
        pool = by_class[c]
        if pool.size == 0:
            continue
        take = min(per_class, pool.size)
        chosen = np.sort(rng.choice(pool, size=take, replace=False))
        indices.extend(int(i) for i in chosen)
    return np.asarray(indices, dtype=np.int64)


def _relative(path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _accuracy(info: Dict[str, np.ndarray]) -> Tuple[float, float]:
    return (
        100.0 * float(np.mean(info["top1"])),
        100.0 * float(np.mean(info["top5"])),
    )


def _row_get(records_index: Dict[Tuple, Dict], session: str, attack: str,
             eps_tag_value: str, mode: str, radius: float) -> Optional[Dict]:
    tag = "clean" if attack == "clean" else eps_tag_value
    return records_index.get((session, attack, tag, mode, float(radius)))


# ===========================================================================
# Data / attack / feature caches
# ===========================================================================

def load_session_pixels(
    session: str,
    manifest: str,
    cfg,
    indices: np.ndarray,
    cache_dir: Path,
    fresh: bool,
    log: Logger,
) -> Dict[str, object]:
    path = cache_dir / "clean" / f"{session}.npz"
    if not fresh and path.exists():
        try:
            data = _load_npz(path)
            if (
                np.array_equal(data["indices"], indices)
                and data["pixels"].shape[1:] == (cfg.data.outchannels, cfg.data.imside, cfg.data.imside)
            ):
                log(f"[Cache] clean pixels {session}: {data['pixels'].shape}")
                return {
                    "pixels": torch.from_numpy(data["pixels"].astype(np.float32)),
                    "labels": data["labels"].astype(np.int64),
                    "paths": [str(p) for p in data["paths"].tolist()],
                    "indices": data["indices"].astype(np.int64),
                }
        except Exception as exc:  # corrupt cache -> recompute
            log(f"[Cache] ignoring {path}: {exc}")

    pixel_tf = T.Compose([T.Resize(cfg.data.imside), T.ToTensor()])
    dataset = PalmROIDataset(
        manifest,
        imside=cfg.data.imside,
        outchannels=cfg.data.outchannels,
        transform=pixel_tf,
    )
    pixels: List[torch.Tensor] = []
    for idx in indices:
        image, _ = dataset[int(idx)]
        pixels.append(image)
    tensor = torch.stack(pixels, 0).float()
    labels = np.asarray([dataset.images_label[int(i)] for i in indices], dtype=np.int64)
    paths = [_relative(dataset.images_path[int(i)]) for i in indices]
    _save_npz(
        path,
        pixels=tensor.numpy().astype(np.float32),
        labels=labels,
        indices=np.asarray(indices, dtype=np.int64),
        paths=np.asarray(paths),
    )
    log(f"[Data] loaded {session}: {tuple(tensor.shape)} from {manifest}")
    return {"pixels": tensor, "labels": labels, "paths": paths,
            "indices": np.asarray(indices, dtype=np.int64)}


def load_or_generate_adversarial(
    model,
    clean: Dict[str, object],
    session: str,
    attack_name: str,
    epsilon: Optional[float],
    cfg,
    device: torch.device,
    seed: int,
    attack_batch_size: int,
    cache_dir: Path,
    fresh: bool,
    log: Logger,
) -> Dict[str, object]:
    tag = epsilon_tag(epsilon)
    path = cache_dir / "adv" / f"{session}_{attack_name}_{tag}.npz"
    labels = clean["labels"]
    indices = clean["indices"]
    if not fresh and path.exists():
        try:
            data = _load_npz(path)
            if (
                data["adv"].shape == clean["pixels"].shape
                and np.array_equal(data["labels"], labels)
                and np.array_equal(data["indices"], indices)
            ):
                log(f"[Cache] adversarial {session}/{attack_name}@{tag}: {data['adv'].shape}")
                return {
                    "adv": torch.from_numpy(data["adv"].astype(np.float32)),
                    "labels": data["labels"].astype(np.int64),
                    "indices": data["indices"].astype(np.int64),
                }
        except Exception as exc:
            log(f"[Cache] ignoring {path}: {exc}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    attacker = CompNetAttacker(
        model,
        device=device,
        preserve_background=cfg.attack.preserve_background,
        input_range=cfg.attack.input_range,
    )
    # This auxiliary spectral analysis evaluates the classification-loss
    # ablation (no gallery reference is built here); the main attack benchmark
    # uses loss="matching" and session1 gallery features.  CW uses its logit
    # margin (`loss="cw"`); PGD/APGD/Patch use CE.
    spectral_loss = "cw" if attack_name == "cw" else "ce"
    attack = build_attack(
        attack_name,
        attacker,
        0.0 if epsilon is None else float(epsilon),
        cfg,
        loss=spectral_loss,
        gallery_features=None,
        gallery_labels=None,
    )

    pixels = clean["pixels"]
    n = int(labels.size)
    chunks: List[torch.Tensor] = []
    start_time = time.perf_counter()
    n_batches = (n + attack_batch_size - 1) // attack_batch_size
    for batch_idx, start in enumerate(range(0, n, attack_batch_size), 1):
        end = min(start + attack_batch_size, n)
        x = pixels[start:end].to(device)
        y = torch.as_tensor(labels[start:end], device=device, dtype=torch.long)
        adv = attack.attack(x, y)
        chunks.append(adv.detach().float().cpu())
        if batch_idx % 10 == 0 or batch_idx == n_batches:
            done = min(batch_idx * attack_batch_size, n)
            log(f"  [{batch_idx}/{n_batches}] {done}/{n} attacked, "
                f"{time.perf_counter() - start_time:.1f}s")
    adv_tensor = torch.cat(chunks, 0)
    _save_npz(
        path,
        adv=adv_tensor.numpy().astype(np.float32),
        labels=labels,
        indices=indices,
    )
    log(f"[Attack] {session}/{attack_name}@{tag}: {tuple(adv_tensor.shape)}, "
        f"{time.perf_counter() - start_time:.1f}s")
    return {"adv": adv_tensor, "labels": labels, "indices": indices}


def load_or_compute_eval(
    model,
    pixels: torch.Tensor,
    labels: np.ndarray,
    path: Path,
    device: torch.device,
    batch_size: int,
    fresh: bool,
    log: Logger,
    name: str,
) -> Dict[str, np.ndarray]:
    if not fresh and path.exists():
        try:
            data = _load_npz(path)
            if data["codes"].shape[0] == labels.size:
                log(f"[Cache] codes/logits {name}")
                return {
                    "codes": data["codes"].astype(np.float32),
                    "logits": data["logits"].astype(np.float32),
                    "prediction": data["prediction"].astype(np.int64),
                    "top1": data["top1"].astype(bool),
                    "top5": data["top5"].astype(bool),
                }
        except Exception as exc:
            log(f"[Cache] ignoring {path}: {exc}")
    log(f"[Features] encoding {name} ({len(labels)} images) ...")
    codes, logits = encode_pixels(model, pixels, device, batch_size=batch_size)
    info = predictions_from_logits(logits, labels)
    result = {
        "codes": codes.numpy().astype(np.float32),
        "logits": logits.numpy().astype(np.float32),
        **info,
    }
    _save_npz(path, **result)
    return result


def load_or_compute_block(
    path: Path,
    compute_fn,
    required: Sequence[str],
    fresh: bool,
    log: Logger,
    name: str,
) -> Dict[str, np.ndarray]:
    if not fresh and path.exists():
        try:
            data = _load_npz(path)
            if all(k in data for k in required):
                return data
        except Exception as exc:
            log(f"[Cache] ignoring {path}: {exc}")
    result = compute_fn()
    _save_npz(path, **result)
    return result


# ===========================================================================
# Summary construction
# ===========================================================================

def make_clean_record(
    session: str,
    mode: str,
    radius: float,
    labels: np.ndarray,
    clean_info: Dict[str, np.ndarray],
    block: Dict[str, np.ndarray],
) -> Dict[str, object]:
    stats = distribution_stats(block["cosine"])
    clean_top1, clean_top5 = _accuracy(clean_info)
    lpf_top1, lpf_top5 = _accuracy(block)
    return {
        "session": session, "attack": "clean", "epsilon": None,
        "epsilon_tag": "clean", "filter_mode": mode, "filter_radius": float(radius),
        "n_samples": int(labels.size),
        "n_clean_correct": int(clean_info["top1"].sum()),
        "n_adv_success": 0, "n_adv_fail": int(labels.size),
        "attack_success_rate": 0.0,
        "clean_cos_mean": stats["mean"], "clean_cos_std": stats["std"],
        "clean_cos_median": stats["median"], "clean_cos_q25": stats["q25"],
        "clean_cos_q75": stats["q75"],
        "adv_cos_mean": stats["mean"], "adv_cos_std": stats["std"],
        "adv_cos_median": stats["median"], "adv_cos_q25": stats["q25"],
        "adv_cos_q75": stats["q75"],
        "delta_c": 0.0,
        "paired_delta_mean": 0.0, "paired_delta_std": 0.0,
        "paired_delta_ci95_low": 0.0, "paired_delta_ci95_high": 0.0,
        "paired_t_statistic": None, "paired_t_pvalue": None,
        "paired_wilcoxon_pvalue": None, "paired_cohens_dz": 0.0,
        "auc": None, "pr_auc": None, "threshold_insession": None,
        "tpr_at_fpr_1pct_insession": None, "fpr_insession": None,
        "cohens_d_scores": None, "ks_statistic": None, "ks_pvalue": None,
        "auc_success": None, "auc_fail": None,
        "tpr_success": None, "tpr_fail": None,
        "clean_top1": clean_top1, "clean_top5": clean_top5,
        "clean_lpf_top1": lpf_top1, "clean_lpf_top5": lpf_top5,
        "adv_top1": clean_top1, "adv_top5": clean_top5,
        "adv_lpf_top1": lpf_top1, "adv_lpf_top5": lpf_top5,
    }


def make_attack_record(
    session: str,
    attack: str,
    epsilon: Optional[float],
    mode: str,
    radius: float,
    labels: np.ndarray,
    clean_info: Dict[str, np.ndarray],
    adv_info: Dict[str, np.ndarray],
    clean_block: Dict[str, np.ndarray],
    adv_block: Dict[str, np.ndarray],
    fpr_target: float,
) -> Dict[str, object]:
    clean_scores = 1.0 - clean_block["cosine"]
    adv_scores = 1.0 - adv_block["cosine"]
    det = detection_metrics(clean_scores, adv_scores, fpr_target=fpr_target)
    threshold = det["threshold"]

    success = ~adv_info["top1"].astype(bool)
    n_success = int(success.sum())
    n_fail = int(labels.size - n_success)
    if n_success:
        det_success = detection_metrics(
            clean_scores, adv_scores[success], fpr_target=fpr_target,
            threshold=threshold,
        )
    else:
        det_success = {}
    if n_fail:
        det_fail = detection_metrics(
            clean_scores, adv_scores[~success], fpr_target=fpr_target,
            threshold=threshold,
        )
    else:
        det_fail = {}

    clean_stats = distribution_stats(clean_block["cosine"])
    adv_stats = distribution_stats(adv_block["cosine"])
    paired = paired_delta_stats(clean_block["cosine"], adv_block["cosine"])

    clean_top1, clean_top5 = _accuracy(clean_info)
    lpf_clean_top1, lpf_clean_top5 = _accuracy(clean_block)
    adv_top1, adv_top5 = _accuracy(adv_info)
    lpf_adv_top1, lpf_adv_top5 = _accuracy(adv_block)

    return {
        "session": session, "attack": attack, "epsilon": epsilon,
        "epsilon_tag": epsilon_tag(epsilon), "filter_mode": mode,
        "filter_radius": float(radius), "n_samples": int(labels.size),
        "n_clean_correct": int(clean_info["top1"].sum()),
        "n_adv_success": n_success, "n_adv_fail": n_fail,
        "attack_success_rate": 100.0 * n_success / max(1, int(labels.size)),
        "clean_cos_mean": clean_stats["mean"], "clean_cos_std": clean_stats["std"],
        "clean_cos_median": clean_stats["median"],
        "clean_cos_q25": clean_stats["q25"], "clean_cos_q75": clean_stats["q75"],
        "adv_cos_mean": adv_stats["mean"], "adv_cos_std": adv_stats["std"],
        "adv_cos_median": adv_stats["median"],
        "adv_cos_q25": adv_stats["q25"], "adv_cos_q75": adv_stats["q75"],
        "delta_c": clean_stats["mean"] - adv_stats["mean"],
        "paired_delta_mean": paired["delta_mean"],
        "paired_delta_std": paired["delta_std"],
        "paired_delta_ci95_low": paired["delta_ci95_low"],
        "paired_delta_ci95_high": paired["delta_ci95_high"],
        "paired_t_statistic": paired["paired_t_statistic"],
        "paired_t_pvalue": paired["paired_t_pvalue"],
        "paired_wilcoxon_pvalue": paired["paired_wilcoxon_pvalue"],
        "paired_cohens_dz": paired["paired_cohens_dz"],
        "auc": det["auc"], "pr_auc": det["pr_auc"],
        "threshold_insession": threshold,
        "tpr_at_fpr_1pct_insession": det["tpr"],
        "fpr_insession": det["fpr"],
        "cohens_d_scores": det["cohens_d"],
        "ks_statistic": det["ks_statistic"], "ks_pvalue": det["ks_pvalue"],
        "auc_success": det_success.get("auc"),
        "auc_fail": det_fail.get("auc"),
        "tpr_success": det_success.get("tpr"),
        "tpr_fail": det_fail.get("tpr"),
        "clean_top1": clean_top1, "clean_top5": clean_top5,
        "clean_lpf_top1": lpf_clean_top1, "clean_lpf_top5": lpf_clean_top5,
        "adv_top1": adv_top1, "adv_top5": adv_top5,
        "adv_lpf_top1": lpf_adv_top1, "adv_lpf_top5": lpf_adv_top5,
    }


def select_radius_and_threshold(
    records_index: Dict[Tuple, Dict],
    sample_scores: Dict[Tuple, Dict[str, np.ndarray]],
    calibration_session: str,
    primary_eps_tag: str,
    attacks: Sequence[str],
    modes: Sequence[str],
    radii: Sequence[float],
    fpr_target: float,
    log: Logger,
) -> Dict[str, object]:
    selection: Dict[str, object] = {
        "criterion": "maximum mean ROC-AUC over main L-inf attacks on the "
                     "calibration session; ties broken by mean Delta_C then "
                     "by the smaller radius",
        "fpr_target": float(fpr_target),
        "main_attacks": list(attacks),
        "primary_epsilon_tag": primary_eps_tag,
        "calibration_session": calibration_session,
        "by_mode": {},
    }

    for mode in modes:
        radius_entries: List[Dict[str, object]] = []
        for radius in radii:
            per_attack_auc: Dict[str, Optional[float]] = {}
            aucs: List[float] = []
            deltas: List[float] = []
            for attack in attacks:
                row = _row_get(records_index, calibration_session, attack,
                               primary_eps_tag, mode, radius)
                auc = row.get("auc") if row else None
                per_attack_auc[attack] = auc
                if auc is not None:
                    aucs.append(float(auc))
                if row and row.get("delta_c") is not None:
                    deltas.append(float(row["delta_c"]))
            radius_entries.append(
                {
                    "radius": float(radius),
                    "mean_auc": float(np.mean(aucs)) if aucs else None,
                    "mean_delta_c": float(np.mean(deltas)) if deltas else None,
                    "per_attack_auc": per_attack_auc,
                }
            )

        valid = [e for e in radius_entries if e["mean_auc"] is not None]
        if not valid:
            raise RuntimeError(
                f"No detection metrics for mode={mode!r}; cannot select a radius."
            )
        best = max(
            valid,
            key=lambda e: (
                float(e["mean_auc"]),
                float(e["mean_delta_c"]) if e["mean_delta_c"] is not None else -np.inf,
                -float(e["radius"]),
            ),
        )
        best_radius = float(best["radius"])

        clean_pair = sample_scores[
            sample_score_key(calibration_session, "clean", primary_eps_tag,
                             mode, best_radius)
        ]
        clean_scores = 1.0 - clean_pair["clean"]
        threshold = float(
            np.quantile(clean_scores, 1.0 - fpr_target, method="higher")
        )

        cal_by_attack: Dict[str, Dict] = {}
        tprs: List[float] = []
        for attack in attacks:
            pair = sample_scores[
                sample_score_key(calibration_session, attack, primary_eps_tag,
                                 mode, best_radius)
            ]
            metrics = detection_metrics(
                1.0 - pair["clean"], 1.0 - pair["adv"], threshold=threshold
            )
            cal_by_attack[attack] = metrics
            if metrics["tpr"] is not None:
                tprs.append(float(metrics["tpr"]))

        selection["by_mode"][mode] = {
            "best_radius": int(round(best_radius)),
            "mean_auc": float(best["mean_auc"]),
            "mean_delta_c": best["mean_delta_c"],
            "per_attack_auc": best["per_attack_auc"],
            "threshold": threshold,
            "calibration_fpr_empirical": float(np.mean(clean_scores > threshold)),
            "calibration_tpr_mean": float(np.mean(tprs)) if tprs else None,
            "calibration_by_attack": cal_by_attack,
            "by_radius": radius_entries,
        }
        log(
            f"[Select] {mode}: r={int(round(best_radius))}, "
            f"mean AUC={best['mean_auc']:.4f}, "
            f"mean Delta_C={best['mean_delta_c']}, "
            f"threshold={threshold:.6f}"
        )

    mode_aucs = {
        m: selection["by_mode"][m]["mean_auc"] for m in selection["by_mode"]
    }
    if "full_frame" in mode_aucs and "roi_aware" in mode_aucs:
        if abs(mode_aucs["full_frame"] - mode_aucs["roi_aware"]) < 1e-9:
            primary_mode = "full_frame"
        else:
            primary_mode = max(mode_aucs, key=mode_aucs.get)
    else:
        primary_mode = max(mode_aucs, key=mode_aucs.get)
    selection["primary_mode"] = primary_mode
    log(f"[Select] primary mode: {primary_mode}")
    return selection


def build_validation(
    selection: Dict[str, object],
    sample_scores: Dict[Tuple, Dict[str, np.ndarray]],
    calibration_session: str,
    validation_session: str,
    primary_eps_tag: str,
    eps_tags: Sequence[str],
    attacks: Sequence[str],
    modes: Sequence[str],
) -> Dict[str, object]:
    validation: Dict[str, object] = {
        "calibration_session": calibration_session,
        "validation_session": validation_session,
        "primary_epsilon_tag": primary_eps_tag,
        "by_epsilon": {},
        "by_mode": {},
    }
    for tag in eps_tags:
        by_mode: Dict[str, Dict] = {}
        for mode in modes:
            entry = selection["by_mode"][mode]
            radius = float(entry["best_radius"])
            threshold = float(entry["threshold"])
            by_attack: Dict[str, Dict] = {}
            for attack in attacks:
                cal_pair = sample_scores[
                    sample_score_key(calibration_session, attack, tag,
                                     mode, radius)
                ]
                val_pair = sample_scores[
                    sample_score_key(validation_session, attack, tag,
                                     mode, radius)
                ]
                cal_metrics = detection_metrics(
                    1.0 - cal_pair["clean"], 1.0 - cal_pair["adv"],
                    threshold=threshold,
                )
                val_metrics = detection_metrics(
                    1.0 - val_pair["clean"], 1.0 - val_pair["adv"],
                    threshold=threshold,
                )
                success = val_pair["adv_success"].astype(bool)
                n_success = int(success.sum())
                n_fail = int(success.size - n_success)
                m_success = (
                    detection_metrics(
                        1.0 - val_pair["clean"], 1.0 - val_pair["adv"][success],
                        threshold=threshold,
                    )
                    if n_success else {}
                )
                m_fail = (
                    detection_metrics(
                        1.0 - val_pair["clean"], 1.0 - val_pair["adv"][~success],
                        threshold=threshold,
                    )
                    if n_fail else {}
                )
                by_attack[attack] = {
                    "auc": val_metrics["auc"],
                    "pr_auc": val_metrics["pr_auc"],
                    "tpr": val_metrics["tpr"],
                    "fpr": val_metrics["fpr"],
                    "cohens_d": val_metrics["cohens_d"],
                    "ks_statistic": val_metrics["ks_statistic"],
                    "ks_pvalue": val_metrics["ks_pvalue"],
                    "threshold": threshold,
                    "n_adv_success": n_success,
                    "n_adv_fail": n_fail,
                    "auc_success": m_success.get("auc"),
                    "auc_fail": m_fail.get("auc"),
                    "tpr_success": m_success.get("tpr"),
                    "tpr_fail": m_fail.get("tpr"),
                    "calibration": cal_metrics,
                }
            aucs = [v["auc"] for v in by_attack.values() if v["auc"] is not None]
            tprs = [v["tpr"] for v in by_attack.values() if v["tpr"] is not None]
            by_mode[mode] = {
                "best_radius": int(round(radius)),
                "threshold": threshold,
                "by_attack": by_attack,
                "mean_auc": float(np.mean(aucs)) if aucs else None,
                "mean_tpr": float(np.mean(tprs)) if tprs else None,
            }
        validation["by_epsilon"][tag] = {
            "epsilon_tag": tag,
            "by_mode": by_mode,
        }
        if tag == primary_eps_tag:
            validation["by_mode"] = by_mode
    return validation


def build_attack_comparison(
    sample_scores: Dict[Tuple, Dict[str, np.ndarray]],
    selection: Dict[str, object],
    calibration_session: str,
    primary_eps_tag: str,
    attacks: Sequence[str],
    modes: Sequence[str],
) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for mode in modes:
        radius = float(selection["by_mode"][mode]["best_radius"])
        deltas = {}
        for attack in attacks:
            pair = sample_scores[
                sample_score_key(calibration_session, attack, primary_eps_tag,
                                 mode, radius)
            ]
            deltas[attack] = pair["clean"] - pair["adv"]
        out[mode] = compare_attacks(deltas)
    return out


# ===========================================================================
# Outputs
# ===========================================================================

def write_sample_level(
    path: Path,
    sessions: Sequence[str],
    modes: Sequence[str],
    radii: Sequence[float],
    attacks: Sequence[str],
    epsilons: Sequence[float],
    clean_data: Dict[str, Dict],
    adv_data: Dict[Tuple, Dict],
    clean_blocks: Dict[Tuple, Dict],
    adv_blocks: Dict[Tuple, Dict],
    log: Logger,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SAMPLE_FIELDS)
        for session in sessions:
            labels = clean_data[session]["labels"]
            paths = clean_data[session]["paths"]
            indices = clean_data[session]["indices"]
            cinfo = clean_data[session]["info"]
            n = int(labels.size)
            for mode in modes:
                for radius in radii:
                    cb = clean_blocks[(session, mode, float(radius))]
                    for i in range(n):
                        writer.writerow([
                            paths[i], int(indices[i]), int(labels[i]), session,
                            "clean", "", "clean", 0, mode, float(radius),
                            float(cb["cosine"][i]), float(cb["cosine"][i]),
                            int(cinfo["prediction"][i]), int(cinfo["prediction"][i]),
                            int(cb["prediction"][i]), int(cb["prediction"][i]),
                            int(cinfo["top1"][i]), int(cinfo["top5"][i]),
                            int(cinfo["top1"][i]), int(cinfo["top5"][i]),
                            int(cb["top1"][i]), int(cb["top5"][i]),
                            int(cb["top1"][i]), int(cb["top5"][i]),
                        ])
                    for attack in attacks:
                        runs = epsilons if attack in NORM_BOUNDED else [None]
                        for eps in runs:
                            tag = epsilon_tag(eps)
                            info = adv_data[(session, attack, tag)]["info"]
                            ab = adv_blocks[(session, attack, tag, mode, float(radius))]
                            for i in range(n):
                                success = 0 if info["top1"][i] else 1
                                writer.writerow([
                                    paths[i], int(indices[i]), int(labels[i]), session,
                                    attack,
                                    "" if eps is None else float(eps), tag,
                                    success, mode, float(radius),
                                    float(cb["cosine"][i]), float(ab["cosine"][i]),
                                    int(cinfo["prediction"][i]),
                                    int(info["prediction"][i]),
                                    int(cb["prediction"][i]),
                                    int(ab["prediction"][i]),
                                    int(cinfo["top1"][i]), int(cinfo["top5"][i]),
                                    int(info["top1"][i]), int(info["top5"][i]),
                                    int(cb["top1"][i]), int(cb["top5"][i]),
                                    int(ab["top1"][i]), int(ab["top5"][i]),
                                ])
    log(f"[Output] sample_level.csv ({time.perf_counter() - start:.1f}s)")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _environment() -> Dict[str, object]:
    import matplotlib
    import scipy
    import sklearn

    env: Dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    return env


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    for session in (args.calibration_session, args.validation_session):
        if session not in args.sessions:
            raise ValueError(
                f"{session} must be included in --sessions (got {args.sessions})."
            )
    if args.calibration_session == args.validation_session:
        raise ValueError("Calibration and validation sessions must differ.")
    if args.fpr_target <= 0 or args.fpr_target >= 1:
        raise ValueError("--fpr-target must be in (0, 1).")
    if args.sample_n < 0:
        raise ValueError("--sample-n must be >= 0.")
    if args.per_class <= 0:
        raise ValueError("--per-class must be > 0.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(output_dir / "logs" / "run.log")

    started = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()

    overrides = {"evaluation": {}}
    if args.device:
        overrides["evaluation"]["device"] = args.device
    cfg = load_config(args.config, overrides)
    if args.steps is not None:
        cfg._steps_override = int(args.steps)

    attacks = list(dict.fromkeys(args.attacks))
    main_attacks = [a for a in attacks if a in NORM_BOUNDED]
    extra_attacks = [a for a in attacks if a not in NORM_BOUNDED]
    if not main_attacks:
        raise ValueError("At least one L-inf attack is required for selection/validation.")
    if not any(
        np.isclose(args.primary_eps, e, rtol=0.0, atol=1e-9) for e in args.eps
    ):
        raise ValueError(
            f"--primary-eps {args.primary_eps} must be one of --eps {args.eps}."
        )
    primary_eps = next(
        float(e) for e in args.eps
        if np.isclose(e, args.primary_eps, rtol=0.0, atol=1e-9)
    )
    primary_tag = epsilon_tag(primary_eps)

    radii = sorted(dict.fromkeys(float(r) for r in args.radii))
    modes = list(dict.fromkeys(args.filter_modes))
    attack_batch_size = args.attack_batch_size or cfg.attack.batch_size

    log("=" * 78)
    log("CompNet spectral-consistency pre-experiment")
    log("=" * 78)
    log(f"  output            : {output_dir}")
    log(f"  sessions          : {args.sessions} "
        f"(cal={args.calibration_session}, val={args.validation_session})")
    log(f"  attacks           : {attacks} (main L-inf: {main_attacks})")
    log(f"  epsilons          : {list(args.eps)} (primary {primary_eps})")
    log(f"  radii             : {radii}")
    log(f"  filter modes      : {modes}")
    log(f"  sample_n/per_class: {args.sample_n}/{args.per_class}, seed={args.seed}")
    log(f"  device            : {cfg.evaluation.device}")

    manifests = ensure_manifests(cfg)
    device = resolve_device(cfg.evaluation.device)
    log(f"[Device] {device}")

    model, device = load_compnet(
        weights=cfg.paths.weights,
        num_classes=cfg.model.num_classes,
        device=str(device),
    )
    model.eval()

    # ---- select the same clean images for every attack --------------------
    session_indices: Dict[str, np.ndarray] = {}
    clean_data: Dict[str, Dict] = {}
    for session in args.sessions:
        manifest = manifests[SESSION_MANIFEST[session]]
        # Read labels only for selection (cheap, no pixel transform).
        base = PalmROIDataset(manifest, imside=cfg.data.imside,
                              outchannels=cfg.data.outchannels,
                              transform=T.Compose([T.Resize(cfg.data.imside),
                                                   T.ToTensor()]))
        base_labels = np.asarray(base.images_label, dtype=np.int64)
        indices = select_stratified_indices(
            base_labels, cfg.data.num_classes, args.per_class,
            args.sample_n, args.seed,
        )
        session_indices[session] = indices
        log(f"[Data] {session}: selected {indices.size} images "
            f"({len(np.unique(base_labels[indices]))} classes)")

    # ---- cache key ---------------------------------------------------------
    weights_sha = sha256_file(Path(cfg.paths.weights))
    key_meta = {
        "cache_version": 1,
        "filter_version": "fft-gaussian-v1",
        "preprocessing": "official NormSingleROI (once)",
        "weights_sha256": weights_sha,
        "model_num_classes": int(cfg.model.num_classes),
        "sessions": {
            s: {
                "n": int(session_indices[s].size),
                "indices_sha1": hashlib.sha1(
                    session_indices[s].tobytes()
                ).hexdigest(),
                "manifest_sha1": hashlib.sha1(
                    Path(manifests[SESSION_MANIFEST[s]]).read_bytes()
                ).hexdigest(),
            }
            for s in args.sessions
        },
        "sample_n": int(args.sample_n),
        "per_class": int(args.per_class),
        "seed": int(args.seed),
        "attacks": attacks,
        "epsilons": [float(e) for e in args.eps],
        "radii": [float(r) for r in radii],
        "filter_modes": modes,
        "attack_config": {
            "epsilons": [float(e) for e in cfg.attack.epsilons],
            "pgd_steps": int(cfg.attack.pgd_steps),
            "pgd_alpha_scale": float(cfg.attack.pgd_alpha_scale),
            "cw_steps": int(cfg.attack.cw_steps),
            "cw_kappa": float(cfg.attack.cw_kappa),
            "cw_alpha_scale": float(cfg.attack.cw_alpha_scale),
            "apgd_steps": int(cfg.attack.apgd_steps),
            "loss": "per-attack (ce; cw for CW)",
            "steps_override": args.steps,
            "preserve_background": bool(cfg.attack.preserve_background),
            "input_range": cfg.attack.input_range,
            "batch_size": int(attack_batch_size),
        },
    }
    cache_key = hashlib.sha256(
        json.dumps(key_meta, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = output_dir / "cache" / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "key_meta.json", "w", encoding="utf-8") as f:
        json.dump(jsonify(key_meta), f, indent=2)
    log(f"[Cache] key={cache_key} -> {cache_dir}")

    # ---- load clean pixels ------------------------------------------------
    for session in args.sessions:
        clean_data[session] = load_session_pixels(
            session, manifests[SESSION_MANIFEST[session]], cfg,
            session_indices[session], cache_dir, args.fresh, log,
        )

    # ---- clean features/logits -------------------------------------------
    for session in args.sessions:
        path = cache_dir / "eval" / f"clean_{session}.npz"
        info = load_or_compute_eval(
            model, clean_data[session]["pixels"], clean_data[session]["labels"],
            path, device, args.feature_batch_size, args.fresh, log,
            f"clean/{session}",
        )
        clean_data[session]["info"] = info
        clean_data[session]["mask"] = roi_mask(clean_data[session]["pixels"])
        # Clean top1-top2 ArcFace margin: a diagnostic for why gradient-based
        # attacks can be much weaker on one session than another.
        top2 = np.sort(info["logits"].astype(np.float64), axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        clean_data[session]["margin_stats"] = {
            "mean": float(margin.mean()),
            "median": float(np.median(margin)),
            "q01": float(np.quantile(margin, 0.01)),
            "min": float(margin.min()),
            "max": float(margin.max()),
        }

    # ---- clean blocks + clean summary rows -------------------------------
    clean_blocks: Dict[Tuple, Dict[str, np.ndarray]] = {}
    sample_scores: Dict[Tuple, Dict[str, np.ndarray]] = {}
    records: List[Dict[str, object]] = []
    for session in args.sessions:
        pixels = clean_data[session]["pixels"]
        labels = clean_data[session]["labels"]
        mask = clean_data[session]["mask"]
        info = clean_data[session]["info"]
        codes = torch.from_numpy(info["codes"])
        for mode in modes:
            for radius in radii:
                path = cache_dir / "blocks" / (
                    f"clean_{session}_{mode}_r{radius:g}.npz"
                )
                block = load_or_compute_block(
                    path,
                    lambda p=pixels, c=codes, l=labels, m=mask, r=radius, md=mode:
                        filtered_similarity(
                            model, p, c, l, r, md, m, device,
                            batch_size=args.feature_batch_size,
                        ),
                    required=("cosine", "prediction", "top1", "top5"),
                    fresh=args.fresh, log=log,
                    name=f"clean/{session}/{mode}/r{radius:g}",
                )
                clean_blocks[(session, mode, float(radius))] = block
                sample_scores[
                    sample_score_key(session, "clean", "clean", mode, radius)
                ] = {
                    "clean": block["cosine"],
                    "adv": block["cosine"],
                    "adv_success": np.zeros(labels.size, dtype=bool),
                }
                records.append(
                    make_clean_record(session, mode, radius, labels, info, block)
                )
        log(f"[Blocks] clean done for {session}")

    # ---- attacks + adversarial blocks ------------------------------------
    adv_data: Dict[Tuple, Dict] = {}
    adv_blocks: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for session in args.sessions:
        labels = clean_data[session]["labels"]
        mask = clean_data[session]["mask"]
        for attack in attacks:
            runs = args.eps if attack in NORM_BOUNDED else [None]
            for eps in runs:
                tag = epsilon_tag(eps)
                log("-" * 70)
                log(f"[Attack] {session}/{attack}@{tag}")
                adv = load_or_generate_adversarial(
                    model, clean_data[session], session, attack, eps, cfg,
                    device, args.seed, attack_batch_size, cache_dir,
                    args.fresh, log,
                )
                eval_path = cache_dir / "eval" / f"adv_{session}_{attack}_{tag}.npz"
                info = load_or_compute_eval(
                    model, adv["adv"], labels, eval_path, device,
                    args.feature_batch_size, args.fresh, log,
                    f"adv/{session}/{attack}@{tag}",
                )
                adv_data[(session, attack, tag)] = {
                    "info": info, "epsilon": eps,
                }
                codes = torch.from_numpy(info["codes"])
                for mode in modes:
                    for radius in radii:
                        path = cache_dir / "blocks" / (
                            f"adv_{session}_{attack}_{tag}_{mode}_r{radius:g}.npz"
                        )
                        block = load_or_compute_block(
                            path,
                            lambda p=adv["adv"], c=codes, l=labels, m=mask,
                                   r=radius, md=mode:
                                filtered_similarity(
                                    model, p, c, l, r, md, m, device,
                                    batch_size=args.feature_batch_size,
                                ),
                            required=("cosine", "prediction", "top1", "top5"),
                            fresh=args.fresh, log=log,
                            name=f"adv/{session}/{attack}@{tag}/{mode}/r{radius:g}",
                        )
                        adv_blocks[
                            (session, attack, tag, mode, float(radius))
                        ] = block
                        sample_scores[
                            sample_score_key(session, attack, tag, mode, radius)
                        ] = {
                            "clean": clean_blocks[(session, mode, float(radius))]["cosine"],
                            "adv": block["cosine"],
                            "adv_success": ~info["top1"].astype(bool),
                        }
                        records.append(
                            make_attack_record(
                                session, attack, eps, mode, radius, labels,
                                clean_data[session]["info"], info,
                                clean_blocks[(session, mode, float(radius))],
                                block, args.fpr_target,
                            )
                        )
                del adv  # release adversarial pixels before the next attack
                log(f"[Blocks] {session}/{attack}@{tag} done")

    # ---- index summary rows ----------------------------------------------
    records = sorted(
        records,
        key=lambda r: (
            r["session"], "clean" if r["attack"] == "clean" else "1",
            str(r["epsilon_tag"]), r["filter_mode"], float(r["filter_radius"]),
        ),
    )
    records_index = {
        (r["session"], r["attack"], r["epsilon_tag"],
         r["filter_mode"], float(r["filter_radius"])): r
        for r in records
    }

    # ---- selection / validation / comparisons ----------------------------
    selection = select_radius_and_threshold(
        records_index, sample_scores, args.calibration_session, primary_tag,
        main_attacks, modes, radii, args.fpr_target, log,
    )
    selection["primary_epsilon"] = primary_eps
    selection["extra_attacks"] = extra_attacks
    selection["all_epsilons"] = [float(e) for e in args.eps]
    eps_tags = [epsilon_tag(e) for e in args.eps]
    validation = build_validation(
        selection, sample_scores, args.calibration_session,
        args.validation_session, primary_tag, eps_tags, main_attacks, modes,
    )
    attack_comparison = build_attack_comparison(
        sample_scores, selection, args.calibration_session, primary_tag,
        main_attacks, modes,
    )
    radius_transfer = compute_radius_transfer(
        sample_scores, args.calibration_session, args.validation_session,
        primary_tag, main_attacks, modes, radii, args.fpr_target,
    )

    # ---- ROI statistics ---------------------------------------------------
    roi_zero_flags: List[bool] = []
    roi_fractions: List[float] = []
    for session in args.sessions:
        pixels = clean_data[session]["pixels"]
        zero_per_image = (pixels == 0).flatten(1).any(dim=1)
        roi_zero_flags.extend(bool(x) for x in zero_per_image.tolist())
        roi_fractions.append(float((pixels > 0).float().mean().item()))
    primary_mode = selection["primary_mode"]
    r0 = float(selection["by_mode"][primary_mode]["best_radius"])
    probe_pixels = clean_data[args.sessions[0]]["pixels"][:16]
    full = gaussian_lowpass(probe_pixels, r0, mode="full_frame")
    roi = gaussian_lowpass(
        probe_pixels, r0, mode="roi_aware", mask=roi_mask(probe_pixels)
    )
    roi_stats = {
        "images_with_zero_pixels_fraction": (
            float(np.mean([1.0 if f else 0.0 for f in roi_zero_flags]))
        ),
        "mean_roi_fraction": float(np.mean(roi_fractions)),
        "mode_equivalence_max_abs_diff": float((full - roi).abs().max().item()),
        "mode_equivalence_radius": r0,
        "n_probe_images": int(probe_pixels.shape[0]),
    }
    log(f"[ROI] zero-pixel images fraction={roi_stats['images_with_zero_pixels_fraction']:.4f}, "
        f"mean ROI fraction={roi_stats['mean_roi_fraction']:.4f}, "
        f"full-vs-ROI max diff={roi_stats['mode_equivalence_max_abs_diff']:.2e}")

    # ---- write summary CSV/JSON ------------------------------------------
    summary_rows = [
        {field: row.get(field) for field in SUMMARY_FIELDS} for row in records
    ]
    write_csv(output_dir / "summary.csv", summary_rows, SUMMARY_FIELDS)
    log(f"[Output] summary.csv ({len(summary_rows)} rows)")

    write_sample_level(
        output_dir / "sample_level.csv", args.sessions, modes, radii,
        attacks, args.eps, clean_data, adv_data, clean_blocks, adv_blocks, log,
    )

    elapsed = time.perf_counter() - started
    reproduce_command = shlex.join(
        [sys.executable, str(Path(__file__).resolve().relative_to(PROJECT_DIR)),
         *sys.argv[1:]]
    )
    effective_attack = {
        "attacks": attacks,
        "main_attacks": main_attacks,
        "extra_attacks": extra_attacks,
        "epsilons": [float(e) for e in args.eps],
        "primary_epsilon": primary_eps,
        "steps_override": args.steps,
        "batch_size": int(attack_batch_size),
    }
    meta = {
        "created_at": created_at,
        "elapsed_seconds": float(elapsed),
        "command": reproduce_command,
        "reproduce_command": reproduce_command,
        "argv": sys.argv,
        "git_commit": _git_commit(),
        "device": str(device),
        "weights": str(Path(cfg.paths.weights)),
        "weights_sha256": weights_sha,
        "data_root": str(cfg.paths.data_root),
        "manifests": {s: manifests[SESSION_MANIFEST[s]] for s in args.sessions},
        "sessions": list(args.sessions),
        "calibration_session": args.calibration_session,
        "validation_session": args.validation_session,
        "sample_n": int(clean_data[args.sessions[0]]["labels"].size),
        "per_class": int(args.per_class),
        "seed": int(args.seed),
        "radii": radii,
        "filter_modes": modes,
        "attacks": attacks,
        "main_attacks": main_attacks,
        "extra_attacks": extra_attacks,
        "epsilons": [float(e) for e in args.eps],
        "primary_epsilon": primary_eps,
        "primary_epsilon_tag": primary_tag,
        "fpr_target": float(args.fpr_target),
        "filter_implementation": "2-D FFT Gaussian low-pass, radius in FFT bins",
        "preprocessing": "official dataset.NormSingleROI applied exactly once",
        "attack_pipeline": "pixel [0,1] -> existing attack/ -> LPF -> NormSingleROI",
        "effective_attack": effective_attack,
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "selection_criterion": selection["criterion"],
        "roi_stats": roi_stats,
        "clean_margin_stats": {
            s: clean_data[s]["margin_stats"] for s in args.sessions
        },
        "environment": _environment(),
        "config": cfg.to_dict(),
        "session_selection": {
            s: {
                "n": int(clean_data[s]["labels"].size),
                "n_classes": int(len(np.unique(clean_data[s]["labels"]))),
                "indices": clean_data[s]["indices"].tolist(),
            }
            for s in args.sessions
        },
    }

    config_payload = {
        "meta": meta,
        "cli": vars(args),
        "cache_key_meta": key_meta,
        "outputs": {
            "summary_json": "summary.json",
            "summary_csv": "summary.csv",
            "sample_level_csv": "sample_level.csv",
            "report": "REPORT.md",
            "plots": "plots/",
            "logs": "logs/run.log",
            "cache": f"cache/{cache_key}/",
        },
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(jsonify(config_payload), f, indent=2, ensure_ascii=False)
    log("[Output] config.json")

    summary_payload = {
        "meta": meta,
        "selection": selection,
        "validation": validation,
        "attack_comparison": attack_comparison,
        "radius_transfer": radius_transfer,
        "roi_stats": roi_stats,
        "per_radius": records,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonify(summary_payload), f, indent=2, ensure_ascii=False)
    log("[Output] summary.json")

    # ---- plots ------------------------------------------------------------
    figures: Dict[str, str] = {}
    if not args.no_plots:
        best_radius = {
            mode: selection["by_mode"][mode]["best_radius"] for mode in modes
        }
        thresholds = {
            mode: selection["by_mode"][mode]["threshold"] for mode in modes
        }
        fig1 = plots_dir / "consistency_vs_radius.png"
        plot_consistency_vs_radius(
            records, main_attacks, primary_tag, args.sessions, modes, fig1
        )
        figures["Figure 1: consistency vs radius"] = "plots/consistency_vs_radius.png"

        fig2 = plots_dir / "consistency_distribution.png"
        plot_distributions(
            records, sample_scores, main_attacks, primary_tag, args.sessions,
            modes, best_radius, thresholds, fig2,
        )
        figures["Figure 2: distributions at selected radius"] = (
            "plots/consistency_distribution.png"
        )

        fig3 = plots_dir / "delta_consistency.png"
        plot_delta_consistency(
            records, main_attacks, primary_tag, args.sessions, modes, fig3
        )
        figures["Figure 3: Delta_C vs radius"] = "plots/delta_consistency.png"

        fig4 = plots_dir / "roc_curves.png"
        plot_roc_curves(
            records, sample_scores, main_attacks, primary_tag,
            args.validation_session, modes, best_radius, fig4,
            fpr_target=args.fpr_target,
        )
        figures["Figure 4: ROC curves across radii (validation session)"] = (
            "plots/roc_curves.png"
        )

        fig5 = plots_dir / "classification_accuracy.png"
        plot_classification_accuracy(
            records, main_attacks, primary_tag, args.validation_session,
            modes, fig5,
        )
        figures["Figure 5: low-pass filtering vs classification accuracy"] = (
            "plots/classification_accuracy.png"
        )

        fig6 = plots_dir / "auc_vs_radius.png"
        plot_auc_vs_radius(
            records, main_attacks, primary_tag, args.sessions, modes,
            best_radius, fig6,
        )
        figures["Supplementary: AUC vs radius"] = "plots/auc_vs_radius.png"
        log(f"[Output] plots -> {plots_dir}")

    # ---- Markdown report --------------------------------------------------
    write_report(
        output_dir / "REPORT.md", meta, selection, validation, records,
        attack_comparison, roi_stats, figures, radius_transfer,
    )
    log("[Output] REPORT.md")

    # ---- console summary --------------------------------------------------
    log("=" * 78)
    log(f"[Done] elapsed {elapsed:.1f}s")
    for mode in modes:
        entry = selection["by_mode"][mode]
        v = validation["by_mode"][mode]
        log(f"  {mode}: r={entry['best_radius']}, "
            f"cal AUC={entry['mean_auc']:.4f}, "
            f"val AUC={v['mean_auc']}, val TPR@1%={v['mean_tpr']}")
    log(f"  outputs: {output_dir}")
    log("=" * 78)
    log.close()


if __name__ == "__main__":
    main()
