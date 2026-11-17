"""
Feature extraction and the CompNet recognition protocols.

Protocols (all from the official ``test.py``):

1. ``verification``             - session1 gallery vs session2 probes, EER.
2. ``aggregated_verification``  - per-probe min distance over each class's
                                  gallery images, then EER.
3. ``dataset_eer``              - all pairs inside session2, EER.
4. ``rank1``                    - nearest-gallery identification accuracy.

Matching distance is ``arccos(cosine) / pi`` on L2-normalized feature codes,
exactly as in the official implementation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import compute_eer, score_statistics, tar_at_far


# ======================== Features ========================

@dataclass
class FeatureSet:
    """Feature codes for one session."""

    features: np.ndarray                    # [N, 512] float32, unit norm
    labels: np.ndarray                      # [N] int64
    paths: List[str]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            features=self.features.astype(np.float32),
            labels=self.labels.astype(np.int64),
            paths=np.asarray(self.paths, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureSet":
        data = np.load(path, allow_pickle=True)
        return cls(
            features=data["features"].astype(np.float32),
            labels=data["labels"].astype(np.int64),
            paths=[str(p) for p in data["paths"].tolist()],
        )


def extract_features(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
    num_workers: int = 0,
    pin_memory: bool = True,
    log_prefix: str = "",
) -> FeatureSet:
    """Run the official ``getFeatureCode`` over a dataset, keeping file order."""
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
    )

    model.eval()
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    n_batches = len(loader)
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader, 1):
            data = data.to(device, non_blocking=True)
            codes = model.getFeatureCode(data)
            features.append(codes.float().cpu().numpy())
            labels.append(target.numpy())

            if batch_idx % 20 == 0 or batch_idx == n_batches:
                elapsed = time.perf_counter() - start_time
                print(f"{log_prefix}[{batch_idx}/{n_batches}] "
                      f"{batch_idx * batch_size if batch_idx < n_batches else len(dataset)}"
                      f"/{len(dataset)} images, {elapsed:.1f}s")

    feats = np.concatenate(features, axis=0)
    labs = np.concatenate(labels, axis=0)

    if not np.isfinite(feats).all():
        raise RuntimeError("Non-finite feature values encountered.")
    norms = np.linalg.norm(feats, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError(
            f"Feature codes are not L2-normalized (min={norms.min():.6f}, "
            f"max={norms.max():.6f})."
        )

    return FeatureSet(features=feats, labels=labs, paths=list(dataset.images_path))


# ======================== Distance helpers ========================

def _as_device_tensor(features: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        np.ascontiguousarray(features, dtype=np.float32), device=device
    )


def _arccos_distance(query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """Matching distance in [0, 1]; smaller = more similar."""
    cosine = query @ gallery.T
    return torch.acos(cosine.clamp(-1.0, 1.0)) / math.pi


# ======================== Protocols ========================

def evaluate_verification(
    probe: FeatureSet,
    gallery: FeatureSet,
    device: torch.device | str,
    chunk: int = 256,
    num_classes: int = 600,
    images_per_class: int = 10,
) -> Dict[str, Dict]:
    """Verification EER + rank-1 + aggregated verification EER.

    The gallery must be ordered by class with exactly ``images_per_class``
    consecutive images per class (the official manifest layout).
    """
    device = torch.device(device)
    g_labels = gallery.labels.astype(np.int64)
    p_labels = probe.labels.astype(np.int64)
    n_gallery = len(gallery)

    if len(gallery) != num_classes * images_per_class:
        raise ValueError(
            f"Gallery has {len(gallery)} images, expected "
            f"{num_classes * images_per_class}."
        )

    class_labels = g_labels[::images_per_class]
    if not np.array_equal(np.repeat(class_labels, images_per_class), g_labels):
        raise ValueError(
            "Gallery labels are not contiguous blocks; run prepare_data.py."
        )
    if not np.array_equal(class_labels, np.arange(num_classes)):
        raise ValueError("Gallery class labels must be 0..num_classes-1.")

    gallery_t = _as_device_tensor(gallery.features, device)

    genuine: List[np.ndarray] = []
    impostor: List[np.ndarray] = []
    agg_genuine: List[np.ndarray] = []
    agg_impostor: List[np.ndarray] = []
    correct = 0

    for start in range(0, len(probe), chunk):
        end = min(start + chunk, len(probe))
        query = _as_device_tensor(probe.features[start:end], device)

        dist = _arccos_distance(query, gallery_t)
        dist_np = dist.float().cpu().numpy()

        # --- standard verification ---
        mask = p_labels[start:end, None] == g_labels[None, :]
        genuine.append(dist_np[mask])
        impostor.append(dist_np[~mask])

        # --- rank-1 identification ---
        nearest = np.argmin(dist_np, axis=1)
        correct += int((g_labels[nearest] == p_labels[start:end]).sum())

        # --- aggregated verification (official test.py) ---
        # Each probe vs each class: min distance over the class's 10 images.
        agg = dist_np.reshape(end - start, num_classes, images_per_class).min(axis=2)
        agg_mask = p_labels[start:end, None] == class_labels[None, :]
        agg_genuine.append(agg[agg_mask])
        agg_impostor.append(agg[~agg_mask])

        del query, dist, dist_np

    genuine_arr = np.concatenate(genuine)
    impostor_arr = np.concatenate(impostor)
    verification = compute_eer(genuine_arr, impostor_arr)
    verification["score_statistics"] = {
        "genuine": score_statistics(genuine_arr),
        "impostor": score_statistics(impostor_arr),
    }
    for far in (1e-2, 1e-3, 1e-4):
        verification[f"tar_at_far_{far:g}"] = tar_at_far(
            genuine_arr, impostor_arr, far
        )
    verification["rank1_accuracy"] = 100.0 * correct / len(probe)

    agg_genuine_arr = np.concatenate(agg_genuine)
    agg_impostor_arr = np.concatenate(agg_impostor)
    aggregated = compute_eer(agg_genuine_arr, agg_impostor_arr)
    aggregated["score_statistics"] = {
        "genuine": score_statistics(agg_genuine_arr),
        "impostor": score_statistics(agg_impostor_arr),
    }

    return {
        "verification": verification,
        "aggregated_verification": aggregated,
        "rank1": {
            "accuracy": 100.0 * correct / len(probe),
            "num_probe": len(probe),
            "num_gallery": n_gallery,
            "num_correct": int(correct),
        },
        # private arrays for plotting
        "_genuine": genuine_arr,
        "_impostor": impostor_arr,
        "_agg_genuine": agg_genuine_arr,
        "_agg_impostor": agg_impostor_arr,
    }


def evaluate_dataset_eer(
    features: FeatureSet,
    device: torch.device | str,
    chunk: int = 256,
) -> Dict:
    """EER over all probe-pair combinations inside a single session.

    This is the official "EER of the test set" protocol (gallery not used):
    every pair ``i < j`` is matched, genuine if labels are equal.
    """
    device = torch.device(device)
    n = len(features)
    labels = features.labels.astype(np.int64)
    feats_t = _as_device_tensor(features.features, device)

    col_idx = np.arange(n, dtype=np.int64)
    dist_chunks: List[np.ndarray] = []
    genuine_chunks: List[np.ndarray] = []

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        query = feats_t[start:end]
        dist = _arccos_distance(query, feats_t).float().cpu().numpy()

        keep = col_idx[None, :] > np.arange(start, end, dtype=np.int64)[:, None]
        dist_chunks.append(dist[keep])
        genuine_chunks.append((labels[start:end, None] == labels[None, :])[keep])

        del query, dist

    distances = np.concatenate(dist_chunks)
    genuine_mask = np.concatenate(genuine_chunks)

    result = compute_eer(distances[genuine_mask], distances[~genuine_mask])
    result["score_statistics"] = {
        "genuine": score_statistics(distances[genuine_mask]),
        "impostor": score_statistics(distances[~genuine_mask]),
    }
    result["num_pairs"] = int(distances.size)
    result["_genuine"] = distances[genuine_mask]
    result["_impostor"] = distances[~genuine_mask]
    return result


def run_protocols(
    probe: FeatureSet,
    gallery: FeatureSet,
    device: torch.device | str,
    protocols: Optional[Sequence[str]] = None,
    chunk: int = 256,
    num_classes: int = 600,
    images_per_class: int = 10,
) -> Dict[str, Dict]:
    """Run the requested protocols and return their results."""
    # An explicit empty list means "no protocols"; only ``None`` selects the
    # legacy defaults.  (``protocols or [...]`` would silently turn [] into
    # the four defaults and crash on a classification-only run.)
    protocols = set(protocols) if protocols is not None else {
        "verification", "aggregated_verification", "dataset_eer", "rank1"
    }
    results: Dict[str, Dict] = {}

    needs_gallery = protocols & {"verification", "aggregated_verification", "rank1"}
    if needs_gallery:
        out = evaluate_verification(
            probe, gallery,
            device=device,
            chunk=chunk,
            num_classes=num_classes,
            images_per_class=images_per_class,
        )
        if "verification" in protocols:
            results["verification"] = out["verification"]
            results["_verification_genuine"] = out["_genuine"]
            results["_verification_impostor"] = out["_impostor"]
        if "aggregated_verification" in protocols:
            results["aggregated_verification"] = out["aggregated_verification"]
            results["_aggregated_genuine"] = out["_agg_genuine"]
            results["_aggregated_impostor"] = out["_agg_impostor"]
        if "rank1" in protocols:
            results["rank1"] = out["rank1"]

    if "dataset_eer" in protocols:
        out = evaluate_dataset_eer(probe, device=device, chunk=chunk)
        results["dataset_eer"] = out
        results["_dataset_genuine"] = out.pop("_genuine")
        results["_dataset_impostor"] = out.pop("_impostor")

    return results


def strip_private(results: Dict) -> Dict:
    """Remove plotting arrays and non-serializable private entries."""
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if not k.startswith("_")}
        return obj

    return clean(results)
