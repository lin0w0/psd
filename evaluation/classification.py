"""Closed-set classification evaluation for CompNet.

The model's ``forward`` returns ArcFace cosine logits ``[N, num_classes]``, so
closed-set palmprint classification is simply ``argmax`` over those logits.
This module computes the metrics used by the classification task:

* top-1 / top-k accuracy
* macro / weighted F1
* per-class accuracy
* the confusion matrix

It works with either a raw :class:`model.compnet.compnet` (inputs already
normalized by ``NormSingleROI``) or an :class:`attack.attacker.CompNetAttacker`
(inputs are pixel tensors in ``[0, 1]``; the wrapper normalizes internally).
"""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def evaluate_classification(
    model: torch.nn.Module,
    dataset,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    num_classes: int = 600,
    topk: Sequence[int] = (1, 5),
    log_prefix: str = "",
) -> Dict:
    """Evaluate closed-set classification over ``dataset``.

    Args:
        model: CompNet, or a wrapper whose ``forward(x)`` returns class logits.
        dataset: Yields ``(input_tensor, int label)``.
        device: Torch device.
        batch_size, num_workers, pin_memory: DataLoader settings.
        num_classes: Number of classes (Tongji: 600).  Extended automatically
            if a larger label is encountered.
        topk: Top-k values to report (``1`` and ``5`` by default).
        log_prefix: Prefix for progress lines.

    Returns:
        Dict with percentage metrics (``top1_accuracy``, ``top5_accuracy``,
        ``macro_f1``, ``per_class_accuracy`` ...), the confusion matrix as
        nested lists, and private ``_predictions`` / ``_labels`` arrays that
        :func:`evaluation.evaluator.strip_private` removes before JSON export.
    """
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
    )

    # Always report top-1 and top-5; honour any additional requested k.
    ks = sorted({1, 5, *(int(k) for k in topk if int(k) > 0)})

    model.eval()
    correct = {k: 0 for k in ks}
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    n_samples = 0
    n_batches = len(loader)
    start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader, 1):
            data = data.to(device, non_blocking=True)
            target = target.to(device)

            logits = model(data)
            if logits.dim() != 2:
                raise RuntimeError(
                    f"model output must be [N, num_classes], got {tuple(logits.shape)}"
                )

            max_k = min(max(ks), logits.shape[1])
            top = logits.topk(max_k, dim=1).indices
            for k in ks:
                kk = min(k, max_k)
                correct[k] += int((top[:, :kk] == target[:, None]).any(1).sum().item())

            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(target.cpu().numpy())
            n_samples += int(target.numel())

            if batch_idx % 20 == 0 or batch_idx == n_batches:
                done = min(batch_idx * batch_size, len(dataset))
                print(
                    f"{log_prefix}[{batch_idx}/{n_batches}] {done}/{len(dataset)} "
                    f"images, {time.perf_counter() - start:.1f}s"
                )

    if n_samples == 0:
        raise ValueError("Empty dataset")

    preds = np.concatenate(predictions).astype(np.int64)
    labels = np.concatenate(targets).astype(np.int64)
    num_classes = int(max(num_classes, int(labels.max()) + 1, int(preds.max()) + 1))

    num_correct = int((preds == labels).sum())
    top1 = 100.0 * num_correct / n_samples

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, preds), 1)
    support = confusion.sum(axis=1)
    per_class = np.divide(
        confusion.diagonal(),
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    per_class_percent = per_class * 100.0

    # Average F1 over classes actually present in the evaluated split.  Using
    # all ``num_classes`` would score absent classes as 0 and make macro-F1 on
    # a stratified subset misleading (e.g. 256 probes -> 344 absent classes).
    present_classes = np.unique(labels)
    macro_f1 = float(
        f1_score(
            labels, preds, labels=present_classes, average="macro", zero_division=0
        )
    )
    weighted_f1 = float(
        f1_score(
            labels, preds, labels=present_classes, average="weighted", zero_division=0
        )
    )

    result = {
        "top1_accuracy": top1,
        "top5_accuracy": 100.0 * correct[5] / n_samples,
        "topk_accuracy": {f"top{k}": 100.0 * c / n_samples for k, c in correct.items()},
        "macro_f1": 100.0 * macro_f1,
        "weighted_f1": 100.0 * weighted_f1,
        "num_samples": int(n_samples),
        "num_correct": num_correct,
        "num_classes": num_classes,
        # Full-length arrays indexed by class id; use per_class_support to
        # mask out classes that do not appear in a stratified subset.
        "per_class_accuracy": per_class_percent.tolist(),
        "per_class_support": support.tolist(),
        "evaluated_class_ids": np.flatnonzero(support > 0).tolist(),
        "per_class_accuracy_mean": float(per_class_percent[support > 0].mean()),
        "per_class_accuracy_min": float(per_class_percent[support > 0].min()),
        "per_class_accuracy_max": float(per_class_percent[support > 0].max()),
        "num_classes_evaluated": int((support > 0).sum()),
        "confusion_matrix": confusion.tolist(),
        # private arrays (stripped before JSON export)
        "_predictions": preds,
        "_labels": labels,
    }
    return result


def classification_summary(result: Dict) -> Dict[str, float]:
    """Scalar subset of a classification result (safe for CSV/JSON rows)."""
    return {
        "top1_accuracy": float(result["top1_accuracy"]),
        "top5_accuracy": float(result["top5_accuracy"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_f1": float(result["weighted_f1"]),
        "num_correct": int(result["num_correct"]),
        "num_samples": int(result["num_samples"]),
    }
