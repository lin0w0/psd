"""Evaluation protocols, metrics and reporting."""

from .classification import classification_summary, evaluate_classification
from .evaluator import (
    FeatureSet,
    evaluate_dataset_eer,
    evaluate_verification,
    extract_features,
    run_protocols,
    strip_private,
)
from .metrics import compute_eer, gi_histogram, score_statistics, tar_at_far
from .report import plot_results, print_summary, write_results

__all__ = [
    "FeatureSet",
    "evaluate_classification",
    "classification_summary",
    "extract_features",
    "run_protocols",
    "evaluate_verification",
    "evaluate_dataset_eer",
    "strip_private",
    "compute_eer",
    "gi_histogram",
    "score_statistics",
    "tar_at_far",
    "print_summary",
    "write_results",
    "plot_results",
]
