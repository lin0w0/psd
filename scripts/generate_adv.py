"""
Generate adversarial palmprint probes and measure the recognition drop.

This is the CompNet counterpart of CSR's ``scripts/generate_adv.py``.  It
runs one or more pixel-space identity-classification attacks from
:mod:`attack` on session2 probes and re-evaluates classification ACC plus
the auxiliary recognition protocols.

Pipeline::

    ROI -> Resize -> ToTensor (pixels in [0,1]) -> attack adds perturbation
        -> NormSingleROI -> CompNet -> ArcFace logits
        -> classification loss (CE / CW logit margin)

Final evaluation is classification accuracy (ACC).  The default
``--loss ce`` uses cross-entropy for PGD/APGD/Patch and the CW logit margin
for CW; ``dlr`` runs the untargeted APGD-DLR objective for PGD/APGD/Patch;
``cw`` forces the CW margin for all L-inf attacks; ``matching`` is the
optional gallery genuine/impostor margin ablation.  ``--eps`` is an L-inf
budget in pixel units, matching CSR (1/255, 2/255, 4/255 by default).
``NormSingleROI`` is applied inside ``CompNetAttacker``.  CW defaults to
zero initialization (``build_attack``'s per-attack policy): its hinge has zero
gradient once a sample succeeds, so a random start would be frozen into the
final perturbation as high-frequency noise.  PGD/APGD/Patch keep random starts.

Usage:
    python scripts/generate_adv.py --attack pgd --sample-n 1000
    python scripts/generate_adv.py --attack apgd --loss dlr --sample-n 1000
    python scripts/generate_adv.py --attack apgd cw --sample-n 1000
    python scripts/generate_adv.py --attack pgd --loss matching --sample-n 1000
    python scripts/generate_adv.py --attack patch --save-adv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from attack import (  # noqa: E402
    APGDAttack,
    CompNetAttacker,
    CWAttack,
    PatchAttack,
    PGDAttack,
)
from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from dataset.dataset import PalmROIDataset  # noqa: E402
from dataset.manifest import (  # noqa: E402
    manifests_match,
    prepare_tongji_manifests,
)
from evaluation.classification import (  # noqa: E402
    classification_summary,
    evaluate_classification,
)
from evaluation.evaluator import (  # noqa: E402
    FeatureSet,
    extract_features,
    run_protocols,
)
from model.loader import load_compnet, resolve_device  # noqa: E402

ATTACK_NAMES = ["pgd", "apgd", "cw", "patch"]
NORM_BOUNDED = {"pgd", "apgd", "cw"}
LOSS_CHOICES = ["matching", "ce", "cw", "dlr"]
PROTOCOL_CHOICES = [
    "classification",
    "verification",
    "aggregated_verification",
    "dataset_eer",
    "rank1",
]


# ======================== Data helpers ========================

class SubsetDataset(Dataset):
    """Deterministic view over a :class:`PalmROIDataset` (preserves metadata)."""

    def __init__(self, base: PalmROIDataset, indices: Sequence[int]):
        self.base = base
        self.indices = [int(i) for i in indices]
        self.images_path = [base.images_path[i] for i in self.indices]
        self.images_label = [base.images_label[i] for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base[self.indices[index]]


class TensorFeatureDataset(Dataset):
    """Feature-extraction view over pre-generated adversarial tensors."""

    def __init__(self, tensors: torch.Tensor, labels: np.ndarray, paths: List[str]):
        self.tensors = tensors
        self.labels = np.asarray(labels, dtype=np.int64)
        self.images_path = list(paths)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.tensors[index], int(self.labels[index])


def ensure_manifests(cfg) -> dict:
    """Load existing manifests or regenerate them (same rules as evaluate.py)."""
    manifest_dir = Path(cfg.paths.manifest_dir)
    train = manifest_dir / "train.txt"
    test = manifest_dir / "test.txt"
    settings = {
        "session1": cfg.paths.session1,
        "session2": cfg.paths.session2,
        "num_classes": cfg.data.num_classes,
        "images_per_class": cfg.data.images_per_class,
        "label_rule": cfg.data.label_rule,
    }
    if not (train.exists() and test.exists()) or not manifests_match(
        manifest_dir, data_root=cfg.paths.data_root, **settings
    ):
        print("[Manifests] Regenerating ...")
        prepare_tongji_manifests(
            data_root=cfg.paths.data_root, manifest_dir=cfg.paths.manifest_dir, **settings
        )
    return {"train_manifest": str(train), "test_manifest": str(test)}


def _evenly_spaced(count: int, total: int) -> List[int]:
    """``count`` distinct integers spread evenly over ``[0, total-1]``."""
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    return np.unique(np.linspace(0, total - 1, count).round().astype(int)).tolist()


def select_probe_indices(
    labels: Sequence[int],
    sample_n: Optional[int],
    num_classes: int,
    images_per_class: int,
) -> np.ndarray:
    """Stratified subset spread across **all** classes.

    When ``sample_n < num_classes`` one probe is taken from ``sample_n``
    classes spread evenly over the full class range (not just the first
    ``sample_n`` classes).  When ``sample_n >= num_classes`` each class gets
    ``sample_n // num_classes`` probes, and the first ``sample_n % num_classes``
    extras are spread evenly over the class range as well.
    """
    n = len(labels)
    if sample_n is None or sample_n <= 0 or sample_n >= n:
        return np.arange(n)

    if n == num_classes * images_per_class:
        base, remainder = divmod(sample_n, num_classes)
        indices: List[int] = []

        if base == 0:
            for c in _evenly_spaced(remainder, num_classes):
                indices.append(c * images_per_class)
            return np.asarray(indices[:sample_n], dtype=np.int64)

        extra_classes = set(_evenly_spaced(remainder, num_classes))
        for c in range(num_classes):
            k = min(base + (1 if c in extra_classes else 0), images_per_class)
            start = c * images_per_class
            indices.extend(range(start, start + k))
        return np.asarray(indices[:sample_n], dtype=np.int64)

    # Fallback for non-standard manifests: evenly spaced indices.
    return np.linspace(0, n - 1, sample_n).astype(np.int64)


# ======================== Attack helpers ========================

def resolve_attack_loss(name: str, loss: str) -> str:
    """Map the run-level loss to the per-attack objective.

    ``loss="ce"`` (classification ACC default) applies CE to PGD/APGD/Patch;
    ``loss="dlr"`` applies the untargeted DLR to PGD/APGD/Patch.  CW always
    uses its logit margin for those two run-level losses, because the CW hinge
    is defined on that margin.  ``loss="cw"`` / ``"matching"`` apply uniformly.
    """
    if loss in ("ce", "dlr") and name == "cw":
        return "cw"
    return loss


def build_attack(
    name: str,
    attacker: CompNetAttacker,
    epsilon: float,
    cfg,
    loss: str,
    gallery_features: torch.Tensor,
    gallery_labels: torch.Tensor,
    random_start: Optional[bool] = None,
):
    """Build an attack instance.

    ``random_start=None`` selects the per-attack default: random initialization
    for PGD/APGD/Patch and zero initialization for CW.  CW's hinge loss has a
    zero gradient once a sample crosses the margin, so a random start would be
    frozen into the final perturbation (uniform noise, spurious high-frequency
    content).  Pass ``True``/``False`` to override for all attacks.
    """
    a = cfg.attack
    steps_override = getattr(cfg, "_steps_override", None)
    if random_start is None:
        random_start = name != "cw"
    else:
        random_start = bool(random_start)
    common = dict(
        loss=loss,
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        random_start=random_start,
    )

    if name == "pgd":
        return PGDAttack(
            attacker,
            epsilon=epsilon,
            steps=steps_override or a.pgd_steps,
            alpha_scale=a.pgd_alpha_scale,
            **common,
        )
    if name == "apgd":
        return APGDAttack(
            attacker,
            epsilon=epsilon,
            steps=steps_override or a.apgd_steps,
            **common,
        )
    if name == "cw":
        return CWAttack(
            attacker,
            epsilon=epsilon,
            steps=steps_override or a.cw_steps,
            alpha_scale=a.cw_alpha_scale,
            kappa=a.cw_kappa,
            **common,
        )
    if name == "patch":
        return PatchAttack(
            attacker,
            patch_size=a.patch_size,
            steps=a.patch_steps,
            lr=a.patch_lr,
            **common,
        )
    raise ValueError(f"Unknown attack: {name}")


def generate_adversarial(
    attack,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks: List[torch.Tensor] = []
    n_batches = len(loader)
    start = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(loader, 1):
        images = images.to(device)
        labels = labels.to(device)
        adv = attack.attack(images, labels)
        chunks.append(adv.detach().cpu())

        if batch_idx % 10 == 0 or batch_idx == n_batches:
            done = min(batch_idx * batch_size, len(dataset))
            print(
                f"{log_prefix}[{batch_idx}/{n_batches}] {done}/{len(dataset)} "
                f"images, {time.perf_counter() - start:.1f}s"
            )

    return torch.cat(chunks, dim=0)


def feature_set_from_tensors(
    model,
    tensors: torch.Tensor,
    labels: np.ndarray,
    paths: List[str],
    batch_size: int,
    device: torch.device,
    log_prefix: str = "",
) -> FeatureSet:
    dataset = TensorFeatureDataset(tensors, labels, paths)
    return extract_features(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        num_workers=0,
        pin_memory=False,
        log_prefix=log_prefix,
    )


def protocol_record(results: Dict, prefix: str) -> Dict[str, float]:
    record: Dict[str, float] = {}
    if "verification" in results:
        v = results["verification"]
        record[f"{prefix}verification_eer_percent"] = v["eer_percent"]
        record[f"{prefix}verification_rank1"] = v["rank1_accuracy"]
        for far in (1e-2, 1e-3, 1e-4):
            key = f"tar_at_far_{far:g}"
            if key in v:
                record[f"{prefix}verification_{key}"] = v[key] * 100.0
    if "aggregated_verification" in results:
        record[f"{prefix}aggregated_eer_percent"] = results[
            "aggregated_verification"
        ]["eer_percent"]
    if "dataset_eer" in results:
        record[f"{prefix}dataset_eer_percent"] = results["dataset_eer"]["eer_percent"]
    if "rank1" in results:
        record[f"{prefix}rank1"] = results["rank1"]["accuracy"]
    return record


# ======================== CLI ========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial probes for CompNet and evaluate the drop."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--attack", nargs="+", choices=ATTACK_NAMES + ["all"],
                        default=["pgd"], help="Attack(s) to run.")
    parser.add_argument("--loss", choices=LOSS_CHOICES, default=None,
                        help="Attack objective: 'ce' (default, classification "
                             "ACC; CW uses its logit margin), 'dlr' (untargeted "
                             "APGD-DLR for PGD/APGD/Patch), 'cw' (logit margin "
                             "for all), or 'matching' (gallery ablation).")
    parser.add_argument("--eps", nargs="+", type=float, default=None,
                        help="L-inf budgets on pixel tensors in [0,1] "
                             "(default: config attack.epsilons, 1/255..4/255).")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override iterations for pgd/apgd/cw.")
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Override attack initialization. "
                             "--random-start forces uniform random init for all "
                             "attacks; --no-random-start forces delta=0. "
                             "Default: per-attack (CW zero-init, PGD/APGD/Patch "
                             "random), because CW's hinge has zero gradient "
                             "after success and would freeze random noise into "
                             "the final perturbation.")
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Probes to attack (stratified, default: config). "
                             "Use 0 for all 6000.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Base output directory "
                             "(default: ./outputs/adv_samples).")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Per-run subdirectory name (default: timestamp). "
                             "Prevents overwriting previous run results.")
    parser.add_argument("--protocols", nargs="+",
                        choices=PROTOCOL_CHOICES, default=None,
                        help="Distance-based protocols to report. Classification "
                             "metrics are always computed as an auxiliary ablation.")
    parser.add_argument("--save-adv", action="store_true",
                        help="Save adversarial tensors (.npz) per attack/epsilon.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    overrides: Dict = {"evaluation": {}}
    if args.device:
        overrides["evaluation"]["device"] = args.device
    cfg = load_config(args.config, overrides)

    if args.steps is not None:
        cfg._steps_override = int(args.steps)

    epsilons = args.eps or list(cfg.attack.epsilons)
    sample_n = cfg.attack.sample_n if args.sample_n is None else args.sample_n
    batch_size = args.batch_size or cfg.attack.batch_size
    base_output_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_DIR / "outputs" / "adv_samples"
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_names = ATTACK_NAMES if "all" in args.attack else args.attack
    loss = args.loss or cfg.attack.loss
    if loss not in LOSS_CHOICES:
        raise ValueError(f"Unknown loss: {loss!r}; choices: {LOSS_CHOICES}")

    print("=" * 70)
    print("CompNet adversarial generation - Tongji ROI (identity classification)")
    print("=" * 70)
    print(f"  attacks     : {', '.join(attack_names)}")
    if loss == "ce":
        loss_note = "CE for PGD/APGD/Patch, CW margin for CW"
    elif loss == "dlr":
        loss_note = "untargeted DLR for PGD/APGD/Patch, CW margin for CW"
    else:
        loss_note = loss
    print(f"  loss        : {loss} ({loss_note})")
    print(f"  epsilons    : {epsilons} (pixel-space L-inf budget, [0,1])")
    if args.random_start is None:
        init_note = "per-attack (CW delta=0; PGD/APGD/Patch random)"
    elif args.random_start:
        init_note = "forced random (uniform)"
    else:
        init_note = "forced zero (delta=0)"
    print(f"  init        : {init_note}")
    print(f"  sample_n    : {sample_n if sample_n else 'all'}")
    print(f"  output      : {output_dir}")

    manifests = ensure_manifests(cfg)
    device = resolve_device(cfg.evaluation.device)
    print(f"[Device] {device}")

    model, device = load_compnet(
        weights=cfg.paths.weights,
        num_classes=cfg.model.num_classes,
        device=str(device),
    )
    attacker = CompNetAttacker(
        model,
        device=device,
        preserve_background=cfg.attack.preserve_background,
        input_range=cfg.attack.input_range,
    )

    # Pixel tensors ([0, 1]); NormSingleROI lives inside CompNetAttacker.
    pixel_transform = T.Compose([T.Resize(cfg.data.imside), T.ToTensor()])
    gallery_dataset = PalmROIDataset(
        manifests["train_manifest"],
        imside=cfg.data.imside,
        outchannels=cfg.data.outchannels,
        transform=pixel_transform,
    )
    full_probe = PalmROIDataset(
        manifests["test_manifest"],
        imside=cfg.data.imside,
        outchannels=cfg.data.outchannels,
        transform=pixel_transform,
    )
    indices = select_probe_indices(
        full_probe.images_label,
        sample_n,
        cfg.data.num_classes,
        cfg.data.images_per_class,
    )
    probe_dataset = SubsetDataset(full_probe, indices)
    print(f"[Data] gallery={len(gallery_dataset)} probe={len(probe_dataset)} "
          f"image={cfg.data.imside}x{cfg.data.imside}, pixel range [0,1]")

    protocols = list(args.protocols or cfg.evaluation.protocols)
    if not protocols:
        raise ValueError("protocols must not be empty")
    unknown = set(protocols) - set(PROTOCOL_CHOICES)
    if unknown:
        raise ValueError(
            f"Unknown protocols: {sorted(unknown)}; choices: {PROTOCOL_CHOICES}"
        )

    # dataset_eer needs at least one genuine (same-class) probe pair.  A
    # one-probe-per-class stratified subset cannot supply one; skip it for the
    # default protocol set, but reject an explicit request instead of silently
    # dropping it.
    skipped_protocols: List[str] = []
    per_class_counts = np.bincount(
        np.asarray(probe_dataset.images_label, dtype=np.int64),
        minlength=cfg.data.num_classes,
    )
    has_genuine_pairs = bool(per_class_counts.max() >= 2)
    if "dataset_eer" in protocols and not has_genuine_pairs:
        if args.protocols is not None:
            raise ValueError(
                "dataset_eer requires at least two probes from some class; "
                f"use --sample-n >= {cfg.data.num_classes + 1} or omit dataset_eer."
            )
        protocols = [p for p in protocols if p != "dataset_eer"]
        skipped_protocols.append("dataset_eer")
        print(
            "[Protocols] dataset_eer skipped: the stratified subset has fewer "
            "than two probes for every class."
        )

    # ---- gallery features (clean, fixed) ----
    # Always needed: it is both the evaluation gallery and the reference for
    # the identity-matching attack loss.
    print("\n[Features] Extracting clean gallery features ...")
    gallery = extract_features(
        attacker, gallery_dataset,
        batch_size=cfg.data.batch_size,
        device=device,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        log_prefix="  gallery ",
    )
    gallery_features_t = torch.as_tensor(gallery.features, device=device)
    gallery_labels_t = torch.as_tensor(gallery.labels, device=device)

    # ---- clean probes & baseline ----
    print("\n[Features] Extracting clean probe features ...")
    clean_probe = extract_features(
        attacker, probe_dataset,
        batch_size=cfg.data.batch_size,
        device=device,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        log_prefix="  probe   ",
    )

    baseline = run_protocols(
        probe=clean_probe,
        gallery=gallery,
        device=device,
        protocols=protocols,
        chunk=cfg.evaluation.probe_chunk,
        num_classes=cfg.data.num_classes,
        images_per_class=cfg.data.images_per_class,
    )
    baseline_record = protocol_record(baseline, prefix="clean_")

    # ---- clean closed-set classification ----
    print("\n[Classification] Clean baseline (argmax over ArcFace logits) ...")
    clean_classification = evaluate_classification(
        attacker,
        probe_dataset,
        device=device,
        batch_size=cfg.data.batch_size,
        num_workers=0,
        pin_memory=False,
        num_classes=cfg.data.num_classes,
        log_prefix="  cls     ",
    )
    baseline_record.update(
        {
            f"clean_{k}": v
            for k, v in classification_summary(clean_classification).items()
        }
    )

    print("\n[Baseline]")
    for key, value in baseline_record.items():
        print(f"  {key}: {value:.6f}")

    # ---- attacks ----
    records: List[Dict] = []
    for name in attack_names:
        runs: List[Optional[float]] = (
            list(epsilons) if name in NORM_BOUNDED else [None]
        )
        for epsilon in runs:
            eps_tag = "none" if epsilon is None else f"{epsilon:.5f}".rstrip("0").rstrip(".")
            label = f"{name}" + ("" if epsilon is None else f"@eps={epsilon}")
            print("\n" + "-" * 70)
            print(f"[Attack] {label}")
            start = time.perf_counter()

            attack_loss = resolve_attack_loss(name, loss)
            attack = build_attack(
                name,
                attacker,
                0.0 if epsilon is None else epsilon,
                cfg,
                attack_loss,
                gallery_features_t,
                gallery_labels_t,
                random_start=args.random_start,
            )
            adv = generate_adversarial(
                attack, probe_dataset, batch_size, device, log_prefix="  adv "
            )
            elapsed_attack = time.perf_counter() - start

            adv_features = feature_set_from_tensors(
                attacker, adv, clean_probe.labels, clean_probe.paths,
                batch_size=cfg.data.batch_size, device=device,
                log_prefix="  feat ",
            )
            results = run_protocols(
                probe=adv_features,
                gallery=gallery,
                device=device,
                protocols=protocols,
                chunk=cfg.evaluation.probe_chunk,
                num_classes=cfg.data.num_classes,
                images_per_class=cfg.data.images_per_class,
            )

            # Closed-set classification metrics on the adversarial pixels.
            adv_classification = evaluate_classification(
                attacker,
                TensorFeatureDataset(adv, clean_probe.labels, clean_probe.paths),
                device=device,
                batch_size=cfg.data.batch_size,
                num_workers=0,
                pin_memory=False,
                num_classes=cfg.data.num_classes,
                log_prefix="  cls     ",
            )
            adv_cls = {
                f"adv_{k}": v
                for k, v in classification_summary(adv_classification).items()
            }
            clean_correct = int(baseline_record.get("clean_num_correct", 0))
            adv_correct = int(adv_cls["adv_num_correct"])
            attack_success_rate = (
                100.0 * max(0, clean_correct - adv_correct) / max(1, clean_correct)
            )
            top1_drop = (
                float(baseline_record["clean_top1_accuracy"])
                - float(adv_cls["adv_top1_accuracy"])
            )

            record = {
                "attack": name,
                "loss": attack_loss,
                "epsilon": epsilon,
                "sample_n": len(probe_dataset),
                "attack_seconds": round(elapsed_attack, 2),
                **baseline_record,
                **protocol_record(results, prefix="adv_"),
                **adv_cls,
                "top1_drop": top1_drop,
                "attack_success_rate": attack_success_rate,
            }
            records.append(record)

            print(
                f"  ACC (top-1)      : {baseline_record['clean_top1_accuracy']:.2f}%"
                f" -> {adv_cls['adv_top1_accuracy']:.2f}%"
                f"  (drop {top1_drop:.2f} pp, ASR {attack_success_rate:.2f}%)"
            )
            print(
                f"  top-5 accuracy   : {baseline_record['clean_top5_accuracy']:.2f}%"
                f" -> {adv_cls['adv_top5_accuracy']:.2f}%"
            )
            clean_eer = baseline_record.get("clean_verification_eer_percent")
            adv_eer = record.get("adv_verification_eer_percent")
            clean_r1 = baseline_record.get("clean_verification_rank1")
            adv_r1 = record.get("adv_verification_rank1")
            if adv_eer is not None:
                print(f"  [aux] EER        : {clean_eer:.4f}% -> {adv_eer:.4f}%")
            if adv_r1 is not None:
                print(f"  [aux] rank-1     : {clean_r1:.2f}% -> {adv_r1:.2f}%")

            if args.save_adv:
                adv_path = output_dir / f"adv_{name}_{eps_tag}.npz"
                np.savez_compressed(
                    adv_path,
                    adv=adv.numpy().astype(np.float32),
                    labels=clean_probe.labels,
                    indices=indices,
                    paths=np.asarray(clean_probe.paths, dtype=object),
                )
                print(f"  saved            : {adv_path}")

    # ---- write summary ----
    if records:
        # Record the values actually used, not just the static YAML defaults.
        # Full baseline confusion matrix / per-class accuracy go to a compact
        # .npz; keep adv_results.json small (scalars only).
        baseline_cls_path = output_dir / "baseline_confusion_matrix.npz"
        np.savez_compressed(
            baseline_cls_path,
            confusion_matrix=np.asarray(
                clean_classification["confusion_matrix"], dtype=np.int64
            ),
            per_class_accuracy=np.asarray(
                clean_classification["per_class_accuracy"], dtype=np.float32
            ),
            per_class_support=np.asarray(
                clean_classification["per_class_support"], dtype=np.int64
            ),
            evaluated_class_ids=np.asarray(
                clean_classification["evaluated_class_ids"], dtype=np.int64
            ),
        )

        effective_cfg = cfg.to_dict()
        effective_cfg["attack"]["loss"] = loss
        effective_cfg["attack"]["epsilons"] = [float(e) for e in epsilons]
        effective_cfg["attack"]["sample_n"] = len(probe_dataset)
        effective_cfg["attack"]["batch_size"] = int(batch_size)
        effective_cfg["evaluation"]["protocols"] = list(protocols)
        if args.steps is not None:
            effective_cfg["attack"]["pgd_steps"] = int(args.steps)
            effective_cfg["attack"]["apgd_steps"] = int(args.steps)
            effective_cfg["attack"]["cw_steps"] = int(args.steps)

        json_path = output_dir / "adv_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "run_name": run_name,
                        "seed": args.seed,
                        "weights": cfg.paths.weights,
                        "data_root": cfg.paths.data_root,
                        "device": str(device),
                        "sample_n": len(probe_dataset),
                        "epsilons": epsilons,
                        "protocols": protocols,
                        "skipped_protocols": skipped_protocols,
                        "input_domain": "pixel [0,1] -> NormSingleROI -> CompNet",
                        "effective_attack": {
                            "loss": loss,
                            "epsilons": [float(e) for e in epsilons],
                            "sample_n": len(probe_dataset),
                            "batch_size": int(batch_size),
                            "steps_override": args.steps,
                            "random_start": args.random_start,
                        },
                        "config": effective_cfg,
                    },
                    "baseline": baseline_record,
                    "baseline_classification": classification_summary(
                        clean_classification
                    ),
                    "results": records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        csv_path = output_dir / "adv_summary.csv"
        fieldnames = sorted({k for r in records for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"\n[Output] {json_path}")
        print(f"[Output] {csv_path}")
        print(f"[Output] {baseline_cls_path}")


if __name__ == "__main__":
    main()
