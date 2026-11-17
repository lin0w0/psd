"""
Evaluate the official CompNet checkpoint on the Tongji ROI dataset.

Evaluates the four palmprint identity-matching protocols of the official
``CompNet/test.py``, plus an optional closed-set classification ablation:

    verification EER            session1 gallery vs session2 probes (main)
    aggregated verification EER per-class minimum distance
    dataset EER                 all pairs inside session2
    rank-1 accuracy             nearest-gallery identification
    classification              optional argmax over 600 ArcFace logits

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --device cpu --batch-size 64
    python scripts/evaluate.py --protocols verification rank1 --no-plots
    python scripts/evaluate.py --no-feature-cache --prepare-manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from dataset.dataset import PalmROIDataset  # noqa: E402
from dataset.manifest import (  # noqa: E402
    manifests_match,
    prepare_tongji_manifests,
    read_manifest_meta,
)
from evaluation.classification import evaluate_classification  # noqa: E402
from evaluation.evaluator import (  # noqa: E402
    FeatureSet,
    extract_features,
    run_protocols,
)
from evaluation.report import (  # noqa: E402
    plot_results,
    print_summary,
    write_results,
)
from model.loader import load_compnet, resolve_device  # noqa: E402

PROTOCOL_CHOICES = [
    "classification",
    "verification",
    "aggregated_verification",
    "dataset_eer",
    "rank1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce CompNet recognition results on Tongji ROI."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="YAML config path (default: config/default.yaml).")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root containing session1/ and session2/.")
    parser.add_argument("--weights", type=str, default=None,
                        help="CompNet checkpoint (default: net_params_tongji_600.pth).")
    parser.add_argument("--manifest-dir", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of palm classes (Tongji: 600). "
                             "Updates both data and model config.")
    parser.add_argument("--images-per-class", type=int, default=None,
                        help="Gallery/probe images per class (Tongji: 10).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Results directory (default: ./results).")
    parser.add_argument("--device", type=str, default=None,
                        help="auto | cuda | cuda:0 | cpu.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None,
                        help="Probe chunk size for distance computation.")
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOL_CHOICES,
                        default=None, help="Recognition protocols to run.")
    parser.add_argument("--prepare-manifests", action="store_true",
                        help="Regenerate train.txt / test.txt.")
    parser.add_argument("--no-feature-cache", action="store_true",
                        help="Do not load or save extracted features.")
    parser.add_argument("--save-scores", action="store_true",
                        help="Save genuine/impostor distance arrays as .npz.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable GI / ROC / DET figures.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {"paths": {}, "data": {}, "model": {}, "evaluation": {}}
    if args.data_root:
        overrides["paths"]["data_root"] = args.data_root
    if args.weights:
        overrides["paths"]["weights"] = args.weights
    if args.manifest_dir:
        overrides["paths"]["manifest_dir"] = args.manifest_dir
    if args.output_dir:
        overrides["paths"]["results_root"] = args.output_dir
    if args.num_classes is not None:
        overrides["data"]["num_classes"] = args.num_classes
        overrides["model"]["num_classes"] = args.num_classes
    if args.images_per_class is not None:
        overrides["data"]["images_per_class"] = args.images_per_class
    if args.batch_size is not None:
        overrides["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["data"]["num_workers"] = args.num_workers
    if args.device:
        overrides["evaluation"]["device"] = args.device
    if args.chunk is not None:
        overrides["evaluation"]["probe_chunk"] = args.chunk
    if args.protocols:
        overrides["evaluation"]["protocols"] = args.protocols
    if args.no_plots:
        overrides["evaluation"]["plots"] = False
    if args.save_scores:
        overrides["evaluation"]["save_scores"] = True
    # Drop empty sections
    return {k: v for k, v in overrides.items() if v}


def ensure_manifests(cfg, force: bool = False) -> dict:
    manifest_dir = Path(cfg.paths.manifest_dir)
    train = manifest_dir / "train.txt"
    test = manifest_dir / "test.txt"
    settings = {
        "data_root": cfg.paths.data_root,
        "session1": cfg.paths.session1,
        "session2": cfg.paths.session2,
        "num_classes": cfg.data.num_classes,
        "images_per_class": cfg.data.images_per_class,
        "label_rule": cfg.data.label_rule,
    }

    manifests_exist = train.exists() and test.exists()
    if not force and manifests_exist and manifests_match(manifest_dir, **settings):
        print(f"[Manifests] Using existing: {train} / {test}")
        return {
            "train_manifest": str(train),
            "test_manifest": str(test),
        }

    if manifests_exist and not force:
        if read_manifest_meta(manifest_dir) is None:
            print("[Manifests] Existing manifests have no metadata; "
                  "regenerating ...")
        else:
            print("[Manifests] Existing manifests do not match the current "
                  "data_root/session/class settings; regenerating ...")

    print("[Manifests] Generating from dataset ...")
    summary = prepare_tongji_manifests(
        data_root=cfg.paths.data_root,
        manifest_dir=cfg.paths.manifest_dir,
        **{k: v for k, v in settings.items() if k != "data_root"},
    )
    print(f"[Manifests] gallery={summary['num_gallery']} "
          f"probe={summary['num_probe']} classes={summary['num_classes']} "
          f"images/class={summary['images_per_class']}")
    return summary


def feature_cache_key(
    weights: str | Path,
    manifest: str | Path,
    imside: int,
    outchannels: int,
) -> str:
    """Short cache key covering weights, manifest content and preprocessing.

    Any change in the checkpoint file, the dataset manifest, or the input
    preprocessing produces a different key, so stale features are never
    silently reused.
    """
    digest = hashlib.sha1()
    weights = Path(weights)
    digest.update(str(weights.resolve()).encode("utf-8"))
    stat = weights.stat()
    digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
    digest.update(Path(manifest).read_bytes())
    digest.update(f"{imside}:{outchannels}".encode("ascii"))
    return digest.hexdigest()[:12]


def validate_protocols(protocols) -> list[str]:
    """Reject empty or misspelled protocol lists from YAML overrides."""
    protocols = list(protocols or [])
    if not protocols:
        raise ValueError("evaluation.protocols must not be empty")
    unknown = set(protocols) - set(PROTOCOL_CHOICES)
    if unknown:
        raise ValueError(
            f"Unknown evaluation.protocols: {sorted(unknown)}; "
            f"choices: {PROTOCOL_CHOICES}"
        )
    return protocols


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, build_overrides(args))
    protocols = validate_protocols(cfg.evaluation.protocols)

    print("=" * 70)
    print("CompNet reproduction - Tongji ROI")
    print("=" * 70)
    print(f"  data_root     : {cfg.paths.data_root}")
    print(f"  weights       : {cfg.paths.weights}")
    print(f"  manifests     : {cfg.paths.manifest_dir}")
    print(f"  output        : {cfg.paths.results_root}")
    print(f"  classes       : {cfg.data.num_classes} x {cfg.data.images_per_class}")
    print(f"  protocols     : {', '.join(protocols)}")
    print()

    manifests = ensure_manifests(cfg, force=args.prepare_manifests)

    device = resolve_device(cfg.evaluation.device)
    print(f"[Device] {device}")

    model, device = load_compnet(
        weights=cfg.paths.weights,
        num_classes=cfg.model.num_classes,
        device=str(device),
    )

    gallery_dataset = PalmROIDataset(
        manifests["train_manifest"],
        imside=cfg.data.imside,
        outchannels=cfg.data.outchannels,
    )
    probe_dataset = PalmROIDataset(
        manifests["test_manifest"],
        imside=cfg.data.imside,
        outchannels=cfg.data.outchannels,
    )
    print(f"[Data] gallery={len(gallery_dataset)} probe={len(probe_dataset)} "
          f"image={cfg.data.imside}x{cfg.data.imside} "
          f"channels={cfg.data.outchannels}")

    output_dir = Path(cfg.paths.results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_tag = Path(cfg.paths.weights).stem
    gallery_key = feature_cache_key(
        cfg.paths.weights, manifests["train_manifest"],
        cfg.data.imside, cfg.data.outchannels,
    )
    probe_key = feature_cache_key(
        cfg.paths.weights, manifests["test_manifest"],
        cfg.data.imside, cfg.data.outchannels,
    )
    cache_gallery = output_dir / f"features_session1_{weight_tag}_{gallery_key}.npz"
    cache_probe = output_dir / f"features_session2_{weight_tag}_{probe_key}.npz"
    use_cache = not args.no_feature_cache

    start_all = time.perf_counter()

    needs_gallery = bool(
        set(protocols) & {"verification", "aggregated_verification", "rank1"}
    )

    # ---- gallery features (only when a gallery-based protocol is requested) ----
    gallery = None
    if needs_gallery:
        if use_cache and cache_gallery.exists():
            print(f"[Features] Loading cached gallery features: {cache_gallery}")
            gallery = FeatureSet.load(cache_gallery)
            expected_labels = np.asarray(gallery_dataset.images_label, dtype=np.int64)
            if len(gallery) != len(gallery_dataset) or not np.array_equal(
                gallery.labels, expected_labels
            ):
                raise RuntimeError(
                    f"Feature cache does not match the dataset: {cache_gallery}. "
                    "Rerun with --no-feature-cache."
                )
        else:
            print("[Features] Extracting gallery (session1) features ...")
            gallery = extract_features(
                model, gallery_dataset,
                batch_size=cfg.data.batch_size,
                device=device,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                log_prefix="  gallery ",
            )
            if use_cache:
                gallery.save(cache_gallery)
                print(f"[Features] Saved: {cache_gallery}")
    else:
        print("[Features] No gallery-based protocol requested; "
              "skipping session1 features.")

    # ---- probe features (not needed for classification-only runs) ----
    needs_features = bool(set(protocols) - {"classification"})
    probe = None
    if needs_features:
        if use_cache and cache_probe.exists():
            print(f"[Features] Loading cached probe features: {cache_probe}")
            probe = FeatureSet.load(cache_probe)
            expected_labels = np.asarray(probe_dataset.images_label, dtype=np.int64)
            if len(probe) != len(probe_dataset) or not np.array_equal(
                probe.labels, expected_labels
            ):
                raise RuntimeError(
                    f"Feature cache does not match the dataset: {cache_probe}. "
                    "Rerun with --no-feature-cache."
                )
        else:
            print("[Features] Extracting probe (session2) features ...")
            probe = extract_features(
                model, probe_dataset,
                batch_size=cfg.data.batch_size,
                device=device,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                log_prefix="  probe   ",
            )
            if use_cache:
                probe.save(cache_probe)
                print(f"[Features] Saved: {cache_probe}")
    else:
        print("[Features] Classification only; skipping feature extraction.")

    gallery_shape = "skipped" if gallery is None else str(gallery.features.shape)
    probe_shape = "skipped" if probe is None else str(probe.features.shape)
    print(f"[Features] gallery={gallery_shape} probe={probe_shape}")

    # ---- distance-based protocols ----
    print("\n[Evaluation] Running recognition protocols ...")
    results = run_protocols(
        probe=probe,
        gallery=gallery,
        device=device,
        protocols=protocols,
        chunk=cfg.evaluation.probe_chunk,
        num_classes=cfg.data.num_classes,
        images_per_class=cfg.data.images_per_class,
    )

    # ---- closed-set classification ----
    if "classification" in protocols:
        print("\n[Evaluation] Closed-set classification (argmax over logits) ...")
        results["classification"] = evaluate_classification(
            model,
            probe_dataset,
            device=device,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            num_classes=cfg.data.num_classes,
            log_prefix="  cls     ",
        )

    elapsed = time.perf_counter() - start_all
    print_summary(results, reference=cfg.reference)

    # ---- optional score export ----
    if cfg.evaluation.save_scores:
        for name, g_key, o_key in (
            ("verification", "_verification_genuine", "_verification_impostor"),
            ("aggregated_verification", "_aggregated_genuine", "_aggregated_impostor"),
            ("dataset_eer", "_dataset_genuine", "_dataset_impostor"),
        ):
            if g_key in results:
                score_path = output_dir / f"scores_{name}.npz"
                np.savez_compressed(
                    score_path,
                    genuine=results[g_key],
                    impostor=results[o_key],
                )
                print(f"[Scores] Saved: {score_path}")

    # ---- report ----
    meta = {
        "weights": cfg.paths.weights,
        "data_root": cfg.paths.data_root,
        "device": str(device),
        "gallery_size": 0 if gallery is None else len(gallery),
        # Classification-only runs skip feature extraction but still evaluate
        # every probe in the dataset.
        "probe_size": len(probe_dataset) if probe is None else len(probe),
        "image_size": cfg.data.imside,
        "num_classes": cfg.data.num_classes,
        "images_per_class": cfg.data.images_per_class,
        "protocols": protocols,
        "feature_cache": use_cache,
        "elapsed_seconds": round(elapsed, 2),
    }
    written = write_results(
        results,
        output_dir=output_dir,
        meta=meta,
        reference=cfg.reference,
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
    written["config_json"] = str(output_dir / "config.json")

    if cfg.evaluation.plots:
        print("\n[Plots] Generating figures ...")
        plots = plot_results(results, output_dir)
        written.update({f"plot_{i}": p for i, p in enumerate(plots)})
        print(f"[Plots] {len(plots)} figures -> {output_dir / 'plots'}")

    print("\n[Output]")
    for key, path in written.items():
        print(f"  {key}: {path}")
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
