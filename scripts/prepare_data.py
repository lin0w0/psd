"""
Generate train.txt / test.txt manifests for the Tongji ROI dataset.

Official layout:
    data/ROI/session1/00001.bmp ... 06000.bmp   -> gallery (train.txt)
    data/ROI/session2/00001.bmp ... 06000.bmp   -> probe   (test.txt)
    label = (int(filename) - 1) // 10           -> palm id 0..599

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --data-root ./data/ROI --force
    python scripts/prepare_data.py --num-classes 460 --images-per-class 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from dataset.manifest import (  # noqa: E402
    manifests_match,
    prepare_tongji_manifests,
    read_manifest_meta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Tongji ROI manifests.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--manifest-dir", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--label-rule", type=str, default=None,
                        choices=["tongji", "sequential"])
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides: dict = {"paths": {}, "data": {}, "model": {}}
    if args.data_root:
        overrides["paths"]["data_root"] = args.data_root
    if args.manifest_dir:
        overrides["paths"]["manifest_dir"] = args.manifest_dir
    if args.num_classes is not None:
        overrides["data"]["num_classes"] = args.num_classes
        overrides["model"]["num_classes"] = args.num_classes
    if args.images_per_class is not None:
        overrides["data"]["images_per_class"] = args.images_per_class
    if args.label_rule:
        overrides["data"]["label_rule"] = args.label_rule
    overrides = {k: v for k, v in overrides.items() if v}

    cfg = load_config(args.config, overrides)

    train_path = Path(cfg.paths.manifest_dir) / "train.txt"
    test_path = Path(cfg.paths.manifest_dir) / "test.txt"
    if not args.force and train_path.exists() and test_path.exists():
        if manifests_match(
            cfg.paths.manifest_dir,
            data_root=cfg.paths.data_root,
            session1=cfg.paths.session1,
            session2=cfg.paths.session2,
            num_classes=cfg.data.num_classes,
            images_per_class=cfg.data.images_per_class,
            label_rule=cfg.data.label_rule,
        ):
            print(f"Manifests already exist: {train_path}, {test_path}")
            print("Use --force to overwrite.")
            return
        if read_manifest_meta(cfg.paths.manifest_dir) is None:
            print("Existing manifests have no metadata; regenerating ...")
        else:
            print("Existing manifests were built with different settings; regenerating ...")

    summary = prepare_tongji_manifests(
        data_root=cfg.paths.data_root,
        manifest_dir=cfg.paths.manifest_dir,
        num_classes=cfg.data.num_classes,
        images_per_class=cfg.data.images_per_class,
        session1=cfg.paths.session1,
        session2=cfg.paths.session2,
        label_rule=cfg.data.label_rule,
    )

    print("Manifests generated:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
