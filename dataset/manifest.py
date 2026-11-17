"""
Manifest generation for the Tongji ROI dataset.

Official Tongji labeling rule (``genText_tongji.py``):
every 10 consecutive ROIs with numeric filenames ``00001.bmp`` ... ``06000.bmp``
belong to the same palm, so ``label = (int(stem) - 1) // 10``.

Session 1 is the gallery (training) set, session 2 the probe (test) set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MANIFEST_META_NAME = "manifest_meta.json"


def list_session_images(session_dir: str | Path) -> List[Path]:
    """Return the session's image files in filename order."""
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    files = [
        p for p in session_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No images found in: {session_dir}")
    return files


def label_for(path: Path, index: int, images_per_class: int, rule: str = "tongji") -> int:
    """Derive the palm id for an image.

    ``tongji``: numeric filename ``00001`` -> ``(1 - 1) // 10 = 0``.
    ``sequential``: index in the sorted file list, grouped by
    ``images_per_class``.
    """
    if rule == "tongji":
        try:
            return (int(path.stem) - 1) // images_per_class
        except ValueError as exc:
            raise ValueError(
                f"Filename {path.name!r} is not numeric; use --label-rule sequential"
            ) from exc
    if rule == "sequential":
        return index // images_per_class
    raise ValueError(f"Unknown label rule: {rule!r}")


def build_manifest(
    session_dir: str | Path,
    images_per_class: int,
    num_classes: int,
    label_rule: str = "tongji",
) -> List[Tuple[Path, int]]:
    """Return ``(path, label)`` pairs and validate the class layout."""
    files = list_session_images(session_dir)
    expected = num_classes * images_per_class
    if len(files) != expected:
        raise ValueError(
            f"{session_dir}: expected {expected} images "
            f"({num_classes} classes x {images_per_class}), found {len(files)}"
        )

    items = [
        (path, label_for(path, i, images_per_class, label_rule))
        for i, path in enumerate(files)
    ]

    counts: Dict[int, int] = {}
    for _, label in items:
        counts[label] = counts.get(label, 0) + 1

    labels = sorted(counts)
    if labels != list(range(num_classes)):
        raise ValueError(
            f"{session_dir}: labels must be 0..{num_classes - 1}, "
            f"found {len(labels)} labels ({labels[:5]}...{labels[-5:]})"
        )
    bad = {label: c for label, c in counts.items() if c != images_per_class}
    if bad:
        examples = dict(list(bad.items())[:5])
        raise ValueError(
            f"{session_dir}: every class must have {images_per_class} images; "
            f"offending classes (label: count) {examples}"
        )

    # Class blocks must be contiguous so gallery index // images_per_class
    # equals the label (required by the official aggregated protocol).
    for i, (_, label) in enumerate(items):
        if label != i // images_per_class:
            raise ValueError(
                f"{session_dir}: files are not grouped by class at index {i}; "
                "the aggregated verification protocol requires contiguous blocks."
            )

    return items


def write_manifest(items: List[Tuple[Path, int]], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for path, label in items:
            f.write(f"{path.resolve()} {label}\n")


def write_manifest_meta(
    manifest_dir: str | Path,
    *,
    data_root: str | Path,
    session1: str,
    session2: str,
    num_classes: int,
    images_per_class: int,
    label_rule: str,
    num_gallery: int,
    num_probe: int,
) -> Path:
    """Write a sidecar JSON describing how the manifests were generated."""
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_root": str(Path(data_root).resolve()),
        "session1": session1,
        "session2": session2,
        "num_classes": int(num_classes),
        "images_per_class": int(images_per_class),
        "label_rule": label_rule,
        "num_gallery": int(num_gallery),
        "num_probe": int(num_probe),
    }
    path = manifest_dir / MANIFEST_META_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return path


def read_manifest_meta(manifest_dir: str | Path) -> Dict | None:
    """Read the manifest sidecar, or ``None`` if missing/invalid."""
    path = Path(manifest_dir) / MANIFEST_META_NAME
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def manifests_match(
    manifest_dir: str | Path,
    *,
    data_root: str | Path,
    session1: str,
    session2: str,
    num_classes: int,
    images_per_class: int,
    label_rule: str,
) -> bool:
    """Return True if existing manifests were built from the same settings."""
    meta = read_manifest_meta(manifest_dir)
    if meta is None:
        return False
    expected = {
        "data_root": str(Path(data_root).resolve()),
        "session1": session1,
        "session2": session2,
        "num_classes": int(num_classes),
        "images_per_class": int(images_per_class),
        "label_rule": label_rule,
    }
    return all(meta.get(key) == value for key, value in expected.items())


def prepare_tongji_manifests(
    data_root: str | Path,
    manifest_dir: str | Path,
    num_classes: int = 600,
    images_per_class: int = 10,
    session1: str = "session1",
    session2: str = "session2",
    label_rule: str = "tongji",
) -> Dict[str, str]:
    """Build ``train.txt`` (session1) and ``test.txt`` (session2).

    Returns a dict with manifest paths and summary statistics.
    """
    data_root = Path(data_root)
    manifest_dir = Path(manifest_dir)

    train_items = build_manifest(
        data_root / session1, images_per_class, num_classes, label_rule
    )
    test_items = build_manifest(
        data_root / session2, images_per_class, num_classes, label_rule
    )

    train_names = {p.name for p, _ in train_items}
    test_names = {p.name for p, _ in test_items}
    if train_names != test_names:
        raise ValueError(
            "session1 and session2 filenames differ; the official protocol "
            "pairs them by filename."
        )

    train_manifest = manifest_dir / "train.txt"
    test_manifest = manifest_dir / "test.txt"
    write_manifest(train_items, train_manifest)
    write_manifest(test_items, test_manifest)
    meta_path = write_manifest_meta(
        manifest_dir,
        data_root=data_root,
        session1=session1,
        session2=session2,
        num_classes=num_classes,
        images_per_class=images_per_class,
        label_rule=label_rule,
        num_gallery=len(train_items),
        num_probe=len(test_items),
    )

    return {
        "train_manifest": str(train_manifest),
        "test_manifest": str(test_manifest),
        "manifest_meta": str(meta_path),
        "num_gallery": len(train_items),
        "num_probe": len(test_items),
        "num_classes": num_classes,
        "images_per_class": images_per_class,
    }
