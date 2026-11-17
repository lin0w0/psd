"""Palmprint dataset loading and manifest generation.

Note: the raw dataset lives in ``data/`` (git-ignored); this package holds
the code, so it is named ``dataset`` to avoid shadowing the common
HuggingFace ``datasets`` package.
"""

from .dataset import NormSingleROI, PalmROIDataset, build_transform
from .manifest import (
    build_manifest,
    list_session_images,
    manifests_match,
    prepare_tongji_manifests,
    read_manifest_meta,
    write_manifest,
    write_manifest_meta,
)

__all__ = [
    "PalmROIDataset",
    "NormSingleROI",
    "build_transform",
    "prepare_tongji_manifests",
    "build_manifest",
    "write_manifest",
    "write_manifest_meta",
    "read_manifest_meta",
    "manifests_match",
    "list_session_images",
]
