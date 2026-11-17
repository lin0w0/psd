"""
Palmprint ROI dataset and preprocessing.

Kept byte-for-byte equivalent to the official CompNet ``models/dataset.py``
inference path:

    Resize(128) -> ToTensor() -> NormSingleROI()

Tongji ROI images are already 128 x 128, so ``Resize`` is a no-op; the
grayscale conversion happens at image load time (``convert("L")``).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class NormSingleROI:
    """Normalize non-black pixels of a single-channel ROI to 0 mean / 1 std.

    Input/output shape: ``[1, H, W]``. Black background pixels stay zero.
    """

    def __init__(self, outchannels: int = 1):
        self.outchannels = outchannels

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        c, h, w = tensor.size()
        if c != 1:
            raise TypeError("only support grayscale image.")

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        m = t.mean()
        s = t.std()
        tensor[idx] = t.sub_(m).div_(s + 1e-6)

        tensor = tensor.view(c, h, w)
        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)
        return tensor


def build_transform(imside: int = 128, outchannels: int = 1) -> T.Compose:
    """Official deterministic inference transform."""
    return T.Compose([
        T.Resize(imside),
        T.ToTensor(),
        NormSingleROI(outchannels=outchannels),
    ])


class PalmROIDataset(Dataset):
    """Load ROI images from a manifest (``path label`` per line).

    Args:
        manifest: Text file produced by ``scripts/prepare_data.py``.
        imside: Network input size.
        outchannels: 1 for grayscale (official), 3 to repeat channels.
        transform: Optional override; defaults to the official transform.

    Returns:
        ``(tensor[outchannels, imside, imside], int label)``
    """

    def __init__(
        self,
        manifest: str | Path,
        imside: int = 128,
        outchannels: int = 1,
        transform: T.Compose | None = None,
    ):
        self.manifest = str(manifest)
        self.imside = imside
        self.outchannels = outchannels
        self.transform = transform or build_transform(imside, outchannels)
        self.images_path, self.images_label = self._read_manifest(self.manifest)

    @staticmethod
    def _read_manifest(manifest: str | Path) -> Tuple[List[str], List[int]]:
        paths: List[str] = []
        labels: List[int] = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    path, label = line.rsplit(" ", 1)
                except ValueError as exc:
                    raise ValueError(
                        f"Malformed manifest line {line_no}: {line!r}"
                    ) from exc
                paths.append(path)
                labels.append(int(label))
        if not paths:
            raise ValueError(f"Empty manifest: {manifest}")
        return paths, labels

    def __getitem__(self, index: int):
        img_path = self.images_path[index]
        label = self.images_label[index]

        data = Image.open(img_path).convert("L")
        data = self.transform(data)
        return data, int(label)

    def __len__(self) -> int:
        return len(self.images_path)
