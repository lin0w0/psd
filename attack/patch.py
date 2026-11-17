"""Localized patch attack for CompNet identity classification.

Adapted from ``CSR/csr/attacks/patch.py``.  The patch is parameterized as
``sigmoid(patch_param)`` so its pixels stay in ``[0, 1]``; the attacker
wrapper applies ``NormSingleROI`` internally.

The default objective is ArcFace cross-entropy (classification ACC);
``loss="cw"`` / ``"dlr"`` are margin-based alternatives and
``loss="matching"`` is the gallery ablation.
"""

from __future__ import annotations

from typing import Optional

import torch

from .attacker import CompNetAttacker
from .objectives import build_margin


class PatchAttack:
    """Localized square patch attack.

    Args:
        attacker: :class:`CompNetAttacker` instance.
        patch_size: Patch side length in pixels.
        steps: Number of optimization steps.
        lr: Adam learning rate.
        random_start: Whether to initialize the patch randomly.
        seed: Random seed for patch locations.
        loss: ``"ce"`` (default), ``"cw"``, ``"dlr"``, or ``"matching"``
            (ablation).
        gallery_features, gallery_labels: only needed for ``loss="matching"``.
    """

    def __init__(
        self,
        attacker: CompNetAttacker,
        patch_size: int = 24,
        steps: int = 100,
        lr: float = 5e-2,
        random_start: bool = True,
        seed: int = 42,
        loss: str = "ce",
        gallery_features: Optional[torch.Tensor] = None,
        gallery_labels: Optional[torch.Tensor] = None,
    ):
        self.attacker = attacker
        self.patch_size = int(patch_size)
        self.steps = int(steps)
        self.lr = float(lr)
        self.random_start = random_start
        self.seed = int(seed)
        self.device = attacker.device
        self.loss = loss

        if gallery_features is not None:
            if gallery_labels is None:
                raise ValueError("gallery_labels are required with gallery_features")
            self.gallery_features = torch.as_tensor(
                gallery_features, device=self.device, dtype=torch.float32
            )
            self.gallery_labels = torch.as_tensor(
                gallery_labels, device=self.device, dtype=torch.long
            )
        else:
            self.gallery_features = None
            self.gallery_labels = None

        # Persistent generator: one set of locations per batch, advanced
        # across batches so every mini-batch gets fresh positions.
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def _sample_locations(
        self, batch_size: int, height: int, width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_y = height - self.patch_size
        max_x = width - self.patch_size
        ys = torch.randint(
            0, max_y + 1, (batch_size,), generator=self._generator
        )
        xs = torch.randint(
            0, max_x + 1, (batch_size,), generator=self._generator
        )
        return ys.to(self.device), xs.to(self.device)

    def _apply_patch(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        ys: torch.Tensor,
        xs: torch.Tensor,
    ) -> torch.Tensor:
        adv = images.clone()
        for i in range(images.shape[0]):
            y0 = int(ys[i].item())
            x0 = int(xs[i].item())
            adv[i, :, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size] = patch[i]
        return self.attacker.project(adv, images)

    @torch.enable_grad()
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial images with learned localized patches.

        Args:
            images: ``[B, 1, H, W]`` clean pixel tensors in ``[0, 1]``.
            labels: ``[B]`` ground-truth class indices.

        Returns:
            Best adversarial pixel tensors found during the optimization.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        batch_size, channels, height, width = images.shape
        if self.patch_size > min(height, width):
            raise ValueError(
                f"patch_size={self.patch_size} exceeds input size {(height, width)}"
            )

        margin = build_margin(
            self.loss,
            self.attacker,
            images,
            labels,
            self.gallery_features,
            self.gallery_labels,
        )
        ys, xs = self._sample_locations(batch_size, height, width)

        if self.random_start:
            patch_param = torch.rand(
                batch_size,
                channels,
                self.patch_size,
                self.patch_size,
                device=self.device,
            )
        else:
            patch_param = torch.zeros(
                batch_size,
                channels,
                self.patch_size,
                self.patch_size,
                device=self.device,
            )
        patch_param.requires_grad = True
        optimizer = torch.optim.Adam([patch_param], lr=self.lr)

        # Never return a candidate worse than the clean image (lower margin
        # is stronger, so "worse" means a higher margin).
        with torch.no_grad():
            best_margin = margin.per_sample(self.attacker, images, labels)
        best_imgs = images.detach().clone()

        for _ in range(self.steps):
            patch = torch.sigmoid(patch_param)
            adv_images = self._apply_patch(images, patch, ys, xs)
            per_sample = margin.per_sample(self.attacker, adv_images, labels)
            loss = per_sample.mean()

            better_mask = per_sample < best_margin
            if better_mask.any():
                best_margin[better_mask] = per_sample[better_mask].detach()
                best_imgs[better_mask] = adv_images[better_mask].detach()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            patch = torch.sigmoid(patch_param)
            adv_images = self._apply_patch(images, patch, ys, xs)
            per_sample = margin.per_sample(self.attacker, adv_images, labels)
            better_mask = per_sample < best_margin
            if better_mask.any():
                best_imgs[better_mask] = adv_images[better_mask]

        return best_imgs.detach()
