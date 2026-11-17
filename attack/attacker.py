"""CompNet attacker wrapper for pixel-space attacks.

Adapted from ``CSR/csr/attacks/attacker.py`` (``CLIPAttacker``).

Attacks operate on **pixel tensors** ``[B, 1, H, W]`` in ``[0, 1]``, exactly
as CSR's attacks do on CLIP inputs.  CompNet, however, expects the output of
:class:`dataset.dataset.NormSingleROI`, so the wrapper applies that
preprocessing inside the forward pass:

    pixels [0,1] -> norm_single_roi -> frozen CompNet -> logits / codes

The official ``NormSingleROI`` uses in-place boolean-index assignment and is
not autograd-safe (``RuntimeError: a view of a leaf Variable ... in-place``).
:func:`norm_single_roi` below is a differentiable, batched re-implementation
that matches the official transform to ~5e-7 (verified on Tongji).

* ``forward(x) -> logits``   -- ArcFace cosine logits (``cosine * s``)
* ``features(x)`` / ``getFeatureCode(x)`` -- L2-normalized 512-d codes
* ``project(x_adv, x_clean)`` -- clamp to the pixel range and, optionally,
  keep the (black, exactly zero) ROI background untouched.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from model.loader import resolve_device


def norm_single_roi(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable, batched equivalent of ``dataset.NormSingleROI``.

    For every pixel that is strictly positive, subtract the image mean and
    divide by the (unbiased) standard deviation of the positive pixels;
    non-positive pixels are left unchanged, matching the official transform.
    For ``[0, 1]`` pixel inputs those are exactly the zero background.

    Args:
        x: pixel tensor ``[B, C, H, W]`` in ``[0, 1]``.
        eps: stabilizer added to the std (the official transform uses 1e-6).

    Returns:
        Normalized tensor with the same shape.
    """
    b, c, h, w = x.shape
    flat = x.reshape(b, c, h * w)
    mask = flat > 0

    count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (flat * mask).sum(dim=-1, keepdim=True) / count
    # Unbiased variance, matching torch.std() used by NormSingleROI.
    var = (((flat - mean) ** 2) * mask).sum(dim=-1, keepdim=True) / (
        count - 1
    ).clamp_min(1)
    std = torch.sqrt(var.clamp_min(0))

    normalized = (flat - mean) / (std + eps)
    normalized = torch.where(mask, normalized, flat)
    return normalized.reshape(b, c, h, w)


class CompNetAttacker(nn.Module):
    """Frozen CompNet wrapper that exposes pixel-space logits and features.

    Args:
        model: CompNet instance (from :func:`model.loader.load_compnet`).
        device: ``"auto"``, ``"cuda"``, ``"cuda:0"`` or ``"cpu"``.
        preprocess: Optional differentiable map ``pixels -> model input``.
            Defaults to :func:`norm_single_roi`.  Do **not** pass the official
            ``NormSingleROI`` for gradient-based attacks -- it is not
            autograd-safe.
        preserve_background: If ``True``, pixels that are exactly zero in the
            clean ROI (black background) are never perturbed.
        input_range: Valid pixel range, default ``(0.0, 1.0)``.  Set to
            ``None`` to disable the clamp (not recommended).
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "auto",
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        preserve_background: bool = False,
        input_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ):
        super().__init__()
        self.device = (
            resolve_device(device) if isinstance(device, str) else torch.device(device)
        )
        self.model = model.to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.preprocess = preprocess or norm_single_roi
        self.preserve_background = preserve_background
        self.input_range = (
            None if input_range is None else (float(input_range[0]), float(input_range[1]))
        )

    # ---- model interface -------------------------------------------------

    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Map pixel tensors to the representation CompNet expects."""
        return self.preprocess(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ArcFace cosine logits ``[B, num_classes]`` for pixel inputs."""
        return self.model(self.preprocess_images(images))

    def features(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized feature codes ``[B, 512]`` for pixel inputs."""
        return self.model.getFeatureCode(self.preprocess_images(images))

    # Evaluation helpers expect ``model.getFeatureCode``; expose the same name
    # so ``evaluation.extract_features(attacker, pixel_dataset, ...)`` works.
    def getFeatureCode(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)

    @torch.no_grad()
    def compute_features(self, images: torch.Tensor) -> torch.Tensor:
        """``features`` under ``torch.no_grad()`` for evaluation loops."""
        return self.features(images)

    # ---- projection helpers used by the attacks --------------------------

    def background_mask(self, clean: torch.Tensor) -> torch.Tensor:
        """Boolean mask of perturbable (non-black) pixels."""
        return clean > 0

    def project(self, adv: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Project an adversarial tensor back onto the valid pixel domain."""
        if self.input_range is not None:
            adv = torch.clamp(adv, self.input_range[0], self.input_range[1])
        if self.preserve_background:
            adv = torch.where(self.background_mask(clean), adv, clean)
        return adv

    def project_delta(self, delta: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        """Keep ``clean + delta`` inside the pixel range, then re-mask."""
        if self.input_range is not None:
            delta = torch.clamp(
                clean + delta, self.input_range[0], self.input_range[1]
            ) - clean
        if self.preserve_background:
            delta = torch.where(
                self.background_mask(clean), delta, torch.zeros_like(delta)
            )
        return delta
