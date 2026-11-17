"""Shared L-inf machinery for CompNet pixel-space attacks.

Pipeline (identity classification ACC)::

    ROI -> Resize -> ToTensor -> [attack adds delta] -> NormSingleROI
        -> CompNet -> ArcFace logits -> classification loss

``CompNetAttacker`` applies ``NormSingleROI`` inside its forward pass, so the
attacks themselves only manipulate pixel tensors in ``[0, 1]``.

The default objective is ArcFace cross-entropy (``loss="ce"``), which directly
maximizes classification error.  ``loss="cw"`` uses the CW logit margin and
``loss="matching"`` keeps the gallery genuine/impostor margin as an ablation.
The perturbation is bounded by ``‖δ‖∞ ≤ ε`` and ``x + δ`` is clamped back to
``[0, 1]``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .attacker import CompNetAttacker
from .objectives import build_margin


class BaseAttack:
    """Common hyperparameters, objective and projection helpers.

    Args:
        attacker: :class:`CompNetAttacker` instance.
        epsilon: Maximum L-inf perturbation budget in pixel units ``[0, 1]``
            (CSR defaults: ``1/255``, ``2/255``, ``4/255``).
        steps: Number of iterations (unused by some subclasses).
        alpha: Step size per iteration.  If ``None``, computed as
            ``(epsilon / steps) * alpha_scale``.
        alpha_scale: Multiplier for the auto-computed step size.
        random_start: Whether to initialize the perturbation randomly inside
            the epsilon ball.
        loss: ``"ce"`` (default), ``"cw"``, ``"dlr"``, or ``"matching"``
            (ablation).
        gallery_features: ``[N, 512]`` session1 feature codes used as the
            matching reference.  Only needed for ``loss="matching"``; if
            ``None``, the clean probe feature is used as the anchor.
        gallery_labels: ``[N]`` gallery labels.
    """

    def __init__(
        self,
        attacker: CompNetAttacker,
        epsilon: float = 4 / 255,
        steps: int = 10,
        alpha: float | None = None,
        alpha_scale: float = 2.5,
        random_start: bool = True,
        loss: str = "ce",
        gallery_features: Optional[torch.Tensor] = None,
        gallery_labels: Optional[torch.Tensor] = None,
    ):
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        if steps <= 0:
            raise ValueError(f"steps must be > 0, got {steps}")

        self.attacker = attacker
        self.epsilon = float(epsilon)
        self.steps = int(steps)
        self.alpha = (
            float(alpha) if alpha is not None else (self.epsilon / self.steps) * alpha_scale
        )
        self.random_start = random_start
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

    # ---- objective -------------------------------------------------------

    def build_margin(self, images: torch.Tensor, labels: torch.Tensor):
        """Construct the per-sample matching margin for this attack."""
        return build_margin(
            self.loss,
            self.attacker,
            images,
            labels,
            self.gallery_features,
            self.gallery_labels,
        )

    # ---- projection ------------------------------------------------------

    def init_delta(self, images: torch.Tensor) -> torch.Tensor:
        """Random (or zero) perturbation inside the pixel-space epsilon ball."""
        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(images)
        return self.attacker.project_delta(delta, images)

    def project_delta(self, delta: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Clamp to the L-inf ball and back into the valid pixel range."""
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        return self.attacker.project_delta(delta, images)

    def finalize(
        self, images: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """Build the final adversarial pixel tensor (detached)."""
        return self.attacker.project(images + delta, images).detach()
