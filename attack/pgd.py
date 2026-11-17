"""PGD (Projected Gradient Descent) attack for palmprint identity classification.

Adapted from ``CSR/csr/attacks/pgd.py``.  The final evaluation is
classification accuracy (ACC), so the default objective is the ArcFace
cross-entropy (``loss="ce"``); ``loss="cw"`` uses the CW logit margin,
``loss="dlr"`` uses the untargeted Difference of Logits Ratio, and
``loss="matching"`` keeps the gallery genuine/impostor margin as an ablation.
"""

from __future__ import annotations

import torch

from .attacker import CompNetAttacker
from .base import BaseAttack


class PGDAttack(BaseAttack):
    """PGD-Linf attack on the identity-classification objective.

    Args:
        attacker: :class:`CompNetAttacker` instance.
        epsilon: Maximum perturbation (L-inf norm, pixel units in ``[0, 1]``).
        steps: Number of PGD iterations.
        alpha: Step size per iteration.  If ``None``, computed as
            ``(epsilon / steps) * alpha_scale``.
        alpha_scale: Multiplier for auto-computed alpha.
        random_start: Whether to initialize with random noise.
        loss: ``"ce"`` (default), ``"cw"``, ``"dlr"``, or ``"matching"``
            (ablation).
        gallery_features, gallery_labels: matching reference (session1).
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
        gallery_features: torch.Tensor | None = None,
        gallery_labels: torch.Tensor | None = None,
    ):
        super().__init__(
            attacker=attacker,
            epsilon=epsilon,
            steps=steps,
            alpha=alpha,
            alpha_scale=alpha_scale,
            random_start=random_start,
            loss=loss,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
        )

    @torch.enable_grad()
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples.

        Args:
            images: ``[B, 1, H, W]`` clean pixel tensors in ``[0, 1]``.
            labels: ``[B]`` ground-truth class indices.

        Returns:
            Adversarial pixel tensors ``[B, 1, H, W]`` in ``[0, 1]``.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        margin = self.build_margin(images, labels)

        delta = self.init_delta(images)
        delta.requires_grad = True

        for _ in range(self.steps):
            adv_images = self.attacker.project(images + delta, images)

            # Maximize -margin == minimize the genuine/impostor margin.
            loss = -margin.per_sample(self.attacker, adv_images, labels).mean()
            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad = delta.grad.detach()
            delta.data = self.project_delta(delta + self.alpha * grad.sign(), images)
            delta.grad.zero_()

        return self.finalize(images, delta)
