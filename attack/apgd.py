"""APGD (Auto-PGD) attack for palmprint identity classification.

Adapted from ``CSR/csr/attacks/apgd.py``.  The default objective is the
ArcFace cross-entropy (``loss="ce"``, classification ACC).  ``loss="dlr"``
gives the untargeted Difference of Logits Ratio (APGD-DLR), ``loss="cw"``
uses the CW logit margin, and ``loss="matching"`` is the gallery ablation.
Keeps APGD's per-sample adaptive step sizes, checkpoint-based step halving
and best-sample tracking.
"""

from __future__ import annotations

import torch

from .attacker import CompNetAttacker
from .base import BaseAttack


class APGDAttack(BaseAttack):
    """Auto-PGD L-inf attack on the identity-classification objective.

    Args:
        attacker: :class:`CompNetAttacker` instance.
        epsilon: Maximum perturbation (L-inf norm, pixel units in ``[0, 1]``).
        steps: Total number of APGD iterations.
        random_start: Whether to initialize with random noise.
        loss: ``"ce"`` (default), ``"cw"``, ``"dlr"`` (APGD-DLR), or
            ``"matching"`` (ablation).
        gallery_features, gallery_labels: matching reference (session1).
    """

    def __init__(
        self,
        attacker: CompNetAttacker,
        epsilon: float = 4 / 255,
        steps: int = 30,
        random_start: bool = True,
        loss: str = "ce",
        gallery_features: torch.Tensor | None = None,
        gallery_labels: torch.Tensor | None = None,
    ):
        super().__init__(
            attacker=attacker,
            epsilon=epsilon,
            steps=steps,
            random_start=random_start,
            loss=loss,
            gallery_features=gallery_features,
            gallery_labels=gallery_labels,
        )

    @torch.enable_grad()
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using APGD on the classification loss.

        Args:
            images: ``[B, 1, H, W]`` clean pixel tensors in ``[0, 1]``.
            labels: ``[B]`` ground-truth class indices.

        Returns:
            Best adversarial pixel tensors found during the optimization.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        margin = self.build_margin(images, labels)
        batch_size = images.shape[0]

        # Best-sample tracking: maximize -margin.
        with torch.no_grad():
            loss_best = -margin.per_sample(self.attacker, images, labels)
        x_best = images.clone()

        # Random start inside the epsilon ball.
        x_adv = self.attacker.project(images + self.init_delta(images), images)
        x_adv.requires_grad = True

        # Per-sample adaptive step size (CSR/AutoAttack convention).
        step_size = torch.full(
            (batch_size,) + (1,) * (images.dim() - 1),
            2.0 * self.epsilon,
            device=self.device,
        )

        # Checkpoints for step-size reduction (AutoAttack paper).
        checkpoints = {
            max(int(0.22 * self.steps), 1),
            max(int(0.48 * self.steps), 1),
            max(int(0.78 * self.steps), 1),
        }

        for i in range(self.steps):
            loss_indiv = -margin.per_sample(self.attacker, x_adv, labels)
            loss_total = loss_indiv.sum()

            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss_total.backward()
            grad = x_adv.grad.detach()

            # Track the best (largest -margin == smallest margin) sample.
            with torch.no_grad():
                is_better = loss_indiv > loss_best
                loss_best[is_better] = loss_indiv[is_better]
                current_x = x_adv.detach()
                x_best[is_better] = current_x[is_better]

            # Step-size halving at checkpoints.
            if (i + 1) in checkpoints:
                step_size = step_size * 0.5

            # Gradient update with projection.
            with torch.no_grad():
                x_next = x_adv + step_size * grad.sign()
                delta = self.project_delta(x_next - images, images)
                x_adv.data = self.attacker.project(images + delta, images)
                x_adv.grad.zero_()

        return x_best.detach()
