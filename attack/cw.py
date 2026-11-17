"""CW (Carlini-Wagner) margin attack for palmprint identity classification.

Adapted from ``CSR/csr/attacks/cw.py``.  The default (``loss="cw"``) applies
the CW hinge to the ArcFace logit margin:

    minimize  relu(-(max_other - real) + kappa)

and stops once the true-class logit exceeds the best other logit by
``kappa``.  ``loss="matching"`` is an optional gallery genuine/impostor
margin ablation.

Unlike PGD/APGD, this attack defaults to **zero initialization**.  The hinge
gradient is exactly zero once a sample is successful, so a uniform random
start would be frozen into the final perturbation; that noise is not part of
the optimized direction and shows up as a spurious high-frequency component.
Pass ``random_start=True`` to restore the old behaviour.
"""

from __future__ import annotations

import torch

from .attacker import CompNetAttacker
from .base import BaseAttack


class CWAttack(BaseAttack):
    """PGD-CW logit-margin attack (classification ACC).

    Args:
        attacker: :class:`CompNetAttacker` instance.
        epsilon: Maximum perturbation (L-inf norm, pixel units in ``[0, 1]``).
        steps: Number of PGD iterations.
        alpha: Step size.  If ``None``, auto-computed.
        alpha_scale: Multiplier for auto alpha.
        kappa: Confidence margin.  The attack stops once the true-class
            logit exceeds the best other logit by ``kappa`` (0 = decision
            boundary).
        random_start: Whether to use random initialization.  Defaults to
            ``False``: the hinge has zero gradient once a sample is
            successful, so a random start would freeze uniform noise into the
            final perturbation (visible as a spurious high-frequency
            component).  Zero initialization keeps the final delta equal to
            the optimized steps only.  Pass ``True`` to restore the old
            behaviour.
        loss: ``"cw"`` (default, logit margin) or ``"matching"`` (ablation).
        gallery_features, gallery_labels: only needed for ``loss="matching"``.
    """

    def __init__(
        self,
        attacker: CompNetAttacker,
        epsilon: float = 4 / 255,
        steps: int = 30,
        alpha: float | None = None,
        alpha_scale: float = 2.5,
        kappa: float = 0.0,
        random_start: bool = False,
        loss: str = "cw",
        gallery_features: torch.Tensor | None = None,
        gallery_labels: torch.Tensor | None = None,
    ):
        if loss in ("ce", "dlr"):
            raise ValueError(
                "CWAttack supports loss='cw' or 'matching'; "
                "use PGD/APGD for the CE/DLR objectives."
            )
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
        self.kappa = float(kappa)

    @torch.enable_grad()
    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using the CW hinge.

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
            per_sample = margin.per_sample(self.attacker, adv_images, labels)

            # Minimize relu(margin + kappa): push until margin <= -kappa.
            loss = torch.clamp(per_sample + self.kappa, min=0).sum()

            if delta.grad is not None:
                delta.grad.zero_()
            loss.backward()

            grad = delta.grad.detach()
            # Descent (lower margin is stronger).
            delta.data = self.project_delta(delta - self.alpha * grad.sign(), images)
            delta.grad.zero_()

        return self.finalize(images, delta)
