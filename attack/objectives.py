"""Attack objectives for palmprint identity classification.

The final evaluation metric is classification accuracy (ACC), i.e. the
``argmax`` over the 600 ArcFace cosine logits.  The primary objectives are
therefore classification losses:

* :class:`ClassificationMargin` ``mode="ce"`` -- maximize cross-entropy
  (default for PGD/APGD/Patch).
* :class:`ClassificationMargin` ``mode="cw"`` -- CW logit margin
  ``-(max_other - real)`` (default for the CW attack).
* :class:`ClassificationMargin` ``mode="dlr"`` -- maximize the untargeted
  Difference of Logits Ratio (DLR), available for PGD/APGD/Patch.

The gallery matching margin below is retained as an optional ablation
(``loss="matching"``): explicitly optimizing the verification/rank-1 statistic
lets us test whether it transfers to classification ACC.

For a probe with label ``y`` and gallery features ``G``::

    s_gen = max_{g in G, label(g) == y} cos(f(x'), f(g))
    s_imp = max_{g in G, label(g) != y} cos(f(x'), f(g))
    margin(x') = s_gen - s_imp

A verification / rank-1 failure occurs when ``margin < 0``.  Every objective
below exposes this per-sample **margin**, where *lower is a stronger attack*:

* :class:`GalleryMargin` -- the gallery-based margin above (ablation).  This
  directly matches the verification statistic and the rank-1 competitor.
* :class:`CleanAnchorMargin` -- fallback ``cos(f(x'), f(x_clean))`` when no
  gallery is available (label-free, weaker).
* :class:`ClassificationMargin` -- ArcFace logits: ``mode="ce"`` returns
  ``-CE``; ``mode="cw"`` returns ``-(max_other - real)``; ``mode="dlr"``
  returns ``-DLR``.

Attack update rules use the margin consistently:

* PGD / APGD maximize ``-margin``.
* CW minimizes ``relu(margin + kappa)`` (stop at ``margin = -kappa``).

For classification ACC, ``-CE``, ``-(max_other - real)`` and ``-DLR`` are the
relevant "lower is stronger" margins; the gallery margin is only an ablation.

DLR (Croce & Hein, untargeted)::

    DLR = (max_{j != y} z_j - z_y) / (z_{pi_1} - z_{pi_3})

where ``z_{pi_1}`` / ``z_{pi_3}`` are the largest / third-largest logits.
The denominator normalizes by the logit spread, making the loss scale-
invariant and often more stable when CE gradients saturate.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class BaseMargin:
    """Per-sample matching margin; lower means a stronger attack."""

    def per_sample(
        self,
        attacker,
        adv_images: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError


class GalleryMargin(BaseMargin):
    """``max genuine cosine - max impostor cosine`` against the gallery."""

    def __init__(
        self,
        gallery_features: torch.Tensor,
        gallery_labels: torch.Tensor,
    ):
        if gallery_features.dim() != 2:
            raise ValueError(
                f"gallery_features must be [N, D], got {tuple(gallery_features.shape)}"
            )
        gallery_features = gallery_features.detach()
        self.gallery_features = F.normalize(gallery_features, dim=1)
        self.gallery_labels = gallery_labels.detach().to(
            gallery_features.device
        ).long()

    def per_sample(self, attacker, adv_images, labels):
        if labels is None:
            raise ValueError("GalleryMargin requires class labels")

        features = attacker.features(adv_images)          # [B, D], unit norm
        similarity = features @ self.gallery_features.t()  # [B, N]
        labels = labels.to(self.gallery_labels.device).long()
        positive = labels[:, None] == self.gallery_labels[None, :]

        # Cosines are in [-1, 1]; -2 is below any valid score.
        s_gen = similarity.masked_fill(~positive, -2.0).amax(dim=1)
        s_imp = similarity.masked_fill(positive, -2.0).amax(dim=1)
        return s_gen - s_imp


class CleanAnchorMargin(BaseMargin):
    """Fallback: cosine similarity to the clean probe's own feature code."""

    def __init__(self, clean_features: torch.Tensor):
        self.clean_features = F.normalize(clean_features.detach(), dim=1)

    def per_sample(self, attacker, adv_images, labels=None):
        features = attacker.features(adv_images)
        return (features * self.clean_features).sum(dim=1)


class ClassificationMargin(BaseMargin):
    """Primary classification objective on ArcFace logits.

    Modes (all return a "lower is stronger" margin):

    * ``"ce"``  -- ``-cross_entropy``.
    * ``"cw"``  -- ``-(max_other - real)`` (CW logit margin).
    * ``"dlr"`` -- ``-DLR`` (untargeted Difference of Logits Ratio).
    """

    def __init__(self, mode: str = "ce"):
        if mode not in ("ce", "cw", "dlr"):
            raise ValueError(
                f"ClassificationMargin mode must be 'ce'/'cw'/'dlr', got {mode!r}"
            )
        self.mode = mode

    def per_sample(self, attacker, adv_images, labels):
        if labels is None:
            raise ValueError("ClassificationMargin requires class labels")

        logits = attacker(adv_images)
        labels = labels.to(logits.device).long()
        real = torch.gather(logits, 1, labels.view(-1, 1)).squeeze(1)

        if self.mode == "ce":
            return -F.cross_entropy(logits, labels, reduction="none")

        if self.mode == "cw":
            # CW logit margin: -(max_other - real).
            others = logits.clone()
            others.scatter_(1, labels.view(-1, 1), -float("inf"))
            max_other = others.max(dim=1).values
            return -(max_other - real)

        # Untargeted DLR: (max_other - real) / (top1 - top3).
        if logits.shape[1] < 3:
            raise ValueError("DLR requires at least 3 classes")
        sorted_logits, sorted_idx = logits.sort(dim=1)  # ascending
        is_top1 = sorted_idx[:, -1].eq(labels)
        max_other = torch.where(
            is_top1, sorted_logits[:, -2], sorted_logits[:, -1]
        )
        numerator = max_other - real
        denominator = (sorted_logits[:, -1] - sorted_logits[:, -3]).clamp_min(1e-12)
        dlr = numerator / denominator
        return -dlr


def build_margin(
    loss: str,
    attacker,
    images: torch.Tensor,
    labels: Optional[torch.Tensor],
    gallery_features: Optional[torch.Tensor] = None,
    gallery_labels: Optional[torch.Tensor] = None,
) -> BaseMargin:
    """Build the per-sample margin for a given loss name.

    Args:
        loss: ``"ce"`` (classification ACC, default), ``"cw"`` (logit
            margin), ``"dlr"`` (untargeted DLR), or ``"matching"`` (gallery
            ablation).
        attacker: :class:`attack.attacker.CompNetAttacker`.
        images: clean pixel tensors (used only for the clean-anchor fallback).
        labels: probe labels (required for gallery/classification margins).
        gallery_features: ``[N, 512]`` reference features (session1).
        gallery_labels: ``[N]`` reference labels.
    """
    if loss == "matching":
        if gallery_features is not None:
            if gallery_labels is None:
                raise ValueError("gallery_labels are required with gallery_features")
            return GalleryMargin(gallery_features, gallery_labels)
        with torch.no_grad():
            clean_features = attacker.features(images).detach()
        return CleanAnchorMargin(clean_features)

    if loss in ("ce", "cw", "dlr"):
        return ClassificationMargin(loss)

    raise ValueError(
        f"loss must be 'matching', 'ce', 'cw' or 'dlr', got {loss!r}"
    )

