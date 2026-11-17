"""Adversarial attacks for CompNet palmprint identity classification.

Adapted from the CSR attack suite (``CSR/csr/attacks/``).

Final evaluation is **identity classification accuracy (ACC)**, i.e. the
``argmax`` over the 600 ArcFace cosine logits.  Attack framework::

    raw ROI -> Resize -> ToTensor (pixels in [0,1])
        -> add PGD / APGD / CW / Patch perturbation
        -> NormSingleROI -> CompNet -> ArcFace logits
        -> classification loss (CE / CW logit margin)

Default objectives:

* PGD / APGD / Patch: ``loss="ce"`` (cross-entropy, direct ACC surrogate).
* APGD / PGD / Patch: ``loss="dlr"`` (untargeted Difference of Logits Ratio,
  APGD-DLR).
* CW: ``loss="cw"`` (CW logit margin, stops at the decision boundary).

``loss="matching"`` keeps the gallery genuine/impostor margin as an optional
ablation (useful for testing whether verification-oriented perturbations
transfer to ACC).

Available attacks
-----------------
====================  ==================================================
:class:`PGDAttack`        PGD-Linf on CE / CW margin
:class:`APGDAttack`       APGD with adaptive step sizes
:class:`CWAttack`         CW hinge on the logit margin
:class:`PatchAttack`      localized square patch
====================  ==================================================

Typical use::

    from attack import CompNetAttacker, PGDAttack

    attacker = CompNetAttacker(model, device=device)
    attack = PGDAttack(attacker, epsilon=4 / 255, steps=10)   # loss="ce"
    adv = attack.attack(images, labels)       # pixels: [B, 1, 128, 128]

Feature codes for the matching protocols and the optional matching ablation
are available as ``attacker.features(x)`` / ``attacker.getFeatureCode(x)``.
"""

from .apgd import APGDAttack
from .attacker import CompNetAttacker, norm_single_roi
from .base import BaseAttack
from .cw import CWAttack
from .objectives import (
    BaseMargin,
    ClassificationMargin,
    CleanAnchorMargin,
    GalleryMargin,
    build_margin,
)
from .patch import PatchAttack
from .pgd import PGDAttack

__all__ = [
    "CompNetAttacker",
    "norm_single_roi",
    "BaseAttack",
    "PGDAttack",
    "APGDAttack",
    "CWAttack",
    "PatchAttack",
    "BaseMargin",
    "GalleryMargin",
    "CleanAnchorMargin",
    "ClassificationMargin",
    "build_margin",
]
