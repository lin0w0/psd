"""
psd project - Centralized configuration.

All paths and hyperparameters are defined here and can be overridden by a YAML
file (``config/default.yaml``) or by command-line arguments.

Path fields are resolved relative to the project root (the repository root),
unless they are already absolute.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]      # repository root (psd/)
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "default.yaml"


# ======================== Path Defaults ========================

@dataclass
class PathConfig:
    """Filesystem paths for the dataset, weights, manifests and results."""

    # Root of the Tongji ROI dataset (contains session1/ and session2/)
    data_root: str = "./data/ROI"
    session1: str = "session1"
    session2: str = "session2"

    # Official pretrained checkpoint
    weights: str = "./net_params_tongji_600.pth"

    # Output directories
    manifest_dir: str = "./manifests"
    results_root: str = "./results"

    @property
    def session1_dir(self) -> Path:
        return Path(self.data_root) / self.session1

    @property
    def session2_dir(self) -> Path:
        return Path(self.data_root) / self.session2

    def resolve(self, base_dir: Path) -> "PathConfig":
        """Return a copy whose path fields are absolute."""
        base_dir = Path(base_dir).resolve()
        out = copy.deepcopy(self)
        for name in ("data_root", "weights", "manifest_dir", "results_root"):
            value = Path(getattr(self, name))
            if not value.is_absolute():
                value = (base_dir / value).resolve()
            setattr(out, name, str(value))
        return out


# ======================== Dataset Config ========================

@dataclass
class DataConfig:
    """Palmprint ROI dataset / preprocessing settings."""

    imside: int = 128                       # network input size (official: 128)
    outchannels: int = 1                    # 1 = grayscale (official)
    num_classes: int = 600                  # Tongji: 600 palms
    images_per_class: int = 10              # session1/session2: 10 ROIs per palm
    label_rule: str = "tongji"              # "tongji" -> (int(stem)-1)//10 | "sequential"
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True


# ======================== Model Config ========================

@dataclass
class ModelConfig:
    """CompNet architecture settings."""

    name: str = "CompNet"
    num_classes: int = 600                  # Tongji: 600 (IITD 460, REST 358, XJTU 200, KTU 145)


# ======================== Evaluation Config ========================

@dataclass
class EvaluationConfig:
    """Recognition protocol settings (mirrors CompNet's test.py)."""

    device: str = "auto"                    # "auto" | "cuda" | "cuda:0" | "cpu"
    probe_chunk: int = 256                  # probes processed per distance chunk
    protocols: List[str] = field(
        default_factory=lambda: [
            "classification",               # primary: identity classification ACC
            "verification",                 # auxiliary: session1 vs session2
            "aggregated_verification",      # auxiliary: per-class min distance
            "dataset_eer",                  # auxiliary: all pairs inside session2
            "rank1",                        # auxiliary: nearest-gallery identification
        ]
    )
    save_scores: bool = False               # save genuine/impostor distance arrays
    plots: bool = True                      # save GI / ROC / DET curves


@dataclass
class AttackConfig:
    """Adversarial attack settings for palmprint identity classification.

    Final evaluation metric is classification accuracy (ACC, top-1).  The
    default ``loss="ce"`` maximizes ArcFace cross-entropy for PGD/APGD/Patch;
    the CW attack uses its logit margin.  ``loss="matching"`` is an optional
    gallery-margin ablation.  ``epsilons`` are L-inf budgets on **pixel
    tensors in [0, 1]** (``ToTensor`` output, before ``NormSingleROI``),
    matching CSR's ``1/255, 2/255, 4/255`` setting.  ``NormSingleROI`` is
    applied inside ``CompNetAttacker.forward``.
    """

    # Objective: "ce" (classification ACC), "cw", "dlr" (APGD-DLR), or
    # "matching" (gallery ablation).
    loss: str = "ce"
    epsilons: List[float] = field(
        default_factory=lambda: [1 / 255, 2 / 255, 4 / 255]
    )
    pgd_steps: int = 10
    pgd_alpha_scale: float = 2.5
    cw_steps: int = 30
    cw_kappa: float = 0.0
    cw_alpha_scale: float = 2.5
    apgd_steps: int = 30
    patch_size: int = 24
    patch_steps: int = 100
    patch_lr: float = 5e-2
    batch_size: int = 16
    sample_n: int = 1000                   # probes attacked per run (null = all)
    preserve_background: bool = False      # keep exactly-zero ROI background intact
    input_range: Optional[List[float]] = field(
        default_factory=lambda: [0.0, 1.0]
    )                                      # valid pixel range (None disables clamp)


@dataclass
class ReferenceResults:
    """Numbers reported in the CompNet paper (Tongji, d=9, m=0.5)."""

    rank1: float = 100.0
    eer: float = 0.018
    source: str = "CompNet, IEEE SPL 2021 - Supplementary Table I (Tongji, d=9, m=0.5)"


# ======================== Top-level Config ========================

@dataclass
class CompNetConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    reference: ReferenceResults = field(default_factory=ReferenceResults)
    base_dir: str = str(PROJECT_DIR)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["base_dir"] = str(self.base_dir)
        return d


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _build_dataclass(cls, values: Optional[Dict[str, Any]]):
    """Instantiate a dataclass, rejecting unknown keys."""
    values = values or {}
    known = set(cls.__dataclass_fields__.keys())
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**values)


_TOP_LEVEL_KEYS = {
    "base_dir", "paths", "data", "model", "evaluation", "attack", "reference"
}


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> CompNetConfig:
    """Load configuration from YAML and apply optional nested overrides.

    Relative path fields are resolved against the repository root. A YAML may
    set ``base_dir`` explicitly to resolve them elsewhere.

    Args:
        path: YAML file. Defaults to ``config/default.yaml``.
        overrides: Nested dict, e.g. ``{"evaluation": {"device": "cpu"}}``.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config root must be a mapping: {path}")
    else:
        if path != DEFAULT_CONFIG_PATH:
            raise FileNotFoundError(f"Config file not found: {path}")

    if overrides:
        raw = _deep_merge(raw, overrides)

    unknown_top = set(raw) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown_top)}")

    base_dir = Path(raw.pop("base_dir", PROJECT_DIR)).resolve()

    paths = _build_dataclass(PathConfig, raw.get("paths")).resolve(base_dir)
    data = _build_dataclass(DataConfig, raw.get("data"))
    model = _build_dataclass(ModelConfig, raw.get("model"))
    evaluation = _build_dataclass(EvaluationConfig, raw.get("evaluation"))
    attack = _build_dataclass(AttackConfig, raw.get("attack"))
    reference = _build_dataclass(ReferenceResults, raw.get("reference"))

    if data.num_classes != model.num_classes:
        raise ValueError(
            "data.num_classes and model.num_classes must match: "
            f"{data.num_classes} vs {model.num_classes}"
        )

    return CompNetConfig(
        paths=paths,
        data=data,
        model=model,
        evaluation=evaluation,
        attack=attack,
        reference=reference,
        base_dir=str(base_dir),
    )
