"""CompNet model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import torch

from .compnet import compnet


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when available."""
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_compnet(
    weights: str | Path,
    num_classes: int = 600,
    device: str = "auto",
    strict: bool = True,
) -> tuple[compnet, torch.device]:
    """Build CompNet and load the official state dict.

    Args:
        weights: Path to ``net_params_tongji_600.pth``.
        num_classes: 600 for Tongji.
        device: ``"auto"``, ``"cuda"``, ``"cuda:0"`` or ``"cpu"``.
        strict: Whether every checkpoint key must match the architecture.

    Returns:
        (model, torch.device) with the model in eval mode and frozen.
    """
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    torch_device = resolve_device(device)

    model = compnet(num_classes=num_classes)
    checkpoint = torch.load(weights, map_location=torch_device, weights_only=True)

    # Support raw state dicts and common training wrappers.
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict):
        state_dict = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=strict)
    model = model.to(torch_device).eval()

    for param in model.parameters():
        param.requires_grad = False

    print(f"[CompNet] Loaded weights: {weights}")
    print(f"[CompNet] device={torch_device}, num_classes={num_classes}, "
          f"strict={strict}")
    return model, torch_device
