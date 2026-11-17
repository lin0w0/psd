"""Gaussian low-pass filtering in pixel space for spectral-consistency analysis.

The filter is implemented with a 2-D FFT so that the radius is defined in
frequency-bin units on the network input grid (128 x 128 for Tongji ROI),
**not** as a spatial-domain ``GaussianBlur`` sigma.

Two modes are provided:

``full_frame``
    ``x_low = IFFT(FFT(x) * H_r)`` over the whole image.

``roi_aware``
    Normalized convolution against the original ROI mask ``M``::

        x_low = M * LPF(x * M) / (LPF(M) + eps)

    and the region outside ``M`` is zeroed afterwards.

Filtering happens on pixel tensors in ``[0, 1]`` (the output of
``ToTensor``), *before* the official ``NormSingleROI`` preprocessing.

Notes
-----
Tongji ROI images shipped with this project have no exactly-zero
background (every pixel is > 0), so ``M`` is all ones and ``roi_aware``
reduces to ``full_frame`` up to the ``eps`` stabilizer.  The implementation
still follows the requested normalized-convolution definition; the
equivalence is measured and reported by the evaluation script.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

FILTER_MODES = ("full_frame", "roi_aware")


def roi_mask(x: torch.Tensor) -> torch.Tensor:
    """Binary ROI mask ``M`` from pixel tensors.

    Args:
        x: ``[B, C, H, W]`` pixel tensors in ``[0, 1]``.

    Returns:
        ``[B, 1, H, W]`` float32 mask, 1 for pixels strictly greater than 0.
    """
    if x.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")
    return (x > 0).to(dtype=x.dtype).mean(dim=1, keepdim=True)


def gaussian_frequency_mask(
    height: int,
    width: int,
    radius: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Centered Gaussian low-pass transfer function ``H_r(u, v)``.

    ``H_r(u,v) = exp(-(u^2 + v^2) / (2 r^2))`` with ``(u, v)`` measured in
    FFT bins relative to the DC component (``fftshift`` layout).

    Returns:
        ``[H, W]`` tensor with maximum 1 at the center.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    device = torch.device(device)
    u = torch.arange(height, device=device, dtype=dtype) - height // 2
    v = torch.arange(width, device=device, dtype=dtype) - width // 2
    dist_sq = u[:, None].pow(2) + v[None, :].pow(2)
    return torch.exp(-dist_sq / (2.0 * float(radius) ** 2))


_MASK_CACHE: Dict[Tuple[int, int, float, str, str], torch.Tensor] = {}


def _cached_mask(
    height: int,
    width: int,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (height, width, float(radius), str(device), str(dtype))
    mask = _MASK_CACHE.get(key)
    if mask is None or mask.device != device or mask.dtype != dtype:
        mask = gaussian_frequency_mask(height, width, radius, device, dtype)
        _MASK_CACHE[key] = mask
    return mask


def _lowpass(x: torch.Tensor, radius: float) -> torch.Tensor:
    """FFT-domain low-pass of ``[B, C, H, W]``; returns the real part."""
    _, _, height, width = x.shape
    transfer = _cached_mask(
        height, width, radius, x.device, torch.float32
    ).to(dtype=x.dtype)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1)
    )
    spectrum = spectrum * transfer[None, None, :, :]
    out = torch.fft.ifft2(
        torch.fft.ifftshift(spectrum, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return out


def gaussian_lowpass(
    x: torch.Tensor,
    radius: float,
    mode: str = "full_frame",
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    clamp: bool = True,
) -> torch.Tensor:
    """Apply a Gaussian low-pass filter to pixel tensors.

    Args:
        x: ``[B, C, H, W]`` pixel tensors in ``[0, 1]``.
        radius: Gaussian standard deviation in FFT-bin units.
        mode: ``"full_frame"`` or ``"roi_aware"``.
        mask: Optional ROI mask ``[B, 1, H, W]`` (values in {0, 1}).  When
            omitted it is derived from ``x``.  For adversarial inputs pass
            the mask of the *clean* original so that the same region is
            filtered for both.
        eps: Stabilizer in the normalized-convolution denominator.
        clamp: Clamp the filtered pixels back to ``[0, 1]``.

    Returns:
        Filtered pixel tensors with the same shape/dtype as ``x``.
    """
    if mode not in FILTER_MODES:
        raise ValueError(f"unknown filter mode {mode!r}; choices={FILTER_MODES}")
    if x.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(x.shape)}")

    if mode == "full_frame":
        out = _lowpass(x, radius)
    else:
        m = roi_mask(x) if mask is None else mask
        if m.shape != x.shape:
            raise ValueError(
                f"mask shape {tuple(m.shape)} does not match input {tuple(x.shape)}"
            )
        m = m.to(dtype=x.dtype)
        low_x = _lowpass(x * m, radius)
        low_m = _lowpass(m, radius)
        out = m * low_x / (low_m + eps)

    if clamp:
        out = out.clamp(0.0, 1.0)
        if mode == "roi_aware":
            m = roi_mask(x) if mask is None else mask
            out = out * m.to(dtype=out.dtype)
    return out


def roi_mask_difference(
    a: torch.Tensor, b: torch.Tensor
) -> float:
    """Fraction of pixels whose ROI (``> 0``) membership differs."""
    return float((roi_mask(a) != roi_mask(b)).float().mean().item())
