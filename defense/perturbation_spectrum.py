"""2-D FFT band-energy analysis of adversarial perturbations.

The perturbation is defined in pixel space (the tensors the attacks operate
on)::

    delta = x_adv - x_clean        # [B, 1, H, W], pixel units in [0, 1]

For every image we take the centered 2-D power spectrum

    P(u, v) = |FFT2(delta)(u, v)|^2 / (H * W)^2

normalized so that ``sum(P) = mean(delta^2)`` (Parseval).  Frequencies are
expressed as a radius ``rho`` normalized to the axis Nyquist frequency
(``rho = 1`` at ``0.5 cycles/pixel``; the corners reach ``sqrt(2)``).

The radial axis is split into three bands (``low``, ``mid``, ``high``) by
default at ``rho = 1/3`` and ``rho = 2/3``, i.e. equal thirds of the radial
Nyquist range.  For each band we report

* the fraction of the total perturbation energy in the band,
* the same fraction after removing the DC component,
* the **enrichment** over an isotropic (white-noise) spectrum, i.e.

      enrichment = observed_fraction / (band_bin_count / total_bin_count)

  This removes the area confound: a high-frequency ring contains many more
  FFT bins than a low-frequency one, so raw band fractions alone would make
  the high band look artificially large.

The module is pure numpy/torch (plus matplotlib for the optional figures) so
that it can be reused by other scripts/tests.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

BAND_NAMES = ("low", "mid", "high")
BAND_LABELS = {"low": "Low", "mid": "Mid", "high": "High"}
BAND_COLORS = {"low": "#0072B2", "mid": "#E69F00", "high": "#D55E00"}
DEFAULT_EDGES = (1.0 / 3.0, 2.0 / 3.0)  # rho, normalized to axis Nyquist
SENSITIVITY_EDGES = (0.25, 0.5)         # alternative split for robustness


# ---------------------------------------------------------------------------
# Frequency grid and band definition
# ---------------------------------------------------------------------------

def frequency_grid(height: int, width: int) -> np.ndarray:
    """Radial frequency map in cycles/pixel, ``fftshift`` layout."""
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


class SpectrumBands:
    """Frequency-band partition and radial-ring bookkeeping for one image size.

    Args:
        height, width: Spatial size of the perturbation.
        edges_norm: Two ascending band edges as fractions of the axis Nyquist
            frequency (``rho``).  Bands are half-open: ``[0, e0)``,
            ``[e0, e1)``, ``[e1, inf)``.
    """

    def __init__(
        self,
        height: int,
        width: int,
        edges_norm: Sequence[float] = DEFAULT_EDGES,
    ):
        edges = np.asarray(edges_norm, dtype=np.float64)
        if edges.shape != (2,) or not (0 < edges[0] < edges[1]):
            raise ValueError(f"edges must be two ascending positive values, got {edges}")
        if height <= 0 or width <= 0:
            raise ValueError(f"invalid image size {height}x{width}")

        self.height = int(height)
        self.width = int(width)
        self.edges_norm = tuple(float(e) for e in edges)
        self.n_bins = self.height * self.width

        freq = frequency_grid(self.height, self.width)
        self.freq_cycles_per_px = freq
        self.rho = freq / 0.5  # 1.0 == axis Nyquist

        self.labels = np.digitize(self.rho, edges).astype(np.int64)  # 0,1,2
        self.bin_counts = np.bincount(self.labels.ravel(), minlength=3)
        self.expected_fraction = self.bin_counts / float(self.n_bins)
        self.masks = torch.from_numpy(
            np.stack([self.labels == k for k in range(3)]).reshape(3, -1)
        )
        self.dc_index = (self.height // 2) * self.width + self.width // 2

        # Radial rings of width 1/min(H, W) cycles/pixel.
        self.scale = float(min(self.height, self.width))
        self.ring_index = np.rint(freq * self.scale).astype(np.int64)
        self.n_rings = int(self.ring_index.max()) + 1
        self.ring_counts = np.bincount(
            self.ring_index.ravel(), minlength=self.n_rings
        )
        self.ring_centers_cycles = np.arange(self.n_rings) / self.scale
        self.ring_centers_norm = self.ring_centers_cycles / 0.5

    @property
    def edge_cycles_per_px(self) -> Tuple[float, float]:
        return tuple(0.5 * e for e in self.edges_norm)  # type: ignore[return-value]

    def describe(self) -> Dict[str, object]:
        """Metadata for JSON/CSV output."""
        return {
            "height": self.height,
            "width": self.width,
            "band_edges_norm_of_nyquist": list(self.edges_norm),
            "band_edges_cycles_per_px": list(self.edge_cycles_per_px),
            "band_bin_counts": self.bin_counts.tolist(),
            "band_expected_fraction_white_noise": self.expected_fraction.tolist(),
            "total_bins": self.n_bins,
            "dc_index": self.dc_index,
        }


# ---------------------------------------------------------------------------
# Core per-image spectrum statistics
# ---------------------------------------------------------------------------

def _as_delta_batch(delta: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(delta):
        delta = torch.as_tensor(delta)
    if delta.dim() != 4:
        raise ValueError(f"expected [B, C, H, W] deltas, got {tuple(delta.shape)}")
    return delta


@torch.no_grad()
def analyze_deltas(
    delta: torch.Tensor,
    band_sets: Sequence[SpectrumBands],
    chunk_size: int = 256,
) -> Dict[str, np.ndarray]:
    """Per-image 2-D spectrum statistics of a batch of perturbations.

    Args:
        delta: ``[B, C, H, W]`` perturbation tensors (pixel units).
        band_sets: One or more :class:`SpectrumBands` on the same grid.  The
            first is the primary partition; the others are sensitivity splits.
        chunk_size: Images per FFT chunk.

    Returns:
        Dict of numpy arrays:

        * ``total_energy`` ``[B]`` -- ``mean(delta^2)`` per image.
        * ``dc_energy`` ``[B]`` -- energy of the DC bin.
        * ``linf`` ``[B]`` -- ``||delta||_inf``.
        * ``rms`` ``[B]`` -- ``sqrt(mean(delta^2))``.
        * ``band_power_0`` ... ``[B, 3]`` -- band energies per band set.
        * ``ring_power`` ``[B, n_rings]`` -- energy per radial ring (primary
          grid; all band sets share the grid).
        * ``ring_centers_norm`` / ``ring_counts``.
    """
    delta = _as_delta_batch(delta)
    if not band_sets:
        raise ValueError("at least one SpectrumBands is required")
    for bands in band_sets:
        if (bands.height, bands.width) != tuple(delta.shape[-2:]):
            raise ValueError(
                "band grid "
                f"{bands.height}x{bands.width} does not match delta "
                f"{delta.shape[-2]}x{delta.shape[-1]}"
            )

    bands0 = band_sets[0]
    height, width = bands0.height, bands0.width
    n = int(delta.shape[0])
    total = np.zeros(n, dtype=np.float64)
    dc = np.zeros(n, dtype=np.float64)
    linf = np.zeros(n, dtype=np.float64)
    rms = np.zeros(n, dtype=np.float64)
    band_power = [np.zeros((n, 3), dtype=np.float64) for _ in band_sets]
    ring_power = np.zeros((n, bands0.n_rings), dtype=np.float64)

    ring_index = torch.from_numpy(bands0.ring_index.ravel())
    norm = float(height * width) ** 2

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d = delta[start:end].to(dtype=torch.float64, device="cpu")
        spec = torch.fft.fftshift(
            torch.fft.fft2(d, dim=(-2, -1)), dim=(-2, -1)
        )
        power = spec.abs().square() / norm  # sum == mean(delta^2)
        flat = power.reshape(d.shape[0], -1)

        total[start:end] = flat.sum(dim=1).numpy()
        dc[start:end] = flat[:, bands0.dc_index].numpy()
        linf[start:end] = d.abs().amax(dim=(1, 2, 3)).numpy()
        rms[start:end] = d.square().mean(dim=(1, 2, 3)).sqrt().numpy()

        for i, bands in enumerate(band_sets):
            for k in range(3):
                band_power[i][start:end, k] = (
                    flat[:, bands.masks[k]].sum(dim=1).numpy()
                )
        rings = torch.zeros(
            d.shape[0], bands0.n_rings, dtype=torch.float64
        ).index_add_(1, ring_index, flat)
        ring_power[start:end] = rings.numpy()

    out: Dict[str, np.ndarray] = {
        "total_energy": total,
        "dc_energy": dc,
        "linf": linf,
        "rms": rms,
        "ring_power": ring_power,
        "ring_centers_norm": bands0.ring_centers_norm.copy(),
        "ring_centers_cycles": bands0.ring_centers_cycles.copy(),
        "ring_counts": bands0.ring_counts.copy(),
    }
    for i in range(len(band_sets)):
        out[f"band_power_{i}"] = band_power[i]
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise division with NaN where the denominator is non-positive."""
    num = np.asarray(num, dtype=np.float64)
    den = np.broadcast_to(np.asarray(den, dtype=np.float64), num.shape)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    valid = den > 0
    out[valid] = num[valid] / den[valid]
    return out


def summarize_bands(
    arrays: Dict[str, np.ndarray],
    bands: SpectrumBands,
    key: str = "band_power_0",
) -> Dict[str, object]:
    """Aggregate per-image band statistics for one attack/epsilon run.

    Args:
        arrays: Output of :func:`analyze_deltas`.
        bands: The band partition corresponding to ``key``.
        key: Which ``band_power_*`` array to summarize.
    """
    total = np.asarray(arrays["total_energy"], dtype=np.float64)
    band_power = np.asarray(arrays[key], dtype=np.float64)
    n = total.size
    eps = np.finfo(np.float64).eps

    fraction = _safe_div(band_power, total[:, None])
    ac_total = total - np.asarray(arrays["dc_energy"], dtype=np.float64)
    # DC always belongs to the low band because both edges are > 0.
    band_power_ac = band_power.copy()
    band_power_ac[:, 0] -= np.asarray(arrays["dc_energy"], dtype=np.float64)
    fraction_ac = _safe_div(band_power_ac, np.maximum(ac_total, eps)[:, None])

    ring_power = np.asarray(arrays["ring_power"], dtype=np.float64)
    cumulative = np.cumsum(ring_power, axis=1) / np.maximum(
        total, eps
    )[:, None]

    bands_out: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(BAND_NAMES):
        f = fraction[:, k]
        f = f[np.isfinite(f)]
        bands_out[name] = {
            "mean_fraction": float(np.mean(f)) if f.size else float("nan"),
            "std_fraction": float(np.std(f, ddof=1)) if f.size > 1 else 0.0,
            "median_fraction": float(np.median(f)) if f.size else float("nan"),
            "q25_fraction": float(np.quantile(f, 0.25)) if f.size else float("nan"),
            "q75_fraction": float(np.quantile(f, 0.75)) if f.size else float("nan"),
            "expected_fraction_white_noise": float(bands.expected_fraction[k]),
            "enrichment_over_white_noise": (
                float(np.mean(f) / bands.expected_fraction[k])
                if f.size and bands.expected_fraction[k] > 0
                else float("nan")
            ),
            "mean_fraction_ac": float(np.nanmean(fraction_ac[:, k])),
            "band_bin_count": int(bands.bin_counts[k]),
        }

    dominant_counts = {name: 0 for name in BAND_NAMES}
    valid_rows = np.isfinite(fraction).all(axis=1)
    if valid_rows.any():
        dominant = np.argmax(fraction[valid_rows], axis=1)
        for k, name in enumerate(BAND_NAMES):
            dominant_counts[name] = int(np.sum(dominant == k))

    return {
        "n_images": n,
        "total_energy": distribution_1d(total),
        "perturbation_linf": distribution_1d(arrays["linf"]),
        "perturbation_rms": distribution_1d(arrays["rms"]),
        "dc_fraction": distribution_1d(_safe_div(arrays["dc_energy"], total)),
        "bands": bands_out,
        "dominant_band_counts": dominant_counts,
        "dominant_band_fractions": {
            name: (count / max(1, int(valid_rows.sum())))
            for name, count in dominant_counts.items()
        },
        "cumulative_energy": cumulative,
        "cumulative_ring_centers_norm": bands.ring_centers_norm.copy(),
    }


def distribution_1d(values: np.ndarray) -> Dict[str, Optional[float]]:
    """Mean/std/median/quartiles of a 1-D sample (NaN-safe)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "mean": None, "std": None, "median": None,
            "q25": None, "q75": None, "min": None, "max": None,
        }
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
    }


def radial_density(
    arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean normalized power per FFT bin for each radial ring.

    Each image's ring power is first expressed as a fraction of that image's
    total perturbation energy, then divided by the number of bins in the ring
    and averaged across images.  The result integrates (over rings weighted by
    ring bin counts) to 1 and is directly comparable across rings.

    Returns:
        ``(rings_norm, density)``.
    """
    ring_power = np.asarray(arrays["ring_power"], dtype=np.float64)
    total = np.asarray(arrays["total_energy"], dtype=np.float64)
    frac = _safe_div(ring_power, total[:, None])
    density = np.nanmean(frac / np.maximum(arrays["ring_counts"], 1)[None, :], axis=0)
    return np.asarray(arrays["ring_centers_norm"]), density


def cumulative_energy(
    arrays: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean cumulative energy fraction vs normalized radius.

    Returns:
        ``(rings_norm, mean, std)`` where ``mean[i]`` is the average fraction
        of perturbation energy in bins with ``rho <= rings_norm[i]``.
    """
    ring_power = np.asarray(arrays["ring_power"], dtype=np.float64)
    total = np.asarray(arrays["total_energy"], dtype=np.float64)
    frac = _safe_div(ring_power, total[:, None])
    cum = np.cumsum(frac, axis=1)
    return (
        np.asarray(arrays["ring_centers_norm"]),
        np.nanmean(cum, axis=0),
        np.nanstd(cum, axis=0),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _line_style(attack: str, eps_index: int) -> Dict[str, object]:
    colors = {
        "pgd": "#0072B2",
        "apgd": "#E69F00",
        "cw": "#009E73",
        "patch": "#56B4E9",
    }
    linestyles = ["-", "--", ":", "-."]
    return {
        "color": colors.get(attack, "#666666"),
        "linestyle": linestyles[eps_index % len(linestyles)],
        "linewidth": 1.5,
    }


def plot_radial_spectrum(
    runs: Sequence[Dict],
    bands: SpectrumBands,
    path,
    attack_order: Sequence[str],
) -> None:
    """Mean normalized radial PSD per attack/epsilon, with band shading."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = bands.ring_centers_norm
    for attack in attack_order:
        attack_runs = [r for r in runs if r["attack"] == attack]
        for i, run in enumerate(attack_runs):
            density = run["radial_density"]
            label = attack.upper() + (
                "" if run["epsilon"] is None
                else f" eps={run['epsilon'] * 255.0:.0f}/255"
            )
            ax.semilogy(x, np.maximum(density, 1e-12), label=label,
                        **_line_style(attack, i))
    # Band shading: low [0, e0), mid [e0, e1), high [e1, sqrt(2)].
    e0, e1 = bands.edges_norm
    rmax = float(bands.ring_centers_norm[-1])
    for lo, hi, name in ((0.0, e0, "low"), (e0, e1, "mid"), (e1, rmax, "high")):
        ax.axvspan(lo, hi, color=BAND_COLORS[name], alpha=0.06, linewidth=0)
    for edge in (e0, e1):
        ax.axvline(edge, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, rmax)
    ax.set_xlabel(r"Radial frequency $\rho$ (1 = Nyquist, 0.5 cycles/px)")
    ax.set_ylabel("Normalized power per FFT bin")
    ax.set_title("Radial power spectrum of the perturbation")
    ax.legend(ncol=2, frameon=False, loc="lower left")
    fig.savefig(path)
    plt.close(fig)


def plot_band_fractions(
    runs: Sequence[Dict],
    bands: SpectrumBands,
    path,
    attack_order: Sequence[str],
) -> None:
    """Band energy fractions per attack/epsilon vs white-noise expectation."""
    import matplotlib.pyplot as plt
    import numpy as np

    _style()
    n_attacks = len(attack_order)
    fig, axes = plt.subplots(1, n_attacks, figsize=(4.0 * n_attacks, 3.6),
                             squeeze=False, sharey=True)
    width = 0.8 / 3
    for ax, attack in zip(axes[0], attack_order):
        attack_runs = [r for r in runs if r["attack"] == attack]
        positions = np.arange(len(attack_runs))
        for k, name in enumerate(BAND_NAMES):
            means = [r["summary"]["bands"][name]["mean_fraction"] for r in attack_runs]
            stds = [r["summary"]["bands"][name]["std_fraction"] for r in attack_runs]
            ax.bar(
                positions + (k - 1) * width, means, width=width,
                color=BAND_COLORS[name], alpha=0.85, label=BAND_LABELS[name],
                yerr=stds, capsize=2.0, error_kw={"elinewidth": 0.7},
            )
            ax.axhline(
                bands.expected_fraction[k], color=BAND_COLORS[name],
                linestyle=":", linewidth=1.0, alpha=0.9,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [r["epsilon_label"] for r in attack_runs], rotation=15, ha="right"
        )
        ax.set_title(attack.upper())
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0][0].set_ylabel("Fraction of perturbation energy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def plot_cumulative_energy(
    runs: Sequence[Dict],
    bands: SpectrumBands,
    path,
    attack_order: Sequence[str],
) -> None:
    """Cumulative energy fraction vs normalized radius."""
    import matplotlib.pyplot as plt

    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for attack in attack_order:
        attack_runs = [r for r in runs if r["attack"] == attack]
        for i, run in enumerate(attack_runs):
            x, mean, std = run["cumulative"]
            label = attack.upper() + (
                "" if run["epsilon"] is None
                else f" eps={run['epsilon'] * 255.0:.0f}/255"
            )
            style = _line_style(attack, i)
            ax.plot(x, mean, label=label, **style)
            ax.fill_between(x, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                            color=style["color"], alpha=0.08, linewidth=0)
    for edge in bands.edges_norm:
        ax.axvline(edge, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlim(0, float(bands.ring_centers_norm[-1]))
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(r"Radius $\rho$ (1 = Nyquist, 0.5 cycles/px)")
    ax.set_ylabel("Cumulative energy fraction")
    ax.set_title("Cumulative perturbation energy vs frequency radius")
    ax.legend(ncol=2, frameon=False, loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_examples(
    examples: Sequence[Dict],
    bands: SpectrumBands,
    path,
) -> None:
    """Perturbation image and log power spectrum for representative samples."""
    import matplotlib.pyplot as plt

    _style()
    n = len(examples)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.6), squeeze=False)
    for j, ex in enumerate(examples):
        delta = np.asarray(ex["delta"], dtype=np.float64)
        spectrum = np.fft.fftshift(
            np.abs(np.fft.fft2(delta)) ** 2
        )
        vmax = float(delta.max())
        im0 = axes[0][j].imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[0][j].set_title(ex["label"])
        axes[0][j].axis("off")
        fig.colorbar(im0, ax=axes[0][j], fraction=0.046, pad=0.02)
        im1 = axes[1][j].imshow(
            np.log10(np.maximum(spectrum, spectrum.max() * 1e-12)),
            cmap="viridis",
        )
        axes[1][j].set_title("log10 |FFT(delta)|^2")
        axes[1][j].axis("off")
        fig.colorbar(im1, ax=axes[1][j], fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
