#!/usr/bin/env python
"""Analyze where adversarial perturbations live in the 2-D frequency plane.

For every attack/epsilon the script regenerates adversarial pixel tensors,
forms ``delta = x_adv - x_clean`` and computes the centered 2-D power
spectrum of each perturbation.  The radial frequency axis is split into
low / mid / high bands, and band energies are reported as

* a fraction of the total perturbation energy, and
* an enrichment factor over an isotropic (white-noise) spectrum,

because raw band fractions alone are dominated by the number of FFT bins in
each ring.

Pipeline (pixel space, same as the attacks)::

    ROI -> Resize -> ToTensor -> [attack] -> delta = adv - clean
        -> FFT2 -> radial bands -> energy fractions

Usage::

    # quick look: 3 attacks x 3 epsilons, 1000 probes
    python scripts/analyze_perturbation_spectrum.py

    # one epsilon and only successfully attacked probes
    python scripts/analyze_perturbation_spectrum.py \
        --eps 0.0078431373 --successful-only

    # all 6000 probes, include the patch attack
    python scripts/analyze_perturbation_spectrum.py \
        --sample-n 0 --attacks pgd apgd cw patch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from attack import CompNetAttacker  # noqa: E402
from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from dataset.dataset import PalmROIDataset  # noqa: E402
from defense.perturbation_spectrum import (  # noqa: E402
    BAND_LABELS,
    BAND_NAMES,
    DEFAULT_EDGES,
    SENSITIVITY_EDGES,
    SpectrumBands,
    analyze_deltas,
    cumulative_energy,
    plot_band_fractions,
    plot_cumulative_energy,
    plot_examples,
    plot_radial_spectrum,
    radial_density,
    summarize_bands,
)
from model.loader import load_compnet, resolve_device  # noqa: E402
from scripts.generate_adv import (  # noqa: E402
    NORM_BOUNDED,
    SubsetDataset,
    build_attack,
    ensure_manifests,
    resolve_attack_loss,
    select_probe_indices,
)

ATTACK_NAMES = ["pgd", "apgd", "cw", "patch"]
DEFAULT_ATTACKS = ["pgd", "apgd", "cw"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-D FFT band-energy analysis of adversarial perturbations."
    )
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--attacks", nargs="+", choices=ATTACK_NAMES, default=None,
                   help="Attacks to analyze. Default: pgd/apgd/cw when "
                        "regenerating; all saved tensors when using --adv-dir.")
    p.add_argument("--adv-dir", type=str, default=None,
                   help="Read saved adversarial tensors from this directory "
                        "(adv_<attack>_<eps>.npz produced by "
                        "`generate_adv.py --save-adv`) instead of generating "
                        "attacks. Clean pixels are rebuilt from the saved "
                        "probe indices. --sample-n/--seed are ignored.")
    p.add_argument("--eps", nargs="+", type=float, default=None,
                   help="L-inf budgets in pixel [0,1] "
                        "(default: config attack.epsilons).")
    p.add_argument("--loss", choices=["ce", "cw", "dlr", "matching"], default=None,
                   help="Attack objective (default: config attack.loss).")
    p.add_argument("--sample-n", type=int, default=None,
                   help="Probes to attack (default: config attack.sample_n; "
                        "0 = all).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Default: results/perturbation_spectrum")
    p.add_argument("--band-edges", nargs=2, type=float,
                   default=list(DEFAULT_EDGES),
                   help="Two ascending band edges as fractions of the axis "
                        "Nyquist frequency (default: 1/3 2/3).")
    p.add_argument("--successful-only", action="store_true", help="Analyze only probes whose prediction changed.")
    p.add_argument("--random-start", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Force random initialization for every attack "
                        "(--random-start) or zero initialization for every "
                        "attack (--no-random-start). Default: per-attack — "
                        "PGD/APGD/Patch use uniform noise, CW uses delta=0, "
                        "because CW's hinge has zero gradient after success "
                        "and would otherwise freeze the random noise into the "
                        "final perturbation.")
    p.add_argument("--save-per-image", action="store_true",
                   help="Write per-image band energies (one row per image).")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pixels(dataset, batch_size: int):
    """Collect pixel tensors ``[N, 1, H, W]`` and labels once."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    xs: List[torch.Tensor] = []
    ys: List[np.ndarray] = []
    for images, labels in loader:
        xs.append(images)
        ys.append(labels.numpy().astype(np.int64))
    return torch.cat(xs, dim=0), np.concatenate(ys)


@torch.no_grad()
def predict(attacker: CompNetAttacker, pixels: torch.Tensor,
            batch_size: int, device: torch.device) -> np.ndarray:
    preds: List[np.ndarray] = []
    for start in range(0, pixels.shape[0], batch_size):
        batch = pixels[start:start + batch_size].to(device)
        preds.append(attacker(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


@torch.enable_grad()
def generate_adversarial(attack, clean: torch.Tensor, labels: np.ndarray,
                         batch_size: int, device: torch.device) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    labels_t = torch.from_numpy(labels)
    for start in range(0, clean.shape[0], batch_size):
        end = min(start + batch_size, clean.shape[0])
        x = clean[start:end].to(device)
        y = labels_t[start:end].to(device)
        chunks.append(attack.attack(x, y).detach().cpu())
    return torch.cat(chunks, dim=0)


def epsilon_label(attack: str, eps: Optional[float]) -> str:
    if eps is None:
        return attack.upper()
    return f"{eps * 255.0:.0f}/255"


def list_saved_runs(
    adv_dir: str | Path,
    attacks: Optional[Sequence[str]] = None,
    epsilons: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Discover ``adv_<attack>_<eps>.npz`` tensors saved by ``generate_adv.py``."""
    adv_dir = Path(adv_dir)
    if not adv_dir.is_dir():
        raise FileNotFoundError(f"adv dir not found: {adv_dir}")
    wanted = set(attacks) if attacks else None
    runs: List[Dict] = []
    for path in sorted(adv_dir.glob("adv_*.npz")):
        rest = path.stem[len("adv_"):]
        if "_" not in rest:
            continue
        attack, tag = rest.rsplit("_", 1)
        if attack not in ATTACK_NAMES:
            continue
        if wanted is not None and attack not in wanted:
            continue
        epsilon = None if tag == "none" else float(tag)
        if epsilons is not None:
            if epsilon is None or not any(
                abs(epsilon - float(e)) <= 5e-5 for e in epsilons
            ):
                continue
        runs.append(
            {"attack": attack, "epsilon": epsilon, "tag": tag, "path": path}
        )
    order = {name: i for i, name in enumerate(ATTACK_NAMES)}
    runs.sort(
        key=lambda r: (
            order.get(r["attack"], 99),
            -1.0 if r["epsilon"] is None else r["epsilon"],
        )
    )
    if not runs:
        raise FileNotFoundError(f"no matching adv_*.npz under {adv_dir}")
    return runs


def load_saved_metadata(adv_dir: str | Path) -> tuple[Dict, Dict]:
    """Read ``adv_results.json`` next to the saved tensors, if present."""
    path = Path(adv_dir) / "adv_results.json"
    if not path.exists():
        return {}, {"epsilon": {}, "loss": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta", {})
    exact_epsilon: Dict[tuple, float] = {}
    losses: Dict[str, str] = {}
    for row in data.get("results", []):
        eps = row.get("epsilon")
        if eps is not None:
            exact_epsilon[(row["attack"], round(float(eps), 5))] = float(eps)
        losses[row["attack"]] = row.get("loss")
    return meta, {"epsilon": exact_epsilon, "loss": losses}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def fmt_pct(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def interp_at(x: np.ndarray, y: np.ndarray, xq: float) -> Optional[float]:
    if x.size == 0:
        return None
    return float(np.interp(xq, x, y))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def flatten_bands(run: Dict, summary: Dict, partition: str,
                  edges: Sequence[float]) -> List[Dict]:
    rows = []
    for name in BAND_NAMES:
        entry = summary["bands"][name]
        rows.append(
            {
                "attack": run["attack"],
                "epsilon": "" if run["epsilon"] is None else run["epsilon"],
                "epsilon_label": run["epsilon_label"],
                "n_images": summary["n_images"],
                "partition": partition,
                "edge_low_norm": edges[0],
                "edge_high_norm": edges[1],
                "band": name,
                "mean_fraction": entry["mean_fraction"],
                "std_fraction": entry["std_fraction"],
                "median_fraction": entry["median_fraction"],
                "q25_fraction": entry["q25_fraction"],
                "q75_fraction": entry["q75_fraction"],
                "mean_fraction_ac": entry["mean_fraction_ac"],
                "expected_fraction_white_noise": entry["expected_fraction_white_noise"],
                "enrichment_over_white_noise": entry["enrichment_over_white_noise"],
                "band_bin_count": entry["band_bin_count"],
                "dominant_image_fraction": summary["dominant_band_fractions"][name],
            }
        )
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    meta: Dict,
    runs: Sequence[Dict],
    bands: SpectrumBands,
    sensitivity_bands: SpectrumBands,
    figures: Dict[str, str],
) -> None:
    """Write a Chinese Markdown report driven by the computed numbers."""
    e0, e1 = bands.edges_norm
    f0, f1 = bands.edge_cycles_per_px
    lines: List[str] = []
    lines.append("# 对抗扰动的二维频谱能量分布")
    lines.append("")
    lines.append(
        "本报告由 `scripts/analyze_perturbation_spectrum.py` 根据实际计算结果自动生成，"
        "所有数值均来自同目录下的 `summary.json` / `band_energy.csv` / `radial_profile.csv`。"
    )
    lines.append("")
    lines.append("## 1. 分析方法")
    lines.append("")
    lines.append(
        "- 扰动定义在攻击操作的像素域：δ = x_adv − x_clean，"
        "与攻击的像素域 L∞ 约束同一空间。"
    )
    lines.append(
        "- 对每张图的 δ 做二维 FFT 并移到中心，功率谱 "
        "P(u,v) = |FFT2(δ)|² / (H·W)²，按 Parseval 有 ΣP = mean(δ²)。"
    )
    lines.append(
        f"- 径向频率 ρ = ƒ / 0.5 cycles/px，ρ=1 为轴向 Nyquist；"
        f"默认在 ρ={e0:.4f} 和 ρ={e1:.4f} 处划分低/中/高三个环带"
        f"（对应 {f0:.4f} 与 {f1:.4f} cycles/px，即轴向 Nyquist 的 "
        f"{e0:.2f} / {e1:.2f} 倍）。"
    )
    lines.append(
        "- **频段能量占比** = 该环带内功率和 / 全频谱功率和。"
        "**白噪声期望占比** = 该环带 FFT bin 数 / 总 bin 数。"
        "**富集度** = 实际占比 / 白噪声期望占比；富集度 > 1 表示该频段"
        "比各向同性噪声更集中，< 1 表示更少。只看占比会被高频环带更大的"
        "面积（更多 bin）误导，因此富集度是主要判据。"
    )
    lines.append(
        f"- 每张图单独归一化后再对图像取平均；表中同时给出均值 ± 样本标准差。"
        f"替代划分 ρ={SENSITIVITY_EDGES[0]:.2f}/{SENSITIVITY_EDGES[1]:.2f} "
        "用于稳健性检查。"
    )
    lines.append("")
    lines.append("## 2. 实验设置")
    lines.append("")
    lines.append(f"- 模型：CompNet（Tongji ROI，{meta['height']}×{meta['width']} 单通道）")
    if meta.get("source") == "saved_npz":
        lines.append(
            f"- 数据来源：读取已保存的对抗张量 `{meta.get('adv_dir')}`"
            "（`generate_adv.py --save-adv`），未重新生成攻击。"
        )
    else:
        lines.append("- 数据来源：本次运行重新生成对抗样本。")
    lines.append(f"- 攻击：{', '.join(a.upper() for a in meta['attacks'])}；"
                 f"目标函数 loss={meta['loss']}")
    lines.append(f"- epsilon：{meta['epsilons']}（像素域 [0,1] L∞；Patch 无 epsilon）")
    lines.append(f"- 样本数：{meta['n_samples']}（seed={meta['seed']}）；"
                 f"攻击成功率随行给出")
    if args.successful_only:
        lines.append("- **只统计成功攻击的样本**（干净正确且对抗预测改变）。")
    init_rs = meta.get("random_start", args.random_start)
    if init_rs is True:
        init_text = "所有攻击 δ ~ U(-ε, ε)（`--random-start`，强制随机初始化）。"
    elif init_rs is False:
        init_text = "所有攻击 δ=0（`--no-random-start`，强制零初始化）。"
    else:
        init_text = (
            "PGD/APGD/Patch 使用 δ ~ U(-ε, ε)；CW 使用 δ=0。"
            "CW 的 hinge 在样本满足 margin 后梯度为 0，随机初始化会被原样"
            "保留在最终扰动中（表现为虚假的高频成分），故默认零初始化。"
        )
    lines.append("- 攻击初始化：" + init_text)
    lines.append(f"- 运行时间：{meta['elapsed_seconds']:.1f} s；设备：{meta['device']}")
    lines.append("")

    lines.append("## 3. 频段能量占比（默认划分）")
    lines.append("")
    header = (
        "| 攻击 | eps | 样本 | ASR | DC 占比 | "
        "Low 占比 | Mid 占比 | High 占比 | 白噪声 L/M/H | 富集度 L/M/H | 主导频段 |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for run in runs:
        s = run["summary"]
        b = s["bands"]
        dominant = max(BAND_NAMES, key=lambda n: b[n]["mean_fraction"])
        lines.append(
            f"| {run['attack'].upper()} | {run['epsilon_label']} | "
            f"{s['n_images']} | {fmt_pct(run['attack_success_rate'])} | "
            f"{fmt_pct(s['dc_fraction']['mean'])} | "
            f"{fmt_pct(b['low']['mean_fraction'])} | "
            f"{fmt_pct(b['mid']['mean_fraction'])} | "
            f"{fmt_pct(b['high']['mean_fraction'])} | "
            f"{fmt_pct(b['low']['expected_fraction_white_noise'], 1)} / "
            f"{fmt_pct(b['mid']['expected_fraction_white_noise'], 1)} / "
            f"{fmt_pct(b['high']['expected_fraction_white_noise'], 1)} | "
            f"{fmt(b['low']['enrichment_over_white_noise'], 2)} / "
            f"{fmt(b['mid']['enrichment_over_white_noise'], 2)} / "
            f"{fmt(b['high']['enrichment_over_white_noise'], 2)} | "
            f"{BAND_LABELS[dominant]} |"
        )
    lines.append("")
    lines.append(
        "注：占比之和为 100%（含 DC）；DC 只在低环带的 ρ=0 处，表里单独列出。"
        "若 DC 占比不可忽略，应结合下一节的 AC 口径一起看。"
    )
    lines.append("")

    lines.append("## 4. 去直流（AC）后的频段占比")
    lines.append("")
    lines.append(
        "DC 是扰动的全局均值（相当于整体加减一个常数），与纹理型频率成分无关。"
        "下表在总能量中扣除 DC 后重新计算，用于排除“整体亮度平移”对低频占比的影响。"
    )
    lines.append("")
    lines.append("| 攻击 | eps | DC 占比 | Low-AC | Mid-AC | High-AC |")
    lines.append("|---|---|---|---|---|---|")
    for run in runs:
        s = run["summary"]
        b = s["bands"]
        lines.append(
            f"| {run['attack'].upper()} | {run['epsilon_label']} | "
            f"{fmt_pct(s['dc_fraction']['mean'])} | "
            f"{fmt_pct(b['low']['mean_fraction_ac'])} | "
            f"{fmt_pct(b['mid']['mean_fraction_ac'])} | "
            f"{fmt_pct(b['high']['mean_fraction_ac'])} |"
        )
    lines.append("")

    lines.append("## 5. 累积能量曲线")
    lines.append("")
    lines.append(
        "累积曲线不依赖人为的频段边界：给出半径 ρ 以下所包含的扰动能量比例。"
        "下表为若干半径处的平均累积占比，可据此自行选择划分。"
    )
    lines.append("")
    lines.append("| 攻击 | eps | ρ≤0.25 | ρ≤0.5 | ρ≤0.75 | ρ≤1.0 | ρ@90% |")
    lines.append("|---|---|---|---|---|---|---|")
    for run in runs:
        x, mean, _ = run["cumulative"]
        vals = [interp_at(x, mean, q) for q in (0.25, 0.5, 0.75, 1.0)]
        # Radius at which 90% of the energy is contained.
        rho90 = None
        above = np.nonzero(mean >= 0.90)[0]
        if above.size:
            rho90 = float(x[above[0]])
        lines.append(
            f"| {run['attack'].upper()} | {run['epsilon_label']} | "
            + " | ".join(fmt_pct(v) for v in vals)
            + f" | {fmt(rho90, 3)} |"
        )
    lines.append("")

    lines.append("## 6. 稳健性：替代频段划分")
    lines.append("")
    se0, se1 = sensitivity_bands.edges_norm
    lines.append(
        f"将边界改为 ρ={se0:.2f} / {se1:.2f}"
        f"（{0.5 * se0:.4f} / {0.5 * se1:.4f} cycles/px）后，各频段占比："
    )
    lines.append("")
    lines.append("| 攻击 | eps | Low | Mid | High | 富集度 L/M/H |")
    lines.append("|---|---|---|---|---|---|")
    for run in runs:
        b = run["sensitivity"]["bands"]
        lines.append(
            f"| {run['attack'].upper()} | {run['epsilon_label']} | "
            f"{fmt_pct(b['low']['mean_fraction'])} | "
            f"{fmt_pct(b['mid']['mean_fraction'])} | "
            f"{fmt_pct(b['high']['mean_fraction'])} | "
            f"{fmt(b['low']['enrichment_over_white_noise'], 2)} / "
            f"{fmt(b['mid']['enrichment_over_white_noise'], 2)} / "
            f"{fmt(b['high']['enrichment_over_white_noise'], 2)} |"
        )
    lines.append("")
    lines.append(
        "若两种划分给出的主导频段一致，则结论不依赖于边界选择；"
        "若不一致，应以后文的累积曲线为准，并明确报告所采用的边界。"
    )
    lines.append("")

    lines.append("## 7. 结论（按攻击/epsilon）")
    lines.append("")
    for run in runs:
        s = run["summary"]
        b = s["bands"]
        dominant = max(BAND_NAMES, key=lambda n: b[n]["mean_fraction"])
        enrich_rank = sorted(
            BAND_NAMES,
            key=lambda n: b[n]["enrichment_over_white_noise"],
            reverse=True,
        )
        x, mean, _ = run["cumulative"]
        rho90 = None
        above = np.nonzero(mean >= 0.90)[0]
        if above.size:
            rho90 = float(x[above[0]])
        lines.append(
            f"- **{run['attack'].upper()} @ {run['epsilon_label']}**："
            f"能量占比 Low/Mid/High = {fmt_pct(b['low']['mean_fraction'])} / "
            f"{fmt_pct(b['mid']['mean_fraction'])} / "
            f"{fmt_pct(b['high']['mean_fraction'])}；"
            f"相对白噪声的富集度依次为 "
            f"{fmt(b['low']['enrichment_over_white_noise'], 2)} / "
            f"{fmt(b['mid']['enrichment_over_white_noise'], 2)} / "
            f"{fmt(b['high']['enrichment_over_white_noise'], 2)} 倍，"
            f"富集度最高的是 **{BAND_LABELS[enrich_rank[0]]}** 频段；"
            f"按原始占比，主导频段为 **{BAND_LABELS[dominant]}**。"
            + (f"约 90% 的能量集中在 ρ≤{rho90:.2f}（≤{0.5 * rho90:.3f} cycles/px）以内。"
               if rho90 is not None else "")
        )
    lines.append("")
    lines.append(
        "**如何解读：** 原始占比回答“能量最多的是哪个频段”，但高频环带的 bin 数远多于低频；"
        "富集度回答“相对于均匀噪声，哪个频段被额外强化”。两者应一起报告。"
        "若原始占比以低频为主且富集度也在低频最大，说明扰动是低频集中的；"
        "若原始占比在中/高频但富集度在低频最大，说明扰动频谱整体仍偏低频，"
        "只是高频环带因为面积大而拿到更多总能量。"
    )
    lines.append("")

    lines.append("## 8. 图表")
    lines.append("")
    for title, rel in figures.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")
    if not figures:
        lines.append("（本次运行使用 `--no-plots`，未生成图表。）")
        lines.append("")

    lines.append("## 9. 注意事项")
    lines.append("")
    lines.append(
        "- 分析在**像素域**进行（攻击实际操作的张量）；CompNet 输入还会经过 "
        "`NormSingleROI` 的逐图标准化。该操作是逐图像素级的仿射缩放，"
        "不改变频谱形状，只整体缩放能量，因此频段占比结论不受影响。"
    )
    lines.append(
        "- 扰动经过 `[0,1]` 裁剪，裁剪会在频谱中引入额外的高频成分；"
        "这在像素域攻击中无法避免，解读时需注意。"
    )
    lines.append(
        "- 各频段边界是约定而非物理常数；请结合累积能量曲线和各向同性期望一起看。"
    )
    lines.append(
        "- 攻击初始化方式见第 2 节；使用随机初始化的攻击在不同 seed 下扰动"
        "细节不同，本报告使用固定 seed。"
    )
    lines.append(
        "- 若 ASR 较低，未被攻破样本的扰动幅度可能较小，会稀释频谱统计；"
        "可用 `--successful-only` 复跑对比。"
    )
    lines.append(
        "- **CW 的随机初始化残留（已修复）：** CW 使用 hinge 损失，样本一旦"
        "满足 margin 梯度即为 0，后续迭代不再更新该样本。旧实现使用随机初始化，"
        "早期成功的样本会把均匀随机噪声保留在最终 δ 中（频谱接近各向同性），"
        "从而把高 epsilon 下的 CW 频谱推向高频。现在 `CWAttack` 与 "
        "`build_attack` 默认对 CW 使用 δ=0；PGD/APGD/Patch 仍使用随机初始化。"
        "`--random-start` 可复现旧行为，`--no-random-start` 可让所有攻击零初始化。"
    )
    lines.append("")
    lines.append("## 10. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(meta["reproduce_command"])
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    overrides: Dict = {"evaluation": {}}
    if args.device:
        overrides["evaluation"]["device"] = args.device
    cfg = load_config(args.config, overrides)

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else PROJECT_DIR / "results" / "perturbation_spectrum"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    epsilons = args.eps or list(cfg.attack.epsilons)
    sample_n = cfg.attack.sample_n if args.sample_n is None else args.sample_n
    batch_size = args.batch_size or cfg.attack.batch_size
    loss = args.loss or cfg.attack.loss
    requested_attacks = args.attacks or (
        None if args.adv_dir else list(DEFAULT_ATTACKS)
    )
    saved_meta: Dict = {}
    saved_lookup: Dict = {"epsilon": {}, "loss": {}}
    if args.adv_dir:
        saved_meta, saved_lookup = load_saved_metadata(args.adv_dir)

    print("=" * 72)
    print("2-D FFT band-energy analysis of adversarial perturbations")
    print("=" * 72)
    print(
        "  attacks     : "
        + (", ".join(requested_attacks) if requested_attacks else "all saved")
    )
    print(f"  source      : {args.adv_dir or 'regenerate'}")
    print(f"  loss        : {loss}")
    if args.adv_dir:
        print("  epsilons    : from saved tensors")
        print("  sample_n    : from saved tensors")
    else:
        print(f"  epsilons    : {epsilons} (pixel-space L-inf, [0,1])")
        print(f"  sample_n    : {sample_n if sample_n else 'all'}")
    print(f"  band edges  : {list(args.band_edges)} (fraction of Nyquist)")
    print(f"  output      : {output_dir}")

    manifests = ensure_manifests(cfg)
    device = resolve_device(cfg.evaluation.device)
    print(f"[Device] {device}")

    model, device = load_compnet(
        weights=cfg.paths.weights,
        num_classes=cfg.model.num_classes,
        device=str(device),
    )
    attacker = CompNetAttacker(
        model,
        device=device,
        preserve_background=cfg.attack.preserve_background,
        input_range=cfg.attack.input_range,
    )

    pixel_transform = T.Compose([T.Resize(cfg.data.imside), T.ToTensor()])
    full_probe = PalmROIDataset(
        manifests["test_manifest"],
        imside=cfg.data.imside,
        outchannels=cfg.data.outchannels,
        transform=pixel_transform,
    )
    if args.adv_dir:
        jobs = list_saved_runs(args.adv_dir, requested_attacks, args.eps)
        clean_full, labels_full = load_pixels(full_probe, batch_size)
        clean_ref = clean_full
        clean_pred_full = predict(attacker, clean_full, batch_size, device)
        clean_correct_full = clean_pred_full == labels_full
        print(
            f"[Data] {len(jobs)} saved runs | "
            f"probe={clean_full.shape[0]} image="
            f"{cfg.data.imside}x{cfg.data.imside}, pixel range [0,1]"
        )
        print(
            f"[Baseline] clean correct = {int(clean_correct_full.sum())}"
            f"/{len(labels_full)}"
        )
    else:
        indices = select_probe_indices(
            full_probe.images_label,
            sample_n,
            cfg.data.num_classes,
            cfg.data.images_per_class,
        )
        probe_dataset = SubsetDataset(full_probe, indices)
        print(f"[Data] probe={len(probe_dataset)} image="
              f"{cfg.data.imside}x{cfg.data.imside}, pixel range [0,1]")
        clean, labels = load_pixels(probe_dataset, batch_size)
        clean_ref = clean
        clean_pred = predict(attacker, clean, batch_size, device)
        clean_correct = clean_pred == labels
        print(
            f"[Baseline] clean correct = {int(clean_correct.sum())}"
            f"/{len(labels)}"
        )
        jobs = []
        for name in requested_attacks:
            attacks_eps: List[Optional[float]] = (
                list(epsilons) if name in NORM_BOUNDED else [None]
            )
            jobs.extend(
                {"attack": name, "epsilon": eps} for eps in attacks_eps
            )

    height, width = int(clean_ref.shape[-2]), int(clean_ref.shape[-1])
    bands = SpectrumBands(height, width, args.band_edges)
    sensitivity_bands = SpectrumBands(height, width, SENSITIVITY_EDGES)
    print(f"[Bands] bins={bands.bin_counts.tolist()} "
          f"(white-noise fractions={bands.expected_fraction.round(4).tolist()})")

    runs: List[Dict] = []
    examples: List[Dict] = []
    per_image_rows: List[Dict] = []
    start_all = time.perf_counter()

    for job in jobs:
        name = job["attack"]
        epsilon = job["epsilon"]
        if args.adv_dir:
            key = (name, round(float(epsilon), 5)) if epsilon is not None else None
            if key is not None and key in saved_lookup["epsilon"]:
                epsilon = saved_lookup["epsilon"][key]
            label = epsilon_label(name, epsilon)
            print("\n" + "-" * 72)
            print(f"[Saved] {name} @ {label}  ({job['path'].name})")
            t0 = time.perf_counter()
            with np.load(job["path"], allow_pickle=True) as archive:
                adv = torch.from_numpy(
                    np.asarray(archive["adv"], dtype=np.float32)
                )
                idx = np.asarray(archive["indices"], dtype=np.int64)
                labels_run = np.asarray(archive["labels"], dtype=np.int64)
            clean_run = clean_full[torch.from_numpy(idx)]
            labels_ref = labels_full[idx]
            if labels_run.shape != labels_ref.shape or not np.array_equal(
                labels_run, labels_ref
            ):
                raise ValueError(
                    f"{job['path']}: saved labels do not match probe indices"
                )
            adv_pred = predict(attacker, adv, batch_size, device)
            clean_correct_run = clean_pred_full[idx] == labels_run
            attack_loss = saved_lookup["loss"].get(name) or loss
            source = f"saved:{job['path'].name}"
        else:
            label = epsilon_label(name, epsilon)
            print("\n" + "-" * 72)
            print(f"[Attack] {name} @ {label}")
            t0 = time.perf_counter()
            attack_loss = resolve_attack_loss(name, loss)
            attack = build_attack(
                name, attacker,
                0.0 if epsilon is None else epsilon,
                cfg, attack_loss, None, None,
                random_start=args.random_start,
            )
            adv = generate_adversarial(attack, clean, labels, batch_size, device)
            adv_pred = predict(attacker, adv, batch_size, device)
            clean_run, labels_run = clean, labels
            clean_correct_run = clean_correct
            source = "generated"

        success = clean_correct_run & (adv_pred != labels_run)
        n_total = int(labels_run.size)
        n_clean_correct = int(clean_correct_run.sum())
        asr = float(success.sum()) / max(1, n_clean_correct)

        delta = adv - clean_run
        if args.successful_only:
            keep = torch.from_numpy(success)
            delta_used = delta[keep]
        else:
            delta_used = delta
        if delta_used.shape[0] == 0:
            print("  no usable perturbations; skipping.")
            continue

        arrays = analyze_deltas(
            delta_used, (bands, sensitivity_bands), chunk_size=256
        )
        summary = summarize_bands(arrays, bands, "band_power_0")
        sens_summary = summarize_bands(arrays, sensitivity_bands, "band_power_1")
        rho, density = radial_density(arrays)
        cum_x, cum_mean, cum_std = cumulative_energy(arrays)

        run = {
            "attack": name,
            "epsilon": epsilon,
            "epsilon_label": label,
            "loss": attack_loss,
            "source": source,
            "n_total": n_total,
            "n_clean_correct": n_clean_correct,
            "n_success": int(success.sum()),
            "attack_success_rate": asr,
            "n_used": int(delta_used.shape[0]),
            "summary": summary,
            "sensitivity": {
                "edges_norm": list(sensitivity_bands.edges_norm),
                "bands": sens_summary["bands"],
                "n_images": sens_summary["n_images"],
                "dominant_band_fractions": sens_summary["dominant_band_fractions"],
            },
            "radial_density": density,
            "cumulative": (cum_x, cum_mean, cum_std),
            "elapsed_seconds": time.perf_counter() - t0,
        }
        runs.append(run)

        b = summary["bands"]
        print(
            f"  used {delta_used.shape[0]}/{n_total} images "
            f"(ASR {100.0 * asr:.1f}%)"
        )
        print(
            "  band fraction L/M/H = "
            f"{100 * b['low']['mean_fraction']:.1f}% / "
            f"{100 * b['mid']['mean_fraction']:.1f}% / "
            f"{100 * b['high']['mean_fraction']:.1f}%  "
            f"(DC {100 * summary['dc_fraction']['mean']:.2f}%)"
        )
        print(
            "  enrichment over white noise L/M/H = "
            f"{b['low']['enrichment_over_white_noise']:.2f} / "
            f"{b['mid']['enrichment_over_white_noise']:.2f} / "
            f"{b['high']['enrichment_over_white_noise']:.2f}"
        )

        if args.save_per_image:
            frac = arrays["band_power_0"] / arrays["total_energy"][:, None]
            for i in range(delta_used.shape[0]):
                per_image_rows.append(
                    {
                        "attack": name,
                        "epsilon": "" if epsilon is None else epsilon,
                        "epsilon_label": label,
                        "image_index": i,
                        "total_energy": float(arrays["total_energy"][i]),
                        "linf": float(arrays["linf"][i]),
                        "rms": float(arrays["rms"][i]),
                        "dc_fraction": float(
                            arrays["dc_energy"][i]
                            / max(float(arrays["total_energy"][i]), 1e-30)
                        ),
                        "low_fraction": float(frac[i, 0]),
                        "mid_fraction": float(frac[i, 1]),
                        "high_fraction": float(frac[i, 2]),
                    }
                )

        # Keep the first perturbation of each run as a figure example.
        if len(examples) < 12:
            examples.append(
                {
                    "label": f"{name.upper()} {label}",
                    "delta": delta_used[0, 0].numpy().astype(np.float64),
                }
            )

        del adv, delta, delta_used

    elapsed = time.perf_counter() - start_all
    if not runs:
        raise RuntimeError("No successful analysis runs; nothing to report.")
    attack_order = list(dict.fromkeys(r["attack"] for r in runs))

    # ---- figures ----
    figures: Dict[str, str] = {}
    if not args.no_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_radial_spectrum(
            runs, bands, plots_dir / "radial_psd.png", attack_order
        )
        plot_band_fractions(
            runs, bands, plots_dir / "band_fractions.png", attack_order
        )
        plot_cumulative_energy(
            runs, bands, plots_dir / "cumulative_energy.png", attack_order
        )
        plot_examples(examples, bands, plots_dir / "examples.png")
        figures["图 1：扰动径向功率谱"] = "plots/radial_psd.png"
        figures["图 2：低/中/高频段能量占比（虚线为白噪声期望）"] = (
            "plots/band_fractions.png"
        )
        figures["图 3：累积能量曲线"] = "plots/cumulative_energy.png"
        figures["图 4：单样本扰动及其对数功率谱"] = "plots/examples.png"
        print(f"[Plots] {plots_dir}")

    # ---- CSV / JSON ----
    band_rows: List[Dict] = []
    radial_rows: List[Dict] = []
    for run in runs:
        band_rows.extend(
            flatten_bands(run, run["summary"], "primary", bands.edges_norm)
        )
        band_rows.extend(
            flatten_bands(
                run, run["sensitivity"], "sensitivity",
                sensitivity_bands.edges_norm,
            )
        )
        x, mean, std = run["cumulative"]
        for i in range(x.size):
            radial_rows.append(
                {
                    "attack": run["attack"],
                    "epsilon": "" if run["epsilon"] is None else run["epsilon"],
                    "epsilon_label": run["epsilon_label"],
                    "ring": i,
                    "rho_norm": float(x[i]),
                    "freq_cycles_per_px": float(0.5 * x[i]),
                    "bin_count": int(bands.ring_counts[i]),
                    "mean_power_fraction_per_bin": float(run["radial_density"][i]),
                    "mean_cumulative_fraction": float(mean[i]),
                    "std_cumulative_fraction": float(std[i]),
                }
            )
    write_csv(
        output_dir / "band_energy.csv", band_rows,
        [
            "attack", "epsilon", "epsilon_label", "n_images", "partition",
            "edge_low_norm", "edge_high_norm", "band", "mean_fraction",
            "std_fraction", "median_fraction", "q25_fraction", "q75_fraction",
            "mean_fraction_ac", "expected_fraction_white_noise",
            "enrichment_over_white_noise", "band_bin_count",
            "dominant_image_fraction",
        ],
    )
    write_csv(
        output_dir / "radial_profile.csv", radial_rows,
        [
            "attack", "epsilon", "epsilon_label", "ring", "rho_norm",
            "freq_cycles_per_px", "bin_count", "mean_power_fraction_per_bin",
            "mean_cumulative_fraction", "std_cumulative_fraction",
        ],
    )
    if per_image_rows:
        write_csv(
            output_dir / "per_image_band_energy.csv", per_image_rows,
            [
                "attack", "epsilon", "epsilon_label", "image_index",
                "total_energy", "linf", "rms", "dc_fraction",
                "low_fraction", "mid_fraction", "high_fraction",
            ],
        )

    run_epsilons = sorted(
        {float(r["epsilon"]) for r in runs if r["epsilon"] is not None}
    )
    if args.adv_dir:
        random_start = (saved_meta.get("effective_attack") or {}).get(
            "random_start"
        )
        saved_loss = (saved_meta.get("effective_attack") or {}).get("loss")
        meta_loss = saved_loss or loss
        meta_seed = saved_meta.get("seed", args.seed)
        meta_sample_n = saved_meta.get("sample_n", max(r["n_total"] for r in runs))
        data_source = "saved_npz"
    else:
        random_start = args.random_start
        meta_loss = loss
        meta_seed = args.seed
        meta_sample_n = sample_n
        data_source = "regenerated"

    meta = {
        "model": "CompNet (Tongji ROI)",
        "height": height,
        "width": width,
        "source": data_source,
        "adv_dir": str(args.adv_dir) if args.adv_dir else None,
        "attacks": attack_order,
        "loss": meta_loss,
        "epsilons": run_epsilons if args.adv_dir else epsilons,
        "sample_n": meta_sample_n,
        "n_samples": int(clean_ref.shape[0]),
        "seed": meta_seed,
        "successful_only": bool(args.successful_only),
        "random_start": random_start,
        "random_start_policy": (
            "all_random" if random_start is True
            else "all_zero" if random_start is False
            else "per_attack_cw_zero"
        ),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "definitions": {
            "delta": "adv_pixels - clean_pixels",
            "power": "|FFT2(delta)|^2 / (H*W)^2, so sum(power) == mean(delta^2)",
            "rho": "frequency / (0.5 cycles/px); 1.0 == axis Nyquist",
            "enrichment": "observed_band_fraction / white_noise_expected_fraction",
            "white_noise_expected_fraction": "band_bin_count / (H*W)",
        },
        "bands_primary": bands.describe(),
        "bands_sensitivity": sensitivity_bands.describe(),
        "reproduce_command": " ".join(
            [Path(sys.executable).name, "scripts/analyze_perturbation_spectrum.py"]
            + sys.argv[1:]
        ),
    }
    summary_json = {
        "meta": meta,
        "runs": [
            {
                "attack": r["attack"],
                "epsilon": r["epsilon"],
                "epsilon_label": r["epsilon_label"],
                "loss": r["loss"],
                "source": r["source"],
                "n_total": r["n_total"],
                "n_clean_correct": r["n_clean_correct"],
                "n_success": r["n_success"],
                "attack_success_rate": r["attack_success_rate"],
                "n_used": r["n_used"],
                "elapsed_seconds": r["elapsed_seconds"],
                "summary": {
                    k: v for k, v in r["summary"].items()
                    if k not in (
                        "cumulative_energy",
                        "cumulative_ring_centers_norm",
                    )
                },
                "sensitivity": r["sensitivity"],
                "cumulative": {
                    "rho_norm": r["cumulative"][0].tolist(),
                    "mean_fraction": r["cumulative"][1].tolist(),
                    "std_fraction": r["cumulative"][2].tolist(),
                },
            }
            for r in runs
        ],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "effective_attack": {
                    "loss": loss,
                    "epsilons": epsilons,
                    "sample_n": sample_n,
                    "batch_size": batch_size,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[Output] {output_dir / 'summary.json'}")
    print(f"[Output] {output_dir / 'band_energy.csv'}")
    print(f"[Output] {output_dir / 'radial_profile.csv'}")

    write_report(
        output_dir / "REPORT.md", args, meta, runs, bands,
        sensitivity_bands, figures,
    )
    print(f"[Report] {output_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
