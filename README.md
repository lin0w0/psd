# psd

掌纹身份分类模型的对抗攻击与测试时防御方法研究

本项目以 **CompNet**（Tongji 掌纹 ROI，600 类）为 **掌纹身份分类**基线：
主评测指标为身份分类准确率 **ACC（Top-1）**；Verification EER、
Aggregated Verification EER 和 Rank-1 保留为辅助指标。`attack/` 已集成
像素空间分类攻击（PGD / APGD / CW / Patch），`defense/` 用于测试时防御
研究。

## 目录结构

```
psd/
├── config/
│   ├── __init__.py              # 配置 dataclass + YAML 加载（路径相对项目根目录解析）
│   ├── default.yaml             # 默认配置（数据 / 模型 / 评测 / 论文参考值）
│   └── csr_reference.yaml       # 原 CSR 配置备份（参考用）
├── data/                        # 原始数据（整目录 git-ignore）
│   └── ROI/{session1,session2}/ # Tongji ROI：600 类 x 10 张 x 2 session
├── dataset/                     # 数据加载代码（避免与 HuggingFace datasets 重名）
│   ├── dataset.py               # PalmROIDataset + NormSingleROI（官方预处理）
│   └── manifest.py              # train.txt / test.txt 生成与校验
├── model/
│   ├── compnet.py               # 官方 CompNet 结构（模块名与权重严格对应）
│   └── loader.py                # 构建模型 + 加载官方权重 + 设备解析
├── attack/                      # 对抗攻击（移植自 CSR，已适配 CompNet）
├── defense/                     # 测试时防御实现（待扩展）
├── evaluation/
│   ├── classification.py        # 主任务：身份分类 ACC、F1、混淆矩阵
│   ├── metrics.py               # EER（对齐官方 getEER.py）、GI 直方图、TAR@FAR
│   ├── evaluator.py             # 特征提取 + 四种识别协议
│   └── report.py                # JSON / CSV / 图表输出
├── scripts/
│   ├── prepare_data.py          # 生成数据清单
│   ├── evaluate.py              # 身份分类 + 辅助匹配评测入口
│   └── generate_adv.py          # 身份分类对抗样本生成与指标对比
├── manifests/                   # 生成的数据清单（git-ignore）
├── results/                     # 评测输出与特征缓存（git-ignore）
├── net_params_tongji_600.pth    # 官方预训练权重（git-ignore）
├── requirements.txt
└── README.md
```

## 环境

已在 `pr` conda 环境验证：Python 3.10、PyTorch 2.5.1 + CUDA 12.1、
RTX 4070 Laptop（8 GB）。

```bash
conda activate pr
pip install -r requirements.txt
```

> 系统默认 `python`（3.12）未安装 torch，请使用 `pr` 环境的解释器。
>
> 注意：当前机器上原 `pr` 环境路径（`D:/code/research/envs/pr`）已不存在，
> 可用 Python 均未安装 torch。运行实验前需先重建该环境；在此之前只能做
> `python -m compileall` 语法检查。

## 复现结果

论文补充材料 Table I（Tongji，方向数 d=9，angular margin m=0.5）：
**Rank-1 100%，EER 0.018%**。

**主任务：身份分类 ACC**。session2 的 6000 张 probe 直接对 `model(x)` 的
600 类 ArcFace logits 取 `argmax`：

| 指标 | 值 |
|------|-----|
| Top-1 ACC | **100.0000%**（6000/6000） |
| Top-5 ACC | **100.0000%** |
| Macro F1 / Weighted F1 | **100.0000% / 100.0000%** |
| 逐类 ACC | 600/600 类均为 100% |

辅助身份匹配指标（session1 作为 gallery，session2 作为 probe）：

| 协议 | EER（fitted） | EER（mean） | Rank-1 |
|------|---------------|-------------|--------|
| Verification（session1 → session2） | **0.018297%** | 0.018315% | **100.00%** (6000/6000) |
| Aggregated verification（类内最小距离） | 0.016667% | 0.021856% | — |
| Dataset EER（session2 内部两两） | 0.006066% | 0.006737% | — |

- 论文的 0.018% EER 对应 **Verification** 协议，复现值与之一致；
  Rank-1 为 6000/6000，与论文的 100% 一致。
- Verification AUC = 0.99999988；TAR@FAR=1e-4 = 99.97%。

## 快速开始

```bash
# 在项目根目录执行
"D:/code/research/envs/pr/python.exe" scripts/prepare_data.py
"D:/code/research/envs/pr/python.exe" scripts/evaluate.py
```

常用参数：

```bash
# 只跑闭集分类（跳过 gallery/probe 特征提取，约 8 秒）
python scripts/evaluate.py --protocols classification

# 只跑验证 EER + Rank-1
python scripts/evaluate.py --protocols verification rank1

# 指定设备 / 批大小 / 输出目录
python scripts/evaluate.py --device cuda:0 --batch-size 64 --output-dir ./results

# 重新生成清单、忽略特征缓存
python scripts/evaluate.py --prepare-manifests --no-feature-cache

# 导出 genuine / impostor 距离，供攻击/防御实验复用
python scripts/evaluate.py --save-scores

# 切换到其他掌纹数据集（同时更新 data/model 的类别数）
python scripts/evaluate.py --data-root ./data/OTHER_ROI --num-classes 460 \
    --images-per-class 3
```

`scripts/` 会自动把项目根目录加入 `sys.path`，因此也可以直接用已激活环境的
`python` 运行。

## 评测协议（身份分类 ACC + 辅助匹配）

默认协议为 `classification + verification + aggregated_verification +
dataset_eer + rank1`。

0. **classification**（主任务）：对 `model(x)` 的 600 类 ArcFace 余弦
   logits 取 `argmax`，统计 Top-1 ACC / Top-5 ACC、macro/weighted F1、
   逐类精度和 600×600 混淆矩阵。输入仍走官方 `NormSingleROI`。
1. **verification**（辅助）：session2 每张图与 session1 全部 6000 张图匹配。
   距离为 `arccos(cosine)/pi`（特征已 L2 归一化），genuine 为同一掌纹。
2. **aggregated_verification**：每个 probe、每个类别取该类 10 张 gallery 图的
   最小距离，再计算 EER（官方 `scores_VeriEER_aggr`）。
3. **dataset_eer**：session2 内部所有两两组合（不使用 gallery）。
4. **rank1**：每个 probe 取距离最近的 gallery 样本，判断类别是否正确。

EER 计算对齐官方 `getEER.py`：`sklearn.metrics.roc_curve` +
`brentq/interp1d` 拟合 EER；同时给出官方脚本中的 `eer_1/2`（mean EER）。

## 输出

```
results/
├── report.json                    # 全部指标 + 分类混淆矩阵 + 配置元信息
├── summary.csv                    # 扁平化指标表
├── config.json                    # 本次运行的完整配置
├── features_session1_*.npz        # gallery 特征缓存（仅识别协议需要）
├── features_session2_*.npz        # probe 特征缓存（仅识别协议需要）
├── scores_verification.npz        # 仅 --save-scores 时生成
├── scores_aggregated_verification.npz
├── scores_dataset_eer.npz
└── plots/
    ├── classification_per_class_accuracy.png
    ├── gi_verification.png        # 官方 getGI 风格分数分布
    ├── scores_verification.png    # 原始距离直方图（对数纵轴）
    ├── gi_aggregated_verification.png
    ├── scores_aggregated_verification.png
    ├── gi_dataset_eer.png
    ├── scores_dataset_eer.png
    └── roc.png / det.png
```

## 对抗攻击 / 防御扩展点

身份分类基线可在几行内加载并复用：

```python
from config import load_config
from dataset import PalmROIDataset
from model import load_compnet
from evaluation import evaluate_classification, extract_features, run_protocols

cfg = load_config()
model, device = load_compnet(
    cfg.paths.weights, cfg.model.num_classes, cfg.evaluation.device
)
probe = PalmROIDataset("manifests/test.txt")

# 主任务：身份分类 ACC
cls = evaluate_classification(model, probe, device=device,
                              num_classes=cfg.data.num_classes)
print(cls["top1_accuracy"], cls["top5_accuracy"], cls["macro_f1"])

# 辅助：特征匹配识别协议
gallery = PalmROIDataset("manifests/train.txt")
g = extract_features(model, gallery, device=device)
p = extract_features(model, probe, device=device)
results = run_protocols(p, g, device=device)
```

- **输入空间**：`[N, 1, 128, 128]`，使用 `NormSingleROI` 归一化
  （黑色背景保持 0，非零像素零均值单位方差）。
- **特征空间**：`model.getFeatureCode(x)` 返回 L2 归一化的 512 维特征；
  匹配距离固定为 `arccos(cosine)/pi`。
- **攻击**：在 `attack/` 中实现，对 `ToTensor` 后的像素张量 `[0,1]` 做约束
  扰动；`NormSingleROI` 在 attacker 内部执行。可复用 `evaluation` 中的指标
  评估攻击前后的 Rank-1/EER 变化。
- **防御**：在 `defense/` 中实现，对输入或特征做测试时校正，再走同一套
  识别协议即可对比干净/受攻击条件下的性能。

## 对抗攻击（attack/）

`attack/` 已移植 CSR 的攻击框架，并按 **身份分类 ACC** 设计攻击损失。
**AdaptivePGD 和 GlobalColor 攻击已移除。** 攻击流程：

```
原始 ROI → Resize → ToTensor (像素 [0,1])
        → 添加 PGD / APGD / CW / Patch 扰动
        → NormSingleROI → CompNet → ArcFace logits
        → 分类损失（CE / CW logit margin）
```

### 攻击损失：分类 ACC

最终评价是 `argmax(model(x))` 的身份分类准确率 ACC，因此攻击损失直接使用
ArcFace 分类 logits：

- **PGD / APGD / Patch**（默认 `loss="ce"`）：最大化交叉熵
  \(\mathcal{L}_{CE}(z, y)\)，这是 ACC 最直接的可微代理。
- **APGD-DLR**（可选 `loss="dlr"`）：最大化 untargeted Difference of Logits
  Ratio
  \[
  \mathrm{DLR}(x,y)=\frac{\max_{j\ne y} z_j - z_y}{z_{\pi_1}-z_{\pi_3}}
  \]
  其中 \(\pi\) 为 logits 降序排列，\(z_{\pi_1},z_{\pi_3}\) 为最大/第三大
  logit。分母按 logit 跨度归一化，对尺度不敏感；在 CE 梯度饱和时通常更
  稳定。PGD/Patch 也支持 `loss="dlr"`，CW 不支持。
- **CW**（默认 `loss="cw"`）：最小化
  \[
  \mathrm{relu}\big(-(z_y - \max_{j\ne y} z_j) + \kappa\big)
  \]
  即 CW logit margin，把样本推过分类决策边界后停止。CW 默认使用 δ=0
  初始化：hinge 在样本满足 margin 后梯度为 0，随机初始化会被原样保留在
  最终扰动中（高频噪声）；PGD/APGD/Patch 仍默认随机初始化。
- **可选 `loss="cw"`**：让 PGD/APGD/Patch 也使用同一个 logit margin。
- **可选 `loss="matching"`**：gallery genuine/impostor margin 消融，用于检验
  “优化 verification/rank-1 统计量是否能迁移到 ACC”。

ArcFace logits 为 \(z_j = 30\cos(f(x'), w_j)\)，所以 logit margin
\(z_y - \max_{j\ne y} z_j\) 与“真实类原型余弦相似度 − 最佳其他类原型余弦
相似度”成比例；CW 实际上优化的是身份分类的原型 margin。

| CSR | 本项目 | 说明 |
|-----|--------|------|
| `CLIPAttacker` | `attack/attacker.py::CompNetAttacker` | 冻结 CompNet；`forward(x)` 内部先做 `NormSingleROI`，返回 ArcFace logits；`features(x)` 返回 512 维 L2 归一化特征 |
| `PGDAttack` | `attack/pgd.py` | 像素空间 L∞ + CE（默认） |
| `APGDAttack` | `attack/apgd.py` | APGD-CE（默认）/ APGD-DLR（`loss="dlr"`） |
| `CWAttack` | `attack/cw.py` | CW logit margin（默认） |
| `PatchAttack` | `attack/patch.py` | 局部方形 patch，CE（默认） |
| — | `attack/objectives.py` | `ClassificationMargin`（主）/ `GalleryMargin`（消融） |

### 输入空间与 epsilon

- 攻击对象是 **`ToTensor` 之后、`NormSingleROI` 之前的像素张量 `[0,1]`**，
  与 CSR 的 CLIP 像素攻击一致。
- 每个攻击步骤都做 `‖δ‖∞ ≤ ε` 与 `x_adv = clamp(x + δ, 0, 1)` 投影；
  epsilon 默认取 CSR 的 `1/255, 2/255, 4/255`。
- `NormSingleROI` 由 `CompNetAttacker` 在 forward 内部执行。官方实现使用原地
  布尔索引赋值，不可反向传播，因此 `attack/attacker.py::norm_single_roi`
  提供了一个可微、批量化的等价实现（与官方结果误差 < 5e-7）。
- `CompNetAttacker(preserve_background=True)` 可保持 ROI 黑色背景
  （像素恰为 0）不被扰动。

### 直接调用

```python
from torch.utils.data import DataLoader
from torchvision import transforms as T

from config import load_config
from dataset import PalmROIDataset
from model.loader import load_compnet
from attack import CompNetAttacker, PGDAttack

cfg = load_config()
model, device = load_compnet(cfg.paths.weights, cfg.model.num_classes, "auto")
attacker = CompNetAttacker(model, device=device)

# 关键：数据集只做 Resize + ToTensor，不做 NormSingleROI。
pixel_tf = T.Compose([T.Resize(128), T.ToTensor()])
loader = DataLoader(
    PalmROIDataset("manifests/test.txt", transform=pixel_tf), batch_size=16
)
images, labels = next(iter(loader))            # [B, 1, 128, 128], [B]

# 默认 loss="ce"：最大化 ArcFace 交叉熵，直接对应 ACC
attack = PGDAttack(attacker, epsilon=4 / 255, steps=10)
adv = attack.attack(images, labels)

# 可选消融：loss="matching" 需要额外传入 session1 gallery 特征
# attack = PGDAttack(attacker, ..., loss="matching",
#                    gallery_features=..., gallery_labels=...)
```

### 批量生成与评测

```bash
# 默认 CE 分类损失；PGD 三个 epsilon，1000 个分层采样的 probe
python scripts/generate_adv.py --attack pgd --sample-n 1000

# 其余 L∞ 攻击（CW 自动使用 logit margin）
python scripts/generate_adv.py --attack apgd cw --eps 0.0156862745

# Patch 并保存像素空间对抗样本
python scripts/generate_adv.py --attack patch --save-adv

# 非定向 APGD-DLR
python scripts/generate_adv.py --attack apgd --loss dlr --sample-n 1000

# 全部 L∞ 攻击统一使用 CW logit margin
python scripts/generate_adv.py --attack pgd apgd cw --loss cw --sample-n 1000

# 可选消融：gallery genuine/impostor margin
python scripts/generate_adv.py --attack pgd apgd --loss matching --sample-n 1000

# 全部攻击 / 全部协议 / 全量 probe
python scripts/generate_adv.py --attack all --sample-n 0
```

脚本默认输出到 `outputs/adv_samples/<run_name>/`（`run_name` 为时间戳，
可用 `--run-name` 指定），每次运行独立目录，避免覆盖历史结果：

- `adv_results.json`：baseline、每次攻击的指标、实际生效的 epsilon/步数/
  批大小/seed（`effective_attack`）与完整配置；
- `adv_summary.csv`：每个 `(attack, epsilon)` 一行；
- `baseline_confusion_matrix.npz`：干净基线的 600×600 混淆矩阵、逐类精度
  与支持数；
- `adv_<attack>_<eps>.npz`：`--save-adv` 时保存的**像素空间**对抗张量。

> 分层子集覆盖**全部 600 类**：`sample_n < 600` 时按类号等距抽样；`> 600`
> 时每类 1 张，余数在全部类上等距分配。若子集中每类都只有 1 张，
> `dataset_eer` 没有 genuine 对，默认协议会自动跳过并写入
> `meta.skipped_protocols`；如果显式 `--protocols dataset_eer` 请求则报错
> （需 `--sample-n >= 601`）。

实测参考（Tongji 全量 6000 probe，600 类 × 10；2026-11-17 全量运行；主任务：
身份分类 ACC，ε=4/255；PGD/APGD/Patch 用 CE，CW 用 logit margin，
APGD-DLR 用 `--loss dlr`；CW 使用当前默认的零初始化）：

| 攻击 | 设置 | Top-1 ACC | Top-5 ACC | ASR |
|------|------|-----------|-----------|-----|
| 干净 | — | 100.00% | 100.00% | 0.00% |
| PGD | 10 步 | 0.00% | 0.02% | 100.00% |
| APGD-CE | 30 步 | 0.00% | 0.00% | 100.00% |
| APGD-DLR | 30 步 | 0.00% | 10.45% | 100.00% |
| CW | 30 步 | 0.00% | **99.95%** | 100.00% |
| Patch (24 px) | 100 步 | 14.32% | 33.52% | 85.68% |

全量 epsilon 扫描（Top-1 ACC / ASR）：

| ε | PGD | APGD-CE | APGD-DLR | CW |
|---|-----|---------|----------|----|
| 1/255 | 49.67% / 50.33% | 42.53% / 57.47% | 44.07% / 55.93% | 45.07% / 54.93% |
| 2/255 | 0.17% / 99.83% | 0.07% / 99.93% | 0.10% / 99.90% | 0.10% / 99.90% |
| 4/255 | 0.00% / 100.00% | 0.00% / 100.00% | 0.00% / 100.00% | 0.00% / 100.00% |

辅助身份匹配指标（同一批对抗样本，ε=4/255）：

| 攻击 | Rank-1 | Verification EER |
|------|--------|------------------|
| 干净 | 100.00% | 0.0183% |
| PGD | 0.00% | 5.14% |
| APGD-CE | 0.00% | 2.13% |
| APGD-DLR | 0.02% | 25.2117% |
| CW | 22.08% | 0.3298% |
| Patch (24 px) | 15.10% | 21.2243% |

原始结果：

- `outputs/adv_samples/full_6000_ce_saved/`（PGD / APGD / CW / Patch；
  CW 零初始化；含全部 `adv_<attack>_<eps>.npz` 像素张量，约 1.1 GB）
- APGD-DLR 的数值来自更早的一次运行，其产物已在缓存清理时删除；
  如需复现：`python scripts/generate_adv.py --attack apgd --loss dlr --sample-n 0 --save-adv`。

> 观察：
> - **APGD-DLR 与 APGD-CE 在 Top-1 ACC 上接近**：1/255 时 DLR 略弱
>   （44.07% vs 42.53%），2/255/4/255 基本持平；差异主要体现在辅助指标
>   （1/255 时 DLR 的 Rank-1 60.57% vs CE 50.18%，EER 更低）。
> - **Patch 结果比历史版本强很多**（Top-1 14.32% vs 旧结果 99.67%）：
>   重构后 Patch 正确执行“最大化 CE”（旧实现实际在最小化 CE，方向反了）。
> - ACC 以 Top-1 为准；CW 的最小扰动只把样本推过 Top-1 决策边界，因此
>   Top-5 仍约 100%。辅助 EER 在极强攻击下可能非单调（genuine/impostor
>   距离同时被推大）。

### 扰动频谱分析（低 / 中 / 高频能量）

对扰动 δ = x_adv − x_clean 做二维 FFT，按径向频率把频谱划为低/中/高三个
环带，同时报告**能量占比**与**相对白噪声的富集度**（高频环带 bin 数远多于
低频，只看占比会高估高频；富集度 = 实际占比 / 白噪声期望占比）。

```bash
# 3 种 L∞ 攻击 x 3 个 epsilon，1000 个 probe（现场生成攻击）
python scripts/analyze_perturbation_spectrum.py --attacks pgd apgd cw --sample-n 1000

# 只看成功攻击的样本
python scripts/analyze_perturbation_spectrum.py --successful-only

# 复现旧行为（CW 也使用随机初始化，观察初始化噪声残留）
python scripts/analyze_perturbation_spectrum.py --attacks cw --random-start

# 复用已保存的对抗张量：先全量生成并落盘，再直接读张量做频谱分析
python scripts/generate_adv.py --attack all --sample-n 0 --save-adv \
    --run-name full_6000_ce_saved
python scripts/analyze_perturbation_spectrum.py \
    --adv-dir outputs/adv_samples/full_6000_ce_saved \
    --attacks pgd apgd cw patch
```

默认边界 ρ = 1/3、2/3（ρ = 1 为轴向 Nyquist），可用 `--band-edges` 修改。
输出到 `results/perturbation_spectrum/`：`REPORT.md`、`summary.json`、
`band_energy.csv`、`radial_profile.csv`、`plots/`。

Tongji / CompNet，全量 6000 probe，seed=0（读取已保存张量）的结果：PGD 与
APGD 在三个 epsilon 下约 63–67% 的扰动能量落在低频带（富集度 7.2–7.7×），
中/高频富集度 < 1；CW 在其优化方向上也以低频为主（约 65–68%）。Patch 由于
是局部方形扰动，频谱更宽：Low 47.6%、High 27.2%，但高频富集度仍只有 0.42。

> CW 默认改为 δ=0 初始化：CW 的 hinge 损失在样本满足 margin 后梯度为 0，
> 旧实现会把随机初始化原样保留在最终扰动中，使 2/255、4/255 下的 CW 频谱
> 出现虚假的高频（High 37% / 54%）。`CWAttack` / `build_attack` 现在默认对
> CW 使用零初始化；PGD/APGD/Patch 仍使用随机初始化。零初始化不降低攻击
> 成功率（1/255：54.93% vs 54.20%；2/255：99.90% vs 99.87%；4/255 均为 100%）。

## 与官方实现的对应关系

| 官方文件 | 本项目 |
|----------|--------|
| `models/compnet.py` | `model/compnet.py`（结构、参数名不变） |
| `models/dataset.py` | `dataset/dataset.py` |
| `data/for_reference/genText_tongji.py` | `dataset/manifest.py` |
| `test.py`（特征提取 + 三种 EER + Rank-1） | `evaluation/evaluator.py` |
| `getEER.py` / `getGI.py` | `evaluation/metrics.py` |

## 注意事项

- **数据清单**：`manifests/manifest_meta.json` 记录生成参数；当 `--data-root`、
  `--num-classes`、`--images-per-class` 或 `--label-rule` 变化时，脚本会自动
  重新生成清单，不会误用旧数据。`--prepare-manifests` 可强制重建。
- **特征缓存**：缓存文件名包含权重文件指纹、清单内容哈希和预处理参数，
  更换权重 / 数据 / 输入尺寸会自动失效；仍可用 `--no-feature-cache` 禁用缓存。
- **内存**：Verification 与 Dataset EER 分别处理约 3600 万 / 1800 万个距离值；
  脚本按 `--chunk` 分块计算，仅保留 genuine/impostor 数组。
- **可复现性**：推理阶段无随机性（`eval()` + `torch.no_grad()`），
  数据顺序与官方一致（文件名排序，每 10 张同一掌纹）。
- **标签规则**：Tongji 官方规则为 `label = (int(文件名) - 1) // 10`；
  文件名非数字时可使用 `--label-rule sequential`。
