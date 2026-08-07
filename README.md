<div align="center">

<h2>Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration</h2>

[Ruimeng Liu](https://github.com/cleste-pome)<sup>1</sup>, [Chang Tang](https://github.com/changtang)<sup>1</sup>, [Bo Wang]()<sup>1</sup>, [Zhenglai Li](https://github.com/guanyuezhen)<sup>2</sup>, [Lianbo Guo]()<sup>1</sup>, [Xinwang Liu](https://github.com/xinwangliu)<sup>3</sup>

<sup>1</sup>[Huazhong University of Science and Technology](https://www.hust.edu.cn/), <sup>2</sup>[Shenzhen Institutes of Advanced Technology](https://www.siat.ac.cn/), <sup>3</sup>[National University of Defense Technology](https://www.nudt.edu.cn/)

</div>

***

Welcome to the official implementation of **MASA** — a robust multi-view clustering framework built on the idea of **calibrating the fusion weights from early fusion to late fusion**: the early-fused global manifold serves as a stable structural anchor, and the view weights of the late-stage fusion are calibrated against it via manifold consistency (ELMC), while cross-view sparsity heterogeneity is first handled by adaptive view-specific encoding (AVE).

<details open><summary>📣 I also have other multi-view clustering projects that may interest you ✨.</summary><p>

> [**SparseMVC: Probing Cross-view Sparsity Variations for Multi-view Clustering**](https://openreview.net/pdf?id=cvJvk6oYfC)<br>
> Ruimeng Liu, Xin Zou, Chang Tang, Xiao Zheng, Xingchen Hu, Kun Sun, Xinwang Liu<br>
> [![Conference](http://img.shields.io/badge/NeurIPS-2025(Spotlight)-f9f107.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117045) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/cleste-pome/SparseMVC)

> [**Learning Disentangled Representations for Generalized Multi-view Clustering**](https://doi.org/10.1109/TPAMI.2026.3687339)<br>
> Xin Zou, Ruimeng Liu, Chang Tang, Zhenglai Li, Xinwang Liu, Kunlun He, Wanqing Li<br>
> [![Journal](http://img.shields.io/badge/TPAMI-2026-Bf107.svg)](https://doi.org/10.1109/TPAMI.2026.3687339) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/cleste-pome/GMAE)

</p></details>

<p align="center">
  <img src="docs/framework.png" alt="MASA framework" width="100%">
</p>

<p align="center">
The flowchart of our proposed MASA framework. Adaptive View-specific Encoding (AVE) probes the sparsity ratio of each view as prior knowledge for view-aware representation learning and constraint modulation; Early-to-late Manifold Consistency Calibration (ELMC) leverages the stable global manifold preserved in early-fused features to reweight the late-stage fusion of adaptively encoded local features.
</p>

### 📑 Table of Contents
- [🔗 Citation](#-citation)
- [📂 Source code list](#-source-code-list)
- [1. 📊 Dataset](#1-dataset)
- [2. ✅ Run](#2-run)
- [3. 🧮 Main Code](#3-main-code)
- [4. 🔬 Loss](#4-loss)
- [5. 🧩 Method Overview](#5-method-overview-three-core-modules)
- [6. 💻 User Guide](#6--user-guide-windows--linux--macos)

### 🔗 Citation
If our paper or code inspires you, please cite this paper when it is available😊:
```
@article{liu2026masa,
  author={Ruimeng, Liu and Chang, Tang and Bo, Wang and Zhenglai, Li and Lianbo, Guo and Xinwang, Liu},
  title={Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration},
  year={2026}
}
```

### 📂 Source code list:

```shell
MASA
├── train.py                          # 训练主程序（两阶段：AVE 预训练 → 一致性训练）
├── test.py                           # 测试程序（加载 .pth + 数据集 → 前向 → K-means 评估）
├── MASA.py                           # 模型定义（Network：Encoder/Decoder、投影头、循环一致性、加权融合）
├── GlobalLocalManifoldCalibration.py # ELMC 核心（拉普拉斯迹对齐视图权重，σ 自适应，可切换消融配置）
├── loss.py                           # 损失函数（对比损失 + 重建/KL 稀疏）
├── metric.py                         # 评价指标计算（ACC/NMI/PUR/ARI，K-means 评估）
├── main.tex                          # 论文 LaTeX 源文件
├── datasets                          # 数据集存放目录（.mat 格式：X 视图 cell 数组 + Y 标签）
└── utils                             # 辅助功能的工具包
    ├── dataloader.py                 # 数据集加载与预处理（min-max 归一化 + 噪声/冲突/缺失/稀疏注入）
    ├── device_check.py               # 设备前置检查（CUDA > MPS > CPU，MASA_DEVICE 可强制指定）
    ├── Logger.py                     # log 文档打印
    ├── metric2csv.py                 # 指标 CSV 记录
    ├── plot.py                       # 训练曲线与 ELMC σ 变化曲线
    └── tsne_visual.py                # t-SNE 可视化（pdf+svg 双输出）
```

## 1. 📊Dataset
- 数据格式：每个数据集一个 `.mat` 文件放入 `datasets/`，包含 `X`（各视图矩阵的 cell 数组）与 `Y`（样本标签）。
- 训练时 `train.py` 自动遍历 `datasets/` 下全部 `.mat` 数据集。
- 常用多视图数据集可参考：https://github.com/wangsiwei2010/awesome-multi-view-clustering

## 2. ✅Run

(1) To run the **training** (two-stage: AVE pretraining → consistency training), use:

```shell
python train.py
```

- 设备自动选择 **CUDA > MPS > CPU**；可用环境变量 `MASA_DEVICE=cuda|mps|cpu|auto` 强制指定。
- 训练内嵌 K-means 评估，输出目录自动创建：`1.logs/`（日志）、`2.results_imgs/`（曲线）、`3.csv/`（指标）、`4.models/`（.pth 权重）、`5.tsne/`（t-SNE 可视化）、`7.ViewWeights/`（每轮 ELMC 视图权重）。

(2) To run the **evaluation** with a trained model:

```shell
python test.py --model 4.models --datasets MSRCV1
```

也可以直接修改 `test.py` 顶部的 `MODEL_PATH` / `DATASETS` 两个变量后运行，或留空交互输入（权重路径优先级：`--model` > `MODEL_PATH` > 交互输入）。

## 🧮3. Main Code

### 3.1 Configuration

The dataset folder and output directories are handled automatically; the device is selected by `utils/device_check.py`.

```py
# Dataset folder path (all .mat files under it are trained in turn)
folder_path = "datasets"
# Output directories are created at runtime:
# 1.logs/ 2.results_imgs/ 3.csv/ 4.models/ 5.tsne/ 7.ViewWeights/
```

### 3.2 Hyperparameters

```py
# Number of epochs for the AVE pretraining stage
parser.add_argument("--pre_epochs", type=int, default=300)
# Number of epochs for the consistency training stage (ELMC + GLDA)
parser.add_argument("--con_epochs", type=int, default=300)
# Learning rate and weight decay
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--weight_decay", type=float, default=0.0)
# Feature dimensions (per-view encoder output / high-level common feature)
parser.add_argument("--feature_dim", type=int, default=64)
parser.add_argument("--high_feature_dim", type=int, default=20)
# Random seed and number of runs
parser.add_argument("--seed", type=int, default=50)
parser.add_argument("--iter", type=int, default=1)
```

### 3.3 Dataset Preprocessing

```py
# Select samples by noise ratio, then add Gaussian noise to (1..view-1) random views.
parser.add_argument('--noise_ratio', type=float, default=0.0)
# Select samples by conflict ratio, then replace one view's data with the same-view
# data of a sample from another class.
parser.add_argument('--conflict_ratio', type=float, default=0.0)
# Select samples by missing ratio, then set (1..view-1) random views to zero.
parser.add_argument('--missing_ratio', type=float, default=0.0)
# Select dimensions by sparsity ratio, then set random dimensions to zero.
parser.add_argument('--sparsity_ratio', type=float, default=0.0)
```

## 4. 🔬Loss

The overall objective integrates three terms:

```py
# 1. loss_rec：重建损失（MSE），约束每个视图与其全局表示的自编码重建
ae_loss_function(mean_average, xs2one, xr_all, activation[0], rho=0.05, beta=1.0)

# 2. loss_sparse：KL 稀疏约束，稀疏系数 C_spa 由各视图稀疏率自适应调节（AVE）
kl_sparse_loss(hidden_layer_activation, rho, sparse_beta)

# 3. loss_con：对比损失，对齐全局融合特征 H 与各视图公共信息（GLDA）
contrastiveloss(H, rs[v], w2[v])
```

Additionally, the **ELMC** module computes per-view fusion weights by Laplacian trace
alignment with an adaptive bandwidth σ (default: global pairwise-distance median). The
score form and σ setting can be switched at the top of `GlobalLocalManifoldCalibration.py`
(`SCORE_FORM` / `SIGMA_MODE`), corresponding to the ablation tables in the paper.

The overall objective of the consistency training stage is the **total loss**:

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{sparse} + \alpha\, \mathcal{L}_{con}, \quad \alpha = 1$$

where L_rec is the reconstruction error (term 1), L_sparse the adaptive KL-sparsity term (term 2), and L_con the contrastive alignment term (term 3).

## 5. 🧩 Method Overview: Three Core Modules

MASA is a robust multi-view clustering framework built on three core modules and trained in **two stages**: first *AVE pretraining* (reconstruction with adaptive sparsity), then *consistency training* (ELMC weighting + GLDA alignment). The aligned global representation is finally clustered by K-means into ACC / NMI / PUR / ARI.

**① AVE — Adaptive View-specific Encoding** handles the cross-view sparsity heterogeneity typical of multi-view data: the sparsity ratio of each view is probed from its input (`zero_value_proportion` in `MASA.py`) and used as prior knowledge to adaptively modulate the strength of the entropy-based sparse constraint — sparser views receive stronger sparse regularization, so that each view's encoder is tuned in a view-aware manner (adaptive coefficient C_spa in `ae_loss_function`, `loss.py`).

**② ELMC — Early-to-late Manifold Consistency Calibration** performs the **fusion-weight calibration from early fusion to late fusion**: the early-fused global representation provides a stable structural anchor; for each view, a Gaussian-kernel graph Laplacian is built with an adaptive bandwidth (median heuristic, re-estimated every epoch), and a consistency score (Laplacian trace alignment) measures how well the view's manifold aligns with the global anchor; after cross-view normalization, the resulting weights reweight the late-stage fusion of the adaptively encoded view features, automatically down-weighting unreliable views (`GlobalLocalManifoldCalibration.py`; the score form and bandwidth setting are switchable for the ablation study, and unsupported MPS ops fall back to CPU automatically).

<p align="center">
  <img src="docs/MSRCV1_acc.png" alt="Clustering accuracy on MSRCV1" width="70%">
</p>

<p align="center">
Clustering accuracy (ACC) on MSRCV1 during training.
</p>

**③ GLDA — Global-local Distribution Alignment** aligns the global fused representation with each view's local shared information: the fused representation H and the per-view common information r_v are L2-normalized before computing pairwise similarities, and a contrastive loss (temperature τ = 1) pulls them together — the normalization keeps the temperature τ meaningful regardless of feature scales; reconstruction and cycle-consistency terms preserve view-specific fidelity (`loss.py` + the consistency training stage of `train.py`). The global representation H is finally clustered by K-means (n_init=100) into ACC / NMI / PUR / ARI.

## 6. 💻 User Guide (Windows / Linux / macOS)

### ⚙️ Requirements

| Library | Version (dev) | 用途（对应论文部分） |
|---|---|---|
| python | 3.12（3.10+ 均可） | 运行环境 |
| pytorch | 2.x（按平台选 cu121 / cpu / MPS 官方 wheel） | 网络构建与两阶段训练（AVE 预训练 → 一致性训练） |
| numpy | ≥ 1.21 | 数值计算（数据与指标处理） |
| scipy | ≥ 1.10 | 读取 .mat 数据集（论文全部数据集） |
| scikit-learn | ≥ 1.0 | K-means 聚类评估：ACC / NMI / PUR / ARI（论文全部实验指标） |
| tqdm | ≥ 4.6 | 训练进度条 |
| matplotlib | ≥ 3.5 | 训练曲线 / ELMC σ 变化 / t-SNE 可视化（论文收敛性分析图） |
| tabulate | ≥ 0.9 | 分段计时报告 |

```shell
pip install numpy scipy scikit-learn tqdm matplotlib tabulate
```

- PyTorch is installed separately per platform (see below); the GPU K-means alternatives (cuML/cuPy) are optional and **not** required by default.

### Install PyTorch per platform

| Platform | Install command | Default device |
|---|---|---|
| Linux / Windows + NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu121` | CUDA |
| Linux / Windows (CPU only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | CPU |
| macOS (Apple Silicon) | `pip install torch` (official wheels include MPS support) | MPS |

### Device selection (automatic, no configuration needed)

- Decision rule: **CUDA > MPS > CPU**（`utils/device_check.py`，训练启动时自动探测）。
- Force a device with the environment variable `MASA_DEVICE=cuda|mps|cpu|auto` (unavailable or invalid values fall back to automatic):

```shell
MASA_DEVICE=cuda python train.py    # force CUDA (Linux / Windows)
MASA_DEVICE=mps  python train.py    # force MPS (macOS)
MASA_DEVICE=cpu  python train.py    # force CPU
```

- Standalone probe (prints the environment summary only, no training): `python utils/device_check.py`

### Platform notes

- **macOS**: MPS requires macOS ≥ 12.3 and an official PyTorch build; unsupported ops under MPS (`torch.cdist` / `torch.diag` in the ELMC module) automatically fall back to CPU — nothing to configure.
- **Linux / Windows without CUDA**: falls back to CPU automatically. `OMP_NUM_THREADS=1` is preset in `train.py` to avoid thread oversubscription on CPU.
- **CUDA**: to choose a specific GPU, set `CUDA_VISIBLE_DEVICES` (train.py presets `"0"`); multi-GPU is not required.

### Run

```shell
python train.py                                          # train on all .mat datasets under datasets/
python test.py --model 4.models --datasets MSRCV1        # evaluate a trained model
```
