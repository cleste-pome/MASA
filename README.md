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
If this work or the code is helpful to you, please cite it when it is available😊:
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
├── train.py                          # Training entry (two-stage: AVE pretrain → consistency training)
├── test.py                           # Evaluation (load .pth + dataset → forward → K-means)
├── MASA.py                           # Model definition (Network: Encoder/Decoder, projection head, cycle consistency, weighted fusion)
├── GlobalLocalManifoldCalibration.py # ELMC core (Laplacian trace-alignment view weights, adaptive σ, switchable ablation configs)
├── loss.py                           # Loss functions (contrastive + reconstruction/KL sparsity)
├── metric.py                         # Metrics (ACC/NMI/PUR/ARI, K-means evaluation)
├── main.tex                          # LaTeX source
├── datasets                          # Dataset directory (.mat: X view-cell array + Y labels)
└── utils                             # Utilities
    ├── dataloader.py                 # Data loading & preprocessing (min-max norm + noise/conflict/missing/sparsity injection)
    ├── device_check.py               # Device probing (CUDA > MPS > CPU, MASA_DEVICE override)
    ├── Logger.py                     # Logging
    ├── metric2csv.py                 # Metric CSV export
    ├── plot.py                       # Training curves & ELMC σ curve
    └── tsne_visual.py                # t-SNE visualization (pdf+svg)
```

## 1. 📊Dataset
- Each dataset is a single `.mat` file placed under `datasets/`, containing `X` (a cell array of view matrices) and `Y` (sample labels).
- `train.py` automatically trains on every `.mat` file under `datasets/`.
- Public multi-view datasets: https://github.com/wangsiwei2010/awesome-multi-view-clustering

## 2. ✅Run

(1) To run the **training** (two-stage: AVE pretraining → consistency training), use:

```shell
python train.py
```

- The device is selected automatically (**CUDA > MPS > CPU**); use the environment variable `MASA_DEVICE=cuda|mps|cpu|auto` to force a specific device.
- Training embeds K-means evaluation; output directories are created automatically: `1.logs/` (logs), `2.results_imgs/` (curves), `3.csv/` (metrics), `4.models/` (.pth weights), `5.tsne/` (t-SNE), `7.ViewWeights/` (per-epoch ELMC view weights).

(2) To run the **evaluation** with a trained model:

```shell
python test.py --model 4.models --datasets MSRCV1
```

Alternatively, edit the `MODEL_PATH` / `DATASETS` variables at the top of `test.py`, or leave them empty for interactive input (weight-path priority: `--model` > `MODEL_PATH` > interactive input).

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
alignment with an adaptive bandwidth (default: the global pairwise-distance median). The
score form and bandwidth setting can be switched at the top of
`GlobalLocalManifoldCalibration.py` (`SCORE_FORM` / `SIGMA_MODE`), corresponding to the
ablation tables.

The overall objective of the consistency training stage is the **total loss**:

$$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{sparse} + \alpha\, \mathcal{L}_{con}, \quad \alpha = 1$$

where the first term is the reconstruction error, the second is the adaptive KL-sparsity term, and the third is the contrastive alignment term.

## 5. 🧩 Method Overview: Three Core Modules

MASA is a robust multi-view clustering framework built on three core modules and trained in **two stages**: first *AVE pretraining* (reconstruction with adaptive sparsity), then *consistency training* (ELMC weighting + GLDA alignment). The aligned global representation is finally clustered by K-means into ACC / NMI / PUR / ARI.

**① AVE — Adaptive View-specific Encoding** handles the cross-view sparsity heterogeneity typical of multi-view data: the sparsity ratio of each view is probed from its input (`zero_value_proportion` in `MASA.py`) and used as prior knowledge to adaptively modulate the strength of the entropy-based sparse constraint — sparser views receive stronger sparse regularization, so that each view's encoder is tuned in a view-aware manner (adaptive sparse coefficient in `ae_loss_function`, `loss.py`).

**② ELMC — Early-to-late Manifold Consistency Calibration** performs the **fusion-weight calibration from early fusion to late fusion**: the early-fused global representation provides a stable structural anchor; for each view, a Gaussian-kernel graph Laplacian is built with an adaptive bandwidth (median heuristic, re-estimated every epoch), and a consistency score (Laplacian trace alignment) measures how well the view's manifold aligns with the global anchor; after cross-view normalization, the resulting weights reweight the late-stage fusion of the adaptively encoded view features, automatically down-weighting unreliable views (`GlobalLocalManifoldCalibration.py`; the score form and bandwidth setting are switchable for the ablation study, and unsupported MPS ops fall back to CPU automatically).

<p align="center">
  <img src="docs/MSRCV1_acc.png" alt="Clustering accuracy on MSRCV1" width="70%">
</p>

<p align="center">
Clustering accuracy (ACC) on MSRCV1 during training.
</p>

**③ GLDA — Global-local Distribution Alignment** aligns the global fused representation with each view's local shared information: both the fused representation and the per-view common information are L2-normalized before computing pairwise similarities, and a contrastive loss (temperature 1) pulls them together — the normalization keeps the temperature meaningful regardless of feature scales; reconstruction and cycle-consistency terms preserve view-specific fidelity (`loss.py` + the consistency training stage of `train.py`). The global representation is finally clustered by K-means (n_init=100) into ACC / NMI / PUR / ARI.

## 6. 💻 User Guide (Windows / Linux / macOS)

### ⚙️ Requirements

| Library | Version | Recommended | Purpose |
|---|---|---|---|
| python | 3.9.25 | 3.9–3.12 | Runtime environment |
| pytorch | 2.7.1+cu128 | 2.7+ (cu121 / cpu / MPS official wheels by platform) | Network construction & two-stage training (AVE pretrain → consistency training) |
| numpy | 2.5.1 | ≥ 1.21 | Numerical computation (data & metrics) |
| scipy | 1.18.0 | ≥ 1.10 | Load .mat datasets |
| scikit-learn | 1.7.2 | ≥ 1.0 | K-means evaluation: ACC / NMI / PUR / ARI |

```shell
pip install numpy scipy scikit-learn
```

- PyTorch is installed separately per platform (see below); the GPU K-means alternatives (cuML/cuPy) are optional and **not** required by default.

### Install PyTorch per platform

| Platform | Install command | Default device |
|---|---|---|
| Linux / Windows + NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu121` | CUDA |
| Linux / Windows (CPU only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | CPU |
| macOS (Apple Silicon) | `pip install torch` (official wheels include MPS support) | MPS |

### Device selection (automatic, no configuration needed)

- Decision rule: **CUDA > MPS > CPU** (`utils/device_check.py`, probed automatically at startup).
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
