<div align="center">

<h2>Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration</h2>

[Ruimeng Liu](https://github.com/cleste-pome)<sup>1</sup>, [Chang Tang](https://github.com/changtang)<sup>1</sup>, [Bo Wang]()<sup>1</sup>, [Zhenglai Li](https://github.com/guanyuezhen)<sup>2</sup>, [Lianbo Guo]()<sup>1</sup>, [Xinwang Liu](https://github.com/xinwangliu)<sup>3</sup>

<sup>1</sup>[Huazhong University of Science and Technology](https://www.hust.edu.cn/), <sup>2</sup>[Shenzhen Institutes of Advanced Technology](https://www.siat.ac.cn/), <sup>3</sup>[National University of Defense Technology](https://www.nudt.edu.cn/)

</div>

***

Welcome to the official implementation of **MASA** — a robust multi-view clustering framework that handles cross-view sparsity heterogeneity via adaptive sparse encoding (AVE), and calibrates view-quality imbalance in late fusion through early-to-late manifold consistency (ELMC).

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
- [🧩 Method Overview: Three Core Modules](#-method-overview-three-core-modules)
- [🔗 Citation](#-citation)
- [📂 Source code list](#-source-code-list)
- [1. 📊 Dataset](#1-dataset)
- [2. ✅ Run](#2-run)
- [3. 🧮 Main Code](#3-main-code)
- [4. 🔬 Loss](#4-loss)
- [7. 💻 User Guide](#7--user-guide-windows--linux--macos)

### 🧩 Method Overview: Three Core Modules

MASA is a robust multi-view clustering framework built on three core modules, trained in **two stages**: ① *AVE pretraining* (reconstruction + adaptive sparsity), followed by ② *consistency training* (ELMC weighting + GLDA alignment). The aligned global representation is finally clustered by K-means into **ACC / NMI / PUR / ARI**.

#### ① AVE — Adaptive View-specific Encoding

Cross-view structural heterogeneity is common in multi-view data, with **sparsity variation** as a typical manifestation. AVE makes each view's encoder aware of its own sparsity:

1. **Sparsity probing**: for view $v$, the sparsity ratio is estimated per sample as the proportion of dimensions below a numerical threshold,
   $$s_v = \frac{1}{N}\sum_{n=1}^{N}\frac{|\{d : |x_{n,d}^v| < \varepsilon_{norm}\}|}{D_v}, \quad \varepsilon_{norm} = 6.1\times10^{-5}\ \text{(FP16 smallest normal)}$$
   (`zero_value_proportion` in `MASA.py`).
2. **Adaptive sparse constraint**: the observed $s_v$ modulates the KL-sparsity strength via an adaptive coefficient,
   $$C_{spa} = \begin{cases} 0, & s_v \le 0.01 \\ (s_v - 0.01)/(1 - 0.01), & s_v > 0.01 \end{cases} \in (0,1]$$
   sparser views thus receive stronger sparse regularization (the *sparse-at* positions and the KL term are applied to the encoder activations).
3. **View-aware loss**: each view (plus a global concatenated view) is reconstructed by its own autoencoder,
   $$\mathcal{L}_{rec} = \|\mathbf{x}_v - \hat{\mathbf{x}}_v\|_F^2 + \beta \cdot KL(\rho \,\|\, \hat{\rho}_v), \quad \rho = 0.05,\ \beta = 1.0$$

#### ② ELMC — Early-to-late Manifold Consistency Calibration

View-quality imbalance is the broader fusion problem: low-quality views introduce unreliable structural relations. ELMC anchors on the **early-fused global manifold** and calibrates the late fusion with a five-step chain (`GlobalLocalManifoldCalibration.py`):

| Step | Operation | Formula |
|---|---|---|
| 1 | Pairwise Euclidean distances of view / global features | $\Delta^v$, $\Delta^G$ |
| 2 | Gaussian kernel with **adaptive bandwidth** (median heuristic, re-estimated every epoch) | $W_{ab} = \exp(-\Delta_{ab}^2 / 2\sigma^2),\quad \sigma = \mathrm{median}(\Delta^G_{a>b})$ |
| 3 | Degree matrix and graph Laplacian | $L_v = D_v - W_v,\quad L_G = D_G - W_G$ |
| 4 | **Consistency score** (manifold alignment with the global anchor) | $S_v = \mathrm{Tr}(L_v L_G)$ |
| 5 | Cross-view normalization → late-fusion weights | $w_v = S_v / \sum_u S_u$ |

The normalized weights reweight the late-stage fusion $H = f_{fusion}\big(\sum_v w_v \cdot z_v\big)$: views whose manifolds align with the global structure dominate, while unreliable views are automatically down-weighted. Both the score form ($S_v$) and the bandwidth setting ($\sigma$) are switchable for the ablation study (`SCORE_FORM` / `SIGMA_MODE` at the top of the file), and unsupported MPS ops (`torch.cdist`/`diag`) fall back to CPU automatically.

<p align="center">
  <img src="docs/MSRCV1_acc.png" alt="Clustering accuracy on MSRCV1" width="70%">
</p>

<p align="center">
Clustering accuracy (ACC) on MSRCV1 during training.
</p>

#### ③ GLDA — Global-local Distribution Alignment

The final stage aligns the **global fused representation** with each view's **local shared information**:

1. **Common-information projection**: each view's encoding $z_v$ is projected to a shared space, $r_v = \mathrm{Proj}(z_v)$.
2. **Contrastive alignment**: a contrastive loss (temperature $\tau = 1$) pulls the global representation $H$ and each view's common information $r_v$ together,
   $$\mathcal{L}_{con} = \sum_{v\in\mathcal{V}} \ell_{con}(H,\ r_v)$$
   aligning the global and local distributions across views.
3. **Overall objective** (consistency training stage):
   $$\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_{sparse} + \alpha\, \mathcal{L}_{con}, \quad \alpha = 1$$
   reconstruction and cycle-consistency terms preserve view-specific fidelity throughout (`loss.py` + `contrastive_train` in `train.py`).
4. **Final clustering**: K-means (n_init=100) on the L2-normalized global representation $H$ yields ACC / NMI / PUR / ARI.

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

### 3.3 Dataset Preprocessing (Robustness Perturbations)

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

Additionally, the **ELMC** module computes per-view weights by Laplacian trace alignment
`w_v = Tr(L_v · L_G) / Σ_u Tr(L_u · L_G)` with an adaptive bandwidth σ (default: global
pairwise-distance median). The score form and σ setting can be switched at the top of
`GlobalLocalManifoldCalibration.py` (`SCORE_FORM` / `SIGMA_MODE`), corresponding to the
ablation tables in the paper.

## 7. 💻 User Guide (Windows / Linux / macOS)

### Dependencies

```shell
pip install numpy scipy scikit-learn tqdm matplotlib tabulate
```

- Python 3.10+ recommended (developed on 3.12).
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
