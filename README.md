<p align="center">
  <b>English</b> | <a href="docs/README_zh.md">中文</a>
</p>

<div align="center">

<h2>Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration</h2>

[Ruimeng Liu](https://github.com/cleste-pome)<sup>1</sup>, [Chang Tang](https://github.com/changtang)<sup>1</sup>, [Bo Wang]()<sup>1</sup>, [Zhenglai Li](https://github.com/guanyuezhen)<sup>2</sup>, [Lianbo Guo]()<sup>1</sup>, [Xinwang Liu](https://github.com/xinwangliu)<sup>3</sup>

<sup>1</sup>[Huazhong University of Science and Technology](https://www.hust.edu.cn/), <sup>2</sup>[Shenzhen Institutes of Advanced Technology](https://www.siat.ac.cn/), <sup>3</sup>[National University of Defense Technology](https://www.nudt.edu.cn/)

</div>

***

Welcome to the official implementation of **MASA**, a multi-view clustering (MVC) framework that handles cross-view sparsity heterogeneity via adaptive view-specific encoding (AVE), calibrates the late-stage fusion weights through early-to-late manifold consistency (ELMC), and aligns the global fused representation with each view's common information via contrastive learning (GLDA). ELMC measures how closely each view's local manifold agrees with the early-fused global manifold. The pipeline is agnostic to the number of views: two or more, fusion proceeds directly.

<details open><summary>📣 I also have other multi-view clustering projects that may interest you ✨.</summary><p>

> [**SparseMVC: Probing Cross-view Sparsity Variations for Multi-view Clustering**](https://openreview.net/pdf?id=cvJvk6oYfC)<br>
> Ruimeng Liu, Xin Zou, Chang Tang, Xiao Zheng, Xingchen Hu, Kun Sun, Xinwang Liu<br>
> [![Conference](http://img.shields.io/badge/NeurIPS-2025-f9f107.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/117045) [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/cleste-pome/SparseMVC)

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
- [1. ✅ Run](#1--run)
  - [1.1 Source code list](#11-source-code-list)
- [2. 🧮 Main Code](#2--main-code)
  - [2.1 Configuration](#21-configuration)
  - [2.2 Hyperparameters](#22-hyperparameters)
  - [2.3 Dataset Preprocessing](#23-dataset-preprocessing)
  - [2.4 Outputs](#24-outputs)
- [3. 🔬 Loss](#3--loss)
- [4. 🧩 Method Overview](#4--method-overview)
- [5. 📊 Dataset](#5--dataset)
  - [5.1 Multi-view Data and the .mat Format](#51-multi-view-data-and-the-mat-format)
  - [5.2 Creating Your Own Dataset](#52-creating-your-own-dataset)
  - [5.3 Evaluation Metrics](#53-evaluation-metrics)
- [6. 💻 User Guide](#6--user-guide-windows--linux--macos)
  - [6.1 Requirements](#61-requirements)
  - [6.2 Install PyTorch per platform](#62-install-pytorch-per-platform)
  - [6.3 Device selection (automatic, no configuration needed)](#63-device-selection-automatic-no-configuration-needed)
  - [6.4 Platform notes](#64-platform-notes)
- [7. 🙏 Acknowledgments](#7--acknowledgments)

---
### 🔗 Citation
If this work or the code is helpful to you, please cite it when it is available😊:
```
@article{liu2026masa,
  author={Ruimeng, Liu and Chang, Tang and Bo, Wang and Zhenglai, Li and Lianbo, Guo and Xinwang, Liu},
  title={Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration},
  year={2026}
}
```

---

This code is my accumulated work on multi-view research throughout my graduate studies, and it would not exist without the help I received along the way: thanks to my lab's supportive atmosphere and the open-source community's generous spirit, I stand on the shoulders of giants. When I first touched MVC, I kept wishing for a complete framework I could just pick up, test, and improve. Now that I have made it through the hardest stretch of the learning curve, I want to help the you of today, who was also the me of yesterday (I hit the same walls with evaluation metrics and multi-view datasets early on; to veterans, a few of the notes below may look over-explained, but they are written for newcomers, so bear with me.). The method may be nothing special, and my writing has much room to improve; a good, impactful paper is a goal I am still working toward (´･ω･`). But for a journal this good, finishing the paper is not the end; the follow-up work, like the open-source framework, deserves the same care. This repository is my answer, a small contribution ( •̀ ω •́ )✧. I want it clean, clear, and a little inspiring. If you also work on multi-view clustering or unsupervised representation learning, I hope it saves you some time. Thank you for reading my paper and using my code; it is my honor. If it helps your research even a little, I will be very happy (●'◡'●).

这套代码是我研究生阶段多视图研究的积累，能完成离不开一路上大家的帮助：感谢实验室互相帮助的氛围，也感谢开源社区慷慨分享的风气，让我有幸站在了巨人的肩膀上。刚接触 MVC 时我就在想，要是有一个完整的框架能直接拿来测试、改进该多好；现在熬过了最艰难的那段学习路，我也想去帮帮现在的你，其实也是曾经的我（比如评价指标、多视图数据集这些方面，我当初也是一路踩坑、各种困惑；对于过来人，我下面这些可能唠叨了些，但初衷只为让初学者理解得更快速，还请见谅。）。我的方法也许并不算出色，我的写作也还有不少需要改进的地方，写一篇“好”的影响力论文一直是我努力的目标(´･ω･`)。但既然投稿给了这么好的期刊，写完论文绝不是终点，后续工作比如开源框架，同样值得用心做好。这个仓库就是我的答案，一点微小的贡献 ( •̀ ω •́ )✧我希望它干净、清晰，能给你一些启发。如果你也在做多视图聚类或者无监督表示学习，希望它能帮你省下一些时间。感谢你阅读我的论文，使用我的代码，这是我的荣幸。哪怕它对你的研究只有一点点帮助，我也会非常开心 (●'◡'●)。

## 1. ✅ Run

(1) To run the **training** (two-stage: AVE pretraining → consistency training), use:

```shell
python train.py
```

- The terminal output is organized into labeled sections: `[Device]` (probed once at startup), then per dataset `[Data]` (dataset info), `[Hyperparams]` (config, once per round), `[Network]` (module structure with parameter counts and shares), and `[Train]` stage banners.
- Each training stage shows a light-blue progress bar that refreshes in place; below it, every epoch reports the loss with its components (pre: `global_ae + view_ae`; con: `global_ae + view_ae + contrastive`), the probed sparsity ratios, and the ELMC view weights.
- When a round finishes, one-line notices point to where the log, curves, and metrics were saved. With `--iter > 1`, a final summary table lists every round's best metrics plus their mean and standard deviation.

(2) To run the **evaluation** with a trained model:

```shell
python test.py --model 4.models --datasets ALOI-100
```

Alternatively, edit the `MODEL_PATH` / `DATASETS` variables at the top of `test.py`, or leave them empty for interactive input (weight-path priority: `--model` > `MODEL_PATH` > interactive input).

### 1.1 Source code list

```shell
MASA
├── train.py                          # Training entry (two-stage: AVE pretrain → consistency training)
├── test.py                           # Evaluation (load .pth + dataset → forward → K-means)
├── MASA.py                           # Model definition (Network: Encoder/Decoder, projection head, weighted fusion)
├── loss.py                           # Loss functions (contrastive + reconstruction/KL sparsity)
├── datasets                          # Dataset directory (.mat: X view-cell array + Y labels)
├── docs                               # Figures used in this README
└── utils                             # Utilities
    ├── count_datasetY.py             # Class distribution statistics & plots
    ├── dataloader.py                 # Data loading & preprocessing (min-max norm + noise/conflict/missing/sparsity injection)
    ├── device_check.py               # Device probing (CUDA > MPS > CPU, MASA_DEVICE override)
    ├── GlobalLocalManifoldCalibration.py # ELMC core (Laplacian trace-alignment view weights, adaptive σ, switchable ablation configs)
    ├── Logger.py                     # Logging
    ├── metric.py                     # Metrics (ACC/NMI/PUR/ARI, K-means evaluation)
    ├── metric2csv.py                 # Metric CSV export
    ├── plot.py                       # Training curves & ELMC σ curve
    ├── scripts.py                    # Shared utilities (seed/timing/formatting, model summary, progress bar format)
    └── tsne_visual.py                # t-SNE visualization (pdf+svg)
```

---

## 2. 🧮 Main Code

### 2.1 Configuration

The dataset folder and output directories are handled automatically; the device is selected by `utils/device_check.py`.

```py
# Dataset folder path (all .mat files under it are trained in turn)
folder_path = "datasets"
# Output directories are created at runtime:
# 1.logs/ 2.results_imgs/ 3.csv/(Metrics, ViewWeights) 4.models/ 5.tsne/
```

### 2.2 Hyperparameters

```py
# Dataset name (defaults to the current .mat file; keep the default, passing it manually desyncs log/model filenames)
parser.add_argument('--dataset', default=Dataname)
# Batch size (the runtime forces the full dataset size; NUSWIDEOBJ keeps 256)
parser.add_argument('--batch_size', default=256, type=int)
# Learning rate
parser.add_argument("--learning_rate", type=float, default=0.0003)
# Number of epochs for the AVE pretraining stage
parser.add_argument("--pre_epochs", type=int, default=300)  # 300
# Number of epochs for the consistency training stage (ELMC + GLDA)
parser.add_argument("--con_epochs", type=int, default=300)  # 300/600
# Feature dimensions: larger for encoding (richer representations), smaller for the contrastive projection (confines the loss to a compact subspace, shielding the encoder from dimensional collapse).
parser.add_argument("--feature_dim", type=int, default=64)       # view-specific features
parser.add_argument("--high_feature_dim", type=int, default=20)  # compressed feature dimension
# Random seed: anchors the random number generators of torch/numpy/python, so the stochastic components of training (weight initialization, data shuffling, dropout, K-means initialization) behave consistently for the same experimental configuration
parser.add_argument("--seed", type=int, default=42)
# Number of runs; --iter > 1 perturbs the seed/lr deterministically each round
parser.add_argument("--iter", type=int, default=1)
# Weight decay (L2 penalty on the network weights, passed to the Adam optimizer to prevent overfitting; 0.0 disables it)
parser.add_argument("--weight_decay", type=float, default=0.0)
```

### 2.3 Dataset Preprocessing

```py
# Select samples by noise ratio, then add Gaussian noise to (1..view-1) random views.
parser.add_argument('--noise_ratio', type=float, default=0.0)
# Select samples by conflict ratio, then replace one view's data with the same-view data of a sample from another class.
parser.add_argument('--conflict_ratio', type=float, default=0.0)
# Select samples by missing ratio, then set (1..view-1) random views to zero.
parser.add_argument('--missing_ratio', type=float, default=0.0)
# Select dimensions by sparsity ratio, then set random dimensions to zero.
parser.add_argument('--sparsity_ratio', type=float, default=0.0)
```

### 2.4 Outputs

All output directories are created automatically. All files of one dataset run share the same timestamp in their names (`{dataset}_{timestamp}`), so results from the same run are easy to pair.

| Directory | Contents | Naming |
|---|---|---|
| `1.logs/` | Training logs per dataset (also holds the class-distribution report figure) | `{dataset}/{dataset}_{timestamp}_{ratio}.log` |
| `2.results_imgs/` | Training curves (per-component loss curves: `global_ae` / `view_ae` / `contrastive`, and ACC / NMI / PUR / ARI) and the ELMC σ curve | `{dataset}_{timestamp}/{dataset}_ep{epochs}_{name}.png` |
| `3.csv/Metrics/` | Per-epoch metrics: summary and per-view + global | `{dataset}_{timestamp}_{ratio}.csv`, `view_{dataset}_{timestamp}_{ratio}.csv` |
| `3.csv/ViewWeights/` | Per-epoch ELMC view weights | `{dataset}_{timestamp}.csv` |
| `4.models/` | Trained model weights (.pth) | `{dataset}/{dataset}_{timestamp}.pth` |
| `5.tsne/` | t-SNE visualizations of features (PDF + SVG) | `{dataset}_{timestamp}/{epoch}_{dataset}[_EarlyFusion].pdf` |

Where `{dataset}` is the dataset name, `{timestamp}` is the run time (one per dataset, shared by all its outputs), `{ratio}` is the perturbation setting `{noise}_{conflict}_{missing}`, and `{name}` is the metric name (`acc` / `nmi` / `pur` / `ari`).

---

## 3. 🔬 Loss

The training objective is assembled per stage, and each `ae_loss_function` already contains the reconstruction (MSE) plus the adaptive KL sparsity term (see `loss.py`), so the terms below are the actual components summed into the per-batch loss `loss = sum(loss_list)` used for backpropagation.

**Pretraining stage (AVE):** the loss is the sum of

```py
# global AE: reconstruction + sparsity for the concatenated (early-fused) input
ae_loss_function(mean_average, xs2one, xr_all, activation[0], rho=0.05, beta=1.0)
# view AE: reconstruction + sparsity for each view (adaptive C_spa modulated by its sparsity ratio)
for v in range(view):
    ae_loss_function(means[v], xs[v], xrs[v], activation[v + 1], rho=0.05, beta=1.0)
```

**Consistency training stage (ELMC + GLDA):** the loss additionally includes, for each view, the contrastive alignment between the global fused representation `H` and that view's common information (GLDA):

```py
loss_list.append(
    ae_loss_function(mean_average, xs2one, xr_all, activation[0], rho=0.05, beta=1.0))       # global AE
for v in range(view):
    loss_list.append(ae_loss_function(means[v], xs[v], xrs[v], activation[v + 1], ...))     # view AE
    loss_list.append(contrastiveloss(H, rs[v], w2[v]))                                       # GLDA contrastive
loss = sum(loss_list)
```

The ELMC module computes per-view fusion weights by Laplacian trace alignment during the consistency stage; the score form and bandwidth setting can be switched at the top of `utils/GlobalLocalManifoldCalibration.py` (`SCORE_FORM` / `SIGMA_MODE`), corresponding to the ablation tables. Based on experimental results, when the data scale is larger or the training epochs are more, the kernel bandwidth can be adjusted to a fixed value to balance the training cost.

The overall objective combines two terms, balanced by a constraint ratio coefficient: the **AVE loss**, summed over all views, of the reconstruction error plus the adaptive entropy-based sparsity penalty (whose strength is modulated by each view's probed sparsity ratio); and the **GLDA loss**, the contrastive alignment between the global fused representation and each view's common information, averaged over views and samples.

---

## 4. 🧩 Method Overview

As the simplest and most intuitive self-supervised task, clustering offers a direct check on the quality of the extracted and fused features: the cleaner the clusters, the more discriminative the learned representation.

MASA is a robust multi-view clustering framework built on three core modules and trained in **two stages**: first *AVE pretraining* (reconstruction with adaptive sparsity), then *consistency training* (ELMC weighting + GLDA alignment). The aligned global representation is finally clustered by K-means into ACC / NMI / PUR / ARI.

**① AVE: Adaptive View-specific Encoding** handles the cross-view sparsity heterogeneity typical of multi-view data: the sparsity ratio of each view is probed from its input (`zero_value_proportion` in `MASA.py`) and used as prior knowledge to adaptively modulate the strength of the entropy-based sparse constraint; sparser views receive stronger sparse regularization, so that each view's encoder is tuned in a view-aware manner (adaptive sparse coefficient in `ae_loss_function`, `loss.py`).

**② ELMC: Early-to-late Manifold Consistency Calibration** quantifies the geometric agreement between each view and the early-fused global representation: a Gaussian-kernel graph Laplacian per view, re-estimated every epoch, is aligned with the global Laplacian into a consistency score; normalized across views, the scores become fusion weights that let the manifold structure of early fusion guide the late-stage fusion and down-weight unreliable views (`utils/GlobalLocalManifoldCalibration.py`; score form and bandwidth setting are switchable for the ablation study; MPS-unsupported ops fall back to CPU automatically).

<p align="center">
  <img src="docs/MSRCV1_acc.png" alt="Clustering accuracy on MSRCV1" width="90%">
</p>

<p align="center">
Clustering accuracy (ACC) on MSRCV1 during training.
</p>

**③ GLDA: Global-local Distribution Alignment** aligns the global fused representation with each view's local shared information: both the fused representation and the per-view common information are L2-normalized before computing pairwise similarities, and a contrastive loss (temperature 1) pulls them together; the normalization keeps the temperature meaningful regardless of feature scales; reconstruction terms preserve view-specific fidelity (`loss.py` + the consistency training stage of `train.py`).

## 5. 📊 Dataset

### 5.1 Multi-view Data and the .mat Format

Multi-view clustering data describes the same set of samples from several complementary views, e.g. different feature extractors, image and text modalities, or gene expression profiles, where each view is one feature matrix. Good multi-view datasets provide views that are informative on their own and complementary to each other.

In this repo, each dataset is a single `.mat` file placed under `datasets/`, containing:
- `X`: a cell array of view matrices; `X{1}, X{2}, ...` are the feature matrices of views 1, 2, ..., each of shape `(num_samples, num_dimensions)`;
- `Y`: a column vector of sample labels, of shape `(num_samples, 1)`.

The loading logic lives in `utils/dataloader.py`:

```python
# MATKind: read the .mat file
data = scipy.io.loadmat(f"{path}/{dataset_name}.mat")
data_X = data['X'][0]          # view matrices of ALL views (cell array)
data_Y = data['Y']             # labels

# MultiViewDataset: the view count is read from the file, never hard-coded
self.num_views = data_X.shape[0]           # 2, 6, 10 views: the same code works
for v in range(self.num_views):
    self.X[v] = self.normalize(data_X[v])  # each view is normalized independently
self.Y = np.squeeze(self.Y)                # labels -> 1-D; if they start at 1, shifted to start at 0
self.num_classes = len(np.unique(self.Y))  # class count derived automatically
```

The number of views is never hard-coded; it is read from the file, so the same code handles datasets with 2, 6, or 10 views without any modification (e.g. `MSRCV1.mat` has 6 views with 1302/48/512/100/256/210 dimensions). Views may also differ in dimensionality, since each view is normalized on its own, and the class count is derived automatically from the labels. All views are min-max normalized to the unit range independently before training.

After loading, `train.py` calls `count_classes(Dataname, dataset.Y)` (`utils/count_datasetY.py`): it tallies the sample count of every class with `np.unique(Y, return_counts=True)` and draws a single class-distribution report (class sizes, long-tail analysis, and related statistics), saved as `1.logs/{dataset_name}/{dataset_name}_ClassReport.png` (300 dpi). It is useful for spotting imbalanced datasets before training.

### 5.2 Creating Your Own Dataset

1. Prepare one feature matrix per view, shape `(num_samples, num_dimensions)`; different views may have different dimensions, but every view must have the same number of rows (one per sample) and the sample order must match across views.
2. Pack the views into `X` as a MATLAB cell array of shape `(1, num_views)` (each element one view matrix), and put the integer labels into `Y` as a `(num_samples, 1)` column vector.
3. Save both into a single `.mat` file (e.g. with `scipy.io.savemat`) and drop it into `datasets/`; `train.py` picks it up automatically, no registration needed. You can inspect a bundled dataset (e.g. `datasets/MSRCV1.mat`) with `scipy.io.loadmat` to see the exact layout.

Common public multi-view datasets can be found at: https://github.com/wangsiwei2010/awesome-multi-view-clustering

---


### 5.3 Evaluation Metrics

The quality of the learned representation is measured by four standard clustering metrics: **ACC** (Accuracy): the proportion of samples correctly matched to the ground-truth labels after optimal label alignment; **NMI** (Normalized Mutual Information): the normalized mutual information between the clustering and the ground-truth partition; **PUR** (Purity): the proportion of samples assigned to their dominant class; and **ARI** (Adjusted Rand Index): the similarity between two clusterings corrected for chance. They are computed on the K-means clustering (n_init=100) of the final global representation, with higher values indicating better clustering quality.

The four metrics are implemented in `utils/metric.py`:

```python
# utils/metric.py: evaluate() computes all four metrics at once
def evaluate(y_true, y_pred):
    return {
        "nmi": v_measure_score(y_true, y_pred),      # NMI: normalized mutual information
        "ari": adjusted_rand_score(y_true, y_pred),  # ARI: adjusted rand index
        "acc": cluster_acc(y_true, y_pred),          # ACC: accuracy after optimal label matching
        "purity": purity(y_true, y_pred),            # PUR: purity
    }
```

Here ACC (`cluster_acc`) matches the predicted cluster ids to the ground-truth labels with the Hungarian algorithm (`linear_sum_assignment`) before counting the correctly classified samples.

---

## 6. 💻 User Guide (Windows / Linux / macOS)

### 6.1 Requirements

| Library | Version | Recommended | Purpose |
|---|---|---|---|
| python | 3.9.25 | 3.9–3.12 | Runtime environment |
| pytorch | 2.7.1+cu128 | 2.7+ (cu121 / cpu / MPS official wheels by platform) | Network construction & two-stage training (AVE pretrain → consistency training) |
| numpy | 2.5.1 | ≥ 1.21 | Numerical computation (data & metrics) |
| scipy | 1.18.0 | ≥ 1.10 | Load .mat datasets |
| scikit-learn | 1.7.2 | ≥ 1.0 | K-means evaluation: ACC / NMI / PUR / ARI |

- PyTorch is installed separately per platform (see below); the GPU K-means alternatives (cuML/cuPy) are optional and **not** required by default.

### 6.2 Install PyTorch per platform

| Platform | Install command | Default device |
|---|---|---|
| Linux / Windows + NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu121` | CUDA |
| Linux / Windows (CPU only) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | CPU |
| macOS (Apple Silicon) | `pip install torch` (official wheels include MPS support) | MPS |

> **提示**：官方源在国内下载较慢，一般 Python 包可换用[清华 TUNA 镜像](https://pypi.tuna.tsinghua.edu.cn/simple)，PyTorch 可换用[阿里云镜像](https://mirrors.aliyun.com/pytorch-wheels/cu121)。

### 6.3 Device selection (automatic, no configuration needed)

- Decision rule: **CUDA > MPS > CPU** (`utils/device_check.py`, probed automatically at startup).
- Force a device with the environment variable `MASA_DEVICE=cuda|mps|cpu|auto` (unavailable or invalid values fall back to automatic):

```shell
MASA_DEVICE=cuda python train.py    # force CUDA (Linux / Windows)
MASA_DEVICE=mps  python train.py    # force MPS (macOS)
MASA_DEVICE=cpu  python train.py    # force CPU
```

- Standalone probe (prints the environment summary only, no training): `python utils/device_check.py`

### 6.4 Platform notes

- **macOS**: MPS requires macOS ≥ 12.3 and an official PyTorch build; unsupported ops under MPS (`torch.cdist` / `torch.diag` in the ELMC module) automatically fall back to CPU; nothing to configure.
- **Linux / Windows without CUDA**: falls back to CPU automatically. `OMP_NUM_THREADS=1` is preset in `train.py` to avoid thread oversubscription on CPU.
- **CUDA**: to choose a specific GPU, set `CUDA_VISIBLE_DEVICES` (train.py presets `"0"`); multi-GPU is not required.

---

## 7. 🤝 Acknowledgments

Our proposed MASA draws inspiration from the works of [SCMVC](https://github.com/SongwuJob/SCMVC), [RCML](https://github.com/jiajunsi/RCML), [DCG](https://github.com/zhangyuanyang21/2025-AAAI-DCG) and the [Awesome-Deep-Multi-View-Clustering](https://github.com/jinjiaqi1998/Awesome-Deep-Multi-View-Clustering) collection. We would like to thank the authors for their valuable contributions to the multi-view clustering community.

This codebase is built on the open-source ecosystem: [PyTorch](https://pytorch.org/) for deep learning, [scikit-learn](https://scikit-learn.org/) for clustering evaluation, [scipy](https://scipy.org/) for loading `.mat` datasets and scientific computing, and [matplotlib](https://matplotlib.org/) for visualization. We sincerely thank the developers of these projects for their free and excellent software. The public multi-view datasets used in the experiments were collected from the community, and we are grateful to the dataset creators for sharing them.
