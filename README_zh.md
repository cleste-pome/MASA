<p align="center"><a href="README.md">English</a> | <b>中文</b></p>

<div align="center">

<h2>Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration</h2>

[Ruimeng Liu](https://github.com/cleste-pome)<sup>1</sup>, [Chang Tang](https://github.com/changtang)<sup>1</sup>, [Bo Wang]()<sup>1</sup>, [Zhenglai Li](https://github.com/guanyuezhen)<sup>2</sup>, [Lianbo Guo]()<sup>1</sup>, [Xinwang Liu](https://github.com/xinwangliu)<sup>3</sup>

<sup>1</sup>[Huazhong University of Science and Technology](https://www.hust.edu.cn/), <sup>2</sup>[Shenzhen Institutes of Advanced Technology](https://www.siat.ac.cn/), <sup>3</sup>[National University of Defense Technology](https://www.nudt.edu.cn/)

</div>

***

这里是 **MASA** 的官方代码实现。MASA 是一个多视图聚类框架：它用自适应视图专属编码（AVE）处理跨视图的稀疏性差异，再用早期到晚期流形一致性校准（ELMC）校正后期融合权重。ELMC 衡量的，是每个视图的局部流形与早期融合全局流形之间的一致性。

作为一个中文母语者，做研究意味着要用一门不属于自己的语言来工作。英语对我来说从来都不轻松，我的写作也还有很多不足之处需要改进。写一篇好论文，也是我一直在努力的目标 (´･ω･`)。这套代码能够完成，离不开一路上大家给予我的帮助，这要感谢实验室里互相帮助的氛围，感谢开源社区慷慨分享的氛围，我有幸站在了巨人的肩膀上。当时我就一直想要是有一个好的框架可以直接拿来测试改进就好了，现在我也想能成为其中之一去帮助后来的你。我的方法本身也许并不出色，我觉得对于这么好的期刊来说，写完论文绝不是一件工作的结束，后续工作比如开源框架同样值得用心做好。这个仓库就是我给出的答案 ( •̀ ω •́ )✧我希望它是干净的清晰的、能给你启发的。如果你也在做多视图聚类或者无监督表示学习，希望它能帮你省下一些时间。感谢你阅读我的论文，使用我的代码，这是我的荣幸。哪怕它对你的研究只有一点点帮助，我也会非常开心 (●'◡'●)。

<details open><summary>📣 我还有其他多视图聚类项目，或许你也会感兴趣 ✨。</summary><p>

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
MASA 框架流程图。自适应视图专属编码（AVE）先探测每个视图的稀疏率，作为视图感知表示学习与约束调制的先验信息；早期到晚期流形一致性校准（ELMC）则利用早期融合特征中保持的稳定全局流形，对自适应编码局部特征的后期融合重新赋权。
</p>

### 📑 目录
- [🔗 引用](#引用)
- [1. ✅ 运行](#1--运行)
- [2. 🧮 主代码](#2--主代码)
- [3. 🔬 损失](#3--损失)
- [4. 🧩 方法概览](#4--方法概览)
- [5. 📊 数据集](#5--数据集)
- [6. 💻 使用指南（Windows / Linux / macOS）](#6--使用指南windows--linux--macos)
- [7. 🙏 致谢](#7--致谢)

---

### 🔗 引用
如果这项工作或代码对你有帮助，欢迎引用😊：
```
@article{liu2026masa,
  author={Ruimeng, Liu and Chang, Tang and Bo, Wang and Zhenglai, Li and Lianbo, Guo and Xinwang, Liu},
  title={Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration},
  year={2026}
}
```

---

## 1. ✅ 运行

(1) 运行**训练**（两阶段：AVE 预训练 → 一致性训练）：

```shell
python train.py
```

- 设备自动选择（**CUDA > MPS > CPU**）；可用环境变量 `MASA_DEVICE=cuda|mps|cpu|auto` 强制指定设备。
- 训练内置 K-means 评估；输出目录自动创建：`1.logs/`（日志）、`2.results_imgs/`（曲线）、`3.csv/`（指标 + 视图权重）、`4.models/`（.pth 权重）、`5.tsne/`（t-SNE）。
- 终端输出分为几个区块：`[Device]`（启动时探测一次），然后每个数据集依次打印 `[Data]`（数据集信息）、`[Hyperparams]`（配置，每轮一次）、`[Network]`（模块结构、参数量与占比）、`[Train]` 阶段横幅。
- 每个训练阶段都有一条淡蓝色进度条，原地刷新。进度条下方，每个 epoch 报告损失及其分量（pre：`global_ae + view_ae`；con：`global_ae + view_ae + contrastive`）、探测到的稀疏率与 ELMC 视图权重。
- 每轮结束时打印加权最优结果（`Max metric: epoch...`，含 ACC/NMI/PUR/ARI），随后一行提示日志、曲线与指标的保存位置。`--iter > 1` 时，最后再输出一张汇总表：每轮的最优指标，以及它们的均值与标准差。

(2) 用训练好的模型运行**评估**：

```shell
python test.py --model 4.models --datasets ALOI-100
```

也可以在 `test.py` 顶部填写 `MODEL_PATH` / `DATASETS` 变量，或留空交互输入（权重路径优先级：`--model` > `MODEL_PATH` > 交互输入）。

### 📂 源码列表：

```shell
MASA
├── train.py                          # 训练入口（两阶段：AVE 预训练 → 一致性训练）
├── test.py                           # 评估（加载 .pth + 数据集 → 前向 → K-means）
├── MASA.py                           # 模型定义（Network：编码器/解码器、投影头、加权融合）
├── loss.py                           # 损失函数（对比 + 重建/KL 稀疏）
├── datasets                          # 数据集目录（.mat：X 视图 cell 数组 + Y 标签）
├── docs                              # 本 README 使用的图片
└── utils                             # 工具模块
    ├── count_datasetY.py             # 类别分布统计与绘图
    ├── dataloader.py                 # 数据加载与预处理（min-max 归一化 + noise/conflict/missing/sparsity 注入）
    ├── device_check.py               # 设备探测（CUDA > MPS > CPU，MASA_DEVICE 覆盖）
    ├── GlobalLocalManifoldCalibration.py # ELMC 核心（拉普拉斯迹对齐视图权重、自适应 σ、可切换消融配置）
    ├── Logger.py                     # 日志
    ├── metric.py                     # 指标（ACC/NMI/PUR/ARI，K-means 评估）
    ├── metric2csv.py                 # 指标 CSV 导出
    ├── plot.py                       # 训练曲线与 ELMC σ 曲线
    ├── scripts.py                    # 共享工具（种子/计时/格式化、模型摘要、进度条格式）
    └── tsne_visual.py                # t-SNE 可视化（pdf+svg）
```

---

## 2. 🧮 主代码

### 2.1 配置

数据集文件夹与输出目录无需手动配置，设备由 `utils/device_check.py` 选择。

```py
# 数据集文件夹路径（其下所有 .mat 依次训练）
folder_path = "datasets"
# 输出目录运行时自动创建：
# 1.logs/ 2.results_imgs/ 3.csv/(Metrics, ViewWeights) 4.models/ 5.tsne/
```

### 2.2 超参数

```py
# 数据集名（默认为当前 .mat 文件；保持默认，手动传入会导致日志/模型文件名不同步）
parser.add_argument('--dataset', default=Dataname)
# 批大小（运行时强制为全量数据集；NUSWIDEOBJ 保持 256）
parser.add_argument('--batch_size', default=256, type=int)
# 学习率
parser.add_argument("--learning_rate", type=float, default=0.0003)
# AVE 预训练阶段的轮数
parser.add_argument("--pre_epochs", type=int, default=300)  # 300
# 一致性训练阶段的轮数（ELMC + GLDA）
parser.add_argument("--con_epochs", type=int, default=300)  # 300/600
# 特征维度：编码侧更大（表示更丰富），对比投影侧更小（把损失限制在紧致子空间中，防止编码器发生维度坍缩）。
parser.add_argument("--feature_dim", type=int, default=64)       # 视图专属特征
parser.add_argument("--high_feature_dim", type=int, default=20)  # 压缩特征维度
# 随机种子（固定值，运行可复现）
parser.add_argument("--seed", type=int, default=42)
# 运行轮数；--iter > 1 时每轮对 seed/lr 做确定性扰动
parser.add_argument("--iter", type=int, default=1)
# 权重衰减（对网络权重的 L2 惩罚，传给 Adam 优化器防止过拟合；0.0 关闭）
parser.add_argument("--weight_decay", type=float, default=0.0)
```

### 2.3 数据集预处理

```py
# 按 noise ratio 选取样本，随机选 (1..view-1) 个视图加入高斯噪声。
parser.add_argument('--noise_ratio', type=float, default=0.0)
# 按 conflict ratio 选取样本，用一个其他类别样本的同视图数据替换其一个视图。
parser.add_argument('--conflict_ratio', type=float, default=0.0)
# 按 missing ratio 选取样本，将 (1..view-1) 个随机视图置零。
parser.add_argument('--missing_ratio', type=float, default=0.0)
# 按 sparsity ratio 选取维度，将随机维度置零。
parser.add_argument('--sparsity_ratio', type=float, default=0.0)
```

### 2.4 输出

所有输出目录自动创建。同一数据集一次运行的所有文件共享同一时间戳（`{数据集}_{时间戳}`），同一次运行的结果易于对应。

| 目录 | 内容 | 命名 |
|---|---|---|
| `1.logs/` | 每个数据集的训练日志（也存放类别分布报告图） | `{dataset}/{dataset}_{timestamp}_{ratio}.log` |
| `2.results_imgs/` | 训练曲线（loss / ACC / NMI / PUR / ARI）与 ELMC σ 曲线 | `{dataset}_{timestamp}/{dataset}_ep{epochs}_{name}.png` |
| `3.csv/Metrics/` | 每轮指标：汇总与分视图 + 全局 | `{dataset}_{timestamp}_{ratio}.csv`、`view_{dataset}_{timestamp}_{ratio}.csv` |
| `3.csv/ViewWeights/` | 每轮 ELMC 视图权重 | `{dataset}_{timestamp}.csv` |
| `4.models/` | 训练好的模型权重（.pth） | `{dataset}/{dataset}_{timestamp}.pth` |
| `5.tsne/` | 特征 t-SNE 可视化（PDF + SVG） | `{dataset}_{timestamp}/{epoch}_{dataset}[_EarlyFusion].pdf` |

其中 `{dataset}` 是数据集名，`{timestamp}` 是运行时间（每个数据集一个，其所有输出共享），`{ratio}` 是扰动设置 `{noise}_{conflict}_{missing}`，`{name}` 是指标名（`acc` / `nmi` / `pur` / `ari`）。

---

## 3. 🔬 损失

总体目标整合三项：

```py
# 1. loss_rec：重建（MSE），约束每个视图的自编码器与全局表示
ae_loss_function(mean_average, xs2one, xr_all, activation[0], rho=0.05, beta=1.0)

# 2. loss_sparse：KL 稀疏，系数 C_spa 由每个视图的稀疏率自适应调制（AVE）
kl_sparse_loss(hidden_layer_activation, rho, sparse_beta)

# 3. loss_con：对比损失，将全局融合表示 H 与每个视图的公共信息对齐（GLDA）
contrastiveloss(H, rs[v], w2[v])
```

此外，**ELMC** 模块通过拉普拉斯迹对齐（自适应带宽，默认为全局两两距离中位数）计算每个视图的融合权重。分数形式与带宽设置可在 `utils/GlobalLocalManifoldCalibration.py` 顶部切换（`SCORE_FORM` / `SIGMA_MODE`），对应消融表。

总体目标由两项组成，通过约束比例系数平衡：**AVE 损失**，对所有视图求和，即重建误差加自适应熵基稀疏惩罚（惩罚强度由每个视图探测到的稀疏率调制）；**GLDA 损失**，即全局融合表示与每个视图公共信息之间的对比对齐，在视图与样本上取平均。

---

## 4. 🧩 方法概览

作为最简单、最直观的自监督任务，聚类可以直接检验所提取与融合特征的质量：簇越干净，说明学到的表示越有判别力。

MASA 是一个鲁棒的多视图聚类框架，基于三个核心模块，分**两个阶段**训练：先进行 *AVE 预训练*（带自适应稀疏的重建），再进行一致性训练（ELMC 加权 + GLDA 对齐）。对齐后的全局表示最终用 K-means 聚类，得到 ACC / NMI / PUR / ARI。

**① AVE — 自适应视图专属编码** 处理多视图数据中常见的跨视图稀疏性差异：先从输入中探测每个视图的稀疏率（`MASA.py` 中的 `zero_value_proportion`），再把它当作先验信息，自适应地调制熵基稀疏约束的强度，越稀疏的视图获得越强的稀疏正则。这样，每个视图的编码器都能按自己的稀疏程度来调整（`loss.py` 中 `ae_loss_function` 的自适应稀疏系数）。

**② ELMC — 早期到晚期流形一致性校准** 量化每个视图与早期融合全局表示之间的几何一致性：先为每个视图构建高斯核图拉普拉斯（带宽自适应，采用中位数启发式，每轮重新估计），再与全局拉普拉斯做对齐，得到一致性分数。跨视图归一化后，这些分数就成为融合权重：早期融合的流形结构借此指导后期融合，并压低不可靠视图的权重（`utils/GlobalLocalManifoldCalibration.py`；分数形式与带宽设置可切换，用于消融研究；MPS 不支持的算子自动回退 CPU）。

<p align="center">
  <img src="docs/MSRCV1_acc.png" alt="MSRCV1 聚类准确率" width="90%">
</p>

<p align="center">
MSRCV1 训练过程中的聚类准确率（ACC）。
</p>

**③ GLDA — 全局-局部分布对齐** 把全局融合表示与每个视图的局部共享信息对齐：融合表示与每个视图的公共信息都先做 L2 归一化，再计算两两相似度，对比损失（温度 1）把它们拉近。归一化保证了温度在不同特征尺度下都有意义；重建项则保留视图专属的保真度（`loss.py` + `train.py` 的一致性训练阶段）。

**评估指标。** 所学表示的质量由四个标准聚类指标衡量：**ACC**（Accuracy，准确率），与真实标签最优匹配后样本被正确归类的比例；**NMI**（Normalized Mutual Information，归一化互信息），聚类结果与真实划分之间的归一化互信息；**PUR**（Purity，纯度），样本被分到其主导类别的比例；**ARI**（Adjusted Rand Index，调整兰德指数），消除随机因素后两种聚类之间的相似度。它们都基于最终全局表示的 K-means 聚类（n_init=100）计算，值越高表示聚类质量越好。

---

## 5. 📊 数据集

多视图聚类数据用若干个互补的视图描述同一组样本，例如不同的特征提取器、图像与文本模态，或者基因表达谱。每个视图都是一个特征矩阵。好的多视图数据集，其各个视图本身就有信息量，且彼此互补。

在本仓库中，每个数据集是一个放在 `datasets/` 下的单个 `.mat` 文件，包含：
- `X`：视图矩阵的 cell 数组，`X{1}, X{2}, ...` 是视图 1, 2, ... 的特征矩阵，每个形状为 `(样本数, 维度数)`；
- `Y`：样本标签列向量，形状为 `(样本数, 1)`。

要使用你自己的数据，把视图放进 cell 数组 `X`，标签放进 `Y`，保存为 `.mat` 文件（例如通过 `scipy.io.savemat`），并把文件放入 `datasets/`。`train.py` 会自动读取并训练文件夹中每个 `.mat` 文件。常用公开多视图数据集见：https://github.com/wangsiwei2010/awesome-multi-view-clustering

---

## 6. 💻 使用指南（Windows / Linux / macOS）

### ⚙️ 环境要求

| 库 | 版本 | 推荐 | 用途 |
|---|---|---|---|
| python | 3.9.25 | 3.9–3.12 | 运行环境 |
| pytorch | 2.7.1+cu128 | 2.7+（按平台 cu121 / cpu / MPS 官方 wheel） | 网络构建与两阶段训练（AVE 预训练 → 一致性训练） |
| numpy | 2.5.1 | ≥ 1.21 | 数值计算（数据与指标） |
| scipy | 1.18.0 | ≥ 1.10 | 加载 .mat 数据集 |
| scikit-learn | 1.7.2 | ≥ 1.0 | K-means 评估：ACC / NMI / PUR / ARI |

- PyTorch 按平台单独安装（见下文）；GPU K-means 备选（cuML/cuPy）是可选的，默认不需要。

### 按平台安装 PyTorch

| 平台 | 安装命令 | 默认设备 |
|---|---|---|
| Linux / Windows + NVIDIA GPU | `pip install torch --index-url https://download.pytorch.org/whl/cu121` | CUDA |
| Linux / Windows（仅 CPU） | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | CPU |
| macOS（Apple Silicon） | `pip install torch`（官方 wheel 含 MPS 支持） | MPS |

> 官方源在国内下载较慢，一般 Python 包可换用[清华 TUNA 镜像](https://pypi.tuna.tsinghua.edu.cn/simple)，PyTorch 可换用[阿里云镜像](https://mirrors.aliyun.com/pytorch-wheels/cu121)。

### 设备选择（自动，无需配置）

- 决策规则：**CUDA > MPS > CPU**（`utils/device_check.py`，启动时自动探测）。
- 用环境变量强制指定设备 `MASA_DEVICE=cuda|mps|cpu|auto`（不可用或非法值自动回退）：

```shell
MASA_DEVICE=cuda python train.py    # 强制 CUDA（Linux / Windows）
MASA_DEVICE=mps  python train.py    # 强制 MPS（macOS）
MASA_DEVICE=cpu  python train.py    # 强制 CPU
```

- 独立探测（只打印环境摘要，不训练）：`python utils/device_check.py`

### 平台说明

- **macOS**：MPS 需要 macOS ≥ 12.3 与官方 PyTorch 构建；MPS 下不支持的算子（ELMC 模块中的 `torch.cdist` / `torch.diag`）会自动回退 CPU，无需任何配置。
- **Linux / Windows 无 CUDA**：自动回退 CPU。`train.py` 中预设 `OMP_NUM_THREADS=1`，避免 CPU 上的线程过载。
- **CUDA**：选择特定 GPU 请设置 `CUDA_VISIBLE_DEVICES`（train.py 预设 `"0"`）；不需要多 GPU。

---

## 7. 🙏 致谢

我们提出的 MASA 受 [SCMVC](https://github.com/SongwuJob/SCMVC)、[RCML](https://github.com/jiajunsi/RCML)、[DCG](https://github.com/zhangyuanyang21/2025-AAAI-DCG) 与 [Awesome-Deep-Multi-View-Clustering](https://github.com/jinjiaqi1998/Awesome-Deep-Multi-View-Clustering) 合集等工作的启发。我们感谢这些作者对多视图聚类社区的宝贵贡献。
