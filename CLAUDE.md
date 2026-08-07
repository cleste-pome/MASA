# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 这是什么

TPAMI'26 投稿论文《Towards Robust Multi-View Clustering via Early-to-late Manifold Consistency Calibration》方法 **MASA**（Manifold Consistency Guided Adaptive Sparse Autoencoder）的 **纯代码副本**（不含 LaTeX，疑似为 GitHub 发表准备的独立副本，目前尚未 git init）。同级目录 `MASA[ClaudeCode核查]/` 是论文+核查工作区，其 `MASA代码/` 是 2026-07 核查后重构的代码版本；**本目录是重构前的旧布局快照（文件时间 2026-02 前后）**，改代码前先确认改的是哪一份。

## 运行

```
python train.py
```

- `main()` 遍历 `datasets/` 下所有 `.mat`，每个数据集自动跑一轮完整的两阶段训练+验证。`--dataset` 的默认值随遍历到的文件而定，命令行传 `--dataset` 会与日志/模型文件名不同步（默认用法即正确用法，不要传）。
- **test.py**（2026-08-02 加入）：独立测试程序——加载 .pth 权重 + 数据集 → 前向 → K-means 评估。**改文件顶部 `MODEL_PATH`/`DATASETS` 两个变量后直接运行即可**（需填写的内容与例子见文件顶部注释）；也可 `python test.py --model 4.models --datasets MSRCV1`，或留空交互输入。权重路径优先级：`--model` > `MODEL_PATH` > 交互输入。训练内嵌评估由 `metric.py` 的 `valid()` 每 N 轮做一次 K-means（n_init=100）；小数据集 pre 阶段每 100 轮、con 阶段每 10 轮验证；样本数 ≥2500 时 con_epochs 强制 600、每 100 轮验证。
- argparse 已补 type（2026-08-01）：`--learning_rate`/`--pre_epochs`/`--con_epochs`/`--iter`/`--feature_dim`/`--high_feature_dim`/`--seed`/`--weight_decay`/`--batch_size` 及四个扰动比例均可安全从命令行覆盖（此前未写 type 的参数传了会以字符串进入代码直接崩）。
- **设备前置检查**（2026-08-01 新增）：`utils/device_check.py`，可独立运行 `python utils/device_check.py` 只探测打印，也可被 train.py 启动时自动调用。决策规则 CUDA > MPS（识别到 Mac 即优先启用）> CPU；环境变量 `MASA_DEVICE=cuda|mps|cpu|auto` 可强制指定（不可用或非法值自动回退）。
- **运行时间报告**（2026-08-01 新增）：train.py 内置分段计时（`measure` 上下文管理器 + `print_timing_report`），训练结束自动打印并写入日志：总耗时、设备检查、各数据集的数据加载/模型构建/预训练/一致性训练/出图与保存；验证评估(KMeans) 与 t-SNE 可视化作为阶段内细分项单列（已含在阶段耗时内）。报告为 tabulate grid 表格形式；Logger 已做 handler 去重（多数据集循环不再重复打印同一行日志）。
- 批大小被强制为全量数据（`batch_size = data_size`，仅 NUSWIDEOBJ 为 256），即全批次训练。
- 输出目录均为运行时自动创建：`1.logs/`（日志，文件名=`{数据集名}{noise}_{conflict}_{missing}.log`，注意数据集名与第一个比值直接拼接）、`2.results_imgs/`（损失/指标曲线）、`3.csv/view/`（每轮指标 csv）、`4.models/`（.pth）、`5.tsne/`（t-SNE pdf+svg）、`7.ViewWeights/`（每轮视图权重 csv）。

## 代码结构（与论文模块对应）

- **train.py**：训练入口。`pretrain()`=AVE 预训练（重建+稀疏），`contrastive_train()`=一致性训练（对比损失+重建+循环一致性，最后一轮触发 feature_separation 图）。`--iter` 默认 1 只跑一轮；多轮时 seed/lr 每轮随机扰动，结尾 `l_max` 取"最后 epoch 指标最优的那次运行"（代码注释注明论文用法是多次取平均，此为辅助实现）。
- **MASA.py**（原 EdgeMVC.py，2026-08-01 改名）：`Network`（对应论文 AVE+MASA）。`zero_value_proportion(xs)` 探测各视图稀疏率（AVE 自适应机制）；per-view `Encoder`/`Decoder` + 1 个拼接全局编码器（稀疏位置 `sparse_at=(1,4,7,)`、dropout 0.2）；`common_information_module`（公共信息投影头）、`cycle_transfer_module`（循环一致性）、`feature_fusion(zs, Wz)`（按视图权重加权融合得全局特征 H）。forward 里保留了注意力/边缘/费舍尔/核对齐等历史备选加权方案的 TODO 注释（全部注释掉，勿启用）。`Network` 记录 `self.device`，forward 不再硬编码 cuda（2026-08-01 修复）。
- **GlobalLocalManifoldCalibration.py**（原 ReWeightView.py，2026-08-01 改名）：**ELMC 核心**（全局-局部流形校准）。`manifold_alignment_weights(zs, z_all, sigma=None, device=None)` 用拉普拉斯矩阵迹对齐算视图权重；`kernel_alignment_weights` 为核对齐备选。顶部 `SCORE_FORM`（**默认 'trace'**=论文 Eq. 9 原式 Tr(L_v L_G)，保留结构对齐与视图能量先验适配 GLDA；'cosine'/'distance'/'l_sym' 谱归一化/'z_norm' 样本级归一化为实验备选，对应论文消融表各行）+ `SIGMA_MODE`（**默认 'median'**=论文 Step 2 σ 取全局距离中位数；'fixed'/'mean'/'quantile'/'local' 为实验备选）。σ 每轮随全局特征重算、随训练演化：每 50 次调用周期打印（含变化率），全量记入 history，训练结束由 `plot_sigma` 自动画变化曲线到 `2.results_imgs/`（'fixed' 模式无历史自动跳过）；算 σ 的全局距离矩阵复用于全局 Laplacian（省一次 O(n²) cdist）。device=None 自动跟随输入张量设备，MPS 下 cdist/diag 自动回退 CPU（本机 MPS 实测）。⚠️ 切换 SCORE_FORM/SIGMA_MODE 会改变权重 w_v 数值，与论文表格的一致性待重跑验证。
- **loss.py**：`ContrastiveLoss`（`loss_type="classical"` 在用，nt_xent/vicreg/barlow_twins/triplet/easy/proco_von 是实验备选）；`ae_loss_function()` = 重建 MSE + KL 稀疏，含 C_spa 自适应稀疏系数（mean≤0.01 置 0，否则线性映射到 (0,1]），ρ=0.05、β=1.0。
- **metric.py**：`valid()` 的 sklearn KMeans 评估，pre/con 两阶段分开打分。
- **utils/**（扁平命名，重构版已改为 `_01` 序号后缀并按 data/output/kmeans/analysis 分子目录）：
  - `dataloader.py`：`MultiViewDataset` + `MATKind`（读 `datasets/*.mat`：`X` 为各视图矩阵的 cell 数组、`Y` 为标签）；min-max 归一化；`addNoise`/`addConflict`/`addMissing`/`addSparsity` 四种鲁棒性扰动注入（对应 `--*_ratio`，0.0=干净数据）。
  - `Logger.py`、`metric2csv.py`（`find_max_weighted_sum_index` 等）、`plot.py`、`tsne_visual.py`（pdf+svg 双输出）、`device_check.py`（设备前置检查，见"运行"节）。
  - `count_datasetY.py`（count_classes）：train.py import 并在数据集加载时调用。**2026-08-01 已删除**（死 import / 零引用孤儿，调用点均在注释里）：`metric_cuda.py`（原 train.py 注释引用，已连同注释移除）、`FisherInformation.py`、`Sample_SelfWeight.py`（原 MASA.py 顶层 import，已连同 import 一并移除）、`SingleDimsDifferentiation.py`（原 train.py import）、`EdgSensitive1/2.py`、`KmeansGPU1/2.py`、`cuda_test.py`；这些工具在父目录 `MASA代码-效果差/` 有同名副本可查。

## 与本副本的"主版本"差异（重要）

`MASA[ClaudeCode核查]/MASA代码-效果差/`（2026-07 核查后重构，目录名注记"效果差"）相对本目录的重构：`EdgeMVC.py`（本目录现已改名 MASA.py）`+ReWeightView.py`（本目录现已改名 GlobalLocalManifoldCalibration.py）→ `network.py`、`metric.py` → `metrics.py`、新增 `test.py`（`python test.py --dataset X --model path.pth`）、去掉对比损失 1/view 系数、全零权重均匀回退防 NaN。本目录 2026-08-01 已对齐：argparse 补 type、设备前置检查（`utils/device_check.py`，等价重构版 auto_device）、ELMC 论文版公式（`SCORE_FORM='trace'` + σ 全局距离中位数）。仍为旧版：文件布局（未拆 network.py/metrics.py/test.py）、对比损失 1/view 系数、全零权重回退。若此副本要发 GitHub，先确认发布对象是旧版还是重构版。**论文表格数值来自历史实验，改代码不会自动改表格。**

## 环境与依赖

- Python 3.12（`__pycache__` 为 cpython-312）；依赖 torch、numpy、scipy、scikit-learn、tqdm、matplotlib；GPU K-means 备选需 cuML/cuPy（可选）。
- `OMP_NUM_THREADS=1` 在 import torch 前设置（train.py 已有，macOS 必需）。
- `.idea/` 的 PyCharm 项目名是 "EdgeSensitive"（历史项目名），无运行配置。
