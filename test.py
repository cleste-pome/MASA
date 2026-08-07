"""
test.py — MASA 测试程序 · 评估入口
=============================================
跑法：改下面「TODO」参数 → 点运行 / python test.py

功能：加载训练好的模型权重（.pth），对指定数据集执行前向推理与聚类评估，
      输出 ACC/NMI/PUR/ARI（metric.py 的 valid()）。

评估做的事：
  1. 加载数据集（与训练相同的归一化流程，utils/dataloader.py）
  2. 构建网络（MASA.py 的 Network）并加载 .pth 权重（严格匹配维度）
  3. 无梯度前向 → 对晚期融合全局特征与各视图压缩特征做 K-means 聚类
  4. 输出 ACC / NMI / PUR / ARI（metric.py 的 valid()）

输出：
  - 控制台表格（每个视图 + 全局 Y 的四个指标）
  - 汇总表（每份权重一行：样本数/视图数/ACC/NMI/PUR/ARI/评估用时 + 总评估用时）

注意：
  ⚠ 权重维度必须与网络结构匹配（feature_dim / high_feature_dim 与训练时一致）
  ⚠ 权重目录下每份 .pth 都是一次独立实验，全部评估、各占一行
  ⚠ 单份权重评估失败不影响其他权重（跳过并标注）
"""

# ── 导入模块 ──
import argparse     # 命令行参数解析（可选，日常改顶部 TODO 即可）
import glob         # 通配符匹配权重文件（目录内找 .pth）
import os           # 路径与文件操作
import sys          # 程序退出（路径不存在 / 未输入）
import time         # 记录每个数据集的评估用时
from itertools import chain  # 展平各视图维度列表

os.environ["OMP_NUM_THREADS"] = "1"  # 必须在 torch/numpy 导入前设置：macOS OpenMP 冲突防护

import numpy as np  # 数值计算
import torch        # 深度学习框架

from tabulate import tabulate                 # 汇总表格（与 metric.py 样式一致）
from MASA import Network                      # 网络结构定义（论文 AVE+MASA）
from utils.metric import valid                 # 聚类评估（ACC/NMI/PUR/ARI）
from utils.device_check import detect_device  # 设备自动选择（CUDA > MPS > CPU）
from utils.dataloader import MATKind          # 数据集加载（.mat 多视图格式）
from utils import GlobalLocalManifoldCalibration as GLMC  # 用于关闭 σ 周期打印（见下行）
GLMC.SIGMA_PRINT_ENABLED = False  # 测试只评估一次，σ 变化观察无意义，关掉


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                               超参数 —— 你只需要改这里                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# TODO 1：模型权重路径 — 评估用的 .pth 文件，或权重目录（推荐目录）
#   填 .pth 文件:  MODEL_PATH = "4.models/MSRCV1/MSRCV120260801-120042.pth"
#   填目录（自动按数据集匹配权重，目录下每份权重都会评估、各占一行）:
#                 MODEL_PATH = "4.models"
#   留空 "" 时运行中会在控制台提示你输入
MODEL_PATH = f"4.models"           # 字符串：权重路径（.pth 文件或目录）

# TODO 2：要评估的数据集 — 逗号分隔；留空 = datasets/ 下全部 .mat（推荐）
#   例: DATASETS = "ALOI-100"
#   例: DATASETS = "ALOI-100,MSRCV1"
#   例: DATASETS = "datasets/MSRCV1.mat"   ← 直接传 .mat 路径也可以（自动去路径）
DATASETS = ""                                 # 字符串：数据集名或 .mat 路径，逗号分隔（留空 = datasets/ 下全部）

# TODO 3：网络维度 — 必须与训练时一致（否则权重加载会报错）
#   ⚠ 训练默认值是 64 / 20；若训练时改过 --feature_dim / --high_feature_dim，这里要同步
#   例: 训练时用了 --feature_dim 128 --high_feature_dim 32
#       → 这里改 FEATURE_DIM = 128，HIGH_FEATURE_DIM = 32
FEATURE_DIM = 64          # 整数：编码器输出维度
HIGH_FEATURE_DIM = 20     # 整数：对比压缩维度

# TODO 4：随机种子 — 影响 K-means 初始化的可复现性
#   例: SEED = 2026   ← 换一个种子，结果可复现但数值可能不同
SEED = 50                 # 整数：随机种子


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 1 步：收集要评估的数据集                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def collect_datasets(names=None):
    """收集要评估的数据集：names 为空时遍历 datasets/ 下全部 .mat。
    条目兼容两种写法：纯数据集名（如 MSRCV1）或 .mat 文件路径
    （如 datasets/MSRCV1.mat，自动去掉路径与扩展名）；不存在的数据集打印警告并跳过。"""
    if names:
        out = []
        for n in names.split(","):
            n = n.strip()
            if not n:
                continue
            if n.lower().endswith(".mat"):       # 兼容传 datasets/MSRCV1.mat 这类路径
                n = os.path.basename(n)[:-4]
            if os.path.isfile(os.path.join("datasets", n + ".mat")):
                out.append(n)
            else:
                print(f"[跳过] datasets/{n}.mat 不存在")
        return out
    return [f[:-4] for f in sorted(os.listdir("datasets")) if f.endswith(".mat")]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  第 2 步：列出某数据集可用的全部权重                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def resolve_weights(model_arg, dataset_name):
    """解析权重路径：文件→[该文件]；目录→{目录}/{数据集}/*.pth 全部（按修改时间旧→新）
    或 {目录}/{数据集}.pth；找不到→[]"""
    if os.path.isfile(model_arg):
        return [model_arg]
    if os.path.isdir(model_arg):
        sub = os.path.join(model_arg, dataset_name)    # 候选子目录 {目录}/{数据集}/
        if os.path.isdir(sub):
            pths = sorted(glob.glob(os.path.join(sub, "*.pth")), key=os.path.getmtime)
            if pths:
                return pths                            # 全部权重（每份 = 一次独立实验）
        direct = os.path.join(model_arg, f"{dataset_name}.pth")  # 备选单文件
        if os.path.isfile(direct):
            return [direct]
    return []


def _fmt_duration(seconds):
    """秒数 → 可读文本（如 12.3s / 1m05s）"""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m{int(sec):02d}s"


if __name__ == '__main__':
    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  第 3 步：确定权重路径与数据集（命令行参数可选，日常改顶部 TODO 即可）  ║
    # ╚══════════════════════════════════════════════════════════════════════╝

    # --model：模型权重路径（必填，优先级：本参数 > 顶部 MODEL_PATH > 控制台交互输入）。填法：
    #   例: --model 4.models/MSRCV1/MSRCV120260801-120042.pth   ← 填 .pth 文件
    #   例: --model 4.models                                    ← 填目录（自动按数据集匹配）
    parser = argparse.ArgumentParser(description='MASA 测试：加载权重并评估聚类性能')
    parser.add_argument('--model', type=str, default=None, help='权重路径：.pth 文件，或目录（自动按数据集匹配）')
    # --datasets：要评估的数据集（优先级：本参数 > 顶部 DATASETS > 全部）。填法：
    #   例: --datasets ALOI-100              ← 只评一个数据集（纯名称）
    #   例: --datasets datasets/MSRCV1.mat   ← 也可以直接传 .mat 路径
    #   例: --datasets MSRCV1,Out-Scene      ← 多个数据集用逗号分隔
    parser.add_argument('--datasets', type=str, default='', help='数据集名或 .mat 路径，逗号分隔（默认：datasets/ 下全部）')
    # --feature_dim / --high_feature_dim：网络维度（默认取顶部 TODO 3，须与训练时一致）
    #   例: --feature_dim 128 --high_feature_dim 32   ← 训练时若改过，这里要同步
    parser.add_argument('--feature_dim', type=int, default=FEATURE_DIM, help='编码器输出维度（须与训练时一致）')
    parser.add_argument('--high_feature_dim', type=int, default=HIGH_FEATURE_DIM, help='对比压缩维度（须与训练时一致）')
    # --seed：随机种子（默认取顶部 TODO 4）
    #   例: --seed 2026
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    args = parser.parse_args()

    # 权重路径优先级：--model > MODEL_PATH > 控制台交互输入
    model_arg = args.model or (MODEL_PATH.strip() or None)
    if model_arg is None:
        print('未指定模型路径：可在文件顶部 TODO 1 填 MODEL_PATH，或直接输入（如 4.models）：')
        if os.path.isdir('4.models'):
            subs = sorted(os.listdir('4.models'))
            print(f'当前可用（4.models/ 下）: {", ".join(subs) if subs else "（空）"}')
        model_arg = input('> ').strip()
        if not model_arg:
            print('未输入路径，退出。')
            sys.exit(0)

    if not os.path.exists(model_arg):   # 路径校验：不存在则报错退出
        print(f'[错误] 权重路径不存在: {model_arg}')
        sys.exit(1)
    args.model = model_arg

    # 数据集优先级：--datasets > DATASETS > 全部
    args.datasets = args.datasets or (DATASETS.strip() or '')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  第 4 步：随机种子 + 设备探测（CUDA > MPS > CPU）                      ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device, _ = detect_device(verbose=False)
    print(f'设备: {device} | 权重: {args.model}')

    datasets = collect_datasets(args.datasets)
    print(f'待评估数据集: {datasets}')

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  第 5 步：逐个数据集评估（加载权重 → 前向 → K-means → 指标）            ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    summary = []   # 每项: (数据集名, 权重文件名, 样本数, 视图数, acc, nmi, pur, ari, 评估用时秒)
    for name in datasets:
        print('\n' + '=' * 66)
        print(f'  数据集: {name}')
        print('=' * 66)
        weights = resolve_weights(args.model, name)          # 该数据集可用的全部权重（空则跳过）
        if not weights:
            print(f'[跳过] 未找到 {name} 的权重文件（--model 为文件时请确认结构匹配）')
            summary.append((name, "—", None, None, None, None, None, None, None))
            continue
        print(f'权重（共 {len(weights)} 份，按时间旧→新）:')
        for w in weights:
            print(f'  - {w}')

        for weight_path in weights:
            print(f'\n--- 评估权重: {os.path.basename(weight_path)} ---')
            t0 = time.perf_counter()                         # 本份权重评估计时起点
            try:
                dataset = MATKind(name, "datasets")          # 加载数据集（与训练相同的归一化流程）
                class_num = dataset.num_classes              # 类别数
                data_size = len(dataset)                     # 样本数
                view = dataset.num_views                     # 视图数
                dims = list(chain.from_iterable(dataset.dims.tolist()))  # 各视图维度
                print(f'样本数={data_size}, 视图数={view}, 类别数={class_num}, 各视图维度={dims}')

                model = Network(view, dims, args.feature_dim, args.high_feature_dim, device)
                state_dict = torch.load(weight_path, map_location=device)
                model.load_state_dict(state_dict)            # 严格匹配：权重维度须与网络结构一致
                model.to(device)
                model.eval()                                 # 测试模式（关闭 Dropout）

                acc, nmi, pur, ari, results = valid(model, device, dataset, view, data_size, class_num,
                                                    pre_train=False, con_train=True)
                summary.append((name, os.path.basename(weight_path), data_size, view,
                                acc, nmi, pur, ari, time.perf_counter() - t0))  # 记录指标与用时
            except Exception as e:
                print(f'[错误] {name} 评估失败: {e}')
                summary.append((name, os.path.basename(weight_path),
                                None, None, None, None, None, None, None))

    # ╔══════════════════════════════════════════════════════════════════════╗
    # ║  第 6 步：汇总 —— 每份权重一行（数据规模 + 指标 + 评估用时）            ║
    # ╚══════════════════════════════════════════════════════════════════════╝
    print('\n' + '=' * 66)
    print('  汇总（晚期融合全局特征 Y，每份权重 = 一次独立实验）')
    print('=' * 66)
    # 组装表格行：权重文件/样本数/视图数/四个指标/评估用时；评估失败用 "—" 占位
    table = []
    for name, wname, data_size, view, acc, nmi, pur, ari, elapsed in summary:
        if acc is not None:
            table.append([name, wname, data_size, view, acc * 100, nmi * 100, pur * 100, ari * 100,
                          _fmt_duration(elapsed)])
        else:
            table.append([name, wname, "—", "—", "—", "—", "—", "—", "—"])
    print(tabulate(table,
                   headers=["数据集", "权重文件", "样本数", "视图数", "ACC", "NMI", "PUR", "ARI", "评估用时"],
                   tablefmt="grid", floatfmt=".2f"))
    # 总评估用时（所有权重合计）
    ok = [r for r in summary if r[8] is not None]
    if ok:
        print(f'\n总评估用时（{len(ok)} 份权重合计）: {_fmt_duration(sum(r[8] for r in ok))}')
