"""
=====================================================================
 通用脚本工具 (utils/scripts.py)
=====================================================================
训练/评估共用的环境配置、随机种子、分段计时、文本格式与终端输出函数。
train.py / test.py 从这里 import；注意 OMP_NUM_THREADS 等环境变量
必须在 import torch 前设置，故保留在 train.py 顶部，不在此处。
=====================================================================
"""

# ===================== 标准库 =====================
import random
import time
import warnings
from contextlib import contextmanager
from datetime import datetime

# ===================== 第三方库 =====================
import numpy as np
import torch
from tabulate import tabulate

warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
warnings.filterwarnings("ignore", category=FutureWarning)

PLOT_SIGMA = False  # 是否绘制 ELMC σ 变化曲线（模块常量，不做命令行超参数）


def setup_seed(Seed):
    torch.manual_seed(Seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(Seed)  # 为所有GPU设置随机种子
    np.random.seed(Seed)  # 为NumPy设置随机种子
    random.seed(Seed)  # 为Python标准库的random模块设置随机种子
    torch.backends.deterministic = True  # 确保CUDA的确定性（即每次运行结果一致）


# ===================== 运行分段计时（训练结束统一打印/记录） =====================
timing_secs = {}  # 分段名 -> 累计秒数（跨数据集、跨 iter 轮次累加）


@contextmanager
def measure(section):
    """with measure('段名'): 自动把整段耗时（秒）累加到 timing_secs['段名']，可重复累加。
    用于记录各阶段运行时间：数据加载/模型构建/预训练/一致性训练/验证/绘图等。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing_secs[section] = timing_secs.get(section, 0.0) + time.perf_counter() - t0


def _fmt_duration(seconds):
    """秒数 → 可读文本（不足 1 分钟显示秒，超过显示 分/时）"""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {sec:.1f} s"


def _fmt_duration_compact(seconds):
    """秒数 → 表格用紧凑文本（2.4s / 29m18s / 1h42m）"""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(sec)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m"


def _fmt_file_size(nbytes):
    """字节数 → 可读文本（512 B / 3.2 MB / 1.4 GB）"""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024


def _linear_chain(module):
    """提取模块内 Linear 层的输入→输出维度链，如 Encoder: [1302, 500, 500, 2000, 64]"""
    chain = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            if not chain:
                chain.append(m.in_features)
            chain.append(m.out_features)
    return chain


def _fmt_chain(chain):
    """维度链 → '1302 → 500 → 2000 → 64'"""
    return " → ".join(str(d) for d in chain)


def _fmt_ratio_list(values, nd=4):
    """数值列表 → '[0.005, 0.129, ...]'（每轮进度行的稀疏率/视图权重对齐显示）"""
    return "[" + ", ".join(f"{v:.{nd}f}" for v in values) + "]"


def _log_file_only(logger, text):
    """只写日志文件、不打印终端（结果字典等长篇记录）；返回日志路径供终端一句提示"""
    for handler in logger.handlers:
        path = getattr(handler, "baseFilename", None)
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {text}\n")
            return path
    return None


def _kv(key, value, width=24):
    """'key = value' 定宽左对齐（终端分区排版：一行三个键值对，总宽一致列对齐）"""
    return f"{key} = {value}".ljust(width)


# 进度条格式：整行淡蓝色（\x1b[96m 亮青色）；loss/稀疏率/视图权重详情另起一行打印
BAR_FORMAT = "\033[96m{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\033[0m"


def print_model_summary(model):
    """打印网络结构关键信息：各模块结构/参数量/占比（对齐排版，替代 print(model) 的长篇输出）"""
    total = sum(p.numel() for p in model.parameters())
    print(f"\n[Network] Network structure (key info) | Total params: {total:,}")
    print(f"  {'Module':<78s}{'Params':>12s}  {'Share':>6s}")
    print(f"  {'-' * 98}")
    for name, child in model.named_children():
        n_params = sum(p.numel() for p in child.parameters())
        drop = next((m.p for m in child.modules() if isinstance(m, torch.nn.Dropout)), None)
        if isinstance(child, torch.nn.ModuleList):
            chains = [_linear_chain(m) for m in child]
            per_view, global_chain = chains[:-1], chains[-1]  # 最后一个是全局（拼接输入）模块
            in_dims = [c[0] for c in per_view]
            out_dims = [c[-1] for c in per_view]
            if len(set(in_dims)) == 1 and len(set(out_dims)) > 1:
                note = f"differ in output: {out_dims[1:]}"
            elif len(set(out_dims)) == 1 and len(set(in_dims)) > 1:
                note = f"differ in input: {in_dims[1:]}"
            else:
                note = ""
            display = (f"{name} ({child[0].__class__.__name__} x{len(per_view)} + global x1"
                       + (f", Dropout({drop})" if drop else "") + ")")
            detail = [f"per-view [{_fmt_chain(per_view[0])}]" + (f" ({note})" if note else ""),
                      f"global   [{_fmt_chain(global_chain)}]"]
        else:
            display = (f"{name} ({child.__class__.__name__}"
                       + (f", Dropout({drop})" if drop else "") + ")")
            detail = [f"[{_fmt_chain(_linear_chain(child))}]"]
        print(f"  {display:<78s}{n_params:>12,}  {n_params / total * 100:>5.1f}%")
        for ln in detail:
            print(f"      {ln}")
    print(f"  {'Total':<78s}{total:>12,}  {'100.0%':>6s}")
    print("-" * 72)


def print_timing_report(logger=None):
    """训练结束时打印运行时间报告（表格形式，按数据集分组），并同步写入日志文件记录。

    说明：验证(KMeans)/t-SNE 是阶段（预训练/一致性训练）内部的细分项，已包含在所属
    阶段耗时中，单独列出便于定位瓶颈；极小未细分开销未计入。"""
    lines = ["=" * 60, "Run Time Report (stage-wise)", "=" * 60]
    lines.append(f"Total (all datasets): {_fmt_duration(timing_secs.get('Total time', 0.0))}"
                 + (f"  | Device check & startup: {_fmt_duration(timing_secs['Device check & startup'])}"
                    if "Device check & startup" in timing_secs else ""))
    phases = ("Data load & preprocessing", "Model construction", "Pretraining",
              "Consistency training", "Plots & saving")
    headers = ["Dataset", "Data load", "Model build", "Pretrain", "Consistency",
               "incl. Valid", "incl. t-SNE", "Plots & save", "Stage total"]
    rows = []
    datasets = sorted({k.split(':')[0] for k in timing_secs if ':' in k})
    for d in datasets:
        rows.append([
            d,
            _fmt_duration_compact(timing_secs.get(f"{d}: Data load & preprocessing", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: Model construction", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: Pretraining", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: Consistency training", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: Validation (KMeans)", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: t-SNE visualization", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: Plots & saving", 0.0)),
            _fmt_duration_compact(sum(timing_secs.get(f"{d}: {p}", 0.0) for p in phases)),
        ])
    if rows:
        lines.append(tabulate(rows, headers=headers, tablefmt="grid"))
    lines.append("Note: validation (KMeans) / t-SNE are sub-items within stages, already included in the pretrain/consistency times.")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)  # 控制台只打印一次
    if logger is not None:
        # 直接写入当前数据集的日志文件（不走 logger.info，避免控制台重复输出）
        for handler in logger.handlers:
            report_path = getattr(handler, "baseFilename", None)
            if report_path:
                try:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Run Time Report\n{text}\n")
                except OSError:
                    pass
                break
