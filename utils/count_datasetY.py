# ===================== 标准库 =====================
import math
import os

# ===================== 第三方库 =====================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 配色（经 dataviz 校验器验证：categorical slot 1/2 + status serious）
COLOR_BLUE = "#2a78d6"    # 主色（categorical slot 1）
COLOR_ORANGE = "#eb6834"  # 辅助色（categorical slot 2）
COLOR_RARE = "#ec835a"    # 稀有类（status serious，配合文字标注使用）
SURFACE = "#fcfcfb"       # 图表表面
GRID_COLOR = "#e1e0d9"    # hairline 网格
INK_SECONDARY = "#52514e"  # 次要文字


def count_classes(dataset_name, Y_RealLabel, show=False, pause_sec=2):
    out_dir = f"1.logs/{dataset_name}"
    """
    绘制单张图（全英文标签）：包含类别统计信息和分析结果
    """
    Y = np.asarray(Y_RealLabel).ravel()
    classes, counts = np.unique(Y, return_counts=True)
    classes = classes.astype(int)
    classes = classes + 1  # display from 1
    total = int(counts.sum())
    proportions = counts / total * 100 if total > 0 else np.zeros_like(counts)

    # 指标
    K = len(classes)
    max_c, min_c = counts.max(), counts.min()
    mean_c = counts.mean()
    max_share = proportions.max()/100
    imbalance_ratio = max_c / min_c if min_c > 0 else np.inf

    # 稀有类
    rare_th = 0.2 * mean_c
    is_rare = counts < rare_th

    # 输出路径
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_ClassReport.png")

    # 全局风格（hairline 网格、细边、克制配色）
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Songti SC", "Heiti SC", "Microsoft YaHei", "SimHei", "sans-serif"],
        "axes.unicode_minus": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK_SECONDARY,
        "axes.grid": False,  # 每轴按需单独开 y 方向网格
        "axes.axisbelow": True,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
    })

    # 建图（使用 constrained_layout，无 tight_layout 警告）
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.3], wspace=0.05, hspace=0.25)
    ax_count = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    # 左图：类别统计（正常=蓝，稀有=serious 橙红，无描边）
    colors = np.where(is_rare, COLOR_RARE, COLOR_BLUE)
    bars = ax_count.bar(classes, counts, color=colors, width=0.7)
    ax_count.set_xlabel("Class ID")
    ax_count.set_ylabel("Count")
    ax_count.grid(True, axis="y")
    # 类别多时只显示部分刻度（显示密度约 15 个）
    if K > 20:
        tick_step = max(1, math.ceil(K / 15))
        ax_count.set_xticks(classes[::tick_step])
    else:
        ax_count.set_xticks(classes)
    ax_count.tick_params(axis="x", labelsize=9)
    ax_count.set_ylim(0, max(counts) * 1.15 if len(counts) else 1)
    ax_count.set_title(f"{dataset_name} - Class Sample Statistics", fontsize=14)

    # 类别少时标注柱顶数值；类别多时省略（与 Y 轴刻度重复且会互相遮挡）
    if K <= 12:
        for b, c, p in zip(bars, counts, proportions):
            ax_count.text(b.get_x() + b.get_width() / 2, b.get_height(),
                          f"{int(c)}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9,
                          color=INK_SECONDARY)

    if is_rare.any():
        ax_count.text(0.01, 0.98, f"Orange bars = rare classes (Count < {rare_th:.1f})",
                      transform=ax_count.transAxes, ha="left", va="top",
                      fontsize=9, bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                            edgecolor=GRID_COLOR, linewidth=0.8))

    # 右图：长尾与累计占比（单轴：两条线都换算为 0-100% 占比，避免双 y 轴误导）
    order = np.argsort(counts)[::-1]
    counts_sorted = counts[order]
    share = counts_sorted / total * 100
    cum = np.cumsum(counts_sorted) / total * 100
    ranks = np.arange(1, len(counts_sorted) + 1)
    # 点数多时 marker 抽样（约每 1/10 处一个点），避免长尾段挤成一团
    mark_every = max(1, len(ranks) // 10)
    ax_rank.plot(ranks, share, marker="o", color=COLOR_BLUE, linewidth=1.8, markersize=7,
                 markevery=mark_every, label="Per-class share")
    ax_rank.plot(ranks, cum, marker=".", color=COLOR_ORANGE, linewidth=1.4, markersize=6,
                 markevery=mark_every, label="Cumulative share")
    ax_rank.set_xlabel("Rank")
    ax_rank.set_ylabel("Share (%)")
    ax_rank.set_title(f"{dataset_name} - Long-tail & Coverage Analysis", fontsize=14)
    ax_rank.set_ylim(0, 105)
    ax_rank.grid(True, axis="y")
    # 强制横坐标刻度为整数，并自动控制显示密度
    ax_rank.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax_rank.legend(frameon=False, fontsize=9, loc="upper right")
    # Top-K 标注：按排名百分比取点（1%,2%,5%,10%,20%,50%,100%），类别数多时均匀铺开、
    # 不挤在长尾头部；自然包含半程点（50%，如 8 类必有 Top-4）；相邻标注上下交替错开；
    # 末位 Top-K（=K）恒为 100% 且贴边易截断，不标注
    top_keys = sorted({max(1, int(round(p * K))) for p in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)})
    label_step = max(share) * 0.06
    for idx, k in enumerate(top_keys):
        if len(cum) > k:
            offset = label_step if idx % 2 == 0 else -label_step
            y_pos = max(share[k - 1] + offset, 0.0)
            ax_rank.text(k, y_pos, f"Top-{k}:{cum[k - 1]:.1f}%",
                         fontsize=9, ha="left", va="bottom" if idx % 2 == 0 else "top",
                         color=INK_SECONDARY)

    # 前 K//2 个类别的累计占比（均匀数据恰为 0.5；K=1 无长尾概念，记 0）
    half_k = K // 2
    imbalance_half = cum[half_k - 1] / 100.0 if half_k >= 1 else 0.0

    if max_share > 0.5:
        status1 = f"severely imbalanced (largest-class share = {max_share:.2f} > 0.5)"
    else:
        status1 = f"roughly balanced (largest-class share = {max_share:.2f} < 0.5)"

    if 0.75 > imbalance_half > 0.5:
        status2 = f"slight long tail (Top K/2 = {imbalance_half:.2f} > 0.5)"
    elif 0.90 > imbalance_half >= 0.75:
        status2 = f"long-tailed (Top K/2 = {imbalance_half:.2f} > 0.75)"
    elif imbalance_half >= 0.90:
        status2 = f"strong long tail (Top K/2 = {imbalance_half:.2f} > 0.9)"
    else:
        status2 = "fairly balanced distribution"

    status = f'{status1}, {status2}'

    # 下方：分析文字（英文原句 + 对应中文，逐句中英双语并列）
    note = "(Many classes; top-of-bar value labels omitted to avoid overlap)" if K > 12 else ""
    msg = (
        f"Dataset: {dataset_name}    数据集：{dataset_name}\n"
        f"Total samples: {total}    Classes: {K}    总样本数：{total}    类别数：{K}\n"
        f"Largest class: {max_c} ({max_share*100:.2f}%)    最大类：{max_c}（占比 {max_share*100:.2f}%）"
        f"    Smallest class: {min_c}    最小类：{min_c}\n"
        f"Imbalance ratio: {imbalance_ratio:.2f}    不平衡比率：{imbalance_ratio:.2f}"
        f"    Status: {status}\n"
        f"Notes: left panel shows per-class counts and shares; right panel shows the sorted long tail\n"
        f"       with cumulative coverage. Orange bars are rare classes (< 20% of mean). {note}\n"
        f"说明：左图显示各类别样本数与占比，右图显示排序后的长尾分布与累计覆盖率；"
        f"橙柱表示稀有类（低于均值的 20%）。"
    )
    # 稀有类说明只在图上确有橙柱时出现（无稀有类时该句不成立）
    rare_en = " Orange bars are rare classes (< 20% of mean)." if is_rare.any() else ""
    rare_cn = "；橙柱表示稀有类（低于均值的 20%）。" if is_rare.any() else "。"
    note_suffix = f"{rare_en} {note}".rstrip()
    msg = (
        f"Dataset: {dataset_name}  数据集：{dataset_name}\n"
        f"Total samples: {total}   Classes: {K}    总样本数：{total}  类别数：{K}\n"
        f"Largest class: {max_c} ({max_share*100:.2f}%)  最大类：{max_c}（{max_share*100:.2f}%）"
        f"    Smallest class: {min_c}  最小类：{min_c}\n"
        f"Imbalance ratio: {imbalance_ratio:.2f}  不平衡比率：{imbalance_ratio:.2f}"
        f"    Status: {status}\n"
        f"Notes: left panel shows per-class counts and shares; right panel shows the sorted long tail\n"
        f"       with cumulative coverage.{note_suffix}\n"
        f"说明：左图显示各类别样本数与占比，右图显示排序后的长尾分布与累计覆盖率{rare_cn}"
    )
    ax_text.text(0.02, 0.5, msg, ha="left", va="center", fontsize=15,
                 bbox=dict(boxstyle="round,pad=0.7", facecolor="#f9f9f7",
                           edgecolor=GRID_COLOR, linewidth=1))

    fig.suptitle(f"{dataset_name} - Class Statistics & Distribution Analysis", fontsize=16)

    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(pause_sec)
        plt.close()
