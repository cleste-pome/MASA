# ===================== 标准库 =====================
import math
import os

# ===================== 第三方库 =====================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

# 配色（经 dataviz 校验器验证：categorical slot 1/2 + status serious）
COLOR_BLUE = "#2a78d6"    # 主色（categorical slot 1）
COLOR_ORANGE = "#eb6834"  # 辅助色（categorical slot 2）
COLOR_RARE = "#ec835a"    # 稀有类（status serious，配合文字标注使用）
SURFACE = "#fcfcfb"       # 图表表面
GRID_COLOR = "#e1e0d9"    # hairline 网格
INK_SECONDARY = "#52514e"  # 次要文字


# 自动选择系统中存在的中文字体
def setup_matplotlib_cn_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "Microsoft JhengHei",
        "PingFang SC", "Heiti SC",
        "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False
    print("未找到中文字体，可能无法正确显示中文。")
    return None


def count_classes(dataset_name, Y_RealLabel, show=False, pause_sec=2):
    out_dir = f"1.logs/{dataset_name}"
    """
    绘制单张图：包含类别统计信息和分析结果
    """
    setup_matplotlib_cn_font()

    Y = np.asarray(Y_RealLabel).ravel()
    classes, counts = np.unique(Y, return_counts=True)
    classes = classes.astype(int)
    classes = classes + 1  # ✅ 新增：显示从1开始
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
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.3], wspace=0.18, hspace=0.25)
    ax_count = fig.add_subplot(gs[0, 0])
    ax_rank = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[1, :])
    ax_text.axis("off")

    # 左图：类别统计（正常=蓝，稀有=serious 橙红，无描边）
    colors = np.where(is_rare, COLOR_RARE, COLOR_BLUE)
    bars = ax_count.bar(classes, counts, color=colors, width=0.7)
    ax_count.set_xlabel("类别编号（Class ID）")
    ax_count.set_ylabel("样本数（Count）")
    ax_count.grid(True, axis="y")
    # 类别多时只显示部分刻度，避免数字挤成一团（显示密度约 15 个）
    if K > 20:
        tick_step = max(1, math.ceil(K / 15))
        ax_count.set_xticks(classes[::tick_step])
    else:
        ax_count.set_xticks(classes)
    ax_count.tick_params(axis="x", labelsize=9)
    ax_count.set_ylim(0, max(counts) * 1.15 if len(counts) else 1)
    ax_count.set_title(f"{dataset_name} - 类别样本统计", fontsize=14)

    # 类别少时标注柱顶数值；类别多时省略（与 Y 轴刻度重复且会互相遮挡）
    if K <= 12:
        for b, c, p in zip(bars, counts, proportions):
            ax_count.text(b.get_x() + b.get_width() / 2, b.get_height(),
                          f"{int(c)}\n({p:.1f}%)", ha="center", va="bottom", fontsize=9,
                          color=INK_SECONDARY)

    if is_rare.any():
        ax_count.text(0.01, 0.98, f"橙色柱体 = 稀有类（Count < {rare_th:.1f}）",
                      transform=ax_count.transAxes, ha="left", va="top",
                      fontsize=9, bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                                            edgecolor=GRID_COLOR, linewidth=0.8))

    # 右图：长尾与累计占比（单轴：两条线都换算为 0-100% 占比，避免双 y 轴误导）
    order = np.argsort(counts)[::-1]
    counts_sorted = counts[order]
    share = counts_sorted / total * 100
    cum = np.cumsum(counts_sorted) / total * 100
    ranks = np.arange(1, len(counts_sorted) + 1)
    ax_rank.plot(ranks, share, marker="o", color=COLOR_BLUE, linewidth=1.8, markersize=7,
                 label="每类占比")
    ax_rank.plot(ranks, cum, marker=".", color=COLOR_ORANGE, linewidth=1.4, markersize=6,
                 label="累计占比")
    ax_rank.set_xlabel("频次排名（Rank）")
    ax_rank.set_ylabel("占比（%）")
    ax_rank.set_title(f"{dataset_name} - 长尾与覆盖率分析", fontsize=14)
    ax_rank.set_ylim(0, 105)
    ax_rank.grid(True, axis="y")
    # ✅ 强制横坐标刻度为整数，并自动控制显示密度（避免挤成一团）
    ax_rank.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax_rank.legend(frameon=False, fontsize=9, loc="upper right")
    # Top-K 标注：保证包含半程点 Top-K//2（与下方文字分析的 Top K/2 结论对应，
    # 如 8 类必有 Top-4）；相邻标注上下交替错开，避免文字互相遮挡；
    # 末位 Top-K 恒为 100% 且贴边易截断，不标注
    top_keys = [1, 3, 5, 10, 20, 50, 100] if K >= 20 else [1, 2, 3, 5, 10]
    if K // 2 >= 1:
        top_keys.append(K // 2)
    top_keys = sorted(set(top_keys))
    label_step = max(share) * 0.05
    for idx, k in enumerate(top_keys):
        if len(cum) > k:
            offset = label_step if idx % 2 == 0 else -label_step
            y_pos = max(share[k - 1] + offset, 0.0)
            ax_rank.text(k, y_pos, f"Top-{k}:{cum[k - 1]:.1f}%",
                         fontsize=9, ha="left", va="bottom" if idx % 2 == 0 else "top",
                         color=INK_SECONDARY)

    half_k = K // 2  # 向下取整
    if K % 2 == 0:
        # K 是偶数：取前 K/2 和 K/2 + 1 个类别中间两点的平均（线性插值）
        imbalance_half = (cum[half_k - 1] / 100.0 +cum[half_k] / 100.0) / 2.0
    else:
        # K 是奇数：正好取前 K/2 个类别
        imbalance_half = cum[half_k] / 100.0

    if max_share > 50:
        status1 = f"最大类别严重不平衡（最大类占比 = {max_share:.2f} > 0.5）"
    else:
        status1 = f"最大类别一般平衡（最大类占比 = {max_share:.2f} < 0.5）"

    if 0.75 > imbalance_half > 0.5:
        status2 = f"存在轻微的长尾分布（Top K/2 = {imbalance_half:.2f} > 0.5）"
    elif 0.90 > imbalance_half >= 0.75:
        status2 = f"存在长尾分布（Top K/2 = {imbalance_half:.2f} > 0.75）"
    elif imbalance_half >= 0.90:
        status2 = f"存在明显长尾分布（Top K/2 = {imbalance_half:.2f} > 0.9）"
    else:
        status2 = f"类别分布较均衡"

    status =f'{status1}, {status2}'

    # 下方：分析文字
    note = "（类别数较多，左图省略柱顶数值标签以避免重叠）" if K > 12 else ""
    msg = (
        f"数据集：{dataset_name}\n"
        f"总样本数：{total}    类别数：{K}\n"
        f"最大类样本数：{max_c}（{max_share*100:.2f}%）    最小类样本数：{min_c}\n"
        f"不平衡比：{imbalance_ratio:.2f}    状态：{status}\n"
        f"说明：左图显示每类样本数与比例；右图展示类别排序后长尾与累计覆盖率。\n"
        f"橙色柱体为稀有类（低于均值20%）。{note}"
    )
    ax_text.text(0.01, 0.5, msg, ha="left", va="center", fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#f9f9f7",
                           edgecolor=GRID_COLOR, linewidth=1))

    fig.suptitle(f"{dataset_name} - 类别统计与分布分析", fontsize=16)

    fig.savefig(out_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(pause_sec)
        plt.close()

    print(f"Plot saved as {out_path}")
