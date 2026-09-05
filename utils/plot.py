# ===================== 标准库 =====================
import math
import os

# ===================== 第三方库 =====================
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

"""
# 用例
plot_loss(imgs_path, preloss_list, Dataname, 'pretrain loss')
plot_acc(imgs_path, acc_list, Dataname, 'acc', x_values=epoch_ticks)
"""


def plot_loss(imgs_path, loss_list, dataset_name, name, total_epochs=None, components=None):
    """绘制损失曲线，可选附带损失分量曲线。

    :param name: 图名（下划线形式，如 'pretrain_loss'，显示时自动转回空格）。
    :param total_epochs: 总训练轮数（pre+con），用于文件名统一 ep 后缀；None 时退回阶段轮数。
    :param components: 分量字典 {标签: 值列表}，如 {'global_ae': [...], 'view_ae': [...]}；
                       非 None 时只画各分量曲线（总损失 = 分量之和，单独画会把分量压扁看不见）。
    """
    os.makedirs(imgs_path, exist_ok=True)

    epochs = len(loss_list)
    total = total_epochs if total_epochs is not None else epochs
    display_name = name.replace('_', ' ')
    plt.figure(figsize=(12, 6))

    # 无分量时画总损失曲线
    if components is None:
        plt.plot(range(1, epochs + 1), loss_list, marker='o', color='tab:blue', linestyle='-', linewidth=2,
                 markersize=5, label=f'{display_name}')
        curve_values = loss_list
    # 有分量时只画各分量曲线（预训练 2 条：global_ae/view_ae；一致性 3 条：再 + contrastive；总损失=分量之和）
    else:
        curve_values = []
        for label, comp_list in components.items():
            plt.plot(range(1, len(comp_list) + 1), comp_list, marker='.', linestyle='-', linewidth=1.2,
                     markersize=4, label=label)
            curve_values.extend(comp_list)

    # 设置坐标轴标签，优化字体大小和加粗
    plt.xlabel('Epoch', fontsize=14, fontweight='bold', color='darkblue')
    plt.ylabel(f'{display_name}', fontsize=14, fontweight='bold', color='darkblue')
    plt.title(f'{dataset_name}[{name}]', fontsize=16, fontweight='bold', color='darkred')

    # 设置x轴和y轴的范围（有分量时按分量曲线的最小最大值取）
    plt.xlim(0, epochs + 1)
    plt.ylim(min(curve_values) - 0.1, max(curve_values) + 0.1)

    # 绘制网格线，调整样式为虚线且增加透明度
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)

    # 设置背景颜色，保持简洁的灰色背景
    plt.gca().set_facecolor('#f7f7f7')

    # 设置坐标轴样式，边框颜色和宽度
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # 添加图例
    plt.legend(fontsize=12)

    # 调整图表布局并保存
    plt.tight_layout()
    plt.savefig(f'{imgs_path}/{dataset_name}_ep{total}_{name}.png', dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    # 窗口显示n秒后自动继续执行
    plt.pause(2)
    # 自动关闭窗口
    plt.close()


# 定义绘制准确率曲线的函数，参数acc_list为各轮训练的准确率列表
def plot_acc(imgs_path, acc_list, dataset_name, name, x_values=None, pre_epochs=None):
    """绘制评价指标曲线（出版风格：彩色细主线、虚线浅灰网格、阶段标注、整数 nice 刻度）。

    横坐标使用点序号（曲线宽度由 CSV 行数决定），刻度标签映射回真实 epoch：
    如 pre 段 30 行 → 30 个点占 30 单位宽，但刻度标 10,20,...,300。

    :param x_values: 每个点对应的真实 epoch 列表（长度与 acc_list 一致，整数）；
                     为 None 时横坐标即点序号、刻度与之一致。
    :param pre_epochs: AVE 预训练轮数；给定时在 pre/con 交界处画分隔线与阶段标注。
    """
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    # 数据与横坐标：点序号（宽度由行数决定），x_labels 为每个点对应的真实 epoch
    n = len(acc_list)
    x = list(range(1, n + 1))
    x_labels = x_values if x_values is not None else x
    last_epoch = x_labels[-1]
    max_acc = max(acc_list)
    max_epoch = x_labels[acc_list.index(max_acc)]
    last_acc = acc_list[-1]

    # 全局风格（出版风格配置）
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "#CCD0D5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # 主线：参考风格中的视图线样式（彩色、细线）
    ax.plot(x, acc_list, color="#0C5DA5", linewidth=2.0, solid_capstyle="round")

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(f"{name.upper()}", fontweight="bold")
    ax.set_title(f"{dataset_name}  —  {name.upper()}", pad=10)

    # 网格 + 去除上/右边框 + 刻度样式
    ax.grid(True, axis="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", colors="#333333", length=4, width=0.8)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")

    # pre/con 分界（点数）：第一个真实 epoch 超过 pre_epochs 的点
    pre_n = 0
    if pre_epochs is not None:
        pre_n = sum(1 for e in x_labels if e <= pre_epochs)

    # 横坐标刻度：刻度值取真实 epoch 的 nice 间隔（5/10/50/100 的倍数），
    # 位置映射回点序号；pre 段取首/中/尾，con 段取 100 的倍数均匀抽样，强制补最后一轮
    if x_values is not None:
        ticks_val = []
        if pre_n > 0:
            mid = x_labels[(pre_n - 1) // 2]
            ticks_val = sorted(set([x_labels[0], mid, x_labels[pre_n - 1]]))
        con_nice = [e for e in x_labels[pre_n:] if e % 100 == 0]
        if con_nice:
            st = max(1, len(con_nice) // 5)
            ticks_val.extend(con_nice[::st])
        ticks_val = sorted(set(ticks_val))
        if last_epoch not in ticks_val:
            ticks_val.append(last_epoch)
        ticks_pos = [x_labels.index(e) + 1 for e in ticks_val]
    else:
        # 无真实 epoch 表：nice 间隔整数刻度 + 强制末点
        locator = mticker.MaxNLocator(nbins=8, integer=True, steps=[1, 5, 10])
        ticks_pos = sorted({int(round(t)) for t in locator.tick_values(1, n) if 1 <= t <= n})
        if n not in ticks_pos:
            ticks_pos.append(n)
        ticks_val = ticks_pos
    ax.set_xticks(ticks_pos)
    ax.set_xticklabels(ticks_val)
    ax.set_xlim(0, n)

    # y 轴：指标在 [0,1] 时留 0.04 边距，范围过窄时扩到至少 0.12
    lo, hi = min(acc_list), max(acc_list)
    if 0 <= lo and hi <= 1:
        lower, upper = max(0.0, lo - 0.04), min(1.0, hi + 0.04)
        if upper - lower < 0.12:
            center = (upper + lower) / 2
            lower, upper = max(0.0, center - 0.06), min(1.0, center + 0.06)
        ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    # 阶段分隔（标注文字与参考一致）：Pretrain (Independent) | Alignment (Co-training)
    if pre_epochs is not None and pre_n > 0 and n > pre_n:
        y_lo, y_hi = ax.get_ylim()
        split = pre_n + 0.5
        ax.axvspan(split, n, color="#888888", alpha=0.045, zorder=0, lw=0)
        ax.axvline(split, color="#AAAAAA", linewidth=0.8, zorder=1, alpha=0.7)
        kw = dict(fontsize=7.5, fontstyle="italic", color="#999999", ha="center")
        ax.text(pre_n / 2, y_hi - (y_hi - y_lo) * 0.04, "Pretrain (Independent)", va="top", **kw)
        ax.text(pre_n + (n - pre_n) / 2, y_lo + (y_hi - y_lo) * 0.04,
                "Alignment (Co-training)", va="bottom", **kw)

    # 标注最大与最后一轮的值（右下角，避免被曲线遮挡）
    ax.text(0.99, 0.04, f"Max {name.upper()}: {max_acc * 100:.2f}%  @ E{max_epoch}\n"
                        f"Last: {last_acc * 100:.2f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#666666")

    # TODO 文件名（四个单独指标曲线统一放 Metrics/ 子文件夹）
    filename = os.path.join(imgs_path, 'Metrics', f'{dataset_name}_ep{last_epoch}_{name}.png')
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename, dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    # 窗口显示n秒后自动继续执行
    plt.pause(2)
    # 自动关闭窗口
    plt.close()


# 2x2 四宫格汇总图：ACC / NMI / PUR / ARI 各占一格，每格曲线用指标对应颜色
def plot_acc_summary(imgs_path, metrics_dict, dataset_name, x_values=None, pre_epochs=None):
    """绘制四宫格汇总曲线（出版风格：2x2 布局、每格一条曲线、指标对应四色、共享横坐标）。

    :param metrics_dict: {'acc': [...], 'nmi': [...], 'pur': [...], 'ari': [...]}，四序列等长。
    :param x_values: 每个点对应的真实 epoch 列表（与各序列等长）；为 None 时用点序号。
    :param pre_epochs: AVE 预训练轮数；给定时在 pre/con 交界处画分隔线与阶段标注。
    """
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    # 全局风格（与 plot_acc 一致的出版风格配置）
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": "#CCD0D5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    n = len(next(iter(metrics_dict.values())))
    x = list(range(1, n + 1))
    x_labels = x_values if x_values is not None else x
    last_epoch = x_labels[-1]
    pre_n = sum(1 for e in x_labels if e <= pre_epochs) if pre_epochs is not None else 0

    # 四个指标：每个指标对应一种颜色（参考 PALETTE 顺序）、标题、y 轴全名
    specs = [
        ("acc", "#0C5DA5", "ACC", "Accuracy"),
        ("nmi", "#DA3B20", "NMI", "Normalized Mutual Information"),
        ("pur", "#00855C", "PUR", "Purity"),
        ("ari", "#9E4CB0", "ARI", "Adjusted Rand Index"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True,
                             gridspec_kw={"hspace": 0.16})

    for ax, (key, color, short, ylab) in zip(axes.ravel(), specs):
        y = metrics_dict[key]
        ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round")
        ax.set_title(short, pad=8)
        ax.set_ylabel(ylab, fontsize=9)

        # 网格 + 去除上/右边框 + 刻度样式
        ax.grid(True, axis="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", colors="#333333", length=4, width=0.8)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontweight("bold")

        # y 轴：指标在 [0,1] 时留 0.04 边距，范围过窄时扩到至少 0.12
        lo, hi = min(y), max(y)
        if 0 <= lo and hi <= 1:
            lower, upper = max(0.0, lo - 0.04), min(1.0, hi + 0.04)
            if upper - lower < 0.12:
                center = (upper + lower) / 2
                lower, upper = max(0.0, center - 0.06), min(1.0, center + 0.06)
            ax.set_ylim(lower, upper)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

        # 阶段分隔（与单图一致的标注文字）
        if pre_epochs is not None and pre_n > 0 and n > pre_n:
            y_lo, y_hi = ax.get_ylim()
            split = pre_n + 0.5
            ax.axvspan(split, n, color="#888888", alpha=0.045, zorder=0, lw=0)
            ax.axvline(split, color="#AAAAAA", linewidth=0.8, zorder=1, alpha=0.7)
            kw = dict(fontsize=6.5, fontstyle="italic", color="#999999", ha="center")
            ax.text(pre_n / 2, y_hi - (y_hi - y_lo) * 0.04, "Pretrain (Independent)", va="top", **kw)
            ax.text(pre_n + (n - pre_n) / 2, y_lo + (y_hi - y_lo) * 0.04,
                    "Alignment (Co-training)", va="bottom", **kw)

        # 右下角标注该指标的最大与最后一轮的值（与单图一致）
        max_y = max(y)
        max_ep = x_labels[y.index(max_y)]
        ax.text(0.99, 0.04, f"Max {short}: {max_y * 100:.2f}%  @ E{max_ep}\n"
                            f"Last: {y[-1] * 100:.2f}%",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#666666")

    # 共享横坐标刻度（与 plot_acc 相同的 nice 映射），只在底行显示 x 轴标签
    if x_values is not None:
        ticks_val = []
        if pre_n > 0:
            mid = x_labels[(pre_n - 1) // 2]
            ticks_val = sorted(set([x_labels[0], mid, x_labels[pre_n - 1]]))
        con_nice = [e for e in x_labels[pre_n:] if e % 100 == 0]
        if con_nice:
            st = max(1, len(con_nice) // 5)
            ticks_val.extend(con_nice[::st])
        ticks_val = sorted(set(ticks_val))
        if last_epoch not in ticks_val:
            ticks_val.append(last_epoch)
        ticks_pos = [x_labels.index(e) + 1 for e in ticks_val]
    else:
        locator = mticker.MaxNLocator(nbins=8, integer=True, steps=[1, 5, 10])
        ticks_pos = sorted({int(round(t)) for t in locator.tick_values(1, n) if 1 <= t <= n})
        if n not in ticks_pos:
            ticks_pos.append(n)
        ticks_val = ticks_pos
    axes[0][0].set_xticks(ticks_pos)
    axes[0][0].set_xticklabels(ticks_val)
    axes[0][0].set_xlim(0, n)

    # 底行 x 轴标签
    for c in range(2):
        axes[1][c].set_xlabel("Epoch", fontweight="bold", fontsize=10, labelpad=2)

    fig.suptitle(f"{dataset_name}  —  Clustering Performance", fontsize=15, fontweight="bold", y=0.995)

    filename = f'{imgs_path}/{dataset_name}_summary.png'
    fig.savefig(filename, dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    plt.pause(2)
    plt.close()


# 定义绘制 ELMC σ 变化曲线的函数（仅非 fixed 模式时有数据）
def plot_sigma(sigma_history, imgs_path, dataset_name):
    if not sigma_history:
        return
    os.makedirs(imgs_path, exist_ok=True)

    epochs = len(sigma_history)
    plt.figure(figsize=(12, 6))

    # 绘制 σ 曲线，缩小折线宽度和点的大小
    plt.plot(range(1, epochs + 1), sigma_history, marker='o', color='tab:purple', linestyle='-', linewidth=1,
             markersize=3)

    # 设置坐标轴标签，优化字体大小和加粗（保持与 plot.py 其他函数一致的英文标签，
    # 避免 matplotlib 默认字体缺中文字形导致乱码）
    plt.xlabel('Epoch (ELMC call index)', fontsize=14, fontweight='bold', color='darkblue')
    plt.ylabel('sigma (global pairwise-distance median)', fontsize=14, fontweight='bold', color='darkblue')
    plt.title(f'{dataset_name} - ELMC sigma curve', fontsize=16, fontweight='bold', color='darkred')

    # 绘制网格线，调整样式为虚线且增加透明度
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.6)

    # 设置背景颜色，保持简洁的灰色背景
    plt.gca().set_facecolor('#f7f7f7')

    # 设置坐标轴样式，边框颜色和宽度
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    # 标注最后一步的 σ 值
    last_sigma = sigma_history[-1]
    plt.axhline(y=last_sigma, color='gray', linestyle='--', linewidth=0.5)
    plt.text(epochs, last_sigma, f'last σ={last_sigma:.4f}', ha='right', va='bottom', fontsize=10)

    # 设置x轴的刻度：点数较多时抽样显示（上限约10个刻度），避免横坐标挤成一团
    step = max(1, math.ceil(epochs / 10))
    plt.xticks(range(1, epochs + 1, step))

    # 调整图表布局并保存
    plt.tight_layout()
    filename = f'{imgs_path}/{dataset_name}_sigma.png'
    plt.savefig(filename, dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    # 窗口显示n秒后自动继续执行
    plt.pause(2)
    # 自动关闭窗口
    plt.close()
