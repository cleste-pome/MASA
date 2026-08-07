# ===================== 标准库 =====================
import math
import os

# ===================== 第三方库 =====================
import numpy as np
import matplotlib.pyplot as plt

"""
# 用例
plot_loss(imgs_path, preloss_list, Dataname, 'pretrain loss')
plot_acc(imgs_path, acc_list, Dataname, 'acc', x_values=epoch_ticks)
"""


def plot_loss(imgs_path, loss_list, dataset_name, name):
    os.makedirs(imgs_path, exist_ok=True)

    epochs = len(loss_list)
    plt.figure(figsize=(12, 6))

    # 绘制损失曲线，缩小折线宽度和点的大小
    plt.plot(range(1, epochs + 1), loss_list, marker='o', color='tab:blue', linestyle='-', linewidth=1, markersize=5,
             label=f'{name}')

    # 设置坐标轴标签，优化字体大小和加粗
    plt.xlabel('Epoch', fontsize=14, fontweight='bold', color='darkblue')
    plt.ylabel(f'{name}', fontsize=14, fontweight='bold', color='darkblue')
    plt.title(f'{dataset_name}[{name}]', fontsize=16, fontweight='bold', color='darkred')

    # 设置x轴和y轴的范围
    plt.xlim(0, epochs + 1)
    plt.ylim(min(loss_list) - 0.1, max(loss_list) + 0.1)

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
    plt.savefig(f'{imgs_path}/{dataset_name}_ep{epochs}_{name}.png', dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    # 窗口显示n秒后自动继续执行
    plt.pause(2)
    # 自动关闭窗口
    plt.close()

    print(f'Plot saved as {imgs_path}/{dataset_name}_ep{epochs}_{name}.png')


# 定义绘制准确率曲线的函数，参数acc_list为各轮训练的准确率列表
def plot_acc(imgs_path, acc_list, dataset_name, name, x_values=None):
    """绘制评价指标曲线。

    :param x_values: 每个评价点对应的真实 epoch 列表（长度与 acc_list 一致，整数）；
                     为 None 时退回按验证点序号 1..len(acc_list) 绘图。
    """
    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    # 获取总的训练轮数（点数）与横坐标（真实 epoch）
    epochs = len(acc_list)
    x = x_values if x_values is not None else list(range(1, epochs + 1))
    last_epoch = x[-1]
    # 设置绘图的大小
    plt.figure(figsize=(12, 6))
    # 绘制准确率曲线，设置线型、点标记、线宽等
    plt.plot(x, acc_list, marker='o', linestyle='-', linewidth=2, markersize=6)

    # 设置x轴和y轴的标签及其字体大小
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(f'{name}', fontsize=14)
    # 设置图表的标题及其字体大小
    plt.title(f'{dataset_name}[{name}]', fontsize=16)

    # 计算最大准确率及其对应的轮数（真实 epoch）
    max_acc = max(acc_list)
    max_epoch = x[acc_list.index(max_acc)]
    # 获取最后一轮的准确率
    last_acc = acc_list[-1]

    # 绘制表示最大准确率的水平线
    plt.axhline(y=max_acc, color='gray', linestyle='--', linewidth=0.5)
    # 在图表上标注最大准确率及其对应的轮数
    plt.text(last_epoch, max_acc, f'Max {name}: {max_acc * 100:.2f}% at Epoch {max_epoch}', ha='right',
             va='bottom',
             fontsize=10)
    # 在图表上标注最后一轮的准确率
    plt.text(1, 0, f'Last {name}: {last_acc * 100:.2f}%', ha='right', va='bottom', fontsize=10,
             transform=plt.gca().transAxes)

    # 设置x轴的刻度（整数）：验证点较多时抽样显示（上限约10个刻度），
    # 并强制把最后一轮标到横坐标上
    step = max(1, math.ceil(epochs / 10))
    ticks = x[::step]
    if last_epoch not in ticks:
        ticks.append(last_epoch)
    plt.xticks(ticks)
    plt.xlim(min(x), max(x))

    # 设置y轴的刻度
    plt.yticks(np.arange(min(acc_list), max(acc_list) + 0.05, step=0.05))
    # 设置仅在y轴方向显示网格线
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    # 自动调整子图参数，确保图表的元素不会重叠
    plt.tight_layout()

    # TODO 文件名
    filename = f'{imgs_path}/{dataset_name}_ep{last_epoch}_{name}.png'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # # 保存图表为PNG文件，指定分辨率为300dpi
    # plt.savefig(filename, dpi=300)
    # # 显示图表
    # plt.show()
    # # 打印保存的图表文件名
    # print(f'Plot saved as {filename}')

    # 保存图表为PNG文件，指定分辨率为300dpi
    plt.savefig(filename, dpi=300)
    # 显示图表，设置为非阻塞
    plt.show(block=False)
    # 窗口显示n秒后自动继续执行
    plt.pause(2)
    # 自动关闭窗口
    plt.close()
    # 打印保存的图表文件名
    print(f'Plot saved as {filename}')


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

    print(f'Plot saved as {filename}')
