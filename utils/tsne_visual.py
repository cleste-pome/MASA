# ===================== 标准库 =====================
import os

# ===================== 第三方库 =====================
import matplotlib.pyplot as plt


# TODO 1. 绘制并保存 SVG 格式的二维嵌入可视化图
def plot_svg_embeddings(embeds, labels, path, epoch, datasetname):
    # 创建正方形图像窗口，大小为 6x6 英寸
    fig = plt.figure(figsize=(6, 6))
    # 创建子图（1行1列中的第1个）
    ax = plt.subplot(111)

    # 绘制二维嵌入的散点图，使用标签作为颜色，tab10 是 10 类分类的常用配色方案
    ax.scatter(embeds[:, 0], embeds[:, 1], c=labels, cmap='tab10', marker='.', s=10)

    # 设置坐标轴单位长度一致，确保 x 和 y 轴比例一致
    ax.set_aspect('equal')

    # 设置坐标轴的显示范围为正方形，确保图像长宽比为 1:1
    x_min, x_max = embeds[:, 0].min(), embeds[:, 0].max()
    y_min, y_max = embeds[:, 1].min(), embeds[:, 1].max()
    lim_min = min(x_min, y_min)
    lim_max = max(x_max, y_max)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # 移除 x 和 y 轴的刻度，使图像更加简洁
    ax.set_xticks([])
    ax.set_yticks([])

    # 如果保存路径不存在，则创建该路径
    if not os.path.exists(path):
        os.makedirs(path)

    # 构建 SVG 文件的保存路径
    svg_path = os.path.join(path, f"{epoch}_{datasetname}.svg")

    # 保存图像为 SVG 格式文件
    plt.savefig(svg_path, format='svg')
    # 关闭图像窗口，释放内存
    plt.close(fig)

    # 返回保存的文件路径
    return svg_path


# TODO 2. 绘制并保存 PDF 格式的二维嵌入可视化图
def plot_embeddings(embeds, labels, path, epoch, datasetname):
    # 设置 PDF 字体为矢量字体，防止放大后失真
    plt.rcParams["pdf.fonttype"] = 42

    # 创建正方形图像窗口，大小为 6x6 英寸
    fig = plt.figure(figsize=(6, 6))
    # 创建子图
    ax = plt.subplot(111)

    # 绘制二维嵌入的散点图，使用标签作为颜色，tab10 配色方案，点大小为 10
    ax.scatter(embeds[:, 0], embeds[:, 1], c=labels, cmap='tab10', marker='.', s=10)

    # 设置坐标轴单位长度一致，确保 x 和 y 轴比例一致
    ax.set_aspect('equal')

    # 设置坐标轴的显示范围为正方形，确保图像长宽比为 1:1
    x_min, x_max = embeds[:, 0].min(), embeds[:, 0].max()
    y_min, y_max = embeds[:, 1].min(), embeds[:, 1].max()
    lim_min = min(x_min, y_min)
    lim_max = max(x_max, y_max)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)

    # 移除 x 和 y 轴的刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 如果保存路径不存在，则创建该路径
    if not os.path.exists(path):
        os.makedirs(path)

    # 构建 PDF 文件的保存路径
    pdf_path = os.path.join(path, f"{epoch}_{datasetname}.pdf")

    # 保存图像为 PDF 格式文件
    plt.savefig(pdf_path, format='pdf')
    # 关闭图像窗口，释放内存
    plt.close(fig)

    # 返回保存的文件路径
    return pdf_path
