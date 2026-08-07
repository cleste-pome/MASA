"""
loss.py — MASA 损失函数
========================
包含三部分：
  1. ContrastiveLoss：GLDA 全局-局部分布对齐使用的对比损失（支持多种损失形式，
     论文默认 'classical'，其余为实验备选）
  2. kl_divergence / kl_sparse_loss：AVE 的 KL 稀疏正则项
  3. ae_loss_function：AVE 自编码损失（重建 MSE + 自适应 KL 稀疏，系数 C_spa
     由视图稀疏率自适应调节，对应论文 AVE 模块）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """对比损失（GLDA 使用，loss_type='classical' 为论文默认）。

    支持的损失形式：
      - 'classical'     余弦（点积）相似度对比（论文默认）
      - 'nt_xent'       归一化温度缩放交叉熵（InfoNCE）
      - 'vicreg'        方差-协方差正则化（VICReg）
      - 'barlow_twins'  协方差去冗余（Barlow Twins）
      - 'triplet'       传统三元组损失
    """

    def __init__(self, batch_size, device, loss_type):
        """初始化。

        :param batch_size: 批大小（构造负样本掩码矩阵用）
        :param device: 计算设备（CUDA / MPS / CPU）
        :param loss_type: 对比损失形式（见类文档，不区分大小写）
        """
        super().__init__()
        self.batch_size = batch_size
        self.temperature = 1.0  # 温度参数：缩放相似度矩阵
        self.device = device
        self.loss_type = loss_type.lower()

        # 损失形式 -> 计算函数 的映射（forward 按此分发）
        self._loss_fns = {
            "classical": self.classical_loss,
            "nt_xent": self.nt_xent_loss,
            "vicreg": self.vicreg_loss,
            "barlow_twins": self.barlow_twins_loss,
            "triplet": self.triplet_loss,
        }

    def forward(self, h_i, h_j, weight=None):
        """计算指定形式的对比损失。

        :param h_i: 第一个嵌入向量 (N, D)
        :param h_j: 第二个嵌入向量 (N, D)
        :param weight: 可选的损失权重（标量）
        :return: 对比损失值
        """
        fn = self._loss_fns.get(self.loss_type)
        if fn is None:
            raise ValueError(f"未知的对比损失类型: {self.loss_type}")
        return fn(h_i, h_j, weight)

    def classical_loss(self, h_i, h_j, weight=None):
        """'classical'：点积相似度对比损失（论文默认）。

        正样本对 = 同一样本跨视图的嵌入（相似度矩阵对角线）；
        负样本对 = 其余所有跨视图组合（经掩码屏蔽对角线后保留）。
        """
        N = self.batch_size
        # 相似度矩阵：跨视图嵌入的点积，除以温度缩放
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        positives = torch.diag(similarity_matrix)          # 正样本对（对角线）
        mask = torch.ones((N, N), device=self.device)
        mask.fill_diagonal_(0)                             # 屏蔽对角线，仅保留负样本
        numerator = torch.exp(positives)                   # 正样本对指数
        denominator = torch.exp(similarity_matrix) * mask  # 负样本对指数
        # 每行：-log(正样本 / 负样本之和)，再取平均
        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N
        if weight is not None:
            loss = weight * loss
        return loss

    def nt_xent_loss(self, h_i, h_j, weight=None):
        """'nt_xent'：NT-Xent（InfoNCE）对比损失。

        与 classical 的区别：输入先做 L2 归一化（相似度退化为余弦相似度），
        分母加 1e-8 防止除零。
        """
        N = self.batch_size
        h_i = F.normalize(h_i, dim=1)
        h_j = F.normalize(h_j, dim=1)
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N), device=self.device)
        mask.fill_diagonal_(0)
        numerator = torch.exp(positives)
        denominator = torch.sum(torch.exp(similarity_matrix) * mask, dim=1)
        loss = -torch.log(numerator / (denominator + 1e-8))
        return loss.mean() if weight is None else weight * loss.mean()

    def vicreg_loss(self, h_i, h_j, weight=None):
        """'vicreg'：VICReg 损失（方差-协方差正则化，无需负样本）。

        三项：不变性（两嵌入差异）、方差（激活标准差趋向 1）、
        协方差（嵌入维度间去相关），系数沿用 VICReg 原文 25/1。
        """
        N, D = h_i.shape
        sim_loss = torch.mean((h_i - h_j) ** 2)                       # 不变性项
        std_i = torch.sqrt(h_i.var(dim=0) + 1e-4)
        std_j = torch.sqrt(h_j.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_i)) + torch.mean(F.relu(1 - std_j))  # 方差项
        cov_i = (h_i.T @ h_i) / (N - 1)
        cov_j = (h_j.T @ h_j) / (N - 1)
        cov_loss = (torch.norm(cov_i - torch.eye(D, device=h_i.device), p="fro")
                    + torch.norm(cov_j - torch.eye(D, device=h_j.device), p="fro"))  # 协方差项
        loss = sim_loss + 25.0 * var_loss + 1.0 * cov_loss
        return loss if weight is None else weight * loss

    def barlow_twins_loss(self, h_i, h_j, weight=None):
        """'barlow_twins'：Barlow Twins 去冗余损失。

        互相关矩阵的对角线逼近 1（不变性）、非对角线逼近 0（去冗余），
        系数 0.0051 沿用原文。
        """
        N, D = h_i.shape
        h_i = (h_i - h_i.mean(dim=0)) / h_i.std(dim=0)   # 零均值单位方差归一化
        h_j = (h_j - h_j.mean(dim=0)) / h_j.std(dim=0)
        c = (h_i.T @ h_j) / N                            # 互相关矩阵
        on_diag = torch.mean((1 - c.diag()) ** 2)        # 对角线 -> 1
        off_diag = torch.mean(c ** 2) - torch.mean(c.diag() ** 2)  # 非对角线 -> 0
        loss = on_diag + 0.0051 * off_diag
        return loss if weight is None else weight * loss

    def triplet_loss(self, h_i, h_j, weight=None, margin=1.0):
        """'triplet'：三元组损失（负样本取 h_j 的循环平移，即随机负样本）。"""
        triplet_loss_fn = nn.TripletMarginLoss(margin=margin)
        loss = triplet_loss_fn(h_i, h_j, torch.roll(h_j, shifts=1, dims=0))
        return loss if weight is None else weight * loss


def kl_divergence(rho, rho_hat):
    """KL 散度：衡量实际平均激活 rho_hat 与稀疏目标 rho 的偏离。

    :param rho: 稀疏目标值（如 0.05）
    :param rho_hat: 实际平均激活值
    :return: KL 散度值
    """
    # 裁剪激活值，避免 log(0) 或溢出
    rho_hat = torch.clamp(rho_hat, 1e-6, 1 - 1e-6)
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def kl_sparse_loss(hidden_layer_activation, rho, sparse_beta):
    """多层激活的 KL 稀疏正则损失：逐层计算 KL 散度取平均后加权。

    :param hidden_layer_activation: 各稀疏层的激活值列表
    :param rho: 稀疏目标值
    :param sparse_beta: 稀疏正则化强度（可为逐视图自适应的系数）
    :return: 加权 KL 稀疏损失
    """
    kl_total = 0.0
    for activation in hidden_layer_activation:
        rho_hat = torch.mean(activation, dim=0)      # 该层平均激活
        kl_total += kl_divergence(rho, rho_hat).mean()  # 逐特征 KL 取平均
    return sparse_beta * kl_total / len(hidden_layer_activation)


def ae_loss_function(mean, reconstructed_x, x, hidden_layer_activation, rho=0.05, beta=1.0):
    """AVE 自编码损失 = 重建 MSE + 自适应 KL 稀疏项（对应论文 AVE 模块）。

    稀疏系数 C_spa 由视图稀疏率 mean 自适应调节：
      - mean <= 阈值（0.01）：稀疏项关闭，退化为标准自编码器；
      - mean >  阈值：C_spa 线性映射到 (0, 1]，越稀疏的视图获得越强的稀疏约束。

    :param mean: 视图稀疏率（zero_value_proportion 的输出均值）
    :param reconstructed_x: 重建输出
    :param x: 原始输入
    :param hidden_layer_activation: 稀疏层激活列表
    :param rho: KL 稀疏目标值（默认 0.05）
    :param beta: 稀疏项基系数（默认 1.0）
    :return: AVE 损失值
    """
    threshold = 0.01  # C_spa 的激活阈值
    # 自适应稀疏系数：mean <= 阈值 置 0，否则线性映射到 (0, 1]
    C_spa = np.where(mean <= threshold, 0, (mean - threshold) / (1 - threshold))

    # 重建损失（MSE）
    reconstruction_loss = torch.nn.MSELoss()(reconstructed_x, x)

    # 稀疏损失：强度 = C_spa * beta（稀疏视图 -> 更强稀疏约束）
    if float(C_spa) > 0:
        sparse_beta = C_spa * beta
        sparse_loss = kl_sparse_loss(hidden_layer_activation, rho, sparse_beta)
        return reconstruction_loss + sparse_loss
    return reconstruction_loss
