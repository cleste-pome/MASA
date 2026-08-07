# ===================== 第三方库 =====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    通用对比损失类（Contrastive Loss）。
    支持多种不同的对比损失计算方法，包括 NT-Xent, VICReg, Barlow Twins, Triplet Loss, ProCo 等。
    """
    # TODO 注意注意：该选择对比损失了੭ ᐕ)੭ bingoվ'ᴗ' ի
    def __init__(self, batch_size, device, loss_type):
        """
        初始化方法。
        :param batch_size: 每次批处理的数据量大小
        :param device: 计算设备（如 'cuda' 或 'cpu'）
        :param loss_type: 选择使用的对比损失方法，可选：
                         - 1.1 "classical" (默认) 余弦 (点积)相似度对比
                         - 1.2 "nt_xent"  对比交叉熵 (InfoNCE): 归一化温度缩放交叉熵损失
                         - 2. "vicreg"  方差-协方差正则化 (VICReg)
                         - 3. "barlow_twins"  协方差去冗余 (Barlow Twins)
                         - 4. "triplet"  传统 Triplet Loss
                         - 5. "easy"  简易快速对比学习
                         - 5. "proco"  概率对比学习 (ProCo)
        """
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = 1.0  # 温度参数，拉伸(torch.matmul)时影响缩放程度
        self.device = device
        self.loss_type = loss_type.lower()

    def forward(self, h_i, h_j, weight=None):
        """
        前向传播，计算指定类型的对比损失。
        :param h_i: 输入的第一个嵌入向量 (N, D)
        :param h_j: 输入的第二个嵌入向量 (N, D)
        :param weight: 可选的损失权重
        :return: 对比损失值
        """
        if self.loss_type == "classical":
            return self.classical_loss(h_i, h_j, weight)
        elif self.loss_type == "nt_xent":
            return self.nt_xent_loss(h_i, h_j, weight)
        elif self.loss_type == "vicreg":
            return self.vicreg_loss(h_i, h_j, weight)
        elif self.loss_type == "barlow_twins":
            return self.barlow_twins_loss(h_i, h_j, weight)
        elif self.loss_type == "triplet":
            return self.triplet_loss(h_i, h_j, weight)
        else:
            raise ValueError(f"未知的对比损失类型: {self.loss_type}")

    def classical_loss(self, h_i, h_j, weight=None):
        """
        1.1 计算 classical(默认) 余弦 (点积)相似度对比损失
        """
        N = self.batch_size  # 批处理大小
        # 计算相似度矩阵（点积相似度除以温度参数）
        # TODO similarity_matrix = torch.matmul(h_i, h_j.T) / (len(h_i[0]) * self.temperature) # 除以维度20，MSRCV1.mat性能会下降20个百分点
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        # 提取正样本对的相似度（相似度矩阵的对角线元素）
        positives = torch.diag(similarity_matrix)
        # 创建一个N*N的掩码矩阵，用于屏蔽对角线元素
        mask = torch.ones((N, N), device=self.device)  # 创建全1矩阵
        mask.fill_diagonal_(0)  # 将对角线置为0
        # 计算分子：正样本对相似度的指数
        numerator = torch.exp(positives)
        # 计算分母：掩码矩阵中保留的所有元素的指数值
        denominator = torch.exp(similarity_matrix) * mask
        # 逐行计算损失：-log(分子/分母之和)
        loss_partial = -torch.log(numerator / torch.sum(denominator, dim=1))
        # 计算平均损失
        loss = torch.sum(loss_partial) / N
        # 如果提供了权重参数，应用权重
        if weight is not None:
            loss = weight * loss
        # TODO loss / len(h_i[0]) ，MSRCV1.mat性能会下降6个百分点
        return loss

    def nt_xent_loss(self, h_i, h_j, weight=None):
        """
        1.2 计算 NT-Xent（InfoNCE）对比损失
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
        """
        2. 计算 VICReg 损失（无负样本）
        """
        N, D = h_i.shape
        sim_loss = torch.mean((h_i - h_j) ** 2)
        std_i = torch.sqrt(h_i.var(dim=0) + 1e-4)
        std_j = torch.sqrt(h_j.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_i)) + torch.mean(F.relu(1 - std_j))
        cov_i = (h_i.T @ h_i) / (N - 1)
        cov_j = (h_j.T @ h_j) / (N - 1)
        cov_loss = torch.norm(cov_i - torch.eye(D, device=h_i.device), p="fro") + \
                   torch.norm(cov_j - torch.eye(D, device=h_j.device), p="fro")
        loss = sim_loss + 25.0 * var_loss + 1.0 * cov_loss
        return loss if weight is None else weight * loss

    def barlow_twins_loss(self, h_i, h_j, weight=None):
        """
        3. 计算 Barlow Twins 损失（去冗余）
        """
        N, D = h_i.shape
        h_i = (h_i - h_i.mean(dim=0)) / h_i.std(dim=0)
        h_j = (h_j - h_j.mean(dim=0)) / h_j.std(dim=0)
        c = (h_i.T @ h_j) / N
        on_diag = torch.mean((1 - c.diag()) ** 2)
        off_diag = torch.mean(c ** 2) - torch.mean(c.diag() ** 2)
        loss = on_diag + 0.0051 * off_diag
        return loss if weight is None else weight * loss

    def triplet_loss(self, h_i, h_j, weight=None, margin=1.0):
        """
        4. 计算 Triplet Loss
        """
        triplet_loss_fn = nn.TripletMarginLoss(margin=margin)
        loss = triplet_loss_fn(h_i, h_j, torch.roll(h_j, shifts=1, dims=0))  # 随机负样本
        return loss if weight is None else weight * loss


def kl_divergence(rho, rho_hat):
    """
    计算KL散度，用于稀疏性约束。
    :param rho: 稀疏目标值（如0.05）
    :param rho_hat: 实际的平均激活值
    :return: KL散度值
    """
    # 限制激活值范围，防止数值计算出现log(0)或溢出问题
    rho_hat = torch.clamp(rho_hat, 0 + 1e-6, 1 - 1e-6)

    # 计算KL散度
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))


def kl_sparse_loss(hidden_layer_activation, rho, sparse_beta):
    """
    计算稀疏损失，用于正则化隐藏层激活值。
    :param hidden_layer_activation: 隐藏层的激活值
    :param rho: 稀疏目标值
    :param beta: 稀疏正则化强度
    :return: 稀疏正则项
    """
    kl_total = 0.0  # 用来累加每层的 KL 稀疏损失

    for layer in range(len(hidden_layer_activation)):

        # 计算隐藏层激活值的平均值
        rho_hat = torch.mean(hidden_layer_activation[layer], dim=0)

        # 计算KL散度损失
        kl_loss = kl_divergence(rho, rho_hat).mean()

        kl_total += kl_loss

    # 返回加权的KL散度损失
    return sparse_beta * kl_total/len(hidden_layer_activation)


def ae_loss_function(mean, reconstructed_x, x, hidden_layer_activation, rho=0.05, beta=1.0):

    threshold = 0.01  # 稀疏系数阈值

    # TODO 1.稀疏项系数 C_spa：小于阈值的值置为0，大于阈值的部分映射到(0, 1]
    C_spa = np.where(mean <= threshold, 0, (mean - threshold) / (1 - threshold))

    # TODO 计算重构误差
    criterion = torch.nn.MSELoss()  # 定义均方误差损失函数
    reconstruction_loss = criterion(reconstructed_x, x)
    # error = reconstructed_x - x
    # squared_error = error ** 2
    # reconstruction_loss = squared_error.mean()

    # 计算稀疏性约束损失
    # rho = 1 - C_spa * 1.0  # TODO loss会快速变为nan，C_spa是ndarray格式乘以1.0后变为float64
    # rho = C_spa * 1.0  # TODO loss会快速变为nan

    # TODO 2.损失函数重建项与稀疏项之间的比例系数 beta：默认为1
    sparse_beta = C_spa*beta

    if C_spa > 0:
        sparse_loss = kl_sparse_loss(hidden_layer_activation, rho, sparse_beta)
        ae_loss = reconstruction_loss + sparse_loss
    else:
        ae_loss = reconstruction_loss

    return ae_loss
