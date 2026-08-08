"""
MASA.py — 模型定义（对应论文 AVE + MASA 模块）
================================================
包含：
  1. zero_value_proportion：各视图稀疏率探测（AVE 自适应机制的先验输入）
  2. Encoder / Decoder：视图特定与全局的自编码器
  3. Network：整体网络（AVE 自适应稀疏编码 → ELMC 加权融合 → 全局特征 H），
     对应论文 AVE（自适应视图特定编码）与 MASA（早期-晚期流形一致性校准）
"""

# ===================== 第三方库 =====================
import torch
import torch.nn as nn
from torch.nn.functional import normalize

# ===================== 项目内部模块 =====================
from utils.GlobalLocalManifoldCalibration import manifold_alignment_weights


def zero_value_proportion(xs):
    """探测各视图的稀疏率（低于数值阈值的维度比例），作为 AVE 自适应稀疏约束的先验。

    :param xs: 视图数据字典 {视图索引: 张量 (N, D_v)}
    :return: {视图索引: (稀疏率均值, 稀疏率方差)}
    """
    threshold = torch.finfo(torch.float16).smallest_normal  # 数值阈值：FP16 最小规格化数
    stats = {}
    for key, view in xs.items():
        # 每个样本中低于阈值的维度比例
        zero_counts = (torch.abs(view) < threshold).sum(dim=1).float()
        proportions = zero_counts / view.shape[1]
        # 稀疏率的均值与方差（保留 4 位小数）
        mean_proportion = round(torch.mean(proportions).item(), 4)
        variance_proportion = round(torch.var(proportions, unbiased=False).item(), 4)
        stats[key] = (mean_proportion, variance_proportion)
    return stats


class Encoder(nn.Module):
    """视图编码器：多层线性 + ReLU + Dropout，输出低维特征。

    sparse_at 指定在哪些层收集 ReLU 激活（供 KL 稀疏正则使用），
    稀疏层索引对应注释中的序号（1/4/7 为隐藏层激活，9 为输出特征）。
    """

    def __init__(self, input_dim, feature_dim, dropout_rate=0.0, sparse_at=(1, 4, 7)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),        # 0
            nn.ReLU(),                        # 1 ← 可稀疏（收集激活）
            nn.Dropout(dropout_rate),         # 2
            nn.Linear(500, 500),              # 3
            nn.ReLU(),                        # 4 ← 可稀疏
            nn.Dropout(dropout_rate),         # 5
            nn.Linear(500, 2000),             # 6
            nn.ReLU(),                        # 7 ← 可稀疏
            nn.Dropout(dropout_rate),         # 8
            nn.Linear(2000, feature_dim),     # 9 输出特征
        )
        self.sparse_at = set(sparse_at)

    def forward(self, x):
        """前向：返回输出特征与稀疏层激活列表（Dropout 之前的 ReLU 输出）。"""
        sparse_acts = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.sparse_at:
                sparse_acts.append(x)
        return x, sparse_acts


class Decoder(nn.Module):
    """视图解码器：将编码特征重建回输入维度。"""

    def __init__(self, input_dim, feature_dim, dropout_rate=0.0):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    """MASA 整体网络。

    结构：
      - 每个视图一个 Encoder/Decoder（AVE 视图特定编码）
      - 一个拼接全局编码器/解码器（早期融合的全局表示）
      - ELMC 视图权重（manifold_alignment_weights）加权晚期融合 -> 全局特征 H
      - common_information_module：视图公共信息投影（GLDA 对比对齐用）
    """

    def __init__(self, view, input_size, feature_dim, high_feature_dim, device):
        super().__init__()
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.view = view
        self.device = device
        self.input_size_all = sum(input_size)

        # 每个视图一个编码器/解码器 + 一个全局（拼接输入）编码器/解码器
        self.encoders = nn.ModuleList(
            [Encoder(input_size[v], feature_dim, 0.2).to(device) for v in range(view)]
            + [Encoder(sum(input_size), feature_dim, 0.2).to(device)]
        )
        self.decoders = nn.ModuleList(
            [Decoder(input_size[v], feature_dim, 0.2).to(device) for v in range(view)]
            + [Decoder(sum(input_size), feature_dim, 0.2).to(device)]
        )

        # 全局特征融合层（晚期融合：加权视图特征 -> 高维公共表示）
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim),
        )
        # 公共信息投影头（GLDA 对比对齐用）
        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )

        # 循环一致性转化器
        self.cycle_transfer_module = nn.Sequential(
            nn.Linear(feature_dim, 256),  # 线性层，将所有视角的特征维度合并并转换为256
            nn.ReLU(),  # 激活函数ReLU
            nn.Dropout(0.1),
            nn.Linear(256, high_feature_dim)  # 线性层，将维度转换为高特征维度
        )

    # TODO (待定)循环一致性转化器函数
    def cycle_transfer(self, z):
        return self.cycle_transfer_module(z)

    def feature_fusion(self, zs, Wz):
        """按视图权重加权融合视图特征，得到全局特征 H（L2 归一化）。

        :param zs: 各视图编码特征列表 [(N, feature)]
        :param Wz: 每样本每视图的权重矩阵 (N, view)
        :return: 全局融合特征 H (N, high_feature_dim)
        """
        Wz_expanded = Wz.unsqueeze(-1)                # (N, view, 1)
        zs_stacked = torch.stack(zs).permute(1, 0, 2)  # (N, view, feature)
        weighted_sum = torch.sum(zs_stacked * Wz_expanded, dim=1)
        return normalize(self.feature_fusion_module(weighted_sum), dim=1)

    def forward(self, xs):
        """前向：返回 (xrs, zs, rs, H, xr_all, z_all, activation, means, wz_view)。

        :param xs: 视图数据字典 {视图索引: 张量 (N, D_v)}
        :return:
          - xrs: 各视图重建输出
          - zs: 各视图编码特征
          - rs: 各视图公共信息（投影头输出，L2 归一化）
          - H: 晚期融合全局特征（ELMC 加权，L2 归一化）
          - xr_all: 全局重建输出
          - z_all: 早期融合全局编码特征
          - activation: 稀疏层激活（[全局] + 各视图）
          - means: 各视图稀疏率
          - wz_view: ELMC 视图权重
        """
        # AVE：探测各视图稀疏率（自适应稀疏约束的先验）
        means = [mean for mean, _ in zero_value_proportion(xs).values()]
        print(f'Sparsity ratio(zero(missing)_value(dims)_proportion mean)[view]:{means}')

        # 早期融合：拼接输入 -> 全局编码器
        xs2one = torch.cat([xs[key] for key in sorted(xs.keys())], dim=1)
        z_all, hidden_activation_all = self.encoders[self.view](xs2one)
        activation = [hidden_activation_all]

        # 各视图编码
        zs = []
        for v in range(self.view):
            z, hidden_activation = self.encoders[v](xs[v])
            zs.append(z)
            activation.append(hidden_activation)

        # ELMC：视图权重（早期融合全局流形作锚点 -> 晚期融合校准）
        Wz_view = manifold_alignment_weights(zs, z_all)  # sigma=None → 全局距离中位数
        Wz = Wz_view.repeat(z_all.shape[0], 1).to(self.device)
        print(f'Wz_view:{Wz_view}')

        # 各视图解码与公共信息投影
        xrs, rs = [], []
        for v in range(self.view):
            xrs.append(self.decoders[v](zs[v]))
            rs.append(normalize(self.common_information_module(zs[v]), dim=1))
        xr_all = self.decoders[self.view](z_all)

        # 晚期融合：ELMC 加权 -> 全局特征 H
        H = self.feature_fusion(zs, Wz)

        return xrs, zs, rs, H, xr_all, z_all, activation, means, Wz_view
