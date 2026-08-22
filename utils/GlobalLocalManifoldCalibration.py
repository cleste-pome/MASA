import torch

# =====================================================================
# 视图权重计算配置（只影响 manifold_alignment_weights）
#
#   SCORE_FORM —— 一致性分数形式（实验对比用，默认 'trace'）：
#     'trace'     原始迹：S_v = Tr(L_v L_G)，同时保留结构对齐与视图
#                 能量/尺度信息——能量作为视图表达强度的隐式先验，
#                 供后续 GLDA 全局-局部对齐适配
#     'cosine'    余弦核对齐：S_v = Tr(L_v L_G)/(||L_v||_F * ||L_G||_F)
#                 （仅结构方向、丢弃尺度，实验备选）
#     'distance'  距离型：S_v = exp(-||L_v - L_G||_F^2)
#     'l_sym'     谱归一化拉普拉斯 L = I - D^{-1/2} W D^{-1/2} + 原始迹
#     'z_norm'    Z 先样本级 L2 归一化再算距离（余弦距离）+ 原始迹
#
#   SIGMA_MODE —— 高斯核带宽 σ 的设置方式（实验对比用，默认 'median'）：
#     'median'    全局两两距离的中位数（median heuristic，Step 2）
#     'fixed'     固定带宽（SIGMA_FIXED_VALUE，默认 1.0）
#     'mean'      全局两两距离的均值
#     'quantile'  两两距离的低分位数（SIGMA_QUANTILE，默认 0.1，强调近邻）
#     'local'     每样本第 k 近邻距离作局部带宽（SIGMA_KNN_K，默认 5，
#                 W_ab = exp(-d_ab^2 / (sigma_a * sigma_b))）
# =====================================================================
SCORE_FORM = 'trace'
SIGMA_MODE = 'median'
SIGMA_FIXED_VALUE = 1.0
SIGMA_QUANTILE = 0.1
SIGMA_KNN_K = 5

# σ 观测（仅供训练时观察）：σ 每轮从当前全局特征重算，随训练演化；
# 全部值记入 history，供训练结束时绘制 σ 变化曲线（仅非 fixed 模式有数据）
_sigma_stats = {"history": []}


def reset_sigma_history():
    """清空 σ 观测状态（train.py 每个数据集开始时调用，曲线按数据集分开展示）"""
    _sigma_stats["history"] = []


def get_sigma_history():
    """返回 σ 历史（复制，避免外部直接改内部状态）；开关关闭时为空列表"""
    return list(_sigma_stats["history"])



def manifold_alignment_weights(zs, z_all, sigma=None, device=None):
    """
    计算基于流形对齐的视图权重 (支持 GPU)
    S_v 的形式与 σ 的自适应方式由文件顶部 SCORE_FORM / SIGMA_MODE 控制
    :param zs: list，每个元素是一个Tensor，表示不同视图的样本矩阵 (m, d_i)
    :param z_all: Tensor，全局视图的样本矩阵 (m, d)
    :param sigma: float，高斯核参数；None 时按 Step 2 取全局空间两两距离的中位数（median heuristic）
    :param device: str/None，目标设备；None 时自动取输入张量所在设备
    :return: Tensor，大小为 (n,)，表示每个视图的权重
    """
    if device is None:
        device = z_all.device  # 自动跟随输入张量设备（CUDA/MPS/CPU 通用）

    # 迁移到目标设备
    z_all = z_all.to(device)
    zs = [z.to(device) for z in zs]

    if SCORE_FORM == 'z_norm':
        # Z 先做样本级 L2 归一化再进入距离计算（球面投影，距离变为余弦距离）
        z_all = torch.nn.functional.normalize(z_all, dim=1)
        zs = [torch.nn.functional.normalize(z, dim=1) for z in zs]

    # 全局成对距离矩阵：算 σ 用的 d_all 与全局 Laplacian 需要的是同一张矩阵，
    # 共享复用省一次 O(n²) 的 cdist（σ 经 .item() 截断，不参与反传）
    global_d_all = None

    sigma_per_sample = None  # 局部带宽模式（'local'）下为每个样本的带宽向量

    if sigma is None:
        z_for_dist = z_all
        if z_for_dist.device.type == 'mps':
            z_for_dist = z_for_dist.cpu()  # MPS 暂不支持 cdist，回退 CPU
        if SIGMA_MODE in ('median', 'mean', 'quantile'):
            # 全局距离上三角（不含对角线零距离），供各标量自适应模式使用
            global_d_all = torch.cdist(z_for_dist, z_for_dist, p=2)
            with torch.no_grad():
                tri = torch.triu(global_d_all, diagonal=1)
                non_zero = tri[tri > 0]
                if SIGMA_MODE == 'median':
                    # Step 2：带宽自适应设为全局空间两两欧氏距离的中位数
                    sigma = non_zero.median().item() if non_zero.numel() > 0 else 1.0
                elif SIGMA_MODE == 'mean':
                    sigma = non_zero.mean().item() if non_zero.numel() > 0 else 1.0
                else:  # quantile
                    sigma = torch.quantile(non_zero, SIGMA_QUANTILE).item() if non_zero.numel() > 0 else 1.0
                if not (sigma > 0):
                    sigma = 1.0  # 防御：距离全为零等退化情形
        elif SIGMA_MODE == 'fixed':
            sigma = SIGMA_FIXED_VALUE
            global_d_all = None
        elif SIGMA_MODE == 'local':
            # 局部带宽：每个样本用其第 k 近邻距离（Zelnik-Manor & Perona 自适应核）
            global_d_all = torch.cdist(z_for_dist, z_for_dist, p=2)
            with torch.no_grad():
                k = min(SIGMA_KNN_K + 1, global_d_all.shape[0])
                knn_d, _ = torch.topk(global_d_all, k, largest=False, dim=1)
                sigma_vec = knn_d[:, -1]  # 第 k 近邻距离（跳过自身 d=0）
                if not (sigma_vec > 0).any():
                    sigma_vec = torch.full_like(sigma_vec, 1.0)
                sigma_per_sample = sigma_vec.to(device)
                sigma = float(sigma_per_sample.mean().item())  # 仅用于观测打印
        else:
            raise ValueError(f"Unknown SIGMA_MODE: {SIGMA_MODE}")

        # σ 观测：记入 history（供训练结束时绘制 σ 变化曲线）
        if SIGMA_MODE != 'fixed':
            _sigma_stats["history"].append(sigma)

    def laplacian_matrix(X, pairwise_distances=None):
        """计算拉普拉斯矩阵，适配 GPU 计算；pairwise_distances 为已算好的 X 全对距离（可复用）"""
        compute_device = X.device
        if pairwise_distances is None:
            # MPS 后端暂不支持 cdist/diag，回退 CPU 计算再迁回原设备
            if compute_device.type == 'mps':
                X = X.cpu()
            pairwise_distances = torch.cdist(X, X, p=2)  # 使用 torch.cdist 计算欧几里得距离
        if sigma_per_sample is not None:
            # 局部带宽：W_ab = exp(-d_ab^2 / (sigma_a * sigma_b))（所有视图共享全局的局部带宽）
            s = sigma_per_sample.to(pairwise_distances.device)
            W = torch.exp(-pairwise_distances ** 2 / torch.outer(s, s))
        else:
            W = torch.exp(-pairwise_distances ** 2 / (2 * sigma ** 2))  # 计算高斯核相似度
        D = torch.diag(W.sum(dim=1))  # 计算度矩阵
        if SCORE_FORM == 'l_sym':
            # 对称归一化拉普拉斯：L = I - D^{-1/2} W D^{-1/2}（孤立点 D=0 时按 0 处理）
            d_inv_sqrt = torch.zeros_like(D)
            nz = D.diag() > 0
            d_inv_sqrt[nz, nz] = 1.0 / torch.sqrt(D.diag()[nz])
            # 注意：MPS 回退路径下 W 在 CPU 上算，eye 必须跟随 W.device（X 可能仍在原设备）
            L = torch.eye(X.shape[0], device=W.device) - d_inv_sqrt @ W @ d_inv_sqrt
            return L.to(compute_device)
        return (D - W).to(compute_device)  # 未归一化拉普拉斯矩阵

    # 全局视图拉普拉斯：优先复用 σ 分支算好的全局距离矩阵（省一次 O(n²) cdist）
    L_G = laplacian_matrix(z_all, pairwise_distances=global_d_all)
    weights = torch.zeros(len(zs), dtype=torch.float32, device=device)

    for i, z in enumerate(zs):
        L_i = laplacian_matrix(z)  # 计算每个视图的拉普拉斯矩阵
        if SCORE_FORM == 'trace' or SCORE_FORM == 'l_sym' or SCORE_FORM == 'z_norm':
            # 原始迹：同时保留结构对齐与视图能量/尺度信息
            weights[i] = torch.trace(torch.mm(L_i, L_G))
        elif SCORE_FORM == 'cosine':
            # 余弦核对齐：归一化消除嵌入尺度与图密度的影响
            norm_i = torch.linalg.matrix_norm(L_i, ord='fro')
            norm_G = torch.linalg.matrix_norm(L_G, ord='fro')
            weights[i] = torch.trace(torch.mm(L_i, L_G)) / (norm_i * norm_G)
        elif SCORE_FORM == 'distance':
            # 距离型：S_v = exp(-||L_v - L_G||_F^2 / (||L_v||_F * ||L_G||_F))，与全局偏差越大分数越低
            # 注：原式 exp(-||L_v-L_G||_F^2) 中拉普拉斯范数在真实数据上远超 1，平方后必下溢为 0
            # 导致 NaN；除以两范数乘积使指数无量纲，值域 (0,1]，语义不变
            norm_i = torch.linalg.matrix_norm(L_i, ord='fro')
            norm_G = torch.linalg.matrix_norm(L_G, ord='fro')
            weights[i] = torch.exp(-torch.linalg.matrix_norm(L_i - L_G, ord='fro') ** 2 / (norm_i * norm_G))

    total = weights.sum()
    # float() 前 detach：weights 在 autograd 图内（参与融合反传），数值防御检查无需梯度
    if not torch.isfinite(weights).all() or float(total.detach()) == 0.0:
        # 防御：数值退化（全零/NaN）时退化为均匀权重，防止 NaN 传播
        return torch.ones(len(zs), dtype=torch.float32, device=device) / len(zs)
    return weights / total  # 归一化视图权重
