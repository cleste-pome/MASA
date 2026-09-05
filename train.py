"""
=====================================================================
 MASA 训练程序 (train.py)
=====================================================================
功能：自动遍历 datasets/ 下全部 .mat，逐个数据集完成两阶段训练 + 评估 + 出图。

【不需要你填写的】直接运行 python train.py 即可；所有超参数（轮数/学习率/
  特征维度等）在 main() 的 argparse 定义处均有默认值，需要调整时改那里
  （或命令行传参，如 python train.py --pre_epochs 300）。
【输出】1.logs/ 日志、2.results_imgs/ 曲线、3.csv/ 指标与视图权重（Metrics/ + ViewWeights/）、
        4.models/ 权重、5.tsne/ 可视化。
=====================================================================
"""

# ===================== 标准库 =====================
import argparse
import gc
import os
import random
import time
from datetime import datetime
from itertools import chain

# ===================== 第三方库 =====================
import numpy as np
import torch
from sklearn.manifold import TSNE
from tabulate import tabulate  # 运行时间报告表格（与 metric.py 样式一致）
from tqdm import tqdm

# ===================== 项目内部模块 =====================
from loss import ContrastiveLoss, ae_loss_function
from utils.metric import valid
from MASA import Network
from utils.GlobalLocalManifoldCalibration import reset_sigma_history, get_sigma_history
from utils import Logger
from utils.count_datasetY import count_classes
from utils.dataloader import MATKind
from utils.device_check import detect_device
from utils.metric2csv import save_lists_to_file, find_max_weighted_sum_index, create_csv, \
    save_results_to_csv, save_wz_view_to_csv
from utils.plot import plot_acc, plot_acc_summary, plot_loss, plot_sigma
from utils.tsne_visual import plot_embeddings, plot_svg_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"  # 设置OMP_NUM_THREADS环境变量

from utils.scripts import PLOT_SIGMA, setup_seed, timing_secs, measure, print_model_summary, \
    print_timing_report, BAR_FORMAT, _kv, _fmt_ratio_list, _log_file_only

_CURRENT_PBAR = None  # 当前阶段进度条（主循环设置，训练函数内据此选择 tqdm.write 或 print 输出详情）


def pretrain(Epoch, Dataset_name, current_time):
    """AVE 预训练一个 epoch：前向 → 重建+稀疏损失 → 反向更新，返回本轮平均损失。

    :param Epoch: 当前轮次（从 0 开始，用于 t-SNE 输出与进度显示）
    :param Dataset_name: 数据集名（t-SNE 输出路径与分段计时分组用）
    :param current_time: 数据集级时间戳（输出文件命名）
    :return: 本轮平均损失（全局 AE + 各视图 AE）
    每轮详情经当前阶段进度条（_CURRENT_PBAR）用 tqdm.write 输出；无进度条时用 print。
    """
    pbar = _CURRENT_PBAR  # 当前阶段进度条，None 表示没有（如日志重定向场景）
    tot_loss = 0.  # 初始化总损失
    tot_global_ae = 0.  # 全局 AE（重建+稀疏）分量累计
    tot_view_ae = 0.  # 各视图 AE（重建+稀疏）分量累计
    # 遍历数据集，enumerate用于获取批次索引和数据
    for batch_idx, (xs, gnd, _) in enumerate(data_loader):
        loss_list = []  # 每 batch 清空损失列表，避免跨 batch 累积使 loss/梯度虚高
        # 将数据从字典中提取并按键的顺序转换为张量列表
        xs_dict2tensors = [xs[key] for key in sorted(xs.keys())]
        # 将所有视角的数据拼接在一起，形成一个大的张量，用于计算整体的重建误差
        xs2one = torch.cat(xs_dict2tensors, dim=1)
        if Epoch == 0:
            with measure(f"{Dataset_name}: t-SNE visualization"):
                embeddings = TSNE(n_components=2, init='pca', random_state=42).fit_transform(
                    xs2one.detach().cpu())  # TODO tensor
                fig_svg = plot_svg_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', Epoch,
                                              Dataset_name)  # TODO plot t-sne svg
                fig = plot_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', Epoch,
                                      Dataset_name)  # TODO plot t-sne pdf
                tqdm.write(f'1.SVG_path:{fig_svg} 2.PDF_path:{fig}')
        # 将每个视角的数据移动到指定设备上（例如，GPU），以便加速计算
        for v in range(view):
            xs[v] = xs[v].to(device)
        # 清空优化器中的梯度
        optimizer.zero_grad()
        # 前向传播：通过模型计算重建后的输入、隐藏表示和其他中间结果
        xrs, zs, rs, H, xr_all, z_all, activation, means, wz_view = model(xs)
        save_wz_view_to_csv(wz_view, Epoch + 1, f'3.csv/ViewWeights/{Dataset_name}_{current_time}')
        # 计算平均值
        mean_average = sum(means) / len(means)
        # TODO pre 1 全局视角
        loss_list.append(
            ae_loss_function(mean_average, xs2one.to(device), xr_all.to(device), activation[0], rho=0.05, beta=1.0))
        # if Epoch == 299:
        if (Epoch + 1) % 100 == 0:
            with measure(f"{Dataset_name}: t-SNE visualization"):
                embeddings = TSNE(n_components=2, init='pca', random_state=42).fit_transform(
                    z_all.detach().cpu())  # TODO 绘制前期融合编码后的全局矩阵
                fig_svg = plot_svg_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', Epoch,
                                              Dataset_name + f'_EarlyFusion')
                fig = plot_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', Epoch, Dataset_name + f'_EarlyFusion')
                tqdm.write(f'1.SVG_path:{fig_svg} 2.PDF_path:{fig}')

        # TODO pre 2 局部视角
        for v in range(view):
            # TODO 视图降维可视化
            # if Epoch == 299:
            # if (Epoch + 1) % 100 == 0:
            #     embeddings = TSNE(n_components=2, init='pca', random_state=42).fit_transform(
            #         zs[v].detach().cpu())  # TODO tensor
            #     fig_svg = plot_svg_embeddings(embeddings, gnd, F'5.tsne/{Dataset_name}/', epoch,
            #                                   Dataset_name + f'_{v + 1}')  # TODO plot t-sne svg
            #     fig = plot_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}/', epoch,
            #                           Dataset_name + f'_{v + 1}')  # TODO plot t-sne pdf
            #     print(f'1.SVG_path:{fig_svg} 2.PDF_path:{fig}')
            loss_list.append(ae_loss_function(means[v], xs[v], xrs[v], activation[v + 1], rho=0.05, beta=1.0))
        # 汇总所有视角的损失
        loss = sum(loss_list)
        # 反向传播计算梯度
        loss.backward()
        # 使用优化器更新模型参数
        optimizer.step()
        # 累加损失，用于计算当前Epoch的平均损失（loss_list[0]=全局 AE，其余=各视图 AE）
        tot_loss += loss.item()
        tot_global_ae += loss_list[0].item()
        tot_view_ae += sum(loss_list[1:]).item()
    # 计算当前轮次的平均损失与各分量，详情另起行打印（进度条下方滚动，条本身保持干净）
    pretrain_loss = tot_loss / len(data_loader)
    global_ae = tot_global_ae / len(data_loader)
    view_ae = tot_view_ae / len(data_loader)
    breakdown = f"loss = {pretrain_loss:.4f} (global_ae {global_ae:.4f} + view_ae {view_ae:.4f})"
    if pbar is not None:
        tqdm.write(breakdown)
        tqdm.write(f"  | sparsity {_fmt_ratio_list(means)}")
        tqdm.write(f"  | view weights {_fmt_ratio_list(wz_view.tolist(), 3)}")
    else:
        print(f'Pre  Epoch [{Epoch + 1}] {breakdown}')
        print(f'     | sparsity {_fmt_ratio_list(means)}')
        print(f'     | view weights {_fmt_ratio_list(wz_view.tolist(), 3)}')
    # 返回本轮总损失与分量（global_ae + view_ae），供分量分解图使用
    return pretrain_loss, global_ae, view_ae


def contrastive_train(Epoch, Dataset_name, Total_epochs, Plot_SDD, current_time):
    """一致性训练一个 epoch：前向 → 重建+稀疏+对比损失 → 反向更新，返回本轮平均损失。

    :param Epoch: 当前轮次（全局轮数，在 pre 阶段之后续算）
    :param Dataset_name: 数据集名（视图权重 CSV 路径与分段计时分组用）
    :param Total_epochs: pre+con 总轮数（进度显示与最后一轮判断用）
    :param Plot_SDD: 最后一轮触发特征分离图（当前为预留开关，未启用）
    :param current_time: 数据集级时间戳（输出文件命名）
    :return: 本轮平均损失（全局 AE + 各视图 AE + 各视图对比损失）
    每轮详情经当前阶段进度条（_CURRENT_PBAR）用 tqdm.write 输出；无进度条时用 print。
    """
    pbar = _CURRENT_PBAR  # 当前阶段进度条，None 表示没有（如日志重定向场景）
    tot_loss = 0.  # 初始化总损失
    tot_global_ae = 0.  # 全局 AE（重建+稀疏）分量累计
    tot_view_ae = 0.  # 各视图 AE（重建+稀疏）分量累计
    tot_con = 0.  # 各视图对比损失分量累计
    for batch_idx, (xs, gnd, _) in enumerate(data_loader):  # 遍历数据集
        for v in range(view):
            xs[v] = xs[v].to(device)  # 将数据移动到指定设备（如GPU）
        optimizer.zero_grad()  # 清空梯度
        xrs, zs, rs, H, xr_all, z_all, activation, means, wz_view = model(xs)  # TODO 2.前向传播，获取重建后的输入、编码特征、视角一致特征和全局特征
        save_wz_view_to_csv(wz_view, Epoch + 1, f'3.csv/ViewWeights/{Dataset_name}_{current_time}')
        # TODO C.3 视图权重, wz_view
        # TODO 降维可视化
        # if Epoch == Total_epochs - 1:
        # if (Epoch + 1) % 100 == 0:
        #     embeddings = TSNE(n_components=2, init='pca', random_state=42).fit_transform(
        #         H.detach().cpu())  # TODO tensor
        #     fig_svg = plot_svg_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}/', epoch,
        #                                   Dataset_name + f'_LateFusion')  # TODO plot t-sne svg
        #     fig = plot_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}/', epoch,
        #                           Dataset_name + f'_LateFusion')  # TODO plot t-sne pdf
        #     print(f'1.SVG_path:{fig_svg} 2.PDF_path:{fig}')

        loss_list = []
        # if Plot_SDD:
        #     xs_list = list(xs.values())
        #     names = ['xs', 'xrs', 'zs', 'rs', 'H', 'xr_all', 'z_all']
        #     feature_separation([xs_list, xrs, zs, rs, H, xr_all, z_all], names, Dataset_name)
        # TODO C.0.0 w2: 每个视角权重取平均
        w2 = []
        for v in range(view):
            w_v = 1 / view
            w2.append(w_v)  # 将 w_v 放入一个列表中，然后再添加到 w 中
        w2 = torch.tensor(w2).to(device)
        xs_dict2tensors = [xs[key] for key in sorted(xs.keys())]
        xs2one = torch.cat(xs_dict2tensors, dim=1)
        mean_average = sum(means) / len(means)
        loss_list.append(
            ae_loss_function(mean_average, xs2one.to(device), xr_all.to(device), activation[0], rho=0.05, beta=1.0))
        for v in range(view):
            loss_list.append(ae_loss_function(means[v], xs[v], xrs[v], activation[v + 1], rho=0.05, beta=1.0))
            # TODO 自加权对比学习损失
            loss_list.append(contrastiveloss(H, rs[v], w2[v]))  # TODO 计算对比损失
        loss = sum(loss_list)  # 汇总所有视角的损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        # 分量累计：loss_list[0]=全局 AE，奇数位=各视图 AE，偶数位=各视图对比损失
        tot_loss += loss.item()  # 累加损失
        tot_global_ae += loss_list[0].item()
        tot_view_ae += sum(loss_list[1::2]).item()
        tot_con += sum(loss_list[2::2]).item()
    con_loss = tot_loss / len(data_loader)
    global_ae = tot_global_ae / len(data_loader)
    view_ae = tot_view_ae / len(data_loader)
    con = tot_con / len(data_loader)
    breakdown = f"loss = {con_loss:.4f} (global_ae {global_ae:.4f} + view_ae {view_ae:.4f} + contrastive {con:.4f})"
    if pbar is not None:
        tqdm.write(breakdown)
        tqdm.write(f"  | sparsity {_fmt_ratio_list(means)}")
        tqdm.write(f"  | view weights {_fmt_ratio_list(wz_view.tolist(), 3)}")
    else:
        print(f'Con  Epoch [{Epoch + 1}/{Total_epochs}] {breakdown}')
        print(f'     | sparsity {_fmt_ratio_list(means)}')
        print(f'     | view weights {_fmt_ratio_list(wz_view.tolist(), 3)}')
    return con_loss


if __name__ == '__main__':
    # loop in data
    folder_path = f"datasets"  # TODO 数据集文件夹地址
    # 只遍历 .mat 数据集文件（过滤 .DS_Store 等无关文件，保证 data_iter 序号连续）
    file_names = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
    # 设备前置检查：运行前探测环境并决定 CUDA/MPS/CPU 配置（可用 MASA_DEVICE 环境变量强制指定）
    with measure("Device check & startup"):
        device, _ = detect_device()
    t_start_all = time.perf_counter()  # 总计时起点（含全部数据集，不含 torch import）
    logger = None  # 数据集循环内会赋值，供结尾计时报告写入日志
    data_iter = 1  # 数据集位次
    for Dataname in file_names:
        if Dataname.endswith(".mat"):
            Dataname = Dataname[:-4]
            parser = argparse.ArgumentParser(description='train') # 超参数
            parser.add_argument('--dataset', default=Dataname)
            parser.add_argument('--batch_size', default=256, type=int)
            parser.add_argument("--learning_rate", type=float, default=0.0003)
            parser.add_argument("--pre_epochs", type=int, default=300)  # 300
            parser.add_argument("--con_epochs", type=int, default=300)  # 300/600
            parser.add_argument("--feature_dim", type=int, default=64)
            parser.add_argument("--high_feature_dim", type=int, default=20)
            parser.add_argument("--seed", type=int, default=42)
            parser.add_argument("--iter", type=int, default=1)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            # TODO 选取noise ratio比例的样本，随机(1到view-1)个视图做添加高斯噪声处理
            parser.add_argument('--noise_ratio', type=float, default=0.0)
            # TODO 选取conflict ratio比例的样本，随机选择一个视图的数据用另一个类别的样本的同视图数据替换
            parser.add_argument('--conflict_ratio', type=float, default=0.0)
            # TODO 选取missing ratio比例样本的随机(1到view-1)个视图做缺失处理
            parser.add_argument('--missing_ratio', type=float, default=0.0)
            # TODO 选取sparsity ratio比例维度的随机(1到dims-1)个维度做置0处理
            parser.add_argument('--sparsity_ratio', type=float, default=0.0)
            args = parser.parse_args()
            # TODO log创建
            log_path = f'1.logs'
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            data_ratio = f'{args.noise_ratio}_{args.conflict_ratio}_{args.missing_ratio}'
            # 数据集级时间戳：本数据集所有输出（日志/曲线/指标/权重/可视化）命名统一
            current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
            logger = Logger.get_logger(__file__, Dataname, data_ratio, current_time)
            with measure(f"{Dataname}: Data load & preprocessing"):
                dataset = MATKind(args.dataset, folder_path)
                count_classes(Dataname, dataset.Y)  # TODO 统计类别数量分布情况（是否长尾分布）

                # 获取数据集中类别的数量
                class_num = dataset.num_classes
                # 获取数据集中样本的总数
                data_size = len(dataset)
                # 获取数据集中视图的数量
                view = dataset.num_views
                # 获取每个视图的维度
                dims = list(chain.from_iterable(dataset.dims.tolist()))
                print(f"\n[Data] Dataset info")
                print(f"  {_kv('samples', data_size)}{_kv('views', view)}{_kv('classes', class_num)}")
                print(f"  {_kv('view dims', str(dims))}")
                print("-" * 72)

                index = np.arange(data_size)
                np.random.shuffle(index)
                # TODO batch size
                if Dataname == 'NUSWIDEOBJ':
                    args.batch_size = 256
                else:
                    args.batch_size = data_size
                # 数据预处理
                dataset.postprocessing(index,
                                       addNoise=True, sigma=0.5, ratio_noise=args.noise_ratio,
                                       addConflict=True, ratio_conflict=args.conflict_ratio,
                                       addMissing=True, missing_rate=args.missing_ratio,
                                       addSparsity=True, sparsity_rate=args.sparsity_ratio)
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True)
            pth_path = f'4.models'
            if not os.path.exists(f'./{pth_path}'):
                os.makedirs(f'./{pth_path}')
            acc_l, nmi_l, pur_l, ari_l, seed_l, lr_l, loss_l = [], [], [], [], [], [], []
            T = args.iter  # 循环测试次数，用于获取更准确地评价指标（平均值和方差）
            seed = args.seed  # 首轮用 --seed 固定值；--iter > 1 时每轮末尾随机扰动，种子序列确定可复现
            lr = args.learning_rate
            imgs_path = f'2.results_imgs/{Dataname}_{current_time}'
            reset_sigma_history()  # σ 历史按数据集清零（曲线按数据集分开画，非 fixed 模式才有数据）
            for i in range(T):
                print(f"\n[Hyperparams] Config (ROUND {i + 1}/{T})")
                print(f"  {_kv('seed', seed)}{_kv('lr', lr)}{_kv('iter', args.iter)}")
                print(f"  {_kv('batch_size', args.batch_size)}{_kv('pre_epochs', args.pre_epochs)}"
                      f"{_kv('con_epochs', args.con_epochs)}")
                print(f"  {_kv('feature_dim', args.feature_dim)}{_kv('high_feature_dim', args.high_feature_dim)}"
                      f"{_kv('weight_decay', args.weight_decay)}")
                print(f"  {_kv('noise', args.noise_ratio)}{_kv('conflict', args.conflict_ratio)}"
                      f"{_kv('missing', args.missing_ratio)}")
                print(f"  {_kv('sparsity', args.sparsity_ratio)}")
                print("-" * 72)
                # 确定本次循环测试的随机数种子：1.固定每次结果 2.保证不同次结果不同
                setup_seed(seed)
                seed_l.append(seed)
                lr_l.append(lr)
                # 建保存评价指标的列表
                acc_list, nmi_list, pur_list, ari_list, preloss_list, conloss_list = [], [], [], [], [], []
                pre_global_ae_list, pre_view_ae_list = [], []  # 预训练损失分量（全局 AE / 各视图 AE）
                valid_loss_list = []  # 每个验证点对应的当轮总损失（与 acc_list 等长，CSV 保存用）
                epoch_ticks = []  # 每个评价点对应的真实 epoch（pre 后接 con 续算，出图横坐标用）
                # TODO 重点来了੭ ᐕ)੭: model
                with measure(f"{Dataname}: Model construction"):
                    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device)
                    if i == 0:
                        print_model_summary(model)  # 网络结构每个数据集只打印一次（各轮结构相同）
                    model = model.to(device)
                    state = model.state_dict()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
                    # contrastiveloss = ContrastiveLoss(args.batch_size, device).to(device)
                    # TODO 初始化损失函数（选择不同的对比损失վ'ᴗ' ի）
                    contrastiveloss = ContrastiveLoss(batch_size=args.batch_size, device=device, loss_type="classical")
                    metric_csv_path, metric_csv_name = create_csv(Dataname, data_ratio, view, current_time)
                    print(f'Metrics csv file has been created: {metric_csv_path}')

                # TODO 调整计算评价指标的轮数间隔，valid_check_num有条件的话最好设置为1
                if data_size >= 2500:  # large
                    args.con_epochs = 600  # small/large 300/600
                    pre_check_num = 100
                    valid_check_num = 10
                else:  # small
                    pre_check_num = 10
                    valid_check_num = 1

                with measure(f"{Dataname}: Pretraining"):
                    print(f'---------------------------------------{Dataname}[{data_iter}]---------------------------------------')
                    print(f"\n[Train] Pretrain stage ({args.pre_epochs} epochs)")
                    pbar = tqdm(total=args.pre_epochs, desc="Pretrain", unit="epoch", bar_format=BAR_FORMAT)
                    _CURRENT_PBAR = pbar  # 训练函数内据此用 tqdm.write 输出详情
                    for epoch in range(args.pre_epochs):
                        preloss, pre_global_ae, pre_view_ae = pretrain(epoch, Dataname, current_time)  # 1.pre-train
                        preloss_list.append(preloss)
                        pre_global_ae_list.append(pre_global_ae)  # 分量：全局 AE（重建+稀疏）
                        pre_view_ae_list.append(pre_view_ae)  # 分量：各视图 AE（重建+稀疏）
                        if (epoch + 1) % pre_check_num == 0:  # TODO pre_check_num 1. pre
                            with measure(f"{Dataname}: Validation (KMeans)"):
                                acc, nmi, pur, ari, zs_Results = valid(model, device, dataset, view, data_size,
                                                                       class_num, pre_train=True,
                                                                       con_train=False)
                            save_results_to_csv(zs_Results, epoch + 1, metric_csv_path, view)
                            # 将本轮pre_epochs评价指标添加到列表中
                            epoch_ticks.append(epoch + 1)
                            acc_list.append(acc)
                            nmi_list.append(nmi)
                            pur_list.append(pur)
                            ari_list.append(ari)
                            valid_loss_list.append(preloss)  # 验证点对应的当轮总损失
                        pbar.update(1)
                    pbar.close()
                    _CURRENT_PBAR = None
                plot_loss(imgs_path, preloss_list, Dataname, 'pretrain_loss', args.pre_epochs + args.con_epochs,
                          components={'global_ae': pre_global_ae_list, 'view_ae': pre_view_ae_list})

                with measure(f"{Dataname}: Consistency training"):
                    print(f"\n[Train] Consistency stage ({args.con_epochs} epochs)")
                    pbar = tqdm(total=args.con_epochs, desc="Consistency", unit="epoch", bar_format=BAR_FORMAT)
                    _CURRENT_PBAR = pbar  # 训练函数内据此用 tqdm.write 输出详情
                    for epoch in range(args.con_epochs):
                        epoch = args.pre_epochs + epoch
                        plot_SDD = False
                        total_epochs = args.pre_epochs + args.con_epochs
                        if epoch + 1 == total_epochs:
                            plot_SDD = True
                        conloss = contrastive_train(epoch, Dataname, total_epochs, plot_SDD, current_time)  # 2.contrastive train
                        conloss_list.append(conloss)
                        # TODO valid_check_num 2. con
                        if (epoch + 1) % valid_check_num == 0:  # TODO con
                            with measure(f"{Dataname}: Validation (KMeans)"):
                                acc, nmi, pur, ari, rs_Results = valid(model, device, dataset, view, data_size,
                                                                       class_num, pre_train=False,
                                                                       con_train=True)
                            save_results_to_csv(rs_Results, epoch + 1, metric_csv_path, view)
                            epoch_ticks.append(epoch + 1)
                            acc_list.append(acc)
                            nmi_list.append(nmi)
                            pur_list.append(pur)
                            ari_list.append(ari)
                            valid_loss_list.append(conloss)  # 验证点对应的当轮总损失
                        max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                                acc_weight=0.25, nmi_weight=0.25,
                                                                pur_weight=0.25, ari_weight=0.25)
                        pbar.update(1)
                    pbar.close()
                    _CURRENT_PBAR = None
                # 一致性阶段总损失 = global_ae + view_ae + contrastive（画总损失曲线，命名为 co-training loss 表明是协同训练总损失，避免与对比损失混淆）
                plot_loss(imgs_path, conloss_list, Dataname, 'co-training_loss', args.pre_epochs + args.con_epochs)
                loss_list = preloss_list + conloss_list
                # TODO 1.保存最后次最后一轮的权重文件(.pth)
                state = model.state_dict()
                pth_path_meta = f'{pth_path}/' + f'{Dataname}'
                if not os.path.exists(pth_path_meta):
                    os.makedirs(pth_path_meta)
                model_path = f'{pth_path_meta}/{Dataname}_{current_time}.pth'
                torch.save(state, model_path)
                print(f'Model(.pth) has been saved at {model_path}')
                # TODO 最后一轮
                info = {"dataset": Dataname,
                        "iter": i + 1,
                        "Last Epoch": epoch_ticks[-1],
                        "acc": acc_list[-1],
                        "Nmi": nmi_list[-1],
                        "Purity": pur_list[-1],
                        "ari": ari_list[-1],
                        "seed": seed,
                        "learning rate": lr}
                # log save（只写日志文件，终端不显示字典；一句提示保存位置）
                _log_file_only(logger, str(info))
                del info
                acc_l.append(acc_list)
                nmi_l.append(nmi_list)
                pur_l.append(pur_list)
                ari_l.append(ari_list)
                loss_l.append(loss_list)
                max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                        acc_weight=0.25, nmi_weight=0.25,
                                                        pur_weight=0.25, ari_weight=0.25)
                # TODO 2.最好一轮(不建议这样做，除非你有early stop的理由)
                info = {"dataset": Dataname,
                        "iter": i + 1,
                        "MAX Epoch": epoch_ticks[max_index],
                        "acc": acc_list[max_index],
                        "Nmi": nmi_list[max_index],
                        "Purity": pur_list[max_index],
                        "ari": ari_list[max_index],
                        "seed": seed,
                        "learning rate": lr}
                log_path_res = _log_file_only(logger, str(info))
                print(f'Results saved to log: {log_path_res}')
                # 本轮最优结果（加权 0.25×4 选出，与日志 MAX Epoch 记录一致）
                print(f'Max metric: epoch{epoch_ticks[max_index]}\n'
                      f'1.acc:{acc_list[max_index] * 100:.2f}%\n'
                      f'2.nmi:{nmi_list[max_index] * 100:.2f}%\n'
                      f'3.pur:{pur_list[max_index] * 100:.2f}%\n'
                      f'4.ari:{ari_list[max_index] * 100:.2f}%')
                # 本轮出图与保存：曲线/指标 CSV（单个文件不再逐条打印，一句提示目录位置）
                with measure(f"{Dataname}: Plots & saving"):
                    plot_acc(imgs_path, acc_list, Dataname, 'acc', epoch_ticks, args.pre_epochs)
                    plot_acc(imgs_path, nmi_list, Dataname, 'nmi', epoch_ticks, args.pre_epochs)
                    plot_acc(imgs_path, pur_list, Dataname, 'pur', epoch_ticks, args.pre_epochs)
                    plot_acc(imgs_path, ari_list, Dataname, 'ari', epoch_ticks, args.pre_epochs)
                    plot_acc_summary(imgs_path,
                                     {'acc': acc_list, 'nmi': nmi_list, 'pur': pur_list, 'ari': ari_list},
                                     Dataname, epoch_ticks, args.pre_epochs)
                    save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, valid_loss_list, Dataname, data_ratio,
                                       epoch_ticks, current_time)
                    # ELMC σ 变化曲线（默认不绘制，PLOT_SIGMA 置 True 开启；'fixed' 模式无历史数据自动跳过）
                    if PLOT_SIGMA:
                        plot_sigma(get_sigma_history(), imgs_path, Dataname)
                    print(f'Curves saved to {imgs_path}')
                offset1 = 100000
                seed = int(abs(seed + random.uniform(-offset1, offset1)))
                offset2 = 0.0001
                lr = abs(lr + random.uniform(-offset2, offset2))
                lr = "{:.5f}".format(lr)
                lr = float(lr)
                del info
            if T > 1:
                # 多轮结果汇总表：每轮最优指标（加权 0.25/0.25/0.25/0.25）+ 均值/标准差
                headers = ["ROUND", "seed", "lr", "ACC%", "NMI%", "PUR%", "ARI%"]
                rows = []
                bests = []  # 每轮最优 (acc, nmi, pur, ari) 百分比，用于最后算均值/标准差
                for i in range(T):
                    mi = find_max_weighted_sum_index(acc_l[i], nmi_l[i], pur_l[i], ari_l[i],
                                                     acc_weight=0.25, nmi_weight=0.25,
                                                     pur_weight=0.25, ari_weight=0.25)
                    acc, nmi, pur, ari = acc_l[i][mi] * 100, nmi_l[i][mi] * 100, \
                        pur_l[i][mi] * 100, ari_l[i][mi] * 100
                    bests.append((acc, nmi, pur, ari))
                    rows.append([i + 1, seed_l[i], lr_l[i],
                                 f"{acc:.2f}", f"{nmi:.2f}", f"{pur:.2f}", f"{ari:.2f}"])
                mean = [f"{np.mean([b[k] for b in bests]):.2f}" for k in range(4)]
                std = [f"{np.std([b[k] for b in bests]):.2f}" for k in range(4)]
                rows.append(["Mean", "", "", *mean])
                rows.append(["Std", "", "", *std])
                print(f"\n===== Multi-round summary (iter={T}; per-round best by weighted metrics) =====")
                print(tabulate(rows, headers=headers, tablefmt="grid"))
            # 显式删除变量
            del dataset
            # 手动调用垃圾回收
            gc.collect()
        data_iter += 1
    # —— 全部数据集运行结束：记录并打印分段运行时间 ——
    timing_secs["Total time"] = time.perf_counter() - t_start_all
    print_timing_report(logger)
