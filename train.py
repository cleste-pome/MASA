"""
=====================================================================
 MASA 训练程序 (train.py)
=====================================================================
功能：自动遍历 datasets/ 下全部 .mat，逐个数据集完成两阶段训练 + 评估 + 出图。

【不需要你填写的】直接运行 python train.py 即可；所有超参数（轮数/学习率/
  特征维度等）在 main() 的 argparse 定义处均有默认值，需要调整时改那里
  （或命令行传参，如 python train.py --pre_epochs 300）。
【输出】1.logs/ 日志、2.results_imgs/ 曲线、3.csv/ 指标、4.models/ 权重、
        5.tsne/ 可视化、6.ViewWeights/ 视图权重。
=====================================================================
"""

# ===================== 标准库 =====================
import argparse
import gc
import os
import random
import time
import warnings
from contextlib import contextmanager
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
from utils.metric2csv import save_lists_to_file, find_max_weighted_sum_index, find_max_last_element_index, create_csv, \
    save_results_to_csv, save_wz_view_to_csv
from utils.plot import plot_acc, plot_loss, plot_sigma
from utils.tsne_visual import plot_embeddings, plot_svg_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"  # 设置OMP_NUM_THREADS环境变量
print(f'1.torch version:{torch.__version__} 2.cuda available:{torch.cuda.is_available()}')
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_seed(Seed):
    torch.manual_seed(Seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed_all(Seed)  # 为所有GPU设置随机种子
    np.random.seed(Seed)  # 为NumPy设置随机种子
    random.seed(Seed)  # 为Python标准库的random模块设置随机种子
    torch.backends.deterministic = True  # 确保CUDA的确定性（即每次运行结果一致）


# ===================== 运行分段计时（训练结束统一打印/记录） =====================
timing_secs = {}  # 分段名 -> 累计秒数（跨数据集、跨 iter 轮次累加）


@contextmanager
def measure(section):
    """with measure('段名'): 自动把整段耗时（秒）累加到 timing_secs['段名']，可重复累加。
    用于记录各阶段运行时间：数据加载/模型构建/预训练/一致性训练/验证/绘图等。"""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing_secs[section] = timing_secs.get(section, 0.0) + time.perf_counter() - t0


def _fmt_duration(seconds):
    """秒数 → 可读文本（不足 1 分钟显示秒，超过显示 分/时）"""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} min {sec:.1f} s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {sec:.1f} s"


def _fmt_duration_compact(seconds):
    """秒数 → 表格用紧凑文本（2.4s / 29m18s / 1h42m）"""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{int(sec)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m"


def print_timing_report(logger=None):
    """训练结束时打印运行时间报告（表格形式，按数据集分组），并同步写入日志文件记录。

    说明：验证(KMeans)/t-SNE 是阶段（预训练/一致性训练）内部的细分项，已包含在所属
    阶段耗时中，单独列出便于定位瓶颈；极小未细分开销未计入。"""
    lines = ["=" * 60, "运行时间报告（分段计时）", "=" * 60]
    lines.append(f"总耗时（所有数据集合计）: {_fmt_duration(timing_secs.get('总耗时', 0.0))}"
                 + (f" ｜ 设备检查与启动: {_fmt_duration(timing_secs['设备检查与启动'])}"
                    if "设备检查与启动" in timing_secs else ""))
    phases = ("数据加载与预处理", "模型构建", "预训练", "一致性训练", "出图与保存")
    headers = ["数据集", "数据加载", "模型构建", "预训练", "一致性训练",
               "其中: 验证", "其中: t-SNE", "出图保存", "阶段合计"]
    rows = []
    datasets = sorted({k.split(':')[0] for k in timing_secs if ':' in k})
    for d in datasets:
        rows.append([
            d,
            _fmt_duration_compact(timing_secs.get(f"{d}: 数据加载与预处理", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: 模型构建", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: 预训练", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: 一致性训练", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: 验证评估(KMeans)", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: t-SNE 可视化", 0.0)),
            _fmt_duration_compact(timing_secs.get(f"{d}: 出图与保存", 0.0)),
            _fmt_duration_compact(sum(timing_secs.get(f"{d}: {p}", 0.0) for p in phases)),
        ])
    if rows:
        lines.append(tabulate(rows, headers=headers, tablefmt="grid"))
    lines.append("注: 验证(KMeans)/t-SNE 为阶段内部细分项，已含在预训练/一致性训练耗时内，单列便于定位瓶颈。")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print(text)  # 控制台只打印一次
    if logger is not None:
        # 直接写入当前数据集的日志文件（不走 logger.info，避免控制台重复输出）
        for handler in logger.handlers:
            report_path = getattr(handler, "baseFilename", None)
            if report_path:
                try:
                    with open(report_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 运行时间报告\n{text}\n")
                except OSError:
                    pass
                break


def pretrain(Epoch, Dataset_name, current_time):
    tot_loss = 0.  # 初始化总损失
    loss_list = []  # 用于存储每个视角的损失
    # 遍历数据集，enumerate用于获取批次索引和数据
    for batch_idx, (xs, gnd, _) in enumerate(data_loader):
        # 将数据从字典中提取并按键的顺序转换为张量列表
        xs_dict2tensors = [xs[key] for key in sorted(xs.keys())]
        # 将所有视角的数据拼接在一起，形成一个大的张量，用于计算整体的重建误差
        xs2one = torch.cat(xs_dict2tensors, dim=1)
        if Epoch == 0:
            with measure(f"{Dataset_name}: t-SNE 可视化"):
                embeddings = TSNE(n_components=2, init='pca', random_state=42).fit_transform(
                    xs2one.detach().cpu())  # TODO tensor
                fig_svg = plot_svg_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', epoch,
                                              Dataset_name)  # TODO plot t-sne svg
                fig = plot_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', epoch,
                                      Dataset_name)  # TODO plot t-sne pdf
                print(f'1.SVG_path:{fig_svg} 2.PDF_path:{fig}')
        # 将每个视角的数据移动到指定设备上（例如，GPU），以便加速计算
        for v in range(view):
            xs[v] = xs[v].to(device)
        # 清空优化器中的梯度
        optimizer.zero_grad()
        # 前向传播：通过模型计算重建后的输入、隐藏表示和其他中间结果
        xrs, zs, rs, H, xr_all, z_all, activation, means, wz_view = model(xs)
        save_wz_view_to_csv(wz_view, Epoch + 1, f'6.ViewWeights/{Dataset_name}/{Dataset_name}_{current_time}')
        # 计算平均值
        mean_average = sum(means) / len(means)
        # TODO pre 1 全局视角
        loss_list.append(
            ae_loss_function(mean_average, xs2one.to(device), xr_all.to(device), activation[0], rho=0.05, beta=1.0))
        # if Epoch == 299:
        if (Epoch + 1) % 100 == 0:
            with measure(f"{Dataset_name}: t-SNE 可视化"):
                embeddings = TSNE(n_components=2, init='pca', random_state=42).fit_transform(
                    z_all.detach().cpu())  # TODO 绘制前期融合编码后的全局矩阵
                fig_svg = plot_svg_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', epoch,
                                              Dataset_name + f'_EarlyFusion')
                fig = plot_embeddings(embeddings, gnd, f'5.tsne/{Dataset_name}_{current_time}/', epoch, Dataset_name + f'_EarlyFusion')
                print(f'1.SVG_path:{fig_svg} 2.PDF_path:{fig}')

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
        # 累加损失，用于计算当前Epoch的平均损失
        tot_loss += loss.item()
    # 计算并打印当前轮次的平均损失
    pretrain_loss = tot_loss / len(data_loader)
    print('Pre Epochs[{}]'.format(Epoch + 1), 'Loss:{:.6f}'.format(pretrain_loss))
    # 返回当前轮次的平均损失和每个视角的权重
    return pretrain_loss


def contrastive_train(Epoch, Dataset_name, Total_epochs, Plot_SDD, current_time):
    """
    CVDA：基于对比的视图级分布对齐训练过程
    :param Epoch: 当前的训练轮次
    Plot_SDD： 我发明的维度分布蜡烛图:D文章还在写（鸽子咕咕咕）
    """
    tot_loss = 0.  # 初始化总损失
    for batch_idx, (xs, gnd, _) in enumerate(data_loader):  # 遍历数据集
        for v in range(view):
            xs[v] = xs[v].to(device)  # 将数据移动到指定设备（如GPU）
        optimizer.zero_grad()  # 清空梯度
        xrs, zs, rs, H, xr_all, z_all, activation, means, wz_view = model(xs)  # TODO 2.前向传播，获取重建后的输入、编码特征、视角一致特征和全局特征
        save_wz_view_to_csv(wz_view, Epoch + 1, f'6.ViewWeights/{Dataset_name}/{Dataset_name}_{current_time}')
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
        tot_loss += loss.item()  # 累加损失
    con_loss = tot_loss / len(data_loader)
    print('Con Epochs[{}/{}]'.format(Epoch + 1, Total_epochs), 'Loss:{:.6f}'.format(con_loss))  # 输出当前轮次的平均损失
    return con_loss


if __name__ == '__main__':
    # loop in data
    folder_path = "datasets"  # TODO 数据集文件夹地址
    # 只遍历 .mat 数据集文件（过滤 .DS_Store 等无关文件，保证 data_iter 序号连续）
    file_names = [f for f in os.listdir(folder_path) if f.endswith(".mat")]
    # 设备前置检查：运行前探测环境并决定 CUDA/MPS/CPU 配置（可用 MASA_DEVICE 环境变量强制指定）
    with measure("设备检查与启动"):
        device, _ = detect_device()
    t_start_all = time.perf_counter()  # 总计时起点（含全部数据集，不含 torch import）
    logger = None  # 数据集循环内会赋值，供结尾计时报告写入日志
    data_iter = 1  # 数据集位次
    for Dataname in tqdm(file_names):
        if Dataname.endswith(".mat"):
            Dataname = Dataname[:-4]
            print(
                f'---------------------------------------{Dataname}[{data_iter}]---------------------------------------')
            parser = argparse.ArgumentParser(description='train')
            parser.add_argument('--dataset', default=Dataname)
            # 超参数
            parser.add_argument('--batch_size', default=256, type=int)
            parser.add_argument("--learning_rate", type=float, default=0.0003)
            parser.add_argument("--pre_epochs", type=int, default=300)  # 300
            parser.add_argument("--con_epochs", type=int, default=1000)  # 300/600
            parser.add_argument("--iter", type=int, default=1)
            parser.add_argument("--feature_dim", type=int, default=64)
            parser.add_argument("--high_feature_dim", type=int, default=20)
            parser.add_argument("--seed", type=int, default=50)
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
            with measure(f"{Dataname}: 数据加载与预处理"):
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
                print(
                    f'dataset information: 1. number of data: {data_size}, 2.views: {view}, 3.clusters: {class_num}, 4.each view: {dims}')

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
            # 默认验证间隔，循环外（如 --iter 0）不会因 valid_check_num 未定义而崩
            pre_check_num, valid_check_num = 100, 10
            T = args.iter  # 循环测试次数，用于获取更准确地评价指标（平均值和方差）
            seed = args.seed
            lr = args.learning_rate
            imgs_path = f'2.results_imgs/{Dataname}_{current_time}'
            reset_sigma_history()  # σ 历史按数据集清零（曲线按数据集分开画，非 fixed 模式才有数据）
            for i in range(T):
                print(f"ROUND:{i + 1}[seed:{seed}][learning rate:{lr}]")
                # 确定本次循环测试的随机数种子：1.固定每次结果 2.保证不同次结果不同
                setup_seed(seed)
                seed_l.append(seed)
                lr_l.append(lr)
                # 建保存评价指标的列表
                acc_list, nmi_list, pur_list, ari_list, preloss_list, conloss_list = [], [], [], [], [], []
                epoch_ticks = []  # 每个评价点对应的真实 epoch（pre 后接 con 续算，出图横坐标用）
                # TODO 重点来了੭ ᐕ)੭: model
                with measure(f"{Dataname}: 模型构建"):
                    model = Network(view, dims, args.feature_dim, args.high_feature_dim, device)
                    print(model)
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
                    args.con_epochs = 1000  # small/large 300/600
                    pre_check_num = 10
                    valid_check_num = 1
                else:  # small
                    pre_check_num = 10
                    valid_check_num = 1

                with measure(f"{Dataname}: 预训练"):
                    for epoch in tqdm(range(args.pre_epochs)):
                        preloss = pretrain(epoch, Dataname, current_time)  # 1.pre-train
                        preloss_list.append(preloss)
                        if (epoch + 1) % pre_check_num == 0:  # TODO pre_check_num 1. pre
                            with measure(f"{Dataname}: 验证评估(KMeans)"):
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
                plot_loss(imgs_path, preloss_list, Dataname, 'pretrain loss')

                with measure(f"{Dataname}: 一致性训练"):
                    for epoch in tqdm(range(args.con_epochs)):
                        epoch = args.pre_epochs + epoch
                        plot_SDD = False
                        total_epochs = args.pre_epochs + args.con_epochs
                        if epoch + 1 == total_epochs:
                            plot_SDD = True
                        conloss = contrastive_train(epoch, Dataname, total_epochs, plot_SDD, current_time)  # 2.contrastive train
                        conloss_list.append(conloss)
                        # TODO valid_check_num 2. con
                        if (epoch + 1) % valid_check_num == 0:  # TODO con
                            with measure(f"{Dataname}: 验证评估(KMeans)"):
                                acc, nmi, pur, ari, rs_Results = valid(model, device, dataset, view, data_size,
                                                                       class_num, pre_train=False,
                                                                       con_train=True)
                            save_results_to_csv(rs_Results, epoch + 1, metric_csv_path, view)
                            epoch_ticks.append(epoch + 1)
                            acc_list.append(acc)
                            nmi_list.append(nmi)
                            pur_list.append(pur)
                            ari_list.append(ari)
                        max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                                acc_weight=0.25, nmi_weight=0.25,
                                                                pur_weight=0.25, ari_weight=0.25)
                plot_loss(imgs_path, conloss_list, Dataname, 'con loss')
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
                # log save
                logger.info(str(info))
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
                logger.info(str(info))
                offset1 = 100000
                seed = int(abs(seed + random.uniform(-offset1, offset1)))
                offset2 = 0.0001
                lr = abs(lr + random.uniform(-offset2, offset2))
                lr = "{:.5f}".format(lr)
                lr = float(lr)
                del info
            # TODO [一般是取平均值，但考虑到需求下面实现了取最大值:D] 找到 acc_l 中最后一个元素最大的列表元素的位次（默认训练一次，所以l_max=0）
            with measure(f"{Dataname}: 出图与保存"):
                l_max = find_max_last_element_index(acc_l)
                acc_list, nmi_list, pur_list, ari_list, loss_list = acc_l[l_max], nmi_l[l_max], pur_l[l_max], ari_l[l_max], \
                    loss_l[l_max]
                max_index = find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list,
                                                        acc_weight=0.25, nmi_weight=0.25,
                                                        pur_weight=0.25, ari_weight=0.25)
                plot_acc(imgs_path, acc_list, Dataname, 'acc', epoch_ticks)
                plot_acc(imgs_path, nmi_list, Dataname, 'nmi', epoch_ticks)
                plot_acc(imgs_path, pur_list, Dataname, 'pur', epoch_ticks)
                plot_acc(imgs_path, ari_list, Dataname, 'ari', epoch_ticks)

                save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, loss_list, Dataname, data_ratio,
                                   valid_check_num, current_time)
                # ELMC σ 变化曲线（非 fixed 模式才有数据；SIGMA_PRINT_ENABLED 关闭则自动跳过）
                plot_sigma(get_sigma_history(), imgs_path, Dataname)
            print(f'Max metric: epoch{(max_index + 1) * valid_check_num}\n'
                  f'1.acc:{acc_list[max_index] * 100:.2f}%\n'
                  f'2.nmi:{nmi_list[max_index] * 100:.2f}%\n'
                  f'3.pur:{pur_list[max_index] * 100:.2f}%\n'
                  f'4.ari:{ari_list[max_index] * 100:.2f}%\n'
                  f'5.best seed[{seed_l[l_max]}] & learning rate[{lr_l[l_max]}] for this dataset')
            # 显式删除变量
            del dataset
            # 手动调用垃圾回收
            gc.collect()
        data_iter += 1
    # —— 全部数据集运行结束：记录并打印分段运行时间 ——
    timing_secs["总耗时"] = time.perf_counter() - t_start_all
    print_timing_report(logger)
