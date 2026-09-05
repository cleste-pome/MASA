# ===================== 标准库 =====================
import csv
import os
import time  # 锁轮询重试用（注意：勿用 datetime 的 time 遮蔽本模块）
from datetime import datetime


def find_max_weighted_sum_index(acc_list, nmi_list, pur_list, ari_list, acc_weight, nmi_weight, pur_weight, ari_weight):
    max_sum = float('-inf')
    max_index = -1

    for i, (acc, nmi, pur, ari) in enumerate(zip(acc_list, nmi_list, pur_list, ari_list)):
        current_sum = acc * acc_weight + nmi * nmi_weight + pur * pur_weight + ari * ari_weight
        if current_sum > max_sum:
            max_sum = current_sum
            max_index = i

    return max_index


def save_lists_to_file(acc_list, nmi_list, pur_list, ari_list, loss_list, data_name, data_rate, epoch_ticks,
                       current_time):
    # 创建logs文件夹（汇总指标与视图指标同放 Metrics/ 下）
    csv_path = f'3.csv/Metrics'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    # 创建以data_name命名的csv文件路径（文件名带时间戳，区分多次实验）
    file_path = os.path.join(csv_path, f'{data_name}_{current_time}_{data_rate}.csv')

    # 各列表长度应一致（每验证点一条），epoch 用训练时记录的 epoch_ticks，不重新算术生成
    assert len(epoch_ticks) == len(acc_list) == len(nmi_list) == len(pur_list) == len(ari_list) == len(loss_list), \
        f'列表长度不一致: epoch_ticks={len(epoch_ticks)}, acc={len(acc_list)}, loss={len(loss_list)}'

    # 写入数据到CSV文件
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # 写入表头
        csvwriter.writerow(['epoch', 'acc', 'nmi', 'pur', 'ari', 'loss'])
        # 写入数据（epoch 为真实验证轮次，loss 为对应验证点的当轮总损失）
        for epoch, acc, nmi, pur, ari, loss in zip(epoch_ticks, acc_list, nmi_list, pur_list, ari_list, loss_list):
            csvwriter.writerow([epoch, acc, nmi, pur, ari, loss])

    pass  # 保存路径不逐个打印，训练结束由 train.py 统一输出一句结果位置


def find_max_last_element_index(acc_l):
    # 使用enumerate来同时获取元素和索引
    max_index = 0
    max_value = acc_l[0][-1]  # 初始化最大值为第一个子列表的最后一个元素

    for index, sublist in enumerate(acc_l):
        if sublist[-1] > max_value:  # 如果当前子列表的最后一个元素大于当前最大值
            max_value = sublist[-1]  # 更新最大值
            max_index = index  # 更新最大值对应的索引

    return max_index


# TODO 实时写入每个视图和全局的评价指标数据
def save_results_to_csv(zs_results, epoch, filepath, views):
    # 打开文件以追加模式写入
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        # 构造数据行
        row = [epoch]  # 开始填充epoch列

        # 将每个视图的评价指标加入数据行
        for view_idx in range(views):
            result = zs_results[view_idx]
            row.extend([result[1], result[2], result[3], result[4]])  # acc, nmi, pur, ari

        # 假设zs_results[views]是全局结果（z_all）
        z_all_result = zs_results[views]  # 全局结果
        row.extend([z_all_result[1], z_all_result[2], z_all_result[3], z_all_result[4]])  # acc, nmi, pur, ari

        # 写入数据行
        writer.writerow(row)


def save_wz_view_to_csv(zs_results, epoch, filepath):
    # 自动补全 .csv 扩展名
    if not filepath.endswith('.csv'):
        filepath += '.csv'

    # 创建保存目录（如不存在）
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 如果当前是 epoch 1 且文件已存在，则将旧文件重命名为备份
    if epoch == 1 and os.path.exists(filepath):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{filepath.rstrip('.csv')}_backup_{timestamp}.csv"
        os.rename(filepath, backup_path)
        print(f"[提示] 检测到重新训练，旧文件已备份为：{backup_path}")

    # 文件锁路径（用于并发安全）
    lock_path = filepath + '.lock'

    # 获取文件锁：轮询等待释放（最多等 5 秒；超时说明可能残留陈旧锁，警告后继续，避免卡死训练）
    for _ in range(100):
        try:
            lock_file = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(lock_file)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        print(f"[警告] 未能获取文件锁 {lock_path}（可能残留陈旧锁），继续写入（多进程并发时存在覆盖风险）")

    try:
        # 写入当前 epoch 的视图权重到 CSV
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [epoch] + zs_results.detach().cpu().numpy().tolist()
            writer.writerow(row)
    finally:
        # 删除锁文件，释放写入权限
        if os.path.exists(lock_path):
            os.remove(lock_path)


def create_csv(data_name, data_rate, views, current_time):
    # 创建logs文件夹，如果不存在（视图指标与汇总指标同放 3.csv/Metrics/ 下）
    csv_dir = '3.csv/Metrics'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # 命名与汇总指标统一：{data_name}_{current_time}_{data_rate}，视图文件用 view_ 前缀区分
    filename = f'view_{data_name}_{current_time}_{data_rate}.csv'
    # 创建以data_name命名的csv文件路径
    filepath = os.path.join(csv_dir, filename)

    # 写入表头
    headers = ['epoch']
    # 为每个视图生成对应的评价指标列
    for view_idx in range(views):
        headers.extend([f'view{view_idx + 1}_acc', f'view{view_idx + 1}_nmi', f'view{view_idx + 1}_pur',
                        f'view{view_idx + 1}_ari'])
    # 如果存在全局评价指标，增加
    headers.extend(['z_all_acc', 'z_all_nmi', 'z_all_pur', 'z_all_ari'])

    # 写入表头到CSV文件
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入表头

    return filepath, filename
