"""
utils/elmb_sigma_test.py — ELMC 带宽模式（SIGMA_MODE）受控对比测试
================================================================
用法（在仓库根目录）：
    python utils/elmb_sigma_test.py                      # 默认参数
    python utils/elmb_sigma_test.py --epochs 200 --samples 2000 --update-md

功能：用合成数据模拟训练过程（特征从噪声逐步收敛到簇结构，模拟编码器学习），
      在每一轮 epoch 用各 SIGMA_MODE 计算 ELMC 视图权重，绘制"对齐视图权重
      随 epoch 演化"曲线，统计权重≈均匀（含防御回退）的轮数。

输出：
  - 曲线图：{--fig-dir}/test{N}_weights_epoch_{条件}.png（raw / l2-norm 两张）
  - 控制台汇总表：每模式 wA@epoch1 / wA@epoch100 / 权重≈均匀的epoch数
  - （可选 --update-md）把"参数+结果+耗时+图引用"追加写入 docs/elmb_sigma_tests.md，
    结论部分留待人工补充

说明：
  - 视图构造：对齐 = 与全局一致；打乱 = structure[perm]（固定置换，结构错配）；
    噪声 = z_all + randn×6（全程强噪声）
  - "权重≈均匀"包含两种情形：防御真回退（total=0/NaN → 返回 1/V）与核过平导致的
    自然均匀，需结合特征尺度人工判断
  - 图与 md 记录文件仅本地使用，不上传 GitHub（见 .gitignore）
"""

import argparse
import datetime
import os
import sys
import time

# 保证从任意目录运行都能导入仓库模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # 无显示环境绘图
import matplotlib.pyplot as plt
import torch

from utils.GlobalLocalManifoldCalibration import manifold_alignment_weights
import utils.GlobalLocalManifoldCalibration as GLMC


# 显示标签（判别力表用）
COND_LABELS = {'raw': 'raw 特征', 'l2-norm': 'L2 归一化'}
MODE_LABELS = {'median': 'median', 'fixed': 'fixed (σ=1)', 'mean': 'mean',
               'quantile': 'quantile (0.1)', 'local': 'local (k=5)'}


def verdict(w):
    """判定：均匀回退 / 最强区分 / 能区分 / 分不出结构"""
    if bool(torch.allclose(w, torch.full_like(w, 1.0 / len(w)), atol=1e-3)):
        return '均匀回退'
    r = float(w[0] / w[1])  # 对齐 / 打乱
    if r >= 1.3:
        return '最强区分'
    if r >= 1.05:
        return '能区分'
    return '分不出结构'


def parse_args():
    p = argparse.ArgumentParser(description='ELMC 带宽模式受控对比测试')
    p.add_argument('--samples', type=int, default=1000, help='样本数（默认 1000）')
    p.add_argument('--dims', type=int, default=50, help='特征维度（默认 50）')
    p.add_argument('--clusters', type=int, default=4, help='簇数（默认 4）')
    p.add_argument('--epochs', type=int, default=100, help='模拟训练轮数，>=100（默认 100）')
    p.add_argument('--seed', type=int, default=42, help='随机种子（默认 42）')
    p.add_argument('--score-form', type=str, default='trace',
                   help='SCORE_FORM（默认 trace=论文默认）')
    p.add_argument('--modes', type=str, default='median,fixed,mean,quantile,local',
                   help='SIGMA_MODE 列表，逗号分隔')
    p.add_argument('--fig-dir', type=str, default='docs/elmb_sigma_test_figs',
                   help='曲线图输出目录')
    p.add_argument('--md-path', type=str, default='docs/elmb_sigma_tests.md',
                   help='测试记录 md 文件路径')
    p.add_argument('--update-md', action='store_true', help='把本次结果追加写入测试记录 md')
    return p.parse_args()


def run_test(args, test_no):
    torch.manual_seed(args.seed)
    n, dg, K, T = args.samples, args.dims, args.clusters, args.epochs
    centers = torch.randn(K, dg) * 4
    labels = torch.randint(0, K, (n,))
    structure = centers[labels]            # 干净的簇结构
    perm = torch.randperm(n)               # 打乱视图的固定置换
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    GLMC.SIGMA_MODE = modes[0]             # 占位，避免未设置
    GLMC.SIGMA_PRINT_ENABLED = False
    GLMC.SCORE_FORM = args.score_form

    def normed(x):
        return torch.nn.functional.normalize(x, dim=1)

    results, fallback, last_w = {}, {}, {}
    t_start = time.perf_counter()
    for cond_name, cond_fn in [('raw', lambda x: x), ('l2-norm', normed)]:
        for mode in modes:
            GLMC.SIGMA_MODE = mode
            wA_list, fb = [], 0
            for t in range(1, T + 1):
                p = t / T
                # 特征演化：全局特征从纯噪声逐步收敛到簇结构（模拟编码器学习）
                z_all = p * structure + (1 - p) * torch.randn(n, dg) * 3
                vA = z_all                                            # 对齐视图
                vB = p * structure[perm] + (1 - p) * torch.randn(n, dg) * 3  # 打乱视图
                vC = z_all + torch.randn(n, dg) * 6                   # 噪声视图
                w = manifold_alignment_weights(
                    [cond_fn(vA), cond_fn(vB), cond_fn(vC)], cond_fn(z_all)).detach()
                wA_list.append(float(w[0]))
                last_w[(cond_name, mode)] = w                         # 训练结束时的三视图权重
                if bool(torch.allclose(w, torch.full_like(w, 1.0 / 3), atol=1e-4)):
                    fb += 1
            results[(cond_name, mode)] = wA_list
            fallback[(cond_name, mode)] = fb
    elapsed = time.perf_counter() - t_start

    # ================= 画图 =================
    os.makedirs(args.fig_dir, exist_ok=True)
    colors = {'median': '#4C72B0', 'fixed': '#D55E00', 'mean': '#8172B2',
              'quantile': '#55A868', 'local': '#CCB974'}
    fig_paths = {}
    for cond_name in ['raw', 'l2-norm']:
        plt.figure(figsize=(10, 6))
        for mode in modes:
            plt.plot(range(1, T + 1), results[(cond_name, mode)], label=mode,
                     color=colors.get(mode), linewidth=1.8)
        plt.axhline(1.0 / 3, color='gray', linestyle='--', linewidth=1, label='uniform (1/3)')
        plt.xlabel('Epoch (simulated training)', fontsize=12)
        plt.ylabel('w_aligned (aligned view weight)', fontsize=12)
        plt.title(f'ELMC SIGMA_MODE test #{test_no}: aligned-view weight over {T} epochs - {cond_name}',
                  fontsize=13)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        path = os.path.join(args.fig_dir, f'test{test_no}_weights_epoch_{cond_name}.png')
        plt.savefig(path, dpi=150)
        plt.close()
        fig_paths[cond_name] = path
        print(f'图已保存: {path}')

    # ================= 控制台汇总 =================
    calls = len(modes) * T * 2
    print(f'\n总耗时: {elapsed:.1f} s（{calls} 次调用, n={n}, T={T}）')
    print(f'\n=== 判别力表（训练结束时 = epoch {T} 的三视图权重） ===')
    print(f"{'条件':<10}{'模式':<14}{'对齐':>7}{'打乱':>7}{'噪声':>7}{'对齐/打乱':>10}  判定")
    for cond_name in ['raw', 'l2-norm']:
        for mode in modes:
            w = last_w[(cond_name, mode)]
            r = float(w[0] / w[1])
            print(f"{COND_LABELS[cond_name]:<10}{MODE_LABELS.get(mode, mode):<14}"
                  f"{w[0]:>7.3f}{w[1]:>7.3f}{w[2]:>7.3f}{r:>9.2f}x  {verdict(w)}")
    print(f'\n=== epoch 演化补充（对齐视图权重） ===')
    print(f"{'条件':<9}{'模式':<9}{'wA@epoch1':>10}{f'wA@epoch{T}':>12}{'权重≈均匀epoch数':>14}")
    for cond_name in ['raw', 'l2-norm']:
        for mode in modes:
            w1, w100 = results[(cond_name, mode)][0], results[(cond_name, mode)][-1]
            print(f"{cond_name:<9}{mode:<9}{w1:>10.3f}{w100:>12.3f}{fallback[(cond_name, mode)]:>14}")

    # ================= 可选：写入 md 记录 =================
    if args.update_md:
        write_md(args, test_no, results, last_w, fallback, elapsed, calls, fig_paths)
        print(f'已追加写入测试记录: {args.md_path}')

    return results, fallback, elapsed


def write_md(args, test_no, results, last_w, fallback, elapsed, calls, fig_paths):
    """把本次测试的 参数+结果+耗时+图引用 追加进 md（结论留人工补充）"""
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    lines = [
        f'### 测试 #{test_no} · {now}：SIGMA_MODE 对比（{args.epochs} epoch 模拟训练, n={args.samples}）',
        '',
        '**本测试参数**：',
        '',
        '| 参数 | 值 |',
        '|---|---|',
        f'| 随机种子 | `torch.manual_seed({args.seed})` |',
        f'| 数据规模 | {args.samples} 样本 × {args.dims} 维，{args.clusters} 簇 |',
        f'| 模拟训练轮数 | T={args.epochs} epoch（特征每轮演化并重算权重） |',
        '| 特征演化模型 | z_all(t) = (t/T)·structure + (1−t/T)·randn×3（从噪声逐步收敛到簇结构） |',
        '| 视图构造 | 对齐 = 全局一致；打乱 = structure[perm]（固定置换）；噪声 = z_all + randn×6 |',
        f'| 分数形式 | SCORE_FORM=\'{args.score_form}\' |',
        f'| 带宽模式 | {args.modes} |',
        '| 特征条件 | raw / l2-norm（对应代码 z_norm 预处理）两种 |',
        f'| 调用次数 | {calls} 次 |',
        '',
        f'**结果（训练结束时 = epoch {args.epochs} 的三视图权重，1/3 = 均匀基准）**：',
        '',
        '| 条件 | 模式 | 对齐 | 打乱 | 噪声 | 对齐/打乱 | 判定 |',
        '|---|---|---|---|---|---|---|',
    ]
    for cond_name in ['raw', 'l2-norm']:
        for mode in [m.strip() for m in args.modes.split(',') if m.strip()]:
            w = last_w[(cond_name, mode)]
            r = float(w[0] / w[1])
            lines.append(f'| {COND_LABELS[cond_name]} | {MODE_LABELS.get(mode, mode)} | '
                         f'{w[0]:.3f} | {w[1]:.3f} | {w[2]:.3f} | {r:.2f}x | {verdict(w)} |')
    lines += [
        '',
        '**epoch 演化补充**（对齐视图权重，权重≈均匀含防御回退与核过平两种情形）：',
        '',
        '| 条件 | 模式 | wA@epoch1 | wA@epoch100 | 权重≈均匀的epoch数 |',
        '|---|---|---|---|---|',
    ]
    for cond_name in ['raw', 'l2-norm']:
        for mode in [m.strip() for m in args.modes.split(',') if m.strip()]:
            w1, w100 = results[(cond_name, mode)][0], results[(cond_name, mode)][-1]
            lines.append(f'| {cond_name} | {mode} | {w1:.3f} | {w100:.3f} | {fallback[(cond_name, mode)]} |')
    lines += [
        '',
        f'**耗时**：{elapsed:.1f} s（{calls} 次调用，n={args.samples}，CPU）',
        f'**曲线图**：`{args.fig_dir}/` 下 test{test_no}_weights_epoch_raw.png 与 test{test_no}_weights_epoch_l2-norm.png',
        '',
        '**结论**：（待补充）',
        '',
        '---',
        '',
        '<!-- 后续测试追加在此处，格式同测试 #1 -->',
    ]
    with open(args.md_path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    args = parse_args()
    # 测试编号：从 md 现有记录推断下一个（图文件名带编号，即使不 --update-md 也一致）
    test_no = 1
    try:
        with open(args.md_path, encoding='utf-8') as f:
            test_no = sum(1 for line in f if line.startswith('### 测试 #')) + 1
    except OSError:
        pass
    run_test(args, test_no)
