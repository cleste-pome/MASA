"""
utils/device_check.py — 设备前置检查（训练前探测环境，决定 CUDA/MPS/CPU 配置）

两种用法：
  1) 独立运行（前置检查，只打印不训练）：
         python utils/device_check.py
  2) 被训练程序引入：
         from utils.device_check import auto_device, detect_device
         device = auto_device()          # 简洁版：返回选定的 torch.device
         device, info = detect_device()  # 详细版：附带环境信息字典

决策规则（优先级从高到低）：
  1) torch.cuda.is_available()  → cuda （Linux 服务器 / NVIDIA 显卡）
  2) macOS（Darwin）+ MPS 可用   → mps  （MacBook 等 Mac 设备，识别到 Mac 就尽量启用 MPS）
  3) 其他                        → cpu

强制指定（环境变量）：
  MASA_DEVICE=cuda|mps|cpu|auto   # auto=按上述决策规则；不设或留空等价于 auto
  （强制值不可用时会回退到自动决策）
"""

# ===================== 标准库 =====================
import os
import platform
import subprocess

# ===================== 第三方库 =====================
import torch


def auto_device():
    """自动检测可用计算设备：CUDA > MPS(Apple Silicon) > CPU，跨平台通用（与重构版 auto_device 同逻辑）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mac_model():
    """读取 macOS 硬件型号（如 MacBookPro17,9 / Mac17,9）；非 macOS 或读取失败返回 None"""
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, timeout=5)
        model = result.stdout.strip()
        return model or None
    except Exception:
        return None


def platform_info():
    """基础平台信息（不含 torch 部分）"""
    return {
        "os": platform.system(),           # Darwin / Linux / Windows
        "os_release": platform.release(),  # 如 27.0.0
        "arch": platform.machine(),        # arm64 / x86_64 ...
        "mac_model": _mac_model(),         # 如 MacBookPro17,9（仅 macOS）
        "is_mac": platform.system() == "Darwin",
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
    }


def _cuda_info():
    """CUDA 可用性汇总；不可用返回 None"""
    if not torch.cuda.is_available():
        return None
    return {
        "count": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }


def _mps_info():
    """MPS 后端汇总"""
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "built": mps_backend.is_built() if mps_backend is not None else False,
        "available": mps_backend.is_available() if mps_backend is not None else False,
    }


def detect_device(verbose=True):
    """设备前置检查：探测环境 → 决定设备 →（verbose 时）打印汇总。

    :param verbose: 是否打印探测汇总
    :return: (device, info)；info 为字典，含 platform/cuda/mps/forced/device 等全部探测结果
    """
    info = {
        "platform": platform_info(),
        "torch": torch.__version__,
        "cuda": _cuda_info(),
        "mps": _mps_info(),
        "forced": None,
    }

    # 环境变量强制指定（测试/CI/服务器场景）
    override = (os.environ.get("MASA_DEVICE", "auto") or "auto").strip().lower()
    if override not in ("auto", "cuda", "mps", "cpu"):
        print(f"[device_check] ignoring unknown MASA_DEVICE={override!r} (options: auto/cuda/mps/cpu)")
        override = "auto"

    if override != "auto":
        if override == "cuda" and not torch.cuda.is_available():
            print("[device_check] MASA_DEVICE=cuda unavailable (no CUDA), falling back to auto")
            override = "auto"
        elif override == "mps" and not info["mps"]["available"]:
            print("[device_check] MASA_DEVICE=mps unavailable, falling back to auto")
            override = "auto"

    if override != "auto":
        device = torch.device(override)
        info["forced"] = override
    else:
        device = auto_device()
    info["device"] = device

    if verbose:
        print_device_summary(info)
    return device, info


def print_device_summary(info):
    """打印设备探测汇总（紧凑信息块，分隔线包裹，与训练分区排版一致）"""
    p = info["platform"]
    platform = (f"macOS {p['os_release']} ({p['arch']}) {p['mac_model']} <- Mac detected, MPS preferred"
                if p["is_mac"] and p["mac_model"]
                else f"{p['os']} {p['os_release']} ({p['arch']})")

    cuda = info["cuda"]
    cuda_txt = f"available x{cuda['count']} ({', '.join(cuda['names'])})" if cuda else "unavailable"
    mps = info["mps"]
    if mps["available"]:
        mps_txt = "available"
    elif mps["built"]:
        mps_txt = "unavailable (torch built with MPS but environment missing; check macOS >=12.3)"
    else:
        mps_txt = "unavailable (current torch built without MPS support)"

    dev = info["device"]
    note = "(forced by MASA_DEVICE)" if info["forced"] else (
        "(Mac device, MPS preferred)" if dev.type == "mps" else "")

    print("-" * 72)
    print(f"[Device] {platform}")
    print(f"  python = {p['python']}    torch = {info['torch']}    cpu cores = {p['cpus']}")
    print(f"  cuda = {cuda_txt}    mps = {mps_txt}")
    print(f"  decision = {dev}" + (f" {note}" if note else ""))
    if dev.type == "mps":
        print("  note = MPS-unsupported ops (torch.cdist/diag) are handled by GlobalLocalManifoldCalibration with automatic CPU fallback")
    if dev.type == "cpu" and p["is_mac"]:
        print("  note = MPS unavailable? check macOS >=12.3 and official PyTorch build")
    print("-" * 72)


def main():
    """独立运行入口：只做设备检查并打印，不启动训练"""
    device, _ = detect_device(verbose=True)
    print(f"Final device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
