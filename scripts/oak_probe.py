#!/usr/bin/env python3
"""检测已连接的 OAK/DepthAI 设备，并输出基础设备信息。"""

from __future__ import annotations

import sys


def _read_attr(obj, *names: str, default: str = "unknown"):
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        return value() if callable(value) else value
    return default


def main() -> int:
    try:
        import depthai as dai
    except ImportError:
        print("错误：当前 Python 环境未安装 depthai。")
        print("请运行：python -m pip install -r requirements.txt")
        return 2

    devices = dai.Device.getAllAvailableDevices()
    print(f"DepthAI 版本：{dai.__version__}")
    print(f"可用设备数量：{len(devices)}")

    if not devices:
        print("未发现 OAK/DepthAI 设备。")
        print("请检查 USB-C 线缆、供电和 macOS USB 设备列表。")
        return 1

    for index, info in enumerate(devices, start=1):
        print(f"\n设备 {index}")
        print(f"  名称：{_read_attr(info, 'name')}")
        print(f"  设备ID：{_read_attr(info, 'mxid', 'deviceId', 'getDeviceId')}")
        print(f"  状态：{_read_attr(info, 'state')}")
        print(f"  协议：{_read_attr(info, 'protocol')}")

    print("\n正在打开第一台设备以验证固件通信...")
    with dai.Device() as device:
        print("已连接。")
        print(f"  设备ID：{device.getMxId()}")
        print(f"  USB速度：{device.getUsbSpeed()}")
        try:
            print(f"  相机：{device.getConnectedCameraFeatures()}")
        except Exception as exc:
            print(f"  相机：不可用（{type(exc).__name__}: {exc}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
