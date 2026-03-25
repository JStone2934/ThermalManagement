#!/usr/bin/env python3
"""
订阅 G1 `rt/lowstate`，校验 CRC 后打印全身 29 关节双通道温度 + IMU 温度。

依赖：按 plan Phase 0 安装 unitree_sdk2_python，并配置 CycloneDDS 网络接口。

用法示例：
  # 使用默认网卡（与官方 g1_low_level_example 一致）
  uv run python scripts/print_body_temperatures.py

  # 指定网卡（常见：有线直连机器人）
  uv run python scripts/print_body_temperatures.py eth0

  # 仅打印摘要、提高刷新间隔
  uv run python scripts/print_body_temperatures.py --summary-only --interval 1.0
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _config import load_merged  # noqa: E402

G1_NUM_MOTOR = 29
DEFAULT_TOPIC = "rt/lowstate"


def _require_unitree():
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as e:
        raise SystemExit(
            "未找到 unitree_sdk2_python。请按宇树文档安装 SDK 并保证 PYTHONPATH 可导入 "
            "unitree_sdk2py。\n"
            f"原始错误: {e}"
        ) from e
    return ChannelFactoryInitialize, ChannelSubscriber, LowState_, CRC


def _joint_names_from_config() -> dict[int, str]:
    cfg = load_merged()
    joints = cfg.get("joint_index", {}).get("joints", [])
    out: dict[int, str] = {}
    for j in joints:
        if isinstance(j, dict) and "i" in j and "name" in j:
            out[int(j["i"])] = str(j["name"])
    return out


def _read_two_channel_temp(motor_state) -> tuple[int, int]:
    """MotorState_.temperature 为长度 2 的数组（IDL 可能为 int16 或 uint8，统一转 int）。"""
    t = motor_state.temperature
    try:
        c0 = int(t[0])
        c1 = int(t[1])
    except (TypeError, IndexError):
        t_list = list(t)
        c0 = int(t_list[0]) if len(t_list) > 0 else 0
        c1 = int(t_list[1]) if len(t_list) > 1 else 0
    return c0, c1


def _format_table(
    tick: int,
    imu_temp: int,
    motor_temps: list[tuple[int, int]],
    names: dict[int, str],
    crc_ok: bool,
) -> str:
    lines = [
        f"tick={tick}  IMU_T={imu_temp}°C  CRC={'OK' if crc_ok else 'FAIL'}",
        f"{'i':>3}  {'joint':<22}  {'T0':>5}  {'T1':>5}  {'max':>5}",
        "-" * 44,
    ]
    for i in range(G1_NUM_MOTOR):
        c0, c1 = motor_temps[i]
        nm = names.get(i, f"J{i}")
        mx = max(c0, c1)
        lines.append(f"{i:3d}  {nm:<22}  {c0:5d}  {c1:5d}  {mx:5d}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Print G1 full-body motor temperatures from rt/lowstate.")
    p.add_argument(
        "interface",
        nargs="?",
        default=None,
        help="网卡名（如 eth0）；省略时与官方示例一致，由 CycloneDDS 默认发现。",
    )
    p.add_argument("--topic", default=DEFAULT_TOPIC, help=f"DDS topic（默认 {DEFAULT_TOPIC}）")
    p.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="终端刷新间隔（秒），基于最近一次有效/无效帧。",
    )
    p.add_argument(
        "--no-crc",
        action="store_true",
        help="跳过 CRC 校验（仅调试用，勿用于入库数据）。",
    )
    p.add_argument(
        "--summary-only",
        action="store_true",
        help="仅打印每关节 max(T0,T1) 的最大值与 IMU 温度一行摘要。",
    )
    args = p.parse_args()

    ChannelFactoryInitialize, ChannelSubscriber, LowState_, CRC = _require_unitree()

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)

    names = _joint_names_from_config()
    crc_engine = CRC()

    state_lock = threading.Lock()
    latest_msg: LowState_ | None = None
    stats = {"frames": 0, "crc_fail": 0}

    def handler(msg: LowState_) -> None:
        nonlocal latest_msg
        with state_lock:
            stats["frames"] += 1
            if not args.no_crc:
                try:
                    if crc_engine.Crc(msg) != msg.crc:
                        stats["crc_fail"] += 1
                        return
                except Exception:
                    stats["crc_fail"] += 1
                    return
            latest_msg = msg

    sub = ChannelSubscriber(args.topic, LowState_)
    sub.Init(handler, 10)

    print(
        f"已订阅 {args.topic}，打印 {G1_NUM_MOTOR} 关节温度；"
        f"CRC={'关' if args.no_crc else '开'}；Ctrl+C 退出。\n"
    )

    try:
        while True:
            time.sleep(args.interval)
            with state_lock:
                msg = latest_msg
                s_frames = stats["frames"]
                s_fail = stats["crc_fail"]
            if msg is None:
                print(f"[{time.strftime('%H:%M:%S')}] 尚无有效帧 (总回调 {s_frames}, CRC失败 {s_fail})")
                continue

            motor_temps: list[tuple[int, int]] = []
            for i in range(G1_NUM_MOTOR):
                motor_temps.append(_read_two_channel_temp(msg.motor_state[i]))

            try:
                imu_t = int(msg.imu_state.temperature)
            except (TypeError, AttributeError):
                imu_t = 0

            crc_ok = True
            if not args.no_crc:
                try:
                    crc_ok = crc_engine.Crc(msg) == msg.crc
                except Exception:
                    crc_ok = False

            ts = time.strftime("%H:%M:%S")
            if args.summary_only:
                max_coil = max(t[0] for t in motor_temps)
                max_shell = max(t[1] for t in motor_temps)
                max_any = max(max(t[0], t[1]) for t in motor_temps)
                print(
                    f"[{ts}] tick={msg.tick}  IMU={imu_t}°C  "
                    f"max_T0={max_coil}  max_T1={max_shell}  max_any={max_any}  "
                    f"crc_ok={crc_ok}  frames={s_frames} crc_fail={s_fail}"
                )
            else:
                print(f"[{ts}]")
                print(_format_table(msg.tick, imu_t, motor_temps, names, crc_ok))
                print(f"(frames={s_frames}, crc_fail={s_fail})\n")
    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
