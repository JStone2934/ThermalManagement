#!/usr/bin/env python3
"""DDS 采集 → 原始 HDF5（plan §5）；需本机安装 unitree_sdk2_python。"""

from __future__ import annotations

import argparse

from _config import CONFIG_FILENAMES, load_merged


def main() -> None:
    p = argparse.ArgumentParser(description="Record thermal session to HDF5.")
    p.add_argument(
        "--extra-config",
        nargs="*",
        default=[],
        help="Extra YAML basenames under configs/ to merge after defaults.",
    )
    args = p.parse_args()
    names = (*CONFIG_FILENAMES, *args.extra_config) if args.extra_config else None
    cfg = load_merged(names)
    _ = cfg  # TODO: thermal_g1.collector
    raise SystemExit("collector not implemented — wire thermal_g1.collector here")


if __name__ == "__main__":
    main()
