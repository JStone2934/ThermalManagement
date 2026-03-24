#!/usr/bin/env python3
"""导出 ONNX（plan §7.1, opset 17）。"""

from __future__ import annotations

import argparse

from _config import CONFIG_FILENAMES, load_merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="thermal_predictor_g1.onnx")
    p.add_argument("--extra-config", nargs="*", default=[])
    args = p.parse_args()
    names = (*CONFIG_FILENAMES, *args.extra_config) if args.extra_config else None
    _ = load_merged(names), args.output
    raise SystemExit("export not implemented — wire thermal_g1.export here")


if __name__ == "__main__":
    main()
