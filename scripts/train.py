#!/usr/bin/env python3
"""训练入口（plan §6）。"""

from __future__ import annotations

import argparse

from _config import CONFIG_FILENAMES, load_merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--extra-config", nargs="*", default=[], help="Extra YAML under configs/")
    args = p.parse_args()
    names = (*CONFIG_FILENAMES, *args.extra_config) if args.extra_config else None
    _ = load_merged(names)
    raise SystemExit("training not implemented — wire thermal_g1.training here")


if __name__ == "__main__":
    main()
