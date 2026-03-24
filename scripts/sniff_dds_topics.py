#!/usr/bin/env python3
"""Phase 0：嗅探 CycloneDDS topic 名称（plan §0.4 / §8）。"""

from __future__ import annotations

import argparse


def main() -> None:
    argparse.ArgumentParser(description="DDS discovery helper (stub).").parse_args()
    raise SystemExit(
        "stub: use cyclonedds CLI or unitree_sdk2py discovery; "
        "then update configs/topics.yaml"
    )


if __name__ == "__main__":
    main()
