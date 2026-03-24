"""Shallow-merge project YAML configs (repo root = parent of scripts/)."""

from __future__ import annotations

from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILENAMES = (
    "default.yaml",
    "topics.yaml",
    "data.yaml",
    "joints_g1_29dof.yaml",
)


def load_merged(names: tuple[str, ...] | None = None) -> dict:
    out: dict = {}
    for name in names or CONFIG_FILENAMES:
        path = ROOT / "configs" / name
        if not path.is_file():
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out.update(data)
    return out
