"""Path helpers for examples."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(*, anchor_file: str) -> Path:
    """Find the repository root by walking upwards until `pyproject.toml` is found."""
    start = Path(anchor_file).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate `pyproject.toml` above this example.")
