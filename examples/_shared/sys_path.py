"""Helpers to make examples runnable from different working directories."""

from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_sys_path(*, anchor_file: str) -> Path:
    """Ensure the repository root is on sys.path.

    The examples are often executed with `dapr run -- python <script.py>`, and
    the current working directory may vary. Adding the project root to
    `sys.path` makes `import examples...` reliable.

    Args:
        anchor_file: Usually pass `__file__` from the caller.

    Returns:
        The detected project root Path.
    """
    anchor = Path(anchor_file).resolve()
    project_root = anchor.parents[2]  # <repo>/examples/<...>/<script.py>
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
