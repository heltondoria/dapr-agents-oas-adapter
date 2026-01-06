"""Resolve Dapr component templates for this example."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_sys_path() -> None:
    """Ensure the repo root (the folder containing `pyproject.toml`) is on sys.path."""
    anchor = Path(__file__).resolve()
    for candidate in [anchor, *anchor.parents]:
        if (candidate / "pyproject.toml").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_ensure_repo_root_on_sys_path()

from examples._shared.env_templates import resolve_env_templates
from examples._shared.optional_dotenv import try_load_dotenv


def main() -> int:
    try_load_dotenv()
    result = resolve_env_templates(Path(__file__).parent / "components")
    if result.missing_env_vars:
        missing = ", ".join(result.missing_env_vars)
        raise SystemExit(f"Missing environment variables: {missing}")
    print(str(result.resolved_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
