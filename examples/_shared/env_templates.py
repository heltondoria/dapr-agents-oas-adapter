"""Resolve env-template placeholders in Dapr component YAMLs.

Quickstarts commonly use placeholders like `{{OPENAI_API_KEY}}` inside YAML.
This module resolves those placeholders using the current environment.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


@dataclass(frozen=True, slots=True)
class ResolvedResources:
    """Result of resolving a directory of YAML templates."""

    source_dir: Path
    resolved_dir: Path
    missing_env_vars: tuple[str, ...] = ()

    def cleanup(self) -> None:
        """Delete the resolved directory (best-effort)."""
        shutil.rmtree(self.resolved_dir, ignore_errors=True)


def resolve_env_templates(
    source_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
    file_suffixes: tuple[str, ...] = (".yaml", ".yml"),
) -> ResolvedResources:
    """Copy `source_dir` to a temp folder replacing `{{VARNAME}}` placeholders.

    Args:
        source_dir: Directory containing YAML component templates.
        env: Environment mapping to use. Defaults to `os.environ`.
        file_suffixes: File suffixes that are treated as templates.

    Returns:
        ResolvedResources pointing to a temporary directory.

    Raises:
        FileNotFoundError: If `source_dir` does not exist.
    """
    src = source_dir.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Components directory not found: {src}")
    if not src.is_dir():
        raise FileNotFoundError(f"Components path is not a directory: {src}")

    env_map: Mapping[str, str] = os.environ if env is None else env
    out_dir = Path(tempfile.mkdtemp(prefix="dapr_resources_")).resolve()

    missing: set[str] = set()

    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = out_dir / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        if path.suffix.lower() not in file_suffixes:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            continue

        raw = path.read_text(encoding="utf-8")

        def _replace(match: re.Match[str]) -> str:
            key = match.group(1)
            value = env_map.get(key)
            if value is None:
                missing.add(key)
                return match.group(0)
            return value

        rendered = _PLACEHOLDER_RE.sub(_replace, raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")

    return ResolvedResources(
        source_dir=src, resolved_dir=out_dir, missing_env_vars=tuple(sorted(missing))
    )
