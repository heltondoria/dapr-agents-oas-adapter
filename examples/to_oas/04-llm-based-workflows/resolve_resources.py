"""Resolve Dapr component templates for this example.

Prints a temp directory path that can be passed to `dapr run --resources-path`.
"""

from __future__ import annotations

from pathlib import Path

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
