"""Optional support for loading `.env` without requiring python-dotenv."""

from __future__ import annotations


def try_load_dotenv() -> None:
    """Load `.env` if `python-dotenv` is installed; otherwise do nothing."""
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()
