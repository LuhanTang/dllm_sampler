# sampler/utils_io.py
from __future__ import annotations

from typing import List


def parse_steps(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _fmt_float_tag(x: float) -> str:
    """
    Make a filesystem-friendly float tag:
      1.0 -> "1"
      1e-4 -> "0.0001" -> "0p0001"
      -0.5 -> "m0p5"
    """
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")


def _sanitize(s: str) -> str:
    """
    Keep only [A-Za-z0-9 . _ -], replace others with underscore.
    """
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)
