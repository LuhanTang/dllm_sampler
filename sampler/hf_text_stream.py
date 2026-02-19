#!/usr/bin/env python3
# sampler/hf_text_stream.py
# ------------------------------------------------------------
# HuggingFace streaming text helpers for building tokenizers + GT.
#
# - Supports streaming=True to avoid downloading huge datasets
# - Extracts text fields robustly across datasets
# - Optional language filter (for The Stack)
# - Light cleaning + per-doc length cap
# - Utilities to mix two streams with a fixed ratio and stop by byte budget
# ------------------------------------------------------------

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterator, Iterable, List, Optional, Tuple

# NOTE: we import datasets lazily to give clear error messages in environments
# where huggingface datasets isn't installed.


_WS_RE = re.compile(r"[ \t]+")


def _lazy_import_datasets():
    try:
        import datasets  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'datasets' is required. Install with:\n"
            "  pip install datasets\n"
            f"Original error: {e}"
        )
    return datasets


def _pick_text_field(ex: Dict) -> Optional[str]:
    # Try common fields first
    for k in ("text", "content", "code", "data", "document"):
        v = ex.get(k, None)
        if isinstance(v, str) and v.strip():
            return v

    # Fall back: find the first string field with decent length
    for k, v in ex.items():
        if isinstance(v, str) and len(v.strip()) >= 8:
            return v
    return None


def _clean_text(s: str, *, normalize_ws: bool, strip: bool) -> str:
    # Keep this light; do NOT over-normalize code.
    if "\r" in s:
        s = s.replace("\r", "")
    if strip:
        s = s.strip("\n")
    if normalize_ws:
        # Convert tabs to spaces, compress runs of spaces
        s = s.replace("\t", " ")
        s = _WS_RE.sub(" ", s)
    return s


def _cap_len(s: str, max_chars_per_doc: int) -> str:
    if max_chars_per_doc <= 0:
        return s
    if len(s) <= max_chars_per_doc:
        return s
    return s[:max_chars_per_doc]


@dataclass
class StreamConfig:
    streaming: bool = True
    split: str = "train"
    max_docs: Optional[int] = None
    max_chars_per_doc: int = 4096
    normalize_ws: bool = False  # for code, default False
    strip: bool = True


def stream_owt_text(
    hf_name: str = "stanford-cs336/owt-sample",
    *,
    cfg: StreamConfig = StreamConfig(),
) -> Iterator[str]:
    datasets = _lazy_import_datasets()
    ds = datasets.load_dataset(hf_name, split=cfg.split, streaming=cfg.streaming)

    n = 0
    for ex in ds:
        txt = _pick_text_field(ex)
        if txt is None:
            continue
        txt = _clean_text(txt, normalize_ws=cfg.normalize_ws, strip=cfg.strip)
        txt = _cap_len(txt, cfg.max_chars_per_doc)
        if txt.strip() == "":
            continue
        yield txt
        n += 1
        if cfg.max_docs is not None and n >= cfg.max_docs:
            break


def stream_stack_python_text(
    hf_name: str = "bigcode/the-stack",
    *,
    cfg: StreamConfig = StreamConfig(),
    language: str = "Python",
    subset: Optional[str] = None,
) -> Iterator[str]:
    """
    The Stack typically supports configuration/subset by language.
    Depending on dataset version, 'subset' may be needed:
      load_dataset("bigcode/the-stack", data_dir="data/python", ...)
    We try best-effort patterns with clear errors.
    """
    datasets = _lazy_import_datasets()

    # Best effort load:
    # 1) Try subset as config name if provided
    # 2) Otherwise try config=language lowercased (common in some datasets)
    # 3) Otherwise load base and filter by ex['language']
    load_errs: List[str] = []
    ds = None

    if subset is not None:
        try:
            ds = datasets.load_dataset(hf_name, subset, split=cfg.split, streaming=cfg.streaming)
        except Exception as e:
            load_errs.append(f"load_dataset(name, subset={subset}) failed: {e}")

    if ds is None:
        try:
            ds = datasets.load_dataset(hf_name, language.lower(), split=cfg.split, streaming=cfg.streaming)
        except Exception as e:
            load_errs.append(f"load_dataset(name, config={language.lower()}) failed: {e}")

    if ds is None:
        try:
            ds = datasets.load_dataset(hf_name, split=cfg.split, streaming=cfg.streaming)
        except Exception as e:
            load_errs.append(f"load_dataset(name) failed: {e}")
            msg = (
                "Failed to load The Stack dataset in streaming mode.\n"
                "Tried several patterns. Errors:\n  - " + "\n  - ".join(load_errs)
            )
            raise RuntimeError(msg)

    n = 0
    for ex in ds:
        # filter by language if field exists
        lang = ex.get("language", None)
        if isinstance(lang, str):
            # The Stack often uses capitalized language names
            if lang.lower() != language.lower():
                continue

        txt = _pick_text_field(ex)
        if txt is None:
            continue
        txt = _clean_text(txt, normalize_ws=cfg.normalize_ws, strip=cfg.strip)
        txt = _cap_len(txt, cfg.max_chars_per_doc)
        if txt.strip() == "":
            continue
        yield txt
        n += 1
        if cfg.max_docs is not None and n >= cfg.max_docs:
            break


def mix_streams(
    a: Iterable[str],
    b: Iterable[str],
    *,
    ratio_a_to_b: Tuple[int, int] = (2, 1),
) -> Iterator[str]:
    """
    Deterministic round-robin mixing:
      yields a,a,b for ratio (2,1), then repeats.
    """
    ra, rb = ratio_a_to_b
    if ra <= 0 or rb <= 0:
        raise ValueError("ratio_a_to_b must be positive integers, e.g. (2,1)")

    ita = iter(a)
    itb = iter(b)
    while True:
        for _ in range(ra):
            yield next(ita)
        for _ in range(rb):
            yield next(itb)


def take_until_bytes(stream: Iterable[str], max_text_bytes: int) -> Iterator[str]:
    """
    Stop after emitting approximately max_text_bytes UTF-8 bytes total.
    """
    if max_text_bytes <= 0:
        for s in stream:
            yield s
        return

    total = 0
    for s in stream:
        bs = len(s.encode("utf-8", errors="ignore"))
        if bs == 0:
            continue
        if total + bs > max_text_bytes:
            break
        yield s
        total += bs
