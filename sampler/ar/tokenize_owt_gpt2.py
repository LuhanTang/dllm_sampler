#!/usr/bin/env python3
# sampler/tokenize_owt_gpt2.py
# ------------------------------------------------------------
# Stage 0: Tokenize OWT using the GPT-2 tokenizer (same tokenizer for gpt2/gpt2-large)
# in a streaming way, and dump token ids to disk incrementally.
#
# Output:
#   out_dir/tokens.bin    (uint16 token ids, concatenated)
#   out_dir/offsets.bin   (uint64 offsets, length = num_docs+1; offsets[0]=0)
#   out_dir/stats.json
#
# Usage example:
# python -m sampler.tokenize_owt_gpt2 \
#   --out_dir sampler/tokenized/owt_gpt2bpe \
#   --owt_name stanford-cs336/owt-sample \
#   --split train \
#   --max_text_bytes 200000000 \
#   --max_docs 0 \
#   --max_chars_per_doc 4096 \
#   --batch_size 64 \
#   --append_eos \
#   --seed 123
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np

from sampler.hf_text_stream import StreamConfig, stream_owt_text, take_until_bytes


def _lazy_import_transformers():
    try:
        from transformers import GPT2TokenizerFast  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'transformers' is required. Install with:\n"
            "  pip install transformers\n"
            f"Original error: {e}"
        )
    return GPT2TokenizerFast


def _write_u16(path: str, arr: np.ndarray) -> None:
    assert arr.dtype == np.uint16
    with open(path, "ab") as f:
        arr.tofile(f)


def _write_u64(path: str, arr: np.ndarray) -> None:
    assert arr.dtype == np.uint64
    with open(path, "ab") as f:
        arr.tofile(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="output directory")
    ap.add_argument("--owt_name", type=str, default="stanford-cs336/owt-sample")
    ap.add_argument("--split", type=str, default="train")

    # streaming limits
    ap.add_argument("--max_text_bytes", type=int, default=200_000_000)
    ap.add_argument("--max_docs", type=int, default=0, help="0 means no limit (bytes budget controls)")
    ap.add_argument("--max_chars_per_doc", type=int, default=4096)
    ap.add_argument("--normalize_ws", action="store_true", help="compress whitespace (optional)")

    # tokenization controls
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--append_eos", action="store_true", help="append GPT-2 EOS token to each doc")
    ap.add_argument("--max_tokens_per_doc", type=int, default=0, help="0 means no truncation by tokens")

    # reproducibility
    ap.add_argument("--seed", type=int, default=123)

    args = ap.parse_args()
    random.seed(args.seed)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tokens_path = os.path.join(out_dir, "tokens.bin")
    offsets_path = os.path.join(out_dir, "offsets.bin")
    stats_path = os.path.join(out_dir, "stats.json")

    # start fresh
    for p in [tokens_path, offsets_path]:
        if os.path.exists(p):
            os.remove(p)

    GPT2TokenizerFast = _lazy_import_transformers()
    tok = GPT2TokenizerFast.from_pretrained("gpt2")  # same tokenizer as gpt2-large
    vocab_size = int(tok.vocab_size)
    eos_id = int(tok.eos_token_id) if tok.eos_token_id is not None else None
    if args.append_eos and eos_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id, cannot --append_eos")

    # Stream config mirrors your previous script style
    cfg = StreamConfig(
        streaming=True,
        split=args.split,
        max_docs=(None if args.max_docs <= 0 else int(args.max_docs)),
        max_chars_per_doc=int(args.max_chars_per_doc),
        normalize_ws=bool(args.normalize_ws),
        strip=True,
    )

    stream = stream_owt_text(args.owt_name, cfg=cfg)
    limited = take_until_bytes(stream, int(args.max_text_bytes))

    # offsets: store as uint64, starting at 0
    cur_offset = np.uint64(0)
    _write_u64(offsets_path, np.array([cur_offset], dtype=np.uint64))

    total_docs = 0
    total_tokens = 0
    total_bytes_seen = 0  # best-effort (approx from utf-8 length)
    max_doc_tokens = 0

    batch: List[str] = []

    def flush_batch(texts: List[str]) -> None:
        nonlocal cur_offset, total_docs, total_tokens, max_doc_tokens, total_bytes_seen

        if not texts:
            return

        # batch encode (fast tokenizer)
        enc = tok(texts, add_special_tokens=False)
        ids_list: List[List[int]] = enc["input_ids"]

        offsets_to_write: List[np.uint64] = []
        token_bufs: List[np.ndarray] = []

        for ids in ids_list:
            if args.max_tokens_per_doc and len(ids) > args.max_tokens_per_doc:
                ids = ids[: int(args.max_tokens_per_doc)]
            if args.append_eos:
                ids = ids + [int(eos_id)]

            # GPT-2 vocab fits in uint16 (50257)
            arr = np.asarray(ids, dtype=np.uint16)
            token_bufs.append(arr)

            cur_offset = np.uint64(int(cur_offset) + int(arr.size))
            offsets_to_write.append(cur_offset)

            total_docs += 1
            total_tokens += int(arr.size)
            max_doc_tokens = max(max_doc_tokens, int(arr.size))

        if token_bufs:
            big = np.concatenate(token_bufs, axis=0) if len(token_bufs) > 1 else token_bufs[0]
            _write_u16(tokens_path, big)

        if offsets_to_write:
            _write_u64(offsets_path, np.asarray(offsets_to_write, dtype=np.uint64))

        # approximate bytes seen from raw texts
        total_bytes_seen += sum(len(t.encode("utf-8", errors="ignore")) for t in texts)

    # Main loop
    for text in limited:
        batch.append(text)
        if len(batch) >= int(args.batch_size):
            flush_batch(batch)
            batch = []

            if total_docs % 2000 == 0 and total_docs > 0:
                print(f"[PROG] docs={total_docs} tokens={total_tokens} cur_offset={int(cur_offset)}")

    flush_batch(batch)

    stats: Dict = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "out_dir": os.path.abspath(out_dir),
        "tokens_path": os.path.abspath(tokens_path),
        "offsets_path": os.path.abspath(offsets_path),
        "owt_name": args.owt_name,
        "split": args.split,
        "seed": int(args.seed),
        "tokenizer": "gpt2 (GPT2TokenizerFast)  # same as gpt2-large tokenizer",
        "vocab_size": int(vocab_size),
        "eos_id": int(eos_id) if eos_id is not None else None,
        "append_eos": bool(args.append_eos),
        "max_text_bytes_budget": int(args.max_text_bytes),
        "max_docs": int(args.max_docs),
        "max_chars_per_doc": int(args.max_chars_per_doc),
        "max_tokens_per_doc": int(args.max_tokens_per_doc),
        "batch_size": int(args.batch_size),
        "normalize_ws": bool(args.normalize_ws),
        "num_docs": int(total_docs),
        "num_tokens": int(total_tokens),
        "max_doc_tokens": int(max_doc_tokens),
        "approx_raw_bytes_seen": int(total_bytes_seen),
        "notes": (
            "tokens.bin is uint16 concatenated token ids; offsets.bin is uint64 offsets "
            "(length num_docs+1, offsets[0]=0). Doc d occupies tokens[offsets[d]:offsets[d+1]]."
        ),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"[OK] wrote:\n  - {tokens_path}\n  - {offsets_path}\n  - {stats_path}")
    print(f"[STATS] docs={total_docs} tokens={total_tokens} vocab={vocab_size} eos={eos_id} append_eos={args.append_eos}")


if __name__ == "__main__":
    main()
