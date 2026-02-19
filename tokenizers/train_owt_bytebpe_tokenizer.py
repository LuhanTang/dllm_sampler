#!/usr/bin/env python3
# tokenizers/train_owt_bytebpe_tokenizer.py
# ------------------------------------------------------------
# Train an OWT-only shared byte-level BPE tokenizer (tokenizer.json)
# using HF streaming text.
#
# Output:
#   <out_dir>/tokenizer.json
#   <out_dir>/train_stats.json
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, Iterator

from sampler.hf_text_stream import (
    StreamConfig,
    stream_owt_text,
    take_until_bytes,
)


def _lazy_import_tokenizers():
    try:
        from tokenizers import Tokenizer  # type: ignore
        from tokenizers.models import BPE  # type: ignore
        from tokenizers.trainers import BpeTrainer  # type: ignore
        from tokenizers.pre_tokenizers import ByteLevel  # type: ignore
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder  # type: ignore
        from tokenizers.normalizers import NFKC  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'tokenizers' is required. Install with:\n"
            "  pip install tokenizers\n"
            f"Original error: {e}"
        )
    return Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder, NFKC


def build_owt_bytebpe_tokenizer(
    *,
    vocab_size: int,
    special_tokens: list[str],
    iterator: Iterator[str],
) -> "Tokenizer":
    Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder, NFKC = _lazy_import_tokenizers()

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.normalizer = NFKC()
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        special_tokens=special_tokens,
        show_progress=True,
        min_frequency=2,
    )

    tok.train_from_iterator(iterator, trainer=trainer)
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="output directory for tokenizer.json")
    ap.add_argument("--vocab_size", type=int, default=4096)
    ap.add_argument("--max_text_bytes", type=int, default=200_000_000)
    ap.add_argument("--seed", type=int, default=123)

    # HF dataset name
    ap.add_argument("--owt_name", type=str, default="stanford-cs336/owt-sample")

    # streaming options
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_docs", type=int, default=0, help="0 means no limit (use bytes budget instead)")
    ap.add_argument("--max_chars_per_doc", type=int, default=4096)

    # cleaning
    ap.add_argument("--normalize_ws", action="store_true", help="compress whitespace for OWT")
    ap.add_argument("--strip", action="store_true", help="strip leading/trailing whitespace per doc")

    args = ap.parse_args()

    random.seed(args.seed)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    cfg = StreamConfig(
        streaming=True,
        split=args.split,
        max_docs=(None if args.max_docs <= 0 else int(args.max_docs)),
        max_chars_per_doc=int(args.max_chars_per_doc),
        normalize_ws=bool(args.normalize_ws),
        strip=bool(args.strip),
    )

    # Stream OWT text (only)
    stream = stream_owt_text(args.owt_name, cfg=cfg)

    # Enforce byte budget
    limited = take_until_bytes(stream, int(args.max_text_bytes))

    # Minimal special tokens
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]

    tok = build_owt_bytebpe_tokenizer(
        vocab_size=int(args.vocab_size),
        special_tokens=special_tokens,
        iterator=limited,
    )

    tok_path = os.path.join(out_dir, "tokenizer.json")
    tok.save(tok_path)

    stats: Dict = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "out_dir": os.path.abspath(out_dir),
        "tokenizer_path": os.path.abspath(tok_path),
        "vocab_size": int(tok.get_vocab_size()),
        "requested_vocab_size": int(args.vocab_size),
        "special_tokens": special_tokens,
        "max_text_bytes": int(args.max_text_bytes),
        "owt_name": args.owt_name,
        "split": args.split,
        "max_docs": int(args.max_docs),
        "max_chars_per_doc": int(args.max_chars_per_doc),
        "normalize_ws": bool(args.normalize_ws),
        "strip": bool(args.strip),
        "seed": int(args.seed),
    }
    with open(os.path.join(out_dir, "train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    print(f"[OK] tokenizer saved: {tok_path}")
    print(f"[TOK] vocab_size={tok.get_vocab_size()}  (requested={args.vocab_size})")
    print(f"[DATA] OWT={args.owt_name} split={args.split}")
    print(f"[BUDGET] max_text_bytes={args.max_text_bytes}")


if __name__ == "__main__":
    main()
