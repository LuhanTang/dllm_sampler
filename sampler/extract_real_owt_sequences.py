#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import numpy as np
import torch

def load_memmap_tokens(tokens_path: str) -> np.memmap:
    # uint16 token ids
    return np.memmap(tokens_path, dtype=np.uint16, mode="r")

def load_offsets(offsets_path: str) -> np.ndarray:
    # uint64 offsets, length = num_docs + 1
    return np.fromfile(offsets_path, dtype=np.uint64)

@torch.no_grad()
def sample_real_sequences_within_docs(
    tokens: np.memmap,
    offsets: np.ndarray,
    T: int,
    N: int,
    seed: int = 123,
    min_doc_len: int | None = None,
) -> torch.LongTensor:
    """
    Randomly sample N sequences of length T from within-document token spans.
    Never crosses document boundaries.
    """
    rng = np.random.default_rng(seed)
    num_docs = len(offsets) - 1
    if num_docs <= 0:
        raise RuntimeError("offsets.bin seems empty.")

    if min_doc_len is None:
        min_doc_len = T

    # Precompute eligible docs (length >= min_doc_len)
    doc_starts = offsets[:-1].astype(np.int64)
    doc_ends = offsets[1:].astype(np.int64)
    doc_lens = doc_ends - doc_starts
    eligible = np.nonzero(doc_lens >= min_doc_len)[0]
    if eligible.size == 0:
        raise RuntimeError(f"No documents with length >= {min_doc_len} tokens.")

    out = torch.empty((N, T), dtype=torch.long)
    for i in range(N):
        d = int(rng.choice(eligible))
        s = int(doc_starts[d])
        e = int(doc_ends[d])
        # sample a start position so that [pos, pos+T) stays in [s,e)
        pos = int(rng.integers(low=s, high=e - T + 1))
        seq = np.asarray(tokens[pos : pos + T], dtype=np.int64)
        out[i] = torch.from_numpy(seq)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=str, required=True)
    ap.add_argument("--offsets", type=str, required=True)
    ap.add_argument("--out_pt", type=str, required=True)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--decode", action="store_true", help="also save decoded text using GPT-2 tokenizer")
    args = ap.parse_args()

    tokens = load_memmap_tokens(args.tokens)
    offsets = load_offsets(args.offsets)

    real_ids = sample_real_sequences_within_docs(
        tokens=tokens,
        offsets=offsets,
        T=int(args.T),
        N=int(args.N),
        seed=int(args.seed),
    )

    payload = {
        "real_samples_ids": real_ids,   # [N,T] torch.long
        "config": {
            "tokens_path": os.path.abspath(args.tokens),
            "offsets_path": os.path.abspath(args.offsets),
            "T": int(args.T),
            "N": int(args.N),
            "seed": int(args.seed),
            "within_doc_only": True,
        },
    }

    if args.decode:
        try:
            from transformers import GPT2TokenizerFast  # type: ignore
        except Exception as e:
            raise RuntimeError("Need transformers for --decode. pip install transformers") from e
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        payload["real_samples_text"] = [tok.decode(seq.tolist()) for seq in real_ids]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pt)) or ".", exist_ok=True)
    torch.save(payload, args.out_pt)
    print(f"[OK] saved real samples to {args.out_pt}  shape={tuple(real_ids.shape)} decode={args.decode}")

if __name__ == "__main__":
    main()
