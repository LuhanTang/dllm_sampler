#!/usr/bin/env python3
# sampler/decode_ids.py
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import torch
from tokenizers import Tokenizer


def decode_one(pt_path: str, tok: Tokenizer, out_txt: str, replace_newlines: bool = True):
    obj = torch.load(pt_path, map_location="cpu")
    x = obj["x"]  # [N,T] long
    with open(out_txt, "w", encoding="utf-8") as f:
        for i in range(x.shape[0]):
            ids: List[int] = x[i].tolist()
            txt = tok.decode(ids)
            if replace_newlines:
                txt = txt.replace("\n", " ")
            f.write(txt + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_json", type=str, required=True)
    ap.add_argument("--in_pt", type=str, default="", help="single .pt file")
    ap.add_argument("--in_dir", type=str, default="", help="directory containing *_step*.pt")
    ap.add_argument("--glob", type=str, default="samples_step*.pt")
    ap.add_argument("--out_dir", type=str, default="", help="output dir (default: same as input)")
    ap.add_argument("--suffix", type=str, default=".txt", help="output suffix")
    ap.add_argument("--keep_newlines", action="store_true")
    args = ap.parse_args()

    if (not args.in_pt) and (not args.in_dir):
        raise ValueError("Provide --in_pt or --in_dir")

    tok = Tokenizer.from_file(args.tokenizer_json)

    pt_files: List[str] = []
    if args.in_pt:
        pt_files = [args.in_pt]
    else:
        pt_files = sorted(glob.glob(os.path.join(args.in_dir, args.glob)))
        if not pt_files:
            raise FileNotFoundError(f"No pt files matched {args.glob} in {args.in_dir}")

    for pt in pt_files:
        base_dir = os.path.dirname(pt)
        out_dir = args.out_dir or base_dir
        os.makedirs(out_dir, exist_ok=True)

        name = os.path.basename(pt)
        # samples_step32.pt -> samples_step32.txt
        out_txt = os.path.join(out_dir, os.path.splitext(name)[0] + args.suffix)

        decode_one(
            pt_path=pt,
            tok=tok,
            out_txt=out_txt,
            replace_newlines=not args.keep_newlines,
        )
        print(f"[OK] {pt} -> {out_txt}")


if __name__ == "__main__":
    main()
