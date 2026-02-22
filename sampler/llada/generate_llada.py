# ------------------------------------------------------------
# LLaDA sampling (merged version)
# - Supports batch prompts + (left/right) padding via attention_mask
# - Forbids predicting MASK as output token
# - Optional EOS/EOT suppression (fixed implementation)
# - Keeps semi-autoregressive block remasking logic
# ------------------------------------------------------------

from __future__ import annotations

import torch
import numpy as np
import torch.nn.functional as F

#from transformers import AutoTokenizer, AutoModel


# ------------------------------
# Gumbel-max style noise
# ------------------------------
def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    NOTE: This matches current implementation.
    If temperature==0: no noise.
    Otherwise uses float64 for "low precision gumbel max" style effect described in their text.
    """
    if temperature == 0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits64.exp() / gumbel_noise


# ------------------------------
# Per-step number of tokens to "transfer" (unmask) inside a block
# ------------------------------
def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    mask_index: bool tensor [B, block_len], True for masked positions
    returns: int64 tensor [B, steps], how many tokens to unmask per step for each sample
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # [B,1]
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base

    # distribute the remainder to early steps
    for i in range(mask_num.size(0)):
        r = int(remainder[i].item())
        if r > 0:
            num_transfer_tokens[i, :r] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    logits_fn=None,
    # Appendix-B.4 style options (robust + bug-fixed)
    eos_id: int = 126081,
    eot_id: int = 126348,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    forbid_mask_prediction: bool = False,
):
    """
    Args:
        model: HF model with .logits output (mask predictor).
        prompt: LongTensor [B, L_prompt]
        attention_mask: optional Long/Bool [B, L_prompt], 1 for real, 0 for padding
        steps: total sampling steps across all blocks (must be divisible by num_blocks)
        gen_length: how many new tokens (masked slots) to generate
        block_length: if < gen_length => semi-autoregressive blocks
        temperature: noise knob for gumbel-like sampling
        cfg_scale: classifier-free guidance scale (0 disables)
        remasking: 'low_confidence' or 'random'
        mask_id: token id for [MASK]
        logits_fn: optional hook:
            logits = logits_fn(x, attention_mask) with shape [B, L, V]
        eos/eot knobs: optionally suppress selecting EOS/EOT (Appendix B.4)
        forbid_mask_prediction: forbid predicting MASK as an output class

    Returns:
        x: LongTensor [B, L_prompt + gen_length]
    """
    device = prompt.device
    B, Lp = prompt.shape

    # Initialize x = [prompt | MASK ... MASK]
    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=device)
    if Lp > 0:
        x[:, :Lp] = prompt.clone()

    # Extend attention mask to include generated positions (all "real" tokens after filled)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        gen_attn = torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, gen_attn], dim=-1)  # [B, Lp+gen_length]

    prompt_index = (x != mask_id)  # True on prompt tokens + (later) filled tokens

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        start = Lp + num_block * block_length
        end = Lp + (num_block + 1) * block_length

        block_mask_index = (x[:, start:end] == mask_id)  # [B, block_length]
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)  # [B, steps_per_block]

        for i in range(steps_per_block):
            mask_index = (x == mask_id)  # [B, L_total]

            # ---- compute logits ----
            if cfg_scale > 0.0:
                # conditional/unconditional batch concat
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None

                if logits_fn is None:
                    logits_all = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits_all = logits_fn(x_, attention_mask_)
                logits, un_logits = torch.chunk(logits_all, 2, dim=0)

                # standard CFG interpolation
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                if logits_fn is None:
                    logits = model(x, attention_mask=attention_mask).logits
                else:
                    logits = logits_fn(x, attention_mask)

            # Safety: do not allow predicting MASK as a normal token
            if forbid_mask_prediction:
                logits = logits.clone()
                logits[..., mask_id] = -1e30

            # Optional: forbid EOS by setting its logits to -inf
            if logits_eos_inf:
                logits = logits.clone()
                logits[..., eos_id] = -torch.inf

            # ---- sample x0 via argmax on "logits_with_noise" ----
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L_total]

            # Optional: forbid selecting EOS/EOT in confidence stage
            # (we implement this by pushing their confidence to -inf)
            forbid_ids_for_conf = None
            if confidence_eos_eot_inf:
                forbid_ids_for_conf = {eos_id, eot_id}

            # ---- compute confidence for remasking ----
            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)  # [B, L, V]
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
                if forbid_ids_for_conf is not None:
                    # if x0 chooses EOS/EOT, set its confidence to -inf
                    bad = torch.zeros_like(x0, dtype=torch.bool)
                    for tid in forbid_ids_for_conf:
                        bad |= (x0 == tid)
                    x0_p = x0_p.masked_fill(bad, -torch.inf)
            elif remasking == "random":
                x0_p = torch.rand((B, x0.shape[1]), device=device)
                if forbid_ids_for_conf is not None:
                    bad = torch.zeros_like(x0, dtype=torch.bool)
                    for tid in forbid_ids_for_conf:
                        bad |= (x0 == tid)
                    x0_p = x0_p.masked_fill(bad, -torch.inf)
            else:
                raise NotImplementedError(remasking)

            # Only allow updates inside current block (and prompt is never mask anyway)
            x0_p[:, end:] = -np.inf

            # Only replace masked positions
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # ---- pick top-k positions to "transfer" this step ----
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(B):
                k = int(num_transfer_tokens[j, i].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = "cuda"

    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
    )

    # left-padding is simpler with this sampler
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    # If PAD == MASK, the sampler needs special handling. Keep this guard.
    assert tokenizer.pad_token_id != 126336

    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
    ]

    # Instruct template
    messages = [{"role": "user", "content": p} for p in prompts]
    prompts = [
        tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False)
        for m in messages
    ]

    encoded = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    out = generate(
        model,
        input_ids,
        attention_mask=attention_mask,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=126336,
        logits_fn=None,                 # <-- your hook goes here if needed
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,   # <-- safe (bug-fixed) implementation
        forbid_mask_prediction=True,
    )

    # NOTE: output slice uses the *padded* prompt length (same across batch)
    output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output_text:
        print(o)
        print("-" * 50)


if __name__ == "__main__":
    main()
