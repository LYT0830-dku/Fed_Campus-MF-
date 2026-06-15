#!/usr/bin/env python3
"""Run a minimal PyTorch/PEFT DPO loop for MF comparison."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


warnings.filterwarnings("ignore", message="Failed to load image Python extension")


@dataclass
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    chosen_response_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor
    rejected_response_mask: torch.Tensor


def read_pairs(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "prompt": str(item["prompt"]),
                    "chosen": str(item["chosen"]),
                    "rejected": str(item["rejected"]),
                }
            )
    return rows


def encode_branch(tokenizer, prompt: str, response: str, seq_len: int, append_eos: bool) -> tuple[list[int], list[int], list[int]] | None:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if append_eos:
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("append_eos requires tokenizer.eos_token_id")
        if not response_ids or response_ids[-1] != eos:
            response_ids.append(eos)
    if not response_ids:
        return None

    ids = prompt_ids + response_ids
    response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    if len(ids) > seq_len:
        ids = ids[:seq_len]
        response_mask = response_mask[:seq_len]
    valid_response = sum(response_mask[1:])
    if valid_response == 0:
        return None

    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    attention = [1] * len(ids)
    while len(ids) < seq_len:
        ids.append(pad)
        attention.append(0)
        response_mask.append(0)
    return ids, attention, response_mask


def build_batches(tokenizer, rows: list[dict[str, str]], seq_len: int, append_eos: bool) -> list[PreferenceBatch]:
    batches: list[PreferenceBatch] = []
    for row in rows:
        chosen = encode_branch(tokenizer, row["prompt"], row["chosen"], seq_len, append_eos)
        rejected = encode_branch(tokenizer, row["prompt"], row["rejected"], seq_len, append_eos)
        if chosen is None or rejected is None:
            continue
        if chosen[0] == rejected[0] and chosen[2] == rejected[2]:
            continue
        c_ids, c_attn, c_resp = chosen
        r_ids, r_attn, r_resp = rejected
        batches.append(
            PreferenceBatch(
                torch.tensor([c_ids], dtype=torch.long),
                torch.tensor([c_attn], dtype=torch.long),
                torch.tensor([c_resp], dtype=torch.bool),
                torch.tensor([r_ids], dtype=torch.long),
                torch.tensor([r_attn], dtype=torch.long),
                torch.tensor([r_resp], dtype=torch.bool),
            )
        )
    return batches


def sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target_ids = input_ids[:, 1:]
    target_mask = response_mask[:, 1:]
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return (gathered * target_mask.float()).sum(dim=-1)


def dpo_loss(policy_chosen_logp, policy_rejected_logp, ref_chosen_logp, ref_rejected_logp, beta: float):
    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    margin = (pi_logratios - ref_logratios).mean()
    acc = ((pi_logratios - ref_logratios) > 0).float().mean()
    return loss, margin, acc


def move_batch(batch: PreferenceBatch, device: torch.device) -> PreferenceBatch:
    return PreferenceBatch(
        batch.chosen_input_ids.to(device),
        batch.chosen_attention_mask.to(device),
        batch.chosen_response_mask.to(device),
        batch.rejected_input_ids.to(device),
        batch.rejected_attention_mask.to(device),
        batch.rejected_response_mask.to(device),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--preference-jsonl", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_pairs(args.preference_jsonl)
    batches = build_batches(tokenizer, rows, args.seq_len, append_eos=True)
    if not batches:
        raise RuntimeError("no usable preference batches after tokenization")

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(policy, lora_config)
    policy.train()

    reference = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    print("========== PyTorch Qwen DPO LoRA Run ==========")
    print(f"model_dir={args.model_dir}")
    print(f"preference_jsonl={args.preference_jsonl}")
    print(f"loaded_pairs={len(batches)} seq_len={args.seq_len} max_steps={args.max_steps}")
    total = 0.0
    for step in range(1, args.max_steps + 1):
        batch = move_batch(batches[(step - 1) % len(batches)], device)
        t0 = time.perf_counter()
        chosen_logits = policy(
            input_ids=batch.chosen_input_ids,
            attention_mask=batch.chosen_attention_mask,
        ).logits
        rejected_logits = policy(
            input_ids=batch.rejected_input_ids,
            attention_mask=batch.rejected_attention_mask,
        ).logits
        with torch.no_grad():
            ref_chosen_logits = reference(
                input_ids=batch.chosen_input_ids,
                attention_mask=batch.chosen_attention_mask,
            ).logits
            ref_rejected_logits = reference(
                input_ids=batch.rejected_input_ids,
                attention_mask=batch.rejected_attention_mask,
            ).logits

        policy_chosen = sequence_logprob(chosen_logits, batch.chosen_input_ids, batch.chosen_response_mask)
        policy_rejected = sequence_logprob(rejected_logits, batch.rejected_input_ids, batch.rejected_response_mask)
        ref_chosen = sequence_logprob(ref_chosen_logits, batch.chosen_input_ids, batch.chosen_response_mask)
        ref_rejected = sequence_logprob(ref_rejected_logits, batch.rejected_input_ids, batch.rejected_response_mask)
        loss, margin, acc = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, args.beta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        seconds = time.perf_counter() - t0
        total += seconds
        response_tokens = int(batch.chosen_response_mask[:, 1:].sum() + batch.rejected_response_mask[:, 1:].sum())
        print(
            f"[step {step}/{args.max_steps}] "
            f"loss={loss.item():.4f} margin={margin.item():.4f} "
            f"acc={acc.item():.4f} response_tokens={response_tokens} time_s={seconds:.4f}",
            flush=True,
        )

    print(f"mean_step_time_s={total / args.max_steps:.4f} total_train_time_s={total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
