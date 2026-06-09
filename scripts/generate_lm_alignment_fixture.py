#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_dir: Path):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
        device_map=None,
    )
    model.eval()
    return model


def build_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
    """Build labels matching MobileFineTuner's CausalLMBatch policy.

    lm_cross_entropy applies the HuggingFace causal shift internally:
    logits[:, s, :] predicts labels[:, s + 1]. For padded batches, especially
    left-padded ones, the first real token after padding should not be
    supervised from a padding-position query. This mirrors
    make_causal_lm_batch(...): label position s is valid only when token s and
    token s - 1 are both real.
    """
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    labels[:, 0] = ignore_index
    previous_token_is_real = torch.zeros_like(attention_mask, dtype=torch.bool)
    previous_token_is_real[:, 1:] = attention_mask[:, :-1] > 0
    labels = labels.masked_fill(~previous_token_is_real, ignore_index)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PyTorch/HuggingFace causal-LM alignment fixture for "
            "MobileFineTuner AutoModelForCausalLM."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Local HuggingFace model snapshot.")
    parser.add_argument("--output", required=True, help="Output JSON fixture path.")
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help=(
            "Prompt used for tokenizer/forward/loss alignment. Pass multiple "
            "times to generate a padded batch fixture."
        ),
    )
    parser.add_argument("--max-length", type=int, default=0, help="Optional max_length padding/truncation.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of last-token logits to record.")
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--abs-tol", type=float, default=5e-3)
    parser.add_argument("--last-logits-batch-index", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    tokenizer = load_tokenizer(model_dir)
    prompts = args.prompt if args.prompt else ["MobileFineTuner alignment prompt."]
    if len(prompts) > 1 and args.max_length <= 0:
        raise ValueError("multiple prompts require --max-length so the batch is rectangular")
    if not (0 <= args.last_logits_batch_index < len(prompts)):
        raise ValueError("--last-logits-batch-index is out of range for the prompt batch")

    encode_kwargs = {
        "add_special_tokens": True,
        "return_attention_mask": True,
        "return_tensors": "pt",
    }
    if args.max_length > 0:
        encode_kwargs.update(
            {
                "padding": "max_length",
                "truncation": True,
                "max_length": args.max_length,
            }
        )

    encoded = tokenizer(prompts, **encode_kwargs)
    input_ids = encoded["input_ids"].to(torch.long)
    attention_mask = encoded["attention_mask"].to(torch.long)
    if input_ids.shape[1] < 2:
        raise ValueError("alignment prompt must tokenize to at least two tokens")

    labels = build_labels(input_ids, attention_mask, args.ignore_index)

    model = load_model(model_dir)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = output.logits.detach().cpu().float()
        loss = float(output.loss.detach().cpu())

    batch_index = int(args.last_logits_batch_index)
    valid_positions = torch.nonzero(attention_mask[batch_index] > 0, as_tuple=False).flatten()
    last_pos = int(valid_positions[-1].item())
    last_logits = logits[batch_index, last_pos]
    topk = torch.topk(last_logits, k=min(args.top_k, last_logits.numel()))

    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}

    record = {
        "schema": "mobilefinetuner.lm_alignment.v1",
        "model_dir": str(model_dir),
        "model_type": str(config.get("model_type", "")),
        "prompt": prompts[0],
        "prompts": prompts,
        "add_special_tokens": True,
        "max_length": int(args.max_length),
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "ignore_index": int(args.ignore_index),
        "abs_tol": float(args.abs_tol),
        "input_ids": input_ids.reshape(-1).tolist(),
        "attention_mask": attention_mask.reshape(-1).tolist(),
        "labels": labels.reshape(-1).tolist(),
        "valid_shifted_label_count": int((labels[:, 1:] != args.ignore_index).sum().item()),
        "expected_loss": loss,
        "last_logits_batch_index": batch_index,
        "last_logits_position": last_pos,
        "last_logits_topk_ids": topk.indices.tolist(),
        "last_logits_topk_values": [float(v) for v in topk.values.tolist()],
        "last_logits_sum": float(last_logits.sum().item()),
        "last_logits_mean": float(last_logits.mean().item()),
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote LM alignment fixture to {output}")
    print(
        f"loss={record['expected_loss']:.8f} "
        f"batch={record['batch_size']}x{record['sequence_length']} "
        f"topk_ids={record['last_logits_topk_ids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
