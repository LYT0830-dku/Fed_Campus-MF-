#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def disable_dropout_modules(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


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
    return model


def build_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, ignore_index: int) -> torch.Tensor:
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, ignore_index)
    labels[:, 0] = ignore_index
    previous_token_is_real = torch.zeros_like(attention_mask, dtype=torch.bool)
    previous_token_is_real[:, 1:] = attention_mask[:, :-1] > 0
    labels = labels.masked_fill(~previous_token_is_real, ignore_index)
    return labels


def canonical_param_name(name: str) -> str:
    for marker in ("model.layers.", "transformer.h.", "h."):
        idx = name.find(marker)
        if idx >= 0:
            tail = name[idx:]
            if tail.startswith("model."):
                tail = tail[len("model.") :]
            if tail.startswith("transformer.h."):
                tail = "blocks." + tail[len("transformer.h.") :]
            if tail.startswith("h."):
                tail = "blocks." + tail[len("h.") :]
            tail = tail.replace(".attn.c_attn.", ".attn.qkv.")
            tail = tail.replace(".attn.c_proj.", ".attn.proj.")
            tail = tail.replace(".mlp.c_fc.", ".mlp.fc_in.")
            tail = tail.replace(".mlp.c_proj.", ".mlp.fc_out.")
            return tail
    prefixes = (
        "base_model.model.",
        "model.",
    )
    out = name
    for prefix in prefixes:
        if out.startswith(prefix):
            out = out[len(prefix) :]
    return out


def infer_model_type(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return ""
    try:
        return str(json.loads(config_path.read_text(encoding="utf-8")).get("model_type", ""))
    except json.JSONDecodeError:
        return ""


def uses_mf_in_rank_layout(model_type: str) -> bool:
    model_type = model_type.lower()
    return model_type.startswith("qwen") or model_type.startswith("llama") or model_type == "gpt2"


def tensor_checksum(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().cpu().float().reshape(-1)
    if flat.numel() == 0:
        return {"sum": 0.0, "mean": 0.0, "l2": 0.0, "max_abs": 0.0}
    return {
        "sum": float(flat.sum().item()),
        "mean": float(flat.mean().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "max_abs": float(flat.abs().max().item()),
    }


def to_mf_layout(name: str, tensor: torch.Tensor, transpose_lora: bool) -> torch.Tensor:
    if not transpose_lora:
        return tensor.detach().cpu().float()
    if ".lora_A." in name or ".lora_B." in name:
        if tensor.ndim != 2:
            raise ValueError(f"LoRA tensor must be 2D for MF layout conversion: {name}")
        return tensor.detach().cpu().float().t().contiguous()
    return tensor.detach().cpu().float()


def lora_tensor_records(model: torch.nn.Module, transpose_lora: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        canonical = canonical_param_name(name)
        mf_tensor = to_mf_layout(canonical, param, transpose_lora)
        records.append(
            {
                "name": canonical,
                "peft_name": name,
                "peft_shape": list(param.shape),
                "mf_shape": list(mf_tensor.shape),
                "checksum": tensor_checksum(mf_tensor),
                "values": mf_tensor.reshape(-1).tolist(),
            }
        )
    records.sort(key=lambda item: item["name"])
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PyTorch/Transformers/PEFT one-step LoRA fixture for "
            "MobileFineTuner family-level training alignment."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Local HuggingFace model snapshot.")
    parser.add_argument("--output", required=True, help="Output JSON fixture path.")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ignore-index", type=int, default=-100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-module", action="append", default=None)
    parser.add_argument(
        "--keep-model-dropout",
        action="store_true",
        help=(
            "Keep stochastic base-model dropout in the PyTorch fixture. By "
            "default base dropout is disabled for deterministic MF parity; "
            "LoRA dropout still follows --dropout because it is attached after "
            "the base model is normalized."
        ),
    )
    parser.add_argument(
        "--mf-layout",
        choices=("auto", "peft", "in_rank"),
        default="auto",
        help=(
            "How LoRA tensors should be stored for MF consumption. auto uses "
            "[in,r]/[r,out] for Qwen/Llama/GPT-2 and PEFT layout otherwise."
        ),
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("This fixture generator requires `peft`; install it in the Python environment.") from exc

    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(model_dir)
    prompts = args.prompt if args.prompt else ["MobileFineTuner PEFT alignment prompt."]
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )
    input_ids = encoded["input_ids"].to(torch.long)
    attention_mask = encoded["attention_mask"].to(torch.long)
    labels = build_labels(input_ids, attention_mask, args.ignore_index)

    model_type = infer_model_type(model_dir)
    target_modules = args.target_module or ["q_proj", "k_proj", "v_proj", "o_proj"]
    base_model = load_model(model_dir)
    if not args.keep_model_dropout:
        disable_dropout_modules(base_model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    transpose_lora = (
        uses_mf_in_rank_layout(model_type) if args.mf_layout == "auto" else args.mf_layout == "in_rank"
    )
    initial_tensors = lora_tensor_records(model, transpose_lora)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    if args.max_grad_norm > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm).item())
    else:
        grad_norm = 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after_tensors = lora_tensor_records(model, transpose_lora)

    record = {
        "schema": "mobilefinetuner.peft_lora_step.v1",
        "model_dir": str(model_dir),
        "model_type": model_type,
        "prompts": prompts,
        "max_length": int(args.max_length),
        "batch_size": int(input_ids.shape[0]),
        "sequence_length": int(input_ids.shape[1]),
        "ignore_index": int(args.ignore_index),
        "input_ids": input_ids.reshape(-1).tolist(),
        "attention_mask": attention_mask.reshape(-1).tolist(),
        "labels": labels.reshape(-1).tolist(),
        "valid_shifted_label_count": int((labels[:, 1:] != args.ignore_index).sum().item()),
        "target_modules": target_modules,
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "base_model_dropout_disabled": not bool(args.keep_model_dropout),
        "mf_lora_layout": "in_rank" if transpose_lora else "peft",
        "expected_loss": float(loss.detach().cpu().item()),
        "grad_norm_before_clip": grad_norm,
        "trainable_tensor_count": len(initial_tensors),
        "initial_lora_tensors": initial_tensors,
        "expected_after_step_lora_tensors": after_tensors,
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_class": base_model.__class__.__name__,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote PEFT LoRA step fixture to {output}")
    print(
        f"loss={record['expected_loss']:.8f} "
        f"batch={record['batch_size']}x{record['sequence_length']} "
        f"trainable_tensors={record['trainable_tensor_count']} "
        f"layout={record['mf_lora_layout']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
