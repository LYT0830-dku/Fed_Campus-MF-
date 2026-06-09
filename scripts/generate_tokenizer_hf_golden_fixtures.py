#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent


MODEL_SPECS = {
    "gpt2_small": {
        "model_type": "gpt2",
        "env": ["GPT2_SMALL_MODEL_DIR"],
        "root_names": ["gpt2", "GPT2-124M", "gpt2-small"],
        "local": REPO_ROOT / "examples/gpt2_small_lora_finetune/pretrained",
        "required": ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
        "add_special_tokens": False,
    },
    "gpt2_medium": {
        "model_type": "gpt2",
        "env": ["GPT2_MEDIUM_MODEL_DIR"],
        "root_names": ["gpt2-medium", "gpt2-medium_official", "GPT2-355M", "gpt2_medium"],
        "local": REPO_ROOT / "examples/gpt2_medium_lora_finetune/pretrained",
        "required": ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
        "add_special_tokens": False,
    },
    "qwen": {
        "model_type": "qwen2",
        "env": ["QWEN_MODEL_DIR"],
        "root_names": ["Qwen2.5-0.5B", "Qwen3-0.6B", "qwen2.5-0.5b", "qwen"],
        "local": REPO_ROOT / "examples/qwen_lora_finetune/pretrained",
        "required": ["config.json", "tokenizer.json", "vocab.json", "merges.txt"],
        "add_special_tokens": False,
    },
    "gemma_270m": {
        "model_type": "gemma",
        "env": ["GEMMA_270M_MODEL_DIR"],
        "root_names": ["gemma-3-270m", "Gemma3-270M/gemma-3-270m", "Gemma3-270M"],
        "local": REPO_ROOT / "examples/gemma_3_270m_lora_finetune/pretrained",
        "required": ["config.json", "tokenizer.json"],
        "add_special_tokens": True,
    },
    "gemma_1b_pt": {
        "model_type": "gemma",
        "env": ["GEMMA_1B_PT_MODEL_DIR", "GEMMA_1B_MODEL_DIR"],
        "root_names": [
            "gemma-3-1b-pt",
            "Gemma3-1B-PT/gemma-3-1b-pt",
            "Gemma3-1B-PT",
            "gemma-3-1b",
            "Gemma3-1B/gemma-3-1b",
            "Gemma3-1B",
        ],
        "local": REPO_ROOT / "examples/gemma_3_1b_pt_lora_finetune/pretrained",
        "required": ["config.json", "tokenizer.json"],
        "add_special_tokens": True,
    },
    "llama_3_2_1b_instruct": {
        "model_type": "llama",
        "env": [
            "LLAMA_MODEL_DIR",
            "LLAMA_3_2_1B_MODEL_DIR",
            "LLAMA_3_2_1B_INSTRUCT_MODEL_DIR",
        ],
        "root_names": [
            "Llama-3.2-1B-Instruct",
            "Meta-Llama-3.2-1B-Instruct",
            "Llama3.2-1B-Instruct",
            "llama-3.2-1b-instruct",
            "llama",
        ],
        "local": REPO_ROOT / "examples/llama_3_2_1b_lora_finetune/pretrained",
        "required": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        "add_special_tokens": True,
    },
}


BASE_CASES = [
    ("empty", ""),
    ("hello", "Hello"),
    (
        "qnli_prompt",
        "Question: Does the sentence contain a negation?\n"
        "Sentence: The cat did not sleep.\n"
        "Answer:",
    ),
    ("training_params", "MobileFineTuner qkvo LoRA, seq_len=64, lr=2e-4."),
    ("cjk_mixed", "\u4e2d\u6587 mixed English 123!"),
    ("whitespace_edges", "  leading\tand trailing \n"),
    ("emoji", "Fine-tune on-device 📱🚀"),
    ("accented", "Café naïve façade"),
]

QWEN_CASES = [
    (
        "qwen_chat_tokens",
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    ),
]


def env_path(names: Iterable[str]) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()
    return None


def model_root() -> Path | None:
    value = os.environ.get("MFT_MODEL_ROOT") or os.environ.get("MOBILEFINETUNER_MODEL_ROOT")
    return Path(value).expanduser() if value else None


def has_required_files(path: Path, required: list[str]) -> bool:
    return path.is_dir() and all((path / rel).is_file() for rel in required)


def candidates_for(model_key: str) -> list[Path]:
    spec = MODEL_SPECS[model_key]
    candidates: list[Path] = []
    explicit = env_path(spec["env"])
    if explicit:
        candidates.append(explicit)
    root = model_root()
    if root:
        candidates.extend(root / name for name in spec["root_names"])
    candidates.append(spec["local"])
    return candidates


def resolve_tokenizer_dir(model_key: str) -> Path | None:
    spec = MODEL_SPECS[model_key]
    for candidate in candidates_for(model_key):
        if has_required_files(candidate, spec["required"]):
            return candidate.resolve()
    return None


def default_cases(model_key: str) -> list[tuple[str, str]]:
    cases = list(BASE_CASES)
    if model_key.startswith("gpt2"):
        cases.append(("literal_eos", "Hello<|endoftext|>world"))
    if model_key == "qwen":
        cases.extend(QWEN_CASES)
        cases.append(("literal_eos", "Hello<|endoftext|>world"))
    if model_key.startswith("llama"):
        cases.append(("literal_llama_specials", "<|begin_of_text|>Hello<|eot_id|>"))
    return cases


def config_model_type(model_dir: Path, fallback: str) -> str:
    config_path = model_dir / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback

    value = data.get("model_type")
    return value if isinstance(value, str) and value else fallback


def load_hf_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to generate HuggingFace tokenizer fixtures. "
            "Install it in the host environment, then rerun this script."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_record(
    model_key: str,
    model_dir: Path,
    case_name: str,
    text: str,
    add_special_tokens_override: bool | None = None,
) -> dict:
    spec = MODEL_SPECS[model_key]
    tokenizer = load_hf_tokenizer(model_dir)
    add_special_tokens = (
        bool(spec["add_special_tokens"])
        if add_special_tokens_override is None
        else bool(add_special_tokens_override)
    )
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    max_length = 16
    padded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    return {
        "schema": "mobilefinetuner.tokenizer_hf_golden.v1",
        "model_key": model_key,
        "model_type": config_model_type(model_dir, str(spec["model_type"])),
        "model_dir": str(model_dir),
        "case_name": case_name,
        "add_special_tokens": add_special_tokens,
        "max_length": max_length,
        "text": text,
        "expected_ids": ids,
        "expected_padded_ids": list(padded["input_ids"]),
        "expected_attention_mask": list(padded["attention_mask"]),
        "expected_decode": decoded,
        "tokenizer_class": tokenizer.__class__.__name__,
    }


def extra_records(model_key: str, model_dir: Path) -> list[dict]:
    if model_key.startswith("gpt2") or model_key == "qwen":
        return [
            build_record(
                model_key,
                model_dir,
                "add_special_tokens_true_no_template_insertion",
                "Hello",
                add_special_tokens_override=True,
            )
        ]
    if model_key.startswith("llama"):
        return [
            build_record(
                model_key,
                model_dir,
                "add_special_tokens_false_no_bos",
                "Hello",
                add_special_tokens_override=False,
            )
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate HuggingFace tokenizer golden JSONL fixtures for MobileFineTuner."
    )
    parser.add_argument(
        "--output",
        default="runs/tokenizer_golden/hf_tokenizer_golden.jsonl",
        help="Output JSONL path. The default lives under ignored runs/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt2_small", "gpt2_medium", "qwen", "gemma_270m", "gemma_1b_pt"],
        choices=sorted(MODEL_SPECS),
        help="Tokenizer/model keys to include.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested tokenizer assets are missing.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    missing: list[str] = []
    for model_key in args.models:
        model_dir = resolve_tokenizer_dir(model_key)
        if model_dir is None:
            checked = ", ".join(str(path) for path in candidates_for(model_key))
            missing.append(f"{model_key}: checked {checked}")
            continue
        for case_name, text in default_cases(model_key):
            records.append(build_record(model_key, model_dir, case_name, text))
        records.extend(extra_records(model_key, model_dir))

    if missing and args.strict:
        for item in missing:
            print(f"[MISSING] {item}")
        return 1

    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    print(f"Wrote {len(records)} tokenizer golden cases to {output}")
    if missing:
        print("Skipped missing tokenizer assets:")
        for item in missing:
            print(f"  - {item}")
    if not records:
        print("No records were generated. Set MFT_MODEL_ROOT or per-model tokenizer dirs.")
        return 1 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
