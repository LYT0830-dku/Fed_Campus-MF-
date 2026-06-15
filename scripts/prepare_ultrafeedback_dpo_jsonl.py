#!/usr/bin/env python3
"""Convert UltraFeedback Binarized parquet rows into MF DPO JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _message_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return str(messages or "")
    assistant_parts: list[str] = []
    fallback_parts: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "")
        role = str(item.get("role") or "")
        if role == "assistant":
            assistant_parts.append(content)
        elif content:
            fallback_parts.append(content)
    if assistant_parts:
        return "\n".join(part.strip() for part in assistant_parts if part.strip()).strip()
    return "\n".join(part.strip() for part in fallback_parts if part.strip()).strip()


def convert(
    input_path: Path,
    output_path: Path,
    limit: int,
    max_prompt_chars: int,
    max_response_chars: int,
) -> int:
    table = pq.read_table(input_path, columns=["prompt", "chosen", "rejected"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in table.to_pylist():
            prompt = str(row.get("prompt") or "").strip()
            chosen = _message_content(row.get("chosen"))
            rejected = _message_content(row.get("rejected"))
            if not prompt or not chosen or not rejected or chosen == rejected:
                continue
            if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
                continue
            if max_response_chars > 0 and (
                len(chosen) > max_response_chars or len(rejected) > max_response_chars
            ):
                continue
            out.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            if limit > 0 and written >= limit:
                break
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    parser.add_argument("--max-response-chars", type=int, default=0)
    args = parser.parse_args()

    written = convert(
        args.input,
        args.output,
        args.limit,
        args.max_prompt_chars,
        args.max_response_chars,
    )
    print(f"wrote {written} preference pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
