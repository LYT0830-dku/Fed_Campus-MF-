# Current Family And Task Matrix

This document defines the maintained support boundary for the current release
line. New model families should not be added until these rows remain green under
the same gates.

## Families

| Family | Representative snapshots | Release status |
| --- | --- | --- |
| GPT-2 | `gpt2`, `gpt2-medium` | Supported |
| Gemma | `gemma-3-270m`, `gemma-3-1b-pt` | Supported |
| Qwen | `Qwen2.5-0.5B` | Supported |
| Llama | `Llama-3.2-1B-Instruct` style snapshots | Experimental validation only |
| Mistral | `mistral` configs | Recognized only; explicitly rejected by AutoModel and Tokenizer |

## Dataset Artifacts

MobileFineTuner uses one standard batch contract across current families:

```text
input_ids       int32   [batch, sequence_length]
attention_mask  float32 [batch, sequence_length], 1 for real tokens and 0 for padding
labels          int32   [batch, sequence_length], -100 for ignored positions
```

`lm_cross_entropy` and `streaming_lm_cross_entropy` apply the HuggingFace
causal-LM shift internally:

```text
logits[:, :-1, :] vs labels[:, 1:]
```

Prepared JSONL task records use this schema:

```json
{"ids":[...], "mask":[...], "attention_mask":[...]}
```

`mask` selects answer tokens for answer-only objectives. Full-token causal-LM
training ignores `mask` and supervises every real shifted token from
`attention_mask`. Records without `attention_mask` are accepted for backwards
compatibility; every token in `ids` is then treated as real, while padding added
by the dataset loader is ignored.

## Maintained Tasks

| Task | Artifact path / loader | Objective modes | Notes |
| --- | --- | --- | --- |
| WikiText-2 | raw `wiki.train.raw` / `wiki.valid.raw` through `WikiText2Dataset` | full-token causal LM | Main perplexity task. |
| MMLU | raw CSV for Qwen; prepared JSONL for GPT-2/Gemma | answer-only or full-token JSONL where applicable | Multiple-choice answer-token supervision remains available. |
| QNLI | `scripts/prepare_qnli_jsonl.py` | answer-only or full-token JSONL | Used by native Android Qwen/QNLI experiments. |
| ARC/HellaSwag/PIQA | `scripts/prepare_mcq_jsonl.py`; legacy ARC helper also emits the same schema | answer-only or full-token JSONL | Candidate tasks; not part of the core Android gate yet. |

## Current Family x Task Status

| Family | WikiText-2 | MMLU | QNLI |
| --- | --- | --- | --- |
| GPT-2 | Train/eval examples maintained | Prepared-JSONL train/eval examples maintained | Dataset artifact compatible; dedicated example pending |
| Gemma | Train/eval examples maintained | Prepared-JSONL train/eval examples maintained | Dataset artifact compatible; dedicated example pending |
| Qwen | Train example maintained | Raw-CSV train example maintained | Native Android experiment path maintained |

## Release Gates

For a family/task row to be promoted from compatible to maintained, it needs:

1. Tokenizer output aligned with HuggingFace on a real local snapshot.
2. Batch labels aligned with the standard shifted CE contract.
3. Real-weight forward/loss fixture for one deterministic batch.
4. PEFT-style one-step LoRA update fixture for q/k/v/o or the documented family default.
5. Example CLI or SDK smoke that opens assets, builds a batch, runs at least one
   train step, reports finite loss, and writes checkpoint/metrics output.
6. Library-level access through `BatchProvider` and `AutoTrainer::fit` when the
   row is intended for generic application use rather than a one-off example.
