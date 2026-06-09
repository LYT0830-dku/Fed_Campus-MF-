# Model And Dataset Assets

MobileFineTuner does not bundle pretrained model weights or benchmark datasets.
The library code, training CLIs, and tests are shipped separately from large
assets. Users provide local HuggingFace-style model snapshots at runtime, similar
to how PyTorch/Transformers accepts either a model identifier or a local
`from_pretrained(...)` path.

MobileFineTuner currently uses local paths only. It does not call
`from_pretrained(...)` and does not download model IDs at runtime. Download or
copy the model snapshot yourself, then pass that directory through an
environment variable or a CLI flag such as `--model_dir`.

## Recommended Layout

Use one shared model root:

```text
/path/to/mft-models/
  gpt2/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    vocab.json
    merges.txt
  gpt2-medium/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    vocab.json
    merges.txt
  gemma-3-270m/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    tokenizer.model
  gemma-3-1b-pt/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    tokenizer.model
  Qwen2.5-0.5B/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json
    vocab.json
    merges.txt
  llama-family-model/
    config.json
    model.safetensors                 # or model.safetensors.index.json + shards
    tokenizer.json                    # Llama 3.x ByteLevel-BPE tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
  mistral-family-model/
    config.json
    model.safetensors                 # recognized by registry only today
    tokenizer assets                  # not consumed until Mistral tokenizer gate lands
```

Then export:

```bash
export MFT_MODEL_ROOT=/path/to/mft-models
```

Every model snapshot must include `config.json` with a supported HuggingFace
`model_type`. MobileFineTuner uses that field in `ModelRegistry` and
`TokenizerFactory`:

```text
gpt2      -> GPT-2 graph + GPT-2 byte-level BPE tokenizer
gemma*    -> Gemma graph + Gemma tokenizer
qwen*     -> Qwen graph + Qwen byte-level BPE tokenizer
llama*    -> Llama graph + Llama 3.x ByteLevel-BPE tokenizer.json adapter
mistral*  -> recognized by ModelRegistry only; AutoModel/Tokenizer reject until gates pass
```

The current Qwen graph supports tied input/output embeddings
(`tie_word_embeddings=true`). A Qwen checkpoint that explicitly sets
`tie_word_embeddings=false` is rejected by `AutoModelForCausalLM` until separate
`lm_head.weight` loading is implemented.

Do not infer the model family from `tokenizer.json` alone. Different model
families reuse that filename with incompatible tokenization rules.

Llama-family graph support is present for `AutoModelForCausalLM` and
SafeTensors loading, including tied and common untied `lm_head.weight`
checkpoints, q/k/v/o LoRA, and Llama3 RoPE scaling. Tokenizer support is
intentionally schema-limited: `TokenizerFactory` supports Llama 3.x fast
tokenizer assets that use ByteLevel BPE in `tokenizer.json` and have passed
HuggingFace golden alignment. It does not claim support for SentencePiece-only
`tokenizer.model`, Unigram, Metaspace-only, GPT2-tokenizer Llama-like models, or
chat-template execution. Unsupported tokenizer schemas fail explicitly instead
of silently producing different token IDs.

Mistral-family configs are recognized so applications can report a precise
unsupported-family error. They are not trainable yet in MF: Mistral tokenizer
assets and graph loading fail explicitly until sliding-window/mask behavior,
tokenization, SafeTensors mapping, and PyTorch alignment gates pass.

The runner scripts also accept per-model overrides:

```bash
export GPT2_SMALL_MODEL_DIR=/path/to/gpt2
export GPT2_MEDIUM_MODEL_DIR=/path/to/gpt2-medium
export GEMMA_270M_MODEL_DIR=/path/to/gemma-3-270m
export GEMMA_1B_PT_MODEL_DIR=/path/to/gemma-3-1b-pt
export QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B
export LLAMA_MODEL_DIR=/path/to/llama-family-model
```

For backwards-compatible local experiments, each model example also checks its
own `pretrained/` directory, for example:

```text
examples/qwen_lora_finetune/pretrained/
examples/gpt2_small_lora_finetune/pretrained/
```

Do not commit these directories. They are intentionally ignored by Git.

## Downloading Models

One practical workflow is to use HuggingFace CLI:

```bash
pip install -U huggingface_hub

huggingface-cli download gpt2 \
  --local-dir "$MFT_MODEL_ROOT/gpt2" \
  --local-dir-use-symlinks False

huggingface-cli download gpt2-medium \
  --local-dir "$MFT_MODEL_ROOT/gpt2-medium" \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-0.5B \
  --local-dir "$MFT_MODEL_ROOT/Qwen2.5-0.5B" \
  --local-dir-use-symlinks False
```

The asset resolver also accepts common local aliases under `MFT_MODEL_ROOT`,
for example `gpt2-medium_official` for GPT-2 Medium.

For gated models such as Gemma, accept the model license on HuggingFace first and
authenticate with `huggingface-cli login`.

## Weight File Contract

The stable C++ loader accepts either:

- one file named `model.safetensors`; or
- a HuggingFace sharded snapshot with `model.safetensors.index.json` and the
  referenced shard files.

Sharded examples look like:

```text
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
model.safetensors.index.json
```

The loader reads the index, opens the listed shards, and resolves requested
tensor names across all shard files. If both `model.safetensors` and an index
are present, the single-file weight takes precedence.

For model onboarding and support tickets, inspect
`AutoModelForCausalLM::safetensors_load_report()` after `from_pretrained`. The
report records:

- total requested mapped keys;
- loaded internal key, HuggingFace key, source shard path, HF dtype, HF shape,
  loaded shape, and transpose policy;
- missing keys when strict loading is disabled;
- checkpoint tensor keys that were present but not consumed by the active
  family mapping.

This makes missing-key and wrong-family failures reproducible without depending
on verbose console logs.

## Dataset Layout

Use one shared data root:

```text
/path/to/mft-data/
  wikitext2/wikitext-2-raw/
    wiki.train.raw
    wiki.valid.raw
    wiki.test.raw
  mmlu/data/
    README.txt
    dev/abstract_algebra_dev.csv
    ...
```

Then export:

```bash
export MFT_DATA_ROOT=/path/to/mft-data
```

The default repository-local fallback is:

```text
data/wikitext2/wikitext-2-raw/
data/mmlu/data/
```

## Validation

Check all local assets before running real training:

```bash
bash scripts/check_local_assets.sh
```

The smoke suite does not require real model weights:

```bash
bash scripts/run_training_smoke.sh
```

The real-asset sanity suite requires valid model and dataset paths:

```bash
bash scripts/run_training_real_assets.sh
```

Tokenizer-only HuggingFace alignment checks do not require model weights. They
require local tokenizer snapshots and the Python `transformers` package:

```bash
export MFT_MODEL_ROOT=/path/to/mft-models
bash scripts/check_tokenizer_hf_alignment.sh
```

To gate a subset while onboarding a new machine, set
`MFT_TOKENIZER_MODELS="qwen gemma_270m gemma_1b_pt"` before running the script.

If `MFT_TOKENIZER_GOLDEN_JSONL` is not set, the `TokenizerHFGolden` CTest target
skips cleanly so source-only builds do not depend on external assets.

Real-weight forward/loss/logits alignment is a separate gate:

```bash
python3 scripts/generate_lm_alignment_fixture.py \
  --model-dir /path/to/hf-model-snapshot \
  --output runs/model_alignment/model_alignment_fixture.json

bash scripts/check_lm_alignment_fixture.sh \
  runs/model_alignment/model_alignment_fixture.json
```

The fixture contains local asset paths and small expected numeric outputs, so it
belongs under ignored `runs/` output rather than the source tree.

For left-padded tokenizers, the fixture generator follows MF's standard
`CausalLMBatch` label policy: padding tokens are ignored, and the first real
token after padding is also ignored so a padding-position query is not trained
to predict real text.

## Runtime Contract

The reusable C++ library does not download assets. Applications must pass asset
paths explicitly. The common patterns are:

```bash
QWEN_MODEL_DIR=/path/to/Qwen2.5-0.5B \
QWEN_DATA_DIR=/path/to/wikitext-2-raw \
./examples/qwen_lora_finetune/run_wikitext.sh
```

or:

```bash
MFT_MODEL_ROOT=/path/to/mft-models \
MFT_DATA_ROOT=/path/to/mft-data \
bash scripts/run_training_real_assets.sh
```
