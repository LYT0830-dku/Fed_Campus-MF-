# MobileFineTuner Architecture

MobileFineTuner uses a PyTorch/Transformers-like asset discovery split, but
keeps the training path in native C++:

```text
HuggingFace-style model_dir
  config.json
  tokenizer assets
  model.safetensors or model.safetensors.index.json
        |
        v
ModelRegistry      -> model family, asset paths, default LoRA targets
ModelFamilyAdapter -> family-specific load layout and LoRA target policy
TokenizerFactory   -> model-specific tokenizer through one interface
make_causal_lm_batch -> text/token rows to input_ids, attention_mask, labels
AutoModelForCausalLM -> concrete graph dispatch + SafeTensors loading + LoRA
AutoTrainer       -> shared one-step causal-LM LoRA training core
Graph class        -> GPT-2 / Gemma / Qwen / Llama forward and backward math
SafeTensorsLoader  -> external checkpoint tensors into graph weights
LoRA modules       -> trainable adapters on named target projections
Optimizer          -> native C++ parameter update
Android SDK AAR    -> Java/JNI packaging over AutoModel + AutoTrainer
```

## Design Principles

- Model files are runtime assets, not source files.
- Each model family keeps its own tokenizer algorithm.
- Applications should depend on stable discovery APIs, not on directory-specific
  training scripts.
- The core library should expose small, auditable interfaces and keep research
  experiments outside the supported product surface.

## Public Discovery Layer

Use the umbrella header:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

Inspect a model directory:

```cpp
auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
```

`ModelRegistry` reads `config.json` and returns:

- `family`: `GPT2`, `Gemma`, `Qwen`, experimental `Llama`, or recognized-only
  `Mistral`;
- `model_type`: the raw HuggingFace `model_type`;
- tokenizer and SafeTensors asset paths;
- single-file vs sharded SafeTensors availability;
- default LoRA target names for that family;
- `tie_word_embeddings` behavior.

`ModelFamilyAdapter` owns the policy that should not be duplicated in app code
or scattered through `AutoModelForCausalLM`:

- family-specific SafeTensors transpose defaults;
- GPT-2 PEFT/fused `c_attn` versus explicit split q/k/v LoRA policy;
- Gemma LoRA target spec construction;
- Qwen/Llama qv-only request detection.

New families should add policy here before wiring model construction into
`AutoModelForCausalLM`.

Load a tokenizer:

```cpp
auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);
auto encoded = tokenizer->encode_with_attention(prompt, 128);
```

`TokenizerFactory` is intentionally an `AutoTokenizer`-style dispatcher. It
standardizes the C++ call site but does not collapse different tokenizers into
one algorithm. GPT-2 byte-level BPE, Qwen byte-level BPE, Gemma tokenizer logic,
and the Llama 3.x ByteLevel-BPE adapter remain separate because their
vocabularies, special tokens, and pre-tokenization rules differ. Mistral
tokenizer loading is deliberately blocked until a dedicated adapter passes
HuggingFace golden alignment.

Build a standard full-token causal-LM batch:

```cpp
ops::CausalLMBatchConfig batch_cfg;
batch_cfg.sequence_length = 128;
auto batch = ops::make_causal_lm_batch(
    *tokenizer,
    {"first training sentence", "second training sentence"},
    batch_cfg);
```

`CausalLMBatch` owns `input_ids`, `attention_mask`, and shifted `labels`
tensors. It follows the same objective as HuggingFace/PyTorch causal LM
training: logits at position `s` predict the token in `labels[s + 1]`, while
padding labels are `-100`. For left-padded batches, MF also masks the first real
token after padding so a padding-position query is not supervised to predict a
real token. `scripts/generate_lm_alignment_fixture.py` uses the same label
policy before asking PyTorch for the reference loss.

## Model Graph Layer

Each supported architecture has a graph class under `operator/finetune_ops/graph`:

```text
gpt2_model.{h,cpp}
gemma_model.{h,cpp}
llama_model.{h,cpp}
qwen_model.{h,cpp}
```

The graph owns:

- architecture configuration;
- tensor allocation for model parameters;
- `forward(input_ids, attention_mask)`;
- HuggingFace weight-name mapping via `assign_weight`;
- LoRA target attachment;
- trainable parameter discovery.

The graph classes remain explicit instead of hidden behind one large
runtime-polymorphic base class. This keeps the math readable and makes it clear
where architecture-specific behavior lives. Generic application code should use
`AutoModelForCausalLM` when it only needs the common causal-LM LoRA surface, and
drop to concrete graph classes when it needs model-specific diagnostics or
alignment hooks.

`AutoModelForCausalLM` uses `ModelRegistry` to construct GPT-2, Gemma, Qwen, or
experimental Llama graphs, load SafeTensors with family-correct layout defaults,
initialize LoRA, run forward, and expose trainable parameters. Mistral is
recognized but rejected at this layer until graph gates pass. It also retains a
structured `SafeTensorsLoadReport` for the most recent weight load, including
loaded keys, missing keys, source shard paths, dtype/shape metadata, and
unmapped checkpoint tensors. `AutoTrainer` sits one layer above it and
implements the shared one-step training core:

```text
input_ids + attention_mask + labels
        |
        v
AutoModelForCausalLM::forward
        |
        v
lm_cross_entropy -> backward -> grad clip -> Adam -> zero_grad
```

Application code can pass either raw tensors or `CausalLMBatch` directly:

```cpp
auto result = trainer.train_step(batch);
```

## Tokenizer Extension Contract

When adding a model:

1. Add or reuse a tokenizer implementation that matches the upstream
   HuggingFace tokenizer behavior.
2. Wrap it in an `ops::Tokenizer` adapter.
3. Register it in `TokenizerFactory::from_pretrained`.
4. Add a factory test that verifies `config.json` inference and adapter
   construction without loading large model weights.

The factory must not infer tokenizer type from `tokenizer.json` alone. Many
model families use that filename with incompatible semantics. Use
`config.json.model_type` or an explicit `TokenizerLoadOptions::model_type`.

Tokenizer correctness is validated in two layers:

- `TokenizerFactory` uses tiny in-tree fixtures to verify loading, special
  tokens, padding masks, and basic encode/decode behavior without external
  assets.
- `TokenizerHFGolden` optionally compares C++ token IDs against HuggingFace
  `AutoTokenizer` standard-answer JSONL fixtures generated from local model
  snapshots. Generate them with:

```bash
python3 scripts/generate_tokenizer_hf_golden_fixtures.py \
  --output runs/tokenizer_golden/hf_tokenizer_golden.jsonl
MFT_TOKENIZER_GOLDEN_JSONL=runs/tokenizer_golden/hf_tokenizer_golden.jsonl \
  ctest --test-dir operator/build --output-on-failure -R TokenizerHFGolden
```

The golden fixture path is intentionally external to the source package because
it contains local model paths. The generated token ID sequences are small, but
they are tied to the exact tokenizer snapshot used to generate them.

Llama tokenizer support is schema-limited by design. The current adapter covers
Llama 3.x fast `tokenizer.json` assets using ByteLevel BPE and rejects
SentencePiece-only, Unigram, Metaspace-only, and GPT2-tokenizer Llama-family
variants until each receives its own HuggingFace golden gate.

Mistral is the next target family, but it is not allowed to borrow the Llama
adapter by assumption. Its tokenizer gate must first prove exact HuggingFace ID,
mask, and decode alignment on a real Mistral snapshot.

Real-weight model alignment is validated separately from tokenizer alignment.
Generate a fixture with PyTorch/Transformers:

```bash
python3 scripts/generate_lm_alignment_fixture.py \
  --model-dir /path/to/hf-model-snapshot \
  --output runs/model_alignment/model_alignment_fixture.json
MFT_LM_ALIGNMENT_FIXTURE=runs/model_alignment/model_alignment_fixture.json \
  ctest --test-dir operator/build --output-on-failure -R AutoModelAlignment
```

or use:

```bash
bash scripts/check_lm_alignment_fixture.sh \
  runs/model_alignment/model_alignment_fixture.json
```

`AutoModelAlignment` compares the C++ tokenizer output, shifted causal-LM loss,
and last-token logits top-k against the fixture. Without the environment
variable it skips, so routine tests do not require large local weights.

For padded batch fixtures, pass multiple `--prompt` values and a fixed
`--max-length`; the generated fixture stores flattened row-major tensors plus
`batch_size` and `sequence_length`:

```bash
python3 scripts/generate_lm_alignment_fixture.py \
  --model-dir /path/to/Llama-3.2-1B-Instruct \
  --prompt "MobileFineTuner Llama alignment prompt." \
  --prompt "Short." \
  --max-length 12 \
  --output runs/model_alignment/llama_padded_batch_alignment_fixture.json
```

## Model Extension Contract

When adding a model family:

1. Add a config parser aligned with the upstream HuggingFace config.
2. Add a graph class under `finetune_ops/graph`.
3. Map SafeTensors keys explicitly in `assign_weight`.
4. Define default LoRA targets using upstream module names.
5. Add load-layout and LoRA target policy in `ModelFamilyAdapter`.
6. Register the family in `ModelRegistry`.
7. Fail explicitly in `TokenizerFactory` and `AutoModelForCausalLM` until the
   tokenizer and graph gates pass.
8. Add small synthetic tests first, then real-asset smoke tests outside the
   source package.

This mirrors the discovery part of the PyTorch/Transformers pattern:

For the full family-by-family expansion plan and acceptance gates, see
`docs/MODEL_LAYER_ROADMAP.md`.

```text
AutoConfig      -> ModelRegistry
AutoTokenizer   -> TokenizerFactory
AutoModel class  -> AutoModelForCausalLM
Trainer core     -> AutoTrainer
PEFT targets    -> ModelFamilyAdapter + graph LoRA injection
```

## Why This Is Cleaner Than Per-Directory Hardcoding

The previous training directories are useful runnable examples, but they should
not be the long-term architecture boundary. A new model should not require
copying a full application directory just to change tokenizer or config parsing.

With the discovery layer:

- application code can validate a `model_dir` before training;
- tokenizer selection is centralized;
- LoRA target defaults are visible and testable;
- asset contracts stay consistent across desktop and Android;
- unsupported model families fail with explicit errors instead of silently using
  the wrong tokenizer.

## Android Packaging Layer

The Android SDK lives under `android-visualizer/mft-sdk` and builds an AAR. It
adds a Java class, `com.mobilefinetuner.sdk.MobileFineTuner`, plus a small JNI
bridge. The bridge owns a native `AutoModelForCausalLM` and `AutoTrainer` and
forwards training requests into the same C++ code path used by desktop and adb
experiments.

The Android layer deliberately does not own model-specific math, tokenizer
implementations, optimizers, or dataset policy. Those stay in the C++ core and
public asset contracts so Android remains a packaging target rather than a fork
of the framework.
