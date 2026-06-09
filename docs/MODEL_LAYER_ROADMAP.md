# Model Layer Roadmap

MobileFineTuner's long-term model-layer goal is to accept common
HuggingFace-style causal-LM model directories and run native C++ LoRA training
with tokenizer, forward pass, loss, gradients, and Android SDK behavior aligned
against PyTorch/Transformers.

The project should grow by model family, not by one-off checkpoint folders. A
family is considered supported only after it passes the acceptance gates in this
document.

## Current Supported Families

| Family | Representative Models | Status | Tokenizer | Notes |
| --- | --- | --- | --- | --- |
| GPT-2 | `gpt2`, `gpt2-medium` | Supported | Byte-level BPE | Legacy decoder with absolute positions and fused QKV. The default/PEFT `c_attn` path is fused, explicit q/k/v requests stay split, and `gpt2` now has a real-weight PyTorch/PEFT one-step LoRA parity fixture consumed by `AutoModelPEFTStepAlignment`. |
| Qwen | `Qwen2.5-0.5B` | Supported | Qwen byte-level BPE | Current graph requires tied input/output embeddings. `Qwen2.5-0.5B` now has a real-weight PyTorch/PEFT one-step LoRA parity fixture consumed by `AutoModelPEFTStepAlignment`. |
| Gemma | `gemma-3-270m`, `gemma-3-1b-pt` | Supported | Gemma tokenizer adapter | Uses Gemma left padding and family-specific RMSNorm behavior. `gemma-3-270m` now has a real-weight PyTorch/PEFT one-step LoRA parity fixture consumed by `AutoModelPEFTStepAlignment`. |
| Llama | `Llama-3.2-1B-Instruct`-style Llama 3.x snapshots | Validated experimental | Llama 3.x `tokenizer.json` adapter | Registry, SafeTensors mapping, AutoModel dispatch, tied/untied output head handling, q/k/v/o LoRA, Llama3 RoPE scaling, tokenizer HF golden, synthetic train-step smoke, real-weight PyTorch forward/loss/logits alignment, and a real-weight PEFT one-step LoRA parity fixture are implemented. Android real-weight smoke remains before claiming full support. |
| Mistral | `mistral` config family | Recognized only | Not supported yet | `ModelRegistry` identifies Mistral and exposes q/k/v/o defaults, but `TokenizerFactory` and `AutoModelForCausalLM` fail explicitly until Mistral tokenizer, sliding-window/mask, and real-weight gates pass. |

## Expansion Order

### Phase 1: Consolidate Existing Families

Before adding new architectures, the current three families must remain clean
under the same gates:

- `ModelRegistry` recognizes each model family from `config.json.model_type`.
- `ModelFamilyAdapter` owns family-specific SafeTensors layout and LoRA target
  policy.
- `TokenizerFactory` aligns with HuggingFace `AutoTokenizer` on real snapshots.
- SafeTensors key mapping is strict: missing required mapped keys fail by
  default.
- `AutoModelForCausalLM` exposes `forward_hidden(...)`,
  `lm_head_weight_for_loss()`, `init_lora(...)`, and trainable parameters.
- `AutoTrainer` computes standard shifted full-token causal-LM CE through the
  streaming loss path by default.
- Android SDK can open the model, initialize LoRA, build text batches, and run
  at least one native train step.

### Phase 2: Llama Family

This should be the next family because many public checkpoints are Llama-like.
Target model types:

- `llama`
- compatible small checkpoints such as TinyLlama/OpenLLaMA-style configs when
  their tensor naming and tokenizer contracts match the Llama adapter

Required work:

- Add `ModelFamily::Llama`. **Done for graph/model registry.**
- Add `LlamaConfig` parser. **Done for the default Llama decoder fields.**
- Add `LlamaModel` graph or a reusable decoder-block base shared with Qwen.
  **Done as a family-specific graph to avoid Qwen bias/tied-head assumptions.**
- Add Llama SafeTensors key mapper. **Done, including optional attention bias
  and untied `lm_head.weight`.**
- Add LoRA targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, optionally
  `gate_proj`, `up_proj`, `down_proj`.
- Add tokenizer adapter for Llama-compatible tokenizer assets. **Done for
  Llama 3.x ByteLevel-BPE `tokenizer.json`; other tokenizer schemas are
  explicitly rejected.**
- Add HF tokenizer golden cases and PyTorch first-step loss fixtures.
  **Done for a local Llama 3.2-1B-Instruct snapshot.**
- Add Llama3 RoPE scaling. **Done for `rope_type=llama3`; unsupported scaling
  variants still fail explicitly.**
- Add LoRA optimizer-step parity against PEFT. **Done for the local
  Llama 3.2-1B-Instruct fixture.**
- Add Android real-weight smoke. **Pending.**

Main risks:

- tokenizer variants: SentencePiece vs fast tokenizer JSON behavior; only
  Llama 3.x ByteLevel-BPE tokenizer JSON is supported today;
- RoPE scaling variants across Llama 2/3/3.1/3.2; Llama3 scaling is
  implemented, while other variants still fail explicitly;
- tied vs untied `lm_head.weight`;
- GQA and head-dimension differences.

### Phase 3: Mistral Family

Mistral is close to Llama but should be treated as a separate family because it
uses sliding-window attention and byte-fallback BPE tokenizer behavior.

Target model types:

- `mistral`
- later `mixtral` only after dense Mistral is stable; MoE should not be folded
  into the first Mistral implementation.

Required work:

- Add `ModelFamily::Mistral`. **Done at registry level only.**
- Reuse as much Llama decoder infrastructure as possible.
- Add sliding-window attention support and tests.
- Add Mistral tokenizer golden fixtures.
- Add PyTorch first-step loss fixture for a small Mistral-compatible snapshot.

Main risks:

- sliding-window attention mask correctness;
- tokenizer byte fallback;
- avoiding accidental Mixtral/MoE partial support claims.

### Phase 4: Phi Family

Phi is useful for smaller edge-oriented checkpoints, but it should come after
Llama/Mistral because its architecture variants need careful mapping.

Target model types:

- `phi`
- `phi3`

Required work:

- Add `ModelFamily::Phi`.
- Add config parser and key mapper for the specific supported Phi generation.
- Add tokenizer adapter or reuse compatible tokenizer logic only after HF
  golden verification.
- Add LoRA target defaults from actual module names.
- Add PyTorch first-step loss fixture.

Main risks:

- Phi generations are not identical;
- tokenizer and special-token behavior differs across releases;
- small local checkpoints must be chosen deliberately for real-asset tests.

### Phase 5: Advanced Families

These should wait until the base decoder framework is stable:

- Qwen3 and Qwen MoE variants
- DeepSeek dense/MoE
- GLM
- Falcon
- StarCoder-style code models
- multimodal models

Do not advertise support for MoE, multimodal, or custom-kernel architectures
until the graph, tokenizer, loss, and first-step PyTorch alignment gates pass.

## Acceptance Gates For A New Family

A model family is not supported until all gates pass.

### Gate 1: Registry And Asset Contract

- `ModelRegistry::inspect_pretrained(model_dir)` returns the correct family.
- Required asset files are documented.
- Unsupported config combinations fail explicitly.
- tied/untied output embedding behavior is validated.

### Gate 2: Tokenizer HF Alignment

- Generate golden cases with HuggingFace `AutoTokenizer`.
- Compare C++ `encode`, padded IDs, attention masks, and
  `decode(skip_special_tokens=true)`.
- Include edge cases:
  - empty string;
  - leading/trailing whitespace;
  - tabs/newlines;
  - CJK;
  - accents;
  - emoji;
  - literal special tokens;
  - `add_special_tokens=true/false`;
  - chat/control token boundaries when the model uses them.

Tokenizer alignment must be exact for IDs and masks. Approximate matching is not
acceptable.

### Gate 3: Strict Weight Loading

- Every required HF weight key maps to an internal tensor.
- Missing required keys fail by default.
- Shape mismatches fail by default.
- Optional keys are documented.
- Sharded SafeTensors and single-file SafeTensors both work where upstream
  snapshots use them.

### Gate 4: Forward Alignment

Using a fixed tiny prompt from the real tokenizer:

- C++ input IDs match PyTorch input IDs.
- final hidden checksum matches within tolerance;
- logits checksum/top-k matches within tolerance;
- attention mask behavior matches for padded batches.

Use `scripts/generate_lm_alignment_fixture.py` plus the optional
`AutoModelAlignment` CTest target for this gate. The fixture is external to the
source tree because it contains local model paths and depends on large model
weights.

### Gate 5: Loss And LoRA Step Alignment

For one deterministic batch:

- shifted full-token CE matches PyTorch;
- LoRA target parameter names/counts match the intended PEFT setup;
- after one optimizer step with fixed seed and hyperparameters, selected LoRA
  tensors or checksums match within tolerance.

This is the strongest test that the family is actually trainable, not merely
loadable.

The core LoRA training primitive already has a PyTorch-aligned unit gate:
`LoRAStepAlignment` checks `x @ W + (x @ A @ B) * scale`, MSE loss, A/B
gradients, and one Adam update against a deterministic PyTorch fixture. Family
support still requires an additional PEFT-style fixture on real model modules
and real tokenizer batches.

Use `scripts/generate_peft_lora_step_fixture.py` to produce the PEFT standard
answer for that family-level gate. The fixture stores the tokenized batch,
initial LoRA tensors in MF-consumable layout, expected loss, and post-Adam LoRA
tensor checksums. `AutoModelPEFTStepAlignment` is the C++ consumer test: it
loads this fixture, installs the initial LoRA tensors by
`AutoModelForCausalLM::named_trainable_parameters()`, runs `AutoTrainer`, and
checks the recorded after-step tensors.

Real-weight PEFT one-step parity has now passed for representative GPT-2, Qwen,
Gemma, and Llama snapshots. GPT-2 uses a precise policy: PEFT-style
`c_attn`/`attn_qkv` stays fused, explicit `q_proj`/`k_proj`/`v_proj` requests
use split LoRA slices, and ambiguous mixed fused/split qkv target lists are
rejected.

### Gate 6: Android Smoke

On Android:

- open model assets;
- initialize LoRA;
- build native text batch;
- run one to three train steps;
- report finite loss, elapsed time, and RSS.

The Android gate can use a small checkpoint or a reduced fixture when full
weights are too large for routine CI.

The maintained family/task boundary is tracked in
`docs/CURRENT_FAMILY_TASK_MATRIX.md`. A dataset row is not considered promoted
until its JSONL or raw loader produces the same `input_ids`, `attention_mask`,
and shifted-label contract used by the PyTorch alignment fixtures.

## Current Real-Asset Alignment Results

The generic `AutoModelAlignment` gate has been added and validated on local
assets:

| Family | Snapshot | Gate Result | Notes |
| --- | --- | --- | --- |
| GPT-2 | `gpt2` HF cache | Passed | tokenizer, shifted CE, last-token logits top-k |
| Qwen | `Qwen2.5-0.5B` staged from local tokenizer + HF weight | Passed | core Qwen path used by phone experiments |
| Gemma | `gemma-3-270m` local snapshot | Passed | uncovered and fixed optional/tied `lm_head.weight` loading |
| Llama | `Llama-3.2-1B-Instruct` local snapshot | Passed | single prompt expected loss `8.540868759`; padded `2x12` batch expected loss `8.67343044`; validated tokenizer, shifted CE, padding masks, and last-token logits top-k with Llama3 RoPE scaling |
| Mistral | synthetic `mistral` config | Registry-only passed | AutoModel/Tokenizer reject explicitly until real gates exist |

## Refactor Direction

The current `AutoModelForCausalLM` switch-based dispatch is acceptable for the
current families while model behavior is still being hardened. Before promoting
Llama from experimental graph support to full support, introduce a clearer
adapter boundary:

```text
ModelFamilyAdapter
  inspect/config defaults
  tokenizer factory hook
  graph construction
  safetensors key mapper
  default LoRA targets
  unsupported-config validation
```

This keeps the public API stable while letting each family own the parts that
must differ.

Avoid a single over-generalized Transformer implementation until repeated
patterns are proven across Qwen, Gemma, Llama, and Mistral. Reuse lower-level
blocks where they are genuinely identical, but keep family-level acceptance
tests separate.

## First Concrete Milestone

The current implementation milestone is now:

```text
Llama family: registry + synthetic graph smoke + Llama 3.x tokenizer golden + real-weight forward/loss/logits gate
Core LoRA primitive: PyTorch-aligned forward/backward/Adam one-step gate
```

Then:

```text
Real-weight Android SDK/native training smoke for GPT-2/Qwen/Gemma/Llama
Clean release commit/tag with fixture-gate instructions
Mistral tokenizer and graph gates
```

Only after that should MobileFineTuner claim Llama support.
