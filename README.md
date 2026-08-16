# Autotune

An agentic post-training system. You describe a business use case and hand it a
small set of mock/seed examples; Autotune (a Claude-driven agent loop) picks a base
model from the Hugging Face Hub, proposes a LoRA fine-tuning recipe, synthesizes the
rest of the training data, trains on [Modal](https://modal.com) with
[Unsloth](https://github.com/unslothai/unsloth) + PEFT, evaluates with an LLM judge,
iterates on the recipe/data when scores are weak, and deploys the result as a Modal
serving endpoint — asking you for guidance at the points that need product judgment.

## How it works

```
use case + seed data
        │
        ▼
  ┌─────────────┐   suggest_base_models (HF Hub search)
  │             │   write_recipe (LoRA + training hyperparams)
  │ Claude agent│   generate_synthetic_data (expands seed set)
  │    loop     │   build_dataset (validate/dedup/split)
  │             │   run_training (Modal: Unsloth + PEFT LoRA)
  │             │   run_eval (Modal generate + Claude-as-judge)
  │             │   ask_user (pauses for your input)
  │             │   deploy (Modal serving endpoint)
  └─────────────┘
        │
        ▼
  runs/<run_id>/manifest.json   ← full history, resumable
```

Every run lives under `runs/<run_id>/` with a `manifest.json` tracking status,
recipe, dataset stats, training/eval history, and a human-readable event log — so a
run is inspectable and resumable outside the agent loop.

## Setup

```bash
pip install -e .
cp .env.example .env   # fill in ANTHROPIC_API_KEY
modal token set        # your own Modal credentials — not managed by Autotune
```

Secrets are loaded via `autotune/config.py`: `.env` by default, or set
`AUTOTUNE_SECRETS_PROVIDER=appconfig` (plus the `AWS_APPCONFIG_*` vars) to pull from
AWS AppConfig instead when deploying the harness somewhere shared.

## Usage

```bash
autotune start \
  --use-case "Draft first-reply support responses for a SaaS invoicing product" \
  --seed-data examples/support_replies.seed.jsonl \
  --guidance "Keep responses under 3 sentences; never promise refunds without escalation"

autotune list
autotune show <run_id>
autotune resume <run_id> --guidance "the model is too verbose, tighten it up"
```

Seed data is JSONL, one chat example per line:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Layout

- `autotune/agent/` — the Claude tool-use loop (`loop.py`), tool schemas/dispatch
  (`tools.py`), system prompt (`prompts.py`).
- `autotune/core/` — plain functions the tools call: `hf_search.py` (base model
  search), `recipe.py` (training config), `data_gen.py` (synthetic data via Claude),
  `dataset.py` (validate/dedup/split), `eval.py` (LLM-judge scoring), `modal_ops.py`
  (push/pull/train/deploy wrappers), `state.py` (run manifest).
- `autotune/modal_app/` — the Modal-side code: `train.py` (Unsloth+PEFT LoRA SFT),
  `eval.py` (batched generation on the held-out split), `serve.py` (FastAPI
  chat-completions endpoint serving the adapter), `common.py` (shared Image/Volume).

## Notes

- Base models are chosen at run time from the HF Hub, biased toward
  `unsloth/*-bnb-4bit` repos for fast, validated loading — there's no hardcoded
  default the agent is stuck with.
- Training/eval GPU defaults to `A10G`; the agent can set a different `gpu` in the
  recipe per run based on model size.
- This is a scaffold: the agent loop, tools, and Modal functions are wired end to
  end, but you should sanity-check a first run's recipe/data before trusting fully
  unattended loops on anything costly.
