"""Tool schemas (Anthropic tool-use format) and the dispatch that executes
them against autotune.core.*. Every tool call reads/writes the run's
RunState so a run is inspectable/resumable outside the loop.
"""
from __future__ import annotations

import json
from pathlib import Path

from autotune.core import data_gen, dataset as dataset_mod, eval as eval_mod, hf_search, modal_ops
from autotune.core.recipe import Recipe
from autotune.core.state import RunState

TOOLS = [
    {
        "name": "suggest_base_models",
        "description": "Search the Hugging Face Hub for candidate base models for this use case. Prefer unsloth-prequantized repos when a good fit exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "size_hint": {"type": "string", "description": "e.g. '3b', '7b', 'small', 'coding'"},
            },
        },
    },
    {
        "name": "write_recipe",
        "description": "Write (or overwrite) the training recipe for this run. Pass only the fields you want to set; unset fields keep their default.",
        "input_schema": {
            "type": "object",
            "properties": {
                "base_model": {"type": "string"},
                "max_seq_length": {"type": "integer"},
                "load_in_4bit": {"type": "boolean"},
                "lora_r": {"type": "integer"},
                "lora_alpha": {"type": "integer"},
                "lora_dropout": {"type": "number"},
                "target_modules": {"type": "array", "items": {"type": "string"}},
                "learning_rate": {"type": "number"},
                "num_train_epochs": {"type": "number"},
                "per_device_train_batch_size": {"type": "integer"},
                "gradient_accumulation_steps": {"type": "integer"},
                "gpu": {"type": "string", "description": "Modal GPU type: A10G, L40S, A100-40GB, H100"},
                "notes": {"type": "string", "description": "Why this recipe, briefly."},
            },
        },
    },
    {
        "name": "generate_synthetic_data",
        "description": "Expand the seed examples into a larger synthetic training pool using Claude.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_count": {"type": "integer"},
                "guidance": {"type": "string", "description": "What to emphasize/cover, if anything."},
            },
            "required": ["target_count"],
        },
    },
    {
        "name": "build_dataset",
        "description": "Validate, dedup, and split the accumulated seed+synthetic pool into train/val/test JSONL files.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_training",
        "description": "Push the run's recipe+data to Modal and run the Unsloth+PEFT training job. Blocks until done.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_eval",
        "description": "Generate predictions on the held-out test split via Modal and judge them with Claude. Returns mean score and per-example reasons.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "deploy",
        "description": "Deploy the trained adapter as a Modal serving endpoint.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "ask_user",
        "description": "Pause and ask the human operator a question when a decision needs their input (product judgment call, ambiguous guidance, or a tradeoff you shouldn't make silently).",
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    },
    {
        "name": "finish",
        "description": "End the run with a summary. Call this once eval scores are good and (if deploying) the endpoint is live.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


class ToolExecutionError(Exception):
    pass


class Dispatcher:
    """Holds mutable per-run context (state, seed examples) and executes tool calls."""

    def __init__(self, state: RunState, seed_examples: list[dict]):
        self.state = state
        self.seed_examples = seed_examples
        self._synthetic_pool: list[dict] = []
        self.paused_for_user: str | None = None
        self.finished_summary: str | None = None

    def dispatch(self, name: str, tool_input: dict) -> str:
        method = getattr(self, f"_tool_{name}", None)
        if method is None:
            raise ToolExecutionError(f"Unknown tool: {name}")
        result = method(**tool_input)
        return json.dumps(result) if not isinstance(result, str) else result

    # -- tools --------------------------------------------------------

    def _tool_suggest_base_models(self, size_hint: str = "") -> dict:
        result = hf_search.suggest_base_models(self.state.use_case, size_hint=size_hint)
        self.state.note(f"Searched base models (size_hint={size_hint!r})")
        return result

    def _tool_write_recipe(self, **fields) -> dict:
        current = Recipe.from_dict(self.state.recipe) if self.state.recipe else Recipe()
        merged = {**current.to_dict(), **fields}
        recipe = Recipe.from_dict(merged)
        recipe.save(self.state.dir)
        self.state.recipe = recipe.to_dict()
        self.state.base_model = recipe.base_model
        self.state.note(f"Recipe updated: base_model={recipe.base_model}, lora_r={recipe.lora_r}")
        return {"recipe": recipe.to_dict()}

    def _tool_generate_synthetic_data(self, target_count: int, guidance: str = "") -> dict:
        new_examples = data_gen.generate_synthetic_dataset(
            self.state.use_case, self.seed_examples, target_count, guidance=guidance
        )
        self._synthetic_pool.extend(new_examples)
        self.state.note(f"Generated {len(new_examples)} synthetic examples (guidance={guidance!r})")
        return {"generated": len(new_examples), "total_synthetic_pool": len(self._synthetic_pool)}

    def _tool_build_dataset(self) -> dict:
        report = dataset_mod.build_dataset(self.state.dir, self.seed_examples, self._synthetic_pool)
        self.state.dataset = report
        self.state.status = "data"
        self.state.note(f"Dataset built: {report['splits']}")
        return report

    def _tool_run_training(self) -> dict:
        if not self.state.recipe:
            raise ToolExecutionError("No recipe set — call write_recipe first")
        if not self.state.dataset:
            raise ToolExecutionError("No dataset built — call build_dataset first")
        modal_ops.push_run(self.state.dir, self.state.run_id)
        gpu = self.state.recipe.get("gpu", "A10G")
        metrics = modal_ops.train(self.state.run_id, gpu=gpu)
        self.state.status = "training"
        self.state.train_history.append({"iteration": len(self.state.train_history), "metrics": metrics})
        self.state.note(f"Training complete: train_loss={metrics.get('train_loss')}")
        return metrics

    def _tool_run_eval(self) -> dict:
        report = eval_mod.run_eval(self.state.dir, self.state.run_id, self.state.use_case)
        self.state.status = "eval"
        self.state.eval_history.append({"iteration": len(self.state.eval_history), "report": report})
        self.state.note(f"Eval complete: mean_score={report.get('mean_score')}")
        return {k: v for k, v in report.items() if k != "graded"} | {
            "worst_examples": sorted(report.get("graded", []), key=lambda g: g["score"])[:5]
        }

    def _tool_deploy(self) -> dict:
        modal_ops.deploy_serving(self.state.run_id)
        self.state.status = "deploy"
        self.state.deployment = {"run_id": self.state.run_id, "deployed": True}
        self.state.note("Deployed serving endpoint")
        return self.state.deployment

    def _tool_ask_user(self, question: str) -> str:
        self.paused_for_user = question
        self.state.note(f"Paused for user: {question}")
        return "Waiting on human operator."

    def _tool_finish(self, summary: str) -> str:
        self.finished_summary = summary
        self.state.status = "done"
        self.state.note(f"Finished: {summary}")
        return "Run marked done."
