"""Evaluation: pulls the trained adapter's predictions on the held-out test
split (generated on Modal by modal_app/eval.py) and scores them locally with
Claude as an LLM judge against the business use case + expected outputs.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from anthropic import Anthropic

from autotune import config
from autotune.core import modal_ops

JUDGE_SYSTEM_PROMPT = """You are grading outputs from a fine-tuned model against a business use case.
For each example you get a prompt, the expected (reference) response, and the model's actual response.
Score the actual response from 1-10 on how well it serves the business use case AND how close it is in
substance to the expected response (exact wording need not match). Be strict: 8-10 means it would be
shippable in production as-is, 4-7 means directionally right but needs work, 1-3 means wrong/unusable.

Output ONLY a JSON array, one object per example, in the same order given:
[{"score": <int 1-10>, "reason": "<one sentence>"}]
No prose, no markdown fences."""


def judge_predictions(use_case: str, predictions: list[dict], model: str | None = None) -> list[dict]:
    client = Anthropic(api_key=config.anthropic_api_key())
    model = model or config.agent_model()

    user_prompt = f"""Business use case:
{use_case}

Examples to grade:
{json.dumps(predictions, indent=2)}"""

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def run_eval(run_dir: Path, run_id: str, use_case: str, split: str = "test") -> dict:
    """Generates predictions on Modal, pulls them locally, judges with Claude,
    and returns an aggregate report."""
    modal_ops.generate_predictions(run_id, split=split)
    local_path = modal_ops.pull_file(run_id, f"predictions_{split}.jsonl", run_dir)

    predictions = [json.loads(l) for l in open(local_path) if l.strip()]
    if not predictions:
        return {"count": 0, "mean_score": None, "graded": []}

    grades = judge_predictions(use_case, predictions)
    scores = [g["score"] for g in grades]

    graded = [
        {**p, "score": g["score"], "reason": g["reason"]}
        for p, g in zip(predictions, grades)
    ]
    report_path = run_dir / f"eval_report_{split}.json"
    report = {
        "count": len(scores),
        "mean_score": statistics.mean(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "graded": graded,
    }
    report_path.write_text(json.dumps(report, indent=2))
    return report
