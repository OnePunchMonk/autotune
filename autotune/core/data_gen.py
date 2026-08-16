"""Synthetic data generation: expands a small mock seed set into a larger
training pool by asking Claude to infer the task schema/style from the seed
examples and the business use case, then generate new, diverse examples that
match it. Batches generations and self-filters obvious near-duplicates.
"""
from __future__ import annotations

import json

from anthropic import Anthropic

from autotune import config

SYSTEM_PROMPT = """You are a synthetic data generator for fine-tuning. You are given:
1. A business use case description.
2. A small set of seed (mock) examples in {"messages": [...]} chat format.
3. Optional user guidance on what kinds of examples to emphasize.

Generate NEW examples that:
- Match the schema, tone, and difficulty distribution of the seed examples.
- Cover edge cases and paraphrase variety the seed set doesn't (different phrasing,
  lengths, edge conditions, adjacent sub-tasks implied by the use case).
- Are NOT near-duplicates of the seed examples or of each other.
- Are realistic for the stated business use case.

Output ONLY a JSON array of objects, each shaped exactly like the seed examples
(a "messages" list with alternating user/assistant turns, optionally a leading
system turn). No prose, no markdown fences, just the JSON array.
"""


def generate_batch(
    use_case: str,
    seed_examples: list[dict],
    n: int,
    guidance: str = "",
    model: str | None = None,
) -> list[dict]:
    client = Anthropic(api_key=config.anthropic_api_key())
    model = model or config.agent_model()

    user_prompt = f"""Business use case:
{use_case}

Seed examples ({len(seed_examples)}):
{json.dumps(seed_examples, indent=2)}

User guidance: {guidance or "(none — use your judgment)"}

Generate exactly {n} new examples as a JSON array."""

    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        examples = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON: {e}\n---\n{text[:2000]}")

    if not isinstance(examples, list):
        raise ValueError("Expected a JSON array of examples")
    return examples


def generate_synthetic_dataset(
    use_case: str,
    seed_examples: list[dict],
    target_count: int,
    guidance: str = "",
    batch_size: int = 25,
    model: str | None = None,
) -> list[dict]:
    """Generates target_count synthetic examples in batches (single calls tend
    to degrade in diversity/quality past ~25-30 examples)."""
    out: list[dict] = []
    while len(out) < target_count:
        remaining = target_count - len(out)
        batch_n = min(batch_size, remaining)
        batch = generate_batch(use_case, seed_examples, batch_n, guidance=guidance, model=model)
        out.extend(batch)
    return out[:target_count]
