"""Dataset ingestion, validation, dedup, and splitting.

Autotune expects examples as JSONL chat-style records:
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
Mock/seed data and synthetic data are merged into one pool, deduped, then split.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

REQUIRED_KEY = "messages"


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def _fingerprint(example: dict) -> str:
    text = json.dumps(example.get(REQUIRED_KEY, example), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def validate_examples(examples: list[dict]) -> tuple[list[dict], list[str]]:
    """Returns (valid_examples, error_messages)."""
    errors = []
    valid = []
    for i, ex in enumerate(examples):
        if REQUIRED_KEY not in ex:
            errors.append(f"example {i} missing '{REQUIRED_KEY}' key")
            continue
        msgs = ex[REQUIRED_KEY]
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"example {i} '{REQUIRED_KEY}' must be a non-empty list")
            continue
        roles = {m.get("role") for m in msgs}
        if not roles.issubset({"system", "user", "assistant"}):
            errors.append(f"example {i} has invalid roles: {roles}")
            continue
        if "assistant" not in roles:
            errors.append(f"example {i} has no assistant turn (nothing to learn from)")
            continue
        valid.append(ex)
    return valid, errors


def dedup(examples: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for ex in examples:
        fp = _fingerprint(ex)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(ex)
    return out


def split(
    examples: list[dict], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, int(n * val_frac)) if n >= 10 else 0
    n_test = max(1, int(n * test_frac)) if n >= 10 else 0
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return {"train": train, "val": val, "test": test}


def build_dataset(
    run_dir: Path, seed_examples: list[dict], synthetic_examples: list[dict]
) -> dict:
    """Merge seed + synthetic, validate, dedup, split, and write to run_dir/data/."""
    pool = seed_examples + synthetic_examples
    valid, errors = validate_examples(pool)
    valid = dedup(valid)
    splits = split(valid)

    data_dir = run_dir / "data"
    for name, exs in splits.items():
        write_jsonl(data_dir / f"{name}.jsonl", exs)

    return {
        "seed_count": len(seed_examples),
        "synthetic_count": len(synthetic_examples),
        "valid_count": len(valid),
        "errors": errors,
        "splits": {name: len(exs) for name, exs in splits.items()},
        "paths": {name: str(data_dir / f"{name}.jsonl") for name in splits},
    }
