"""Base-model search: lets the agent pick a base model from the Hugging Face
Hub instead of relying on a hardcoded default. Biased toward Unsloth's
pre-quantized 4bit repos when available (they load faster and are what
autotune/modal_app/train.py expects to work best with), but falls back to
any instruct-tuned model matching the query.
"""
from __future__ import annotations

from huggingface_hub import HfApi

_api = HfApi()


def search_models(query: str, limit: int = 15, unsloth_only: bool = False) -> list[dict]:
    """Searches the HF Hub for candidate base models.

    query: free-text, e.g. "qwen2.5 instruct 7b" or "small instruct model coding".
    unsloth_only: restrict to the `unsloth/` org, whose repos are pre-quantized
      and validated to work with FastLanguageModel.from_pretrained.
    """
    search = f"unsloth {query}" if unsloth_only else query
    models = _api.list_models(
        search=search,
        sort="downloads",
        direction=-1,
        limit=limit * 3 if unsloth_only else limit,
    )

    results = []
    for m in models:
        if unsloth_only and not m.id.startswith("unsloth/"):
            continue
        results.append(
            {
                "id": m.id,
                "downloads": getattr(m, "downloads", None),
                "likes": getattr(m, "likes", None),
                "tags": getattr(m, "tags", None) or [],
                "pipeline_tag": getattr(m, "pipeline_tag", None),
            }
        )
        if len(results) >= limit:
            break
    return results


def suggest_base_models(use_case: str, size_hint: str = "", limit: int = 10) -> dict:
    """Convenience wrapper for the agent tool: runs both an unsloth-scoped
    search (preferred, ready-to-quantize) and a general search (for coverage
    of newer/smaller models unsloth may not have re-uploaded yet), and returns
    both lists so the agent can weigh recency/fit vs. plug-and-play ease.
    """
    query = f"{use_case} instruct {size_hint}".strip()
    return {
        "unsloth_prequantized": search_models(query, limit=limit, unsloth_only=True),
        "general": search_models(f"{query} instruct", limit=limit, unsloth_only=False),
        "note": (
            "Prefer an unsloth/*-bnb-4bit repo when one reasonably matches the "
            "use case/size — it's pre-quantized and known to work with "
            "FastLanguageModel.from_pretrained. Otherwise pick from `general` "
            "and set load_in_4bit accordingly in the recipe; Unsloth can "
            "quantize most causal LMs on the fly."
        ),
    }
