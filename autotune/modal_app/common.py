"""Shared Modal App/Image definitions for training and serving.

Requires the user to have run `modal token set` locally (or set
MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars) — Autotune does not manage Modal
credentials itself.
"""
import modal

app = modal.App("autotune")

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "unsloth",
        "peft>=0.11.0",
        "trl>=0.9.6",
        "transformers>=4.43.0",
        "datasets>=2.19.0",
        "accelerate>=0.31.0",
        "bitsandbytes>=0.43.1",
        "xformers",
    )
)

serving_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "unsloth",
        "peft>=0.11.0",
        "transformers>=4.43.0",
        "accelerate>=0.31.0",
        "bitsandbytes>=0.43.1",
        "fastapi[standard]",
    )
)

# Persistent volume: checkpoints/adapters/datasets survive across runs and
# across the train -> eval -> deploy steps of a single Autotune loop.
volume = modal.Volume.from_name("autotune-runs", create_if_missing=True)
VOLUME_MOUNT = "/vol/runs"
