"""Thin wrapper around the Modal CLI/SDK for moving a run's files to/from the
shared volume and invoking the train/eval/serve Modal functions. Uses the
`modal` CLI via subprocess for volume put/get (simplest, matches what a user
gets from `modal token set`) and the Python SDK for function calls.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

VOLUME_NAME = "autotune-runs"


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def push_run(run_dir: Path, run_id: str) -> None:
    """Uploads local runs/<run_id>/ to the Modal volume at /<run_id>/."""
    _run(["modal", "volume", "put", VOLUME_NAME, str(run_dir), f"/{run_id}", "-f"])


def pull_file(run_id: str, remote_relpath: str, local_dir: Path) -> Path:
    """Downloads a single file from the volume back to local_dir."""
    local_dir.mkdir(parents=True, exist_ok=True)
    remote = f"/{run_id}/{remote_relpath}"
    _run(["modal", "volume", "get", VOLUME_NAME, remote, str(local_dir), "-f"])
    return local_dir / Path(remote_relpath).name


def train(run_id: str, gpu: str = "A10G") -> dict:
    import modal

    fn = modal.Function.from_name("autotune", "train")
    return fn.with_options(gpu=gpu).remote(run_id)


def generate_predictions(run_id: str, split: str = "test", gpu: str = "A10G") -> str:
    import modal

    fn = modal.Function.from_name("autotune", "generate_predictions")
    return fn.with_options(gpu=gpu).remote(run_id, split)


def deploy_serving(run_id: str) -> None:
    """Deploys autotune/modal_app/serve.py with AUTOTUNE_RUN_ID set so the
    Model class picks up this run's adapter."""
    import os

    env = {**os.environ, "AUTOTUNE_RUN_ID": run_id}
    subprocess.run(
        ["modal", "deploy", "-m", "autotune.modal_app.serve"],
        env=env,
        check=True,
    )
