from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich import print_json

from autotune.agent.loop import run_loop
from autotune.core.state import RunState

app = typer.Typer(help="Autotune: agentic post-training. Describe a use case, hand it mock data, let it fine-tune and deploy.")
console = Console()


@app.command()
def start(
    use_case: str = typer.Option(..., "--use-case", "-u", help="Business use case description."),
    seed_data: Path = typer.Option(..., "--seed-data", "-d", help="JSONL file of mock/seed examples."),
    guidance: str = typer.Option("", "--guidance", "-g", help="Optional steering guidance for the agent."),
    max_turns: int = typer.Option(40, help="Max agent loop turns before giving up."),
):
    """Start a new Autotune run."""
    seed_examples = [json.loads(l) for l in seed_data.read_text().splitlines() if l.strip()]
    state = RunState.create(use_case=use_case)
    console.print(f"[bold]Run created:[/bold] {state.run_id}")
    run_loop(state, seed_examples, guidance=guidance, max_turns=max_turns)


@app.command()
def resume(
    run_id: str,
    guidance: str = typer.Option("", "--guidance", "-g"),
    max_turns: int = typer.Option(40),
):
    """Resume an existing run (re-enters the loop with its current state as context)."""
    state = RunState.load(run_id)
    data_path = state.dir / "data" / "train.jsonl"
    seed_examples = []
    if data_path.exists():
        seed_examples = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    run_loop(state, seed_examples, guidance=guidance or "Resuming a previous run — see run history in your context.", max_turns=max_turns)


@app.command("list")
def list_runs():
    """List all runs."""
    for run_id in RunState.list_runs():
        state = RunState.load(run_id)
        console.print(f"{run_id}  [bold]{state.status}[/bold]  {state.use_case[:60]}")


@app.command()
def show(run_id: str):
    """Show a run's full manifest."""
    state = RunState.load(run_id)
    print_json(data=state.__dict__)


if __name__ == "__main__":
    app()
