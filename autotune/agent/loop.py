"""The Claude tool-use agentic loop that drives a run end to end, pausing for
user input when the agent calls `ask_user` and stopping when it calls
`finish`.
"""
from __future__ import annotations

from anthropic import Anthropic
from rich.console import Console

from autotune import config
from autotune.agent.prompts import SYSTEM_PROMPT
from autotune.agent.tools import TOOLS, Dispatcher
from autotune.core.state import RunState

console = Console()


def _user_kickoff_message(state: RunState, seed_examples: list[dict], guidance: str) -> str:
    return f"""Business use case:
{state.use_case}

Seed/mock examples ({len(seed_examples)} provided):
{seed_examples!r}

Initial user guidance: {guidance or "(none)"}

Begin the run."""


def run_loop(
    state: RunState,
    seed_examples: list[dict],
    guidance: str = "",
    max_turns: int = 40,
) -> None:
    client = Anthropic(api_key=config.anthropic_api_key())
    model = config.agent_model()
    dispatcher = Dispatcher(state, seed_examples)

    messages = [{"role": "user", "content": _user_kickoff_message(state, seed_examples, guidance)}]

    for turn in range(max_turns):
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        for block in resp.content:
            if block.type == "text" and block.text.strip():
                console.print(f"[bold cyan]agent[/bold cyan] {block.text.strip()}")

        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            console.print(f"[dim]→ {block.name}({block.input})[/dim]")
            try:
                output = dispatcher.dispatch(block.name, block.input)
                is_error = False
            except Exception as e:  # noqa: BLE001
                output = f"Error: {e}"
                is_error = True
            console.print(f"[dim]← {output[:500]}[/dim]")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})

        if dispatcher.paused_for_user:
            console.print(f"\n[bold yellow]Autotune needs your input:[/bold yellow] {dispatcher.paused_for_user}")
            answer = console.input("[bold]> [/bold]")
            dispatcher.paused_for_user = None
            messages.append({"role": "user", "content": f"User's answer: {answer}"})
            continue

        if dispatcher.finished_summary:
            console.print(f"\n[bold green]Run finished:[/bold green] {dispatcher.finished_summary}")
            break
    else:
        console.print("[bold red]Max turns reached without finishing.[/bold red]")

    state.save()
