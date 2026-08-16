"""Run state: every Autotune run is a directory under runs/<run_id>/ with a
manifest.json that the agent reads/updates on every loop iteration. This is
what makes a run resumable and inspectable outside the agent loop.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent.parent / "runs"


@dataclass
class RunState:
    run_id: str
    use_case: str
    base_model: str | None = None
    status: str = "planning"  # planning -> data -> training -> eval -> deploy -> done
    recipe: dict = field(default_factory=dict)
    dataset: dict = field(default_factory=dict)  # paths, counts, split sizes
    train_history: list = field(default_factory=list)  # list of {iteration, config, metrics}
    eval_history: list = field(default_factory=list)  # list of {iteration, metrics, judge_notes}
    deployment: dict = field(default_factory=dict)  # modal endpoint url, adapter path
    log: list = field(default_factory=list)  # human-readable event log
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def dir(self) -> Path:
        d = RUNS_DIR / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def save(self) -> None:
        self.updated_at = time.time()
        self.manifest_path.write_text(json.dumps(asdict(self), indent=2))

    def note(self, message: str) -> None:
        self.log.append({"t": time.time(), "message": message})
        self.save()

    @classmethod
    def create(cls, use_case: str) -> "RunState":
        run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        state = cls(run_id=run_id, use_case=use_case)
        state.save()
        return state

    @classmethod
    def load(cls, run_id: str) -> "RunState":
        path = RUNS_DIR / run_id / "manifest.json"
        data = json.loads(path.read_text())
        return cls(**data)

    @classmethod
    def list_runs(cls) -> list[str]:
        if not RUNS_DIR.exists():
            return []
        return sorted(p.name for p in RUNS_DIR.iterdir() if (p / "manifest.json").exists())
