"""Training recipe: a plain JSON config the agent proposes/edits between loop
iterations, and that autotune/modal_app/train.py consumes directly. Kept as
data (not code) so the agent can tweak individual fields without regenerating
a script.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"


@dataclass
class Recipe:
    base_model: str = DEFAULT_BASE_MODEL
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    # LoRA / PEFT
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # Optimization
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    seed: int = 0

    # Modal / infra
    gpu: str = "A10G"  # e.g. "A10G", "L40S", "A100-40GB", "H100"
    timeout_s: int = 60 * 60 * 2

    notes: str = ""  # freeform: agent's rationale for this recipe, for the run log

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, run_dir: Path) -> Path:
        path = run_dir / "recipe.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def load(cls, run_dir: Path) -> "Recipe":
        path = run_dir / "recipe.json"
        return cls.from_dict(json.loads(path.read_text()))
