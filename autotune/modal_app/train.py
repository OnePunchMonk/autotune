"""Modal training entrypoint. Reads recipe.json + data/{train,val}.jsonl for
a run_id from the shared volume (pushed there by the CLI via `modal volume
put` before this is invoked), runs an Unsloth + PEFT LoRA fine-tune, writes
the adapter and a metrics.json back to the volume.

Invoked from the local agent loop via:
    modal.Function.lookup("autotune", "train").with_options(gpu=recipe.gpu).remote(run_id)
"""
import json
import os

import modal

from autotune.modal_app.common import VOLUME_MOUNT, app, training_image, volume


@app.function(
    image=training_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=60 * 60 * 4,
    gpu="A10G",
)
def train(run_id: str) -> dict:
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    run_dir = f"{VOLUME_MOUNT}/{run_id}"
    recipe = json.loads(open(f"{run_dir}/recipe.json").read())

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=recipe["base_model"],
        max_seq_length=recipe["max_seq_length"],
        load_in_4bit=recipe["load_in_4bit"],
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=recipe["lora_r"],
        lora_alpha=recipe["lora_alpha"],
        lora_dropout=recipe["lora_dropout"],
        target_modules=recipe["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=recipe["seed"],
    )

    def formatting(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )

    train_ds = load_dataset("json", data_files=f"{run_dir}/data/train.jsonl", split="train")
    val_path = f"{run_dir}/data/val.jsonl"
    val_ds = (
        load_dataset("json", data_files=val_path, split="train")
        if os.path.exists(val_path)
        else None
    )

    output_dir = f"{run_dir}/checkpoints"
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=recipe["per_device_train_batch_size"],
        gradient_accumulation_steps=recipe["gradient_accumulation_steps"],
        num_train_epochs=recipe["num_train_epochs"],
        learning_rate=recipe["learning_rate"],
        warmup_ratio=recipe["warmup_ratio"],
        weight_decay=recipe["weight_decay"],
        lr_scheduler_type=recipe["lr_scheduler_type"],
        seed=recipe["seed"],
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=50 if val_ds is not None else None,
        save_strategy="no",
        report_to=[],
        max_seq_length=recipe["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
        formatting_func=formatting,
    )

    result = trainer.train()

    adapter_dir = f"{run_dir}/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    eval_metrics = trainer.evaluate() if val_ds is not None else {}

    metrics = {
        "train_loss": result.training_loss,
        "eval": eval_metrics,
        "adapter_dir": adapter_dir,
        "base_model": recipe["base_model"],
    }
    with open(f"{run_dir}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    volume.commit()
    return metrics


@app.local_entrypoint()
def main(run_id: str):
    print(train.remote(run_id))
