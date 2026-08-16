"""Modal function that runs the freshly trained adapter over the held-out
test split and writes predictions back to the volume. Judging (LLM-as-judge
via Claude) happens locally in autotune/core/eval.py so it can use the same
Anthropic key as the agent loop, not a Modal secret.
"""
import json

from autotune.modal_app.common import VOLUME_MOUNT, app, serving_image, volume


@app.function(image=serving_image, volumes={VOLUME_MOUNT: volume}, gpu="A10G", timeout=60 * 30)
def generate_predictions(run_id: str, split: str = "test", max_new_tokens: int = 512) -> str:
    from unsloth import FastLanguageModel

    run_dir = f"{VOLUME_MOUNT}/{run_id}"
    metrics = json.loads(open(f"{run_dir}/train_metrics.json").read())

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=metrics["base_model"], max_seq_length=4096, load_in_4bit=True
    )
    model.load_adapter(metrics["adapter_dir"])
    FastLanguageModel.for_inference(model)

    examples = [json.loads(l) for l in open(f"{run_dir}/data/{split}.jsonl") if l.strip()]
    predictions = []
    for ex in examples:
        msgs = ex["messages"]
        prompt_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
        expected = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else None

        inputs = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

        predictions.append({"prompt": prompt_msgs, "expected": expected, "actual": completion})

    out_path = f"{run_dir}/predictions_{split}.jsonl"
    with open(out_path, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    volume.commit()
    return out_path


@app.local_entrypoint()
def main(run_id: str, split: str = "test"):
    print(generate_predictions.remote(run_id, split))
