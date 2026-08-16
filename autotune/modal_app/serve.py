"""Modal deployment: serves a fine-tuned adapter behind an OpenAI-compatible
/v1/chat/completions endpoint using Unsloth's fast inference. One serving
class per run_id, deployed on demand by the agent loop's `deploy` tool via
`modal deploy` with AUTOTUNE_RUN_ID set, or programmatically through
modal.Cls lookups.
"""
import os

import modal
from fastapi import FastAPI
from pydantic import BaseModel

from autotune.modal_app.common import VOLUME_MOUNT, app, serving_image, volume

RUN_ID = os.environ.get("AUTOTUNE_RUN_ID", "")

web_app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]
    max_new_tokens: int = 512
    temperature: float = 0.7


@app.cls(
    image=serving_image,
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G",
    scaledown_window=60 * 5,
)
class Model:
    run_id: str = modal.parameter(default=RUN_ID)

    @modal.enter()
    def load(self):
        import json

        from unsloth import FastLanguageModel

        run_dir = f"{VOLUME_MOUNT}/{self.run_id}"
        metrics = json.loads(open(f"{run_dir}/train_metrics.json").read())
        base_model = metrics["base_model"]
        adapter_dir = metrics["adapter_dir"]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model, max_seq_length=4096, load_in_4bit=True
        )
        self.model.load_adapter(adapter_dir)
        FastLanguageModel.for_inference(self.model)

    @modal.method()
    def generate(self, messages: list[dict], max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        text = self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        return text

    @modal.asgi_app()
    def web(self):
        @web_app.post("/v1/chat/completions")
        def chat(req: ChatRequest):
            reply = self.generate.local(req.messages, req.max_new_tokens, req.temperature)
            return {"choices": [{"message": {"role": "assistant", "content": reply}}]}

        return web_app
