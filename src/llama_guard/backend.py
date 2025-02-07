from collections import OrderedDict

import torch
import numpy as np
import huggingface_hub as hf_hub
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.rating import RatingBackend, RatingResult

# TODO: MAYBE FINISH IMPLEMENTING LATER
class LlamaGuardBackend(RatingBackend):
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-3-1B",
        device: str | None = None,
        login_token: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device

        hf_hub.login(login_token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)

    @torch.inference_mode()
    def forward(self, prompts: list[str]):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, pad_token_id=0)
        return outputs

    def rate(self, prompt: str) -> RatingResult:
        return self.rate_batch([prompt])[0]

    @torch.inference_mode()
    def rate_batch(self, prompts: list[str]) -> list[RatingResult]:
        try:
            pass

        except Exception as e:
            # Mark all prompts with the error if an exception occurs.
            return [RatingResult(prompt=p, error=str(e)) for p in prompts]
