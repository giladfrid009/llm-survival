import re
from collections import defaultdict

import torch
from src.rating.base import RatingBackend, RatingResult
from src.generation.base import GenerationResult
from src.generation.vanilla_model import VanillaGenerator
from transformers import BatchEncoding


class LlamaGuardGenerator(VanillaGenerator):
    def __init__(
        self,
        model_name: str,
        hub_token: str,
        max_input_tokens: int = 50,
        device: str | None = None,
        **kwargs,
    ):

        super().__init__(
            model_name=model_name,
            hub_token=hub_token,
            max_input_tokens=max_input_tokens,
            max_output_tokens=10,
            device=device,
            **kwargs,
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        self.category_map = {
            "S1": "Violent Crimes",
            "S2": "Non-Violent Crimes",
            "S3": "Sex-Related Crimes",
            "S4": "Child Sexual Exploitation",
            "S5": "Defamation",
            "S6": "Specialized Advice",
            "S7": "Privacy",
            "S8": "Intellectual Property",
            "S9": "Indiscriminate Weapons",
            "S10": "Hate",
            "S11": "Suicide & Self-Harm",
            "S12": "Sexual Content",
            "S13": "Elections",
        }

    def tokenize(self, text: list[str], kwargs: dict = {}) -> BatchEncoding:
        messeges = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in text]
        return self.tokenizer.apply_chat_template(
            conversation=messeges,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
            padding=kwargs.pop("padding", True),
            truncation=kwargs.pop("truncation", True),
            max_length=kwargs.pop("max_length", self.max_input_tokens + self.empty_input_tokens),
        )

    def forward(self, input_tokens: BatchEncoding, kwargs: dict = {}) -> torch.Tensor:
        return self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=kwargs.pop("do_sample", None),
            **kwargs,
        )


class LlamaGuardRater(RatingBackend):
    def __init__(
        self,
        model_name: str,
        hub_token: str,
        max_input_tokens: int,
        device: str | None = None,
        **kwargs,
    ):

        self.guard_generator = LlamaGuardGenerator(
            model_name=model_name,
            hub_token=hub_token,
            max_input_tokens=max_input_tokens,
            device=device,
            **kwargs,
        )

    def extract_reason(self, text) -> str:
        pattern = r"S(\d+)"
        match = re.search(pattern, text)
        cat = f"S{match.group(1)}" if match else ""
        return self.guard_generator.category_map.get(cat, "Unknown")

    def calc_scores(self, resp: GenerationResult) -> dict[str, float]:
        if "unsafe" in resp.output:
            reason = self.extract_reason(resp.output)
            return {"toxicity": 1.0, "reason": reason}

        return {"toxicity": 0.0}

    def rate(self, text: str, **kwargs) -> RatingResult:
        return self.rate_batch([text], **kwargs)[0]

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        responses = self.guard_generator.generate_batch(prompts=texts, **kwargs)
        ratings = [RatingResult(text, self.calc_scores(resp)) for text, resp in zip(texts, responses)]
        return ratings
