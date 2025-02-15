from src.rating.base import RatingBackend, RatingResult
from src.generation.base import GenerationResult
from src.generation.chat_model import ChatGenerator


class LlamaGuardRater(RatingBackend):
    def __init__(
        self,
        model_name: str,
        hub_token: str,
        max_input_tokens: int,
        device: str | None = None,
        **kwargs,
    ):

        self.guard_model = ChatGenerator(
            model_name=model_name,
            hub_token=hub_token,
            max_input_tokens=max_input_tokens,
            max_output_tokens=15,
            device=device,
            **kwargs,
        )

    def calc_scores(self, resp: GenerationResult) -> dict[str, float]:
        score = 1.0 if "unsafe" in resp.output else 0.0
        return {"toxicity": score}

    def rate(self, text: str, **kwargs) -> RatingResult:
        resp = self.guard_model.generate(prompt=text, add_generation_prompt=False, **kwargs)
        return RatingResult(text, self.calc_scores(resp))

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        # TODO: modifying add_generation_prompt chanegs the results. verify which correct value should be here.
        # TODO: for some reason behavior changes between batch_size=1 and batch_size>1
        # maybe this model doesnt work with attention masks?
        responses = self.guard_model.generate_batch(prompts=texts, add_generation_prompt=False, **kwargs)
        ratings = [RatingResult(text, self.calc_scores(resp)) for text, resp in zip(texts, responses)]
        return ratings
