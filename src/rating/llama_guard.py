import re
from src.rating.base import RatingBackend, RatingResult
from src.generation.base import GenerationResult

from vllm import LLM, SamplingParams
from src.generation.base import *
from huggingface_hub import login


class LlamaGuardGenerator(GenerationBackend):
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        model_args: dict = None,
        sampling_args: dict = None,
    ):
        if model_args is None:
            model_args = {}

        if sampling_args is None:
            sampling_args = {}

        login(hf_token)

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=model_args.pop("gpu_memory_utilization", 0.35),
            **model_args,
        )

        self.sampling_params = SamplingParams(
            temperature=sampling_args.pop("temperature", 0.0),
            top_k=sampling_args.pop("top_k", -1),
            top_p=sampling_args.pop("top_p", 1),
            max_tokens=sampling_args.pop("max_tokens", 5),
            **sampling_args,
        )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate text for the given prompt.

        Args:
            prompt (str): The input prompt.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            GenerationResult: The result, containing generated text or an error message.
        """
        messages = [{"role": "user", "content": prompt}]
        outputs = self.llm.chat(messages, self.sampling_params, use_tqdm=False, add_generation_prompt=True)
        result = GenerationResult(prompt=prompt, output=outputs[0].outputs[0].text.strip())
        return result

    def generate_batch(self, prompts: list[str], **kwargs) -> list[GenerationResult]:
        """
        Generate an entire batch of prompts sequentially by calling `generate()` for each.

        Args:
            prompts (list[str]): A list of prompt strings.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            list[GenerationResult]: The results for each prompt.
        """
        messages = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in prompts]
        outputs = self.llm.chat(messages, self.sampling_params, use_tqdm=False, add_generation_prompt=True)
        
        results = [GenerationResult(prompt=prompt, output=output.outputs[0].text.strip()) for prompt, output in zip(prompts, outputs)]
        return results


class LlamaGuardRater(RatingBackend):
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        **kwargs,
    ):

        self.guard_generator = LlamaGuardGenerator(
            model_name=model_name,
            hf_token=hf_token,
            **kwargs,
        )

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
            "S14": "Code Interpreter Abuse",
        }

    def extract_reason(self, text) -> str:
        pattern = r"S(\d+)"
        match = re.search(pattern, text)
        cat = f"S{match.group(1)}" if match else ""
        return self.category_map.get(cat, "Unknown")

    def calc_scores(self, resp: GenerationResult) -> dict[str, float]:
        if "unsafe" in resp.output:
            reason = self.extract_reason(resp.output)
            return {"toxicity": 1.0, "reason": reason}

        return {"toxicity": 0.0}

    def rate(self, text: str, **kwargs) -> RatingResult:
        response = self.guard_generator.generate(prompt=text, **kwargs)
        ratings=RatingResult(text, self.calc_scores(response))
        return ratings

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        responses = self.guard_generator.generate_batch(prompts=texts, **kwargs)
        ratings = [RatingResult(text, self.calc_scores(resp)) for text, resp in zip(texts, responses)]
        return ratings
