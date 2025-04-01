import re
from src.rating.base import RatingBackend, RatingResult
from src.generation.base import GenerationResult

from vllm import LLM, SamplingParams
from src.generation.base import *
from huggingface_hub import login
from transformers import AutoTokenizer


class LlamaGuardRater(RatingBackend, GenerationBackend):
    def __init__(
        self,
        hf_token: str,
        gpu_memory_utilization: float = 0.3,
        max_input_tokens: int = 100,
        max_batch_size: int = 1000,
        model_args: dict = None,
        sampling_args: dict = None,
    ):
        if model_args is None:
            model_args = {}

        if sampling_args is None:
            sampling_args = {}

        login(hf_token)

        model_name = "meta-llama/Llama-Guard-3-1B"
        overhead_tokens = 100
        max_output_tokens = 10

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        empty_input = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        empty_input_tokens = len(tokenizer.apply_chat_template(empty_input, add_generation_prompt=True))
        total_tokens = max_input_tokens + max_output_tokens + empty_input_tokens

        print("INFO: Overhead tokens: ", overhead_tokens)
        print("INFO: Empty input tokens: ", empty_input_tokens)
        print("INFO: Total sequence tokens: ", total_tokens)

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=total_tokens + overhead_tokens,
            max_seq_len_to_capture=total_tokens + overhead_tokens,
            max_num_seqs=max_batch_size,
            max_num_batched_tokens=total_tokens * max_batch_size + overhead_tokens,
            **model_args,
        )

        self.sampling_params = SamplingParams(
            temperature=sampling_args.pop("temperature", 0.0),
            max_tokens=max_output_tokens,
            **sampling_args,
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

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        outputs = self.llm.chat(messages, self.sampling_params, use_tqdm=False, add_generation_prompt=True)
        result = GenerationResult(prompt=prompt, output=outputs[0].outputs[0].text.strip())
        return result

    def generate_batch(self, prompts: list[str], **kwargs) -> list[GenerationResult]:
        messages = [[{"role": "user", "content": [{"type": "text", "text": t}]}] for t in prompts]
        outputs = self.llm.chat(messages, self.sampling_params, use_tqdm=False, add_generation_prompt=True)

        results = [GenerationResult(prompt=prompt, output=output.outputs[0].text.strip()) for prompt, output in zip(prompts, outputs)]
        return results

    def rate(self, text: str, **kwargs) -> RatingResult:
        response = self.generate(prompt=text, **kwargs)
        ratings = RatingResult(text, self.calc_scores(response))
        return ratings

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        responses = self.generate_batch(prompts=texts, **kwargs)
        ratings = [RatingResult(text, self.calc_scores(resp)) for text, resp in zip(texts, responses)]
        return ratings
