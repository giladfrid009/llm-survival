from vllm import LLM, SamplingParams
from src.generation.base import *
from huggingface_hub import login


class VanillaGeneratorVLLM(GenerationBackend):
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        max_output_tokens: int = 100,
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
            gpu_memory_utilization=model_args.pop("gpu_memory_utilization", 0.95),
            **model_args,
        )

        self.sampling_params = SamplingParams(
            temperature=sampling_args.pop("temperature", 1.0),
            top_k=sampling_args.pop("top_k", -1),
            top_p=sampling_args.pop("top_p", 0.8),
            max_tokens=max_output_tokens,
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
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
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
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=True)
        results = [GenerationResult(prompt=prompt, output=output.outputs[0].text.strip()) for prompt, output in zip(prompts, outputs)]
        return results
