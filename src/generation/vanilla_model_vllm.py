from vllm import LLM, SamplingParams
from src.generation.base import *
from src import utils
from transformers import AutoTokenizer
from logging import getLogger


class VanillaGeneratorVLLM(GenerationBackend):
    def __init__(
        self,
        model_name: str,
        hf_token: str,
        gpu_memory_utilization: float = 0.5,
        max_input_tokens: int = 100,
        max_output_tokens: int = 100,
        max_batch_size: int = 1000,
        model_args: dict = None,
        sampling_args: dict = None,
    ):
        if model_args is None:
            model_args = {}

        if sampling_args is None:
            sampling_args = {}

        # login(hf_token, new_session=False)

        overhead_tokens = 100
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        empty_input_tokens = len(tokenizer("")["input_ids"])
        total_tokens = max_input_tokens + max_output_tokens + empty_input_tokens

        logger = getLogger(__name__)
        logger.info("Overhead tokens: %d", overhead_tokens)
        logger.info("Empty input tokens: %d", empty_input_tokens)
        logger.info("Total sequence tokens: %d", total_tokens)

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
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        results = [GenerationResult(prompt=prompt, output=output.outputs[0].text.strip()) for prompt, output in zip(prompts, outputs)]
        return results

    def __del__(self):
        """
        Clean up resources when the object is deleted.
        """
        # Delete the llm object and free the memory
        if hasattr(self, "llm"):
            del self.llm
            utils.full_cleanup()