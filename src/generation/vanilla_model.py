import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BatchEncoding
import huggingface_hub
from src.generation.base import GenerationResult, GenerationBackend


class VanillGenerator(GenerationBackend):
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype | str | None = "auto",
        device: str | None = None,
        api_key: str | None = None,
        **model_kwargs,
    ):
        """
        Initializes a HuggingFace vanilla non-tuned model backend for text generation.
        These model types are intended for sentance completion, and not for chatbot-like interactions.

        Args:
            model_name (str): The model identifier (or path).
            dtype (torch.dtype or str, optional): The data type to use for the model.
                If 'auto', it will infer dtype from model weights.
                If None, it will use torch.float32 dtype.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                If not provided, it will automatically select 'cuda' if available.
            api_key (str, optional): The HuggingFace API key.
                If not provided, a pop-up will appear to enter the key.
            **model_kwargs: Additional keyword arguments passed to the model creation function.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.dtype = dtype
        self.device = device

        huggingface_hub.login(token=api_key)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left",
        )

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            **model_kwargs,
        ).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt: str, max_len: int, **kwargs) -> GenerationResult:
        return self.generate_batch([prompt], max_len=max_len, **kwargs)[0]

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], max_len, **kwargs) -> list[GenerationResult]:
        try:

            input_tokens = self.tokenizer(
                conversation=prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            output_tokens = self.model.generate(
                **input_tokens,
                max_new_tokens=max_len,
                do_sample=True,
                **kwargs,
            )

            # Decode only newly generated tokens.
            start_idx = input_tokens["attention_mask"].shape[1]
            responses = self.tokenizer.batch_decode(
                sequences=output_tokens[:, start_idx:],
                skip_special_tokens=True,
            )

            return [GenerationResult(prompt=prompt, output=resp) for prompt, resp in zip(prompts, responses)]

        except Exception as e:
            # In case of any error during generation, capture the error message.
            return [GenerationResult(prompt=prompt, error=str(e)) for prompt in prompts]
