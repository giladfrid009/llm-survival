import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, BatchEncoding
from src.generation.base import GenerationResult, GenerationBackend


class VanillaGeneratorHF(GenerationBackend):
    """
    Vanilla generation backend using a Hugging Face model.
    These models are trained for pure next-token generation, and not for conversational or chat-like responses.
    """

    def __init__(
        self,
        model_name: str,
        hub_token: str,
        max_input_tokens: int = 50,
        max_output_tokens: int = 50,
        device: str | None = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str): The model identifier (or path).
            hub_token (str): The HuggingFace API token.
            max_input_tokens (int): The maximum number of tokens to use as input.
            max_output_tokens (int): The maximum number of new tokens to generate.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                If not provided, it will automatically select 'cuda' if available.
            **kwargs: Additional keyword arguments passed to the model initialization.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model and tokenizer.
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side=kwargs.pop("padding_side", "left"),
            token=hub_token,
        )

        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            token=hub_token,
            **kwargs,
        ).eval()

        # Set the padding token to the EOS if it is None.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            model.generation_config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        # Set parameters.
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.device = device
        self.model = model
        self.tokenizer = tokenizer

        # Compute the number of tokens for an empty input.
        self.empty_input_tokens = 0
        empty_input = self.tokenize(text=[""], kwargs={"max_length": None, "truncation": False, "padding": False})
        self.empty_input_tokens: int = empty_input["input_ids"].shape[1]

    def tokenize(self, text: list[str], kwargs: dict = {}) -> BatchEncoding:
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=kwargs.pop("padding", True),
            truncation=kwargs.pop("truncation", True),
            max_length=kwargs.pop("max_length", self.max_input_tokens + self.empty_input_tokens),
        )

    def forward(self, input_tokens: BatchEncoding, kwargs: dict = {}) -> torch.Tensor:
        return self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_output_tokens,
            do_sample=kwargs.pop("do_sample", True),
            **kwargs,
        )

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        return self.generate_batch([prompt], **kwargs)[0]

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], **kwargs) -> list[GenerationResult]:
        input_tokens = self.tokenize(prompts, kwargs).to(self.model.device)
        output_tokens = self.forward(input_tokens, kwargs)

        # Decode only newly generated tokens.
        start_idx = input_tokens["attention_mask"].shape[1]
        responses = self.tokenizer.batch_decode(
            sequences=output_tokens[:, start_idx:],
            skip_special_tokens=True,
        )

        return [GenerationResult(prompt=prompt, output=resp) for prompt, resp in zip(prompts, responses)]
