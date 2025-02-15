import torch
from transformers import BatchEncoding
from src.generation.vanilla_model import VanillaGenerator


class ChatGenerator(VanillaGenerator):
    """
    Chat generation backend using a Hugging Face model.
    These models are trained for conversational or chat-like responses (like chat-GPT).
    """
    def __init__(
        self,
        model_name: str,
        hub_token: str,
        max_input_tokens: int = 128,
        max_output_tokens: int = 50,
        device: str | None = None,
        **kwargs,
    ):

        super().__init__(
            model_name=model_name,
            hub_token=hub_token,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            device=device,
            **kwargs,
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def tokenize(self, text: list[str], **kwargs) -> BatchEncoding:
        messeges = [[{"role": "user", "content": t}] for t in text]
        return self.tokenizer.apply_chat_template(
            conversation=messeges,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=kwargs.pop("padding", True),
            truncation=kwargs.pop("truncation", True),
            max_length=self.max_input_tokens,
        )

    def forward(self, input_tokens: BatchEncoding, **kwargs) -> torch.Tensor:
        return self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            **kwargs,
        )
