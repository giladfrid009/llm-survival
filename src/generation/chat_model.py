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
        max_input_tokens: int = 50,
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

    def tokenize(self, text: list[str], kwargs: dict = {}) -> BatchEncoding:
        # Create a conversation structure for each text input
        messages = [[{"role": "user", "content": t}] for t in text]
        add_generation_prompt = kwargs.pop("add_generation_prompt", True)
        inputs = []
        for conversation in messages:
            # Build a string from the conversation messages.
            conversation_str = ""
            for message in conversation:
                # You can adjust this formatting as needed by your model.
                conversation_str += f"{message['role']}: {message['content']}\n"
            if add_generation_prompt:
                # Append a prompt marker (modify as appropriate for your use case).
                conversation_str += "Assistant:"
            inputs.append(conversation_str.strip())

        return self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=kwargs.pop("padding", True),
            truncation=kwargs.pop("truncation", True),
            max_length=kwargs.pop("max_length", self.max_input_tokens + self.empty_input_tokens),
        )

    def forward(self, input_tokens: BatchEncoding, kwargs: dict = {}) -> torch.Tensor:
        return self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_output_tokens,
            eos_token_id=self.terminators,
            do_sample=kwargs.pop("do_sample", True),
            **kwargs,
        )
