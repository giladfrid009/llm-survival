import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import huggingface_hub
from src.generation.base import GenerationResult, GenerationBackend


class ChatGenerator(GenerationBackend):
    def __init__(
        self,
        model_name: str,
        max_input_tokens: int = 50,
        max_new_tokens: int = 50,
        dtype: torch.dtype | str | None = "auto",
        device: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initializes a HuggingFace chat-based model backend for text generation.
        These model types are intended for chatbot-like interactions.

        Args:
            model_name (str): The model identifier (or path).
            max_input_tokens (int): The maximum number of tokens to use as input.
            max_new_tokens (int): The maximum number of tokens to generate.
            dtype (torch.dtype or str, optional): The data type to use for the model.
                If 'auto', it will infer dtype from model weights.
                If None, it will use torch.float32 dtype.
            device (str, optional): Device to run the model on ('cpu' or 'cuda').
                If not provided, it will automatically select 'cuda' if available.
            api_key (str, optional): The HuggingFace API key.
                If not provided, a pop-up will appear to enter the key.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
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
        ).eval()

        # if does not exist create a new padding token, 
        # add it to the tokenizer and resize token embedding accordingly
        if self.tokenizer.pad_token is None:
            pad_token = "<PAD>"
            self.tokenizer.add_special_tokens({"pad_token": pad_token})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token)
            self.model.generation_config.pad_token_id = self.tokenizer.convert_tokens_to_ids(pad_token)
            
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        return self.generate_batch([prompt] **kwargs)[0]

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str], **kwargs) -> list[GenerationResult]:           
        messeges = [[{"role": "user", "content": prompt}] for prompt in prompts]

        input_tokens = self.tokenizer.apply_chat_template(
            conversation=messeges,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=kwargs.pop("padding", True),
            truncation=kwargs.pop("truncation", True),
            max_length=self.max_input_tokens,
        ).to(self.device)

        output_tokens = self.model.generate(
            **input_tokens,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
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

        
