import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from ..generation import GenerationResult


class AutoModelBackend:
    def __init__(self, num_outputs: int = 1, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", device: Optional[str] = None):
        """
        Initializes the generator backend.

        Args:
            num_outputs (int): Number of outputs to generate per prompt.
            model_name (str): The model identifier (or path) for llama3.2-7B.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). If not provided,
                                    it will automatically select 'cuda' if available.
        """
        self.num_outputs = num_outputs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 50, **generation_kwargs) -> GenerationResult:
        """
        Generates output(s) for a single prompt.

        Args:
            prompt (str): The prompt for generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            **generation_kwargs: Additional keyword arguments passed to the model's generate() method.

        Returns:
            GenerationResult: The generation result for the prompt.
        """
        try:
            # Encode the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            prompt_token_length = inputs["input_ids"].shape[1]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate outputs
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=self.num_outputs,
                do_sample=True,  # For diversity; adjust as needed
                **generation_kwargs,
            )

            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)
            
            # Decode each generated sequence.
            decoded_outputs = [
                self.tokenizer.decode(output[prompt_token_length:], skip_special_tokens=True)
                for output in outputs
            ]
            return GenerationResult(prompt=prompt, outputs=decoded_outputs)

        except Exception as e:
            # In case of any error during generation, capture the error message
            return GenerationResult(prompt=prompt, outputs=[], error=str(e))

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 50, **generation_kwargs) -> List[GenerationResult]:
        """
        Generates output(s) for a batch of prompts.

        Args:
            prompts (list[str]): List of prompts.
            max_new_tokens (int): Maximum number of new tokens to generate per prompt.
            **generation_kwargs: Additional keyword arguments passed to the model's generate() method.

        Returns:
            list[GenerationResult]: A list of generation results, one per prompt.
        """
        try:
            # Tokenize the prompts. Note that 'padding=True' pads all inputs to the same length.
            # We do NOT want to use the padded length as the prompt length.
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            # Compute each prompt's actual length by summing the attention mask (1 for real tokens, 0 for padding)
            prompt_token_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch_size,)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate outputs.
            # The returned tensor will have shape (batch_size * num_return_sequences, sequence_length)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=self.num_outputs,
                do_sample=True,  # For diversity; adjust as needed
                **generation_kwargs,
            )

            # Reshape outputs into (batch_size, num_return_sequences, sequence_length)
            batch_size = len(prompts)
            outputs = outputs.view(batch_size, self.num_outputs, -1)

            # For each prompt, decode only the tokens generated after the original prompt tokens.
            decoded_outputs = []
            for i in range(batch_size):
                prompt_length = prompt_token_lengths[i].item()
                prompt_generated = []
                for j in range(self.num_outputs):
                    generated_ids = outputs[i, j, :]
                    # Slice out the tokens corresponding to the new generation
                    new_ids = generated_ids[prompt_length:]
                    text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                    prompt_generated.append(text)
                decoded_outputs.append(prompt_generated)

            return [
                GenerationResult(prompt=prompt, outputs=outs)
                for prompt, outs in zip(prompts, decoded_outputs)
            ]

        except Exception as e:
            # In case of any error during generation, capture the error message.
            return [GenerationResult(prompt=prompt, outputs=[], error=str(e)) for prompt in prompts]
