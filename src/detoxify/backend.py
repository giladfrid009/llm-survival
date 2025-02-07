from collections import OrderedDict

import torch
import numpy as np
from detoxify import Detoxify

from src.rating import RatingBackend, RatingResult


class DetoxifyBackend(RatingBackend):
    """
    Detoxify rating backend using the Detoxify model.
    """

    def __init__(self, model_type: str = "original", device: str | None = None, amp: bool = False):
        """
        Initializes the Detoxify model.

        Args:
            model_type (str): Specifies which Detoxify model to load. Defaults to "original".
            device (str, optional): The device to run the model on ("cpu" or "cuda"). If None, uses "cuda" if available.
            amp (bool): Whether to use automatic mixed precision. Defaults to False.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = Detoxify(model_type=model_type, device=device)
        self.amp = amp

    def rate(self, prompt: str) -> RatingResult:
        return self.rate_batch([prompt])[0]

    @torch.inference_mode()
    def rate_batch(self, prompts: list[str]) -> list[RatingResult]:
        try:
            if self.amp:
                with torch.autocast(device_type=self.device):
                    preds: dict[str, list[np.ndarray]] = self.model.predict(prompts)
            else:
                preds: dict[str, list[np.ndarray]] = self.model.predict(prompts)

            keys = list(preds.keys())
            values = list(preds.values())

            results = []
            for i, prompt in enumerate(prompts):
                scores = OrderedDict()
                for k, v in zip(keys, values):
                    scores[k] = v[i]

                rating = RatingResult(prompt=prompt, scores=scores)
                results.append(rating)

            return results

        except Exception as e:
            # Mark all prompts with the error if an exception occurs.
            return [RatingResult(prompt=p, error=str(e)) for p in prompts]
