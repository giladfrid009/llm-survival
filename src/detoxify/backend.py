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

    @torch.inference_mode()
    def forward(self, texts: list[str]) -> dict[str, list[np.ndarray]]:
        if self.amp:
            with torch.autocast(device_type=self.device):
                return self.model.predict(texts)
        else:
            return self.model.predict(texts)

    def rate(self, text: str) -> RatingResult:
        return self.rate_batch([text])[0]

    def rate_batch(self, texts: list[str]) -> list[RatingResult]:
        try:
            preds = self.forward(texts)
            keys = list(preds.keys())
            values = list(preds.values())

            results = []
            for i, text in enumerate(texts):
                scores = OrderedDict()
                for k, v in zip(keys, values):
                    scores[k] = v[i]

                rating = RatingResult(text=text, scores=scores)
                results.append(rating)

            return results

        except Exception as e:
            # Mark all texts with the error if an exception occurs.
            return [RatingResult(text=t, error=str(e)) for t in texts]
