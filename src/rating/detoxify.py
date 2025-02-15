import torch
import numpy as np
from detoxify import Detoxify
from src.rating.base import RatingBackend, RatingResult


class DetoxifyRater(RatingBackend):
    """
    Detoxify rating backend using the Detoxify model.
    """

    def __init__(self, model_type: str = "original", device: str | None = None, amp: bool = False):
        """
        Initializes the Detoxify model.

        Args:
            model_type (str): Specifies which Detoxify model to load.
            device (str, optional): The device to run the model on ("cpu" or "cuda"). If None, uses "cuda" if available.
            amp (bool): Whether to use automatic mixed precision. 
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

    def rate(self, text: str, **kwargs) -> RatingResult:
        return self.rate_batch([text], **kwargs)[0]

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        preds = self.forward(texts)
        keys = list(preds.keys())
        values = list(preds.values())

        results = []
        for i, text in enumerate(texts):
            scores = {k: v[i] for k, v in zip(keys, values)}
            rating = RatingResult(text, scores=scores)
            results.append(rating)

        return results
