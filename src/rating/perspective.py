import time
import threading

from googleapiclient import discovery
from src.rating.base import RatingBackend, RatingResult

PERSPECTIVE_API_ATTRIBUTES = (
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "THREAT",
    "PROFANITY",
    "SEXUALLY_EXPLICIT",
    "FLIRTATION",
)


class PerspectiveRater(RatingBackend):
    """
    Rating backend using Google's Perspective API.

    NOTE on performance:
      - Enforces a per-request rate limit using a simple lock.
      - For each call, the API waits (if necessary) so that no more than
        `rate_limit` requests are issued per second.
    """

    def __init__(self, api_key: str, rate_limit: int = 1):
        """
        Args:
            api_key (str): The API key for the Perspective API.
            rate_limit (int): Maximum number of API requests per second.
        """
        self.api_key = api_key
        self.rate_limit = rate_limit

        self._lock = threading.Lock()
        self._last_request_time = time.time() - 1.0
        self.service = self._build_service()

    def _build_service(self):
        """
        Build and return the Google API client for Perspective.
        """
        return discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=self.api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

    def _enforce_rate_limit(self):
        """
        Enforce that at least (1 / rate_limit) seconds have passed between calls.
        Uses a lock to coordinate between threads.
        """
        with self._lock:
            now = time.time()
            min_interval = 1.0 / self.rate_limit
            elapsed = now - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_request_time = time.time()

    def rate(self, text: str) -> RatingResult:

        self._enforce_rate_limit()

        # see https://support.perspectiveapi.com/s/about-the-api-methods?language=en_US
        request_body = {
            "comment": {"text": text, "type": "PLAIN_TEXT"},
            "requestedAttributes": {attr: {"scoreType": "PROBABILITY"} for attr in PERSPECTIVE_API_ATTRIBUTES},
            "spanAnnotations": False,
            "languages": ["en"],
            "doNotStore": True,
        }

        response = self.service.comments().analyze(body=request_body).execute()

        # Extract only the summary scores (score of entire text).
        attribute_scores = response.get("attributeScores", {})
        scores = {}
        for attr, data in attribute_scores.items():
            summary_scores = data.get("summaryScore", {})
            score_value = summary_scores.get("value")
            scores[attr.lower()] = score_value

        return RatingResult(text, scores=scores)
