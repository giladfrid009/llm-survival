import pickle
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from .rating.base import RatingResult
from .survival_runner import SurvivalResult
from typing import List, Tuple
import numpy as np

def make_survival(unfiltered_result: SurvivalResult, score_name: str, threshold: float, max_samples: int = np.inf) -> Tuple[str, int, bool]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    e = False
    max_samples = min(max_samples, len(unfiltered_result.ratings))
    ratings = unfiltered_result.ratings[:max_samples]
    t_tilde = len(ratings)
    for i, rating in enumerate(ratings):
        if rating.scores[score_name] >= threshold:
            e = True
            t_tilde = i+1
            break
    return unfiltered_result.prompt, t_tilde, e

def make_prop(unfiltered_result: SurvivalResult, score_name: str, threshold: float, max_samples: int = np.inf) -> Tuple[str, float, int]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    max_samples = min(max_samples, len(unfiltered_result.ratings))
    ratings = unfiltered_result.ratings[:max_samples]
    avg_score_violations = 0
    for rating in ratings:
        if rating.scores[score_name] >= threshold:
            avg_score_violations += 1
    avg_score_violations /= len(ratings)
    return unfiltered_result.prompt, avg_score_violations, len(ratings)

def make_multisample(unfiltered_result: SurvivalResult, score_name: str, threshold: float, max_samples: int = np.inf) -> List[Tuple[str, bool]]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    max_samples = min(max_samples, len(unfiltered_result.ratings))
    ratings = unfiltered_result.ratings[:max_samples]
    results = []
    for rating in ratings:
        results.append((unfiltered_result.prompt, rating.scores[score_name] >= threshold))
    return results

def make_classification(unfiltered_result: SurvivalResult, score_name: str, threshold: float, max_samples: int = np.inf) -> Tuple[str, bool]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    first_rating = unfiltered_result.ratings[0]
    is_above_threshold = first_rating.scores[score_name] >= threshold
    return unfiltered_result.prompt, is_above_threshold

# Base class for single-item transformations
class BaseDataset(Dataset, ABC):
    def __init__(self, pkl_file: str, score_name: str, threshold: float, max_samples: int = np.inf):
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f)
        self.score_name = score_name
        self.threshold = threshold
        self.max_samples = max_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        unfiltered_result = self.data[idx]
        transformed = self.transform(unfiltered_result)
        x = transformed[0]
        y = transformed[1:] if len(transformed) > 2 else transformed[1]
        return x, y

    @abstractmethod
    def transform(self, unfiltered_result):
        """
        Transform the raw item into the desired format.
        Implement this in subclasses.
        """
        pass

# Subclasses for single-item outputs
class SurvivalDataset(BaseDataset):
    def transform(self, unfiltered_result):
        return make_survival(unfiltered_result, self.score_name, self.threshold, self.max_samples)

class PropDataset(BaseDataset):
    def transform(self, unfiltered_result):
        return make_prop(unfiltered_result, self.score_name, self.threshold, self.max_samples)

class ClassificationDataset(BaseDataset):
    def transform(self, unfiltered_result):
        return make_classification(unfiltered_result, self.score_name, self.threshold, self.max_samples)

# Dataset class for multi-item output from make_multisample
class MultiSampleDataset(Dataset):
    def __init__(self, pkl_file: str, score_name: str, threshold: float, max_samples: int = np.inf):
        with open(pkl_file, "rb") as f:
            raw_data = pickle.load(f)
        self.score_name = score_name
        self.threshold = threshold
        self.samples = []  # This will hold each individual sample
        self.max_samples = max_samples

        # For each raw item, apply make_multisample and flatten the results.
        for unfiltered_result in raw_data:
            # Each call returns a list of tuples, e.g., [(prompt, bool), ...]
            transformed = make_multisample(unfiltered_result, self.score_name, self.threshold, self.max_samples)
            self.samples.extend(transformed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample[0]
        y = sample[1:] if len(sample) > 2 else sample[1]
        return x, y

class PromptOnlyDataset(Dataset):
    def __init__(self, pkl_file: str):
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f)
        for i in range(len(self.data)):
            self.data[i] = self.data[i].prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
