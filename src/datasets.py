import pickle
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from .rating.base import RatingResult
from .survival_runner import SurvivalResult
from typing import List, Tuple

def make_survival(unfiltered_result: SurvivalResult, score_name: str, threshold: float) -> Tuple[str, int, bool]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    e = False
    t_tilde = len(unfiltered_result.ratings)
    for i, rating in enumerate(unfiltered_result.ratings):
        if rating.scores[score_name] >= threshold:
            e = True
            t_tilde = i+1
            break
    return unfiltered_result.prompt, t_tilde, e

def make_prop(unfiltered_result: SurvivalResult, score_name: str, threshold: float) -> Tuple[str, float, int]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    avg_score_violations = 0
    for rating in unfiltered_result.ratings:
        if rating.scores[score_name] >= threshold:
            avg_score_violations += 1
    avg_score_violations /= len(unfiltered_result.ratings)
    return unfiltered_result.prompt, avg_score_violations, len(unfiltered_result.ratings)

def make_multisample(unfiltered_result: SurvivalResult, score_name: str, threshold: float) -> List[Tuple[str, bool]]:
    try:
        assert(isinstance(unfiltered_result, SurvivalResult))
        assert(isinstance(unfiltered_result.ratings[0], RatingResult))
        assert(score_name in unfiltered_result.ratings[0].scores)
        assert(threshold >= 0)
        assert(threshold <= 1)
    except AssertionError as e:
        print(f"Invalid input: {e}")
    results = []
    for rating in unfiltered_result.ratings:
        results.append((unfiltered_result.prompt, rating.scores[score_name] >= threshold))
    return results

def make_classification(unfiltered_result: SurvivalResult, score_name: str, threshold: float) -> Tuple[str, bool]:
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
    def __init__(self, pkl_file: str, score_name: str, threshold: float):
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f)
        self.score_name = score_name
        self.threshold = threshold

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
        return make_survival(unfiltered_result, self.score_name, self.threshold)

class PropDataset(BaseDataset):
    def transform(self, unfiltered_result):
        return make_prop(unfiltered_result, self.score_name, self.threshold)

class ClassificationDataset(BaseDataset):
    def transform(self, unfiltered_result):
        return make_classification(unfiltered_result, self.score_name, self.threshold)

# Dataset class for multi-item output from make_multisample
class MultiSampleDataset(Dataset):
    def __init__(self, pkl_file: str, score_name: str, threshold: float):
        with open(pkl_file, "rb") as f:
            raw_data = pickle.load(f)
        self.score_name = score_name
        self.threshold = threshold
        self.samples = []  # This will hold each individual sample

        # For each raw item, apply make_multisample and flatten the results.
        for unfiltered_result in raw_data:
            # Each call returns a list of tuples, e.g., [(prompt, bool), ...]
            transformed = make_multisample(unfiltered_result, self.score_name, self.threshold)
            self.samples.extend(transformed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample[0]
        y = sample[1:] if len(sample) > 2 else sample[1]
        return x, y
