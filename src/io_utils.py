import json
import logging
from pathlib import Path
from typing import Iterator

# Configure logging for production use.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from src.rating import RatingResult


def load_prompts(file_path: Path) -> list[str]:
    """
    Load prompts from a file, one prompt per line.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.exception(f"Failed to load prompts from {file_path}: {e}")
        raise


def save_ratings(ratings: list[RatingResult], file_path: Path, mode: str = "w") -> None:
    """
    Save a list of rating results to a file in JSON Lines format.

    Args:
        ratings (list[RatingResult]): The rating results to be saved.
        file_path (Path): The path to the output file.
        mode (str): File opening mode ('w' for overwrite, 'a' for append).
    """
    try:
        with file_path.open(mode, encoding="utf-8") as f:
            for rating in ratings:
                json.dump(rating.__dict__, f)
                f.write("\n")
    except Exception as e:
        logger.exception(f"Failed to save ratings to {file_path}: {e}")
        raise


def iter_ratings(file_path: Path) -> Iterator[RatingResult]:
    """
    Generator that yields RatingResult objects from a JSON Lines file.

    Args:
        file_path (Path): The path to the JSON Lines file containing rating results.

    Yields:
        RatingResult: Each rating result read from the file.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield RatingResult(**data)
                except Exception as e:
                    logger.error(f"Failed to decode line: {line}. Error: {e}")
    except Exception as e:
        logger.exception(f"Failed to read ratings from {file_path}: {e}")
        raise


def load_ratings(file_path: Path) -> list[RatingResult]:
    """
    Load all rating results from a JSON Lines file.

    Args:
        file_path (Path): The path to the JSON Lines file containing rating results.

    Returns:
        list[RatingResult]: A list of rating results.
    """
    return list(iter_ratings(file_path))
