from pathlib import Path
import os


def api_key_from_file(path: str) -> str:
    """
    Read Perspective API key from a file.

    Args:
        path (str): Path to the file containing the API key.

    Returns:
        str: The API key.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    key_file = Path(path)
    if key_file.exists():
        with key_file.open("r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise FileNotFoundError("API key file not found")


def api_key_from_env() -> str:
    """
    Read Perspective API key from the environment variable `PERSPECTIVE_API_KEY`.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the environment variable is not set.
    """
    key = os.environ.get("PERSPECTIVE_API_KEY", None)
    if key is None:
        raise ValueError("Please run `export PERSPECTIVE_API_KEY=`key` if you wish to use PerspectiveAPI.")
    return key
