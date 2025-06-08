"""Central configuration values shared across scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import torch
from src import utils

###############################################################################
# Default paths and model hyper-parameters
###############################################################################

# Path to a text file containing a HuggingFace API key used for generator models
hf_key_path = "HF_KEY.txt"

# Default model and generation settings
default_model_name = "meta-llama/Llama-3.2-1B"  # model identifier used for generation
default_max_input_tokens = 40  # max tokens from the prompt passed to the model
default_max_output_tokens = 30  # max tokens to sample from the model
default_batch_size = 1500  # number of prompts per survival batch
default_max_attempts = 40  # default maximum number of generations per prompt

# Dataset and model paths used across experiments
# ``default_test_prompts_path`` and ``default_test_surv_time_path`` are produced
# by ``prepare_test_set.py``
default_cal_prompts_path = "data/rtp_500/split_1_0.5_0.1_0.2_0.2/cal.pkl"  # calibration split
default_test_split_path = "data/rtp_500/split_1_0.5_0.1_0.2_0.2/test.pkl"  # base test split
default_test_prompts_path = "data/test_prompt_only.pkl"  # extracted test prompts
default_test_surv_time_path = "data/test_surv_times.npy"  # numpy array of survival times
default_model_path = "saved/Prop_rtp_500_ModernBERT/lightning_logs/version_3/checkpoints/epoch=0-step=99.ckpt"

# Locations of experiment results
# ``real_data_experiments.py`` appends to ``default_exp_results_path`` while
# ``real_data_uncalib_experiments.py`` writes to ``default_uncalib_results_path``.
default_exp_results_path = "results.csv"  # output file for calibrated experiments
default_uncalib_results_path = "results_uncalib.csv"  # output file for uncalibrated baseline


def get_hf_key(path: str | None = None) -> str:
    """Return the HuggingFace API key from ``path`` or ``hf_key_path``."""
    key_path = Path(path or hf_key_path)
    return utils.api_key_from_file(str(key_path))


def generator_params(
    *,
    model_name: str = default_model_name,
    hf_key: str | None = None,
    max_input_tokens: int = default_max_input_tokens,
    max_output_tokens: int = default_max_output_tokens,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Return parameters for ``VanillaGeneratorVLLM``.

    allows multiple scripts to share consistent defaults.
    """
    hub_token = hf_key if hf_key is not None else get_hf_key()
    return {
        "model_name": model_name,
        "hub_token": hub_token,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "torch_dtype": torch_dtype,
    }


def rater_params(model_type: str = "original", amp: bool = True) -> Dict[str, Any]:
    """Return parameters for ``DetoxifyRater``."""
    return {"model_type": model_type, "amp": amp}


###############################################################################
# Persistence utilities
###############################################################################


_config_fields = {
    "hf_key_path": hf_key_path,
    "default_model_name": default_model_name,
    "default_max_input_tokens": default_max_input_tokens,
    "default_max_output_tokens": default_max_output_tokens,
    "default_batch_size": default_batch_size,
    "default_max_attempts": default_max_attempts,
    "default_cal_prompts_path": default_cal_prompts_path,
    "default_test_split_path": default_test_split_path,
    "default_test_prompts_path": default_test_prompts_path,
    "default_test_surv_time_path": default_test_surv_time_path,
    "default_model_path": default_model_path,
    "default_exp_results_path": default_exp_results_path,
    "default_uncalib_results_path": default_uncalib_results_path,
}


def save_config(path: str) -> None:
    """Save current configuration values to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_config_fields, f, indent=2)


def load_config(path: str) -> dict:
    """Load configuration values from ``path`` and update module globals."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    globals().update(data)
    _config_fields.update(data)
    return data
