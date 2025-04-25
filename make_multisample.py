from src import utils
from src.rating.base import *
from src.generation.base import *
import torch

import os
import pickle

utils.set_seed(42)
utils.clear_memory()

# %%
# load data

import pandas as pd

df = pd.read_json("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl", lines=True)

input_dicts = df["prompt"].values.flatten().tolist()
inputs = [d["text"] for d in input_dicts]

# %%
hf_key = utils.api_key_from_file("HF_KEY.txt")

# %%
rater_params = {
    "model_type": "original",
    "amp": True,
}

# %%
generator_params = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "hub_token": hf_key,
    "max_input_tokens": 40,
    "max_output_tokens": 30,
    "torch_dtype": torch.bfloat16,
}

# %%
# create survival analysis runner

from src.survival_runner import (
    SurvivalResult,
    generate_survival_results_generic,
)

batch_size = 1500
max_attempts = 5000
checkpoint_file = "multisample_checkpoint.pkl"

survival_results = generate_survival_results_generic(
    prompts=inputs,
    prompt_attempts=None,
    generate_params={"batch_size": batch_size, "checkpoint_file": checkpoint_file},
    generator_params=generator_params,
    rater_params=rater_params,
    max_attempts=max_attempts,
    toxicity_func="no_toxicity",
    text_prep_func="sentence_completion",
    conserve_memory_ratings=True,
    multi_gpu=True,
)

# %%
# run survival analysis and collect results

survival_list: list[SurvivalResult] = [res for res in survival_results]

# %%
# Save final results to disk

with open("multisample_results.pkl", "wb") as f:
    pickle.dump(survival_list, f)