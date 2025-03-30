# %%
from src import utils
from src.rating.base import *
from src.generation.base import *
import torch

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
from src.rating.detoxify import DetoxifyRater

rater_backend = DetoxifyRater(
    model_type="original",
    amp=True,
)

# %%
from src.generation.vanilla_model_vllm import VanillaGeneratorVLLM

# TODO: FIX ARGS
generator_backend = VanillaGeneratorVLLM(
    model_name="meta-llama/Llama-3.2-3B",
    hf_token=hf_key,
    max_output_tokens=30,
)

# %%
# create survival analysis runner

from functools import partial
from src.survival_runner import (
    SurvivalRunner,
    SurvivalResult,
    default_toxicity_func,
    default_text_prep_func,
)

batch_size = 300
max_attempts = 40

survival_runner = SurvivalRunner(
    generator=generator_backend,
    rater=rater_backend,
    max_attempts=max_attempts,
    toxicity_func=(lambda gen: False),
    text_prep_func=lambda gen: gen.prompt + gen.output,
    conserve_memory=False,
)

survival_results = survival_runner.generate(
    prompts=inputs,
    batch_size=batch_size,
)

# %%
# run survival analysis and print results

survival_list: list[SurvivalResult] = [res for res in survival_results]

# %%
# Save to disk the survival results
import pickle

with open("multisample_results.pkl", "wb") as f:
    pickle.dump(survival_list, f)


