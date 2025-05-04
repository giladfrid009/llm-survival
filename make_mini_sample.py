from src import utils
from src.rating.base import *
from src.generation.base import *
import torch
from huggingface_hub.hf_api import HfFolder
import numpy as np

import pickle

from src.survival_runner import SurvivalResult, generate_survival_results_generic

from src.datasets import PromptOnlyDataset

import os
import logging

utils.clear_memory()


###########
# CONFIG
###########


PROMPTS_PATH = "data/rtp_500/split_1_0.5_0.1_0.2_0.2/test.pkl"

HF_KEY = utils.api_key_from_file("HF_KEY.txt")

RATER_PARAMS = {
    "model_type": "original",
    "amp": True,
}

GENERATOR_PARAMS = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "hub_token": HF_KEY,
    "max_input_tokens": 40,
    "max_output_tokens": 30,
    "torch_dtype": torch.bfloat16,
}

BATCH_SIZE = 1500
MAX_ATTEMPTS = 100
TOTAL_DATASETS = 100
RESULTS_FILE = "mini_datasets/multisample_mini_{}.pkl"  # NOTE: make sure name has place for int formatting

# start index for saving the results file,
# i.e saving first dataset in RESULTS_FILE.format(START_IDX), second in RESULTS_FILE.format(START_IDX + 1), ...
START_IDX = 0


###########
# FUNCTIONS
###########


def configure_logging(level=logging.ERROR):
    logging.captureWarnings(True)
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)

    # set level for all loggers
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
            for h in logger.handlers:
                h.setLevel(level)

    os.environ["LOGLEVEL"] = logging.getLevelName(level)
    os.environ["VLLM_LOGGING_LEVEL"] = logging.getLevelName(level)
    logging.getLogger("lightning.pytorch").setLevel(level)
    torch._logging.set_logs(all=level)


def validate_save_path(save_path):

    # make save_path absolute if it is not
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    # Ensure directory exists
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        print(f"Results directory {directory} does not exist. Creating it.")
        os.makedirs(directory)
        print(f"Directory {directory} created.")

    i = START_IDX
    while os.path.exists(save_path.format(i)):
        print(f"Results file {save_path.format(i)} already exists. Incrementing index.")
        i += 1
    return i


def create_datasets():

    # Load the prompts
    prompts = PromptOnlyDataset(PROMPTS_PATH)
    print(f"Loaded {len(prompts)} prompts from {PROMPTS_PATH}")

    # validate the save path
    start = validate_save_path(RESULTS_FILE)
    start = max(start, START_IDX)

    # run
    for i in range(start, start + TOTAL_DATASETS):

        print("-" * 40)
        print(f"Running iter ({i + 1} / {TOTAL_DATASETS} )")
        print("-" * 40)

        utils.set_seed(i)
        utils.clear_memory()

        survival_results = generate_survival_results_generic(
            prompts=prompts,
            prompt_ids=list(range(len(prompts))),
            prompt_attempts=[MAX_ATTEMPTS] * len(prompts),
            generate_params={"batch_size": BATCH_SIZE},
            generator_params=GENERATOR_PARAMS,
            rater_params=RATER_PARAMS,
            max_attempts=MAX_ATTEMPTS,
            toxicity_func="no_toxicity",
            text_prep_func="sentence_completion",
            conserve_memory_ratings=False,
            conserve_memory=False,
            multi_gpu=torch.cuda.device_count() > 1,
        )

        # Sort the results by the original order of the prompts
        ids = np.array([r.id for r in survival_results])
        sorted_indices = np.argsort(ids).flatten()
        survival_results = [survival_results[i] for i in sorted_indices]

        # Save final results to disk
        with open(RESULTS_FILE.format(i), "wb") as f:
            pickle.dump(survival_results, f)


def main():

    # NOTE: open a new terminal session after running this if you encounter HF token issues
    # in any of the workers or the main process
    HfFolder.save_token(HF_KEY)

    # NOTE: to stop pytorch breaking
    torch.multiprocessing.set_start_method("spawn", force=True)

    configure_logging(logging.ERROR)

    create_datasets()


if __name__ == "__main__":
    main()
