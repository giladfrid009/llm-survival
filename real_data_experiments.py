import pytorch_lightning as pl
from huggingface_hub.hf_api import HfFolder
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd

from src.failure_model import ToxicClassifier
from src.datasets import PromptOnlyDataset
from src.survival_runner import generate_survival_results_generic
from src.conformal import conformalize
from src import utils

import time
import os
import sys
import logging


#############################################################################################################
# EXPERIMENT PARAMETERS
#############################################################################################################


# Get the Hugging Face key and save it
HF_KEY = utils.api_key_from_file("HF_KEY.txt")

# experiment parameters

DS_CAL_PATH = "data/rtp_500/split_1_0.5_0.1_0.2_0.2/cal.pkl"
DS_TEST_PATH = "data/rtp_500/split_1_0.5_0.1_0.2_0.2/test.pkl"
MODEL_PATH = "saved/Jigsaw_BERT/lightning_logs/version_1/checkpoints/epoch=4-step=970.ckpt"

# Create the parameter dictionary for the rating backend.
RATER_PARAMS = {
    "model_type": "original",
    "amp": True,
}

# Create the parameter dictionary for the generation backend.
GENERATOR_PARAMS = {
    "model_name": "meta-llama/Llama-3.2-1B",
    "hub_token": HF_KEY,
    "max_input_tokens": 40,
    "max_output_tokens": 30,
    "torch_dtype": torch.bfloat16,
}

TARGET_TAUS = torch.tensor([0.1])
MIN_TAU_EXP = -8
MAX_TAU_EXP = -0.33
NUM_TAUS = 500
TAUS_RANGE = torch.tensor(np.logspace(MIN_TAU_EXP, MAX_TAU_EXP, NUM_TAUS))

TOXICITY_FUNC = None
TEXT_PREP_FUNC = "sentence_completion"
BATCH_SIZE = 1500

# name, min_sample_size, share_budget, naive
EXPERIMENTS = [
    # ("Fixed Budgeting", None, False, True),
    # ("Adaptive Budgeting", None, False, False),
    # ("Capped Adaptive Budgeting", 0.5, False, False),
    ("Global Budgeting", 0.5, True, False),
]

NUM_RUNS = 5
# BUDGET_RANGE = torch.logspace(start=1, end=3, steps=10, base=10).int().unique().tolist()

NUM_RUNS = 2
BUDGET_RANGE = [1, 2]

SAVE_PATH = "results_v2.csv"


#############################################################################################################
# UTILITY FUNCTIONS
#############################################################################################################


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

    # If the file already exists, ask what to do
    if os.path.exists(save_path):
        choice = input(
            f"Warning: results file '{save_path}' already exists.\n"
            "continue from last experiment (c), overwrite file (o), or exit (e)? [c/o/e]: "
        )
        choice = choice.strip().lower()
        if choice == "o":
            print(f"Overwriting '{save_path}'.")
            return None  # signal to start with empty DataFrame
        elif choice == "c":
            return load_results(save_path)
        else:
            print("Exiting without changes.")
            sys.exit(1)

    return None


def save_results(save_path, df):
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


def load_results(save_path):
    df = pd.read_csv(save_path, index_col=None)
    print(f"Results loaded from {save_path}")
    return df


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


def print_config():
    print("CONFIG:")

    print(f" - Paths:")
    print(f"   - DS_CAL_PATH:      {DS_CAL_PATH}")
    print(f"   - DS_TEST_PATH:     {DS_TEST_PATH}")
    print(f"   - MODEL_PATH:       {MODEL_PATH}")
    print(f"   - SAVE_PATH:        {SAVE_PATH}")

    print(f" - Parameters:")
    print(f"   - GENERATOR_PARAMS: {GENERATOR_PARAMS}")
    print(f"   - RATER_PARAMS:     {RATER_PARAMS}")
    print(f"   - BATCH_SIZE:       {BATCH_SIZE}")
    print(f"   - TOXICITY_FUNC:    {TOXICITY_FUNC}")
    print(f"   - TEXT_PREP_FUNC:   {TEXT_PREP_FUNC}")

    print(f" - Experiment:")
    print(f"   - TARGET_TAU:       {TARGET_TAUS}")
    print(f"   - TEST TAUS:        logspace({MIN_TAU_EXP}, {MAX_TAU_EXP}, {NUM_TAUS})")
    print(f"   - EXPERIMENTS:      {EXPERIMENTS}")
    print(f"   - NUM_RUNS:         {NUM_RUNS}")
    print(f"   - BUDGET_RANGE:     {BUDGET_RANGE}")

    print("-" * 100)


def print_result(result_dict):
    print("-" * 60)
    print("EXPERIMENT RESULTS:")
    for key, value in result_dict.items():
        print(f" - {key.ljust(30)}: {value}")
    print("-" * 60)


#############################################################################################################
# MAIN FUNCTION
#############################################################################################################


def run_experiments():

    print_config()

    results_df = validate_save_path(SAVE_PATH)
    if results_df is None:
        results_df = pd.DataFrame()

    # load data
    ds_cal = PromptOnlyDataset(DS_CAL_PATH)
    ds_test = PromptOnlyDataset(DS_TEST_PATH)
    dl_test = DataLoader(ds_test, batch_size=1500, shuffle=False)

    print(f"Loaded {len(ds_cal)} calibration samples and {len(ds_test)} test samples.")

    # load model
    model = ToxicClassifier.load_from_checkpoint(MODEL_PATH)
    _ = model.eval()

    model.set_taus(TAUS_RANGE)
    model.set_min_p_for_q_tau(1e-20)

    # NOTE: dont enable multiple-gpus for inference, as it causes weird bugs
    trainer = pl.Trainer(enable_progress_bar=False, accelerator="gpu", devices=1)

    for run_num in range(NUM_RUNS):

        for exp_type in EXPERIMENTS:

            name, min_sample_size, share_budget, naive = exp_type

            for budget in BUDGET_RANGE:

                print("-" * 60)
                print(f"Running {name} with budget {budget} (run {run_num + 1}/{NUM_RUNS})")
                print("-" * 60)

                # check if experiment already exists
                if (
                    not results_df.empty
                    and (
                        (results_df["exp_run_num"] == run_num)
                        & (results_df["exp_name"] == name)
                        & (results_df["exp_min_sample_size"] == min_sample_size)
                        & (results_df["exp_share_budget"] == share_budget)
                        & (results_df["exp_naive"] == naive)
                        & (results_df["exp_budget"] == budget)
                    ).any()
                ):
                    print(f"Skipping {name} with budget {budget} (run {run_num + 1}/{NUM_RUNS}) - already done.")
                    continue

                cal_start_time = time.time()

                utils.clear_memory()

                # Call the conformalize function with the specified parameters.
                result_tuple = conformalize(
                    trainer=trainer,
                    model=model,
                    target_taus=TARGET_TAUS,
                    canidate_taus=TAUS_RANGE,
                    X=ds_cal,
                    generator_params=GENERATOR_PARAMS,
                    rater_params=RATER_PARAMS,
                    budget_per_sample=budget,
                    share_budget=share_budget,
                    min_sample_size=min_sample_size,
                    naive=naive,
                    toxicity_func=TOXICITY_FUNC,
                    text_prep_func=TEXT_PREP_FUNC,
                    multi_gpu=torch.cuda.device_count() > 1,
                    plot=False,
                    return_extra=True,
                    batch_size=BATCH_SIZE,
                )

                (
                    tau_hat,  # chosen tau for the target miscoverage
                    max_est,  # maximum quantile prediction
                    q_hats,  # quantile predictions for the chosen tau
                    T_tilde,  # sampled survival time for all samples
                    C,  # censoring time
                    quantile_est,  # predicted quantile estimates for all taus
                    prior_quantile_est,  # each output is sampled at most prior_quantile_est times
                    C_probs,  # sampling probability of each sample
                    weights,  # weights used for the weighted miscoverage
                    miscoverage,  # miscoverage rate for each tau
                ) = result_tuple

                cal_hours = time.time() - cal_start_time
                cal_hours = round(cal_hours / (60 * 60), 3)  # convert to hours

                # compute empirical miscoverage on calibration set
                tau_hat_idx = np.argmin(torch.abs(TAUS_RANGE - tau_hat)).item()
                cal_miscoverage = miscoverage[tau_hat_idx].item()

                # compute total number of generated samples
                cal_mean_generated_samples = T_tilde.mean().item()
                cal_mean_c_value = C.mean().item()

                test_start_time = time.time()

                # compute LPB on test set
                test_pred_raw = trainer.predict(model, dataloaders=dl_test)
                test_quantile_est = np.vstack([p["tau"].T for p in test_pred_raw]).clip(min=1, max=max_est)
                tau_hat_pred = test_quantile_est[:, tau_hat_idx].flatten().astype(np.int64)
                test_mean_lpb = tau_hat_pred.mean().item()

                # Compute empirical miscoverage on test set
                survival_results = generate_survival_results_generic(
                    prompts=ds_test,
                    prompt_ids=list(range(len(ds_test))),
                    prompt_attempts=(tau_hat_pred - 1).clip(min=0).tolist(),
                    generate_params={"batch_size": BATCH_SIZE},
                    generator_params=GENERATOR_PARAMS,
                    rater_params=RATER_PARAMS,
                    max_attempts=int(10e6),
                    toxicity_func=TOXICITY_FUNC,
                    text_prep_func=TEXT_PREP_FUNC,
                    multi_gpu=torch.cuda.device_count() > 1,
                    conserve_memory=True,
                    conserve_memory_ratings=True,
                )

                is_miscovered = [(1 if res.is_toxic else 0) for res in survival_results]
                test_miscoverage = np.mean(is_miscovered)

                test_hours = time.time() - test_start_time
                test_hours = round(test_hours / (60 * 60), 3)  # convert to hours

                # add results to dataframe
                result_dict = {
                    "exp_name": name,
                    "exp_min_sample_size": min_sample_size,
                    "exp_share_budget": share_budget,
                    "exp_naive": naive,
                    "exp_budget": budget,
                    "exp_run_num": run_num,
                    "tau_hat": tau_hat,
                    "max_est": max_est,
                    "cal_hours": cal_hours,
                    "cal_mean_generated_samples": cal_mean_generated_samples,
                    "cal_mean_c_value": cal_mean_c_value,
                    "cal_miscoverage": cal_miscoverage,
                    "test_miscoverage": test_miscoverage,
                    "test_mean_lpb": test_mean_lpb,
                    "test_hours": test_hours,
                }

                print_result(result_dict)

                results_df = pd.concat([results_df, pd.DataFrame([result_dict])], ignore_index=True)

                save_results(SAVE_PATH, results_df)


def main():

    # NOTE: open a new terminal session after running this if you encounter HF token issues
    # in any of the workers or the main process
    HfFolder.save_token(HF_KEY)

    # NOTE: to stop pytorch breaking
    torch.multiprocessing.set_start_method("spawn", force=True)

    configure_logging(logging.ERROR)

    run_experiments()


if __name__ == "__main__":
    main()
