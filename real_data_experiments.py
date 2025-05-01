import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader  # Import DataLoader
from src.failure_model import ToxicClassifier
from src.datasets import PromptOnlyDataset
import pandas as pd
from src.conformal import conformalize
import time
from src import utils
import pandas as pd
import os
import sys
from huggingface_hub.hf_api import HfFolder


#############################################################################################################
# EXPERIMENT PARAMETERS
#############################################################################################################


# Get the Hugging Face key and save it
HF_KEY = utils.api_key_from_file("HF_KEY.txt")
HfFolder.save_token(HF_KEY)

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
TAUS_RANGE = torch.tensor(np.logspace(-8, -0.33, 500))
target_tau_idx = torch.argmin(torch.abs(TAUS_RANGE - TARGET_TAUS))

# name, min_sample_size, share_budget, naive
EXPERIMENTS = [
    # ("Fixed Budgeting", None, False, True),
    # ("Adaptive Budgeting", None, False, False),
    ("Capped Adaptive Budgeting", 0.5, False, False),
    # ("Global Budgeting", 0.5, True, False),
]

# NUM_RUNS = 5
# BUDGET_RANGE = torch.logspace(start=1, end=3, steps=10, base=10).int().unique().tolist()

NUM_RUNS = 1
BUDGET_RANGE = [10, 30]

SAVE_PATH = "results.csv"


#############################################################################################################
# UTILITY FUNCTIONS
#############################################################################################################


def validate_save_path(save_path):
    # make save_path absolute if it is not
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    # Check if the directory exists
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        print(f"Results directory {directory} does not exist. Creating it.")
        os.makedirs(directory)
        print(f"Directory {directory} created.")

    # Check if the file already exists
    if os.path.exists(save_path):
        # Ask the user if they want to overwrite
        overwrite = input(f"Warning: file '{save_path}' already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != "y":
            print("Exiting without overwriting.")
            sys.exit(1)
        else:
            print(f"Continuing, overwriting file '{save_path}'.")


def save_results(save_path, df):
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")


def load_results(save_path):
    df = pd.read_csv(save_path, index_col=None)
    print(f"Results loaded from {save_path}")
    return df


def print_config():
    print("Configuration:")
    print(f" - TAUS_RANGE:       {TAUS_RANGE}")
    print(f" - target_taus:      {TARGET_TAUS}")
    print(f" - RATER_PARAMS:     {RATER_PARAMS}")
    print(f" - GENERATOR_PARAMS: {GENERATOR_PARAMS}")
    print(f" - EXPERIMENTS:      {EXPERIMENTS}")
    print(f" - NUM_RUNS:         {NUM_RUNS}")
    print(f" - BUDGET_RANGE:     {BUDGET_RANGE}")
    print(f" - DS_CAL_PATH:      {DS_CAL_PATH}")
    print(f" - DS_TEST_PATH:     {DS_TEST_PATH}")
    print(f" - MODEL_PATH:       {MODEL_PATH}")
    print(f" - SAVE_PATH:        {SAVE_PATH}")
    print("-" * 100)


def print_result(result_dict):
    print("-" * 60)
    print("Experiment Results:")
    for key, value in result_dict.items():
        print(f" - {key.ljust(30)}: {value}")
    print("-" * 60)


#############################################################################################################
# MAIN FUNCTION
#############################################################################################################


def run_experiments():
    print_config()
    validate_save_path(SAVE_PATH)

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

    # run

    # NOTE: dont enable multiple-gpus for inference, as it causes weird bugs
    trainer = pl.Trainer(enable_progress_bar=False, accelerator="gpu", devices=1)
    
    results_df = pd.DataFrame()

    for run_num in range(NUM_RUNS):

        for exp_type in EXPERIMENTS:

            name, min_sample_size, share_budget, naive = exp_type

            for budget in BUDGET_RANGE:

                print("-" * 60)
                print(f"Running {name} with budget {budget} (run {run_num + 1}/{NUM_RUNS})")
                print("-" * 60)

                start_time = time.time()

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
                    text_prep_func="sentence_completion",
                    multi_gpu=True,
                    plot=False,
                    return_extra=True,
                    batch_size=1500,
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

                time_delta = time.time() - start_time

                tau_hat_idx = np.argmin(torch.abs(TAUS_RANGE - tau_hat)).item()
                tau_hat_miscoverage = miscoverage[tau_hat_idx].item()
                tau_target_miscoverage = miscoverage[target_tau_idx].item()

                # compute total number of generated samples
                mean_generated_samples = T_tilde.mean().item()
                mean_c_value = C.mean().item()

                # compute LPB
                test_pred_raw = trainer.predict(model, dataloaders=dl_test)
                test_quantile_est = np.vstack([p["tau"].T for p in test_pred_raw])
                tau_hat_lpb = test_quantile_est[:, tau_hat_idx].mean().item()
                tau_target_lpb = test_quantile_est[:, target_tau_idx].mean().item()

                # add results to dataframe
                result_dict = {
                    "exp_name": name,
                    "exp_min_sample_size": min_sample_size,
                    "exp_share_budget": share_budget,
                    "exp_naive": naive,
                    "budget": budget,
                    "run_num": run_num,
                    "tau_hat": tau_hat,
                    "max_est": max_est,
                    "calib_tau_hat_miscoverage": tau_hat_miscoverage,
                    "calib_tau_target_miscoverage": tau_target_miscoverage,
                    "calib_mean_generated_samples": mean_generated_samples,
                    "calib_mean_c_value": mean_c_value,
                    "test_tau_hat_lpb": tau_hat_lpb,
                    "test_tau_target_lpb": tau_target_lpb,
                    "time_delta": time_delta,
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

    run_experiments()


if __name__ == "__main__":
    main()
