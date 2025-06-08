# NOTE: works

"""Main calibrated real-data experiments used in the paper.

This script loads the split dataset, fine-tuned toxicity classifier and runs the
conformalized budgeting strategies described in the paper.  Results are appended
to ``results.csv`` by default and can later be visualized with
``real_data_plots.py``.
"""

import pytorch_lightning as pl
from huggingface_hub.hf_api import HfFolder
import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import argparse
import config

from src.failure_model import ToxicClassifier
from src.datasets import PromptOnlyDataset
from src.conformal import conformalize
from src import utils

import time
import os
import sys
import logging


import torch._dynamo

# NOTE: supress more errors, great
torch._dynamo.config.suppress_errors = True

#############################################################################################################
# EXPERIMENT PARAMETERS
#############################################################################################################


EXPERIMENT_OPTIONS = {
    "fixed": ("Fixed Budgeting", None, False, True),
    "adaptive": ("Adaptive Budgeting", None, False, False),
    "capped": ("Capped Adaptive Budgeting", 0.5, False, False),
    "global": ("Global Budgeting", 0.5, True, False),
}

# Generation and rating settings used across all experiments
TOXICITY_FUNC = "no_toxicity"
TEXT_PREP_FUNC = "sentence_completion"


def parse_args() -> argparse.Namespace:
    """CLI arguments for the real-data experiments."""
    parser = argparse.ArgumentParser(description="Run real data experiments")
    parser.add_argument("--cal_prompts_path", default=config.default_cal_prompts_path, help="Pickle file of calibration prompts")
    parser.add_argument("--test_prompts_path", default=config.default_test_prompts_path, help="Pickle file of test prompts")
    parser.add_argument("--test_surv_time_path", default=config.default_test_surv_time_path, help="Numpy file of survival times")
    parser.add_argument("--model_path", default=config.default_model_path, help="Path to trained toxicity model checkpoint")
    parser.add_argument("--save_path", default=config.default_exp_results_path, help="CSV file to append experiment results")
    parser.add_argument("--batch_size", type=int, default=1300, help="Batch size for model predictions")
    parser.add_argument("--budgets", type=int, nargs="*", default=[1200], help="Budgets (samples per prompt) to evaluate")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of independent experiment runs")
    parser.add_argument("--target_tau", type=float, default=0.1, help="Target miscoverage level")
    parser.add_argument("--min_tau_exp", type=float, default=-3, help="log10 of smallest tau in search grid")
    parser.add_argument("--max_tau_exp", type=float, default=-0.25, help="log10 of largest tau in search grid")
    parser.add_argument("--num_taus", type=int, default=1000, help="Number of tau values in the grid")
    parser.add_argument("--hf_key_path", default=config.hf_key_path, help="Path to HF API key")
    parser.add_argument("--model_name", default=config.default_model_name, help="Name of the generator model")
    parser.add_argument(
        "--max_input_tokens", type=int, default=config.default_max_input_tokens, help="Maximum prompt tokens for generation"
    )
    parser.add_argument("--max_output_tokens", type=int, default=config.default_max_output_tokens, help="Maximum tokens to generate")
    parser.add_argument(
        "--experiments", default="all", help="Comma-separated list of experiment types to run: fixed, adaptive, capped, global, or 'all'"
    )
    
    parsed = parser.parse_args()
    
    # make all paths absolute
    parsed.cal_prompts_path = utils.abs_path(parsed.cal_prompts_path)
    parsed.test_prompts_path = utils.abs_path(parsed.test_prompts_path)
    parsed.test_surv_time_path = utils.abs_path(parsed.test_surv_time_path)
    parsed.model_path = utils.abs_path(parsed.model_path)
    parsed.save_path = utils.abs_path(parsed.save_path)
    parsed.hf_key_path = utils.abs_path(parsed.hf_key_path)
    
    # print all args
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed


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


def print_config(args: argparse.Namespace) -> None:
    print(f" - Experiment:")
    print(f"   - TARGET_TAU:        {args.target_tau}")
    print(f"   - TEST TAUS:         logspace({args.min_tau_exp}, {args.max_tau_exp}, {args.num_taus})")
    print(f"   - EXPERIMENTS:       {args.experiments}")
    print(f"   - NUM_RUNS:          {args.num_runs}")
    print(f"   - BUDGET_RANGE:      {args.budgets}")

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


def run_experiments(args: argparse.Namespace) -> None:
    """Perform all experiment runs and append results to ``args.save_path``."""

    print_config(args)

    sel = [s.strip().lower() for s in args.experiments.split(",") if s.strip()]
    if "all" in sel:
        experiments = list(EXPERIMENT_OPTIONS.values())
    else:
        experiments = [EXPERIMENT_OPTIONS[s] for s in sel]

    results_df = validate_save_path(args.save_path)
    if results_df is None:
        results_df = pd.DataFrame()

    # load data
    ds_cal = PromptOnlyDataset(args.cal_prompts_path)
    ds_test = PromptOnlyDataset(args.test_prompts_path)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)

    # load test set survival times
    test_t_tilde = np.load(args.test_surv_time_path)

    print(f"Loaded {len(ds_cal)} calibration samples and {len(ds_test)} test samples.")

    # load model
    model = ToxicClassifier.load_from_checkpoint(args.model_path)
    _ = model.eval()

    taus_range = torch.tensor(np.logspace(args.min_tau_exp, args.max_tau_exp, args.num_taus))
    target_taus = torch.tensor([args.target_tau])
    model.set_taus(taus_range)
    model.set_min_p_for_q_tau(1e-20)

    # NOTE: dont enable multiple-gpus for inference, as it causes weird bugs
    trainer = pl.Trainer(enable_progress_bar=False, accelerator="gpu", devices=1)

    for run_num in range(args.num_runs):

        for exp_type in experiments:

            name, min_sample_size, share_budget, naive = exp_type

            for budget in args.budgets:

                print("-" * 60)
                print(f"Running {name} with budget {budget} (run {run_num + 1}/{args.num_runs})")
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
                    print(f"Skipping {name} with budget {budget} (run {run_num + 1}/{args.num_runs}) - already done.")
                    continue

                cal_start_time = time.time()

                utils.clear_memory()

                # Call the conformalize function with the specified parameters.
                result_tuple = conformalize(
                    trainer=trainer,
                    model=model,
                    target_taus=target_taus,
                    canidate_taus=taus_range,
                    X=ds_cal,
                    generator_params=config.generator_params(
                        model_name=args.model_name,
                        hf_key=config.get_hf_key(args.hf_key_path),
                        max_input_tokens=args.max_input_tokens,
                        max_output_tokens=args.max_output_tokens,
                    ),
                    rater_params=config.rater_params(),
                    budget_per_sample=budget,
                    share_budget=share_budget,
                    min_sample_size=min_sample_size,
                    naive=naive,
                    toxicity_func=TOXICITY_FUNC,
                    text_prep_func=TEXT_PREP_FUNC,
                    multi_gpu=torch.cuda.device_count() > 1,
                    plot=False,
                    return_extra=True,
                    batch_size=args.batch_size,
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
                tau_hat_idx = np.argmin(torch.abs(taus_range - tau_hat)).item()
                cal_miscoverage = miscoverage[tau_hat_idx].item()

                # compute total number of generated samples
                cal_mean_generated_samples = T_tilde.mean().item()
                cal_mean_c_value = C.mean().item()

                # compute LPB on test set
                test_pred_raw = trainer.predict(model, dataloaders=dl_test)
                test_quantile_est = np.vstack([p["tau"].T for p in test_pred_raw]).clip(min=1, max=max_est)
                tau_hat_pred = test_quantile_est[:, tau_hat_idx].flatten().astype(np.int64)
                test_mean_lpb = tau_hat_pred.mean().item()

                # compute LPB only for predictions which are correct
                test_mean_covered_lpb = np.mean(tau_hat_pred[test_t_tilde >= tau_hat_pred])

                # compute miscoverage upper-bound on test set
                test_miscoverage_lowerbound = np.mean(test_t_tilde < np.clip(tau_hat_pred, min=1, max=test_t_tilde.max()))
                test_miscoverage_upperbound = np.mean(test_t_tilde < tau_hat_pred)

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
                    "test_miscoverage_lowerbound": test_miscoverage_lowerbound,
                    "test_miscoverage_upperbound": test_miscoverage_upperbound,
                    "test_mean_lpb": test_mean_lpb,
                    "test_mean_covered_lpb": test_mean_covered_lpb,
                }

                if test_miscoverage_lowerbound == test_miscoverage_upperbound:
                    result_dict["test_miscoverage"] = test_miscoverage_lowerbound
                else:
                    result_dict["test_miscoverage_lowerbound"] = test_miscoverage_lowerbound
                    result_dict["test_miscoverage_upperbound"] = test_miscoverage_upperbound

                print_result(result_dict)

                results_df = pd.concat([results_df, pd.DataFrame([result_dict])], ignore_index=True)

                save_results(args.save_path, results_df)


def main() -> None:
    """CLI entry point for the calibrated experiments."""
    args = parse_args()
    HfFolder.save_token(config.get_hf_key())
    torch.multiprocessing.set_start_method("spawn", force=True)
    utils.configure_logging(logging.WARNING)
    run_experiments(args)


if __name__ == "__main__":
    main()
