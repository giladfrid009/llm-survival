"""Evaluate the uncalibrated baseline model on the test prompts."""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from huggingface_hub.hf_api import HfFolder

from src.datasets import PromptOnlyDataset
from src.failure_model import ToxicClassifier
import config
from src import utils


def parse_args() -> argparse.Namespace:
    """Arguments for evaluating the uncalibrated baseline."""
    parser = argparse.ArgumentParser(description="Run uncalibrated real data experiments")
    parser.add_argument("--test_prompts_path", default=config.default_test_prompts_path, help="Pickle containing test prompts")
    parser.add_argument("--test_surv_time_path", default=config.default_test_surv_time_path, help="Numpy file with survival times")
    parser.add_argument("--model_path", default=config.default_model_path, help="Path to trained toxicity model")
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="Batch size for predictions")
    parser.add_argument("--target_tau", type=float, default=0.1, help="Target miscoverage level")
    parser.add_argument("--output", default=config.default_uncalib_results_path, help="CSV file to write results")
    parser.add_argument("--hf_key_path", default=config.hf_key_path, help="Path to HuggingFace API key")
    
    # print all args
    parsed = parser.parse_args()
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed


def main() -> None:
    """Run the uncalibrated baseline evaluation and write ``args.output``."""
    args = parse_args()
    HfFolder.save_token(config.get_hf_key(args.hf_key_path))
    torch.multiprocessing.set_start_method("spawn", force=True)

    ds_test = PromptOnlyDataset(args.test_prompts_path)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False)
    test_t_tilde = np.load(args.test_surv_time_path)
    print(f"Loaded {len(ds_test)} test samples.")

    model = ToxicClassifier.load_from_checkpoint(args.model_path)
    _ = model.eval()
    taus_range = torch.tensor([args.target_tau])
    target_idx = 0
    model.set_taus(taus_range)
    model.set_min_p_for_q_tau(1e-20)

    trainer = pl.Trainer(enable_progress_bar=False, accelerator="gpu", devices=1)
    pred_raw = trainer.predict(model, dataloaders=dl_test)
    test_quantile_est = np.vstack([p["tau"].T for p in pred_raw])
    tau_hat_pred = test_quantile_est[:, target_idx].astype(np.int64)

    test_mean_lpb = tau_hat_pred.mean().item()
    test_mean_covered_lpb = np.mean(tau_hat_pred[test_t_tilde >= tau_hat_pred])
    test_miscoverage_upperbound = np.mean(test_t_tilde < tau_hat_pred)
    test_miscoverage_lowerbound = np.mean(test_t_tilde < np.clip(tau_hat_pred, min=1, max=test_t_tilde.max()))

    result_dict = {
        "exp_name": "Uncalibrated",
        "exp_min_sample_size": np.nan,
        "exp_share_budget": False,
        "exp_naive": False,
        "exp_budget": np.nan,
        "exp_run_num": 0,
        "tau_hat": args.target_tau,
        "max_est": np.inf,
        "cal_hours": np.nan,
        "cal_mean_generated_samples": np.nan,
        "cal_mean_c_value": np.nan,
        "cal_miscoverage": np.nan,
        "test_miscoverage_upperbound": test_miscoverage_upperbound,
        "test_miscoverage_lowerbound": test_miscoverage_lowerbound,
        "test_mean_lpb": test_mean_lpb,
        "test_mean_covered_lpb": test_mean_covered_lpb,
    }
    df = pd.DataFrame([result_dict])
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
