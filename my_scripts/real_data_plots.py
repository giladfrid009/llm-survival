# NOTE: works

"""Generate publication-quality plots from experiment CSV results."""

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from my_scripts import config
from src import utils

def parse_args() -> argparse.Namespace:
    """Return CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate real data plots")
    parser.add_argument("--results", default=config.default_exp_results_path, help="CSV file produced by real_data_experiments.py")
    parser.add_argument(
        "--results_uncalib", default=config.default_uncalib_results_path, help="CSV file from real_data_uncalib_experiments.py"
    )
    parser.add_argument("--output", default="figures/real_data_results.png", help="Path for the output plot image")
    
    parsed = parser.parse_args()
    
    # make all paths absolute
    parsed.results = utils.abs_path(parsed.results)
    parsed.results_uncalib = utils.abs_path(parsed.results_uncalib)
    parsed.output = utils.abs_path(parsed.output)
    
    # print all args
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed
    

def main() -> None:
    """Read CSVs and produce ``args.output``."""
    args = parse_args()
    results = pd.read_csv(args.results)
    results_uncalib = pd.read_csv(args.results_uncalib)

    sns.set(style="whitegrid", font_scale=2.5)
    results["test_coverage"] = 1 - results["test_miscoverage"]
    results["test_coverage_lowerbound"] = 1 - results["test_miscoverage_lowerbound"]
    results["test_coverage_upperbound"] = 1 - results["test_miscoverage_upperbound"]
    results["exp_name"] = results["exp_name"].replace({"Fixed Budgeting": "Naive", "Global Budgeting": "Optimized"})

    colors = sns.color_palette("tab10", 5)
    plt.figure(figsize=(25, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel("Average budget per prompt")
    plt.ylabel("Coverage")
    sns.lineplot(data=results, x="exp_budget", y=0.9, color="gray", linestyle="--", label="Ideal Coverage")
    uncalib_cov_upper = 1 - results_uncalib["test_miscoverage_lowerbound"].mean()
    uncalib_cov_lower = 1 - results_uncalib["test_miscoverage_upperbound"].mean()
    sns.lineplot(data=results, x="exp_budget", y=uncalib_cov_upper, color="black", linestyle=":")
    sns.lineplot(data=results, x="exp_budget", y=uncalib_cov_lower, color="black", linestyle=":")
    plt.fill_between(results["exp_budget"], uncalib_cov_upper, uncalib_cov_lower, color="gray", alpha=0.5)
    sns.lineplot(data=results, x="exp_budget", y="test_coverage_upperbound", marker="o", color=colors[0], linestyle=":")
    sns.lineplot(data=results, x="exp_budget", y="test_coverage_lowerbound", marker="o", color=colors[0], linestyle=":")
    sns.lineplot(data=results, x="exp_budget", y="test_coverage", marker="o", color=colors[3])
    plt.legend().remove()

    plt.subplot(1, 2, 2)
    plt.xlabel("Average budget per prompt")
    plt.ylabel("Average LPB")
    palette = {"Naive": colors[0], "Optimized": colors[3]}
    sns.lineplot(data=results, x="exp_budget", y="test_mean_lpb", marker="o", hue="exp_name", palette=palette)
    plt.hlines(
        y=results_uncalib["test_mean_lpb"].mean(),
        xmin=results["exp_budget"].min(),
        xmax=results["exp_budget"].max(),
        color="black",
        label="Uncalibrated",
    )
    plt.legend()

    plt.gcf().set_dpi(300)
    plt.savefig(args.output)


if __name__ == "__main__":
    main()
