import numpy as np
from src import survival_runner
from src.loss import survival_loss
from src.survival_runner import (
    SurvivalRunner,
    SurvivalResult,
    default_toxicity_func,
    default_text_prep_func,
    generate_survival_results_generic,
)
from scipy.optimize import bisect
import torch
from torch.utils.data import DataLoader  # Import DataLoader
import math
import matplotlib.pyplot as plt

def geom_cdf_stable(p: float, k: np.ndarray):
    """
    Numerically stable CDF of Geometric(p) at k (support 1,2,3,...).
    """
    # computes 1 - (1-p)^k accuratly
    cdf = -np.expm1(k * np.log1p(-p))
    return cdf.clip(0, 1)


def get_probs(budget_per_sample, prior_quantile_est, needed_prob=1):
    C_probs = budget_per_sample / prior_quantile_est
    above = C_probs > needed_prob
    below = C_probs < needed_prob
    while above.any() and below.any():
        leftover = ((C_probs[above] - needed_prob) * prior_quantile_est[above]).sum()
        C_probs[above] = needed_prob
        below = C_probs < needed_prob
        above = C_probs > needed_prob
        # Distribute the leftover budget to the bellow one, proportionally to their prior quantile estimate
        leaftover_per_sample = leftover / below.sum()
        C_probs[below] += leaftover_per_sample / prior_quantile_est[below]
        below = C_probs < needed_prob
        above = C_probs > needed_prob
    C_probs = np.minimum(C_probs, 1)
    return C_probs


def sample_calibration_set(
    generator_params,
    rater_params,
    prior_quantile_est,
    C_probs,
    X,
    toxicity_func=None,
    text_prep_func="sentence_completion",
    batch_size=1500,
    multi_gpu=True,
):
    """
    Generate a calibration set using the generic survival runner.

    Instead of supplying already-created generation and rating backends,
    this function accepts parameter dictionaries from which the backends
    are constructed.

    Args:
        generator_params: Dict of parameters to configure the generation backend.
        rater_params: Dict of parameters to configure the rating backend.
        prior_quantile_est: Array of quantile estimates.
        C_probs: Array of probabilities used to determine the number of samples.
        X: List or iterable of prompt strings.
        toxicity_func: Function to decide whether a generated output is toxic.
        text_prep_func: Function to prepare the generated text for rating.
        batch_size: Batch size for the generation process.

    Returns:
        T_tilde: Array of number of attempts per prompt.
        C: Array of per-prompt maximum attempts (calibration counts).
    """
    # Determine the number of samples per instance.
    # Note: C is computed as a (n, 1) array. The generic runner accepts per-prompt max attempts.
    C = np.where(np.random.uniform(size=len(prior_quantile_est)) < C_probs, prior_quantile_est, 0).astype(int).reshape(-1, 1)
    
    # Generate the calibration set using the generic survival runner.
    # Note: We set max_attempts high (here int(10e6)) and pass C as per–prompt attempt limits.
    survival_results = generate_survival_results_generic(
        prompts=X,
        prompt_ids=list(range(C.size)),
        prompt_attempts=C,
        generate_params={"batch_size": batch_size},
        generator_params=generator_params,
        rater_params=rater_params,
        max_attempts=int(10e6),
        toxicity_func=toxicity_func,
        text_prep_func=text_prep_func,
        conserve_memory=True,
        multi_gpu=multi_gpu,
    )
    
    T_tilde = np.array([r.num_attempts for r in survival_results]).reshape(-1, 1)
    ids = np.array([r.id for r in survival_results])

    # Sort the results by the original order of X.
    sorted_indices = np.argsort(ids)
    T_tilde = T_tilde[sorted_indices]
    # C = C[sorted_indices]    
    return T_tilde, C


def constraint_violation(lambda_val, w, b):
    """
    Computes the difference between the left-hand side of the constraint
    sum(w_i * p_i) and b, where p_i = min{1, 1/sqrt(lambda*w_i)}.
    """
    p = np.minimum(1, 1 / np.sqrt(lambda_val * w))
    return np.sum(w * p) - b


def solve_optimization(w, b, tol=1e-8):
    """
    Solves the optimization problem:
         min   sum(1/p_i)
         s.t.  sum(w_i * p_i) = b,   0 < p_i <= 1,
    by finding the Lagrange multiplier lambda such that the constraint holds.

    Parameters:
        w   : array-like, weights (all positive)
        b   : positive scalar, right-hand side of the constraint
        tol : tolerance for the bisection algorithm

    Returns:
        p   : optimal vector p*, where each p_i = min{1, 1/sqrt(lambda*w_i)}
        lambda_star : the Lagrange multiplier found
    """
    w = np.array(w, dtype=float)

    # Check feasibility: b must be no more than sum(w) since p_i <= 1.
    if b > np.sum(w):
        return np.ones_like(w), None

    # Set a lower bound for lambda.
    lambda_low = 1e-12
    # Set an initial upper bound: choose lambda_high so that for every i, 1/sqrt(lambda_high*w_i) < 1.
    # A sufficient condition is lambda_high > max(1/w).
    lambda_high = max(1 / w) * 10.0

    # Ensure that our bounds bracket a zero of the function.
    # When lambda is very small, p_i = 1 for each i, so the constraint is sum(w) - b (which is >= 0).
    # We need constraint_violation(lambda_low, w, b) >= 0 and constraint_violation(lambda_high, w, b) <= 0.
    f_low = constraint_violation(lambda_low, w, b)
    f_high = constraint_violation(lambda_high, w, b)

    # Increase lambda_high until f_high is negative.
    while f_high > 0:
        lambda_high *= 2
        f_high = constraint_violation(lambda_high, w, b)

    # Use bisection to find lambda such that the constraint is met.
    lambda_star = bisect(constraint_violation, lambda_low, lambda_high, args=(w, b), xtol=tol)

    # With lambda_star in hand, compute the optimal p.
    p_opt = np.minimum(1, 1 / np.sqrt(lambda_star * w))
    return p_opt, lambda_star


def conformalize(
    trainer,
    model,
    target_taus,
    canidate_taus,
    X,
    generator_params,
    rater_params,
    budget_per_sample,
    share_budget=False,
    min_sample_size=None,
    naive: bool = False,
    toxicity_func=None,
    text_prep_func="sentence_completion",
    batch_size=1500,
    multi_gpu=True,
    plot: bool = False,
    return_extra: bool = False,
):
    """
    Run the full conformalization procedure.

    This version uses parameter dictionaries (generator_params and rater_params)
    to build the survival runner via the generic code.

    Args:
        trainer: A trainer instance with a predict() method.
        model: The model to be used for prediction.
        target_taus: Array of target quantiles.
        canidate_taus: Array of candidate quantile values.
        X: The dataset (or prompts) for which to perform conformalization.
        generator_params: Dict for configuring the generation backend.
        rater_params: Dict for configuring the rating backend.
        budget_per_sample: The per-sample budget for the procedure.
        share_budget: Boolean flag for sharing budget.
        min_sample_size: Minimum sample size (if applicable).
        toxicity_func: Function to determine toxicity.
        text_prep_func: Function to prepare text for rating.
        batch_size: Batch size for the generation process.
        plot: Boolean flag for plotting results.
        return_extra: Boolean flag for returning extra results.

    Returns:
        tau_hat: The chosen quantile threshold.
        max_estimator: The maximum quantile estimator used.
        q_hats: The per-sample quantile estimates.
    """
    # Compute the quantile estimators.
    predict_dataloader = DataLoader(X, batch_size=1500, shuffle=False)
    pred_raw = trainer.predict(model, dataloaders=predict_dataloader)
    quantile_est = np.vstack([p["tau"].T for p in pred_raw])
    prior_quantile_est = quantile_est[:, -1]

    max_estimator = np.inf

    if naive:
        p_C = 1 / budget_per_sample
        C_probs = 1 - geom_cdf_stable(p=p_C, k=prior_quantile_est)
    elif share_budget:
        min_sample_size = min_sample_size if min_sample_size else 0
        max_estimator = min(max_estimator, int(budget_per_sample / min_sample_size) if min_sample_size else np.inf)
        quantile_est = np.minimum(quantile_est, max_estimator)
        prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)
        C_probs, _ = solve_optimization(prior_quantile_est, budget_per_sample * len(prior_quantile_est), tol=1e-8)
    else:
        C_probs = budget_per_sample / prior_quantile_est
        C_probs = np.minimum(C_probs, 1)
        if min_sample_size:
            max_estimator = min(max_estimator, int(budget_per_sample / min_sample_size))
            C_probs = np.maximum(C_probs, min_sample_size)
            quantile_est = np.minimum(quantile_est, max_estimator)
            prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)

    # Resample the calibration set using the generic survival runner.
    T_tilde, C = sample_calibration_set(
        generator_params,
        rater_params,
        prior_quantile_est,
        C_probs,
        X,
        toxicity_func=toxicity_func,
        text_prep_func=text_prep_func,
        batch_size=batch_size,
        multi_gpu=multi_gpu,
    )
        
    # Compute the weights – 1/conditional_probability.
    weights = 1 / C_probs.reshape(-1, 1)
    weights = np.where(quantile_est <= C, weights, 0)
    
    # Compute the estimated miscoverage for each quantile.
    miscoverage_unweighted = np.where(T_tilde < quantile_est, 1, 0)
    miscoverage = (weights * miscoverage_unweighted).mean(axis=0)

    tau_diff = target_taus - miscoverage[:, np.newaxis]
    smallest_pos = np.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(axis=0).argmax(axis=0)
    tau_hat = canidate_taus[smallest_pos]
    q_hats = quantile_est[:, smallest_pos]

    if plot:
        
        for (tau, miscov) in zip(canidate_taus, miscoverage):
            print(f"tau: {tau}, miscoverage: {miscov:.4f}")

        valid_mask = (C > 0).flatten()

        # plot weighted coverage vs quantile
        plt.plot(canidate_taus, miscoverage)
        plt.axhline(y=target_taus[0], color="r", linestyle="--", label="Target Miscoverage")
        plt.xlabel("Quantile")
        plt.ylabel("Miscoverage")
        plt.title("Estimated Miscoverage")
        plt.legend()
        plt.show()

        # plot T_tilde where C > 0 histogram
        # plot q_hats histogram where C > 0
        # plot vertical line of max_estimator if its not inf
        plt.hist(
            x=[
                T_tilde[valid_mask].flatten(),
                q_hats[valid_mask].flatten(),
            ],
            bins=50,
            label=["T_tilde", "q_hats"],
            align="right",
            histtype="stepfilled",
            alpha=0.5,
        )
        if max_estimator != np.inf:
            plt.axvline(x=max_estimator, color="r", linestyle="--", label="Max Estimator")
        plt.xlabel("Survival Time")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of T_tilde and q_hats \n (tau={canidate_taus[smallest_pos].item()})")
        plt.legend()
        plt.show()

        # find candidate tau that is closest to the target tau and plot
        # let its index be target_pos
        # plot T_tilde where C > 0 histogram
        # plot quantile_est[:, target_pos] histogram where C > 0
        # plot vertical line of max_estimator if its not inf
        target_pos = np.abs(canidate_taus - target_taus[0]).argmin()

        plt.hist(
            x=[
                T_tilde[valid_mask].flatten(),
                quantile_est[:, target_pos][valid_mask].flatten(),
            ],
            bins=50,
            label=["T_tilde", "q_target"],
            align="right",
            histtype="stepfilled",
            alpha=0.5,
        )
        if max_estimator != np.inf:
            plt.axvline(x=max_estimator, color="r", linestyle="--", label="Max Estimator")
        plt.xlabel("Survival Time")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of T_tilde and q_target \n (tau={canidate_taus[target_pos].item()})")
        plt.legend()
        plt.show()

    if return_extra:
        return (
            tau_hat.item(), # float
            max_estimator, # int
            q_hats.flatten(), # np.ndarray; shape (n_samples,)
            T_tilde.flatten(), # np.ndarray; shape (n_samples,)
            C.flatten(), # np.ndarray; shape (n_samples,)
            quantile_est, # np.ndarray; shape (n_samples, n_taus)
            prior_quantile_est, # np.ndarray; shape (n_samples,)
            C_probs, # np.ndarray; shape (n_samples,)
            weights, # np.ndarray; shape (n_samples, n_taus)
            miscoverage, # np.ndarray; shape (n_taus,)
        )

    return tau_hat.item(), max_estimator, q_hats.flatten()