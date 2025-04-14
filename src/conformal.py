import numpy as np
from src import survival_runner
from src.loss import survival_loss
from src.survival_runner import (
    SurvivalRunner,
    SurvivalResult,
    default_toxicity_func,
    default_text_prep_func,
    run_survival_sampling_generic
)
from scipy.optimize import bisect
import torch
from torch.utils.data import DataLoader  # Import DataLoader

def get_probs(budget_per_sample, prior_quantile_est, needed_prob=1):
    C_probs = budget_per_sample/prior_quantile_est
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
    toxicity_func=default_toxicity_func,
    text_prep_func=default_text_prep_func,
    batch_size=1500,
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
    C = np.where(np.random.uniform(size=len(prior_quantile_est)) < C_probs, 
                 prior_quantile_est, 0).astype(int).reshape(-1, 1)

    # Generate the calibration set using the generic survival runner.
    # Note: We set max_attempts high (here int(10e6)) and pass C as per–prompt attempt limits.
    survival_results = run_survival_sampling_generic(
        prompts=X,
        prompt_attempts=C,
        generate_params={"batch_size": batch_size},
        generator_params=generator_params,
        rater_params=rater_params,
        max_attempts=int(10e6),
        toxicity_func=toxicity_func,
        text_prep_func=text_prep_func,
        conserve_memory=True,
    )

    T_tilde = np.array([r.num_attempts for r in survival_results]).reshape(-1, 1)
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
    needed_prob=1,
    toxicity_func=default_toxicity_func,
    text_prep_func=default_text_prep_func,
    batch_size=1500,
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
        needed_prob: The probability threshold required.
        toxicity_func: Function to determine toxicity.
        text_prep_func: Function to prepare text for rating.
        batch_size: Batch size for the generation process.
    
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
    print(quantile_est.shape)

    max_estimator = np.inf
    if share_budget:
        min_sample_size = min_sample_size if min_sample_size else 0
        max_estimator = int(budget_per_sample / min_sample_size) if min_sample_size else np.inf
        quantile_est = np.minimum(quantile_est, max_estimator)
        prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)
        C_probs, _ = solve_optimization(prior_quantile_est, budget_per_sample * len(prior_quantile_est), tol=1e-8)
    else:
        C_probs = budget_per_sample / prior_quantile_est
        C_probs = np.minimum(C_probs, 1)
        if min_sample_size:
            max_estimator = int(budget_per_sample / min_sample_size)
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
    )

    # Compute the weights – 1/conditional_probability.
    weights = 1 / C_probs.reshape(-1, 1)
    weights = np.where(quantile_est <= C, weights, 0)
    print(weights)

    T_tilde_miscoverage = np.where(T_tilde < quantile_est, 1, 0)
    print(T_tilde_miscoverage)

    # Compute the estimated miscoverage for each quantile.
    tau_hats = (weights * T_tilde_miscoverage).sum(axis=0) / weights.sum(axis=0)
    print(tau_hats)

    tau_diff = target_taus - tau_hats[:, np.newaxis]
    smallest_pos = np.where(tau_diff > 0, 1, -1.0 * np.inf).cumsum(axis=0).argmax(axis=0)
    tau_hat = canidate_taus[smallest_pos]
    q_hats = quantile_est[:, smallest_pos]

    return tau_hat, max_estimator, q_hats