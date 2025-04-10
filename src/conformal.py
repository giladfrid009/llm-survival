import numpy as np
from src import survival_runner
from src.loss import survival_loss
from src.survival_runner import (
    SurvivalRunner,
    SurvivalResult,
    default_toxicity_func,
    default_text_prep_func,
)
from scipy.optimize import bisect

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


# Generate a new calibration set - each observation has a random number of samples:
def sample_calibration_set(generation_backend, rating_backend, prior_quantile_est, C_probs, X, toxicity_func=default_toxicity_func, text_prep_func=default_text_prep_func, batch_size=1500):
    # Determine the number of samples per instance
    C = np.where(np.random.uniform(size=len(prior_quantile_est)) < C_probs, 
                                    prior_quantile_est, 0).astype(int).reshape(-1, 1)
    
    
    # Generate the calibration set
    survival_runner = SurvivalRunner(
        generator=generation_backend,
        rater=rating_backend,
        max_attempts=10e6,
        toxicity_func=toxicity_func,
        text_prep_func=text_prep_func,
        conserve_memory=True,
    )

    survival_results = survival_runner.generate(
        prompts=X,
        batch_size=batch_size,
        max_attempts=C,
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

# Define the conformalizing function
def conformalize(trainer, model, target_taus, canidate_taus, X, generation_backend, rating_backend, budget_per_sample, share_budget=False, min_sample_size=None, needed_prob=1, toxicity_func=default_toxicity_func, text_prep_func=default_text_prep_func, batch_size=1500):
    # Compute the quantile estimators
    pred = trainer.predict(model, X)
    quantile_est = np.vstack([item['tau'].cpu().numpy() for item in pred])
    prior_quantile_est = quantile_est[:, -1]

    max_estimator = np.inf
    if share_budget:
        min_sample_size = min_sample_size if min_sample_size else 0
        max_estimator = int(budget_per_sample / min_sample_size) if min_sample_size else np.inf
        quantile_est = np.minimum(quantile_est, max_estimator)
        prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)
        C_probs, _ = solve_optimization(prior_quantile_est, budget_per_sample * len(prior_quantile_est), tol=1e-8)
    else:
        C_probs = budget_per_sample/prior_quantile_est
        C_probs = np.minimum(C_probs, 1)
        if min_sample_size:
            max_estimator = int(budget_per_sample / min_sample_size)
            C_probs = np.maximum(C_probs, min_sample_size)
            quantile_est = np.minimum(quantile_est, max_estimator)
            prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)

    # Resample the calibration set
    T_tilde, C = sample_calibration_set(generation_backend, rating_backend, prior_quantile_est, C_probs, X, toxicity_func=toxicity_func, text_prep_func=text_prep_func, batch_size=batch_size)

    # Compute the weights - 1/conditional_probability
    weights = 1 / C_probs.reshape(-1, 1)
    weights = np.where(quantile_est <= C, weights, 0)
    print(weights)

    T_tilde_miscoverage = np.where(T_tilde < quantile_est, 1, 0)
    print(T_tilde_miscoverage)

    # Compute the estimated miscoverage for each quantile
    tau_hats = (weights * T_tilde_miscoverage).sum(axis=0) / weights.sum(axis=0)
    print(tau_hats)

    tau_diff = target_taus - tau_hats[:, np.newaxis]
    smallest_pos = np.where(tau_diff > 0, 1, -1. * np.inf).cumsum(axis=0).argmax(axis=0)
    tau_hat = canidate_taus[smallest_pos]
    q_hats = quantile_est[:,smallest_pos]
    
    return tau_hat, max_estimator, q_hats

# Define the conformalizing function
# def conformalize(trainer, model, target_taus, canidate_taus, X, generation_backend, rating_backend, budget_per_sample, share_budget=False, min_sample_size=None, needed_prob=1, toxicity_func=default_toxicity_func, text_prep_func=default_text_prep_func, batch_size=1500):
#     assert (np.sort(canidate_taus) == canidate_taus).all(), "Canidate taus must be sorted"
    
#     # Compute the quantile estimators
#     model.set_taus(canidate_taus)
#     pred = trainer.predict(model, X)
#     quantile_est = np.vstack([item['tau'].cpu().numpy() for item in pred])

#     prior_quantile_est = quantile_est[:, -1]
#     max_estimator = np.inf
#     if share_budget:
#         C_probs = get_probs(budget_per_sample, prior_quantile_est, needed_prob=needed_prob)
#         under_needed_prob = C_probs < needed_prob
#         if under_needed_prob.any():
#             budget_per_sample = (C_probs * prior_quantile_est)[under_needed_prob].mean()
#     else:
#         C_probs = budget_per_sample/prior_quantile_est
#         C_probs = np.minimum(C_probs, 1)
#     if min_sample_size:
#         max_estimator = int(budget_per_sample / min_sample_size)
#         C_probs = np.maximum(C_probs, min_sample_size)
#         quantile_est = np.minimum(quantile_est, max_estimator)
#         prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)

#     # Resample the calibration set
#     T_tilde, C = sample_calibration_set(generation_backend, rating_backend, prior_quantile_est, C_probs, X, toxicity_func=toxicity_func, text_prep_func=text_prep_func, batch_size=batch_size)
#     print(f"Total budget used per sample is {C.mean()}")

#     # Compute the weights - 1/conditional_probability
#     weights = 1 / C_probs
#     weights = np.where(quantile_est <= C.reshape(-1, 1), weights.reshape(-1, 1), 0)

#     T_tilde_miscoverage = np.where(T_tilde.reshape(-1, 1) < quantile_est, 1, 0)

#     # Compute the estimated miscoverage for each quantile
#     tau_hats = (weights * T_tilde_miscoverage).sum(axis=0) / weights.sum(axis=0)
#     print(tau_hats)
#     print(weights)
#     print(C_probs)
#     print(T_tilde_miscoverage)

#     tau_diff = target_taus - tau_hats[:, np.newaxis]
#     smallest_pos = np.where(tau_diff > 0, 1, -1. * np.inf).cumsum(axis=0).argmax(axis=0)
#     a_hats = canidate_taus[smallest_pos]
    
#     return a_hats, max_estimator