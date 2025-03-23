import numpy as np
from src import survival_runner
from src.loss import survival_loss
from src.survival_runner import (
    SurvivalRunner,
    SurvivalResult,
    default_toxicity_func,
    default_text_prep_func,
)

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
def resample_calibration_set(generation_backend, rating_backend, prior_quantile_est, C_probs, X, toxicity_func=default_toxicity_func, text_prep_func=default_text_prep_func, batch_size=1500):
    # Determine the number of samples per instance
    C = np.where(np.random.uniform(size=len(prior_quantile_est)) < C_probs, 
                                    prior_quantile_est, 0).astype(int)
    
    
    # Generate the calibration set
    survival_runner = SurvivalRunner(
        generator=generation_backend,
        rater=rating_backend,
        max_attempts=C,
        toxicity_func=toxicity_func,
        text_prep_func=text_prep_func,
        conserve_memory=True,
    )

    survival_results = survival_runner.generate(
        prompts=X,
        batch_size=batch_size,
    )

    T_tilde = np.array([r.num_attempts for r in survival_results])
 
    return T_tilde, C

# Define the conformalizing function
def conformalize(trainer, model, target_taus, canidate_taus, X, generation_backend, rating_backend, budget_per_sample, share_budget=False, min_sample_size=None, needed_prob=1, toxicity_func=default_toxicity_func, text_prep_func=default_text_prep_func, batch_size=1500):
    assert (np.sort(canidate_taus) == canidate_taus).all(), "Canidate taus must be sorted"
    
    # Compute the quantile estimators
    model.set_taus(canidate_taus)
    pred = trainer.predict(model, X)
    quantile_est = np.vstack([item['tau'].cpu().numpy() for item in pred])

    prior_quantile_est = quantile_est[:, -1]
    max_estimator = np.inf
    if share_budget:
        C_probs = get_probs(budget_per_sample, prior_quantile_est, needed_prob=needed_prob)
        budget_per_sample = (C_probs * prior_quantile_est)[C_probs < needed_prob].mean()
    else:
        C_probs = budget_per_sample/prior_quantile_est
        C_probs = np.minimum(C_probs, 1)
    if min_sample_size:
        max_estimator = int(budget_per_sample / min_sample_size)
        C_probs = np.maximum(C_probs, min_sample_size)
        quantile_est = np.minimum(quantile_est, max_estimator)
        prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)

    # Resample the calibration set
    T_tilde, C = resample_calibration_set(generation_backend, rating_backend, prior_quantile_est, C_probs, X, toxicity_func=toxicity_func, text_prep_func=text_prep_func, batch_size=batch_size)
    print(f"Total budget used per sample is {C.mean()}")

    # Compute the weights - 1/conditional_probability
    weights = 1 / C_probs
    weights = np.where(quantile_est <= C.reshape(-1, 1), weights.reshape(-1, 1), 0)

    T_tilde_miscoverage = np.where(T_tilde.reshape(-1, 1) < quantile_est, 1, 0)

    # Compute the estimated miscoverage for each quantile
    tau_hats = (weights * T_tilde_miscoverage).sum(axis=0) / weights.sum(axis=0)
    print(tau_hats)
    print(weights)
    print(C_probs)
    print(T_tilde_miscoverage)

    tau_diff = target_taus - tau_hats[:, np.newaxis]
    smallest_pos = np.where(tau_diff > 0, 1, -1. * np.inf).cumsum(axis=0).argmax(axis=0)
    a_hats = canidate_taus[smallest_pos]
    
    return a_hats, max_estimator