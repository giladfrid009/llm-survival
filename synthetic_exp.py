# %%
"""
Imports and Global Settings
---------------------------
All required packages are imported and global settings are defined.
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import tqdm
import torch.nn.functional as F
from scipy.stats import geom
import argparse

parser = argparse.ArgumentParser(description="Synthetic experiment")
parser.add_argument("--n_x", type=int, default=100000)
args = parser.parse_args()

# Import custom loss functions (assumed to be correctly defined in src/loss.py)
from src.loss import survival_loss, prop_loss

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

"""
Data Generation
---------------
Generate synthetic data and split it into training, calibration, and testing sets.
"""

# Number of observations
n_x = args.n_x

# Determine probabilities p:
# Create two log-spaced arrays and concatenate them
p1 = np.logspace(-3, -2, (n_x * 9) // 10)
p2 = np.logspace(-5, -4, n_x - len(p1))
p = np.concatenate([p1, p2])

# Set covariates x:
input_size = 10
taus = np.linspace(0.1, 0.9, input_size)
# For each quantile level, compute the geometric quantile (raised to the 0.25 power) and stack as columns.
x = np.stack([stats.geom.ppf(tau, p) ** 0.25 for tau in taus], axis=1)
x /= x.mean()
x += np.random.normal(0, 0.1, x.shape)
n_x, input_size = x.shape  # Update n_x if needed

# For each observation, sample a random number of trials from a uniform distribution between 1 and 1000.
n_samples = 500 * np.ones(n_x, dtype=int)

# Generate binary outcomes for each observation:
# For each observation, perform binomial trials with probability p and sample size (n_samples)
y = [np.random.binomial(1, prob, n) for prob, n in zip(p, n_samples)]
# Sum the successes for each observation.
b = np.array([yi.sum() for yi in y])

# Compute the "first success" (t_tilde) for each observation using a geometric random variable;
# then cap it at the number of trials.
t_tilde = np.random.geometric(p, size=len(p))
t_tilde = np.minimum(t_tilde, n_samples)

# Event indicator: 1 if there is at least one success, else 0.
e = np.array([1 if yi.any() else 0 for yi in y])

(
    p_train,
    p_test,
    x_train,
    x_test,
    y_train,
    y_test,
    t_tilde_train,
    t_tilde_test,
    e_train,
    e_test,
    b_train,
    b_test,
    n_samples_train,
    n_samples_test,
) = train_test_split(p, x, y, t_tilde, e, b, n_samples, test_size=0.1, random_state=42)

(
    p_train,
    p_cal,
    x_train,
    x_cal,
    y_train,
    y_cal,
    t_tilde_train,
    t_tilde_cal,
    e_train,
    e_cal,
    b_train,
    b_cal,
    n_samples_train,
    n_samples_cal,
) = train_test_split(p_train, x_train, y_train, t_tilde_train, e_train, b_train, n_samples_train, test_size=0.5, random_state=42)

# For use in later plots, sort p_test and adjust corresponding predictions accordingly.
sort_idx = np.argsort(p_test)
p_test = p_test[sort_idx]

import numpy as np


def geom_cdf_stable(k, p):
    """
    Numerically stable CDF of Geometric(p) at k (support 1,2,3,...).
    """
    # log1p(-p) is ln(1-p) computed accurately even if p is tiny
    # expm1(...) is exp(...) - 1 computed accurately even if the exponent is near 0
    # Is still instable for large k, so clip the return value.
    return -np.expm1(k * np.log1p(-p))

"""
Neural Network Model and Training Function Definitions
--------------------------------------------------------
Define the model, loss functions, optimizer, dataset preparation, training loop, and prediction helper.
"""


# --- Loss Functions ---
def L1_loss(pred, target):
    """Compute L1 loss on sigmoid-transformed output."""
    return (torch.sigmoid(pred[:, 1]) - target).abs().mean()


def L2_loss(pred, target):
    """Compute L2 loss on sigmoid-transformed output."""
    return (torch.sigmoid(pred[:, 1]) - target).pow(2).mean()


# --- Model Definition ---
class Model(nn.Module):
    def __init__(self, hidden_dims):
        """
        Build a network with input layer size `input_size`, specified hidden layers, and a single output.
        """
        super().__init__()
        layer_dims = [input_size] + hidden_dims + [1]
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            # Use ReLU activation for all layers except the last.
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# --- Optimizer ---
def get_optimizer(model, optimizer_params):
    return optim.AdamW(model.parameters(), **optimizer_params)


# --- Helper to Prepare Dataset Tensors ---
def prepare_dataset(x_data, y_data, gt_p_data):
    """
    Converts numpy arrays (or tuples) into a TensorDataset.
    Ensures that y_data and gt_p_data have the correct shape.
    """
    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    y_arr = np.array(y_data)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    else:
        # If y_arr has shape (channels, n) and channels <= 2, transpose if needed.
        if y_arr.shape[0] in [1, 2] and y_arr.shape[1] == len(x_data):
            y_arr = y_arr.T
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)

    p_arr = np.array(gt_p_data)
    if p_arr.ndim == 1:
        p_arr = p_arr.reshape(-1, 1)
    p_tensor = torch.tensor(p_arr, dtype=torch.float32)

    return torch.utils.data.TensorDataset(x_tensor, y_tensor, p_tensor)


# --- Training Function ---
def train(loss_fn, model_params, optimizer_params, x_train, y_train, gt_p_train, x_val=None, y_val=None, gt_p_val=None, n_epochs=100):
    """
    Train a model using the given loss function.
    """
    model = Model(**model_params)
    optimizer = get_optimizer(model, optimizer_params)

    dataset_train = prepare_dataset(x_train, y_train, gt_p_train)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=100, shuffle=True)

    if x_val is not None:
        dataset_val = prepare_dataset(x_val, y_val, gt_p_val)
        dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=100, shuffle=False)

    def validate():
        total_loss, total_gt_ce, total_L1 = 0.0, 0.0, 0.0
        for batch in dataloader_val:
            x_batch, y_batch, p_batch = batch
            out = model(x_batch)
            # Combine outputs into two channels.
            loss = F.binary_cross_entropy_with_logits(out.reshape(-1), y_batch[:, 0].reshape(-1))
            gt_ce = torch.nn.functional.binary_cross_entropy_with_logits(out.reshape(-1), p_batch.reshape(-1))
            L1 = (torch.sigmoid(out.reshape(-1)) - p_batch.reshape(-1)).abs().mean()
            total_loss += loss.item()
            total_gt_ce += gt_ce.item()
            total_L1 += L1.item()
        n_batches = len(dataloader_val)
        return total_loss / n_batches, total_gt_ce / n_batches, total_L1 / n_batches

    # Training loop
    for epoch in range(n_epochs):
        total_loss, total_gt_ce, total_L1 = 0.0, 0.0, 0.0
        for batch in dataloader_train:
            x_batch, y_batch, p_batch = batch
            optimizer.zero_grad()
            out = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(out.reshape(-1), y_batch[:, 0].reshape(-1))
            with torch.no_grad():
                gt_ce = torch.nn.functional.binary_cross_entropy_with_logits(out.reshape(-1), p_batch.reshape(-1))
                L1 = (torch.sigmoid(out.reshape(-1)) - p_batch.reshape(-1)).abs().mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_gt_ce += gt_ce.item()
            total_L1 += L1.item()

        if x_val is not None and epoch % 50 == 0:
            val_loss, val_gt_ce, val_L1 = validate()
            print(
                f"Epoch {epoch} - Train Loss: {total_loss/len(dataloader_train):.4f} - "
                f"GT CE: {total_gt_ce/len(dataloader_train):.4f} - GT L1: {total_L1/len(dataloader_train):.4f} - "
                f"Val Loss: {val_loss:.4f} - Val GT CE: {val_gt_ce:.4f} - Val GT L1: {val_L1:.4f}"
            )
        elif x_val is None and epoch % 50 == 49:
            print(
                f"Epoch {epoch} - Train Loss: {total_loss/len(dataloader_train):.4f} - "
                f"GT CE: {total_gt_ce/len(dataloader_train):.4f} - GT L1: {total_L1/len(dataloader_train):.4f}"
            )

    return model


# --- Prediction Helper ---
def predict(model, input_data):
    """
    Runs the model on the provided input data and returns sigmoid-transformed predictions.
    """
    x_tensor = torch.tensor(input_data, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(x_tensor))
    return prediction.detach().numpy()


# --- Utility Function for Clipping Probabilities ---
def clip_p(p_vals):
    """Clip probability values so that they are never 0."""
    return np.maximum(1e-30, p_vals)


"""
Train the Proportional Model
----------------------------
Train only the proportional model; other models are commented out.
In this case, the target y is a tuple: the observed proportion and a tensor of ones.
"""

y_train_prop = (b_train / n_samples_train, np.ones(b_train.shape))
y_test_prop = (b_test / n_samples_test, np.ones(b_test.shape))

prop_model = train(
    loss_fn=prop_loss,
    model_params={"hidden_dims": [32, 32, 32, 32]},
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-5},
    x_train=x_train,
    y_train=y_train_prop,
    gt_p_train=p_train,
    x_val=x_test,
    y_val=y_test_prop,
    gt_p_val=p_test,
    n_epochs=10,
)

# Obtain predictions on the test data.
prop_pred = predict(prop_model, x_test)
prop_pred = prop_pred[sort_idx]  # Ensure predictions are sorted along with p_test.

"""
Quantile Estimation Function
-----------------------------
Given model predictions and quantile levels taus, estimate the corresponding quantile 
from a geometric distribution.
"""


def quantile_estimators(preds, taus):
    """
    For each tau value, return the corresponding quantile from a geometric distribution
    with probability estimates given by preds.
    """
    preds = clip_p(preds.flatten())
    return np.array([stats.geom.ppf(q, preds) for q in taus])


# Compute the 0.1 quantile for the proportional model predictions.
q_prop = quantile_estimators(prop_pred, [0.1])[0]

"""
Coverage and Utility Functions
------------------------------
Define functions for computing coverage, redistributing sample budgets,
resampling calibration sets, and solving an optimization problem.
"""


def coverage(quantile_preds, p_vals):
    """Computes the coverage probability for given quantile predictions and true p-values."""
    return 1 - stats.geom.cdf(quantile_preds, p=p_vals)


def get_probs(budget_per_sample, prior_quantile_est, needed_prob=1):
    """
    Compute an allocation probability for each sample and adjust any leftover budget.
    """
    C_probs = budget_per_sample / prior_quantile_est
    above = C_probs > needed_prob
    below = C_probs < needed_prob
    while above.any() and below.any():
        leftover = ((C_probs[above] - needed_prob) * prior_quantile_est[above]).sum()
        C_probs[above] = needed_prob
        below = C_probs < needed_prob
        above = C_probs > needed_prob
        leftover_per_sample = leftover / below.sum()
        C_probs[below] += leftover_per_sample / prior_quantile_est[below]
        below = C_probs < needed_prob
        above = C_probs > needed_prob
    return np.minimum(C_probs, 1)


def resample_calibration_set(p_cal, prior_quantile_est, C_probs):
    """
    Resample the calibration set:
    - Each observation obtains C samples based on its probability.
    - For each, generate a geometric random variable and cap it at C.
    """
    C = np.where(np.random.uniform(size=prior_quantile_est.shape) < C_probs, prior_quantile_est, 0).astype(int)
    T = np.random.geometric(p_cal, size=len(p_cal)).astype(int)
    T_tilde = np.minimum(T, C)
    return T_tilde, C


def constraint_violation(lambda_val, w, b_rhs):
    """
    Returns the difference between the weighted sum of probabilities and b_rhs.
    """
    p_vals = np.minimum(1, 1 / np.sqrt(lambda_val * w))
    return np.sum(w * p_vals) - b_rhs


def solve_optimization(w, b_rhs, tol=1e-8):
    """
    Solve for lambda in the constrained optimization problem to find optimal probabilities.
    """
    w = np.array(w, dtype=float)
    if b_rhs > np.sum(w):
        return np.ones_like(w), np.inf

    lambda_low = 1e-12
    lambda_high = max(1 / w) * 10.0
    f_low = constraint_violation(lambda_low, w, b_rhs)
    f_high = constraint_violation(lambda_high, w, b_rhs)

    while f_high > 0:
        lambda_high *= 2
        f_high = constraint_violation(lambda_high, w, b_rhs)

    from scipy.optimize import bisect

    lambda_star = bisect(constraint_violation, lambda_low, lambda_high, args=(w, b_rhs), xtol=tol)
    p_opt = np.minimum(1, 1 / np.sqrt(lambda_star * w))
    return p_opt, lambda_star


import scipy

"""
Conformalization Functions
----------------------------
Define a function that adjusts quantile estimates via conformalization.
The function now requires the calibration probabilities (p_cal) to be passed explicitly.
"""


def conformalize(
    preds,
    p_cal,
    target_taus,
    candidate_taus,
    C,
    T_tilde,
    budget_per_sample,
    share_budget=False,
    min_sample_size=None,
    needed_prob=1,
    naive=False,
):
    """
    Adjust quantile estimates to achieve target coverage.

    Parameters:
      preds            : Model predictions.
      p_cal            : Calibration set probabilities.
      target_taus      : The desired target coverage levels.
      candidate_taus   : Candidate quantile levels.
      C                : Number of samples per calibration instance.
      T_tilde         : Observed “first success” numbers for calibration.
      budget_per_sample: Budget allocated per sample.
      share_budget     : Whether the leftover budget should be shared.
      min_sample_size  : Minimum sample size if enforcing a floor.
      needed_prob      : Upper limit for allocated probability.

    Returns:
      a tuple containing:
       - a_hats: Adjusted quantile estimates.
       - weights_adjusted: Corresponding weights.
       - miscoverage indicator.
       - max_estimator (if applicable).
       - C_probs (calibration allocation probabilities).
    """
    # Use maximum candidate tau as the baseline.
    prior_tau = candidate_taus.max()
    quantile_est = quantile_estimators(preds, candidate_taus).astype(int)
    prior_quantile_est = quantile_estimators(preds, [prior_tau])[0].astype(int)

    max_estimator = np.inf
    if share_budget:
        if min_sample_size:
            max_estimator = int(budget_per_sample / min_sample_size)
            quantile_est = np.minimum(quantile_est, max_estimator)
            prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)
            C_probs, _ = solve_optimization(prior_quantile_est, budget_per_sample * len(p_cal), tol=1e-8)
        else:
            C_probs = get_probs(budget_per_sample, prior_quantile_est, needed_prob=needed_prob)
            budget_per_sample = (C_probs * prior_quantile_est)[C_probs < needed_prob].mean()
    else:
        C_probs = budget_per_sample / prior_quantile_est
        C_probs = np.minimum(C_probs, 1)
        if min_sample_size:
            max_estimator = int(budget_per_sample / min_sample_size)
            C_probs = np.maximum(C_probs, min_sample_size)
            quantile_est = np.minimum(quantile_est, max_estimator)
            prior_quantile_est = np.minimum(prior_quantile_est, max_estimator)

    if naive:
        # Use the naive approach. If C~Geom(p), both the indicator and probability of \hat{q} \leq C are the same as for \hat{q} \leq Ber(1-F(\hat{q})) * \hat{q}
        p_C = 1 / budget_per_sample
        C_probs = 1 - geom_cdf_stable(quantile_est, p=p_C)

    # Resample calibration set using the computed probabilities.
    T_tilde_resampled, C_resampled = resample_calibration_set(p_cal, quantile_est if naive else prior_quantile_est, C_probs)

    # Compute weights and miscoverage indicators.
    weights = 1 / np.clip(C_probs, 1e-30, 1 - 1e-30)
    weights_adjusted = np.where(quantile_est <= C_resampled, weights, 0)
    T_tilde_miscoverage = np.where(T_tilde_resampled < quantile_est, 1, 0)

    # Estimate miscoverage for each candidate quantile.
    tau_hats = (weights_adjusted * T_tilde_miscoverage).mean(axis=1)
    tau_diff = target_taus - tau_hats[:, np.newaxis]
    smallest_pos = np.where(tau_diff > 0, 1, -1 * np.inf).cumsum(axis=0).argmax(axis=0)
    a_hats = candidate_taus[smallest_pos]
    mean_n_samples = np.random.geometric(p_C, size=p_cal.shape).mean() if naive else C_resampled.mean()

    return a_hats, max_estimator, C_probs, mean_n_samples

"""
Calibration and Conformalization
----------------------------------
Predict on the calibration set and apply conformalization to adjust quantile estimates.
"""

# Define the target and candidate quantiles.
target_taus = np.array([0.1])
candidate_taus = np.logspace(-3, -0.75, 300)

# Predict probabilities on the calibration set for the proportional model.
prop_pred_cal = predict(prop_model, x_cal)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import seaborn as sns

def vary_budget_per_sample(
    budgets,
    p_cal,
    p_test,
    prop_pred_cal,
    prop_pred_test,
    n_samples_cal,
    t_tilde_cal,
    n_conformalizations=20,
    min_sample_size=0,
    share_budget=False,
    naive=False,
    resplit=False,
):
    """
    Vary the budget_per_sample and compute mean coverage and LPB for each budget.

    Parameters:
        budgets (iterable): The different values to use for budget_per_sample.
        p_cal: Calibration p-values or any calibration parameter required by conformalize.
        p_test: Test p-values or any test parameter required by coverage.
        prop_pred_cal: Calibrated propensity score predictions.
        prop_pred_test: Test propensity score predictions.
        n_samples_cal: Number of random samples for calibration.
        t_tilde_cal: A calibration threshold or related parameter.
        n_conformalizations (int): Number of conformalization runs per budget value.
        min_sample_size (float): Minimum sample size parameter to pass to conformalize.
        share_budget (bool): Whether to share the budget among the samples.
        naive (bool): Whether to use the naive method for conformalization.
        resplit (bool): Whether to resplit the calibration and test sets.

    Returns:
        coverage_df (pd.DataFrame): DataFrame with rows as budget values and columns as
            the conformalization iterations containing the mean coverage.
        LPB_df (pd.DataFrame): DataFrame with the corresponding LPB values.
    """
    # Initialize DataFrames with budgets as index and conformalization iteration indices as columns.
    coverage_df = pd.DataFrame(index=budgets, columns=range(n_conformalizations))
    LPB_df = pd.DataFrame(index=budgets, columns=range(n_conformalizations))
    mean_samples_df = pd.DataFrame(index=budgets, columns=range(n_conformalizations))

    # Resplit the calibration and test sets
    if resplit:
        # Merge the calibration and test sets.
        p_cal = np.concatenate([p_cal, p_test])
        prop_pred_cal = np.concatenate([prop_pred_cal, prop_pred_test])
        # Shuffle the merged set.
        sort_idx = np.random.permutation(len(p_cal))
        p_cal = p_cal[sort_idx]
        prop_pred_cal = prop_pred_cal[sort_idx]
        # Split the merged set back into calibration and test sets.
        p_test = p_cal[:len(p_test)]
        prop_pred_test = prop_pred_cal[:len(p_test)]
        p_cal = p_cal[len(p_test):]
        prop_pred_cal = prop_pred_cal[len(p_test):]

    # Loop over each budget value.
    for budget in budgets:
        for i in tqdm.tqdm(range(n_conformalizations), desc=f"Processing budget {budget}"):
            # Run the conformalization function, passing the additional parameters.
            c_prop, max_est, _, mean_n_samples = conformalize(
                prop_pred_cal,
                p_cal,
                target_taus,
                candidate_taus,
                n_samples_cal,
                t_tilde_cal,
                budget,
                min_sample_size=min_sample_size,
                share_budget=share_budget,
                naive=naive,
            )

            max_est = np.inf if max_est is None else max_est

            # Obtain the quantile estimator output.
            q_prop_conf = quantile_estimators(prop_pred, [c_prop[0]])[0].astype(int)
            q_prop_conf = np.minimum(q_prop_conf, max_est).astype(int)

            # Compute the mean coverage (using your coverage function and test set values).
            cov = coverage(q_prop_conf, p_vals=p_test).mean()
            # Compute the mean LPB as the mean value of the quantile estimator.
            lpb = q_prop_conf.mean()

            # Store the values in the DataFrames.
            coverage_df.loc[budget, i] = cov
            LPB_df.loc[budget, i] = lpb
            mean_samples_df.loc[budget, i] = mean_n_samples

    return coverage_df, LPB_df, mean_samples_df


# Define a range of budgets to test.
budgets = np.linspace(10, 10000, 10)

# Dictionary to store results for each experimental condition.
results = {}

# Define experiments with parameters: (label, min_sample_size, share_budget, naive).
experiments = [
    ("Naive", 0, False, True),
    ("Basic", 0, False, False),
    ("Trimmed", 0.01, False, False),
    ("Optimized", 0.01, True, False),
]

# Run the experiments.
for label, min_sample_size, share_budget, naive in experiments:
    print(f"Running experiment: {label}")
    print(label, min_sample_size, share_budget, naive)
    cov_df, lpb_df, mean_samples_df = vary_budget_per_sample(
        budgets,
        p_cal,
        p_test,
        prop_pred_cal,
        prop_pred,
        n_samples_cal,
        t_tilde_cal,
        n_conformalizations=20,
        min_sample_size=min_sample_size,
        share_budget=share_budget,
        naive=naive,
    )
    results[label] = (cov_df, lpb_df, mean_samples_df)

# --- Prepare Data for Seaborn ---

# For Coverage: Convert each DataFrame from wide to long format
coverage_list = []
for label, (cov_df, _, _) in results.items():
    # Reset index to turn the budget values into a column,
    # then melt the iteration columns into a long format.
    cov_long = cov_df.reset_index().rename(columns={"index": "Budget"})
    cov_melt = cov_long.melt(id_vars="Budget", var_name="Iteration", value_name="Coverage")
    cov_melt["Experiment"] = label
    coverage_list.append(cov_melt)
coverage_all = pd.concat(coverage_list, ignore_index=True)

# For LPB: Convert each DataFrame from wide to long format.
lpb_list = []
for label, (_, lpb_df, _) in results.items():
    lpb_long = lpb_df.reset_index().rename(columns={"index": "Budget"})
    lpb_melt = lpb_long.melt(id_vars="Budget", var_name="Iteration", value_name="LPB")
    lpb_melt["Experiment"] = label
    lpb_list.append(lpb_melt)
lpb_all = pd.concat(lpb_list, ignore_index=True)

# For Mean Samples: Convert each DataFrame from wide to long format.
mean_samples_list = []
for label, (_, _, mean_samples_df) in results.items():
    mean_samples_long = mean_samples_df.reset_index().rename(columns={"index": "Budget"})
    mean_samples_melt = mean_samples_long.melt(id_vars="Budget", var_name="Iteration", value_name="Mean Samples")
    mean_samples_melt["Experiment"] = label
    mean_samples_list.append(mean_samples_melt)
mean_samples_all = pd.concat(mean_samples_list, ignore_index=True)


# %%

# --- Plotting with Seaborn ---

sns.set(
    style="whitegrid", context="notebook", font_scale=2.5,
    rc={
      "lines.linewidth": 5,    # default line width
      "lines.markersize": 15   # default marker size
    }
)

# Set the seaborn style.
# sns.set(style="whitegrid")

plt.figure(figsize=(30, 8))

# Coverage plot.
plt.subplot(1, 3, 1)
sns.lineplot(data=coverage_all, x="Budget", y="Coverage", hue="Experiment", marker="o", errorbar="sd")
# Plot the 90% coverage line.
plt.hlines(0.9, budgets.min(), budgets.max(), label="90% coverage", color="gray", linestyle="--")
# Plot the uncalibrated coverage line.
plt.hlines(coverage(q_prop, p_vals=p_test).mean(), budgets.min(), budgets.max(), label="Uncalibrated", color="black")
plt.xlabel("Average budget per prompt")
plt.xlim(0, budgets.max())
plt.ylabel("Coverage")
# Remove the legend.
plt.legend().remove()

# LPB plot, log scale.
plt.subplot(1, 3, 3)
plt.yscale("log")
sns.lineplot(data=lpb_all, x="Budget", y="LPB", hue="Experiment", marker="o", errorbar="sd")
# Plot the uncalibrated prediction line.
plt.hlines(q_prop.mean(), budgets.min(), budgets.max(), label="Uncalibrated", color="black")
plt.xlabel("Average budget per prompt")
plt.xlim(0, budgets.max())
plt.ylabel("Average LPB")
plt.legend(loc="lower right", fontsize=26)

# Mean samples plot.
plt.subplot(1, 3, 2)
sns.lineplot(data=mean_samples_all, x="Budget", y="Mean Samples", hue="Experiment", marker="o", errorbar="sd")
plt.xlabel("Average budget per prompt")
plt.xlim(0, budgets.max())
plt.ylabel("Average # of generations")

plt.tight_layout()
# Set the resolution of the plot.
plt.gcf().set_dpi(300)
plt.legend().remove()
plt.show()
# Allow running as script
if __name__ == "__main__":
    pass
plt.savefig("figures/coverage_budget.png", dpi=300, bbox_inches="tight")


# %%
# Now do the same axperiment but varying the minimum sample size
def vary_min_sample_size(
    min_sample_sizes,
    p_cal,
    p_test,
    prop_pred_cal,
    prop_pred_test,
    n_samples_cal,
    t_tilde_cal,
    n_conformalizations=20,
    budget_per_sample=1000,
    share_budget=False,
    resplit=False,
):
    """
    Vary the min_sample_size and compute mean coverage and LPB for each sample size.

    Parameters:
        min_sample_sizes (iterable): The different values to use for min_sample_size.
        p_cal: Calibration p-values or any calibration parameter required by conformalize.
        p_test: Test p-values or any test parameter required by conformalize.
        prop_pred_cal: Calibrated propensity score predictions.
        prop_pred_test: Test set predictions.
        n_samples_cal: Number of random samples for calibration.
        t_tilde_cal: A calibration threshold or related parameter.
        n_conformalizations (int): Number of conformalization runs per sample size value.
        budget_per_sample (float): Budget allocated per sample.
        share_budget (bool): Whether to share the budget among the samples.
        resplit (bool): Whether to mix and resplit the calibration and test sets.

    Returns:
        coverage_df (pd.DataFrame): DataFrame with rows as min_sample_size values and columns as
            the conformalization iterations containing the mean coverage.
        LPB_df (pd.DataFrame): DataFrame with the corresponding LPB values.
    """
    # Initialize DataFrames with min_sample_sizes as index and conformalization iteration indices as columns.
    coverage_df = pd.DataFrame(index=min_sample_sizes, columns=range(n_conformalizations))
    LPB_df = pd.DataFrame(index=min_sample_sizes, columns=range(n_conformalizations))

    if resplit:
        # Merge the calibration and test sets.
        p_cal = np.concatenate([p_cal, p_test])
        prop_pred_cal = np.concatenate([prop_pred_cal, prop_pred_test])
        # Shuffle the merged set.
        sort_idx = np.random.permutation(len(p_cal))
        p_cal = p_cal[sort_idx]
        prop_pred_cal = prop_pred_cal[sort_idx]
        # Split the merged set back into calibration and test sets.
        p_test = p_cal[:len(p_test)]
        prop_pred_test = prop_pred_cal[:len(p_test)]
        p_cal = p_cal[len(p_test):]
        prop_pred_cal = prop_pred_cal[len(p_test):]

    # Loop over each min_sample_size value.
    for min_sample_size in min_sample_sizes:
        for i in tqdm.tqdm(range(n_conformalizations), desc=f"Processing min sample size {min_sample_size}"):
            # Run the conformalization function, passing the additional parameters.
            c_prop, max_est, _, mean_n_samples = conformalize(
                prop_pred_cal,
                p_cal,
                target_taus,
                candidate_taus,
                n_samples_cal,
                t_tilde_cal,
                budget_per_sample,
                min_sample_size=min_sample_size,
                share_budget=share_budget,
            )

            max_est = np.inf if max_est is None else max_est

            # Obtain the quantile estimator output.
            q_prop_conf = quantile_estimators(prop_pred, [c_prop[0]])[0].astype(int)
            q_prop_conf = np.minimum(q_prop_conf, max_est).astype(int)

            # Compute the mean
            cov = coverage(q_prop_conf, p_vals=p_test).mean()
            # Compute the mean LPB as the mean value of the quantile estimator.
            lpb = q_prop_conf.mean()
            # Store the values in the DataFrames.
            coverage_df.loc[min_sample_size, i] = cov
            LPB_df.loc[min_sample_size, i] = lpb
    return coverage_df, LPB_df


# Define a range of min_sample_sizes to test.
max_weight_sizes = np.logspace(1, 14, 14, base=2)
min_sample_sizes = 1 / max_weight_sizes
# Dictionary to store results for each experimental condition.
results_min_sample = {}
# Define experiments with parameters: (label, n_samples_cal, share_budget).
experiments_min_sample = [
    ("Trimmed", 100 * np.ones(len(p_cal)), False),
    ("Optimized", 100 * np.ones(len(p_cal)), True),
]
# Run the experiments.
for label, n_samples_cal, share_budget in experiments_min_sample:
    print(f"Running experiment: {label}")
    cov_df, lpb_df = vary_min_sample_size(
        min_sample_sizes,
        p_cal,
        p_test,
        prop_pred_cal,
        prop_pred,
        n_samples_cal,
        t_tilde_cal,
        n_conformalizations=20,
        budget_per_sample=1000,
        share_budget=share_budget,
    )
    results_min_sample[label] = (cov_df, lpb_df)
# --- Prepare Data for Seaborn ---
# For Coverage: Convert each DataFrame from wide to long format
coverage_list_min_sample = []
for label, (cov_df, _) in results_min_sample.items():
    # Reset index to turn the min_sample_sizes into a column,
    # then melt the iteration columns into a long format.
    cov_long = cov_df.reset_index().rename(columns={"index": "Min Sample Size"})
    cov_melt = cov_long.melt(id_vars="Min Sample Size", var_name="Iteration", value_name="Coverage")
    cov_melt["Experiment"] = label
    coverage_list_min_sample.append(cov_melt)
coverage_all_min_sample = pd.concat(coverage_list_min_sample, ignore_index=True)

# For LPB: Convert each DataFrame from wide to long format.
lpb_list_min_sample = []
for label, (_, lpb_df) in results_min_sample.items():
    lpb_long = lpb_df.reset_index().rename(columns={"index": "Min Sample Size"})
    lpb_melt = lpb_long.melt(id_vars="Min Sample Size", var_name="Iteration", value_name="LPB")
    lpb_melt["Experiment"] = label
    lpb_list_min_sample.append(lpb_melt)
lpb_all_min_sample = pd.concat(lpb_list_min_sample, ignore_index=True)
# %%
# --- Plotting with Seaborn ---
# Set the seaborn style.
sns.set(
    style="whitegrid", context="notebook", font_scale=2.5,
    rc={
      "lines.linewidth": 5,    # default line width
      "lines.markersize": 15   # default marker size
    }
)
# Set the resolution of the plot.
plt.gcf().set_dpi(300)
plt.figure(figsize=(14, 6))

coverage_all_min_sample["Max Weight"] = 1.0 / coverage_all_min_sample["Min Sample Size"]
lpb_all_min_sample["Max Weight"] = 1.0 / lpb_all_min_sample["Min Sample Size"]

# Coverage plot.
plt.subplot(1, 2, 1)
# Set the color of the optimized method to red and the trimmed method to green, both in the plot and the legend.
colors = sns.color_palette("tab10", 5)
palette = {
    "Optimized": colors[3],
    "Trimmed":   colors[2]
}

# draw the lineplot
ax = sns.lineplot(
    data=coverage_all_min_sample,
    x="Max Weight", 
    y="Coverage",
    hue="Experiment",
    palette=palette,
    marker="o",
    errorbar="sd"
)
# Plot the 90% coverage line.
plt.hlines(0.9, coverage_all_min_sample["Max Weight"].min(), coverage_all_min_sample["Max Weight"].max(), label="90% coverage", color="gray", linestyle="--")
# Write the title in Latex, Coverage vs \pi_{\text{min}}
plt.xlabel(r"$w_{\text{max}}$")
plt.xlim(coverage_all_min_sample["Max Weight"].min(), coverage_all_min_sample["Max Weight"].max())
plt.ylabel("Mean Coverage")
# LPB plot, log scale.
plt.xscale("log")
plt.legend().remove()
plt.hlines(0.9, 0, 0.1, label="90% coverage", color="gray", linestyle="--")
plt.subplot(1, 2, 2)
plt.yscale("log")
plt.xscale("log")
sns.lineplot(data=lpb_all_min_sample, x="Max Weight", y="LPB", hue="Experiment", marker="o", errorbar="sd", palette=palette)
plt.xlabel(r"$w_{\text{max}}$")
plt.xlim(lpb_all_min_sample["Max Weight"].min(), lpb_all_min_sample["Max Weight"].max())
plt.ylabel("Mean LPB")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
plt.savefig("figures/coverage_min_sample_size.png", dpi=300, bbox_inches="tight")
# %%
# Run the same budget experiment but with resplitting the calibration and test sets
# Use all the same parameters as before

# Define a range of budgets to test.
budgets = np.linspace(10, 10000, 10)

# Dictionary to store results for each experimental condition.
results = {}

# Define experiments with parameters: (label, min_sample_size, share_budget, naive).
experiments = [
    ("Naive", 0, False, True),
    ("Basic", 0, False, False),
    ("Trimmed", 0.01, False, False),
    ("Optimized", 0.01, True, False),
]

# Run the experiments.
for label, min_sample_size, share_budget, naive in experiments:
    print(f"Running experiment: {label}")
    print(label, min_sample_size, share_budget, naive)
    cov_df, lpb_df, mean_samples_df = vary_budget_per_sample(
        budgets,
        p_cal,
        p_test,
        prop_pred_cal,
        prop_pred,
        n_samples_cal,
        t_tilde_cal,
        n_conformalizations=20,
        min_sample_size=min_sample_size,
        share_budget=share_budget,
        naive=naive,
    )
    results[label] = (cov_df, lpb_df, mean_samples_df)

# --- Prepare Data for Seaborn ---

# For Coverage: Convert each DataFrame from wide to long format
coverage_list = []
for label, (cov_df, _, _) in results.items():
    # Reset index to turn the budget values into a column,
    # then melt the iteration columns into a long format.
    cov_long = cov_df.reset_index().rename(columns={"index": "Budget"})
    cov_melt = cov_long.melt(id_vars="Budget", var_name="Iteration", value_name="Coverage")
    cov_melt["Experiment"] = label
    coverage_list.append(cov_melt)
coverage_all = pd.concat(coverage_list, ignore_index=True)

# For LPB: Convert each DataFrame from wide to long format.
lpb_list = []
for label, (_, lpb_df, _) in results.items():
    lpb_long = lpb_df.reset_index().rename(columns={"index": "Budget"})
    lpb_melt = lpb_long.melt(id_vars="Budget", var_name="Iteration", value_name="LPB")
    lpb_melt["Experiment"] = label
    lpb_list.append(lpb_melt)
lpb_all = pd.concat(lpb_list, ignore_index=True)

# For Mean Samples: Convert each DataFrame from wide to long format.
mean_samples_list = []
for label, (_, _, mean_samples_df) in results.items():
    mean_samples_long = mean_samples_df.reset_index().rename(columns={"index": "Budget"})
    mean_samples_melt = mean_samples_long.melt(id_vars="Budget", var_name="Iteration", value_name="Mean Samples")
    mean_samples_melt["Experiment"] = label
    mean_samples_list.append(mean_samples_melt)
mean_samples_all = pd.concat(mean_samples_list, ignore_index=True)


# %%

# --- Plotting with Seaborn ---

sns.set(
    style="whitegrid", context="notebook", font_scale=2.5,
    rc={
      "lines.linewidth": 5,    # default line width
      "lines.markersize": 15   # default marker size
    }
)

# Set the seaborn style.
# sns.set(style="whitegrid")

plt.figure(figsize=(30, 8))

# Coverage plot.
plt.subplot(1, 3, 1)
sns.lineplot(data=coverage_all, x="Budget", y="Coverage", hue="Experiment", marker="o", errorbar="sd")
# Plot the 90% coverage line.
plt.hlines(0.9, budgets.min(), budgets.max(), label="90% coverage", color="gray", linestyle="--")
# Plot the uncalibrated coverage line.
plt.hlines(coverage(q_prop, p_vals=p_test).mean(), budgets.min(), budgets.max(), label="Uncalibrated", color="black")
plt.xlabel("Average budget per prompt")
plt.xlim(0, budgets.max())
plt.ylabel("Coverage")
# Remove the legend.
plt.legend().remove()

# LPB plot, log scale.
plt.subplot(1, 3, 3)
plt.yscale("log")
sns.lineplot(data=lpb_all, x="Budget", y="LPB", hue="Experiment", marker="o", errorbar="sd")
# Plot the uncalibrated prediction line.
plt.hlines(q_prop.mean(), budgets.min(), budgets.max(), label="Uncalibrated", color="black")
plt.xlabel("Average budget per prompt")
plt.xlim(0, budgets.max())
plt.ylabel("Average LPB")
plt.legend(loc="lower right", fontsize=26)

# Mean samples plot.
plt.subplot(1, 3, 2)
sns.lineplot(data=mean_samples_all, x="Budget", y="Mean Samples", hue="Experiment", marker="o", errorbar="sd")
plt.xlabel("Average budget per prompt")
plt.xlim(0, budgets.max())
plt.ylabel("Average # of generations")

plt.tight_layout()
# Set the resolution of the plot.
plt.gcf().set_dpi(300)
plt.legend().remove()
plt.show()
plt.savefig("figures/coverage_budget_resplit.png", dpi=300, bbox_inches="tight")

# %%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters for experiments
budgets = np.linspace(10, 10000, 10)
experiments = [
    ("Naive",     0,     False, True),
    ("Basic",     0,     False, False),
    ("Trimmed",   0.01,  False, False),
    ("Optimized", 0.01,  True,  False),
]
test_target_taus = np.array([0.05, 0.15, 0.2])

# Run experiments and collect results
all_results = {}
for target_tau in test_target_taus:
    print(f"Running experiments for target τ = {target_tau}")
    results = {}
    target_taus = np.array([target_tau])
    for label, min_ss, share_budg, naive in experiments:
        cov_df, lpb_df, mean_samples_df = vary_budget_per_sample(
            budgets,
            p_cal,
            p_test,
            prop_pred_cal,
            prop_pred,
            n_samples_cal,
            t_tilde_cal,
            n_conformalizations=20,
            min_sample_size=min_ss,
            share_budget=share_budg,
            naive=naive,
        )
        results[label] = (cov_df, lpb_df, mean_samples_df)
    all_results[target_tau] = results

# %%

# Plotting code using the precomputed results
sns.set(
    style="whitegrid", context="notebook", font_scale=3.0,
    rc={
      "lines.linewidth": 5,
      "lines.markersize": 15
    }
)

fig, axes = plt.subplots(
    nrows=2,
    ncols=len(test_target_taus),
    figsize=(30, 14),
    sharex=True
)

for i, target_tau in enumerate(test_target_taus):
    results = all_results[target_tau]

    # Melt results for seaborn
    cov_list = []
    lpb_list = []
    for label, (cov_df, lpb_df, _) in results.items():
        tmp_cov = cov_df.reset_index().rename(columns={"index": "Budget"})
        tmp_cov = tmp_cov.melt(id_vars="Budget", var_name="Iteration", value_name="Coverage")
        tmp_cov["Experiment"] = label
        cov_list.append(tmp_cov)

        tmp_lpb = lpb_df.reset_index().rename(columns={"index": "Budget"})
        tmp_lpb = tmp_lpb.melt(id_vars="Budget", var_name="Iteration", value_name="LPB")
        tmp_lpb["Experiment"] = label
        lpb_list.append(tmp_lpb)

    coverage_all = pd.concat(cov_list, ignore_index=True)
    lpb_all = pd.concat(lpb_list, ignore_index=True)

    # Coverage plot
    ax_cov = axes[0, i]
    sns.lineplot(
        data=coverage_all,
        x="Budget", y="Coverage",
        hue="Experiment", marker="o",
        errorbar="sd", ax=ax_cov
    )
    ax_cov.hlines(1 - target_tau, budgets.min(), budgets.max(),
                  label="Target coverage (1 - τ)", linestyle="--", color="gray")
    uncal = coverage(q_prop, p_vals=p_test).mean()
    ax_cov.hlines(uncal, budgets.min(), budgets.max(),
                  label="Uncalibrated", color="black")
    ax_cov.set_xlim(0, budgets.max())
    ax_cov.set_xlabel("Average budget per prompt")
    if i == 0:
        ax_cov.set_ylabel("Coverage")
    else:
        ax_cov.set_ylabel('')
    ax_cov.set_title(f"Target coverage = {1 - target_tau}")
    ax_cov.get_legend().remove()

    # LPB plot
    ax_lpb = axes[1, i]
    ax_lpb.set_yscale("log")
    sns.lineplot(
        data=lpb_all,
        x="Budget", y="LPB",
        hue="Experiment", marker="o",
        errorbar="sd", ax=ax_lpb
    )
    uncal_lpb = quantile_estimators(prop_pred, [target_tau])[0].mean()
    ax_lpb.hlines(uncal_lpb, budgets.min(), budgets.max(),
                   label="Uncalibrated", color="black")
    ax_lpb.set_xlim(0, budgets.max())
    ax_lpb.set_xlabel("Average budget per prompt")
    if i == 0:
        ax_lpb.set_ylabel("Average LPB")
    else:
        ax_lpb.set_ylabel('')
    ax_lpb.get_legend().remove()

# Single legend on the right
handles, labels = axes[1, -1].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="center right",
    bbox_to_anchor=(1.13, 0.5),
    ncol=1,
    frameon=False,
    fontsize=30
)

plt.subplots_adjust(right=0.85)
fig.tight_layout(pad=1.0, h_pad=3.0)
# Set high resolution
plt.gcf().set_dpi(300)

# Save and show
fig.savefig("figures/coverage_budget_multiple_taus_with_legend.png", dpi=300, bbox_inches="tight")
plt.show()
# %%
