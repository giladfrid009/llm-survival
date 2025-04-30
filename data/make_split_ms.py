import os
import pickle as pkl
import random
import sys
from typing import List
import argparse

# Parse command-line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--data_path', type=str, required=True, help="Path to the pickled data file.")
argparser.add_argument('--seed', type=int, required=True, help="Random seed for shuffling.")
argparser.add_argument('--proportions', type=str, required=True,
                       help="Comma-separated list of proportions for the split (e.g., 0.7,0.1,0.1,0.1).")
argparser.add_argument('--take_first_N', type=int, default=None, help="Take the first N samples from the data.")
argparser.add_argument('--take_last_N', type=int, default=None, help="Take the last N samples from the data.")
args = argparser.parse_args()

# Assert that take_first_N and take_last_N are not both provided.
if args.take_first_N is not None and args.take_last_N is not None:
    sys.exit("Only one of take_first_N and take_last_N can be provided.")

data_path = args.data_path
seed = args.seed
# Convert the string of proportions to a list of floats.
proportions = [float(p) for p in args.proportions.split(",")]

# Normalize proportions if they do not sum to 1.
total = sum(proportions)
if abs(total - 1.0) > 1e-6:
    proportions = [p / total for p in proportions]

# Set the random seed for reproducibility.
random.seed(seed)

# Load the data (assumed to be a pickled list)
with open(data_path, "rb") as f:
    data = pkl.load(f)

# Optionally take the first N samples.
if args.take_first_N is not None:
    data = data[:args.take_first_N]

# Optionally take the last N samples.
if args.take_last_N is not None:
    data = data[-args.take_last_N:]

n = len(data)
indices = list(range(n))
random.shuffle(indices)

# Compute the boundaries for each split.
split_points = []
cumulative = 0.0
for p in proportions[:-1]:
    cumulative += p
    split_points.append(int(cumulative * n))

# Partition the indices into splits.
split_indices = []
prev = 0
for point in split_points:
    split_indices.append(indices[prev:point])
    prev = point
split_indices.append(indices[prev:])  # The last split takes the remainder.

# Determine the output directory.
data_dir = os.path.dirname(data_path)
# Replace commas with underscores in the proportions string for a valid folder name.
proportions_str = args.proportions.replace(",", "_")
first_last_postfix = ""
if args.take_first_N is not None:
    first_last_postfix = f"_first_{args.take_first_N}"
if args.take_last_N is not None:
    first_last_postfix = f"_last_{args.take_last_N}"
split_folder = os.path.join(data_dir, f"split_{seed}_{proportions_str}{first_last_postfix}")
os.makedirs(split_folder, exist_ok=True)

# Define split names.
# If exactly 4 splits are provided, use train/val/cal/test names; otherwise use generic names.
if len(proportions) == 4:
    split_names = ["train", "val", "cal", "test"]
else:
    split_names = [f"split_{i}" for i in range(len(proportions))]

# Save each split as a separate pickle file.
for name, idx_list in zip(split_names, split_indices):
    split_data = [data[i] for i in idx_list]
    split_file_path = os.path.join(split_folder, f"{name}.pkl")
    with open(split_file_path, "wb") as f:
        pkl.dump(split_data, f)
    print(f"Saved {len(split_data)} items to {split_file_path}")
