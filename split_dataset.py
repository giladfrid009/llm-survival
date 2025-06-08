"""Split a pickled survival dataset into train/val/cal/test sets."""

from __future__ import annotations

import argparse
import os
import pickle as pkl
import random
from typing import List
from src import utils

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a pickled dataset into subsets")
    parser.add_argument("data_path", help="Path to the pickled dataset list")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for shuffling")
    parser.add_argument(
        "--proportions",
        default="0.5,0.1,0.2,0.2",
        help="Comma-separated proportions for train,val,cal,test",
    )
    parser.add_argument(
        "--take_first",
        type=int,
        default=None,
        help="Use only the first N samples from the dataset",
    )
    parser.add_argument(
        "--take_last",
        type=int,
        default=None,
        help="Use only the last N samples from the dataset",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write the split pickles (default derived from data_path)",
    )
    
    parsed = parser.parse_args()
    
    # make all paths absolute
    parsed.data_path = utils.abs_path(parsed.data_path)
    parsed.output_dir = utils.abs_path(parsed.output_dir, ignore=None)
    
    # print all args
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    
    return parsed


def main() -> None:
    args = parse_args()
    if args.take_first is not None and args.take_last is not None:
        raise ValueError("Only one of --take_first and --take_last may be given")

    proportions = [float(p) for p in args.proportions.split(",")]
    total = sum(proportions)
    if abs(total - 1.0) > 1e-6:
        proportions = [p / total for p in proportions]

    random.seed(args.seed)

    with open(args.data_path, "rb") as f:
        data: List = pkl.load(f)

    if args.take_first is not None:
        data = data[: args.take_first]
    if args.take_last is not None:
        data = data[-args.take_last :]

    n = len(data)
    indices = list(range(n))
    random.shuffle(indices)

    cumulative = 0.0
    split_points = []
    for p in proportions[:-1]:
        cumulative += p
        split_points.append(int(cumulative * n))

    split_indices = []
    prev = 0
    for point in split_points:
        split_indices.append(indices[prev:point])
        prev = point
    split_indices.append(indices[prev:])

    if args.output_dir is None:
        data_dir = os.path.dirname(args.data_path)
        prop_str = args.proportions.replace(",", "_")
        postfix = ""
        if args.take_first is not None:
            postfix = f"_first_{args.take_first}"
        if args.take_last is not None:
            postfix = f"_last_{args.take_last}"
        args.output_dir = os.path.join(data_dir, f"split_{args.seed}_{prop_str}{postfix}")
    os.makedirs(args.output_dir, exist_ok=True)

    split_names = ["train", "val", "cal", "test"] if len(proportions) == 4 else [f"split_{i}" for i in range(len(proportions))]

    for name, idx_list in zip(split_names, split_indices):
        split_data = [data[i] for i in idx_list]
        path = os.path.join(args.output_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pkl.dump(split_data, f)
        print(f"Saved {len(split_data)} items to {path}")


if __name__ == "__main__":
    main()
