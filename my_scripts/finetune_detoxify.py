# NOTE: working

# Adapted github.com/unitaryai/detoxify
import argparse
import json
import os
from typing import Literal
from src import utils

# Disable tokenizer parallelism to avoid deadlocks due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from src import datasets
from src.failure_model import ToxicClassifier
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import torch._dynamo
import logging

torch._dynamo.config.suppress_errors = True


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="comma-separated indices of GPUs to enable (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers used in the data loader (default: 10)",
    )
    parser.add_argument("-e", "--n_epochs", default=2, type=int, help="if given, override the num")

    args = parser.parse_args()

    # make all paths absolute
    if args.config is not None:
        args.config = utils.abs_path(args.config)
    if args.resume is not None:
        args.resume = utils.abs_path(args.resume)

    # print all args
    print("Command line arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    # data
    def get_instance(module, name, config, stage: Literal["train", "val", "cal", "test"] = "train", *args, **kwargs):
        return getattr(module, config[name][stage]["type"])(config[name][stage]["pkl_path"], *args, **config[name]["args"], **kwargs)

    dataset = get_instance(datasets, "dataset", config, stage="train")
    val_dataset = get_instance(datasets, "dataset", config, stage="val")

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )

    # model
    model = ToxicClassifier(config)
    model = model.train()

    # training
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_ce",
        mode="min",
    )

    if args.device is None:
        devices = "auto"
    else:
        devices = [int(d.strip()) for d in args.device.split(",")]

    trainer = pl.Trainer(
        devices=devices,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        default_root_dir="saved/" + config["name"],
        deterministic=True,
    )

    trainer.fit(
        model=model,
        train_dataloaders=data_loader,
        val_dataloaders=valid_data_loader,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    utils.configure_logging(logging.WARNING)
    cli_main()
