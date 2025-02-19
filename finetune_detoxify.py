# Adapted github.com/unitaryai/detoxify
import argparse
import json
from math import ceil
import os
from typing import Literal
import transformers
from torch import sigmoid

# Disable tokenizer parallelism to avoid deadlocks due to forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pytorch_lightning as pl
from src import datasets
from src.loss import survival_loss, prop_loss
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchmetrics

def get_model_and_tokenizer(model_type, model_name, tokenizer_name, num_classes):
    model = getattr(transformers, model_name).from_pretrained(model_type, num_labels=num_classes)
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(model_type)

    return model, tokenizer

class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification for the Jigsaw challenges.
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.model, self.tokenizer = get_model_and_tokenizer(**self.model_args)
        self.auroc = torchmetrics.AUROC(task="binary")

        self.num_main_classes = self.num_classes

        self.config = config

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def training_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        if isinstance(meta, list):
            meta = [item.to(output.device) for item in meta]
        else:
            meta = meta.to(output.device)
        loss = getattr(self, self.config["loss"])(output, meta)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        ce = self.binary_cross_entropy(output, meta[0])
        # acc = self.binary_accuracy(output, meta)
        variance = torch.var(output[:,1])
        
        # preds = output[:, 1]  
        # targets = meta.int().to(preds.device)
        # self.auroc.update(torch.sigmoid(preds), targets)

        self.log("val_ce", ce, prog_bar=True)
        # self.log("val_acc", acc)
        self.log("variance", variance, prog_bar=True)
        return {"ce": ce}
    
    # def on_validation_epoch_end(self):
    #     # Compute the AUROC over the validation epoch
    #     auroc_value = self.auroc.compute()
    #     self.log('val_auroc', auroc_value, prog_bar=True)
    #     # Reset the metric for the next epoch
    #     self.auroc.reset()

    def test_step(self, batch, batch_idx):
        x, meta = batch
        output = self.forward(x)
        ce = self.binary_cross_entropy(output, meta)
        acc = self.binary_accuracy(output, meta)
        self.log("test_ce", ce)
        self.log("test_acc", acc)
        return {"loss": ce, "acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])

    def binary_cross_entropy(self, input, meta):
        """Custom binary_cross_entropy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([torch.tensor]): meta tensor including targets

        Returns:
            [torch.tensor]: model loss
        """
        return F.binary_cross_entropy_with_logits(input[:,1], meta.float())

    def survival_loss(self, input, meta):
        return survival_loss(input, meta)
    
    def prop_loss(self, input, meta):
        return prop_loss(input, meta)

    def binary_accuracy(self, output, meta):
        """Custom binary_accuracy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        target = meta.to(output.device)
        with torch.no_grad():
            mask = target != -1
            pred = torch.sigmoid(output[:,1][mask]) >= 0.5
            correct = torch.sum(pred.to(output[mask].device) == target[mask])
            if torch.sum(mask).item() != 0:
                correct = correct.item() / torch.sum(mask).item()
            else:
                correct = 0

        return torch.tensor(correct)


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

    # training

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
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
    cli_main()
