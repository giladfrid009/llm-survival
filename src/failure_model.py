# Adapted github.com/unitaryai/detoxify
import transformers

import pytorch_lightning as pl
from src.loss import survival_loss, prop_loss
import torch
from torch.nn import functional as F
import torchmetrics
from scipy.stats import geom

def get_model_and_tokenizer(model_type, model_name, tokenizer_name, num_classes):
    model = getattr(transformers, model_name).from_pretrained(model_type, num_labels=num_classes)
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(model_type)

    return model, tokenizer

class ToxicClassifier(pl.LightningModule):
    """Toxic comment classification originally built for the Jigsaw challenges.
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
        self.taus = None
        self.min_p = None

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
        loss = getattr(self, self.config["loss"])(output, meta, self.config["L1_reg"] if "L1_reg" in self.config.keys() else 0.0)
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

    def predict_proba(self, x):
        with torch.no_grad():
            output = self.forward(x)
            return torch.sigmoid(output[:,1])
        
    def set_min_p_for_q_tau(self, min_p):
        self.min_p = min_p

    # Survival time T is the time until a failure occurs
    # It is Geometrically distributed with probability p
    # Predict the tau quantile of the distribution
    def predict_q_tau(self, p, taus):
        """
        Predicts the quantiles taus of the survival time distribution

        Args:
            p ([torch.tensor]): predicted probabilities
            taus ([torch.tensor]): quantiles to predict

        Returns:
            [torch.tensor]: predicted quantiles
        """
        with torch.no_grad():
            if self.min_p:
                p = torch.max(p, self.min_p)
            assert torch.all(p > 0), "p must be greater than 0"
            assert torch.all(p < 1), "p must be less than 1"
            p_np = p.cpu().numpy()
            q_tau = geom.ppf(taus.cpu().numpy(), p_np)
            return torch.tensor(q_tau).to(p.device)


    def set_taus(self, taus):
        self.taus = taus
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch

        proba = self.predict_proba(x)

        if self.taus != None:
            q_taus = self.predict_q_tau(proba, self.taus)
        else:
            q_taus = None
        
        return {"proba": proba, "tau": q_taus}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])

    def binary_cross_entropy(self, input, meta, L1_reg=0.0):
        """Custom binary_cross_entropy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([torch.tensor]): meta tensor including targets

        Returns:
            [torch.tensor]: model loss
        """
        if L1_reg == 0.0:
            return F.binary_cross_entropy_with_logits(input[:,1], meta.float())
        else:
            return F.binary_cross_entropy_with_logits(input[:,1], meta.float()) + L1_reg * torch.mean(torch.sigmoid(input[:,1]))

    def survival_loss(self, input, meta, L1_reg=0.0):
        return survival_loss(input, meta, L1_reg)
    
    def prop_loss(self, input, meta, L1_reg=0.0):
        return prop_loss(input, meta, L1_reg)

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