import sys, os, glob, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, CalibrationError
import torch.optim as optim
import wandb

from .utils import *

import yaml
import argparse


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = get_model(config)

        # metrics
        self.accuracy_metric = Accuracy(num_classes=config["data"]["num_classes"], task="multiclass")
        self.precision_metric = Precision(num_classes=config["data"]["num_classes"], average='macro', task="multiclass")
        self.recall_metric = Recall(num_classes=config["data"]["num_classes"], average='macro', task="multiclass")
        self.f1_metric = F1Score(num_classes=config["data"]["num_classes"], average='macro', task="multiclass")
        self.ece_metric = CalibrationError(num_classes=config["data"]["num_classes"], norm='l1', task="multiclass")
        self.loss_module = nn.CrossEntropyLoss(reduction='none')

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.config["training"]["optimizer"] == "Adam":
            optimizer = optim.AdamW(self.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        elif self.config["training"]["optimizer"] == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])
        else:
            assert False, f'Unknown optimizer: "{self.config["training"]["optimizer"]}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=self.config["training"]["scheduler"]["gamma"])
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        logits = preds["logits"] # (..., C)
        kl_loss = preds["kl_loss"]

        # Get probabilities
        probs = logits.softmax(dim=-1) # (..., C)

        # If sampling, average over samples
        if probs.dim() == 3:
            probs = probs.mean(dim=1) # (B, S, C) -> (B, C)

        # Log training metrics
        if "accuracy" in self.config["training"]["metrics"]:
            self.log("train_accuracy", self.accuracy_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "precision" in self.config["training"]["metrics"]:
            self.log("train_precision", self.precision_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "recall" in self.config["training"]["metrics"]:
            self.log("train_recall", self.recall_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "f1" in self.config["training"]["metrics"]:
            self.log("train_f1", self.f1_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "ece" in self.config["training"]["metrics"]:
            self.log("train_ece", self.ece_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)

        # If sampling, compute loss for each sample and average
        # (B, S, C)
        if logits.dim() == 3:
            logits = logits.permute(0, 2, 1) # (B, S, C) -> (B, C, S)
            labels = labels.unsqueeze(-1) # (B, ) -> (B, 1)
            labels = labels.expand(-1, logits.shape[-1]) # (B, 1) -> (B, S)
            criterion_loss = self.loss_module(logits, labels) # (B, S)
            criterion_loss = criterion_loss.mean(-1) # average over samples
            criterion_loss = criterion_loss.sum() # sum over minibatch
        # (B, C)
        else:
            criterion_loss = self.loss_module(logits, labels).sum()

        combined_loss = 0.0
        combined_loss += criterion_loss
        if kl_loss:
            combined_loss += kl_loss

        self.log("train_loss", combined_loss, sync_dist=True, on_step=False, on_epoch=True)
        if kl_loss:
            self.log("train_ce_loss", criterion_loss, sync_dist=True, on_step=False, on_epoch=True)
            self.log("train_kl_loss", kl_loss, sync_dist=True, on_step=False, on_epoch=True)
        return combined_loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.model(imgs)["logits"] # (..., C)

        # Get probabilities
        probs = logits.softmax(dim=-1) # (..., C)

        # If sampling, average over samples
        if probs.dim() == 3:
            probs = probs.mean(dim=1) # (B, S, C) -> (B, C)

        # Log validation metrics
        if "accuracy" in self.config["validation"]["metrics"]:
            self.log("val_accuracy", self.accuracy_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "precision" in self.config["validation"]["metrics"]:
            self.log("val_precision", self.precision_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "recall" in self.config["validation"]["metrics"]:
            self.log("val_recall", self.recall_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "f1" in self.config["validation"]["metrics"]:
            self.log("val_f1", self.f1_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "ece" in self.config["validation"]["metrics"]:
            self.log("val_ece", self.ece_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # assumes 1 batch size
        imgs, labels = batch
        logits = self.model(imgs)["logits"] # (..., C)

        # Get probabilities
        probs = logits.softmax(dim=-1) # (..., C)

        # If sampling, average over samples
        if probs.dim() == 3:
            probs = probs.mean(dim=1) # (B, S, C) -> (B, C)

        if "accuracy" in self.config["testing"]["metrics"]:
            self.log("test_accuracy", self.accuracy_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "precision" in self.config["testing"]["metrics"]:
            self.log("test_precicsion", self.precision_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "recall" in self.config["testing"]["metrics"]:
            self.log("test_recall", self.recall_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "f1" in self.config["testing"]["metrics"]:
            self.log("test_f1", self.f1_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)
        if "ece" in self.config["testing"]["metrics"]:
            self.log("test_ece", self.ece_metric(probs, labels), sync_dist=True, on_step=False, on_epoch=True)

    def predict_single(self, x):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # Disable gradient computation
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float)  # Convert input to tensor if needed
            if x.ndim == 1:
                x = x.unsqueeze(0)  # Add batch dimension for single data point
            probs = self(x)["logits"].softmax(dim=-1).squeeze()
            return probs



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config file path.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Config file path"
    )

    # Optional arguments for prior_a and prior_l
    parser.add_argument("-a", "--prior_a", type=float, help="outputscale", required=False, default=100.0)
    parser.add_argument("-l", "--prior_l", type=float, help="lengthscale", required=False, default=1.0)
    parser.add_argument("-k", "--prior_k", type=str, help="kernel", required=False, default="RBF")

    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        exit

    with open(args.file_path, "r") as file:
        config = yaml.safe_load(file)

    config["model"]["prior_kernel"]["params"]["a"] = args.prior_a
    config["model"]["prior_kernel"]["params"]["l"] = args.prior_l
    config["model"]["prior_kernel"]["name"] = args.prior_k
    config["model"]["kernel"]["name"] = args.prior_k

    config["experiment_name"] = config["experiment_name"] + f" a={args.prior_a} l={args.prior_l} k={args.prior_k}"
    config["logging"]["save_name"] = config["experiment_name"]

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available. Number of GPUs: {num_gpus}")
    else:
        print("GPU is not available.")

    use_gpu = True if config["device"] == "gpu" else False


    pl.seed_everything(42)

    # prep data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=config["experiment_name"],
        log_model=False
    )
    wandb_logger.experiment.config.update(config)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator=config["device"],
        devices=num_gpus if use_gpu else "auto",
        # How many epochs to train for if no patience is set
        max_epochs=config["training"]["epochs"],
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]),
                save_weights_only=True, save_last=True, save_top_k=0
            ),
            LearningRateMonitor("epoch"),
        ],
        logger=wandb_logger
    )

    if config["action"] == "test":
        model = LightningModule.load_from_checkpoint(
        checkpoint_path=config["testing"]["checkpoint_path"],
        config=config)
        trainer.test(model, test_loader)
    else:
        model = LightningModule(config)

        # "DNC" cases will break out automatically without running test
        # exceptions are logged in wandb logs
        trainer.fit(model, train_loader, val_loader)

        if config["action"] != "train":
            # train and test
            # model = LightningModule.load_from_checkpoint(
            # checkpoint_path=glob.glob(os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]) + "/*.ckpt")[0],
            # config=config)
            trainer.test(model, test_loader)
