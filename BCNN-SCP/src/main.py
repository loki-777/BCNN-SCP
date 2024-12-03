
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score
import torch.optim as optim

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
        logits = preds["logits"]
        kl_loss = preds["kl_loss"]

        if len(logits.shape) == 3:
            logits = logits.permute(0, 2, 1) # (batch_size, num_samples, num_classes) -> (batch_size, num_classes, num_samples)
            labels = labels.unsqueeze(-1) # (batch_size, ) -> (batch_size, 1)
            labels = labels.expand(-1, logits.shape[-1]) # (batch_size, 1) -> (batch_size, num_samples)
            criterion_loss = self.loss_module(logits, labels) # (batch_size, num_samples)
            criterion_loss = criterion_loss.mean(-1) # average over samples
            criterion_loss = criterion_loss.sum() # sum over minibatch
        else:
            criterion_loss = self.loss_module(logits, labels).sum()

        combined_loss = criterion_loss
        if kl_loss:
            combined_loss += kl_loss

        self.log("train_loss", combined_loss)
        return combined_loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)["logits"].squeeze().argmax(dim=-1)

        if "accuracy" in self.config["validation"]["metrics"]:
            self.log("val_accuracy", self.accuracy_metric(preds, labels))
        if "precision" in self.config["validation"]["metrics"]:
            self.log("val_precision", self.precision_metric(preds, labels))
        if "recall" in self.config["validation"]["metrics"]:
            self.log("val_recall", self.recall_metric(preds, labels))
        if "f1" in self.config["validation"]["metrics"]:
            self.log("val_f1", self.f1_metric(preds, labels))

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)["logits"].squeeze().argmax(dim=-1)

        if "accuracy" in self.config["validation"]["metrics"]:
            self.log("val_accuracy", self.accuracy_metric(preds, labels))
        if "precision" in self.config["validation"]["metrics"]:
            self.log("val_precision", self.precision_metric(preds, labels))
        if "recall" in self.config["validation"]["metrics"]:
            self.log("val_recall", self.recall_metric(preds, labels))
        if "f1" in self.config["validation"]["metrics"]:
            self.log("val_f1", self.f1_metric(preds, labels))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Config file path.")
    parser.add_argument(
        "file_path",
        type=str,
        help="Config file path"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        print(f"Error: File '{args.file_path}' does not exist.")
        exit

    with open(args.file_path, "r") as file:
        config = yaml.safe_load(file)

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available. Number of GPUs: {num_gpus}")
    else:
        print("GPU is not available.")

    # prep data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    pl.seed_everything(42)

    wandb_logger = WandbLogger(
        project=config["project_name"],
        name=config["experiment_name"],
        log_model=False
    )
    wandb_logger.experiment.config.update(config)

    trainer = pl.Trainer(
        default_root_dir=os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="cpu",
        # devices=num_gpus,
        # How many epochs to train for if no patience is set
        max_epochs=config["training"]["epochs"],
        callbacks=[
            ModelCheckpoint(
                dirpath=os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]),
                save_weights_only=True, mode="max", monitor=config["logging"]["monitor_metric"],
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],
        logger=wandb_logger
    )

    model = LightningModule(config)
    print(model)
    trainer.fit(model, train_loader, val_loader)
