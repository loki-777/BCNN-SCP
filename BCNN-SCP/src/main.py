
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

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
        self.loss_module = nn.CrossEntropyLoss()

        # metrics
        self.accuracy = Accuracy(num_classes=config["data"]["num_classes"])
        self.precision = Precision(num_classes=config["data"]["num_classes"], average='macro')
        self.recall = Recall(num_classes=config["data"]["num_classes"], average='macro')
        self.f1 = F1Score(num_classes=config["data"]["num_classes"], average='macro')

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
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=self.config["training"]["gamma"])
        return [optimizer], [scheduler]

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
            criterion_loss = self.loss_module(logits, labels, reduction='none') # (batch_size, num_samples)
            criterion_loss = criterion_loss.mean(-1) # average over samples
            criterion_loss = criterion_loss.sum() # sum over minibatch
        else:
            criterion_loss = self.loss_module(logits, labels, reduction='sum')
        
        combined_loss = criterion_loss
        if kl_loss:
            combined_loss += kl_loss

        self.log("train_loss", combined_loss)
        return combined_loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).squeeze().argmax(dim=-1)

        if "accuracy" in self.config["validation"]["metrics"]:
            self.log("val_accuracy", self.accuracy(preds, labels))
        if "precision" in self.config["validation"]["metrics"]:
            self.log("val_precision", self.precision(preds, labels))
        if "recall" in self.config["validation"]["metrics"]:
            self.log("val_recall", self.recall(preds, labels))
        if "f1" in self.config["validation"]["metrics"]:
            self.log("val_f1", self.f1(preds, labels))

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).squeeze().argmax(dim=-1)
        
        if "accuracy" in self.config["validation"]["metrics"]:
            self.log("val_accuracy", self.accuracy(preds, labels))
        if "precision" in self.config["validation"]["metrics"]:
            self.log("val_precision", self.precision(preds, labels))
        if "recall" in self.config["validation"]["metrics"]:
            self.log("val_recall", self.recall(preds, labels))
        if "f1" in self.config["validation"]["metrics"]:
            self.log("val_f1", self.f1(preds, labels))

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

    trainer = pl.Trainer(
        default_root_dir=os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]),  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="auto",
        devices=num_gpus,
        # How many epochs to train for if no patience is set
        max_epochs=config["training"]["epochs"],
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True, mode="max", monitor="val_acc"
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],  # Log learning rate every epoch
    )

    model = LightningModule(config)
    trainer.fit(model, train_loader, val_loader)