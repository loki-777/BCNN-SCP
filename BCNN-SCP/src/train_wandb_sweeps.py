import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.main import LightningModule
from .utils import *

import os, argparse, yaml

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
    
    sweep_config = {
        'method': 'bayes',  # Choose search strategy: grid, random, or bayes
        'metric': {
            'name': 'val_loss',  # Metric to optimize
            'goal': 'minimize'   # Minimize or maximize the metric
        },
        'parameters': {
            'prior_a': {
                'values': [16, 32, 64]
            },
            'prior_l': {
            },
            'kernel_a_mu': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_a_sigma': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_a_min': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_a_max': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_l_mu': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_l_sigma': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_l_min': {
                'values': [0.1, 0.2, 0.3]
            },
            'kernel_l_max': {
                'values': [0.1, 0.2, 0.3]
            }
        }
    }

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available. Number of GPUs: {num_gpus}")
    else:
        print("GPU is not available.")

    use_gpu = True if config["device"] == "gpu" else False

    # prep data
    train_loader, val_loader, test_loader = get_dataloaders(config)

    pl.seed_everything(42)

    wandb_logger = WandbLogger(
        project=config["project_name"],
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
        logger=wandb_logger
    )

    def train_model(config=None):
        with wandb.init(config=config):
            config = wandb.config
            
            # Update hyperparameters in your Lightning Module
            model = LightningModule(learning_rate=config.learning_rate, dropout=config.dropout)
            
            # Set up DataLoaders
            train_loader = ...
            val_loader = ...
            
            trainer = pl.Trainer(
                max_epochs=10,
                logger=wandb_logger,
                gpus=1
            )
            trainer.fit(model, train_loader, val_loader)
