import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from src.main import LightningModule
from src.utils import *

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
        "method": 'bayes',
        "metric": {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        "parameters": {
            'prior_a': {
                'min': 0.1,    # Fixed: Added quotes to min/max
                'max': 1000.0
            },
            'prior_l': {
                'min': 0.1,    # Fixed: Added quotes to min/max
                'max': 3.0
            },
        }
    }

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"GPU is available. Number of GPUs: {num_gpus}")
    else:
        print("GPU is not available.")

    use_gpu = True if config["device"] == "gpu" else False

    pl.seed_everything(42)

    wandb_logger = WandbLogger(
        project=config["project_name"],
        log_model=False
    )

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
        with wandb.init():
            # Get the sweep config from wandb, not from the input parameter
            sweep_config = wandb.config
            
            # Update hyperparameters in your Lightning Module
            config["model"]["prior_kernel"]["params"]["a"] = sweep_config.prior_a  # Fixed: Access parameters directly
            config["model"]["prior_kernel"]["params"]["l"] = sweep_config.prior_l  # Fixed: Access parameters directly
            model = LightningModule(config)

            train_loader, val_loader, _ = get_dataloaders(config)

            wandb_logger = WandbLogger(
                project=config["project_name"],
                log_model=False
            )

            trainer = pl.Trainer(
                default_root_dir=os.path.join(config["logging"]["checkpoint_dir"], config["logging"]["save_name"]),  # Where to save models
                # We run on a single GPU (if possible)
                accelerator=config["device"],
                devices=num_gpus if use_gpu else "auto",
                # How many epochs to train for if no patience is set
                max_epochs=config["training"]["epochs"],
                logger=wandb_logger
            )

            trainer.fit(model, train_loader, val_loader)
    
    sweep_id = wandb.sweep(sweep_config, project=config["project_name"])
    # Start the sweep agent
    wandb.agent(sweep_id, train_model)  