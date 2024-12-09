import torch
import pytorch_lightning as pl
import wandb

from src.main import LightningModule

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
                save_weights_only=True, mode="max", monitor=config["logging"]["monitor_metric"],
            ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("epoch"),
        ],
        logger=wandb_logger
    )

    if config["action"] == "train":
        model = LightningModule(config)
        trainer.fit(model, train_loader, val_loader)
    else:
        model = LightningModule.load_from_checkpoint(
        checkpoint_path=config["test"]["checkpoint_path"],
        config=config)

        trainer.test(model, test_loader)
