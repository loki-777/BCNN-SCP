from src.models.networks import *

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_model(config):
    if config["model"]["model_name"] == "BCNN":
        return BCNN(config["model"]["prior_kernel"],
                    config["model"]["prior_kernel_params"],
                    out_channels_conv1=config["model"]["out_channels_conv1"],
                    out_channels_conv2=config["model"]["out_channels_conv2"],
                    num_samples_training=config["model"]["num_samples_training"],
                    num_samples_predict=config["model"]["num_samples_predict"],
                    kernel=config["model"]["kernel"],
                    kernel_params_init=config["model"]["kernel_params_init"])
    elif config["model"]["model_name"] == "CNN":
        return CNN(out_channels_conv1=config["model"]["out_channels_conv1"],
                   out_channels_conv2=config["model"]["out_channels_conv2"])


def get_dataloaders(config):
    if config["data"]["dataset"] == "MNIST":
        # Set up data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((config["data"]["normalize_mean"],), (config["data"]["normalize_std"],))
        ])

        # Download and load the MNIST dataset
        dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

        # Split the training dataset into train and validation subsets
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = len(dataset) - train_size  # 20% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create dataloaders
        batch_size = config["data"]["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["data"]["num_workers"])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config["data"]["num_workers"])
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config["data"]["num_workers"])

        return train_loader, val_loader, test_loader
