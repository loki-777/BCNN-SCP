from src.models.networks import *

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_model(config):
    if config["model"]["model_name"] == "BCNN":
        return BCNN(config["model"]["num_samples"], config["model"]["kernel"])
    elif config["model"]["model_name"] == "CNN":
        return CNN()


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
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader