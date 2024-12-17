import glob, os
import cv2 as cv
from PIL import Image
from src.models.networks import *

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_model(config):
    if config["model"]["model_name"] == "BCNN":
        return BCNN(out_channels_conv1=config["model"]["out_channels_conv1"],
                    out_channels_conv2=config["model"]["out_channels_conv2"],
                    filter_size_conv1=config["model"]["filter_size_conv1"],
                    filter_size_conv2=config["model"]["filter_size_conv2"],
                    num_samples_training=config["model"]["num_samples_training"],
                    num_samples_predict=config["model"]["num_samples_predict"],
                    prior_kernel=config["model"]["prior_kernel"],
                    kernel=config["model"]["kernel"])
    elif config["model"]["model_name"] == "CNN":
        return CNN(out_channels_conv1=config["model"]["out_channels_conv1"],
                   out_channels_conv2=config["model"]["out_channels_conv2"],
                   filter_size_conv1=config["model"]["filter_size_conv1"],
                   filter_size_conv2=config["model"]["filter_size_conv2"])

class PneumoniaDataset(pl.LightningDataModule):
    def __init__(self, path):
        super(PneumoniaDataset, self).__init__()
        self.path = path
        files_0 = [(pth, 0) for pth in glob.glob(os.path.join(path, "NORMAL", "*.jpeg"))]
        files_1 = [(pth, 1) for pth in glob.glob(os.path.join(path, "PNEUMONIA", "*.jpeg"))]
        self.img_files = files_0 + files_1
        
    def __getitem__(self, index):
        img_pth, label = self.img_files[index]
        image = cv.cvtColor(cv.imread(img_pth), cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (224, 224), interpolation=cv.INTER_AREA)

        return image, label
    
    def __len__(self):
        return len(self.img_files) 
    

def get_dataloaders(config):
    if config["data"]["dataset"] == "pneumonia":
        train_dataset_path = "../data/pneumonia_dataset/chest_xray/train"
        val_dataset_path = "../data/pneumonia_dataset/chest_xray/val"
        test_dataset_path = "../data/pneumonia_dataset/chest_xray/test"
        
        train_dataset = PneumoniaDataset(train_dataset_path)
        val_dataset = PneumoniaDataset(val_dataset_path)
        test_dataset = PneumoniaDataset(test_dataset_path)

        batch_size = config["data"]["batch_size"]
        val_batch_size = config["data"]["val_batch_size"]


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
        train_size = int(0.8 * len(dataset)) * (config["data"]["percent"] / 100)  # 80% for training
        val_size = len(dataset) - train_size  # 20% for validation
        unused_size = len(dataset) - train_dataset - val_size
        
        train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, unused_size])

        # Create dataloaders
    batch_size = config["data"]["batch_size"]
    val_batch_size = config["data"]["val_batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=config["data"]["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=config["data"]["num_workers"])

    return train_loader, val_loader, test_loader
