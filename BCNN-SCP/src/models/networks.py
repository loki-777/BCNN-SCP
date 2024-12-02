import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.models.layers import *

class BCNN(pl.LightningModule):
    def __init__(self, num_samples=1, kernel="RBF"):
        super(BCNN, self).__init__()
        self.conv1 = BBBConv2d(1, 32, filter_size=3, stride=1, padding=1, num_samples=num_samples, kernel=kernel)
        self.conv2 = BBBConv2d(32, 64, filter_size=3, stride=1, padding=1, num_samples=num_samples, kernel=kernel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.num_samples = num_samples

    def forward(self, x_in):
        x = x_in.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        y=[]
        num_iters = self.num_samples if self.training else 1
        for i in range(num_iters):
            y.append(self.pool(x[:,i,:,:,:]))
        y = torch.stack(y, dim=1)
        y = y.view(y.size(0), y.size(1), -1)  # Flatten
        y = torch.relu(self.fc1(y))
        y = self.fc2(y)

        kl = 0.0
        for module in self.children():
            if hasattr(module, 'kl_loss'):
                module_kl_loss = module.kl_loss()
                kl = kl + module_kl_loss

        return {
            "logits": y,
            "kl_loss": kl
        }

class CNN(pl.LightningModule):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return {
            "logits": x,
            "kl_loss": None
        }