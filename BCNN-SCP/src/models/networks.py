import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.models.layers import *

class BCNN(pl.LightningModule):
    def __init__(self,
                 out_channels_conv1=16, out_channels_conv2=32,
                 num_samples_training=None, num_samples_predict=None,
                 prior_kernel=None, kernel=None):
        super(BCNN, self).__init__()

        self.conv1 = BBBConv2d(1, out_channels_conv1, prior_kernel=prior_kernel, kernel=kernel)
        self.conv2 = BBBConv2d(out_channels_conv1, out_channels_conv2, prior_kernel=prior_kernel, kernel=kernel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(out_channels_conv2 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.num_samples_training = num_samples_training
        self.num_samples_predict = num_samples_predict

    def forward(self, x):
        num_samples = self.num_samples_training if self.training else self.num_samples_predict

        # If sampling
        if num_samples is not None:
            x = x.unsqueeze(1) # (B, C, H, W) -> (B, 1, C, H, W)
            x = x.expand(-1, num_samples, -1, -1, -1) # (B, 1, C, H, W) -> (B, S, C, H, W)

            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))

            x = x.view(x.size(0) * num_samples, x.size(2), x.size(3), x.size(4))
            x = self.pool(x)
            x = x.view(x.size(0) // num_samples, num_samples, -1)  # Flatten

            x = torch.relu(self.fc1(x))
            x = self.fc2(x) # (B, S, 10)

        # If not sampling
        else:
            x = torch.relu(self.conv1(x))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten
            x = torch.relu(self.fc1(x))
            x = self.fc2(x) # (B, 10)

        kl = 0.0
        if self.training:
            for module in self.children():
                if hasattr(module, 'kl_loss'):
                    module_kl_loss = module.kl_loss()
                    kl = kl + module_kl_loss

        return {
            "logits": x,
            "kl_loss": kl
        }

class CNN(pl.LightningModule):
    def __init__(self, out_channels_conv1=16, out_channels_conv2=32):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, out_channels_conv1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels_conv1, out_channels_conv2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(out_channels_conv2 * 14 * 14, 128)
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
