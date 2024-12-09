import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from src.models.kernels import *
from src.models.losses import *

# For parameter initialization
def log_normal(size, mu=0, sigma=0.5, min=0.1, max=3):
    z = torch.randn(size)
    x = torch.exp(mu + sigma * z)
    x = x.clamp(min, max)
    return x

class BBBConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1, padding=1, dilation=1,
                 prior_kernel=None, kernel=None):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_shape = filter_size if isinstance(filter_size, tuple) else (filter_size, filter_size)
        self.filter_size = self.filter_shape[0] * self.filter_shape[1]
        self.filter_num = in_channels * out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1

        # Setting up priors
        if prior_kernel is None:
            self.prior_kernel = IndependentKernel()
        elif prior_kernel["name"] == "Independent":
            self.prior_kernel = IndependentKernel(**prior_kernel["params"])
        elif prior_kernel["name"] == "RBF":
            self.prior_kernel = RBFKernel(**prior_kernel["params"])
        elif prior_kernel["name"] == "Matern":
            self.prior_kernel = MaternKernel(**prior_kernel["params"])
        elif prior_kernel["name"] == "RQ":
            self.prior_kernel = RationalQuadraticKernel(**prior_kernel["params"])
        else:
            raise NotImplementedError

        # Prior mean and convariance
        self.prior_mu = torch.tensor(0) # shape: ()
        self.prior_sigma = self.prior_kernel(self.filter_shape[0], self.filter_shape[1]) # shape: (filter_size, filter_size)

        # Precomputing inverse and logdet for KL divergence
        self.prior_sigma_inv = torch.linalg.inv(self.prior_sigma)
        self.prior_sigma_logdet = torch.logdet(self.prior_sigma)

        # Setting up variational posteriors
        if kernel is None:
            self.a = nn.Parameter(log_normal((self.filter_num, self.filter_size))) # learnable
        if kernel["name"] == "Independent":
            self.a = nn.Parameter(log_normal((self.filter_num, self.filter_size), **kernel["params_init"]["a"])) # learnable
            self.posterior_kernel = IndependentKernel(self.a)
        elif kernel["name"] == "RBF":
            self.a = nn.Parameter(log_normal(self.filter_num, **kernel["params_init"]["a"])) # learnable
            self.l = nn.Parameter(log_normal(self.filter_num, **kernel["params_init"]["l"])) # learnable
            self.posterior_kernel = RBFKernel(self.a, self.l)
        elif kernel["name"] == "Matern":
            self.a = nn.Parameter(log_normal(self.filter_num, **kernel["params_init"]["a"])) # learnable
            self.l = nn.Parameter(log_normal(self.filter_num, **kernel["params_init"]["l"])) # learnable
            self.nu = self.prior_kernel.nu # use prior
            self.posterior_kernel = MaternKernel(self.a, self.l, self.nu)
        elif kernel["name"] == "RQ":
            self.a = nn.Parameter(log_normal(self.filter_num, **kernel["params_init"]["a"])) # learnable
            self.l = nn.Parameter(log_normal(self.filter_num, **kernel["params_init"]["l"])) # learnable
            self.alpha = self.prior_kernel.alpha # use prior
            self.posterior_kernel = RationalQuadraticKernel(self.a, self.l, self.alpha)
        else:
            raise NotImplementedError

        # Variational mean
        self.W_mu = nn.Parameter(torch.randn((self.filter_num, self.filter_size))) # learnable, shape: (filter_num, filter_size)

    # Variational covariance
    @property
    def W_sigma(self):
        return self.posterior_kernel(self.filter_shape[0], self.filter_shape[1], device=self.device) # shape: (filter_num, filter_size, filter_size)

    def to(self, device):
        super().to(device)
        self.prior_mu = self.prior_mu.to(self.device)
        self.prior_sigma_inv = self.prior_sigma_inv.to(self.device)
        self.prior_sigma_logdet = self.prior_sigma_logdet.to(self.device)

    def forward(self, inputs):
        # If sampling, sample weights and forward for each sample
        # (B,S,C,H,W)
        if inputs.dim() == 5:
            # Number of samples
            num_samples = inputs.shape[1]

            # Sample weights from W_mu and W_sigma, shape: (num_samples, filter_num, filter_size)
            sampled_weights = self.sample_weights(num_samples)

            # Forward for each sample
            outputs = []
            for s in range(num_samples):
                weight = sampled_weights[s].view(self.out_channels, self.in_channels, self.filter_shape[0], self.filter_shape[1])
                outputs.append(F.conv2d(inputs[:,s,:,:,:], weight, None, self.stride, self.padding, self.dilation, self.groups))
            return torch.stack(outputs, dim=1)

        # If not sampling, use the mean weights
        # (B,C,H,W)
        else:
            weight = self.W_mu.view(self.out_channels, self.in_channels, self.filter_shape[0], self.filter_shape[1])
            return F.conv2d(inputs, weight, None, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        return KL_DIV(self.prior_mu, self.prior_sigma_inv, self.prior_sigma_logdet, self.W_mu, self.W_sigma)

    def sample_weights(self, num_samples):
        L = torch.linalg.cholesky(self.W_sigma) # shape: (filter_num, filter_size, filter_size)
        noise = torch.randn((num_samples, self.filter_num, self.filter_size), device=self.device) # shape: (num_samples, filter_num, filter_size)
        sampled_weights = self.W_mu + torch.einsum("fij,sfj->sfi", L, noise) # shape: (num_samples, filter_num, filter_size)
        return sampled_weights
