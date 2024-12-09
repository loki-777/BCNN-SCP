import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from src.models.kernels import *
from src.models.losses import *

# For parameter initialization
def log_normal(init_params, size):
    mu, sigma, x_min, x_max = init_params
    z = torch.randn(size)
    x = torch.exp(mu + sigma * z)
    x = x.clamp(x_min, x_max)
    return x

class BBBConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=1, dilation=1,
                 prior_kernel=None, prior_kernel_params=None,
                 kernel=None, kernel_init=None):

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

        # Default kernel
        if prior_kernel is None:
            prior_kernel = "Independent"
        if prior_kernel_params is None:
            prior_kernel_params = [1, 1, 1]

        if kernel is None:
            kernel = "Independent"
        if kernel_init is None:
            kernel_init = [-1, 0.5, 0.1, 1.0]

        # Setting up priors
        if (prior_kernel == "RBF"):
            prior_kernel = RBFKernel(prior_kernel_params[0], prior_kernel_params[1])
        elif (prior_kernel == "Matern"):
            prior_kernel = MaternKernel(prior_kernel_params[0], prior_kernel_params[1], prior_kernel_params[2])
        elif (prior_kernel == "RQ"):
            prior_kernel = RationalQuadraticKernel(prior_kernel_params[0], prior_kernel_params[1], prior_kernel_params[2])
        elif (prior_kernel == "Independent"):
            prior_kernel = IndependentKernel(prior_kernel_params[0])
        else:
            raise NotImplementedError

        # Prior mean and convariance
        self.prior_mu = torch.tensor(0) # shape: ()
        self.prior_sigma = prior_kernel(self.filter_shape[0], self.filter_shape[1]) # shape: (filter_size, filter_size)

        # Precomputing inverse and logdet for KL divergence
        self.prior_sigma_inv = torch.linalg.inv(self.prior_sigma)
        self.prior_sigma_logdet = torch.logdet(self.prior_sigma)

        # Setting up variational posteriors
        if (kernel == "RBF"):
            self.a = nn.Parameter(log_normal(kernel_init[0], size=self.filter_num)) # learnable
            self.l = nn.Parameter(log_normal(kernel_init[1], size=self.filter_num)) # learnable
            self.posterior_kernel = RBFKernel(self.a, self.l)
        elif (kernel == "Matern"):
            self.a = nn.Parameter(log_normal(kernel_init[0], size=self.filter_num)) # learnable
            self.l = nn.Parameter(log_normal(kernel_init[1], size=self.filter_num)) # learnable
            self.nu = prior_kernel_params[2] # use prior
            self.posterior_kernel = MaternKernel(self.a, self.l, self.nu)
        elif (kernel == "RQC"):
            self.a = nn.Parameter(log_normal(kernel_init[0], size=self.filter_num)) # learnable
            self.l = nn.Parameter(log_normal(kernel_init[1], size=self.filter_num)) # learnable
            self.alpha = prior_kernel_params[2] # use prior
            self.posterior_kernel = RationalQuadraticKernel(self.a, self.l, self.alpha)
        elif (kernel == "Independent"):
            self.a = nn.Parameter(log_normal(kernel_init[0], size=(self.filter_size, self.filter_num))) # learnable
            self.posterior_kernel = IndependentKernel(self.a)
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
