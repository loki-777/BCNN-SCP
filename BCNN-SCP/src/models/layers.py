import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from src.models.kernels import *
from src.models.losses import *

def uniform(a, b, size):
    return a + (b - a) * torch.rand(size)

# priors can be provided as input, if not provided, (1,1) RBF Kernel is used by default
# kernel can be provided, RBF used by default
class BBBConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, filter_size, priors={"kernel": "RBF", "kernel_params": [1, 1]},
                 stride=1, padding=0, dilation=1, kernel="RBF"):

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
        self.kernel = kernel

        # setting up priors
        if (priors["kernel"] == "RBF"):
            prior_kernel = RBFKernel(priors["kernel_params"][0], priors["kernel_params"][1])
        elif (priors["kernel"] == "Matern"):
            prior_kernel = MaternKernel(priors["kernel_params"][0], priors["kernel_params"][1], priors["kernel_params"][2])
        elif (priors["kernel"] == "RQ"):
            prior_kernel = RationalQuadraticKernel(priors["kernel_params"][0], priors["kernel_params"][1], priors["kernel_params"][2])
        elif (priors["kernel"] == "Independent"):
            prior_kernel = IndependentKernel(priors["kernel_params"][0])
        else:
            raise NotImplementedError

        # prior mean and convariance
        self.prior_mu = torch.tensor(0) # shape: ()
        self.prior_sigma = prior_kernel(self.filter_shape[0], self.filter_shape[1]) # shape: (filter_size, filter_size)

        # precomputing inverse and logdet for KL divergence
        self.prior_sigma_inv = torch.linalg.inv(self.prior_sigma)
        self.prior_sigma_logdet = torch.logdet(self.prior_sigma)

        # setting up variational posteriors
        if (kernel == "RBF"):
            self.a = nn.Parameter(uniform(0.1, 0.2, self.filter_num)) # learnable
            self.l = nn.Parameter(uniform(0.1, 0.2, self.filter_num)) # learnable
            self.posterior_kernel = RBFKernel(self.a, self.l)
        elif (kernel == "Matern"):
            self.a = nn.Parameter(uniform(0.1, 0.2, self.filter_num)) # learnable
            self.l = nn.Parameter(uniform(0.1, 0.2, self.filter_num)) # learnable
            self.nu = nn.Parameter(uniform(0.1, 4, self.filter_num)) # learnable
            self.posterior_kernel = MaternKernel(self.a, self.l, self.nu)
        elif (kernel == "RQ"):
            self.a = nn.Parameter(uniform(0.1, 0.2, self.filter_num)) # learnable
            self.l = nn.Parameter(uniform(0.1, 0.2, self.filter_num)) # learnable
            self.alpha = nn.Parameter(uniform(0.1, 4, self.filter_num)) # learnable
            self.posterior_kernel = RationalQuadraticKernel(self.a, self.l, self.alpha)
        elif (kernel == "Independent"):
            self.a = nn.Parameter(uniform(0.1, 0.2, (self.filter_size, self.filter_num))) # learnable
            self.posterior_kernel = IndependentKernel(self.a)
        else:
            raise NotImplementedError

        # variational mean
        self.W_mu = nn.Parameter(torch.randn(self.filter_num, self.filter_size)) # learnable, shape: (filter_num, filter_size)

    # variational covariance
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
            # number of samples
            num_samples = inputs.shape[1]

            # sample weights from W_mu and W_sigma, shape: (num_samples, filter_num, filter_size)
            sampled_weights = self.sample_weights(num_samples)

            # forward for each sample
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
