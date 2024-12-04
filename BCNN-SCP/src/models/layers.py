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
    def __init__(self, in_channels, out_channels, filter_size, priors,
                 stride=1, padding=0, dilation=1, num_samples=1, kernel="RBF"):

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
        self.num_samples = num_samples
        self.kernel = kernel

        # setting up priors
        if (priors["kernel"] == "RBF"):
            prior_kernel = RBFCovariance(priors["kernel_params"][0], priors["kernel_params"][1])
        elif (priors["kernel"] == "Matern"):
            prior_kernel = MaternCovariance(priors["kernel_params"][0], priors["kernel_params"][1], priors["kernel_params"][2])
        elif (priors["kernel"] == "RQC"):
            prior_kernel = RationalQuadraticCovariance(priors["kernel_params"][0], priors["kernel_params"][1], priors["kernel_params"][2])
        else:
            raise NotImplementedError

        # prior mean and convariance
        self.prior_mu = torch.zeros([]) # shape: ()
        self.prior_sigma = prior_kernel(self.filter_shape[0], self.filter_shape[1]) # shape: (filter_size, filter_size)

        # precomputing inverse and logdet for KL divergence
        self.prior_sigma_inv = torch.linalg.inv(self.prior_sigma)
        self.prior_sigma_logdet = torch.logdet(self.prior_sigma)

        # setting up variational posteriors
        if (kernel == "RBF"):
            self.a = nn.Parameter(uniform(1, 5, self.filter_num))
            self.l = nn.Parameter(uniform(0.1, 1, self.filter_num))
        elif (kernel == "Matern"):
            self.a = nn.Parameter(uniform(1, 5, self.filter_num))
            self.l = nn.Parameter(uniform(0.1, 1, self.filter_num))
            self.nu = nn.Parameter(uniform(0.1, 4, self.filter_num))
        elif (kernel == "RQC"):
            self.a = nn.Parameter(uniform(1, 5, self.filter_num))
            self.l = nn.Parameter(uniform(0.1, 1, self.filter_num))
            self.alpha = nn.Parameter(uniform(0.1, 4, self.filter_num))
        else:
            raise NotImplementedError

        # variational mean and covariance
        self.W_mu = nn.Parameter(torch.randn(self.filter_num, self.filter_size)) # shape: (filter_num, filter_size)
        self.W_sigma = None # computed in sample_weights, shape: (filter_num, filter_size, filter_size)

        # sampled weights
        self.sampled_weights = None # sampled in sample_weights, shape: (num_samples, filter_num, filter_size)

    def forward(self, input, sample=True):
        # (B,S,C,H,W)
        if self.training:
            # sample weights in self.sampled_weights, shape: (num_samples, filter_num, filter_size)
            self.sample_weights()

            # now we have sampled "num_samples" at once
            # we iterate through the S dimension and run each set of weights for each sample
            # it is guaranteed that self.num_samples matches with S (except for first layer, where input will have S = 1)
            S = input.shape[1]
            outputs = []
            num_iters = self.num_samples if self.training else 1
            for i in range(num_iters):
                weight = self.sampled_weights[i].view(self.out_channels, self.in_channels, self.filter_shape[0], self.filter_shape[1])

                input_index = 0 if S == 1 else i
                outputs.append(F.conv2d(input[:,input_index,:,:,:], weight, None, self.stride, self.padding, self.dilation, self.groups))
            return torch.stack(outputs, dim=1)
        else:
            weight = self.W_mu.view(self.out_channels, self.in_channels, self.filter_shape[0], self.filter_shape[1])
            return F.conv2d(input[:,0,:,:,:], weight, None, self.stride, self.padding, self.dilation, self.groups).unsqueeze(1)

    def kl_loss(self):
        self.prior_mu = self.prior_mu.to(self.device)
        self.prior_sigma_inv = self.prior_sigma_inv.to(self.device)
        self.prior_sigma_logdet = self.prior_sigma_logdet.to(self.device)
        return KL_DIV(self.prior_mu, self.prior_sigma_inv, self.prior_sigma_logdet, self.W_mu, self.W_sigma)

    def sample_weights(self):
        if (self.kernel == "RBF"):
            posterior_kernel = RBFCovariance(self.a, self.l)
        elif (self.kernel == "Matern"):
            posterior_kernel = MaternCovariance(self.a, self.l, self.nu)
        elif (self.kernel == "RQC"):
            posterior_kernel = RationalQuadraticCovariance(self.a, self.l, self.alpha)

        self.W_sigma = posterior_kernel(self.filter_shape[0], self.filter_shape[1], device=self.device) # shape: (filter_num, filter_size, filter_size)
        L = torch.linalg.cholesky(self.W_sigma) # shape: (filter_num, filter_size, filter_size)
        noise = torch.randn((self.num_samples, self.filter_num, self.filter_size), device=self.device) # shape: (num_samples, filter_num, filter_size)
        self.sampled_weights = self.W_mu + torch.einsum("fij,sfj->sfi", L, noise) # shape: (num_samples, filter_num, filter_size)
