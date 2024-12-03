import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

from src.models.kernels import *
from src.models.losses import *

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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.num_samples = num_samples
        self.kernel = kernel

        self.a = None
        self.l = None
        self.w = None
        self.alpha = None

        if (kernel == "RBF"):
            self.a = nn.Parameter(torch.rand(in_channels*out_channels))
            self.l = nn.Parameter(torch.rand(in_channels*out_channels))
            
        elif (kernel == "Matern"):
            self.a = nn.Parameter(torch.rand(in_channels*out_channels))
            self.l = nn.Parameter(torch.rand(in_channels*out_channels))
            self.w = nn.Parameter(torch.rand(in_channels*out_channels))
            
        elif (kernel == "RQC"):
            self.a = nn.Parameter(torch.rand(in_channels*out_channels))
            self.l = nn.Parameter(torch.rand(in_channels*out_channels))
            self.alpha = nn.Parameter(torch.rand(in_channels*out_channels))
            
        else:
            raise NotImplementedError
        
        # setting up priors
        if (priors["kernel"] == "RBF"):
            prior_kernel = RBFCovariance(priors["kernel_params"][0], priors["kernel_params"][1])
        elif (priors["kernel"] == "Matern"):
            prior_kernel = MaternCovariance(priors["kernel_params"][0], priors["kernel_params"][1], priors["kernel_params"][2])
        elif (priors["kernel"] == "RQC"):
            prior_kernel = RationalQuadraticCovariance(priors["kernel_params"][0], priors["kernel_params"][1], priors["kernel_params"][2])

        # prior mean and convariance
        self.prior_mu = torch.zeros(1)
        self.prior_cov = prior_kernel(self.filter_shape[0], self.filter_shape[1])

        self.prior_cov_inv = torch.linalg.inv(self.prior_cov)
        self.prior_cov_logdet = torch.logdet(self.prior_cov)

        self.W_mu = nn.Parameter(torch.rand(in_channels*out_channels*self.filter_size))
        self.sampled_weights = torch.rand((self.num_samples,) + self.W_mu.shape)

    def forward(self, input, sample=True):
        # (B,S,C,H,W)
        if sample:
            # W_mu, a and l are learnable parameters
            self.sample_weights()

            # now we have sampled "num_samples" at once
            # we iterate through the S dimension and run each set of weights for each sample
            # it is guarenteed that self.num_samples matches with S (except for first layer, where input will have S = 1)
            S = input.shape[1]
            outputs = []
            num_iters = self.num_samples if self.training else 1
            for i in range(num_iters):
                weight = self.sampled_weights[i,:].view(self.out_channels, self.in_channels, self.filter_shape[0], self.filter_shape[1]).to(input.device)

                input_index = 0 if S == 1 else i
                outputs.append(F.conv2d(input[:,input_index,:,:,:], weight, None, self.stride, self.padding, self.dilation, self.groups))
            return torch.stack(outputs, dim=1)
        else:
            weight = self.W_mu.view(self.out_channels, self.in_channels, self.filter_shape[0], self.filter_shape[1])
            return F.conv2d(input[:,0,:,:,:], weight, None, self.stride, self.padding, self.dilation, self.groups)

    def kl_loss(self):
        device = self.W_mu.device
        self.prior_mu = self.prior_mu.to(device)
        self.prior_cov_inv = self.prior_cov_inv.to(device)
        self.prior_cov_logdet = self.prior_cov_logdet.to(device)
        return KL_DIV(self.prior_mu, self.prior_cov_inv, self.prior_cov_logdet, self.W_mu, self.W_cov)

    def sample_weights(self):
        blocks = []
        for i in range(self.in_channels*self.out_channels):
            # different a and l for different filter (x,y)
            if (self.kernel == "RBF"):
                posterior_kernel = RBFCovariance(self.a[i], self.l[i])
            elif (self.kernel == "Matern"):
                posterior_kernel = MaternCovariance(self.a[i], self.l[i], self.w[i])
            elif (self.kernel == "RQC"):
                posterior_kernel = RationalQuadraticCovariance(self.a[i], self.l[i], self.alpha[i])

            filterwise_covariance_matrix = posterior_kernel(self.filter_shape[0], self.filter_shape[1], device=self.device)
            jitter = 1e-6 * torch.eye(filterwise_covariance_matrix.size(0), device=filterwise_covariance_matrix.device)
            filterwise_covariance_matrix += jitter

            # samples "num_samples" weights at once
            mvn = torch.distributions.MultivariateNormal(self.W_mu[self.filter_size*i : self.filter_size*i+self.filter_size], filterwise_covariance_matrix)
            self.sampled_weights[:, self.filter_size*i : self.filter_size*i+self.filter_size] = mvn.sample((self.num_samples,))

            blocks.append(filterwise_covariance_matrix)

        self.W_cov = torch.stack(blocks) # concats only the blocks of the block diagonal covariance matrix
