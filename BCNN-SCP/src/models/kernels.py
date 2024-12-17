import matplotlib.pyplot as plt
import torch


class IndependentKernel:
    def __init__(self, a=1):
        self.a = a  # output scale, scalar or tensor with shape (d, h*w)

    def __call__(self, h, w, device=None):
        covar = torch.eye(h * w, device=device)  # (h*w, h*w)

        a = self.a
        # If a has shape (d, h*w), expand dimension for broadcasting
        if isinstance(a, torch.Tensor) and a.dim() == 2:
            a = a.unsqueeze(-1)  # (d, h*w) -> (d, h*w, 1)

        return a**2 * covar  # (h*w, h*w) or (d, h*w, h*w)


class SpatialKernel:
    def __init__(self, a=1, l=1):
        self.a = a  # output scale, scalar or tensor with shape (d,)
        self.l = l  # length scale, scalar or tensor with shape (d,)

    def __call__(self, h, w, device=None):
        # Create grid of points
        ys = torch.arange(h, dtype=torch.float, device=device)
        xs = torch.arange(w, dtype=torch.float, device=device)
        points = torch.cartesian_prod(ys, xs)  # (h*w, 2)

        # Compute pairwise squared distances
        sq_norms = torch.sum(points**2, dim=1)  # p_i.T @ p_i
        sq_dists = (
            sq_norms.unsqueeze(-1)  # p_i.T @ p_i
            - 2 * torch.matmul(points, points.T)  # -2 * p_i.T @ p_j
            + sq_norms.unsqueeze(0)  # p_j.T @ p_j
        )  # (h*w, h*w)

        # Compute covariance matrix
        # Expand dimension for broadcasting
        sq_dists = sq_dists.unsqueeze(-1)  # (h*w, h*w, 1)
        covar = self.scaled_kernel(sq_dists)  # (h*w, h*w, d)
        covar = covar.permute(2, 0, 1)  # (d, h*w, h*w)
        covar = covar.squeeze(0)  # (d, h*w, h*w) or (h*w, h*w)
        return covar

    def scaled_kernel(self, sq_dists):
        # Scale distances
        sq_dists = sq_dists / self.l**2
        # Compute covariance matrix
        covar = self.kernel(sq_dists)
        # Scale covariance
        return self.a**2 * covar

    def kernel(self, sq_dists):
        raise NotImplementedError


class RBFKernel(SpatialKernel):
    def __init__(self, a=1, l=1):
        super().__init__(a=a, l=l)

    def kernel(self, sq_dists):
        return torch.exp(-sq_dists / 2)


class MaternKernel(SpatialKernel):
    def __init__(self, a=1, l=1, nu=0.5):
        if nu not in {0.5, 1.5, 2.5}:
            raise NotImplementedError
        super().__init__(a=a, l=l)
        self.nu = nu

    def kernel(self, sq_dists):
        dists = sq_dists**0.5

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = 1 + 3**0.5 * dists
        elif self.nu == 2.5:
            constant_component = 1 + 5**0.5 * dists + 5 / 3 * sq_dists

        exp_component = torch.exp(-((2 * self.nu) ** 0.5) * dists)
        return constant_component * exp_component


class RQKernel(SpatialKernel):
    def __init__(self, a=1, l=1, alpha=1):
        super().__init__(a=a, l=l)
        self.alpha = alpha

    def kernel(self, sq_dists):
        return (1 + sq_dists / (2 * self.alpha)) ** (-self.alpha)


if __name__ == "__main__":
    # Increase font size
    plt.rcParams.update({'font.size': 16})

    # Visualize RBF covariance
    h, w = 3, 3
    covar = RBFKernel(a=1, l=1)(h, w)
    covar_p1 = covar[0, :].reshape(h, w)
    plt.imshow(covar_p1, cmap="Reds")
    plt.yticks(range(h))
    plt.xticks(range(w))
    plt.colorbar()
    plt.title("Spatial Covariance of $(0, 0)$")
    plt.savefig("../figures/covariance_example.pdf", bbox_inches="tight")
    plt.close()

    # Visualize RBF kernel
    diff = torch.linspace(-3, 3, 100)
    for a, l in [(1, 1), (1, 2), (2, 1)]:
        covars = RBFKernel(a=a, l=l).scaled_kernel(diff**2)
        plt.plot(diff, covars, label=f"RBF$(a={a}, \\ell={l})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("RBF Kernel")
    plt.savefig("../figures/rbf_kernel.pdf", bbox_inches="tight")
    plt.close()

    # Visualize Matern kernel
    diff = torch.linspace(-3, 3, 100)
    for a, l, nu in [(1, 1, 0.5), (1, 2, 0.5), (2, 1, 0.5), (1, 1, 1.5)]:
        covars = MaternKernel(a=a, l=l, nu=nu).scaled_kernel(diff**2)
        plt.plot(diff, covars, label=f"Matérn$(a={a}, \\ell={l}, \\nu={nu})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Matérn Kernel")
    plt.savefig("../figures/matern_kernel.pdf", bbox_inches="tight")
    plt.close()

    # Visualize RQ kernel
    diff = torch.linspace(-3, 3, 100)
    for a, l, alpha in [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)]:
        covars = RQKernel(a=a, l=l, alpha=alpha).scaled_kernel(diff**2)
        plt.plot(diff, covars, label=f"RQ$(a={a}, \\ell={l}, \\alpha={alpha})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("RQ Kernel")
    plt.savefig("../figures/rq_kernel.pdf", bbox_inches="tight")
    plt.close()

    # Visualize different kernels
    kernels = {
        "RBF$(a=1, \\ell=1)$": RBFKernel(a=1, l=1),
        "Matern$(a=1, \\ell=1, \\nu=0.5)$": MaternKernel(a=1, l=1, nu=0.5),
        "RQ$(a=1, \\ell=1, \\alpha=1)$": RQKernel(a=1, l=1, alpha=1),
    }
    diff = torch.linspace(-3, 3, 100)
    for name, kernel in kernels.items():
        covars = kernel.scaled_kernel(diff**2)
        plt.plot(diff, covars, label=name)
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Kernels")
    plt.savefig("../figures/kernels.pdf", bbox_inches="tight")
    plt.close()
