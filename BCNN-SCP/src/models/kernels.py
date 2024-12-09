import matplotlib.pyplot as plt
import torch
from scipy.special import gamma, kv


class SpatialKernel:
    def __call__(self, h, w, device=None):
        ys = torch.linspace(0, 1, h, device=device)
        xs = torch.linspace(0, 1, w, device=device)
        points = torch.cartesian_prod(ys, xs)

        squared_norms = torch.sum(points**2, dim=1) # p_i.T @ p_i
        dists = torch.sqrt(
            squared_norms.unsqueeze(-1) # p_i.T @ p_i
            - 2 * torch.matmul(points, points.T) # -2 * p_i.T @ p_j
            + squared_norms.unsqueeze(0) # p_j.T @ p_j
        )

        # Expand dimension for broadcasting
        dists = dists.unsqueeze(-1) # (h*w, h*w, 1)
        covar = self.kernel(dists) # (h*w, h*w, d)
        covar = covar.permute(2, 0, 1) # (d, h*w, h*w)
        covar = covar.squeeze(0) # (d, h*w, h*w) or (h*w, h*w)

        return covar

    def kernel(self, dists):
        raise NotImplementedError


class RBFKernel(SpatialKernel):
    def __init__(self, a, l):
        self.a = a
        self.l = l

    def kernel(self, dists):
        return self.a**2 * torch.exp(-(dists**2) / (2 * self.l**2))


class MaternKernel(SpatialKernel):
    def __init__(self, a, l, nu):
        self.a = a
        self.l = l
        self.nu = nu

    def kernel(self, dists):
        return (
            self.a**2
            * (2 ** (1 - self.nu) / gamma(self.nu))
            * ((2 * self.nu) ** (1 / 2) * dists / self.l) ** self.nu
            * kv(self.nu, dists / self.l)
        )


class RationalQuadraticKernel(SpatialKernel):
    def __init__(self, a, l, alpha):
        self.a = a
        self.l = l
        self.alpha = alpha

    def kernel(self, dists):
        return self.a**2 * (1 + dists**2 / (2 * self.alpha * self.l**2)) ** (-self.alpha)


class IndependentKernel(SpatialKernel):
    def __init__(self, a):
        self.a = a # () or (h*w, d)

    def kernel(self, dists):
        eye = torch.eye(dists.shape[0], device=dists.device) # (h*w, h*w)
        # Expand dimension for broadcasting
        eye = eye.unsqueeze(-1) # (h*w, h*w, 1)
        return self.a**2 * eye # (h*w, h*w, 1) or (h*w, h*w, d)


if __name__ == "__main__":
    # Visualize RBF covariance
    h, w = 3, 3
    covar = RBFKernel(a=1, l=1)(h, w)
    covar_p1 = covar[0, :].reshape(h, w)
    plt.imshow(covar_p1)
    plt.yticks(range(h))
    plt.xticks(range(w))
    plt.colorbar()
    plt.savefig("figures/covariance_example.pdf")
    plt.close()

    # Visualize RBF kernel
    dists = torch.linspace(-5, 5, 100)
    for a, l in [(1, 1), (1, 2), (2, 1)]:
        covars = RBFKernel(a, l).kernel(dists.abs())
        plt.plot(dists, covars, label=f"RBF$(a={a}, l={l})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("RBF Kernel")
    plt.savefig("figures/rbf_kernel.pdf")
    plt.close()

    # Visualize Matern kernel
    dists = torch.linspace(-5, 5, 100)
    for a, l, nu in [(1, 1, 1.5), (1, 2, 1.5), (2, 1, 1.5), (1, 1, 2.5)]:
        covars = MaternKernel(a, l, nu).kernel(dists.abs())
        plt.plot(dists, covars, label=f"Matérn$(a={a}, l={l}, \\nu={nu})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Matérn Kernel")
    plt.savefig("figures/matern_kernel.pdf")
    plt.close()

    # Visualize Rational Quadratic kernel
    dists = torch.linspace(-5, 5, 100)
    for a, l, alpha in [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)]:
        covars = RationalQuadraticKernel(a, l, alpha).kernel(dists.abs())
        plt.plot(dists, covars, label=f"RQ$(a={a}, l={l}, \\alpha={alpha})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Rational Quadratic Kernel")
    plt.savefig("figures/rational_quadratic_kernel.pdf")
    plt.close()
