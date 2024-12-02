import matplotlib.pyplot as plt
import torch
from scipy.special import gamma, kv


class SpatialCovariance:
    def __call__(self, h, w, device=None):
        ys = torch.linspace(0, 1, h)
        xs = torch.linspace(0, 1, w)
        points = torch.cartesian_prod(ys, xs)

        squared_norms = torch.sum(points**2, dim=1)
        dists = torch.sqrt(
            squared_norms.unsqueeze(-1)
            - 2 * torch.matmul(points, points.T)
            + squared_norms.unsqueeze(0)
        )

        if device is not None:
            dists = dists.to(device)

        covar = self.kernel(dists)
        return covar

    def kernel(self, dists):
        raise NotImplementedError


class RBFCovariance(SpatialCovariance):
    def __init__(self, a, l):
        self.a = a
        self.l = l

    def kernel(self, dists):
        return self.a**2 * torch.exp(-(dists**2) / (2 * self.l**2))


class MaternCovariance(SpatialCovariance):
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


class RationalQuadraticCovariance(SpatialCovariance):
    def __init__(self, a, l, alpha):
        self.a = a
        self.l = l
        self.alpha = alpha

    def kernel(self, dists):
        return self.a**2 * (1 + dists**2 / (2 * self.alpha * self.l**2)) ** (
            -self.alpha
        )


if __name__ == "__main__":
    # Visualize RBF covariance matrix
    h, w = 3, 3
    covar = RBFCovariance(a=1, l=1)(h, w)
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
        covars = RBFCovariance(a, l).kernel(dists.abs())
        plt.plot(dists, covars, label=f"RBF$(a={a}, l={l})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("RBF Kernel")
    plt.savefig("figures/rbf_kernel.pdf")
    plt.close()

    # Visualize Matern kernel
    dists = torch.linspace(-5, 5, 100)
    for a, l, nu in [(1, 1, 1.5), (1, 2, 1.5), (2, 1, 1.5), (1, 1, 2.5)]:
        covars = MaternCovariance(a, l, nu).kernel(dists.abs())
        plt.plot(dists, covars, label=f"Matérn$(a={a}, l={l}, \\nu={nu})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Matérn Kernel")
    plt.savefig("figures/matern_kernel.pdf")
    plt.close()

    # Visualize Rational Quadratic kernel
    dists = torch.linspace(-5, 5, 100)
    for a, l, alpha in [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)]:
        covars = RationalQuadraticCovariance(a, l, alpha).kernel(dists.abs())
        plt.plot(dists, covars, label=f"RQ$(a={a}, l={l}, \\alpha={alpha})$")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Rational Quadratic Kernel")
    plt.savefig("figures/rational_quadratic_kernel.pdf")
    plt.close()
