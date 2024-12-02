import matplotlib.pyplot as plt
import torch
from scipy.special import gamma, kv


class SpatialCovariance:
    def __call__(self, h, w):
        ys = torch.linspace(0, 1, h)
        xs = torch.linspace(0, 1, w)
        points = torch.cartesian_prod(ys, xs)
        covar = torch.zeros(len(points), len(points))
        for i in range(len(points)):
            for j in range(i, len(points)):
                p1 = points[i]
                p2 = points[j]
                covar[i, j] = self.kernel(p1, p2)
                covar[j, i] = covar[i, j]
        return covar

    def kernel(self, p1, p2):
        raise NotImplementedError


class RBFCovariance(SpatialCovariance):
    def __init__(self, a, l):
        self.a = a
        self.l = l

    def kernel(self, p1, p2):
        diff = torch.norm(p1 - p2)
        return self.a * torch.exp(-(diff**2) / (2 * self.l**2))


class MaternCovariance(SpatialCovariance):
    def __init__(self, a, l, nu):
        self.a = a
        self.l = l
        self.nu = nu

    def kernel(self, p1, p2):
        diff = torch.linalg.vector_norm(p1 - p2)
        return (
            self.a**2
            * (2 ** (1 - self.nu) / gamma(self.nu))
            * ((2 * self.nu) ** (1 / 2) * diff / self.l) ** self.nu
            * kv(self.nu, diff / self.l)
        )


class RationalQuadraticCovariance(SpatialCovariance):
    def __init__(self, a, l, alpha):
        self.a = a
        self.l = l
        self.alpha = alpha

    def kernel(self, p1, p2):
        diff = torch.linalg.vector_norm(p1 - p2)
        return self.a**2 * (1 + diff**2 / (2 * self.alpha * self.l**2)) ** (-self.alpha)


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
    ps = torch.linspace(-5, 5, 100)
    for a, l in [(1, 1), (1, 2), (2, 1)]:
        plt.plot(
            ps,
            [RBFCovariance(a, l).kernel(p, 0.0) for p in ps],
            label=f"RBF$(a={a}, l={l})$",
        )
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("RBF Kernel")
    plt.savefig("figures/rbf_kernel.pdf")
    plt.close()

    # Visualize Matern kernel
    ps = torch.linspace(-5, 5, 100)
    for a, l, nu in [(1, 1, 1.5), (1, 2, 1.5), (2, 1, 1.5), (1, 1, 2.5)]:
        plt.plot(
            ps,
            [MaternCovariance(a, l, nu).kernel(p, 0.0) for p in ps],
            label=f"Matérn$(a={a}, l={l}, \\nu={nu})$",
        )
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Matérn Kernel")
    plt.savefig("figures/matern_kernel.pdf")
    plt.close()

    # Visualize Rational Quadratic kernel
    ps = torch.linspace(-5, 5, 100)
    for a, l, alpha in [(1, 1, 1), (1, 2, 1), (2, 1, 1), (1, 1, 2)]:
        plt.plot(
            ps,
            [RationalQuadraticCovariance(a, l, alpha).kernel(p, 0.0) for p in ps],
            label=f"RQ$(a={a}, l={l}, \\alpha={alpha})$",
        )
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.title("Rational Quadratic Kernel")
    plt.savefig("figures/rational_quadratic_kernel.pdf")
    plt.close()