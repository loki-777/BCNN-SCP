import math

import matplotlib.pyplot as plt
import torch
from scipy.special import gamma, kv


class SpatialCovariance:
    def __call__(self, w, h):
        xs = torch.linspace(0, 1, w)
        ys = torch.linspace(0, 1, h)
        points = torch.cartesian_prod(xs, ys)
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

    def kernel(self, x, y):
        diff = torch.norm(x - y)
        return self.a * torch.exp(-(diff**2) / (2 * self.l**2))


class MaternCovariance(SpatialCovariance):
    def __init__(self, a, l, nu):
        self.a = a
        self.l = l
        self.nu = nu

    def kernel(self, x, y):
        diff = torch.linalg.vector_norm(x - y)
        return (
            self.a**2
            * (2 ** (1 - self.nu) / gamma(self.nu))
            * (math.sqrt(2 * self.nu) * diff / self.l) ** self.nu
            * kv(self.nu, diff / self.l)
        )


class RationalQuadraticCovariance(SpatialCovariance):
    def __init__(self, a, l, alpha):
        self.a = a
        self.l = l
        self.alpha = alpha

    def kernel(self, x, y):
        diff = torch.linalg.vector_norm(x - y)
        return self.a**2 * (1 + diff**2 / (2 * self.alpha * self.l**2)) ** (-self.alpha)


if __name__ == "__main__":
    # Visualize RBF kernel
    xs = torch.linspace(-5, 5, 100)
    for a, l in [(1, 1), (1, 2), (2, 1)]:
        rbf = RBFCovariance(a, l)
        plt.plot(xs, [rbf.kernel(x, 0.0) for x in xs], label=f"RBF(a={a}, l={l})")
    plt.xlabel("$p_1 - p_2$")
    plt.legend()
    plt.show()

    # Visualize RBF covariance matrix
    rbf = RBFCovariance(1, 1)
    xs = torch.tensor([0.0, 1.0, 2.0])
    ys = torch.tensor([0.0, 1.0, 2.0])
    covar = rbf(xs, ys)
    covar_p1 = covar[0, :].reshape(len(xs), len(ys))
    plt.imshow(covar_p1)
    plt.xticks(range(len(xs)), range(1, len(xs) + 1))
    plt.yticks(range(len(ys)), range(1, len(ys) + 1))
    plt.colorbar()
    plt.savefig("covariance_example.pdf")

    # Visualize Matern kernel
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(
        ys,
        [MaternCovariance(1, 1, 3 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(1, 2, 3 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=2, nu=3/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(2, 1, 3 / 2).kernel(x, y) for y in ys],
        label="Matern(a=2, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(1, 1, 5 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=1, nu=5/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(1, 2, 5 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=2, nu=5/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(2, 1, 5 / 2).kernel(x, y) for y in ys],
        label="Matern(a=2, l=1, nu=5/2)",
    )
    plt.xlabel("|x - y|")
    plt.legend()
    plt.show()

    # Visualize Matern covariance matrix
    matern = MaternCovariance(1, 1, 3 / 2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Matern covariance matrix
    matern = MaternCovariance(1, 1, 5 / 2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Rational Quadratic(RQ) kernel
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 1, 1).kernel(x, y) for y in ys],
        label="RQ(a=1, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 2, 1).kernel(x, y) for y in ys],
        label="RQ(a=1, l=2, nu=3/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(2, 1, 2).kernel(x, y) for y in ys],
        label="RQ(a=2, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 1, 2).kernel(x, y) for y in ys],
        label="RQ(a=1, l=1, nu=5/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 2, 2).kernel(x, y) for y in ys],
        label="RQ(a=1, l=2, nu=5/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(2, 1, 2).kernel(x, y) for y in ys],
        label="RQ(a=2, l=1, nu=5/2)",
    )
    plt.xlabel("|x - y|")
    plt.legend()
    plt.show()

    # Visualize Rational Quadratic(RQ) covariance matrix
    matern = RationalQuadraticCovariance(1, 1, 2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Rational Quadratic(RQ) covariance matrix
    matern = RationalQuadraticCovariance(1, 1, 2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Matern kernel
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(
        ys,
        [MaternCovariance(1, 1, 3 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(1, 2, 3 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=2, nu=3/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(2, 1, 3 / 2).kernel(x, y) for y in ys],
        label="Matern(a=2, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(1, 1, 5 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=1, nu=5/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(1, 2, 5 / 2).kernel(x, y) for y in ys],
        label="Matern(a=1, l=2, nu=5/2)",
    )
    plt.plot(
        ys,
        [MaternCovariance(2, 1, 5 / 2).kernel(x, y) for y in ys],
        label="Matern(a=2, l=1, nu=5/2)",
    )
    plt.xlabel("|x - y|")
    plt.legend()
    plt.show()

    # Visualize Matern covariance matrix
    matern = MaternCovariance(1, 1, 3 / 2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Matern covariance matrix
    matern = MaternCovariance(1, 1, 5 / 2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Rational Quadratic(RQ) kernel
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 1, 1).kernel(x, y) for y in ys],
        label="RQ(a=1, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 2, 1).kernel(x, y) for y in ys],
        label="RQ(a=1, l=2, nu=3/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(2, 1, 2).kernel(x, y) for y in ys],
        label="RQ(a=2, l=1, nu=3/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 1, 2).kernel(x, y) for y in ys],
        label="RQ(a=1, l=1, nu=5/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(1, 2, 2).kernel(x, y) for y in ys],
        label="RQ(a=1, l=2, nu=5/2)",
    )
    plt.plot(
        ys,
        [RationalQuadraticCovariance(2, 1, 2).kernel(x, y) for y in ys],
        label="RQ(a=2, l=1, nu=5/2)",
    )
    plt.xlabel("|x - y|")
    plt.legend()
    plt.show()

    # Visualize RBF covariance matrix
    rbf = RBFCovariance(1, 1)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = rbf(xs, ys)
    plt.imshow(covar)
    plt.show()
