from abc import ABC, abstractmethod

import torch


class SpatialCovariance(ABC):
    def __call__(self, xs, ys):
        points = torch.cartesian_prod(xs, ys)
        covar = torch.zeros(len(points), len(points))
        for i in range(len(points)):
            for j in range(i, len(points)):
                x = points[i]
                y = points[j]
                covar[i, j] = self.kernel(x, y)
                covar[j, i] = covar[i, j]
        return covar

    @abstractmethod
    def kernel(self, x, y):
        raise NotImplementedError


class RBFCovariance(SpatialCovariance):
    def __init__(self, a, l):
        self.a = a
        self.l = l

    def kernel(self, x, y):
        dist = torch.linalg.vector_norm(x - y)
        return self.a * torch.exp(-(dist**2) / (2 * self.l**2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Visualize RBF kernel
    xs = torch.linspace(-5, 5, 100)
    for a, l in [(1, 1), (1, 2), (2, 1)]:
        rbf = RBFCovariance(a, l)
        plt.plot(xs, [rbf.kernel(x, 0.0) for x in xs], label=f"RBF(a={a}, l={l})")
    plt.xlabel("|x - y|")
    plt.legend()
    plt.show()

    # Visualize RBF covariance matrix
    rbf = RBFCovariance(1, 1)
    xs = torch.tensor([0.0, 1.0, 2.0])
    ys = torch.tensor([0.0, 1.0, 2.0])
    covar = rbf(xs, ys)
    plt.imshow(covar)
    plt.show()
