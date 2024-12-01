from abc import ABC, abstractmethod

import torch


class SpatialCovariance(ABC):
    def __call__(self, xs, ys):
        points = [torch.tensor([x, y], dtype=torch.float) for x in xs for y in ys]
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
        diff = torch.norm(x - y)
        return self.a * torch.exp(-(diff**2) / (2 * self.l**2))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Visualize RBF kernel
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(ys, [RBFCovariance(1, 1).kernel(x, y) for y in ys], label="RBF(a=1, l=1)")
    plt.plot(ys, [RBFCovariance(1, 2).kernel(x, y) for y in ys], label="RBF(a=1, l=2)")
    plt.plot(ys, [RBFCovariance(2, 1).kernel(x, y) for y in ys], label="RBF(a=2, l=1)")
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
