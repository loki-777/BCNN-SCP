from abc import ABC, abstractmethod
import torch
from scipy.special import gamma, kv
import math
import matplotlib.pyplot as plt

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
        diff = torch.linalg.vector_norm(x - y)
        return self.a * torch.exp(-(diff**2) / (2 * self.l**2))
    
class MaternCovariance(SpatialCovariance):
    def __init__(self, a, l, nu):
        self.a = a
        self.l = l
        self.nu = nu

    def kernel(self, x, y):
        diff = torch.linalg.vector_norm(x - y)
        return self.a**2 * (2**(1-self.nu) / gamma(self.nu)) * (math.sqrt(2*self.nu)*diff / self.l)**self.nu * kv(self.nu, diff / self.l)
    
class RationalQuadraticCovariance(SpatialCovariance):
    def __init__(self, a, l, alpha):
        self.a = a
        self.l = l
        self.alpha = alpha

    def kernel(self, x, y):
        diff = torch.linalg.vector_norm(x - y)
        return self.a**2 * (1 + diff**2 / (2 * self.alpha * self.l**2))**(-self.alpha)


if __name__ == "__main__":
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

    # Visualize Matern kernel
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(ys, [MaternCovariance(1, 1, 3/2).kernel(x, y) for y in ys], label="Matern(a=1, l=1, nu=3/2)")
    plt.plot(ys, [MaternCovariance(1, 2, 3/2).kernel(x, y) for y in ys], label="Matern(a=1, l=2, nu=3/2)")
    plt.plot(ys, [MaternCovariance(2, 1, 3/2).kernel(x, y) for y in ys], label="Matern(a=2, l=1, nu=3/2)")
    plt.plot(ys, [MaternCovariance(1, 1, 5/2).kernel(x, y) for y in ys], label="Matern(a=1, l=1, nu=5/2)")
    plt.plot(ys, [MaternCovariance(1, 2, 5/2).kernel(x, y) for y in ys], label="Matern(a=1, l=2, nu=5/2)")
    plt.plot(ys, [MaternCovariance(2, 1, 5/2).kernel(x, y) for y in ys], label="Matern(a=2, l=1, nu=5/2)")
    plt.xlabel("|x - y|")
    plt.legend()
    plt.show()

    # Visualize Matern covariance matrix
    matern = MaternCovariance(1, 1, 3/2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Matern covariance matrix
    matern = MaternCovariance(1, 1, 5/2)
    xs = torch.tensor([0, 1, 2])
    ys = torch.tensor([0, 1, 2])
    covar = matern(xs, ys)
    plt.imshow(covar)
    plt.show()

    # Visualize Rational Quadratic(RQ) kernel 
    x = torch.tensor([0])
    ys = torch.linspace(-5, 5, 100)
    plt.plot(ys, [RationalQuadraticCovariance(1, 1, 1).kernel(x, y) for y in ys], label="RQ(a=1, l=1, nu=3/2)")
    plt.plot(ys, [RationalQuadraticCovariance(1, 2, 1).kernel(x, y) for y in ys], label="RQ(a=1, l=2, nu=3/2)")
    plt.plot(ys, [RationalQuadraticCovariance(2, 1, 2).kernel(x, y) for y in ys], label="RQ(a=2, l=1, nu=3/2)")
    plt.plot(ys, [RationalQuadraticCovariance(1, 1, 2).kernel(x, y) for y in ys], label="RQ(a=1, l=1, nu=5/2)")
    plt.plot(ys, [RationalQuadraticCovariance(1, 2, 2).kernel(x, y) for y in ys], label="RQ(a=1, l=2, nu=5/2)")
    plt.plot(ys, [RationalQuadraticCovariance(2, 1, 2).kernel(x, y) for y in ys], label="RQ(a=2, l=1, nu=5/2)")
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