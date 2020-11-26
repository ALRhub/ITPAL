import numpy as np
from util.Gaussian import Gaussian
import nlopt
import matplotlib.pyplot as plt
from plots.plot_util import draw_2d_covariance, build_to_cov

class FwdKLProjection:

    def __init__(self):
        self._eta = None
        self._grad = None
        self._succ = False
        self._eps = None
        self._old_dist = None
        self._target_dist = None
        self._proj_mean = None
        self._proj_covar = None

    def project(self, eps, old_mean, old_covar, target_mean, target_covar):
        self._eps = eps
        self._succ = False

        self._old_dist = Gaussian(old_mean, old_covar)
        self._target_dist = Gaussian(target_mean, target_covar)

        opt = nlopt.opt(nlopt.LD_LBFGS, 1)
        opt.set_lower_bounds(0.0)
        opt.set_upper_bounds(1e12)
        opt.set_ftol_abs(1e-12)
        opt.set_xtol_abs(1e-12)
        opt.set_maxeval(10000)
        opt.set_min_objective(self._dual)
        opt_eta = opt.optimize([10.0])
        opt_eta = opt_eta[0]
        self._proj_mean, self._proj_covar = self._new_params(opt_eta)
        self._succ = True
        return self._proj_mean, self._proj_covar

    def _new_params(self, eta):
        mean = (self._target_dist.mean + eta * self._old_dist.mean) / (eta + 1)
        target_outer = np.outer(mean - self._target_dist.mean, mean - self._target_dist.mean)
        old_outer = np.outer(mean - self._old_dist.mean, mean - self._old_dist.mean)
        cov = (self._target_dist.covar + target_outer + eta * (old_outer + self._old_dist.covar)) / (eta + 1)
        return mean, cov

    def _dual(self, eta, grad):
        """
        dual of the more problem
        """
        eta = eta[0] if eta[0] > 0.0 else 0.0
        self._eta = eta
        new_mean, new_cov = self._new_params(eta)
        new_dist = Gaussian(new_mean, new_cov)

        kl_target = self._target_dist.kl(new_dist)
        kl_old = self._old_dist.kl(new_dist)

        # invert sign as dual is maximized
        grad[0] = - (- self._eps + kl_old)
        self._grad = grad
        dual = - (kl_target - self._eta * self._eps + self._eta * kl_old)
        return dual


if __name__ == "__main__":
    import matplotlib

    cmap = matplotlib.cm.get_cmap('winter')

    mean_old = np.zeros(2)
    cov_old = build_to_cov(np.array([5, 0.5]), 0)
    mean_target = 10 * np.ones(2)
    cov_target = build_to_cov(np.array([5, 0.5]), np.pi / 2)
    print("Distance old, target", Gaussian(mean_old, cov_old).kl(Gaussian(mean_target, cov_target)))

    projector = FwdKLProjection()

    #eps = 0.5

    epss = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 150]
    plt.figure()
    plt.title("Forward KL")
    draw_2d_covariance(mean_old, cov_old, chisquare_val=2, c="black")
    draw_2d_covariance(mean_target, cov_target, chisquare_val=2, c="black", linestyle="dashed")
    for i, eps in enumerate(epss):
        mean, cov = projector.project(eps, mean_old, cov_old, mean_target, cov_target)
        print(Gaussian(mean_old, cov_old).kl(Gaussian(mean, cov)))
        draw_2d_covariance(mean, cov, chisquare_val=2, c=cmap(i / len(epss)))
    plt.legend(["old", "target"] + ["eps=" + str(x) for x in epss])
    plt.show()

