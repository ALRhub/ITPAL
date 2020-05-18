from util.Gaussian import Gaussian
import numpy as np
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection
import cpp_projection as projection
import time as t

"""
IMPORTANT: This currently only checks if cpp and python produce the same result, there is no check
if thats the correct one!
benchmark and regression test for backward pass"""

np.random.seed(0)
run_cpp = True  # whether to run cpp implementation
run_py = True   # whether to run python implementation
regression_test = True #whether to run the regression tests

num_gaussians = 128
dim = 100

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))

d_means = np.random.uniform(-1, 1, [num_gaussians, dim])
d_covs = np.stack([sample_sympd(dim) for _ in range(num_gaussians)])

eps = 0.1
beta_loss = 0.05

if run_cpp:
    mp_cpp = projection.BatchedProjection(num_gaussians, dim)
    projeted_dists_cpp = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)
    means, covs = mp_cpp.forward(epss, betas, old_means, old_covs, target_means, target_covs)
    t0 = t.time()
    cpp_d_means, cpp_d_covs = mp_cpp.backward(d_means, d_covs)
    print("cpp", t.time() - t0)

if run_py:
    mp_py = MoreProjection(dim)
    py_d_means = []
    py_d_covs = []

    t_accu = 0
    for i in range(num_gaussians):
        mean, cov = mp_py.more_step(eps, old_dists[i].entropy() - beta_loss,
                                    old_dists[i].mean, old_dists[i].covar,
                                    target_dists[i].mean, target_dists[i].covar)
        t0 = t.time()
        m, c = mp_py.backward(d_means[i], d_covs[i])
        t_accu += t.time() - t0
        py_d_means.append(m)
        py_d_covs.append(c)

    py_d_means = np.stack(py_d_means, axis=0)
    py_d_covs = np.stack(py_d_covs, axis=0)
    print("py", t_accu)

if regression_test:
    for i in range(num_gaussians):
        print("-----------Distribution:", i, "------------")
        print("max dmean diff", np.max(np.abs(py_d_means[i] - cpp_d_means[i])))
        print("max dcov diff", np.max(np.abs(py_d_covs[i] - cpp_d_covs[i])))




