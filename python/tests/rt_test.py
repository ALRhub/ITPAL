from util.Gaussian import Gaussian
import numpy as np
from util.sample import sample_sympd

from cvx.cvxlayers_projection import CVXProjector
from projection.MoreProjection import MoreProjection
from projection.pytorch_op import ProjectionSimpleNetCppBatched, ProjectionSimpleNetPyBatched
import cpp_projection as projection
import tensorflow as tf
import time as t
import torch

"""benchmark and regression test for forward pass"""

np.random.seed(0)
run_cpp = True  # whether to run cpp implementation
run_py = True   # whether to run python implementation
run_cvx = False  # whether to run cvx layers implementation (this can be very slow for large batch sizes and dims)

run_cpp_torch = False # whether to run pytorch with cpp implementation layers
run_py_torch = False # whether to run pytorch with python implementation layers

regression_test = False #whether to run the regression tests

num_gaussians = 512
dim = 24

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))


eps = 100.0
beta_loss = 10.0

mp_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=True)
diag_cov_only_cpp = projection.BatchedDiagCovOnlyProjection(num_gaussians, dim)

projeted_dists_cpp = []

old_means = np.stack([od.mean for od in old_dists])
old_covs = np.stack([od.covar for od in old_dists])
target_means = np.stack([td.mean for td in target_dists])
target_covs = np.stack([td.covar for td in target_dists])
betas = np.array([od.entropy() - beta_loss for od in old_dists])
epss = eps * np.ones(num_gaussians)

d_means = np.zeros(old_means.shape)
d_covs = np.zeros(old_covs.shape)

fwd_times = []
bwd_times = []

for i in range(10):
    t0 = t.time()
    means, covs = mp_cpp.forward(epss, betas, old_means, old_covs, target_means, target_covs)
    fwd_times.append(t.time() - t0)

    t0 = t.time()
    cpp_d_means, cpp_d_covs = mp_cpp.backward(d_means, d_covs)
    bwd_times.append(t.time() - t0)



print(np.mean(fwd_times), np.std(fwd_times))
print(np.mean(bwd_times), np.std(bwd_times))

fwd_times = []
bwd_times = []

old_covs = np.ascontiguousarray(np.diagonal(old_covs, axis2=-1, axis1=-2))
target_covs = np.ascontiguousarray(np.diagonal(target_covs, axis2=-1, axis1=-2))
d_covs = np.zeros(old_covs.shape)

for i in range(10):
    t0 = t.time()
    covs = diag_cov_only_cpp.forward(epss, old_covs, target_covs)
    fwd_times.append(t.time() - t0)

    t0 = t.time()
    cpp_d_covs = diag_cov_only_cpp.backward(d_covs)
    bwd_times.append(t.time() - t0)

print(np.mean(fwd_times), np.std(fwd_times))
print(np.mean(bwd_times), np.std(bwd_times))