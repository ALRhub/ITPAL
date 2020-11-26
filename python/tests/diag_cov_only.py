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

num_gaussians = 1024
dim = 100

old_dists = []
target_dists = []
old_vars = []
target_vars = []
for i in range(num_gaussians):
    means = np.random.uniform(low=-0.5, high=0.5, size=dim)
    old_vars.append(np.exp(np.random.uniform(low=-0.5, high=0.5, size=dim)))
    old_dists.append(Gaussian(means, np.diag(old_vars[-1])))
    target_vars.append(np.exp(np.random.uniform(low=-0.5, high=0.5, size=dim)))
    target_dists.append(Gaussian(means, np.diag(target_vars[-1])))


uc_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=False, max_eval=100)
diag_cov_only_cpp = projection.BatchedDiagCovOnlyProjection(num_gaussians, dim, 100)
projected_dists_cpp = []

old_means = np.stack([od.mean for od in old_dists])
old_covs = np.stack([od.covar for od in old_dists])
old_vars = np.stack(old_vars)
target_means = np.stack([td.mean for td in target_dists])
target_covs = np.stack([td.covar for td in target_dists])
target_vars = np.stack(target_vars)
eps = 0.001

uc_betas = np.nan * np.ones(num_gaussians)
epss = eps * np.ones(num_gaussians)

d_means = np.zeros([num_gaussians, dim]) #np.random.normal(size=[num_gaussians, dim])
d_covs_var = np.random.normal(size=[num_gaussians, dim])
d_covs = np.stack([np.diag(d_covs_var[i]) for i in range(num_gaussians)], axis=0)
t0 = t.time()
ref_means, ref_covs = uc_cpp.forward(epss, uc_betas, old_means, old_covs, target_means, target_covs)
print(t.time() - t0)
ref_bw = uc_cpp.backward(d_means, d_covs)

#means = target_means
#covs = []
grad_mean = np.zeros([num_gaussians, dim])
#grad_cov = []
t0 = t.time()
covs = diag_cov_only_cpp.forward(epss, old_vars, target_vars)
print(t.time() - t0)
grad_cov = diag_cov_only_cpp.backward(d_covs_var)
    #covs.append(c)
    #grad_cov.append(gc)

#covs = np.stack(covs, 0)
#grad_cov = np.stack(grad_cov, 0)

print("Forward")
print(np.max(np.abs(target_means - ref_means)))
print(np.max(np.abs(covs - np.stack([np.diag(ref_covs[i]) for i in range(num_gaussians)], 0))))

print("Backward")
print(np.max(np.abs(ref_bw[0] - grad_mean)))
print(np.max(np.abs(np.stack([np.diag(ref_bw[1][i]) for i in range(num_gaussians)], 0) - grad_cov)))