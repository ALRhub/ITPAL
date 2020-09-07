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

num_gaussians = 32
dim = 32

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))


uc_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=False)
c_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=True)
projeted_dists_cpp = []

old_means = np.stack([od.mean for od in old_dists])
old_covs = np.stack([od.covar for od in old_dists])
target_means = np.stack([td.mean for td in target_dists])
target_covs = np.stack([td.covar for td in target_dists])

eps = 0.1

uc_betas = np.nan * np.ones(num_gaussians)
c_betas = - 100 * np.ones(num_gaussians)
epss = eps * np.ones(num_gaussians)
t0 = t.time()
uc_means, uc_covs = uc_cpp.forward(epss, uc_betas, old_means, old_covs, target_means, target_covs)
c_means, c_covs = c_cpp.forward(epss, c_betas, old_means, old_covs, target_means, target_covs)
print("Forward")
print(np.max(np.abs(c_means - uc_means)))
print(np.max(np.abs(c_covs - uc_covs)))

d_means = np.random.normal(size=uc_means.shape)
d_covs = np.random.normal(size=uc_covs.shape)
uc_bw = uc_cpp.backward(d_means, d_covs)
c_bw = c_cpp.backward(d_means, d_covs)
print("Backward")
for i in range(2):
    print(np.max(np.abs(uc_bw[i] - c_bw[i])))