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


mp_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=False)
projeted_dists_cpp = []

old_means = np.stack([od.mean for od in old_dists])
old_covs = np.stack([od.covar for od in old_dists])
target_means = np.stack([td.mean for td in target_dists])
target_covs = np.stack([td.covar for td in target_dists])

eps = 0.1

betas = np.nan * np.ones(num_gaussians)
epss = eps * np.ones(num_gaussians)
t0 = t.time()
means, covs = mp_cpp.forward(epss, betas, old_means, old_covs, target_means, target_covs)