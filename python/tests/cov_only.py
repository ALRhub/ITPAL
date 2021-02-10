import numpy as np

import cpp_projection as projection
import time as t
from util.sample import sample_sympd

from python.util.Gaussian import Gaussian

num_gaussians = 32
dim = 5

old_dists = []
target_dists = []
old_covs = []
target_covs = []
for i in range(num_gaussians):
    means = np.random.uniform(low=-0.5, high=0.5, size=dim)
    old_covs.append(sample_sympd(dim))
    old_dists.append(Gaussian(means, old_covs[-1]))
    target_covs.append(sample_sympd(dim))
    target_dists.append(Gaussian(means, target_covs[-1]))

uc_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=False, max_eval=100)
cov_only_cpp = projection.BatchedCovOnlyProjection(num_gaussians, dim, 100)
projected_dists_cpp = []

old_means = np.stack([od.mean for od in old_dists])
old_covs = np.stack([od.covar for od in old_dists])
target_means = np.stack([td.mean for td in target_dists])
target_covs = np.stack([td.covar for td in target_dists])
eps = 0.001

uc_betas = np.nan * np.ones(num_gaussians)
epss = eps * np.ones(num_gaussians)

d_means = np.zeros([num_gaussians, dim])  # np.random.normal(size=[num_gaussians, dim])
d_covs_var = np.random.normal(size=[num_gaussians, dim, dim])
t0 = t.time()
ref_means, ref_covs = uc_cpp.forward(epss, uc_betas, old_means, old_covs, target_means, target_covs)
print(t.time() - t0)
ref_bw = uc_cpp.backward(d_means, d_covs_var)
# means = target_means
# covs = []
grad_mean = np.zeros([num_gaussians, dim])
# grad_cov = []
t0 = t.time()
covs = cov_only_cpp.forward(epss, old_covs, target_covs)
print(t.time() - t0)
grad_cov = cov_only_cpp.backward(d_covs_var)
# covs.append(c)
# grad_cov.append(gc)

# covs = np.stack(covs, 0)
# grad_cov = np.stack(grad_cov, 0)

print("Forward")
print(np.max(np.abs(target_means - ref_means)))
print(np.max(np.abs(covs - ref_covs)))

print("Backward")
print(np.max(np.abs(ref_bw[0] - grad_mean)))
print(np.max(np.abs(ref_bw[1] - grad_cov)))
