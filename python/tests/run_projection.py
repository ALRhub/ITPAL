from util.Gaussian import Gaussian
import numpy as np
from util.sample import sample_sympd

from projection.MoreProjection import MoreProjection
import cpp_projection as projection
import time as t

"""benchmark and regression test python and cpp versions"""

np.random.seed(0)


num_gaussians = 1024
dim = 100

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))

mp_cpp = projection.BatchedProjection(num_gaussians, dim)
mp_py = MoreProjection(dim)
eps = 0.1
beta_loss = 0.05

projeted_dists_cpp = []
projeted_dists_py = []

old_means = np.stack([od.mean for od in old_dists])
old_covs = np.stack([od.covar for od in old_dists])
target_means = np.stack([td.mean for td in target_dists])
target_covs = np.stack([td.covar for td in target_dists])

betas = np.array([od.entropy() - beta_loss for od in old_dists])
epss = eps * np.ones(num_gaussians)
t0 = t.time()
means, covs = mp_cpp.more_step(epss, betas, old_means, old_covs, target_means, target_covs)
print("cpp", t.time() - t0)
for i in range(num_gaussians):
    projeted_dists_cpp.append(Gaussian(means[i], covs[i]))

t0 = t.time()
for i in range(num_gaussians):
    #print(i, target_dists[i].kl(old_dists[i]), target_dists[i].entropy())

    mean, cov = mp_py.more_step(eps, old_dists[i].entropy() - beta_loss,
                                old_dists[i].mean, old_dists[i].covar,
                                target_dists[i].mean, target_dists[i].covar)
    projeted_dists_py.append(Gaussian(mean, cov))

print("py", t.time() - t0)

"""uncomment for regression tests between python and c++"""
#for i in range(num_gaussians):
#    print(np.max(np.abs(projeted_dists_py[i].mean - projeted_dists_cpp[i].mean)))
#    print(np.max(np.abs(projeted_dists_py[i].covar - projeted_dists_cpp[i].covar)))
#    print(i, projeted_dists_cpp[i].kl(old_dists[i]), old_dists[i].entropy() - projeted_dists_cpp[i].entropy())


