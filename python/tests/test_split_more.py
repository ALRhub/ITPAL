from util.Gaussian import Gaussian
import numpy as np
from util.sample import sample_sympd
from projection.SplitMoreProjection import SplitMoreProjection

np.random.seed(0)
dim = 5

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)


mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)

eps_mean = 0.1
eps_cov = 0.01

smp = SplitMoreProjection(dim)

mean_proj, cov_proj = smp.more_step(eps_mean, eps_cov, mean_old, cov_old, mean_target, cov_target)

print("dist mean:", 0.5 * np.dot(mean_old - mean_proj, np.linalg.solve(cov_old, mean_old - mean_proj)))

cov_dist = 0.5 * (np.trace(np.linalg.solve(cov_old, cov_proj)) - dim + np.linalg.slogdet(cov_old)[1] -
                  np.linalg.slogdet(cov_proj)[1])
print("dist cov:", cov_dist)



