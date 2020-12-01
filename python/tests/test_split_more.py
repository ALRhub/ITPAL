import numpy as np

from python.projection.SplitMoreProjection import SplitMoreProjection
import cpp_projection as proj
from python.util.Gaussian import Gaussian
from python.util.sample import sample_sympd

np.random.seed(0)
dim = 5

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = np.exp(np.random.uniform(low=-0.5, high=0.5, size=dim))

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = np.exp(np.random.uniform(low=-0.5, high=0.5, size=dim))

eps_mean = 0.1
eps_cov = 0.01

smp = SplitMoreProjection(dim)

cpp_smp = proj.SplitDiagMoreProjection(dim, max_eval=100)
mean_proj_cpp, cov_proj_cpp = cpp_smp.forward(eps_mean, eps_cov, mean_old, cov_old, mean_target, cov_target)

print("cpp done")
cov_old = cov_old * np.eye(dim)
cov_target = cov_target * np.eye(dim)
mean_proj, cov_proj = smp.more_step(eps_mean, eps_cov, mean_old, cov_old, mean_target, cov_target)

print("dist mean:", 0.5 * np.dot(mean_old - mean_proj, np.linalg.solve(cov_old, mean_old - mean_proj)))

cov_dist = 0.5 * (np.trace(np.linalg.solve(cov_old, cov_proj)) - dim + np.linalg.slogdet(cov_old)[1] -
                  np.linalg.slogdet(cov_proj)[1])
print("dist cov:", cov_dist)

print("diff py, cpp", np.max(np.abs(mean_proj_cpp - mean_proj)))
print("diff py, cpp", np.max(np.abs(cov_proj_cpp - np.diag(cov_proj))))