import numpy as np
import time
from python.projection.SplitMoreProjection import SplitMoreProjection
import cpp_projection as proj
from python.util.Gaussian import Gaussian
from python.util.sample import sample_sympd

np.random.seed(0)
dim = 15
batch_size = 1

mean_old = np.random.uniform(low=-1, high=1, size=(batch_size, dim))
cov_old = np.exp(np.random.uniform(low=-0.5, high=0.5, size=(batch_size, dim)))

mean_target = np.random.uniform(low=-1, high=1, size=(batch_size, dim))
cov_target = np.exp(np.random.uniform(low=-0.5, high=0.5, size=(batch_size, dim)))

d_mean = np.random.normal(size=(batch_size, dim))
d_cov = np.random.normal(size=(batch_size, dim))

eps_mean = 0.1 * np.ones(batch_size)
eps_cov = 0.01 * np.ones(batch_size)

smp = SplitMoreProjection(dim)

cpp_smp = proj.BatchedSplitDiagMoreProjection(batch_size, dim, max_eval=100)
mean_proj_cpp, cov_proj_cpp = cpp_smp.forward(eps_mean, eps_cov, mean_old, cov_old, mean_target, cov_target)
dtarget_mean_cpp, dtarget_cov_cpp = cpp_smp.backward(d_mean, d_cov)

print("cpp done")
cov_old = cov_old[..., None] * np.eye(dim)[None, ...]
cov_target = cov_target[..., None] * np.eye(dim)[None, ...]
d_cov = d_cov[..., None] * np.eye(dim)[None, ...]

for i in range(batch_size):
    mean_proj, cov_proj = smp.more_step(eps_mean[i], eps_cov[i], mean_old[i], cov_old[i], mean_target[i], cov_target[i])
    print("dist mean:", 0.5 * np.dot(mean_old[i] - mean_proj, np.linalg.solve(cov_old[i], mean_old[i] - mean_proj)))

    dtarget_mean, dtarget_cov = smp.backward(d_mean[i], d_cov[i])

    cov_dist = 0.5 * (np.trace(np.linalg.solve(cov_old[i], cov_proj)) - dim + np.linalg.slogdet(cov_old[i])[1] -
                      np.linalg.slogdet(cov_proj)[1])
    print("dist cov:", cov_dist)
    print("forward")
    print("mean diff py, cpp", np.max(np.abs(mean_proj_cpp[i] - mean_proj)))
    print("cov diff py, cpp", np.max(np.abs(cov_proj_cpp[i] - np.diag(cov_proj))))
    print("backward")
    print("mean diff py, cpp", np.max(np.abs(dtarget_mean_cpp[i] - dtarget_mean)))
    print("cov diff py, cpp", np.max(np.abs(dtarget_cov_cpp[i] - np.diag(dtarget_cov))))
