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

regression_test = True #whether to run the regression tests

num_gaussians = 32
dim = 32

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))


eps = 0.1
beta_loss = 0.05

## these two cases using pytorch over py and cpp projections
if run_py_torch:
    #mp_cpp = projection.BatchedProjection(num_gaussians, dim)
    projeted_dists_py_torch = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)

    # pytorch net
    mp_cpp_torch = ProjectionSimpleNetPyBatched(dim, num_gaussians, epss, betas, old_means, old_covs)


    t0 = t.time()

    # keep the input to layers in tensor (instead of array)
    t_means = torch.from_numpy(target_means).double()
    t_covs = torch.from_numpy(target_covs).double()

    means, covs = mp_cpp_torch(t_means, t_covs)
    print("py pytorch", t.time() - t0)
    for i in range(num_gaussians):
        projeted_dists_py_torch.append(Gaussian(means[i].detach().numpy(), covs[i].detach().numpy()))


if run_cpp_torch:
    #mp_cpp = projection.BatchedProjection(num_gaussians, dim)
    projeted_dists_cpp_torch = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)

    # pytorch net
    mp_cpp_torch = ProjectionSimpleNetCppBatched(dim, num_gaussians, epss, betas, old_means, old_covs)


    t0 = t.time()

    # keep the input to layers in tensor (instead of array)
    t_means = torch.from_numpy(target_means).double()
    t_covs = torch.from_numpy(target_covs).double()

    means, covs = mp_cpp_torch(t_means, t_covs)
    print("cpp pytorch", t.time() - t0)
    for i in range(num_gaussians):
        projeted_dists_cpp_torch.append(Gaussian(means[i].detach().numpy(), covs[i].detach().numpy()))

#####


if run_cpp:
    mp_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=True)
    projeted_dists_cpp = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)
    t0 = t.time()
    means, covs = mp_cpp.forward(epss, betas, old_means, old_covs, target_means, target_covs)
    print("cpp", t.time() - t0)
    for i in range(num_gaussians):
        projeted_dists_cpp.append(Gaussian(means[i], covs[i]))


if run_py:
    mp_py = MoreProjection(dim)
    projeted_dists_py = []

    t0 = t.time()
    for i in range(num_gaussians):
        mean, cov = mp_py.more_step(eps, old_dists[i].entropy() - beta_loss,
                                    old_dists[i].mean, old_dists[i].covar,
                                    target_dists[i].mean, target_dists[i].covar)
        projeted_dists_py.append(Gaussian(mean, cov))
    print("py", t.time() - t0)

if run_cvx:
    mp_cvx = CVXProjector(dim)
    projected_dists_cvx = []
    t0 = t.time()
    for i in range(num_gaussians):
        np_inputs = [eps, old_dists[i].entropy() - beta_loss, old_dists[i].mean, old_dists[i].covar,
                     target_dists[i].mean, target_dists[i].covar]
        mean, cov = mp_cvx.project(*[tf.constant(np_in) for np_in in np_inputs])
        projected_dists_cvx.append(Gaussian(mean.numpy(), cov.numpy()))
    print("cvx", t.time() - t0)

if regression_test:
    for i in range(num_gaussians):
        print("-----------Distribution:", i, "------------")
        print("Constraints")

        if run_py_torch:
            print("Py Pytorch:", projeted_dists_py_torch[i].kl(old_dists[i]),
                  old_dists[i].entropy() - projeted_dists_py_torch[i].entropy())
        if run_cpp_torch:
            print("CPP Pytorch:", projeted_dists_cpp_torch[i].kl(old_dists[i]),
                  old_dists[i].entropy() - projeted_dists_cpp_torch[i].entropy())

        if run_cpp:
            print("CPP:", projeted_dists_cpp[i].kl(old_dists[i]), old_dists[i].entropy() - projeted_dists_cpp[i].entropy())
        if run_py:
            print("PY:", projeted_dists_cpp[i].kl(old_dists[i]), old_dists[i].entropy() - projeted_dists_cpp[i].entropy())
        if run_cvx:
            print("CVX:", projeted_dists_cpp[i].kl(old_dists[i]), old_dists[i].entropy() - projeted_dists_cpp[i].entropy())
        if run_py_torch and run_py:
            print("PY - Py Torch")
            print(np.max(np.abs(projeted_dists_py[i].mean - projeted_dists_py_torch[i].mean)))
            print(np.max(np.abs(projeted_dists_py[i].covar - projeted_dists_py_torch[i].covar)))
        if run_cpp and run_py:
            print("PY - CPP")
            print(np.max(np.abs(projeted_dists_py[i].mean - projeted_dists_cpp[i].mean)))
            print(np.max(np.abs(projeted_dists_py[i].covar - projeted_dists_cpp[i].covar)))
        if run_cpp and run_cpp_torch:
            print("Cpp - CPP Torch")
            print(np.max(np.abs(projeted_dists_cpp_torch[i].mean - projeted_dists_cpp[i].mean)))
            print(np.max(np.abs(projeted_dists_cpp_torch[i].covar - projeted_dists_cpp[i].covar)))
        if run_cvx and run_cpp:
            print("CPP- CVX")
            print(np.max(np.abs(projected_dists_cvx[i].mean - projeted_dists_cpp[i].mean)))
            print(np.max(np.abs(projected_dists_cvx[i].covar - projeted_dists_cpp[i].covar)))
        if run_py and run_cvx:
            print("PY - CVX")
            print(np.max(np.abs(projeted_dists_py[i].mean - projected_dists_cvx[i].mean)))
            print(np.max(np.abs(projeted_dists_py[i].covar - projected_dists_cvx[i].covar)))



