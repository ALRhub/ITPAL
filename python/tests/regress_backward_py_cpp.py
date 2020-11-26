from util.Gaussian import Gaussian
import numpy as np
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection
import cpp_projection as projection
import time as t

from projection.pytorch_op import ProjectionSimpleNetCppBatched, ProjectionSimpleNetPyBatched
import torch
import torch.nn as nn

"""
IMPORTANT: This currently only checks if cpp and python produce the same result, there is no check
if thats the correct one!
benchmark and regression test for backward pass"""

np.random.seed(0)
run_cpp = True  # whether to run cpp implementation
run_py = True   # whether to run python implementation
regression_test = True #whether to run the regression tests

run_py_torch = False # whether to run pytorch with py implementation layers
run_cpp_torch = False # whether to run pytorch with cpp implementation layers



num_gaussians = 8
dim = 100

old_dists = []
target_dists = []
for i in range(num_gaussians):
    old_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))
    #target_dists.append(old_dists[-1])
    target_dists.append(Gaussian(np.random.uniform(low=-0.1, high=0.1, size=dim), sample_sympd(dim)))

means_true = np.random.uniform(-1, 1, [num_gaussians, dim])
covs_true = np.stack([sample_sympd(dim) for _ in range(num_gaussians)])

def get_loss_derivative(means_,covs_):
    '''
    a simple quadratic loss function applied at the last layer
    
    :param means_: 
    :param covs_: 
    :return: 
    '''

    d_means = means_ - means_true
    d_covs = covs_ - covs_true
    return d_means, d_covs

# the same lost, but for pytorch
def get_loss_pytorch(means_, covs_):
    '''
    a simple quadratic loss function applied at the last layer

    :param means_:
    :param covs_:
    :return:
    '''
    # individual loss on each sample
    criteria = nn.MSELoss(reduction='none')
    losses_mean = 0.5*torch.sum(criteria(means_, torch.from_numpy(means_true).double()),1)
    losses_cov  = 0.5*torch.sum(criteria(covs_, torch.from_numpy(covs_true).double()),(1,2))

    return torch.sum(losses_mean + losses_cov)

eps = 0.1
beta_loss = 0.05

if run_py_torch:
    #mp_cpp = projection.BatchedProjection(num_gaussians, dim)
    projeted_dists_cpp = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)

    # pytorch net
    t0 = t.time()

    proj_mores = []
    proj_mores = [MoreProjection(dim) for i in range(num_gaussians)]

    mp_py_torch = ProjectionSimpleNetPyBatched(proj_mores, num_gaussians, epss, betas, old_means, old_covs)

    t_means = torch.from_numpy(target_means).double()
    t_covs = torch.from_numpy(target_covs).double()
    t_means.requires_grad = True
    t_covs.requires_grad  = True

    means_py_torch, covs_py_torch = mp_py_torch(t_means, t_covs)
    torch_losses = get_loss_pytorch(means_py_torch, covs_py_torch)
    torch_losses.backward()

    py_d_means_torch, py_d_covs_torch = t_means.grad, t_covs.grad
    print("pytorch py", t.time() - t0)


if run_cpp_torch:
    #mp_cpp = projection.BatchedProjection(num_gaussians, dim)
    projeted_dists_cpp = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)

    # pytorch net
    t0 = t.time()

    mp_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=True)

    mp_cpp_torch = ProjectionSimpleNetCppBatched(mp_cpp, epss, betas, old_means, old_covs)
    t_means = torch.from_numpy(target_means).double()
    t_covs = torch.from_numpy(target_covs).double()
    t_means.requires_grad = True
    t_covs.requires_grad = True

    means_torch, covs_torch = mp_cpp_torch(t_means, t_covs)
    torch_losses = get_loss_pytorch(means_torch, covs_torch)
    torch_losses.backward()



    cpp_d_means_torch, cpp_d_covs_torch = t_means.grad, t_covs.grad
    print("pytorch cpp", t.time() - t0)


if run_cpp:
    mp_cpp = projection.BatchedProjection(num_gaussians, dim, eec=False, constrain_entropy=True)
    projeted_dists_cpp = []

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])
    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    betas = np.array([od.entropy() - beta_loss for od in old_dists])
    epss = eps * np.ones(num_gaussians)
    means, covs = mp_cpp.forward(epss, betas, old_means, old_covs, target_means, target_covs)
    t0 = t.time()
    d_means, d_covs = get_loss_derivative(means, covs)
    cpp_d_means, cpp_d_covs = mp_cpp.backward(d_means, d_covs)
    print("cpp", t.time() - t0)

if run_py:
    mp_py = MoreProjection(dim)
    py_d_means = []
    py_d_covs = []

    t_accu = 0
    for i in range(num_gaussians):
        mean, cov = mp_py.more_step(eps, old_dists[i].entropy() - beta_loss,
                                    old_dists[i].mean, old_dists[i].covar,
                                    target_dists[i].mean, target_dists[i].covar)
        t0 = t.time()
        m, c = mp_py.backward(d_means[i], d_covs[i])
        t_accu += t.time() - t0
        py_d_means.append(m)
        py_d_covs.append(c)

    py_d_means = np.stack(py_d_means, axis=0)
    py_d_covs = np.stack(py_d_covs, axis=0)
    print("py", t_accu)

if regression_test:
    for i in range(num_gaussians):
        print("-----------Distribution:", i, "------------")
        print("max dmean diff", np.max(np.abs(py_d_means[i] - cpp_d_means[i])))
      #  print("max dmean diff (vs. cpp pytorch)", np.max(np.abs(py_d_means[i] - cpp_d_means_torch[i].detach().numpy())))
      #  print("max dmean diff (vs. py pytorch)", np.max(np.abs(py_d_means[i] - py_d_means_torch[i].detach().numpy())))

        print("max dcov diff", np.max(np.abs(py_d_covs[i] - cpp_d_covs[i])))
       # print("max dcov diff (vs. cpp pytorch)", np.max(np.abs(py_d_covs[i] - cpp_d_covs_torch[i].detach().numpy())))
       # print("max dcov diff (vs. py pytorch)", np.max(np.abs(py_d_covs[i] - py_d_covs_torch[i].detach().numpy())))




