import torch
from torch.autograd import Function, Variable
from torch.nn import Module
import numpy as np

from numpy.fft import rfft2, irfft2
from util.Gaussian import Gaussian
from util.sample import sample_sympd

from projection.MoreProjection import MoreProjection
import cpp_projection as projection


import torch.nn as nn


# NOTE: keep the input to layers in tensor (instead of numpy array)

def TorchProjMore(dim, eps, beta, q_old):
    class TorchProjection(Function):
        @staticmethod
        def forward(ctx, mean_target, cov_target):

            mean = mean_target.detach().numpy()
            cov  = cov_target.detach().numpy()

            proj_more = MoreProjection(dim)
            new_mean, new_cov = proj_more.more_step(eps, beta, q_old.mean, q_old.covar, mean, cov)
            ctx.proj = proj_more

            return torch.Tensor(new_mean), torch.Tensor(new_cov)

        @staticmethod
        def backward(ctx, d_mean, d_cov):
            proj_more = ctx.proj
            df_mean, df_cov = proj_more.backward(d_mean.detach().numpy(), d_cov.detach().numpy())
            return torch.Tensor(df_mean), torch.Tensor(df_cov)
    return TorchProjection.apply


# Below are batch versions

def TorchProjMoreBatched(proj_more_list, num_gaussians, epss, betas, old_means, old_covs):
    class TorchProjection(Function):
        @staticmethod
        def forward(ctx, mean_targets, cov_targets):


            means = mean_targets.detach().numpy()
            covs = cov_targets.detach().numpy()
            projeted_means = []
            projeted_covs = []
            for i in range(num_gaussians):
                proj_more = proj_more_list[i]
                mean, cov = proj_more.more_step(epss[i], betas[i],
                                                old_means[i], old_covs[i],
                                                means[i], covs[i])
                projeted_means.append(mean)
                projeted_covs.append(cov)

            new_means = np.stack(projeted_means)
            new_covs  = np.stack(projeted_covs)
            ctx.projs = proj_more_list

            return torch.Tensor(new_means).double(), torch.Tensor(new_covs).double()

        @staticmethod
        def backward(ctx, d_means, d_covs):
            proj_more_list = ctx.projs
            df_means=[]
            df_covs=[]

            for i in range(num_gaussians):
                df_mean, df_cov = proj_more_list[i].backward(d_means[i].detach().numpy(), d_covs[i].detach().numpy())
                df_means.append(df_mean)
                df_covs.append(df_cov)
            df_means = np.stack(df_means)
            df_covs = np.stack(df_covs)
            return torch.Tensor(df_means).double(), torch.Tensor(df_covs).double()
    return TorchProjection.apply



def TorchProjMoreCppBatched(projection_op, epss, betas, old_means, old_covs):
    class TorchProjection(Function):
        @staticmethod
        def forward(ctx, target_means, target_covs):
            means = target_means.detach().numpy()
            covs = target_covs.detach().numpy()

            #mp_cpp = projection.BatchedProjection(num_gaussians, dim)
            means, covs = projection_op.forward(epss, betas, old_means, old_covs, means, covs)
            ctx.proj = projection_op

            return torch.Tensor(means).double(), torch.Tensor(covs).double()

        @staticmethod
        def backward(ctx, d_means, d_covs):
            proj = ctx.proj
            df_means, df_covs = proj.backward(d_means.detach().numpy(), d_covs.detach().numpy())
            return torch.Tensor(df_means), torch.Tensor(df_covs)
    return TorchProjection.apply



'''
A simple module (with 1 layer) using python-MORE projection layer (non-batched version)
'''
class ProjectionSimpleNet(nn.Module):
    def __init__(self, dim, eps, beta, q_old):
        super(ProjectionSimpleNet, self).__init__()
        self.dim = dim
        self.eps = eps
        self.beta = beta
        self.q_old = q_old
        self.torch_project = TorchProjMore(self.dim, self.eps, self.beta, self.q_old)

    def forward(self, q_target_mean, q_target_covar):
        new_mean, new_cov = self.torch_project(q_target_mean, q_target_covar)
        return new_mean, new_cov

'''
A simple module (with 1 layer) using Cpp projection layer (batched version)
'''
class ProjectionSimpleNetCppBatched(nn.Module):
    def __init__(self, projection_op, epss, betas, old_means, old_covs):
        super(ProjectionSimpleNetCppBatched, self).__init__()
        self.projection_op = projection_op
        self.old_means = old_means
        self.old_covs = old_covs
        self.epss = epss
        self.betas = betas
        self.torch_project = TorchProjMoreCppBatched(self.projection_op, self.epss, self.betas, self.old_means, self.old_covs)

    def forward(self, q_target_means, q_target_covars):
        new_means, new_covs = self.torch_project(q_target_means, q_target_covars)
        return new_means, new_covs



'''
A simple module (with 1 layer) using python projection layer (batched version)
'''
class ProjectionSimpleNetPyBatched(nn.Module):
    def __init__(self, dim, num_gaussians, epss, betas, old_means, old_covs):
        super(ProjectionSimpleNetPyBatched, self).__init__()
        self.dim = dim
        self.num_gaussians = num_gaussians
        self.old_means = old_means
        self.old_covs = old_covs
        self.epss = epss
        self.betas = betas
        self.torch_project = TorchProjMoreBatched(self.dim, self.num_gaussians, self.epss, self.betas, self.old_means, self.old_covs)

    def forward(self, q_target_means, q_target_covars):
        new_means, new_covs = self.torch_project(q_target_means, q_target_covars)
        return new_means, new_covs