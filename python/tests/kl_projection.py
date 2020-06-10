from typing import Any
import numpy as np

import cpp_projection
import torch as ch


class KLProjection(ch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        p, q, eps, beta = args
        mean, cov = p
        old_mean, old_cov = p
        mean = mean.detach().numpy()
        cov = cov.detach().numpy()
        old_mean = old_mean.detach().numpy()
        old_cov = old_cov.detach().numpy()

        batch_shape, dim = mean.shape

        projection_op = cpp_projection.BatchedProjection(batch_shape, dim, eec=False)
        proj_mean, proj_cov = projection_op.forward(eps * np.ones(batch_shape), beta, old_mean, old_cov, mean, cov)
        ctx.proj = projection_op

        return ch.from_numpy(proj_mean), ch.from_numpy(proj_cov)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        projection_op = ctx.proj
        d_means, d_covs = grad_outputs
        df_means, df_covs = projection_op.backward(d_means.detach().numpy(), d_covs.detach().numpy())
        return ch.tensor(df_means), ch.tensor(df_covs)


def kl_projection(p, q, eps, beta):
    mean, chol = p
    old_mean, old_chol = q

    cov = chol @ chol.permute(0, 2, 1)
    old_cov = old_chol @ old_chol.permute(0, 2, 1)

    return KLProjection.apply((mean, cov), (old_mean, old_cov), eps, beta)


def entropy(std):
    k = std.shape[-1]

    logdet = 2 * std.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    return .5 * (k * np.log(2 * np.e * np.pi) + logdet)


def sample_sympd(batch_size, dim):
    c_raw = ch.randn(size=[batch_size, dim, dim])
    return c_raw @ c_raw.permute(0, 2, 1) + ch.eye(dim)


np.random.seed(0)
dim = 3
batch_size = 10

mean_old = ch.randn(batch_size, dim)
chol_old = ch.cholesky(sample_sympd(batch_size, dim))

eps = 0.1
beta = entropy(chol_old) - 0.05

# shift to ensure projection is needed
mean = ch.randn(batch_size, dim) + 2
chol = ch.cholesky(sample_sympd(batch_size, dim)) + 2 * ch.eye(dim)

kl_projection((mean, chol), (mean_old, chol_old), eps, beta)
