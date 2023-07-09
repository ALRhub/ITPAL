import unittest
import numpy as np
# from python.projection.MoreProjection import MoreProjection
from python.util.Gaussian import Gaussian
from python.util.sample import sample_sympd
from python.projection.CovOnlyMoreProjection import CovOnlyMoreProjection

import cpp_projection


class TestBatchCovOnly(unittest.TestCase):
    np.random.seed(42)
    num_gaussians = 32
    dim = 5
    eps = 0.001

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
    
    cov_only_cpp = cpp_projection.BatchedCovOnlyProjection(num_gaussians, dim, 10000)
    cov_only_py = CovOnlyMoreProjection(dim)

    old_means = np.stack([od.mean for od in old_dists])
    old_covs = np.stack([od.covar for od in old_dists])

    target_means = np.stack([td.mean for td in target_dists])
    target_covs = np.stack([td.covar for td in target_dists])

    
    uc_betas = np.nan * np.ones(num_gaussians)
    epss = eps * np.ones(num_gaussians)

    d_means = np.zeros([num_gaussians, dim])  # np.random.normal(size=[num_gaussians, dim])
    d_covs_var = np.random.normal(size=[num_gaussians, dim, dim])

    grad_mean = np.zeros([num_gaussians, dim])

    # epss, old_chols, target_chols,target_covars
    covs_cpp = cov_only_cpp.forward(epss, np.linalg.cholesky(old_covs), np.linalg.cholesky(target_covs), target_covs)
    grad_cov_cpp = cov_only_cpp.backward(d_covs_var)

    def test_forward_backward(self):
        for i in range(self.num_gaussians):
            cov_py = self.cov_only_py.more_step(self.eps, self.old_covs[i], self.target_covs[i])
            self.assertTrue(np.allclose(cov_py, self.covs_cpp[i]))

            grad_cov_py = self.cov_only_py.backward(self.d_covs_var[i])
            self.assertTrue(np.allclose(grad_cov_py, self.grad_cov_cpp[i]))

if __name__ == '__main__':
    unittest.main()