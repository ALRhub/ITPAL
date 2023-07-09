import unittest
import numpy as np

from python.util.sample import sample_sympd
from python.projection.CovOnlyMoreProjection import CovOnlyMoreProjection

import cpp_projection


class TestConvOnly(unittest.TestCase):
    np.random.seed(42)
    dim = 15
    eps = 0.001

    means = np.random.uniform(low=-0.5, high=0.5, size=dim)
    old_cov= sample_sympd(dim)
    target_cov = sample_sympd(dim)
    
    cov_only_cpp = cpp_projection.CovOnlyMoreProjection(dim, 100)
    cov_only_py = CovOnlyMoreProjection(dim)

    d_covs_var = np.random.normal(size=[dim, dim])

    cov_cpp = cov_only_cpp.forward(eps, np.linalg.cholesky(old_cov).T, target_cov)
    grad_cov_cpp = cov_only_cpp.backward(d_covs_var)

    cov_py = cov_only_py.more_step(eps, old_cov, target_cov)
    grad_cov_py = cov_only_py.backward(d_covs_var)

    def test_forward(self):
        self.assertTrue(np.allclose(self.cov_py, self.cov_cpp))
    
    def test_backward(self):
        self.assertTrue(np.allclose(self.grad_cov_py, self.grad_cov_cpp))

if __name__ == '__main__':
    unittest.main()