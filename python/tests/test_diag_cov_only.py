import sys
sys.path.append('/home/hussam/Documents/Documents/Work/Workspace/ALR-HiWi/ITPAL/')

import unittest
import numpy as np

import unittest
import numpy as np

from python.util.sample import sample_sympd
from python.projection.DiagCovOnlyMoreProjection import DiagCovOnlyMoreProjection

import cpp_projection


class TestConvOnly(unittest.TestCase):
    # np.random.seed(42)
    dim = 5
    eps = 0.001

    means = np.random.uniform(low=-0.5, high=0.5, size=dim)
    old_var= np.exp(np.random.uniform(low=-0.5, high=0.5, size=dim))
    target_var = np.exp(np.random.uniform(low=-0.5, high=0.5, size=dim))
    
    cov_only_cpp = cpp_projection.DiagCovOnlyMoreProjection(dim, 100)
    cov_only_py = DiagCovOnlyMoreProjection(dim)

    d_covs_var = np.random.normal(size=[dim])

    cov_cpp = cov_only_cpp.forward(eps, old_var, target_var)
    grad_cov_cpp = cov_only_cpp.backward(d_covs_var)

    cov_py = cov_only_py.more_step(eps, old_var, target_var)
    print(cov_cpp)
    print(cov_py)
    # grad_cov_py = cov_only_py.backward(d_covs_var)

    # def test_forward(self):
    #     self.assertTrue(np.allclose(self.cov_py, self.cov_cpp))
    
    # def test_backward(self):
    #     self.assertTrue(np.allclose(self.grad_cov_py, self.grad_cov_cpp))

if __name__ == '__main__':
    unittest.main()