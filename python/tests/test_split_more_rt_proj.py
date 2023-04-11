import sys
sys.path.append('/home/hussam/Documents/Documents/Work/Workspace/ALR-HiWi/ITPAL/')

import unittest
import numpy as np
from python.projection.MoreProjection import MoreProjection
import cpp_projection as proj

class TestSplitMoreRTProj(unittest.TestCase):
    # np.random.seed(0)
    dim = 5
    batch_size = 32

    mean_old = np.random.uniform(low=-1, high=1, size=(batch_size, dim))
    cov_old = np.exp(np.random.uniform(low=-0.5, high=0.5, size=(batch_size, dim)))

    mean_target = np.random.uniform(low=-1, high=1, size=(batch_size, dim))
    cov_target = np.exp(np.random.uniform(low=-0.5, high=0.5, size=(batch_size, dim)))

    d_mean = np.random.normal(size=(batch_size, dim))
    d_cov = np.random.normal(size=(batch_size, dim))

    eps_mean = 0.1 * np.ones(batch_size)
    eps_cov = 0.01 * np.ones(batch_size)

    smp = MoreProjection(dim)

    cpp_smp = proj.BatchedSplitDiagMoreProjection(batch_size, dim, max_eval=100)
    mean_proj_cpp, cov_proj_cpp = cpp_smp.forward(eps_mean, eps_cov, mean_old, cov_old, mean_target, cov_target)
    dm, dc = cpp_smp.backward(d_mean, d_cov)

    cov_old = cov_old[..., None] * np.eye(dim)[None, ...]
    cov_target = cov_target[..., None] * np.eye(dim)[None, ...]
    d_cov = d_cov[..., None] * np.eye(dim)[None, ...]

    def test_forward_backward(self):
        for i in range(self.batch_size):
            mean_proj, cov_proj = self.smp.more_step(self.eps_mean[i], self.eps_cov[i], self.mean_old[i], self.cov_old[i], self.mean_target[i], self.cov_target[i])
            dtarget_mean, dtarget_cov = self.smp.backward(self.d_mean[i], self.d_cov[i])

            if not np.allclose(self.mean_proj_cpp[i], mean_proj):
                print(np.max(np.abs(self.mean_proj_cpp[i]- mean_proj)))

            self.assertTrue(np.allclose(self.mean_proj_cpp[i], mean_proj))
            self.assertTrue(np.allclose(self.cov_proj_cpp[i], np.diag(cov_proj)))
            if not np.allclose(self.dm[i], dtarget_mean):
                print(np.max(np.abs(self.dm[i]- dtarget_mean)))
            self.assertTrue(np.allclose(self.dm[i], dtarget_mean))
            self.assertTrue(np.allclose(self.dc[i], np.diag(dtarget_cov)))

if __name__ == '__main__':
    unittest.main()