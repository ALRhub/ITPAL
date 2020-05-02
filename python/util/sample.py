import numpy as np

def sample_sympd(dim):
    c_raw = np.random.normal(size=[dim, dim])
    return c_raw @ c_raw.T + np.eye(dim)