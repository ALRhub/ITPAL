import numpy as np
from util.sample import sample_sympd
from util.Gaussian import Gaussian
dim = 10

def np_kl(lin1, prec1, lin2, prec2):
    ld_terms = - np.linalg.slogdet(prec2)[1] + np.linalg.slogdet(prec1)[1]
    trace_term = np.trace(prec2 @ np.linalg.inv(prec1))
    m_term = np.dot(lin2, np.linalg.solve(prec2, lin2)) - 2 * np.dot(lin1, np.linalg.solve(prec1, lin2))
    m_term += np.dot(lin1, np.linalg.inv(prec1) @ prec2 @ np.linalg.inv(prec1) @ lin1)
    return 0.5 * (ld_terms + trace_term + m_term - lin1.shape[0])

for i in range(100):
    target_dist = Gaussian(np.random.normal(size=dim), sample_sympd(dim))
    old_dist = Gaussian(np.random.normal(size=dim), sample_sympd(dim))

    print("kl", target_dist.kl(old_dist))
    print("npkl", np_kl(target_dist.lin_term, target_dist.precision, old_dist.lin_term, old_dist.precision))