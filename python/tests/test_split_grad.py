import numpy as np

from python.projection.SplitMoreProjection import SplitMoreProjection
from python.util.Gaussian import Gaussian
from python.util.central_differences import central_differences
from python.util.sample import sample_sympd

"""test grads for optimal eta_mu and eta_sigma w.r.t target distribution parameters q_tilde, Q_tilde"""

diff_delta = 1e-4  # delta for the central different gradient approximator, the whole thing is kind of sensitve to that

np.random.seed(0)
dim = 1

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)

proj_more = SplitMoreProjection(dim)


def eval_fn(p, eps_mu, eps_sigma):
    lin = p[:dim]
    prec = np.reshape(p[dim:], [dim, dim])
    prec = 0.5 * (prec + prec.T)

    covar = np.linalg.solve(prec, np.eye(dim))
    mean = covar @ lin

    _, _ = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, mean, covar)
    assert proj_more.success
    return np.array([proj_more.last_eta_mu, proj_more.last_eta_sig])


p0 = np.concatenate([q_target.lin_term, np.reshape(q_target.precision, [-1])], axis=-1)


def run_test(eps_mu, eps_sigma):
    _, _ = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, q_target.mean, q_target.covar)
    # deq, deQ = proj_more.backward(np.zeros((dim,)), np.zeros((dim, dim)))
    deq, deQ = proj_more.get_last_eo_grad()

    eta_mu = proj_more.last_eta_mu
    eta_sigma = proj_more.last_eta_sig
    print("Eta_mu", eta_mu, "Eta_sigma", eta_sigma)
    numerical_grads = central_differences(lambda p: eval_fn(p, eps_mu, eps_sigma), p0, dim=2, delta=diff_delta)
    # print("d eta_mu d q")
    # print("analytical", deq)
    # print("numerical", numerical_grads[0, :dim])
    # print("fraction", numerical_grads[0, :dim] / deq)
    print("max diff  d_eta_mu d_q", np.max(np.abs(numerical_grads[0, :dim] - deq)))

    # print("d eta_mu d Q")
    # print("analytical", deQ)
    # print("numerical", np.reshape(numerical_grads[0, dim:], [dim, dim]))
    # print("fraction", np.reshape(numerical_grads[0, dim:], [dim, dim]) / deQ)
    print("max diff d_eta_sigma d_Q", np.max(np.abs(np.reshape(numerical_grads[0, dim:], [dim, dim]) - deQ)))


print("--------BOTH INACTIVE------------------")
run_test(eps_mu=10.0, eps_sigma=10.0)

print("--------MEAN ACTIVE------------------")
run_test(eps_mu=0.01, eps_sigma=10.0)

print("--------COVARIANCE ACTIVE------------------")
run_test(eps_mu=10.0, eps_sigma=0.01)

print("--------BOTH ACTIVE------------------")
run_test(eps_mu=0.01, eps_sigma=0.001)
