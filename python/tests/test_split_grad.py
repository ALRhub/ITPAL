import numpy as np

from python.projection.SplitMoreProjection import SplitMoreProjection
from python.util.Gaussian import Gaussian
from python.util.central_differences import central_differences
from python.util.sample import sample_sympd

"""test grads for optimal eta_mu and eta_sigma w.r.t target distribution parameters q_tilde, Q_tilde"""

diff_delta = 1e-4  # delta for the central different gradient approximator, the whole thing is kind of sensitve to that

np.random.seed(0)
dim = 10

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)

proj_more = SplitMoreProjection(dim)

p0 = np.concatenate([q_target.lin_term, np.reshape(q_target.precision, [-1])], axis=-1)


def eval_fn(p, eps_mu, eps_sigma):
    lin = p[:dim]
    prec = np.reshape(p[dim:], [dim, dim])
    prec = 0.5 * (prec + prec.T)

    covar = np.linalg.solve(prec, np.eye(dim))
    mean = covar @ lin

    _, _ = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, mean, covar)
    assert proj_more.success
    return np.array([proj_more.last_eta_mu, proj_more.last_eta_sig])


def run_test(eps_mu, eps_sigma):
    _, _ = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, q_target.mean, q_target.covar)
    # deq, deQ = proj_more.backward(np.zeros((dim,)), np.zeros((dim, dim)))
    dmu_dq, dmu_dQ, dsigma_dq, dsigma_dQ = proj_more.get_last_eo_grad()

    eta_mu = proj_more.last_eta_mu
    eta_sigma = proj_more.last_eta_sig
    print("Eta_mu", eta_mu, "Eta_sigma", eta_sigma)
    numerical_grads = central_differences(lambda p: eval_fn(p, eps_mu, eps_sigma), p0, dim=2, delta=diff_delta)
    # if np.any(deq != 0):
    #     print("deq fraction", deq / numerical_grads[0, :dim])
    diff_q = numerical_grads[0, :dim] - dmu_dq
    print("max diff d_eta_mu d_q", np.max(np.abs(diff_q)))
    # print("mean diff  d_eta_mu d_q", np.mean(diff_q))
    diff_q = numerical_grads[1, :dim] - dsigma_dq
    print("max diff d_eta_sigma d_q", np.max(np.abs(diff_q)))

    # if np.any(deQ != 0):
    #     print("deQ fraction", deQ / np.reshape(numerical_grads[1, dim:], [dim, dim]))
    diff_Q = np.reshape(numerical_grads[0, dim:], [dim, dim]) - dmu_dQ
    print("max diff d_eta_mu d_Q", np.max(np.abs(diff_Q)))
    diff_Q = np.reshape(numerical_grads[1, dim:], [dim, dim]) - dsigma_dQ
    print("max diff d_eta_sigma d_Q", np.max(np.abs(diff_Q)))
    # print("mean diff d_eta_sigma d_Q", np.mean(diff_Q))
    # print(numerical_grads)


print("--------BOTH INACTIVE------------------")
run_test(eps_mu=100.0, eps_sigma=100.0)

print("--------MEAN ACTIVE------------------")
run_test(eps_mu=0.01, eps_sigma=100.0)

print("--------COVARIANCE ACTIVE------------------")
run_test(eps_mu=100.0, eps_sigma=0.01)

print("--------BOTH ACTIVE------------------")
run_test(eps_mu=0.01, eps_sigma=0.001)
