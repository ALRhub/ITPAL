import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from python.projection.SplitMoreProjection import SplitMoreProjection
from python.util.Gaussian import Gaussian
from python.util.central_differences import central_differences
from python.util.sample import sample_sympd

"""test grads for optimal eta_mu and eta_sigma w.r.t target distribution parameters q_tilde, Q_tilde"""

diff_delta = 1e-4  # delta for the central different gradient approximator, the whole thing is kind of sensitve to that
do_test_lagrange = False

np.random.seed(0)
dim = 10

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)
# q_target = Gaussian(mean_target, cov_old)
samples = q_target.sample(10)

proj_more = SplitMoreProjection(dim)

p0 = np.concatenate([q_target.lin_term, np.reshape(q_target.precision, [-1])], axis=-1)


def sym_wrapper(x):
    x = np.reshape(x, [dim, dim])
    return 0.5 * (x + x.T)


def eval_fn_lagrange(p, eps_mu, eps_sigma):
    lin = p[:dim]
    prec = sym_wrapper(p[dim:])

    covar = np.linalg.solve(prec, np.eye(dim))
    mean = covar @ lin

    _, _ = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, mean, covar)
    assert proj_more.success
    return np.array([proj_more.last_eta_mu, proj_more.last_eta_sig])


def eval_fn_before(proj_mean, proj_cov):
    mean = tf.constant(proj_mean)
    cov = tf.constant(sym_wrapper(proj_cov))

    new_dist = tfd.MultivariateNormalFullCovariance(mean, cov)
    py_loss = tf.reduce_mean(new_dist.log_prob(samples))
    # py_loss = tf.reduce_mean(mean ** 2) + tf.reduce_mean(cov * 0)
    return np.array(py_loss)


def eval_fn_after(eps_mu, eps_sigma, target_mean, target_cov):
    target_cov = sym_wrapper(target_cov)
    proj_mean, proj_cov = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, target_mean, target_cov)

    proj_mean = tf.constant(proj_mean)
    proj_cov = tf.constant(proj_cov)
    new_dist = tfd.MultivariateNormalFullCovariance(proj_mean, proj_cov)
    py_loss = tf.reduce_mean(new_dist.log_prob(samples))
    # py_loss = tf.reduce_mean(proj_mean ** 2) + tf.reduce_mean(proj_cov * 0)
    return np.array(py_loss)


def run_test_lagrange(eps_mu, eps_sigma):
    _, _ = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, q_target.mean, q_target.covar)
    # deq, deQ = proj_more.backward(np.zeros((dim,)), np.zeros((dim, dim)))
    dmu_dq, dmu_dQ, dsigma_dq, dsigma_dQ = proj_more.get_last_eo_grad()

    eta_mu = proj_more.last_eta_mu
    eta_sigma = proj_more.last_eta_sig
    print("Eta_mu", eta_mu, "Eta_sigma", eta_sigma)
    numerical_grads = central_differences(lambda p: eval_fn_lagrange(p, eps_mu, eps_sigma), p0, dim=2, delta=diff_delta)
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


def run_test_full(eps_mu, eps_sigma):
    # proj_more
    new_mean, new_cov = proj_more.more_step(eps_mu, eps_sigma, q_old.mean, q_old.covar, q_target.mean, q_target.covar)
    eta_mu = proj_more.last_eta_mu
    eta_sigma = proj_more.last_eta_sig
    print("Eta_mu", eta_mu, "Eta_sigma", eta_sigma)
    py_mean = tf.constant(new_mean)
    py_cov = tf.constant(new_cov)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(py_mean)
        tape.watch(py_cov)
        new_dist = tfd.MultivariateNormalFullCovariance(py_mean, py_cov)
        py_loss = tf.reduce_mean(new_dist.log_prob(samples))
        # py_loss = tf.reduce_mean(py_mean ** 2) + tf.reduce_mean(py_cov * 0)
    py_g1 = tape.gradient(py_loss, [py_mean, py_cov])
    py_g2 = proj_more.backward(np.array(py_g1[0]), np.array(py_g1[1]))

    # numpy
    num_mean_before = central_differences(lambda x: eval_fn_before(x, py_cov), py_mean, delta=1e-4)
    num_mean_after = central_differences(lambda x: eval_fn_after(eps_mu, eps_sigma, x, q_target.covar), q_target.mean,
                                         delta=1e-4)

    num_cov_before = central_differences(lambda x: eval_fn_before(py_mean, x), np.reshape(py_cov, -1), delta=1e-4)
    num_cov_before = np.reshape(num_cov_before, [dim, dim])

    num_cov_after = central_differences(lambda x: eval_fn_after(eps_mu, eps_sigma, q_target.mean, x),
                                        np.reshape(q_target.covar, -1),
                                        delta=1e-4)
    num_cov_after = np.reshape(num_cov_after, [dim, dim])

    print("loss", np.array(py_loss))
    print("mean")
    print("backward")
    print("before:")
    print("diff py, num", np.max(np.abs(np.array(py_g1[0]) - num_mean_before)))

    print("after:")
    print("diff py, num", np.max(np.abs(np.array(py_g2[0]) - num_mean_after)))

    print("covar")
    print("backward")
    print("before:")
    print("diff py, num", np.max(np.abs(np.array(py_g1[1]) - num_cov_before)))

    print("after:")
    print("diff py, num", np.max(np.abs(np.array(py_g2[1]) - num_cov_after)))


if do_test_lagrange:
    print("--------BOTH INACTIVE------------------")
    run_test_lagrange(eps_mu=100.0, eps_sigma=100.0)

    print("--------MEAN ACTIVE------------------")
    run_test_lagrange(eps_mu=0.01, eps_sigma=100.0)

    print("--------COVARIANCE ACTIVE------------------")
    run_test_lagrange(eps_mu=100.0, eps_sigma=0.01)

    print("--------BOTH ACTIVE------------------")
    run_test_lagrange(eps_mu=0.01, eps_sigma=0.001)

else:
    print("--------BOTH INACTIVE------------------")
    run_test_full(eps_mu=100.0, eps_sigma=100.0)

    print("--------MEAN ACTIVE------------------")
    run_test_full(eps_mu=0.01, eps_sigma=100.0)

    print("--------COVARIANCE ACTIVE------------------")
    run_test_full(eps_mu=100.0, eps_sigma=0.01)

    print("--------BOTH ACTIVE------------------")
    run_test_full(eps_mu=0.01, eps_sigma=0.001)
