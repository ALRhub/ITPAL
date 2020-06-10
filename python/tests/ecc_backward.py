import numpy as np
from util.Gaussian import Gaussian
from util.central_differences import central_differences
from util.sample import sample_sympd
import cpp_projection as proj
from cvx.cvxlayers_projection import CVXProjector
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from projection.pytorch_op import ProjectionSimpleNet
from torch.distributions import MultivariateNormal
import torch

"""
Regression test for backward pass against numerical approximation and cvxopt
There seems to be some issue with the cvxopt backward for the covariance but our version checks out
against its own central differences approximation
"""


np.random.seed(0)
dim = 2

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)


mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)
samples = q_target.sample(10)

proj_more = proj.MoreProjection(dim, eec=True)

def sym_wrapper(x):
    x = np.reshape(x, [dim, dim])
    return 0.5 * (x + x.T)


def eval_fn_before(proj_mean, proj_cov):
    mean = tf.constant(proj_mean)
    cov = tf.constant(sym_wrapper(proj_cov))

    new_dist = tfd.MultivariateNormalFullCovariance(mean, cov)
    py_loss = tf.reduce_mean(new_dist.log_prob(samples))
    return np.array(py_loss)


def eval_fn_after(eps, beta, target_mean, target_cov):
    target_cov = sym_wrapper(target_cov)
    proj_mean, proj_cov = proj_more.forward(eps, beta, q_old.mean, q_old.covar,
                                              target_mean, target_cov)
    proj_mean = tf.constant(proj_mean)
    proj_cov = tf.constant(proj_cov)
    new_dist = tfd.MultivariateNormalFullCovariance(proj_mean, proj_cov)
    py_loss = tf.reduce_mean(new_dist.log_prob(samples))
    return np.array(py_loss)


def run_test(eps, beta_seed=0):
    np.random.seed(beta_seed)
    beta = q_old.entropy() + np.random.uniform(-0.5, 0.5)

    #proj_more
    new_mean, new_cov = proj_more.forward(eps, beta, q_old.mean, q_old.covar,
                                            q_target.mean, q_target.covar)
    print("eta, omega", proj_more.last_eta, proj_more.last_omega)
    py_mean = tf.constant(new_mean)
    py_cov = tf.constant(new_cov)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(py_mean)
        tape.watch(py_cov)
        new_dist = tfd.MultivariateNormalFullCovariance(py_mean, py_cov)
        py_loss = tf.reduce_mean(new_dist.log_prob(samples))
    py_g1 = tape.gradient(py_loss, [py_mean, py_cov])
    py_g2 = proj_more.backward(np.array(py_g1[0]), np.array(py_g1[1]))

    #numpy
    num_mean_before = central_differences(lambda x: eval_fn_before(x, py_cov), py_mean, delta=1e-4)
    num_mean_after = central_differences(lambda x: eval_fn_after(eps, beta, x, q_target.covar), q_target.mean,
                                         delta=1e-4)

    num_cov_before = central_differences(lambda x: eval_fn_before(py_mean, x), np.reshape(py_cov, -1), delta=1e-4)
    num_cov_before = np.reshape(num_cov_before, [dim, dim])

    num_cov_after = central_differences(lambda x: eval_fn_after(eps, beta, q_target.mean, x),
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


print("--------KL INACTIVE------------------")
run_test(eps=10.0)

print("--------KL ACTIVE------------------")
run_test(eps=0.1)


