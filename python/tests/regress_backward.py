import numpy as np
from util.Gaussian import Gaussian
from util.central_differences import central_differences
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection
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

proj_more = MoreProjection(dim)

proj_cvx = CVXProjector(dim)




def proj_wapper(eps, beta, old_mean, old_cov, target_mean, target_cov):
    cvx_mean, cvx_cov = proj_cvx.project(tf.constant(eps), tf.constant(beta), tf.constant(old_mean),
                                         tf.constant(old_cov), tf.constant(target_mean), tf.constant(target_cov))
    return np.array(cvx_mean), np.array(cvx_cov)


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
    proj_mean, proj_cov = proj_more.more_step(eps, beta, q_old.mean, q_old.covar,
                                              target_mean, target_cov)
    proj_mean = tf.constant(proj_mean)
    proj_cov = tf.constant(proj_cov)
    new_dist = tfd.MultivariateNormalFullCovariance(proj_mean, proj_cov)
    py_loss = tf.reduce_mean(new_dist.log_prob(samples))
    return np.array(py_loss)


def run_test(eps, beta_loss):
    beta = q_old.entropy() - beta_loss

    #torch_proj_more (pytorch layer with python custom op implementation)
    proj_net_model = ProjectionSimpleNet(dim, eps, beta, q_old)

    mean_torch  = torch.from_numpy(q_target.mean).double()
    covar_torch = torch.from_numpy(q_target.covar).double()
    covar_torch.requires_grad= True
    mean_torch.requires_grad = True

    def get_output_and_loss(mean_, covar_):
        torch_mean_before, torch_cov_before = proj_net_model(mean_, covar_) # NOTE: these non-leaf variables’ gradients can not be retained by torch by design.
        new_dist = MultivariateNormal(torch_mean_before.double(), torch_cov_before.double())
        torch_loss = torch.mean(new_dist.log_prob(torch.from_numpy(samples).double()))

        return torch_loss, torch_mean_before.detach().numpy(), torch_cov_before.detach().numpy()

    torch_loss, t_mean, t_cov = get_output_and_loss(mean_torch,covar_torch)
    torch_loss.backward()

    #proj_more
    new_mean, new_cov = proj_more.more_step(eps, beta, q_old.mean, q_old.covar,
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

    #cvx
    eps = tf.constant(eps)
    beta = tf.constant(beta)
    old_mean = tf.constant(q_old.mean)
    old_cov = tf.constant(q_old.covar)
    target_mean = tf.constant(q_target.mean)
    target_cov = tf.constant(q_target.covar)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(target_mean)
        tape.watch(target_cov)
        cvx_mean, cvx_cov = proj_cvx.project(eps, beta, old_mean, old_cov, target_mean, target_cov)
        cvx_dist = tfd.MultivariateNormalFullCovariance(cvx_mean, cvx_cov)
        cvx_loss = tf.reduce_mean(cvx_dist.log_prob(samples))
    cvx_g1 = tape.gradient(cvx_loss, [cvx_mean, cvx_cov])
    cvx_g2 = tape.gradient(cvx_loss, [target_mean, target_cov])

    print("loss", np.array(py_loss), np.array(cvx_loss), torch_loss)
    print("mean")
    print("forward:")
    print("diff py, cvx:", np.max(np.abs(np.array(py_mean) - np.array(cvx_mean))))
    print("diff torch_projection, cvx:", np.max(np.abs(t_mean - np.array(cvx_mean))))
    print("backward")
    print("before:")
    print("diff py, cvx", np.max(np.abs(np.array(py_g1[0]) - np.array(cvx_g1[0]))))
    print("diff py, num", np.max(np.abs(np.array(py_g1[0]) - num_mean_before)))


    print("after:")
    print("diff py, cvx", np.max(np.abs(np.array(py_g2[0]) - np.array(cvx_g2[0]))))
    print("diff py, num", np.max(np.abs(np.array(py_g2[0]) - num_mean_after)))
    print("diff py, tch", np.max(np.abs(np.array(py_g2[0]) - mean_torch.grad.detach().numpy())))

    print("covar")
    print("forward:")
    print("diff py, cvx:", np.max(np.abs(np.array(py_cov) - np.array(cvx_cov))))
    print("diff torch_projection, cvx:", np.max(np.abs(t_cov - np.array(cvx_cov))))
    print("backward")
    print("before:")
    print("diff py, cvx", np.max(np.abs(np.array(py_g1[1]) - np.array(cvx_g1[1]))))
    print("diff py, num", np.max(np.abs(np.array(py_g1[1]) - num_cov_before)))

    print("after:")
    print("diff py, cvx", np.max(np.abs(np.array(py_g2[1]) - np.array(cvx_g2[1]))))
    print("diff py, num", np.max(np.abs(np.array(py_g2[1]) - num_cov_after)))
    print("diff py, tch", np.max(np.abs(np.array(py_g2[1]) - covar_torch.grad.detach().numpy())))


print("--------BOTH INACTIVE------------------")
run_test(eps=10.0, beta_loss=10.0)

print("--------ENTROPY ACTIVE------------------")
run_test(eps=10.0, beta_loss=0.01)

print("--------KL ACTIVE------------------")
run_test(eps=0.01, beta_loss=10.0)

print("--------Both ACTIVE------------------")
run_test(eps=0.01, beta_loss=0.001)
