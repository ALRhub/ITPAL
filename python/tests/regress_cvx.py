import numpy as np
from util.Gaussian import Gaussian
from util.central_differences import central_differences
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection
from cvx.cvxlayers_projection import CVXProjector
import tensorflow as tf
"""test grads for optimal eta and omega w.r.t target distribution parameters q_tilde, Q_tilde"""

np.random.seed(0)
dim = 3

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)
#Q_old = np.linalg.inv(cov_old)
#q_old = Q_old @ mean_old

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)

proj_more = MoreProjection(dim)

proj_cvx = CVXProjector(dim)


def proj_wapper(eps, beta, old_mean, old_cov, target_mean, target_cov):
    cvx_mean, cvx_cov = proj_cvx.project(tf.constant(eps), tf.constant(beta), tf.constant(old_mean),
                                         tf.constant(old_cov), tf.constant(target_mean), tf.constant(target_cov))
    return np.array(cvx_mean), np.array(cvx_cov)


def sym_wrapper(x):
    x = np.reshape(x, [dim, dim])
    return 0.5 * (x + x.T)


def run_test(eps, beta_loss):
    beta = q_old.entropy() - beta_loss
    new_mean, new_cov = proj_more.more_step(eps, beta, q_old.mean, q_old.covar,
                                            q_target.mean, q_target.covar)
    cvx_mean, cvx_cov = proj_cvx.project(tf.constant(eps), tf.constant(beta), tf.constant(q_old.mean),
                                         tf.constant(q_old.covar), tf.constant(q_target.mean), tf.constant(q_target.covar))
    new_dist = Gaussian(new_mean, new_cov)
    cvx_dist = Gaussian(np.array(cvx_mean), np.array(cvx_cov))

    print("forward")
    print("to old", new_dist.kl(q_old), cvx_dist.kl(q_old))
    print("to target", new_dist.kl(q_target), cvx_dist.kl(q_target))
    print("mean max diff", np.max(np.abs(new_mean - cvx_mean)))
    print("cov max diff", np.max(np.abs(new_cov - cvx_cov)))

    print(proj_cvx.backward(tf.constant(eps), tf.constant(beta), tf.constant(q_old.mean),
                            tf.constant(q_old.covar), tf.constant(q_target.mean), tf.constant(q_target.covar)))

    grad_py = proj_more.get_last_full_grad()[0]
    dm_dm_target, dm_dcov_target, dcov_dm_target, dcov_dcov_target = \
        proj_cvx.get_grad(tf.constant(eps), tf.constant(beta), tf.constant(q_old.mean),
                          tf.constant(q_old.covar), tf.constant(q_target.mean), tf.constant(q_target.covar))

    eval_fn = lambda x: (proj_wapper(eps, beta, q_old.mean, q_old.covar, x, q_target.covar)[0])
    cvx_num_dm_dm_target = central_differences(eval_fn, q_target.mean, dim=dim, delta=1e-4)

    eval_fn = lambda x: (proj_wapper(eps, beta, q_old.mean, q_old.covar, q_target.mean, sym_wrapper(x))[0])
    cvx_num_dm_dcov_target = central_differences(eval_fn, np.reshape(q_target.covar, -1), dim=dim, delta=1e-6)
    cvx_num_dm_dcov_target = np.reshape(cvx_num_dm_dcov_target, [dim, dim, dim])

    #print(dm_dcov_target.numpy())
    #print(cvx_num_dm_dcov_target)

    py_num_grad = central_differences(
        lambda x: proj_more.more_step(eps, beta, q_old.mean, q_old.covar, x, q_target.covar)[0],
        q_target.mean, dim=dim, delta=1e-4)

    print("grad")
    print("grad max diff", np.max(np.abs(grad_py - dm_dm_target)))
    print("grad py num max diff", np.max(np.abs(grad_py - py_num_grad)))
    print("grad cvx num max diff", np.max(np.abs(dm_dm_target - cvx_num_dm_dm_target)))


print("--------BOTH INACTIVE------------------")
run_test(eps=10.0, beta_loss=10.0)

print("--------ENTROPY ACTIVE------------------")
run_test(eps=10.0, beta_loss=0.01)


print("--------KL ACTIVE------------------")
run_test(eps=0.01, beta_loss=10.0)

print("--------Both ACTIVE------------------")
run_test(eps=0.01, beta_loss=0.001)
