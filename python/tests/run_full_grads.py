from util.central_differences import central_differences
from util.sample import sample_sympd
import numpy as np
from projection.MoreProjection import MoreProjection

"""test full gradients output distribution w.r.t input distribution and backward pass"""

np.random.seed(0)

dim = 3

mean_old = np.random.uniform(low=-0.1, high=0.1, size=dim)
cov_old = sample_sympd(dim)
Q_old = np.linalg.inv(cov_old)
q_old = Q_old @ mean_old

mean_target = np.random.uniform(low=-0.1, high=0.1, size=dim)
cov_target = sample_sympd(dim)
Q_target = np.linalg.inv(cov_target)
q_target = Q_target @ mean_target
#mu_target = np.linalg.solve(Q_target, q_target)

eta = 12.2456
omega = 0.0

def get_q(eta_, omega_, q_target_):
    return (eta_ * q_old + q_target_) / (eta_ + omega_ + 1)

def get_Q(eta_, omega_, Q_target_):
    return (eta_ * Q_old + Q_target_) / (eta_ + omega_ + 1)

q = get_q(eta, omega, q_target)
Q = get_Q(eta, omega, Q_target)

mu = np.linalg.solve(Q, q)

print("d_mu d_q", end=" ")
grad_numeric = central_differences(lambda x: np.linalg.solve(Q, x), q, dim=dim)
grad_analytic = np.linalg.inv(Q)
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_q d_eo", end=" ")
grad_numeric = central_differences(lambda x: (x[0] * q_old + q_target) / (x[0] + x[1] + 1), np.array([eta, omega]), dim=dim)
grad_analytic = np.stack([(omega + 1) * q_old - q_target, - (eta * q_old + q_target)], axis=-1) / ((eta + omega + 1)**2)
print(np.max(np.abs(grad_numeric - grad_analytic)))

print("d_q_tilde d_mu_tilde", end=" ")
grad_numeric = central_differences(lambda x: np.linalg.solve(cov_target, x), mean_target, dim=dim)
grad_analytic = Q_target
print(np.max(np.abs(np.reshape(grad_numeric, [dim, dim]) - grad_analytic)))

print("d_mu d_mutarget")
proj_more = MoreProjection(dim)
err_mean = np.random.normal(size=dim)

proj_mean, proj_cov = proj_more.more_step(0.1, -5.0, mean_old, cov_old, mean_target, cov_target)
#print(proj_more.last_eta, proj_more.last_omega)
grad_analytic, _ = proj_more.get_last_full_grad()
print("grad analytic, t", err_mean.T @ grad_analytic)
print("backward", proj_more.backward(err_mean, np.zeros([dim, dim]))[0])
print("grad analytic", grad_analytic)


def eval_fn(_mean_target):
    proj_mean, proj_cov = proj_more.more_step(0.1, -5.0, mean_old, cov_old, _mean_target, cov_target)
    return proj_mean

grad_numeric = central_differences(eval_fn, mean_target, dim=dim)
print("grad numeric", grad_numeric)




#print("d_Sigma d_Q", end=" ")
#grad_numeric = central_differences(lambda x: np.linalg.inv(np.reshape(x, [dim, dim])), np.reshape(Q, -1), dim=[dim, dim])
#print()
#print(grad_numeric)
