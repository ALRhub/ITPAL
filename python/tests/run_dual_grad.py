import numpy as np
from util.Gaussian import Gaussian
from util.central_differences import central_differences
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection

#m1 = np.array([-0.5, 1])
#c1 = np.array([[0.3, -0.1],
#               [-0.1, 0.5]])

#m2 = np.array([0.3, -0.6])
#c2 = np.array([[0.4, 0.05],
#               [0.05, 0.2]])
np.random.seed(1)
dim = 4

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)
#Q_old = np.linalg.inv(cov_old)
#q_old = Q_old @ mean_old

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)

proj_more = MoreProjection(dim)

def eval_fn(p, eps, beta):
    lin = p[:dim]
    prec = np.reshape(p[dim:], [dim, dim])
    prec = 0.5 * (prec + prec.T)

    covar = np.linalg.inv(prec)
    mean = covar @ lin

    _, _ = proj_more.more_step(eps, beta, q_old.mean, q_old.covar, mean, covar)
    assert proj_more.success
    return np.array([proj_more.last_eta, proj_more.last_omega])


p0 = np.concatenate([q_target.lin_term, np.reshape(q_target.precision, [-1])], axis=-1)

def run_test(eps, beta_loss):
    beta = q_old.entropy() - beta_loss
    new_mean, new_cov = proj_more.more_step(eps, beta, q_old.mean, q_old.covar,
                                            q_target.mean, q_target.covar)
    deq, deQ, doq, doQ = proj_more.get_last_eo_grad()

    eta = proj_more.last_eta
    omega = proj_more.last_omega
    print("Eta", eta, "Omega", omega)
    numerical_grads = central_differences(lambda p: eval_fn(p, eps, beta), p0, dim=2, delta=1e-9)
    print("analytical", deq)
    print("numerical", numerical_grads[0, :dim])
    print("fraction", numerical_grads[0, :dim] / deq)
    print("d_eta d_q", np.max(np.abs(numerical_grads[0, :dim] - deq)))
    #print("d_eta d_Q", np.max(np.abs(np.reshape(numerical_grads[0, dim:], [dim, dim]) - deQ)))
    print("analytical", doq)
    print("numerical", numerical_grads[1, :dim])
    print("fraction", numerical_grads[1, :dim] / doq)
    print("d_omega d_q", np.max(np.abs(numerical_grads[1, :dim] - doq)))
    #print("d_omega d_Q", np.max(np.abs(np.reshape(numerical_grads[1, dim:], [dim, dim]) - doQ)))

print("--------BOTH INACTIVE------------------")
#run_test(eps=10.0, beta_loss=10.0)

print("--------ENTROPY ACTIVE------------------")
#run_test(eps=10.0, beta_loss=0.01)


print("--------KL ACTIVE------------------")
#run_test(eps=0.01, beta_loss=10.0)

print("--------Both ACTIVE------------------")
run_test(eps=0.01, beta_loss=0.01)


"""
print("analytical:")
print("d_eta d_q", np.zeros(dim))
print("d_eta d_Q", np.zeros([dim, dim]))
print("d_omega d_q", np.zeros(dim))
print("d_omega d_Q", 0.5 * q.covar)
"""

""" Test 2: Entropy Inactive, KL active"""
"""
eps = 0.1
beta = q_old.entropy() - 1.0

new_mean, new_cov = proj_more.more_step(eps, beta, q_old, q_target.lin_term, q_target.precision)
q = Gaussian(new_mean, new_cov)

eta = proj_more.last_eta
omega = proj_more.last_omega
print("Eta", eta, "Omega", omega)
numerical_grads = central_differences(lambda p: eval_fn(p, eps, beta), p0, dim=2, delta=1e-9)

print("numerical:")
print("d_eta d_q", numerical_grads[:dim, 0])
print("d_eta d_Q", np.reshape(numerical_grads[dim:, 0], [dim, dim]))
print("d_omega d_q", numerical_grads[:dim, 1])
print("d_omega d_Q", np.reshape(numerical_grads[dim:, 1], [dim, dim]))

dq_deta = ((omega + 1) * q_old.lin_term - q_target.lin_term) / ((omega + eta + 1)**2)
dQ_deta = ((omega + 1) * q_old.precision - q_target.precision) / ((omega + eta + 1)**2)

dtm1_dq = np.eye(dim) / (omega + eta + 1) @ q.covar @ q_old.lin_term
dtm2_dq = 2 * np.eye(dim) / (omega + eta + 1) @ q.covar @ q_old.precision @ q.covar @ q.lin_term
deta_dq = - 2 * dtm1_dq + dtm2_dq

dtlogdet_dQ = q.covar / (omega + eta + 1)
dttrace_dQ = - q.covar @ q_old.precision @ q.covar / (omega + eta + 1)
dtm1_dQ = - q.covar @ np.outer(q.lin_term, q_old.lin_term) @ q.covar / (omega + eta + 1)
tmp = q_old.precision @ q.covar @ np.outer(q.lin_term, q.lin_term)
dtm2_dQ = - q.covar @ (tmp + tmp.T) @ q.covar / (omega + eta + 1)
deta_dQ = dtlogdet_dQ + dttrace_dQ - 2 * dtm1_dQ + dtm2_dQ

dlogdet_deta = np.trace(q.covar @ dQ_deta)
dtrace_deta = np.trace(- q.covar @ q_old.precision @ q.covar @ dQ_deta)
dtm1_deta = np.dot(q.covar @ q_old.lin_term, dq_deta) + \
            np.trace(- q.covar @ np.outer(q.lin_term, q_old.lin_term) @ q.covar @ dQ_deta)
tmp = q_old.precision @ q.covar @ np.outer(q.lin_term, q.lin_term)
dtm2_deta = np.dot(2 * q.covar @ q_old.precision @ q.covar @ q.lin_term, dq_deta) + \
             np.trace(- q.covar @ (tmp + tmp.T) @ q.covar @ dQ_deta)
c_deta_dq = - 1 / (dlogdet_deta + dtrace_deta - 2 * dtm1_deta + dtm2_deta)

deta_dq = deta_dq * c_deta_dq
deta_dQ = deta_dQ * c_deta_dq
print("analytical:")
print("d_eta d_q", deta_dq)
print("d_eta d_Q", deta_dQ)
print("d_omega d_q", np.zeros(dim))
print("d_omega d_Q", np.zeros([dim, dim]))











"""
"""
print(proj_more.last_eta, proj_more.last_omega)
print(new_mean, new_cov)
new_lin = np.linalg.inv(new_cov) @ new_mean
new_precision = np.linalg.inv(new_cov)
print("d_omega d_tildeQ:", new_cov / 2.0)


dQ = ((omega + 1) * old_dist.precision - target_dist.precision) / ((eta + omega + 1)**2)
dq = ((omega + 1) * old_dist.lin_term - target_dist.lin_term) / ((eta + omega + 1)**2)
# Q terms

lhs_logdet = new_cov @ dQ
lhs_trace = -new_cov @ old_dist.precision @ new_cov @ dQ
lhs_m1 = new_cov @ np.outer(old_dist.lin_term, dq) - new_cov @ np.outer(new_lin, old_dist.lin_term) @ new_cov @ dQ
t = old_dist.precision @ new_cov @ np.outer(new_lin, new_lin)
lhs_m2 = 2 * new_cov @ old_dist.precision @ new_cov @ np.outer(new_lin, dq) - new_cov @ (t + t.T) @ new_cov @ dQ
lhs = - 0.5 * (lhs_logdet + lhs_trace - 2 * lhs_m1 + lhs_m2)

rhs_logdet = 0
rhs_trace = 0
rhs_m1 = new_cov @ old_dist.lin_term / (eta + omega + 1)
rhs_m2 = 2 * new_cov @ old_dist.precision @ new_cov @ new_lin / (eta + omega + 1)

rhs = 0.5 * (rhs_logdet + rhs_trace - 2 * rhs_m1 + rhs_m2)  # -1

d_eta = np.linalg.solve(lhs, rhs)
print("d_eta", d_eta)




p0 = np.concatenate([target_dist.lin_term,
                     np.array([target_dist.precision[0, 0],
                               target_dist.precision[1, 1],
                               target_dist.precision[0, 1]])])

print(project(p0))
cd_grad = central_differences(project, p0, dim=2, delta=1e-9)
print(cd_grad)

#print(cd_grad[2:4, 0] - np.diag(x))
"""