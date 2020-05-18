import numpy as np
from util.Gaussian import Gaussian
from util.central_differences import central_differences
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection

"""test grads for optimal eta and omega w.r.t target distribution parameters q_tilde, Q_tilde"""

diff_delta = 1e-4  # delta for the central different gradient approximator, the whole thing is kind of sensitve to that

np.random.seed(0)
dim = 10

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)

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
    _, _ = proj_more.more_step(eps, beta, q_old.mean, q_old.covar,
                                            q_target.mean, q_target.covar)
    deq, deQ, doq, doQ = proj_more.get_last_eo_grad()

    eta = proj_more.last_eta
    omega = proj_more.last_omega
    print("Eta", eta, "Omega", omega)
    numerical_grads = central_differences(lambda p: eval_fn(p, eps, beta), p0, dim=2, delta=diff_delta)
    #print("d eta d q")
    #print("analytical", deq)
    #print("numerical", numerical_grads[0, :dim])
    #print("fraction", numerical_grads[0, :dim] / deq)
    print("max diff  d_eta d_q", np.max(np.abs(numerical_grads[0, :dim] - deq)))

    #print("d eta d Q")
    #print("analytical", deQ)
    #print("numerical", np.reshape(numerical_grads[0, dim:], [dim, dim]))
    #print("fraction", np.reshape(numerical_grads[0, dim:], [dim, dim]) / deQ)
    print("max diff d_eta d_Q", np.max(np.abs(np.reshape(numerical_grads[0, dim:], [dim, dim]) - deQ)))

    #print("d omega d q")
    #print("analytical", doq)
    #print("numerical", numerical_grads[1, :dim])
    #print("fraction", numerical_grads[1, :dim] / doq)
    print("max diff d_omega d_q", np.max(np.abs(numerical_grads[1, :dim] - doq)))

    #print("d omega d Q")
    #print("analytical", doQ)
    #print("numerical", np.reshape(numerical_grads[1, dim:], [dim ,dim]))
    print("max diff d_omega d_Q", np.max(np.abs(np.reshape(numerical_grads[1, dim:], [dim, dim]) - doQ)))

print("--------BOTH INACTIVE------------------")
run_test(eps=10.0, beta_loss=10.0)

print("--------ENTROPY ACTIVE------------------")
run_test(eps=10.0, beta_loss=0.01)


print("--------KL ACTIVE------------------")
run_test(eps=0.01, beta_loss=10.0)

print("--------Both ACTIVE------------------")
run_test(eps=0.01, beta_loss=0.001)
