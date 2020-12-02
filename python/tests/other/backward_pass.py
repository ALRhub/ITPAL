import numpy as np
from util.Gaussian import Gaussian
from util.central_differences import central_differences, directed_central_differences
from util.sample import sample_sympd
from projection.MoreProjection import MoreProjection
import cpp_projection

"""test grads for optimal eta and omega w.r.t target distribution parameters q_tilde, Q_tilde"""

np.random.seed(0)
dim = 2

mean_old = np.random.uniform(low=-1, high=1, size=dim)
cov_old = sample_sympd(dim)
#Q_old = np.linalg.inv(cov_old)
#q_old = Q_old @ mean_old

mean_target = np.random.uniform(low=-1, high=1, size=dim)
cov_target = sample_sympd(dim)

q_old = Gaussian(mean_old, cov_old)
q_target = Gaussian(mean_target, cov_target)

py_proj_more = MoreProjection(dim)
cpp_proj_more = cpp_projection.MoreProjection(dim)

def run_test(eps, beta_loss):
    beta = q_old.entropy() - beta_loss
    py_new_mean, py_new_cov = py_proj_more.more_step(eps, beta, q_old.mean, q_old.covar, q_target.mean, q_target.covar)
    cpp_new_mean, cpp_new_cov = cpp_proj_more.forward(eps, beta, q_old.mean, q_old.covar, q_target.mean, q_target.covar)

    print("Max diff projected mean", np.max(np.abs(py_new_mean - cpp_new_mean)))
    print("Max diff projected covar", np.max(np.abs(py_new_cov - cpp_new_cov)))

    py_eta = py_proj_more.last_eta
    py_omega = py_proj_more.last_omega

    cpp_eta = cpp_proj_more.last_eta
    cpp_omega = cpp_proj_more.last_omega
    print("Eta: Py:", py_eta, "cpp:", cpp_eta, "Omega: Py:", py_omega, "cpp:", cpp_omega)

    mean_err = np.ones(dim)
    cov_err = np.ones([dim, dim])
    #cov_err[1, 0] = 0.5
    #cov_err[0, 1] = 0.5

    cpp_mean_bw, cpp_cov_bw = cpp_proj_more.backward(mean_err, cov_err)
    py_mean_bw, py_cov_bw = py_proj_more.backward(mean_err, cov_err)

    print("Max diff backward mean", np.max(np.abs(cpp_mean_bw - py_mean_bw)))

    #delta = 1e-4
    #norm = np.linalg.norm(np.concatenate([py_mean_bw, np.reshape(py_cov_bw, -1)]))
    #mean_dir = delta * py_mean_bw #/ norm
    #cov_dir = delta * py_cov_bw #/ norm
    #p_u = py_proj_more.more_step(eps_mu, beta, q_old.mean, q_old.covar, q_target.mean + mean_dir, q_target.covar + cov_dir)
    #p_l = py_proj_more.more_step(eps_mu, beta, q_old.mean, q_old.covar, q_target.mean - mean_dir, q_target.covar - cov_dir)
    #num_mean_err = (p_u[0] - p_l[0]) / (2 * delta)
    #num_cov_err = (p_u[1] - p_l[1]) / (2 * delta)

    #print(num_mean_err)
    #print(num_cov_err)

    #print()

    #print(cpp_mean_bw)

    #py_mean_grad = py_proj_more.get_last_full_grad()[0]
    #eval_fn = lambda x: py_proj_more.more_step(eps_mu, beta, q_old.mean, q_old.covar, x, q_target.covar)[0]
    #py_mean_num_grad = central_differences(eval_fn, q_target.mean, dim=dim)
    #print(py_mean_grad, "\n", py_mean_num_grad)
    #print("Max diff grad mean numeric python", np.max(np.abs(py_mean_grad - py_mean_num_grad)))

#    py_mean_num_bw = directed_central_differences(eval_fn, q_target.mean, mean_err)
    #py_mean_num_bw = mean_err @ py_mean_grad
  #  py_mean_num_bw2 = py_mean_grad @ mean_err
    #print(py_mean_bw, "\n", py_mean_num_bw)
    #print("Max diff backward mean numeric python", np.max(np.abs(py_mean_bw - py_mean_num_bw)))

    #eval_fn = lambda x: py_proj_more.more_step(eps_mu, beta, q_old.mean, q_old.covar, q_target.mean, x)[1]
    #py_cov_num_bw = directed_central_differences(eval_fn, q_target.covar, cov_err)
   # print(py_cov_bw[::-1, ::-1], "\n", py_cov_num_bw)
    #print("Max diff backward cov numeric python", np.max(np.abs(py_cov_bw - py_cov_num_bw)))

    #eval_fn = lambda x: cpp_proj_more.forward(eps_mu, beta, q_old.mean, q_old.covar, x, q_target.covar)[0]
    #cpp_mean_num_bw = directed_central_differences(eval_fn, q_target.mean, mean_err)
    #print("Max diff backward mean numeric cpp", np.max(np.abs(cpp_mean_bw - cpp_mean_num_bw)))

    #eval_fn = lambda x: cpp_proj_more.forward(eps_mu, beta, q_old.mean, q_old.covar, q_target.mean, x)[1]
    #cpp_cov_num_bw = directed_central_differences(eval_fn, q_target.covar, cov_err)
    #print("Max diff backward cov numeric cpp", np.max(np.abs(cpp_cov_bw - cpp_cov_num_bw)))



    #numerical_grads = central_differences(lambda p: eval_fn(p, eps_mu, beta), p0, dim=2, delta=1e-9)


    #print("max diff  d_eta d_q", np.max(np.abs(numerical_grads[0, :dim] - deq)))

    #print("d eta d Q")
    #print("analytical", deQ)
    #print("numerical", np.reshape(numerical_grads[0, dim:], [dim, dim]))
    #print("fraction", np.reshape(numerical_grads[0, dim:], [dim, dim]) / deQ)
    #print("max diff d_eta d_Q", np.max(np.abs(np.reshape(numerical_grads[0, dim:], [dim, dim]) - deQ)))

    #print("d omega d q")
    #print("analytical", doq)
    #print("numerical", numerical_grads[1, :dim])
    #print("fraction", numerical_grads[1, :dim] / doq)
    #print("max diff d_omega d_q", np.max(np.abs(numerical_grads[1, :dim] - doq)))

    #print("d omega d Q")
   # print("max diff d_omega d_Q", np.max(np.abs(np.reshape(numerical_grads[1, dim:], [dim, dim]) - doQ)))

print("--------BOTH INACTIVE------------------")
run_test(eps=10.0, beta_loss=10.0)

print("--------ENTROPY ACTIVE------------------")
run_test(eps=10.0, beta_loss=0.01)

print("--------KL ACTIVE------------------")
run_test(eps=0.01, beta_loss=10.0)

print("--------Both ACTIVE------------------")
run_test(eps=0.01, beta_loss=0.001)
