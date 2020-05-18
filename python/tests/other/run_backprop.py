from util.central_differences import directed_central_differences, central_differences
from util.sample import sample_sympd
import numpy as np
from projection.MoreProjection import MoreProjection
from util.Gaussian import Gaussian
"""test backward pass"""

np.random.seed(0)

dim = 2

mean_old = np.random.uniform(low=-0.1, high=0.1, size=dim)
cov_old = sample_sympd(dim)
Q_old = np.linalg.inv(cov_old)
q_old = Q_old @ mean_old

mean_target = mean_old + np.random.uniform(low=-0.1, high=0.1, size=dim)
cov_target = 0.8 * cov_old
Q_target = np.linalg.inv(cov_target)
q_target = Q_target @ mean_target

proj_more = MoreProjection(dim)
err_mean = np.random.normal(size=dim)

err_cov = np.zeros([dim, dim])
err_cov[1, 0] = 1.0
err_cov = 0.5 * (err_cov + err_cov.T)

td = Gaussian(mean_target, cov_target)
od = Gaussian(mean_old, cov_old)
print("kl", td.kl(od), "entropy_diff", od.entropy() - td.entropy())

eps = 0.005
beta = Gaussian(mean_old, cov_old).entropy() - 10.0

proj_mean, proj_cov = proj_more.more_step(eps, beta, mean_old, cov_old, mean_target, cov_target)
proj_prec = np.linalg.inv(proj_cov)
pd = Gaussian(proj_mean, proj_cov)
print("kl to target", pd.kl(td), "kl to old", pd.kl(od))
grad_analytic_mean, grad_analytic_cov = proj_more.get_last_full_grad()
eta = proj_more.last_eta
omega = proj_more.last_omega
print("eta", eta, "omega", omega)
bw_analytic_mean, bw_analytic_cov = proj_more.backward(err_mean, err_cov)

bw_grad = err_mean.T @ grad_analytic_mean

#def eval_fn_mean_bw(_mean_target):
#    proj_mean, proj_cov = proj_more.more_step(eps, beta, mean_old, cov_old, _mean_target, cov_target)
#    return proj_mean

def eval_fn_mean_grad(_mean_target):
    proj_mean, proj_cov = proj_more.more_step(eps, beta, mean_old, cov_old, _mean_target, cov_target)
    return proj_mean

#bw_numeric = directed_central_differences(eval_fn_mean_bw, mean_target, err_mean)
grad_numeric = central_differences(eval_fn_mean_grad, mean_target, dim=dim)

print("bw_analytic_mean", bw_analytic_mean)
#print("bw_numeric", bw_numeric)
print("grad_analytic_mean", grad_analytic_mean)
print("grad_numeric", grad_numeric)
print("max diff grad", np.max(np.abs(grad_numeric - grad_analytic_mean)))
print(grad_numeric, "\n", grad_analytic_mean)
#print("max diff bw", np.max(np.abs(bw_analytic_mean - bw_numeric)))
print("max diff bw analytic, grad", np.max(np.abs(bw_analytic_mean - bw_grad)))
#print("max diff bw numeric bw grad", np.max(np.abs(bw_numeric - bw_grad)))

#def eval_fn_cov_bw(_cov_target):
#    ct = np.reshape(_cov_target, [dim, dim])
#    ct = 0.5 * (ct + ct.T)
#    proj_mean, proj_cov = proj_more.more_step(eps, beta, mean_old, cov_old, mean_target, ct)
#    return proj_cov

#def eval_fn_cov(_cov_target):
#    __cov_target = np.reshape(_cov_target, [dim, dim])
#    __cov_target = 0.5 * (__cov_target + __cov_target.T)
#    proj_mean, proj_cov = proj_more.more_step(eps, beta, mean_old, cov_old, mean_target, __cov_target)
#    return proj_cov

#grad_numeric = central_differences(eval_fn_cov, np.reshape(cov_target, -1), dim=[dim, dim])
#grad_numeric = np.reshape(grad_numeric, [dim] * 4)
#print("grad num")
#print(bw_analytic_cov)
#for i in range(dim):
#    for j in range(dim):
#        print(grad_numeric[i, j])
#        gan = grad_numeric[i, j]
#        print(i, np.max(np.abs(gan - bw_analytic_cov)))



#print(grad)

#t = proj_cov @ np.linalg.inv(cov_target)
#print(t @ t)
#print(np.max(np.abs((t@t)-grad)))
