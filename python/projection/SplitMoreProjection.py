import numpy as np
from projection.ITPS import ITPS
import nlopt

class SplitMoreProjection():

    def __init__(self, dim):

        self._lp = None
        self._grad = None
        self._succ = False
        self._dim = dim
        self._dual_const_part = dim * np.log(2 * np.pi)
        self._entropy_const_part = 0.5 * (self._dual_const_part + dim)

    def opt_dual(self, fn):
        opt = nlopt.opt(nlopt.LD_LBFGS, 1)
        opt.set_lower_bounds(0.0)
        opt.set_upper_bounds(1e12)
        opt.set_ftol_abs(1e-12)
        opt.set_xtol_abs(1e-12)
        opt.set_maxeval(10000)
        opt.set_min_objective(fn)
        try:
            opt_lp = opt.optimize([10.0])
            opt_lp = opt_lp[0]
            return opt_lp
        except Exception as e:
            # NLOPT somtimes throws error very close to convergence, we check for this and return preliminary result
            # if its close enough:
            # 1.Case: Gradient near 0
            # 2.Case: eta near bound and d_omega near 0
            # 3.Case: omega near bound and d_eta near 0
            if np.abs(self._grad[0]) < ITPS.grad_bound or self._lp < ITPS.value_bound:
                return self._lp
            else:
                raise e

    def more_step(self, eps_mu, eps_sigma, old_mean, old_covar, target_mean, target_covar):
        self._eps_mu = eps_mu
        self._eps_sigma = eps_sigma
        self._succ = True

        self._old_mean = old_mean
        self._old_precision = np.linalg.inv(old_covar)
        self._old_lin = self._old_precision @ self._old_mean

        self._old_chol_precision_t = np.linalg.cholesky(self._old_precision).T

        self._target_precision = np.linalg.inv(target_covar)
        self._target_lin = self._target_precision @ target_mean

        old_logdet = - 2 * np.sum(np.log(np.diagonal(self._old_chol_precision_t) + 1e-25))
        self._old_term = - 0.5 * (self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        print("mean")
        try:
            opt_eta_mu = self.opt_dual(self._dual_mean)
            self._proj_mean = self._new_mean(opt_eta_mu)
        except Exception:
            self._succ = False
            self._proj_mean = None
        print("cov")
        try:
            opt_eta_sig = self.opt_dual(self._dual_cov)
            self._proj_covar = self._new_cov(opt_eta_sig)
        except Exception:
            self._succ = False
            self._proj_covar = None
        return self._proj_mean, self._proj_covar

    def _new_mean(self, eta_mu):
        mat = self._target_precision + eta_mu * self._old_precision
        vec = self._target_lin + eta_mu * self._old_lin
        return np.linalg.solve(mat, vec)

    def _new_cov(self, eta_sig):
        return np.linalg.inv(self._target_precision + eta_sig * self._old_precision) * (eta_sig + 1)

    def _dual_mean(self, eta_mu, grad):

        eta_mu = eta_mu[0] if eta_mu[0] > 0.0 else 0.0
        self._lp = eta_mu

        try:

            proj_lin = self._target_lin + eta_mu * self._old_lin
            proj_mean = np.linalg.solve(self._target_precision + eta_mu * self._old_precision, proj_lin)

            dual = eta_mu * self._eps_mu - 0.5 * eta_mu * np.dot(self._old_lin, self._old_mean)
            dual += 0.5 * np.dot(proj_lin, proj_mean)

            grad[0] = self._eps_mu - 0.5 * np.sum(np.square(self._old_chol_precision_t @ (self._old_mean - proj_mean)))
            self._grad = grad
            #print("eta", eta_mu)
            #print("dual", dual)
            #print("grad", grad)
            return dual
        except np.linalg.LinAlgError as e:
            grad[0] = -1.0
            return 1e12

    def _dual_cov(self, eta_sig, grad):
        eta_sig = eta_sig[0] if eta_sig[0] > 0.0 else 0.0
        self._lp = eta_sig
        try:
            proj_cov = self._new_cov(eta_sig)
            proj_cov_chol = np.linalg.cholesky(proj_cov)
            new_logdet = 2 * np.sum(np.log(np.diagonal(proj_cov_chol) + 1e-25))

            dual = eta_sig * self._eps_sigma + eta_sig * self._old_term
            dual += 0.5 * (eta_sig + 1) * (self._dual_const_part + new_logdet)

            trace_term = np.sum(np.square(self._old_chol_precision_t @ proj_cov_chol))
            grad[0] = self._eps_sigma - 0.5 * (self._kl_const_part - new_logdet + trace_term)
            self._grad = grad

            #print("eta", eta_sig)
            #print("dual", dual)
            #print("grad", grad)
            return dual

        except np.linalg.LinAlgError as e:
            print(e)
            grad[0] = -1.0
            return 1e12







