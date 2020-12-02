import numpy as np
import nlopt

from python.projection.ITPS import ITPS


class SplitMoreProjection(object):

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
        self._target_mean = target_mean
        self._target_covar = target_covar
        self._succ = True

        self._old_mean = old_mean
        self._old_precision = np.linalg.solve(old_covar, np.eye(self._dim))
        self._old_lin = self._old_precision @ self._old_mean

        self._old_chol_precision_t = np.linalg.cholesky(self._old_precision).T

        self._target_precision = np.linalg.inv(target_covar)
        self._target_lin = self._target_precision @ target_mean

        old_logdet = - 2 * np.sum(np.log(np.diagonal(self._old_chol_precision_t) + 1e-25))
        self._old_term = - 0.5 * (self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        # print("mean")
        try:
            opt_eta_mu = self.opt_dual(self._dual_mean)
            self._eta_mu = opt_eta_mu
            self._proj_mean = self._new_mean(opt_eta_mu)
            self._Q_mu = (self._eta_mu * self._old_precision + self._target_precision) / (self._eta_mu + 1)
            self._lin = self._Q_mu @ self._proj_mean
        except Exception:
            self._succ = False
            self._proj_mean = None
        # print("cov")
        try:
            opt_eta_sig = self.opt_dual(self._dual_cov)
            self._eta_sig = opt_eta_sig
            self._proj_covar = self._new_cov(opt_eta_sig)
            self._proj_precision = np.linalg.solve(self._proj_covar, np.eye(self._dim))
        except Exception:
            self._succ = False
            self._proj_covar = None

        return self._proj_mean, self._proj_covar

    def _new_mean(self, eta_mu):
        mat = self._target_precision + eta_mu * self._old_precision
        vec = self._target_lin + eta_mu * self._old_lin
        return np.linalg.solve(mat, vec)

    def _new_cov(self, eta_sig):
        return np.linalg.solve(self._target_precision + eta_sig * self._old_precision, np.eye(self._dim)) * (
                eta_sig + 1)

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
            # print("eta", eta_mu)
            # print("dual", dual)
            # print("grad", grad)
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

            # print("eta", eta_sig)
            # print("dual", dual)
            # print("grad", grad)
            return dual

        except np.linalg.LinAlgError as e:
            print(e)
            grad[0] = -1.0
            return 1e12

    def backward(self, d_mean: np.ndarray, d_cov: np.ndarray):
        """
        Error signal backward propagation based on values from last forward pass. This corresponds to the actual
        backward pass we would need in tensorflow
        TODO: Implement backward propagation of err_cov
        """
        assert self._succ, "INVALID STATE, No previous successful execution!"
        deta_mu_dq_target, deta_sig_dQ_target = self.get_last_eo_grad()

        Q_mu = (self._eta_mu * self._old_precision + self._target_precision) / (self._eta_mu + 1)
        covar_mu = np.linalg.solve(Q_mu, np.eye(self._dim))

        dq_deta_mu = (self._eta_mu * self._old_lin - self._target_lin) / (self._eta_mu + 1) ** 2
        dQ_deta_sig = (self._eta_sig * self._old_precision - self._target_precision) / (self._eta_sig + 1) ** 2

        # mean
        dq = covar_mu @ d_mean
        dQ = - self._proj_covar @ d_cov @ self._proj_covar
        tmp = np.outer(d_mean, Q_mu @ self._proj_mean)
        dQ_mu = - covar_mu @ (0.5 * tmp + 0.5 * tmp.T) @ covar_mu
        dQ_mu2 = - covar_mu @ (d_mean * self._target_lin) @ covar_mu

        deta_mu = np.dot(dq, dq_deta_mu) + np.trace(dQ_mu @ dq_deta_mu)
        deta_sig = np.trace(dQ @ dQ_deta_sig)

        dq_target = deta_mu * deta_mu_dq_target + dq / (self._eta_mu + 1)
        dQ_target = deta_sig * deta_sig_dQ_target + dQ / (self._eta_sig + 1) + dQ_mu / (self._eta_mu + 1)

        d_mu_target = self._target_precision @ dq_target
        tmp = np.outer(dq_target, self._target_mean)
        d_cov_target = - self._target_precision @ (0.5 * tmp + 0.5 * tmp.T + dQ_target) @ self._target_precision

        return d_mu_target, d_cov_target

    def get_last_eo_grad(self):
        """
        gradients of eta_mu and eta_sigma w.r.t. inputs, based on last forward pass
        For each case (except the first) there is a "baseline" implementation, which should correspond to the
        equations in the overleaf, and a tuned version, optimized to reduce number of performed operations
        """
        assert self._succ, "INVALID STATE, No previous successful execution!"
        if self._eta_mu == 0 and self._eta_sig == 0:  # case 1

            return np.zeros(self._dim), np.zeros([self._dim, self._dim])

        at, bt = np.zeros(self._dim), np.zeros([self._dim, self._dim])

        if self._eta_mu > 0:
            at = self.mu_grad_baseline()
            # at, bt, ct, dt = self._case2_tuned()
            # print(np.max(np.abs(cb - ct)), np.max(np.abs(cb - ct)))

        if self._eta_sig > 0:  # case 3
            bt = self.sig_grad_baseline()
            # print(np.max(np.abs(ab - at)), np.max(np.abs(bb - bt)))

        return at, bt

    def mu_grad_baseline(self):

        covar_mu = np.linalg.solve(self._Q_mu, np.eye(self._dim))
        dQ_mu_deta_mu = (self._old_precision - self._target_precision) / (self._eta_mu + 1) ** 2
        dsig_mu_deta_mu = - covar_mu @ dQ_mu_deta_mu @ covar_mu
        dq_deta_mu = (self._old_lin - self._target_lin) / (self._eta_mu + 1) ** 2

        dmean_dq = - covar_mu @ self._old_lin + covar_mu @ self._old_precision @ covar_mu @ self._lin

        tmp1 = self._lin @ covar_mu @ dQ_mu_deta_mu @ covar_mu @ self._old_lin
        tmp2 = self._lin @ dsig_mu_deta_mu @ self._old_precision @ covar_mu @ self._lin

        # TODO: not sure this is faster, no difference in error, though
        # lin_old = np.outer(self._lin, self._old_lin)
        # tmp1 = covar_mu @ (0.5 * (lin_old + lin_old.T)) @ covar_mu
        # lin_lin = self._old_precision @ covar_mu @ np.outer(self._lin, self._lin)
        # tmp2 = - 0.5 * covar_mu @ (lin_lin + lin_lin.T) @ covar_mu
        # dcov_dQ_mu = tmp1 + tmp2

        dmean_deta_mu = np.dot(dmean_dq, dq_deta_mu) + tmp1 + tmp2
        # dmean_deta_mu = np.dot(dmean_dq, dq_deta_mu) + np.trace(dcov_dQ_mu @ dQ_mu_deta_mu)

        lhs = dmean_dq / (self._eta_mu + 1)
        deta_mu_dq_target = lhs / -dmean_deta_mu

        # return deta_mu_dq_target / (self._eta_mu + 1)
        return deta_mu_dq_target

    def mu_grad_tuned(self):
        # Q_mu stuff
        covar_mu = np.linalg.solve(self._Q_mu, np.eye(self._dim))
        dQ_mu_deta_mu = (self._old_precision - self._target_precision) / (self._eta_mu + 1) ** 2
        dsig_mu_deta_mu = - covar_mu @ dQ_mu_deta_mu @ covar_mu

        dq_deta_mu = (self._old_lin - self._target_lin) / (self._eta_mu + 1) ** 2
        dmean_dq = - covar_mu @ self._old_lin + covar_mu @ self._old_precision @ covar_mu @ self._lin

        tmp1 = self._lin @ covar_mu @ dQ_mu_deta_mu @ covar_mu @ self._old_lin
        tmp2 = self._lin @ dsig_mu_deta_mu @ self._old_precision @ covar_mu @ self._lin
        # tmp2 = 0.5 * self._lin @ dsig_mu_deta_mu @ self._old_precision @ covar_mu @ self._lin
        # tmp3 = 0.5 * self._lin @ covar_mu @ self._old_precision @ dsig_mu_deta_mu @ self._lin

        # old_cov = np.linalg.inv(self._old_precision)
        # tmp2 = - self._lin @ self._Q_mu @ old_cov @ self._Q_mu @ old_cov @ self._Q_mu @ dQ_mu_deta_mu @ self._Q_mu @ old_cov @ self._Q_mu @ self._lin
        dmean_deta_mu = dq_deta_mu @ dmean_dq + tmp1 + tmp2

        lhs = dmean_dq / (self._eta_mu + 1)
        deta_mu_dq_target = lhs / -dmean_deta_mu

        # return deta_mu_dq_target / (self._eta_mu + 1)
        return deta_mu_dq_target

    def sig_grad_baseline(self):
        dQ_deta_sig = (self._old_precision - self._target_precision) / (self._eta_sig + 1) ** 2
        dcov_dQ = 0.5 * (-self._proj_covar @ self._old_precision @ self._proj_covar + self._proj_covar)
        dcov_deta_sig = 0.5 * np.trace(
            (-self._proj_covar @ self._old_precision @ self._proj_covar + self._proj_covar) @ dQ_deta_sig)

        lhs = dcov_dQ / (self._eta_sig + 1)
        deta_sig_dQ_target = lhs / -dcov_deta_sig

        return deta_sig_dQ_target

    def sig_grad_tuned(self):
        dQ_deta_sig = (self._old_precision - self._proj_precision) / (self._eta_sig + 1)
        dcov_dQ = -self._proj_covar @ self._old_precision @ self._proj_covar + self._proj_covar
        dcov_deta_sig = np.trace(dcov_dQ @ dQ_deta_sig)

        lhs = dcov_dQ / (self._eta_sig + 1)
        deta_sig_dQ_target = lhs / -dcov_deta_sig

        return deta_sig_dQ_target

    @property
    def last_eta_mu(self):
        return self._eta_mu

    @property
    def last_eta_sig(self):
        return self._eta_sig

    @property
    def success(self):
        return self._succ
