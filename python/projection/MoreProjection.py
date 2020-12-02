import numpy as np
from python.projection.ITPS import ITPS


class MoreProjection(ITPS):

    def __init__(self, dim):
        super().__init__(0.0, 1.0, True)

        self._dim = dim
        self._dual_const_part = dim * np.log(2 * np.pi)
        self._entropy_const_part = 0.5 * (self._dual_const_part + dim)

    def more_step(self, eps, beta, old_mean, old_covar, target_mean, target_covar):
        self._eps = eps
        self._beta = beta
        self._succ = False

        self._old_mean = old_mean
        self._old_precision = np.linalg.inv(old_covar)
        self._old_lin = self._old_precision @ self._old_mean

        self._old_chol_precision_t = np.linalg.cholesky(self._old_precision).T

        self._target_mean = target_mean
        self._target_precision = np.linalg.inv(target_covar)
        self._target_lin = self._target_precision @ target_mean

        old_logdet = - np.linalg.slogdet(self._old_precision)[1]
        self._old_term = -0.5 * (np.dot(self._old_lin, self._old_mean) + self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        try:
            opt_eta, opt_omega = self.opt_dual()
            self._eta = opt_eta
            self._omega = opt_omega
            self._proj_lin, self._proj_precision = \
                self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            self._proj_covar = np.linalg.inv(self._proj_precision)
            self._proj_mean = self._proj_covar @ self._proj_lin
            self._succ = True
            return self._proj_mean, self._proj_covar
        except Exception:
            self._succ = False
            return None, None

    def _new_params(self, eta, omega):
        new_lin = (eta * self._old_lin + self._target_lin) / (eta + omega)
        new_precision = (eta * self._old_precision + self._target_precision) / (eta + omega)
        return new_lin, new_precision

    def _dual(self, eta_omega, grad):
        """
        dual of the more problem
        """
        eta = eta_omega[0] if eta_omega[0] > 0.0 else 0.0
        omega = eta_omega[1] if eta_omega[1] > 0.0 else 0.0
        self._eta = eta
        self._omega = omega

        eta_off = eta + self._eta_offset
        omega_off = omega + self._omega_offset

        new_lin, new_precision = self._new_params(eta_off, omega_off)
        try:
            new_covar = np.linalg.inv(new_precision)
            new_chol_covar = np.linalg.cholesky(new_covar)

            new_mean = new_covar @ new_lin
            new_logdet = 2 * np.sum(np.log(np.diagonal(new_chol_covar) + 1e-25))

            dual = eta * self._eps - omega * self._beta + eta_off * self._old_term
            dual += 0.5 * (eta_off + omega_off) * (self._dual_const_part + new_logdet + np.dot(new_lin, new_mean))

            trace_term = np.sum(np.square(self._old_chol_precision_t @ new_chol_covar))
            kl = self._kl_const_part - new_logdet + trace_term
            diff = self._old_mean - new_mean
            kl = 0.5 * (kl + np.sum(np.square(self._old_chol_precision_t @ diff)))

            grad[0] = self._eps - kl
            grad[1] = (self._entropy_const_part + 0.5 * new_logdet - self._beta) if self._constrain_entropy else 0.0
            self._grad = grad
            return dual

        except np.linalg.LinAlgError as e:
            grad[0] = -1.0
            grad[1] = 0.0
            return 1e12

    def get_last_full_grad(self):
        """
        Full gradient of projected mean and covariance w.r.t. input mean and covariance based on values from last
        forward pass.
        TODO: Implement d Sigma d tilde Sigma
        """
        assert self._succ, "INVALID STATE, No previous successfull execution!"

        deta_dq_target, deta_dQ_target, domega_dq_target, domega_dQ_target = self.get_last_eo_grad()
        deo_dq_target = np.stack([deta_dq_target, domega_dq_target], axis=0)  # 2 x d
        deo_dQ_target = np.stack([np.reshape(deta_dQ_target, -1), np.reshape(domega_dQ_target, -1)], axis=0)  # 2 x d**2

        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dq_domega = -(self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dq_deo = np.stack([dq_deta, dq_domega], axis=1)  # d x 2

        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (
                    (self._omega + self._eta + 1) ** 2)
        dQ_domega = -(self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1) ** 2)
        dQ_deo = np.stack([np.reshape(dQ_deta, -1), np.reshape(dQ_domega, -1)], axis=1)  # d**2 x 2

        t1 = dQ_deo @ deo_dQ_target + np.eye(self._dim ** 2) / (self._eta + self._omega + 1)
        x = np.reshape(t1, [self._dim, self._dim, self._dim, self._dim])
        x = self._proj_precision @ self._target_precision @ x @ self._proj_covar @ self._target_precision

        dq_dqtilde = np.outer(dq_deta, deta_dq_target) + np.outer(dq_domega, domega_dq_target) + np.eye(self._dim) / (
                    self._eta + self._omega + 1)
        dmu_dmutilde = self._proj_covar @ dq_dqtilde @ self._target_precision
        # dmu_dmutilde = self._target_precision @  dq_dqtilde @ self._proj_covar
        return dmu_dmutilde, x

    def backward(self, d_mean, d_cov):
        """
        Error signal backward propagation based on values from last forward pass. This corresponds to the actual
        backward pass we would need in tensorflow
        TODO: Implement backward propagation of err_cov
        """
        assert self._succ, "INVALID STATE, No previous successfull execution!"
        deta_dq_target, deta_dQ_target, domega_dq_target, domega_dQ_target = self.get_last_eo_grad()

        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (
                    (self._omega + self._eta + 1) ** 2)
        dq_domega = -(self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dQ_domega = -(self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1) ** 2)

        # mean

        d_q = self._proj_covar @ d_mean
        tmp = np.outer(d_mean, self._proj_lin)
        d_Q = - self._proj_covar @ (0.5 * tmp + 0.5 * tmp.T + d_cov) @ self._proj_covar

        d_eta = np.dot(d_q, dq_deta) + np.trace(d_Q @ dQ_deta)
        d_omega = np.dot(d_q, dq_domega) + np.trace(d_Q @ dQ_domega)

        d_q_target = d_eta * deta_dq_target + d_omega * domega_dq_target + d_q / (self._eta + self._omega + 1)
        d_Q_target = d_eta * deta_dQ_target + d_omega * domega_dQ_target + d_Q / (self._eta + self._omega + 1)

        d_mu_target = self._target_precision @ d_q_target
        tmp = np.outer(d_q_target, self._target_mean)
        d_cov_target = - self._target_precision @ (0.5 * tmp + 0.5 * tmp.T + d_Q_target) @ self._target_precision

        return d_mu_target, d_cov_target

    def get_last_eo_grad(self):
        """
        gradients of eta and omega w.rt. inputs, based on last forward pass
        For each case (except the first) there is a "baseline" implementation, which should correspond to the
        equations in the overleaf, and a tuned version, optimized to reduce number of performed operations
        """
        assert self._succ, "INVALID STATE, No previous successful execution!"
        if self._eta == 0 and self._omega == 0:  # case 1
            return np.zeros(self._dim), np.zeros([self._dim, self._dim]), \
                   np.zeros(self._dim), np.zeros([self._dim, self._dim])

        elif self._eta == 0 and self._omega > 0:  # case 2
            at, bt, ct, dt = self._case2_tuned()
            # ab, bb, cb, db = self._case2_baseline()
            # print(np.max(np.abs(cb - ct)), np.max(np.abs(cb - ct)))
            return at, bt, ct, dt

        elif self._eta > 0 and self._omega == 0:  # case 3
            at, bt, ct, dt = self._case3_baseline2()
            # ab, bb, cb, db = self._case3_baseline()
            # print(np.max(np.abs(ab - at)), np.max(np.abs(bb - bt)))
            return at, bt, ct, dt

        elif self._eta > 0 and self._omega > 0:
            at, bt, ct, dt = self._case4_tuned()
            # ab, bb, cb, db = self._case4_baseline()
            # print(np.max(np.abs(ab - at)), np.max(np.abs(bb - bt)), np.max(np.abs(cb - ct)), np.max(np.abs(db - dt)))
            return at, bt, ct, dt

        else:
            raise AssertionError("WTF?")

    def _case2_baseline(self):
        dQ_omega = - (self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1) ** 2)
        lhs = np.trace(self._proj_covar @ dQ_omega)
        rhs = - self._proj_covar / (self._omega + self._eta + 1)
        domega_dQ = rhs / lhs
        return np.zeros(self._dim), np.zeros([self._dim, self._dim]), np.zeros(self._dim), domega_dQ

    def _case2_tuned(self):
        domega_dQ = self._proj_covar / self._dim
        return np.zeros(self._dim), np.zeros([self._dim, self._dim]), np.zeros(self._dim), domega_dQ

    def _case3_baseline2(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (
                    (self._omega + self._eta + 1) ** 2)
        dq_domega = - (self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1)) ** 2
        dQ_domega = - (self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1)) ** 2

        dtm1_dq = self._proj_covar @ self._old_lin
        dtm2_dq = 2 * self._proj_covar @ self._old_precision @ self._proj_covar @ self._proj_lin
        f2_dq = - 2 * dtm1_dq + dtm2_dq

        dtlogdet_dQ = self._proj_covar
        dttrace_dQ = - self._proj_covar @ self._old_precision @ self._proj_covar
        tmp = np.outer(self._proj_lin, self._old_lin)
        dtm1_dQ = - self._proj_covar @ (0.5 * (tmp + tmp.T)) @ self._proj_covar
        tmp = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        dtm2_dQ = - self._proj_covar @ (tmp + tmp.T) @ self._proj_covar
        f2_dQ = dtlogdet_dQ + dttrace_dQ - 2 * dtm1_dQ + dtm2_dQ

        lhs = np.dot(f2_dq, dq_deta) + np.trace(f2_dQ @ dQ_deta)
        rhs_dq = -f2_dq / (self._omega + self._eta + 1)  # - dq_deta - dq_domega
        rhs_dQ = -f2_dQ / (self._omega + self._eta + 1)  # - dQ_deta - dQ_domega
        deta_dq = rhs_dq / lhs
        deta_dQ = rhs_dQ / lhs
        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])

    def _case3_baseline(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (
                    (self._omega + self._eta + 1) ** 2)

        dtm1_dq = self._proj_covar @ self._old_lin
        dtm2_dq = 2 * self._proj_covar @ self._old_precision @ self._proj_covar @ self._proj_lin
        f2_dq = - 2 * dtm1_dq + dtm2_dq

        dtlogdet_dQ = self._proj_covar
        dttrace_dQ = - self._proj_covar @ self._old_precision @ self._proj_covar
        tmp = np.outer(self._proj_lin, self._old_lin)
        dtm1_dQ = - self._proj_covar @ (0.5 * (tmp + tmp.T)) @ self._proj_covar
        tmp = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        dtm2_dQ = - self._proj_covar @ (tmp + tmp.T) @ self._proj_covar
        f2_dQ = dtlogdet_dQ + dttrace_dQ - 2 * dtm1_dQ + dtm2_dQ

        lhs = np.dot(f2_dq, dq_deta) + np.trace(f2_dQ @ dQ_deta)
        rhs_dq = -f2_dq / (self._omega + self._eta + 1)
        rhs_dQ = -f2_dQ / (self._omega + self._eta + 1)
        deta_dq = rhs_dq / lhs
        deta_dQ = rhs_dQ / lhs
        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])

    def _case3_tuned(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / (self._omega + self._eta + 1)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (self._omega + self._eta + 1)

        f2_dq = 2 * self._proj_covar @ (self._old_precision @ self._proj_mean - self._old_lin)

        tmp_m1 = np.outer(self._proj_lin, self._old_lin)
        tmp_m2 = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        tmp = (np.eye(self._dim) + (-self._old_precision + tmp_m1 + tmp_m1.T - tmp_m2 - tmp_m2.T) @ self._proj_covar)
        f2_dQ = self._proj_covar @ tmp

        c = - 1 / (np.sum(f2_dQ * dQ_deta) + np.dot(f2_dq, dq_deta))
        deta_dq = f2_dq * c
        deta_dQ = f2_dQ * c

        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])

    def _case4_baseline(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1)) ** 2
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (
        (self._omega + self._eta + 1)) ** 2
        dq_domega = - (self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1)) ** 2
        dQ_domega = - (self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1)) ** 2

        dtm1_dq = self._proj_covar @ self._old_lin
        dtm2_dq = 2 * self._proj_covar @ self._old_precision @ self._proj_covar @ self._proj_lin
        f2_dq = - 2 * dtm1_dq + dtm2_dq

        dtlogdet_dQ = self._proj_covar
        dttrace_dQ = - self._proj_covar @ self._old_precision @ self._proj_covar
        tmp = np.outer(self._proj_lin, self._old_lin)
        dtm1_dQ = - self._proj_covar @ (0.5 * (tmp + tmp.T)) @ self._proj_covar
        tmp = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        dtm2_dQ = - self._proj_covar @ (tmp + tmp.T) @ self._proj_covar
        f2_dQ = dtlogdet_dQ + dttrace_dQ - 2 * dtm1_dQ + dtm2_dQ

        rhs_q = - f2_dq / (self._omega + self._eta + 1)

        lhs_q_eta = np.dot(f2_dq, dq_deta) + np.trace(f2_dQ @ dQ_deta)
        fact_eta = - np.trace(dtlogdet_dQ @ dQ_deta) / np.trace(dtlogdet_dQ @ dQ_domega)
        lhs_q_eta = lhs_q_eta + fact_eta * (np.dot(f2_dq, dq_domega) + np.trace(f2_dQ @ dQ_domega))

        lhs_q_omega = np.dot(f2_dq, dq_domega) + np.trace(f2_dQ @ dQ_domega)
        fact_omega = - np.trace(dtlogdet_dQ @ dQ_domega) / np.trace(dtlogdet_dQ @ dQ_deta)
        lhs_q_omega = lhs_q_omega + fact_omega * (np.dot(f2_dq, dq_deta) + np.trace(f2_dQ @ dQ_deta))

        deta_dq = rhs_q / lhs_q_eta
        domega_dq = rhs_q / lhs_q_omega

        rhs_Q_eta2 = (1 / np.trace(dtlogdet_dQ @ dQ_domega)) * (
                    np.dot(f2_dq, dq_domega) + np.trace(f2_dQ @ dQ_domega)) * dtlogdet_dQ / (
                                 self._omega + self._eta + 1)
        rhs_Q_eta1 = - f2_dQ / (self._omega + self._eta + 1)
        rhs_Q_eta = rhs_Q_eta1 + rhs_Q_eta2

        deta_d_Q = rhs_Q_eta / lhs_q_eta

        rhs_Q_omega2 = (1 / np.trace(dtlogdet_dQ @ dQ_deta)) * (
                    np.dot(f2_dq, dq_deta) + np.trace(f2_dQ @ dQ_deta)) * dtlogdet_dQ / (self._omega + self._eta + 1)
        rhs_Q_omega1 = - f2_dQ / (self._omega + self._eta + 1)
        rhs_Q_omega = rhs_Q_omega1 + rhs_Q_omega2
        domega_d_Q = rhs_Q_omega / lhs_q_omega

        return deta_dq, deta_d_Q, domega_dq, domega_d_Q

    def _case4_tuned(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / (self._omega + self._eta + 1)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (self._omega + self._eta + 1)
        dq_domega = - self._proj_lin
        dQ_domega = - self._proj_precision

        # d eta d q rhs
        f2_dq = 2 * self._proj_covar @ (self._old_precision @ self._proj_mean - self._old_lin)

        # d eta d Q rhs
        tmp_m1 = np.outer(self._proj_lin, self._old_lin)
        tmp_m2 = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        tmp = (np.eye(self._dim) + (-self._old_precision + tmp_m1 + tmp_m1.T - tmp_m2 - tmp_m2.T) @ self._proj_covar)
        f2_dQ = self._proj_covar @ tmp

        f2_domega = np.dot(f2_dq, dq_domega) + np.sum(f2_dQ * dQ_domega)
        f2_deta = np.dot(f2_dq, dq_deta) + np.sum(f2_dQ * dQ_deta)
        rhs_q = - f2_dq

        tr_dlogdet_deta = np.sum(self._proj_covar * dQ_deta)
        tr_dlogdet_domega = np.sum(self._proj_covar * dQ_domega)

        lhs_q_eta = np.dot(f2_dq, dq_deta) + np.sum(f2_dQ * dQ_deta)
        fact_eta = - tr_dlogdet_deta / tr_dlogdet_domega
        lhs_q_eta = lhs_q_eta + fact_eta * f2_domega

        lhs_q_omega = f2_domega
        fact_omega = - tr_dlogdet_domega / tr_dlogdet_deta
        lhs_q_omega = lhs_q_omega + fact_omega * f2_deta

        deta_dq = rhs_q / lhs_q_eta
        domega_dq = rhs_q / lhs_q_omega

        deta_d_Q = (f2_domega * self._proj_covar / tr_dlogdet_domega - f2_dQ) / lhs_q_eta
        domega_d_Q = (f2_deta * self._proj_covar / tr_dlogdet_deta - f2_dQ) / lhs_q_omega

        return deta_dq, deta_d_Q, domega_dq, domega_d_Q
