import numpy as np
from projection.ITPS import ITPS


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

        self._target_precision = np.linalg.inv(target_covar)
        self._target_lin = self._target_precision @ target_mean

        old_logdet = - np.linalg.slogdet(self._old_precision)[1]
        self._old_term = -0.5 * (np.dot(self._old_lin, self._old_mean) + self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        try:
            opt_eta, opt_omega = self.opt_dual()
            self._eta = opt_eta
            self._omega = opt_omega
            self._proj_lin, self._new_precision = \
                self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            self._proj_covar = np.linalg.inv(self._new_precision)
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
        deta_dq_target, _, domega_dq_target, _ = self.get_last_eo_grad()
        deo_dq_target = np.stack([deta_dq_target, domega_dq_target], axis=0)   #  2 x d

        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1)**2)
        dq_domega = -(self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1)**2)
        dq_deo = np.stack([dq_deta, dq_domega], axis=1)   # d x 2

        dq_dqtilde = (dq_deo @ deo_dq_target + np.eye(self._dim) / (self._eta + self._omega +1))
        dmu_dmutilde = self._proj_covar @ dq_dqtilde @ self._target_precision
        #dmu_dmutilde = self._target_precision @  dq_dqtilde @ self._proj_covar
        return dmu_dmutilde, np.zeros([self._dim, self._dim, self._dim, self._dim])

    def backward(self, err_mean, err_cov):
        deta_dq_target, _, domega_dq_target, _ = self.get_last_eo_grad()
        deo_dq_target = np.stack([deta_dq_target, domega_dq_target], axis=0)   #  2 x d

        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1)**2)
        dq_domega = -(self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1)**2)
        dq_deo = np.stack([dq_deta, dq_domega], axis=1)

        # mean
        t = err_mean
        t = self._proj_covar @ t
        t = dq_deo @ (deo_dq_target @ t) + t / (self._eta + self._omega + 1)
        t = self._target_precision @ t

        return t, np.zeros([self._dim, self._dim])


    def get_last_eo_grad(self):
        assert self._succ, "INVALID STATE, No previous successfull execution!"
        if self._eta == 0 and self._omega == 0:    # case 1
            return np.zeros(self._dim), np.zeros([self._dim, self._dim]), \
                   np.zeros(self._dim), np.zeros([self._dim, self._dim])

        elif self._eta == 0 and self._omega > 0:     # case 2
            at, bt, ct, dt = self._case2_tuned()
            ab, bb, cb, db = self._case2_baseline()
            print(np.max(np.abs(cb - ct)), np.max(np.abs(cb - ct)))
            return at, bt, ct, dt


        elif self._eta > 0 and self._omega == 0:     # case 3
            at, bt, ct, dt = self._case3_tuned()
            ab, bb, cb, db = self._case3_baseline()
            print(np.max(np.abs(ab - at)), np.max(np.abs(bb - bt)))
            return ab, bb, cb, db

        elif self._eta > 0 and self._omega > 0:
            return self._case4_baseline()
            #raise NotImplementedError

        else:
            raise AssertionError("WTF?")

    def _case2_baseline(self):
        dQ_omega = - (self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1)**2)
        lhs = np.trace(self._proj_covar @ dQ_omega)
        rhs = - self._proj_covar / (self._omega + self._eta + 1)
        domega_dQ = rhs / lhs
        return np.zeros(self._dim), np.zeros([self._dim, self._dim]), np.zeros(self._dim), domega_dQ

    def _case2_tuned(self):
        domega_dQ = self._proj_covar / self._dim
        return np.zeros(self._dim), np.zeros([self._dim, self._dim]), np.zeros(self._dim), domega_dQ

    def _case3_baseline(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / ((self._omega + self._eta + 1) ** 2)

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

        # d eta d q rhs
        f2_dq = 2 * self._proj_covar @ (self._old_precision @ self._proj_mean - self._old_lin)

        # d eta d Q rhs
        tmp_m1 = np.outer(self._proj_lin, self._old_lin)
        tmp_m2 = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        tmp = (np.eye(self._dim) + (-self._old_precision + tmp_m1 + tmp_m1.T - tmp_m2 - tmp_m2.T) @ self._proj_covar)
        f2_dQ = self._proj_covar @ tmp

        # lhs
        c = - 1 / (np.sum(f2_dQ * dQ_deta) + np.dot(f2_dq, dq_deta))
        deta_dq = f2_dq * c
        deta_dQ = f2_dQ * c
        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])

    def _case4_baseline(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1))**2
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / ((self._omega + self._eta + 1))**2
        dq_domega = - (self._eta * self._old_lin + self._target_lin) / ((self._omega + self._eta + 1))**2
        dQ_domega = - (self._eta * self._old_precision + self._target_precision) / ((self._omega + self._eta + 1))**2

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

        rhs_Q_eta2 = (1 / np.trace(dtlogdet_dQ @ dQ_domega)) * (np.dot(f2_dq, dq_domega) + np.trace(f2_dQ @ dQ_domega)) * dtlogdet_dQ / (self._omega + self._eta + 1)
        rhs_Q_eta1 = - f2_dQ / (self._omega + self._eta + 1)
        rhs_Q_eta = rhs_Q_eta1 + rhs_Q_eta2

        deta_d_Q = rhs_Q_eta / lhs_q_eta

        rhs_Q_omega2 = (1 / np.trace(dtlogdet_dQ @ dQ_deta)) * (np.dot(f2_dq, dq_deta) + np.trace(f2_dQ @ dQ_deta)) * dtlogdet_dQ / (self._omega + self._eta + 1)
        rhs_Q_omega1 = - f2_dQ / (self._omega + self._eta + 1)
        rhs_Q_omega = rhs_Q_omega1 + rhs_Q_omega2
        domega_d_Q = rhs_Q_omega / lhs_q_omega
       # pt_pq = 2 * self._proj_covar @ self._old_precision @ self._proj_covar @ self._proj_lin - 2 * self._proj_covar @ self._old_lin
        #pt_pq = 2 * self._proj_covar @ (self._old_precision @ self._proj_covar @ self._proj_lin - self._old_lin)

        #m = - np.linalg.solve(dQ_domega, dQ_deta)
        #c = - 1 / np.dot(pt_pq, dq_deta)

        #deta_dq = pt_pq * c

        #domega_dq = m @ deta_dq
        return deta_dq, deta_d_Q, domega_dq, domega_d_Q





    """
    def _case4_tuned2(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / (self._omega + self._eta + 1)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (self._omega + self._eta + 1)

        # d eta d q rhs
#        pt_pq = 2 * self._proj_covar @ (self._old_precision @ self._proj_covar @ self._proj_lin - self._old_lin)

        # d eta d Q rhs
        tmp_m1 = np.outer(self._proj_lin, self._old_lin)
        tmp_m2 = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        tmp = (np.eye(self._dim) + (-self._old_precision + tmp_m1 + tmp_m1.T - tmp_m2 - tmp_m2.T) @ self._proj_covar)
        pt_pQ = self._proj_covar @ tmp


        # lhs
        c = - 1 / (np.trace(pt_pQ @ dQ_deta) + np.dot(pt_pq, dq_deta))
        deta_dq = pt_pq * c
        deta_dQ = pt_pQ * c
        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])


    def case3_tuned(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / (self._omega + self._eta + 1)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (self._omega + self._eta + 1)

        pc_op = self._proj_covar @ self._old_precision

        dtm1_dq = self._proj_covar @ self._old_lin
        dtm2_dq = 2 * pc_op @ self._proj_mean
        deta_dq = - 2 * dtm1_dq + dtm2_dq

        dttrace_dQ = - pc_op @ self._proj_covar
        o_pl_ol = np.outer(self._proj_lin, self._old_lin)
        dtm1_dQ = self._proj_covar @ (o_pl_ol + o_pl_ol.T) @ self._proj_covar
        tmp = self._old_precision @ np.outer(self._proj_mean, self._proj_lin)
        dtm2_dQ = - self._proj_covar @ (tmp + tmp.T) @ self._proj_covar
        deta_dQ = self._proj_covar + dttrace_dQ + dtm1_dQ + dtm2_dQ

        pc_dq_de = self._proj_covar @ dQ_deta
        dlogdet_deta = np.trace(pc_dq_de)
        dtrace_deta = np.trace(- pc_op @ pc_dq_de)
        dtm1_deta = np.dot(self._proj_covar @ self._old_lin, dq_deta) + \
                    np.trace(- self._proj_covar @ o_pl_ol @ pc_dq_de)
        tmp = np.outer(self._proj_lin, self._proj_lin) @ pc_op
        dtm2_deta = np.dot(2 * pc_op @ self._proj_mean, dq_deta) + \
                    np.trace(- self._proj_covar @ (tmp + tmp.T) @ pc_dq_de)

        c_deta_dq = - 1 / (dlogdet_deta + dtrace_deta - 2 * dtm1_deta + dtm2_deta)

        deta_dq = deta_dq * c_deta_dq
        deta_dQ = deta_dQ * c_deta_dq
        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])

    def _case3_baseline2(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / ((self._omega + self._eta + 1) ** 2)
        dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (
                    (self._omega + self._eta + 1) ** 2)

        dtm1_dq = self._proj_covar @ self._old_lin / (self._omega + self._eta + 1)
        dtm2_dq = 2 * self._proj_covar @ self._old_precision @ self._proj_covar @ self._proj_lin / (
                    self._omega + self._eta + 1)
        deta_dq = - 2 * dtm1_dq + dtm2_dq

        dtlogdet_dQ = self._proj_covar / (self._omega + self._eta + 1)
        dttrace_dQ = - self._proj_covar @ self._old_precision @ self._proj_covar / (self._omega + self._eta + 1)
        tmp = np.outer(self._proj_lin, self._old_lin)
        # todo : check!!
        dtm1_dQ = - self._proj_covar @ (0.5 * (tmp + tmp.T)) @ self._proj_covar / (self._omega + self._eta + 1)
        tmp = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        dtm2_dQ = - self._proj_covar @ (tmp + tmp.T) @ self._proj_covar / (self._omega + self._eta + 1)
        deta_dQ = dtlogdet_dQ + dttrace_dQ - 2 * dtm1_dQ + dtm2_dQ

        dlogdet_deta = np.trace(self._proj_covar @ dQ_deta)
        dtrace_deta = np.trace(- self._proj_covar @ self._old_precision @ self._proj_covar @ dQ_deta)
        dtm1_deta = np.dot(self._proj_covar @ self._old_lin, dq_deta) + \
                    np.trace(- self._proj_covar @ np.outer(self._proj_lin, self._old_lin) @ self._proj_covar @ dQ_deta)
        tmp = self._old_precision @ self._proj_covar @ np.outer(self._proj_lin, self._proj_lin)
        dtm2_deta = np.dot(2 * self._proj_covar @ self._old_precision @ self._proj_covar @ self._proj_lin, dq_deta) + \
                    np.trace(- self._proj_covar @ (tmp + tmp.T) @ self._proj_covar @ dQ_deta)

        c_deta_dq = - 1 / (dlogdet_deta + dtrace_deta - 2 * dtm1_deta + dtm2_deta)

        deta_dq = deta_dq * c_deta_dq
        deta_dQ = deta_dQ * c_deta_dq
        return deta_dq, deta_dQ, np.zeros(self._dim), np.zeros([self._dim, self._dim])





    """
    """
    def _case4_baseline(self):
        dq_deta = ((self._omega + 1) * self._old_lin - self._target_lin) / (self._omega + self._eta + 1)
        dQ_eta = ((self._omega + 1) * self._old_precision - self._target_precision) / (self._omega + self._eta + 1)
        dq_domega = - (self._eta * self._old_lin + self._target_lin)  / (self._omega + self._eta + 1)
        dQ_omega = - (self._eta * self._old_precision + self._target_precision)  / (self._omega + self._eta + 1)

        m = - np.linalg.inv(dQ_omega, dQ_eta)

        f1 = self._target_precision@self._proj_covar @ self._proj_lin
        f2 = - self._old_lin

        lhs_q_eta =
        rhs_q_eta = f1 + f2

#        m = (self._eta * self._old_precision + self._target_precision)
#        m = np.linalg.solve(m,  (self._omega + 1) * self._old_precision - self._target_precision)

 #       deta_dq = 1 / (m @ (- dq_domega) - dq_deta)

        return deta_dq, np.zeros([self._dim, self._dim]), np.zeros(self._dim), np.zeros([self._dim, self._dim])


        #dQ_deta = ((self._omega + 1) * self._old_precision - self._target_precision) / (self._omega + self._eta + 1)
    """


