import tensorflow as tf
import numpy as np
import cvxpy as cp
from cvxpylayers.tensorflow import CvxpyLayer

class CVXProjector:

    def __init__(self, dim):
        self._dim = dim
        self._proj_layer = self.build_cvx_layer(self._dim)

    def project(self, eps, beta, old_mean, old_covar, target_mean, target_covar):
        """ runs projection layer and constructs cholesky of covariance"""

        solver_args = {
            "max_iters": 50000,
            'eps': 1e-7,
            # "raise_on_error": False
            # "verbose": True,
            # "gpu": True,
            # "use_indirect": True
        }

        tar_transposed_chol_quad = tf.transpose(tf.linalg.cholesky(tf.linalg.inv(target_covar)))
        mahalanobis_tar_part = tf.linalg.matvec(tar_transposed_chol_quad, target_mean)

        old_transposed_chol_quad = tf.transpose(tf.linalg.cholesky(tf.linalg.inv(old_covar)))
        mahalanobis_old_part = tf.linalg.matvec(old_transposed_chol_quad, old_mean)

        proj_res = self._proj_layer(mahalanobis_tar_part, tar_transposed_chol_quad,
                                    mahalanobis_old_part, old_transposed_chol_quad,
                                    eps, beta, solver_args=solver_args)
        projected_mean, pcc = proj_res
        projected_chol = tf.stack([tf.concat([pcc[sum(range(i + 1)):sum(range(i + 1)) + i + 1],
                                              tf.zeros((self._dim - i - 1,), dtype=pcc.dtype)],
                                             axis=0) for i in range(self._dim)], axis=0)
        projected_covar = tf.matmul(projected_chol, projected_chol, transpose_b=True)
        return projected_mean, projected_covar

    def get_grad(self,  eps, beta, old_mean, old_covar, target_mean, target_covar):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(target_mean)
            tape.watch(target_covar)
            new_mean, new_cov = self.project(eps, beta, old_mean, old_covar, target_mean, target_covar)
        dm_dm_target = tape.jacobian(new_mean, target_mean, experimental_use_pfor=False)
        dm_dcov_target = tape.jacobian(new_mean, target_covar, experimental_use_pfor=False)
        dcov_dm_target= tape.jacobian(new_cov, target_mean, experimental_use_pfor=False)
        dcov_dcov_target =tape.jacobian(new_cov, target_covar, experimental_use_pfor=False)
        return dm_dm_target, dm_dcov_target, dcov_dm_target, dcov_dcov_target

    def backward(self, eps, beta, old_mean, old_covar, target_mean, target_covar):
        with tf.GradientTape() as tape:
            tape.watch(target_mean)
            tape.watch(target_covar)
            new_mean, new_cov = self.project(eps, beta, old_mean, old_covar, target_mean, target_covar)
        return tape.gradient([new_mean, new_cov], [target_mean, target_covar])


    @staticmethod
    def build_cvx_layer(dim):
        """build up the cvpx layer"""
        cvpx_proj_mean = cp.Variable(dim)

        chol_size = dim * (dim + 1) // 2
        chol_variable = cp.Variable(chol_size)
        cvx_proj_chol = cp.vstack([cp.hstack([chol_variable[sum(range(i + 1)):sum(range(i + 1)) + i + 1], cp.Constant(
            np.zeros((dim - i - 1,)))]) if i != dim - 1 else cp.hstack(
            [chol_variable[sum(range(i + 1)):sum(range(i + 1)) + i + 1]]) for i in range(dim)])


        cvpx_tar_transposed_chol_quad = cp.Parameter((dim, dim))
        cvpx_mahalanobis_tar_part = cp.Parameter(dim)

        cvpx_old_transposed_chol_quad = cp.Parameter((dim, dim))
        cvpx_mahalanobis_old_part = cp.Parameter(dim)

        cvpx_entropy_bound = cp.Parameter()
        cvpx_kl_bound = cp.Parameter()

        """objective"""
        trace_term_t = cp.sum_squares(cp.matmul(cvpx_tar_transposed_chol_quad, cvx_proj_chol))
        covar_logdet_t = 2 * cp.sum(cp.log(cp.diag(cvx_proj_chol)))
        tar_covar_logdet_t = - 2 * cp.sum(cp.log(cp.diag(cvpx_tar_transposed_chol_quad)))
        mahalanobis_part_t = cp.sum_squares(cvpx_mahalanobis_tar_part -
                                         cp.matmul(cvpx_tar_transposed_chol_quad, cvpx_proj_mean))
        kl_to_target = 0.5 * (mahalanobis_part_t + tar_covar_logdet_t - covar_logdet_t - dim + trace_term_t)
        obj = cp.Minimize(kl_to_target)

        """kl"""
        trace_term_o = cp.sum_squares(cp.matmul(cvpx_old_transposed_chol_quad, cvx_proj_chol))
        covar_logdet_o = 2 * cp.sum(cp.log(cp.diag(cvx_proj_chol)))
        old_covar_logdet_o = - 2 * cp.sum(cp.log(cp.diag(cvpx_old_transposed_chol_quad)))
        mahalanobis_part_o = cp.sum_squares(cvpx_mahalanobis_old_part -
                                          cp.matmul(cvpx_old_transposed_chol_quad, cvpx_proj_mean))
        kl_to_old = 0.5 * (mahalanobis_part_o + old_covar_logdet_o - covar_logdet_o - dim + trace_term_o)

        """entropy"""
        entropy_const_part = dim * np.log(2 * np.pi * np.e)
        proj_entropy = 0.5 * (entropy_const_part + 2 * cp.sum(cp.log(cp.diag(cvx_proj_chol))))

        """problem"""
        const = [kl_to_old <= cvpx_kl_bound, proj_entropy >= cvpx_entropy_bound]
        cvx_problem = cp.Problem(obj, const)

        params = [cvpx_mahalanobis_tar_part, cvpx_tar_transposed_chol_quad,
                  cvpx_mahalanobis_old_part, cvpx_old_transposed_chol_quad,
                  cvpx_kl_bound, cvpx_entropy_bound]

        projection_layer = CvxpyLayer(cvx_problem, parameters=params,
                                      variables=[cvpx_proj_mean, chol_variable])
        return projection_layer