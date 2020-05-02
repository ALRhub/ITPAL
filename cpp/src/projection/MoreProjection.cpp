#include <projection/MoreProjection.h>

MoreProjection::MoreProjection(uword dim) :
    dim(dim)
{
    dual_const_part = dim * log(2 * M_PI);
    entropy_const_part = 0.5 * (dual_const_part + dim);
    omega_offset = 1.0;  // Default value, might change due to rescaling!
}

std::tuple<vec, mat> MoreProjection::more_step(double eps, double beta,
                                               const vec  &old_mean, const mat &old_covar,
                                               const vec &target_mean, const mat &target_covar){
    this->eps = eps;
    this->beta = beta;
    succ = false;
    // eta_inc_ct = 0;

    this->old_mean = old_mean;
    old_precision = inv_sympd(old_covar);
    old_lin = old_precision * old_mean;
    old_chol_precision_t = chol(old_precision, "lower").t();

    target_precision = inv_sympd(target_covar);
    target_lin = target_precision * target_mean;

    // - logdet(Q_old)
    double old_logdet = - 2 * sum(log(diagvec(old_chol_precision_t) + 1e-25));


    // - 0.5 q^T_old Q_old^-1 q_old + 0.5 * logdet(Q_old) -0.5 * dim * log(2pi)
    old_term = -0.5 * (dot(old_lin, old_mean) + old_logdet + dual_const_part);

    kl_const_part = old_logdet - dim;

    // opt
    nlopt::opt opt(nlopt::LD_LBFGS, 2);

    opt.set_min_objective([](const std::vector<double> &eta_omega, std::vector<double> &grad, void *instance){
        return ((MoreProjection *) instance)->dual(eta_omega, grad);}, this);

    std::vector<double> opt_eta_omega;

    std::tie(succ, opt_eta_omega, res_txt) = NlOptUtil::opt_dual(opt);

    if (!succ) {
        opt_eta_omega[0] = eta;
        opt_eta_omega[1] = omega;
        succ = NlOptUtil::valid_despite_failure(opt_eta_omega, grad);
    }

    std::tuple<vec, mat> res;
    if (succ) {
        eta = opt_eta_omega[0];
        omega = opt_eta_omega[1];

        vec new_lin = (eta * old_lin + target_lin) / (eta + omega + omega_offset);
        mat new_covar = inv_sympd((eta * old_precision + target_precision) / (eta + omega + omega_offset));
        //std::tie(new_lin, new_covar) = new_params_internal(eta, omega);
        res_txt = " ";
        res = std::make_tuple(new_covar * new_lin, new_covar);
    } else{
        res_txt += "Failure, last grad " + std::to_string(grad[0]) + " " + std::to_string(grad[1]) + " - skipping ";
        res = std::make_tuple(vec(dim, fill::zeros), mat(dim, dim, fill::zeros));
    }
    if (eta_inc_ct > 0){
        res_txt += "Increased eta " + std::to_string(eta_inc_ct) + " times.";
    }
    return res;
}

/*
std::tuple<vec, mat> MoreProjection::new_params_internal(double eta, double omega) {
    omega = constrain_entropy ? omega : 0.0;
    vec new_lin = ((eta + eta_offset) * old_lin + reward_lin) / (eta + eta_offset + omega + omega_offset);
    mat new_covar;
    while (eta < 1e12) {
        try {
            new_covar = inv_sympd((eta + eta_offset) * old_precision + reward_quad) * (eta + eta_offset + omega + omega_offset);
            break;
        } catch (std::runtime_error &err) {
            if (eta < 1e-12){
                eta = 1e-12;
            }
            eta *= 1.1;
            eta_inc_ct++;
        }
    }
    return std::make_tuple(new_lin, new_covar);

}
*/

double MoreProjection::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
    eta = eta_omega[0] > 0.0 ? eta_omega[0] : 0.0;
    omega = eta_omega[1] > 0.0 ? eta_omega[1] : 0.0;
    double omega_off = omega + omega_offset;

    vec new_lin = (eta * old_lin + target_lin) / (eta + omega_off);
    mat new_quad = (eta * old_precision + target_precision) / (eta + omega_off);
    try {
        mat new_covar = inv_sympd(new_quad);
        mat new_chol_covar = chol(new_covar, "lower");

        vec new_mean = new_covar * new_lin;
        double new_logdet = 2 * sum(log(diagvec(new_chol_covar) + 1e-25));

        double dual = eta * eps - omega * beta + eta * old_term;
        dual += 0.5 * (eta + omega_off) * (dual_const_part + new_logdet + dot(new_lin, new_mean));

        vec diff = old_mean - new_mean;
        double trace_term = accu(square(old_chol_precision_t * new_chol_covar));
        double kl = 0.5 * (sum(square(old_chol_precision_t * diff)) + kl_const_part - new_logdet + trace_term);

        grad[0] = eps - kl;
        grad[1] = entropy_const_part + 0.5 * new_logdet - beta;
        this->grad[0] = grad[0];
        this->grad[1] = grad[1];
        return dual;
    } catch (std::runtime_error &err) {
        grad[0] = -1.0;
        grad[1] = 0.0;
        this->grad[0] = grad[0];
        this->grad[1] = grad[1];
        return 1e12;
    }
}
