#include <projection/CovOnlyMoreProjection.h>

CovOnlyMoreProjection::CovOnlyMoreProjection(uword dim, int max_eval) :
    dim(dim),
    max_eval(max_eval)
{
    //dual_const_part = dim * log(2 * M_PI);
    omega_offset = 1.0;  // Default value, might change due to rescaling!
}

mat CovOnlyMoreProjection::forward(double eps, const mat &old_cov, const mat &target_cov){
    this->eps = eps;
    succ = false;

    /** Prepare **/
    old_prec = inv_sympd(old_cov);
    target_prec = inv_sympd(target_cov);

    old_chol_prec = chol(old_prec, "lower").t();
    double old_logdet = - 2 * sum(log(diagvec(old_chol_prec) + 1e-25));

    //old_term = -0.5 * (old_logdet + dual_const_part);
    kl_const_part = old_logdet - dim;

    /** Optimize **/
    nlopt::opt opt(nlopt::LD_LBFGS, 1);

    opt.set_min_objective([](const std::vector<double> &eta, std::vector<double> &grad, void *instance){
        return ((CovOnlyMoreProjection *) instance)->dual(eta, grad);}, this);

    std::vector<double> opt_eta_omega;

    std::tie(succ, opt_eta_omega) = NlOptUtil::opt_dual_eta(opt, 0.0, max_eval);
    if (!succ) {
        opt_eta_omega[0] = eta;
        succ = NlOptUtil::valid_despite_failure_eta(opt_eta_omega, grad);
    }

    /** Post process**/
    if (succ) {
        eta = opt_eta_omega[0];

        projected_prec = (eta * old_prec + target_prec) / (eta + omega_offset);
        projected_covar = 1.0 / projected_prec;
    } else{
        throw std::logic_error("NLOPT failure");
        //res = std::make_tuple(vec(dim, fill::zeros), mat(dim, dim, fill::zeros));
    }
    return projected_covar;
}

mat CovOnlyMoreProjection::backward(const mat &d_cov) {
    /** takes derivatives of loss w.r.t to projected mean and covariance and propagates back through optimization
      yielding derivatives of loss w.r.t to target mean and covariance **/
    if (!succ){
        throw std::exception();
    }
    /** Prepare **/

    mat deta_dQ_target;
    deta_dQ_target = last_eta_grad();

    double eo = omega_offset + eta;
    double eo_squared = eo * eo;
    mat dQ_deta = (omega_offset * old_prec - target_prec) / eo_squared;

    mat d_Q = - projected_covar * d_cov * projected_covar;

    double d_eta = trace(d_Q * dQ_deta);

    mat d_Q_target = d_eta * deta_dQ_target + d_Q / eo;

    mat d_cov_target = - target_prec * d_Q_target * target_prec;

    return d_cov_target;
}

double CovOnlyMoreProjection::dual(std::vector<double> const &eta_omega, std::vector<double> &grad){
    eta = eta_omega[0] > 0.0 ? eta_omega[0] : 0.0;
    mat new_prec = (eta * old_prec + target_prec) / (eta + omega_offset);
    try {
        /** dual **/
        mat new_covar = inv_sympd(new_prec);
        mat new_chol_covar = chol(new_covar, "lower");
        double new_logdet = 2 * sum(log(diagvec(new_chol_covar) + 1e-25));

        double dual = eta * eps - 0.5 * eta * old_logdet;
        dual += 0.5 * (eta + omega_offset) * new_logdet;

        /** gradient **/
        double trace_term = accu(square(old_chol_prec * new_chol_covar));

        double kl = 0.5 * (kl_const_part - new_logdet + trace_term);
        grad[0] = eps - kl;
        this->grad[0] = grad[0];

        return dual;
    } catch (std::runtime_error &err) {
        grad[0] = -1.0;
        this->grad[0] = grad[0];
        return 1e12;
    }
}

mat CovOnlyMoreProjection::last_eta_grad() const {

    /** case 1, constraint inactive **/
    if(eta == 0.0) {
        return mat(dim, dim, fill::zeros);

    /** case 2, constraint active **/
    }  else if(eta > 0.0){
        mat dQ_deta = (omega_offset * old_prec - target_prec) / (eta + omega_offset);


        mat tmp = mat(dim, dim, fill::eye) - old_prec * projected_covar;
        mat f2_dQ = projected_covar * tmp;

        double c = - 1  / trace(f2_dQ * dQ_deta);
        return c * f2_dQ;

    } else {
        throw std::exception();
    }
}
