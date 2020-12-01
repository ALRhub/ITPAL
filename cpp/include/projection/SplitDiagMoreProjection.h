#ifndef CPP_SPLITDIAGMOREPROJECTION_H
#define CPP_SPLITDIAGMOREPROJECTION_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class SplitDiagMoreProjection{

public:
    SplitDiagMoreProjection(uword dim, int max_eval);

    std::tuple<vec, vec> forward(double eps_mu, double eps_sigma,
                                 const vec  &old_mean, const vec &old_var,
                                 const vec &target_mean, const vec &target_var);

    //std::tuple<vec, mat> backward(const vec &dl_dmu_projected, const vec &d_cov);

    double get_last_lp() const { return lp;};
    bool was_succ() const {return succ_mu && succ_sig;};

private:

    //std::tuple<vec, mat, vec, mat> last_eo_grad() const;

    double dual_mean(std::vector<double> const &eta_mu, std::vector<double> &grad);
    double dual_cov(std::vector<double> const &eta_sig, std::vector<double> &grad);

    //std::tuple<vec, mat> new_params_internal(double eta, double omega);

    double eps_mu, eps_sig;
    bool succ_mu, succ_sig;
    uword dim, eta_inc_ct;
    double lp=1.0;
    std::vector<double> grad = std::vector<double>(1.0, 10);

    int max_eval;
    double old_dot, old_logdet, kl_const_part;

    vec old_lin, old_mean, old_var, old_prec, old_chol_prec;
    vec target_lin, target_mean, target_prec;

};
#endif //CPP_SPLITDIAGMOREPROJECTION_H
