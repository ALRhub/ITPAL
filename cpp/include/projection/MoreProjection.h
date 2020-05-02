#ifndef CPP_MOREPROJECTION_H
#define CPP_MOREPROJECTION_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class MoreProjection{

public:
    MoreProjection(uword dim);

    std::tuple<vec, mat> more_step(double eps, double beta,
            const vec  &old_mean, const mat &old_covar, const vec &target_mean, const mat &target_covar);

    double get_last_eta() const { return eta;};
    double get_last_omega() const { return omega;};
    bool was_succ() const {return succ;}
    std::string get_res_txt() const { return res_txt;};

    void set_omega_offset(double omega_offset){this->omega_offset = omega_offset;};
    double get_omega_offset() const { return omega_offset;};

private:

    double dual(std::vector<double> const &eta_omega, std::vector<double> &grad);
    //std::tuple<vec, mat> new_params_internal(double eta, double omega);

    double eps, beta, omega_offset;
    bool succ;
    uword dim, eta_inc_ct;
    double eta=1, omega=1;
    std::vector<double> grad = std::vector<double>(2, 10);

    double dual_const_part, old_term, entropy_const_part, kl_const_part;

    std::string res_txt;

    vec old_lin, old_mean, target_lin;
    mat old_precision, old_chol_precision_t, target_precision;

};
#endif //CPP_MOREPROJECTION_H
