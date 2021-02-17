#ifndef CPP_COVONLYMOREPROJECTION_H
#define CPP_COVONLYMOREPROJECTION_H

#define ARMA_DONT_PRINT_ERRORS

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>


using namespace arma;

class CovOnlyMoreProjection{

public:
    CovOnlyMoreProjection(uword dim, int max_eval);

    mat forward(double eps, const mat &old_cov, const mat &target_cov);

    mat backward(const mat &d_cov);

    double get_last_eta() const { return eta;};
    bool was_succ() const {return succ;}

    void set_omega_offset(double omega_offset){this->omega_offset = omega_offset;};
    double get_omega_offset() const { return omega_offset;};

private:

    mat last_eta_grad() const;

    double dual(std::vector<double> const &eta, std::vector<double> &grad);

    double eps, omega_offset;
    bool succ;
    uword dim;
    double eta=1;
    std::vector<double> grad = std::vector<double>(1, 10);
    int max_eval;
    double old_logdet, old_term, kl_const_part;

    mat old_prec, old_chol_prec, target_prec, projected_covar, projected_prec;


};
#endif //CPP_COVONLYMOREPROJECTION_H
