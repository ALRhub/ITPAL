#ifndef IPP_BATCHEDDIAGCOVONLYPROJECTION_H
#define IPP_BATCHEDDIAGCOVONLYPROJECTION_H

#include <armadillo>
#include <nlopt.hpp>
#include <NlOptUtil.h>
#include <projection/CovOnlyMoreProjection.h>
#include <cblas.h>

using namespace arma;

class BatchedCovOnlyProjection{

public:

    BatchedCovOnlyProjection(uword batch_size, uword dim, int max_eval);

    cube forward(const vec &epss, const cube &old_covars, const cube &target_covars);
    cube backward(const cube &d_covs);

private:

    double kl_cov_only(const vec& cc1, const vec& cc2) const;
    std::vector<CovOnlyMoreProjection> projectors;
    uword batch_size, dim;
    std::vector<bool> projection_applied;

};

#endif //IPP_BATCHEDDIAGCOVONLYPROJECTION_H
