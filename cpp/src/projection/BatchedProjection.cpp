#include <projection/BatchedProjection.h>
#include <chrono>
using namespace std::chrono;


BatchedProjection::BatchedProjection(uword batch_size, uword dim) : batch_size(batch_size), dim(dim) {
    for (int i = 0; i < batch_size; ++i) {
        projectors.emplace_back(MoreProjection(dim));
        projection_applied.emplace_back(false);
    }
    entropy_const_part = 0.5 * (dim * log(2 * M_PI * M_E)); 

    openblas_set_num_threads(1);
}

std::tuple<mat, cube> BatchedProjection::more_step(const vec &epss, const vec &betas,
                                                   const mat &old_means, const cube &old_covars,
                                                   const mat &target_means, const cube &target_covars) {

    mat means(size(old_means));
    cube covs(size(old_covars));
    auto start = high_resolution_clock::now();

#pragma omp parallel for default(none) schedule(static) shared(epss, betas, old_means, old_covars, target_means, target_covars, means, covs)
    for (int i = 0; i < batch_size; ++i) {
        double eps = epss.at(i);
        double beta = betas.at(i);
        const vec &old_mean = old_means.col(i);
        const mat &old_cov = old_covars.slice(i);
        const vec &target_mean = target_means.col(i);
        const mat &target_cov = target_covars.slice(i);

        mat occ = chol(old_cov, "lower");
        mat tcc = chol(target_cov, "lower");
        double kl_ = kl(target_mean, tcc, old_mean, occ);
        double entropy_ = entropy(tcc);

        if (kl_ <= eps && entropy_ >= beta){
            means.col(i) = target_mean;
            covs.slice(i) = target_cov;
            projection_applied.at(i) = false;
        } else {
            vec mean;
            mat cov;
            std::tie(mean, cov) = projectors[i].more_step(eps, beta, old_mean, old_cov, target_mean, target_cov);
            //std::cout << mean << cov;
            means.col(i) = mean;
            covs.slice(i) = cov;
            projection_applied.at(i) = true;
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "time " << duration.count() << endl;
    return std::make_tuple(means, covs);
}

double BatchedProjection::kl(const vec& m1, const mat& cc1, const vec& m2, const mat& cc2) const {
    mat cc2_inv_t = inv(cc2);
    double logdet_term = 2 * (sum(log(diagvec(cc2) + 1e-25)) - sum(log(diagvec(cc1) + 1e-25)));
    double trace_term = accu(square(cc2_inv_t * cc1));
    double mahal_term = sum(square(cc2_inv_t * (m2 - m1)));
    double kl = 0.5 * (logdet_term + trace_term + mahal_term - dim);
    return kl;
}

double BatchedProjection::entropy(const mat& cc) const {
    return entropy_const_part + sum(log(diagvec(cc) + 1e-25));
}
