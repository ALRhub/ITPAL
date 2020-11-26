#include <pybind11/pybind11.h>
#include <PyArmaConverter.h>

#include <projection/MoreProjection.h>
#include <projection/BatchedProjection.h>
#include <projection/DiagCovOnlyMoreProjection.h>
#include <projection/BatchedDiagCovOnlyProjection.h>


namespace py = pybind11;

PYBIND11_MODULE(cpp_projection, p){
    /* ------------------------------------------------------------------------------
     MORE PROJECTION
     --------------------------------------------------------------------------------*/
    py::class_<MoreProjection> mp(p, "MoreProjection");

    mp.def(py::init([](uword dim, bool eec, bool constrain_entropy){return new MoreProjection(dim, eec, constrain_entropy);}),
            py::arg("dim"), py::arg("eec"), py::arg("constrain_entropy"));

    mp.def("forward", [](MoreProjection* obj, double eps, double beta,
                                  dpy_arr old_mean, dpy_arr old_covar,
                                  dpy_arr target_mean, dpy_arr target_covar){
               return from_mat<double>(obj->forward(eps, beta, to_vec<double>(old_mean), to_mat<double>(old_covar),
                                                    to_vec<double>(target_mean), to_mat<double>(target_covar)));},
    py::arg("eps"), py::arg("beta"), py::arg("old_mean"),
    py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar"));

    mp.def("backward", [](MoreProjection* obj, dpy_arr dl_dmu_projected, dpy_arr dl_dcovar_projected){
        vec dl_dmu_target;
        mat dl_dcovar_target;
        std::tie(dl_dmu_target, dl_dcovar_target) = obj->backward(to_vec<double>(dl_dmu_projected),
                                                                        to_mat<double>(dl_dcovar_projected));
        return std::make_tuple(from_mat<double>(dl_dmu_target), from_mat<double>(dl_dcovar_target));},
    py::arg("dl_dmu_projected"), py::arg("dl_dcovar_projected"));

    mp.def_property_readonly("last_eta", &MoreProjection::get_last_eta);
    mp.def_property_readonly("last_omega", &MoreProjection::get_last_omega);
    mp.def_property_readonly("was_succ", &MoreProjection::was_succ);

    /* ------------------------------------------------------------------------------
    DIAG COVAR ONLY PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<DiagCovOnlyMoreProjection> dcop(p, "DiagCovOnlyMoreProjection");

    dcop.def(py::init([](uword dim){return new DiagCovOnlyMoreProjection(dim);}),
           py::arg("dim"));

    dcop.def("forward", [](DiagCovOnlyMoreProjection* obj, double eps, dpy_arr old_covar, dpy_arr target_covar){
               return from_mat<double>(obj->forward(eps, to_vec<double>(old_covar), to_vec<double>(target_covar)));},
           py::arg("eps"),py::arg("old_covar"), py::arg("target_covar"));

    dcop.def("backward", [](DiagCovOnlyMoreProjection* obj, dpy_arr dl_dcovar_projected){
               return from_mat<double>(obj->backward(to_vec<double>(dl_dcovar_projected)));},
      py::arg("dl_dcovar_projected"));

    dcop.def_property_readonly("last_eta", &DiagCovOnlyMoreProjection::get_last_eta);
    dcop.def_property_readonly("was_succ", &DiagCovOnlyMoreProjection::was_succ);

    /* ------------------------------------------------------------------------------
    BATCHED PROJECTION
    --------------------------------------------------------------------------------*/
    py::class_<BatchedProjection> bp(p, "BatchedProjection");
    bp.def(py::init([](uword batch_size, uword dim, bool eec, bool constrain_entropy){
        return new BatchedProjection(batch_size, dim, eec, constrain_entropy);}),
        py::arg("batchsize"), py::arg("dim"), py::arg("eec"), py::arg("constrain_entropy"));

    bp.def("forward", [](BatchedProjection* obj, dpy_arr epss, dpy_arr betas,
                           dpy_arr old_means, dpy_arr old_covars, dpy_arr target_means, dpy_arr target_covars){
        mat means;
        cube covs;
        try {
            std::tie(means, covs) = obj->forward(
                    to_vec<double>(epss), to_vec<double>(betas),
                    to_mat<double>(old_means), to_cube<double>(old_covars),
                    to_mat<double>(target_means), to_cube<double>(target_covars));
        } catch (std::invalid_argument &e) {
            PyErr_SetString(PyExc_AssertionError, e.what());
        }
        return std::make_tuple(from_mat(means), from_cube(covs));
        },
           py::arg("epss"), py::arg("beta"), py::arg("old_mean"),
           py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar")
    );

    bp.def("backward", [](BatchedProjection* obj, dpy_arr d_means, dpy_arr d_covs){
        mat d_means_target;
        cube d_covs_target;
        std::tie(d_means_target, d_covs_target) = obj->backward(to_mat<double>(d_means), to_cube<double>(d_covs));
        return std::make_tuple(from_mat(d_means_target), from_cube(d_covs_target));
        },
           py::arg("d_means"), py::arg("d_covs"));


    /* ------------------------------------------------------------------------------
    BATCHED DIAG COVAR ONLY PROJECTION

    --------------------------------------------------------------------------------*/
    py::class_<BatchedDiagCovOnlyProjection> bdcop(p, "BatchedDiagCovOnlyProjection");
    bdcop.def(py::init([](uword batch_size, uword dim){return new BatchedDiagCovOnlyProjection(batch_size, dim);}),
           py::arg("batchsize"), py::arg("dim"));

    bdcop.def("forward", [](BatchedDiagCovOnlyProjection* obj, dpy_arr epss, dpy_arr old_vars,
            dpy_arr target_vars){
           try {
                   mat vars = obj->forward(to_vec<double>(epss), to_mat<double>(old_vars), to_mat<double>(target_vars));
                   return from_mat<double>(vars);
               } catch (std::invalid_argument &e) {
                   PyErr_SetString(PyExc_AssertionError, e.what());
               }
           },
           py::arg("epss"),py::arg("old_var"), py::arg("target_var")
    );

    bdcop.def("backward", [](BatchedDiagCovOnlyProjection* obj, dpy_arr d_vars){
               mat d_vars_d_target = obj->backward(to_mat<double>(d_vars));
               return from_mat<double>(d_vars_d_target);}, py::arg("d_vars"));
}