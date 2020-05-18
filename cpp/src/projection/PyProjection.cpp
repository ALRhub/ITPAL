#include <pybind11/pybind11.h>
#include <PyArmaConverter.h>

#include <projection/MoreProjection.h>
#include <projection/BatchedProjection.h>


namespace py = pybind11;

PYBIND11_MODULE(cpp_projection, p){

    py::class_<MoreProjection> mp(p, "MoreProjection");

    mp.def(py::init([](uword dim){return new MoreProjection(dim);}), py::arg("dim"));

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
    mp.def_property_readonly("res_text", &MoreProjection::get_res_txt);

    py::class_<BatchedProjection> bp(p, "BatchedProjection");
    bp.def(py::init([](uword batch_size, uword dim){
        return new BatchedProjection(batch_size, dim);}),
        py::arg("batchsize"), py::arg("dim"));

    bp.def("forward", [](BatchedProjection* obj, dpy_arr epss, dpy_arr betas,
                           dpy_arr old_means, dpy_arr old_covars, dpy_arr target_means, dpy_arr target_covars){
        mat means;
        cube covs;
        std::tie(means, covs) = obj->forward(
                to_vec<double>(epss), to_vec<double>(betas),
                to_mat<double>(old_means), to_cube<double>(old_covars),
                to_mat<double>(target_means), to_cube<double>(target_covars));
        return std::make_tuple(from_mat(means), from_cube(covs));
        },
           py::arg("eps"), py::arg("beta"), py::arg("old_mean"),
           py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar")
    );

    bp.def("backward", [](BatchedProjection* obj, dpy_arr d_means, dpy_arr d_covs){
        mat d_means_target;
        cube d_covs_target;
        std::tie(d_means_target, d_covs_target) = obj->backward(to_mat<double>(d_means), to_cube<double>(d_covs));
        return std::make_tuple(from_mat(d_means_target), from_cube(d_covs_target));
        },
           py::arg("d_means"), py::arg("d_covs")
    );
}