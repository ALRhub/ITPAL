#include <pybind11/pybind11.h>
#include <PyArmaConverter.h>

#include <projection/MoreProjection.h>
#include <projection/BatchedProjection.h>


namespace py = pybind11;

PYBIND11_MODULE(cpp_projection, p){

    py::class_<MoreProjection> mp(p, "MoreProjection");

    mp.def(py::init([](uword dim){return new MoreProjection(dim);}), py::arg("dim"));

    mp.def("more_step", [](MoreProjection* obj, double eps, double beta,
                                  dpy_arr old_mean, dpy_arr old_covar, dpy_arr target_mean, dpy_arr target_covar){
               return from_mat<double>(obj->more_step(eps, beta, to_vec<double>(old_mean), to_mat<double>(old_covar),
                                       to_vec<double>(target_mean), to_mat<double>(target_covar)));},
    py::arg("eps"), py::arg("beta"), py::arg("old_mean"),
    py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar"));

    mp.def_property_readonly("last_eta", &MoreProjection::get_last_eta);
    mp.def_property_readonly("last_omega", &MoreProjection::get_last_omega);
    mp.def_property_readonly("was_succ", &MoreProjection::was_succ);
    mp.def_property_readonly("res_text", &MoreProjection::get_res_txt);

    py::class_<BatchedProjection> bp(p, "BatchedProjection");
    bp.def(py::init([](uword batch_size, uword dim){
        return new BatchedProjection(batch_size, dim);}),
        py::arg("batchsize"), py::arg("dim"));

    bp.def("more_step", [](BatchedProjection* obj, dpy_arr epss, dpy_arr betas,
                           dpy_arr old_means, dpy_arr old_covars, dpy_arr target_means, dpy_arr target_covars){
        mat means;
        cube covs;
        std::tie(means, covs) = obj->more_step(
                to_vec<double>(epss), to_vec<double>(betas),
                to_mat<double>(old_means), to_cube<double>(old_covars),
                to_mat<double>(target_means), to_cube<double>(target_covars));
        return std::make_tuple(from_mat(means), from_cube(covs));
        },
           py::arg("eps"), py::arg("beta"), py::arg("old_mean"),
           py::arg("old_covar"), py::arg("target_mean"), py::arg("target_covar")
    );
}