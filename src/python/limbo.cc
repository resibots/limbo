#include <iostream>
#include <limbo/model.hpp>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

//  clang++ -bundle -I /usr/local/include/eigen3 -I../ -I /usr/local/include/ -L /usr/local/lib   -std=c++14 -L/usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin/  -lpython3.7m -ldl -framework CoreFoundation -framework CoreFoundation  `python3.7-config --cflags --libs` -lpython3.7m   ./limbo.cc -o limbo.cpython-37m-darwin.so

using namespace limbo;

namespace py = pybind11;

struct Params {
    struct kernel : public defaults::kernel {
        BO_DYN_PARAM(double, noise);
        BO_DYN_PARAM(bool, optimize_noise);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public defaults::opt_rprop {
        BO_DYN_PARAM(int, iterations);
        BO_DYN_PARAM(double, eps_stop);
    };
    struct mean_constant {
        BO_DYN_PARAM(double, constant);
    };
};

BO_DECLARE_DYN_PARAM(double, Params::kernel, noise);
BO_DECLARE_DYN_PARAM(bool, Params::kernel, optimize_noise);
BO_DECLARE_DYN_PARAM(int, Params::opt_rprop, iterations);
BO_DECLARE_DYN_PARAM(double, Params::opt_rprop, eps_stop);
BO_DECLARE_DYN_PARAM(double, Params::mean_constant, constant);

using mean_t = mean::Constant<Params>;
using kernel_t = kernel::SquaredExpARD<Params>;
using GP_t = model::GP<Params, kernel_t, mean_t, model::gp::KernelLFOpt<Params>>;
using MultiGP_t = model::MultiGP<Params, model::GP, kernel_t, mean_t,
    model::multi_gp::ParallelLFOpt<Params, model::gp::KernelLFOpt<Params>>>;
bool verbose = false;

void _set_params(py::kwargs kwargs)
{
    Params::kernel::set_noise(0.01);
    Params::kernel::set_optimize_noise(false);
    Params::opt_rprop::set_iterations(300);
    Params::opt_rprop::set_eps_stop(0.0);
    Params::mean_constant::set_constant(0.0);

    for (auto item : kwargs) {
        auto key = py::cast<std::string>(item.first);
        if (key == "noise") {
            Params::kernel::set_noise(py::cast<double>(item.second));
        }
        else if (key == "optimize_noise") {
            Params::kernel::set_optimize_noise(py::cast<bool>(item.second));
        }
        else if (key == "iterations") {
            Params::opt_rprop::set_iterations(py::cast<int>(item.second));
        }
        else if (key == "eps_stop") {
            Params::opt_rprop::set_eps_stop(py::cast<double>(item.second));
        }
        else if (key == "mean") {
            Params::mean_constant::set_constant(py::cast<double>(item.second));
        }
        else if (key == "verbose") {
            verbose = py::cast<bool>(item.second);
        }
        else {
            std::cerr << "Unrecognized parameter:" << item.first << std::endl;
        }
    }
    if (verbose) {
        std::cout << "opt_rprop::iterations (iterations) => " << Params::opt_rprop::iterations()
                  << std::endl
                  << "opt_rprop::eps_stop (eps_stop)=> " << Params::opt_rprop::eps_stop()
                  << std::endl
                  << "kernel::noise (noise) => " << Params::kernel::noise() << std::endl
                  << "WARNING: if not optimized, noise is set for all the GPs !" << std::endl
                  << "kernel::optimize_noise (optimize_noise)=> "
                  << Params::kernel::optimize_noise() << std::endl;
    }
}

template <typename GP>
GP make_gp(py::args args, py::kwargs kwargs)
{
    auto train_x = py::cast<std::vector<Eigen::VectorXd>>(args[0]);
    auto train_y = py::cast<std::vector<Eigen::VectorXd>>(args[1]);
    assert(train_x.size() == train_y.size());

    _set_params(kwargs);

    GP gp;
    gp.compute(train_x, train_y, true);
    gp.optimize_hyperparams();
    if (verbose) {
        std::cout << "Learning done. data points => " << train_x.size() << std::endl;
    }
    return gp;
}

PYBIND11_MODULE(limbo, m)
{
    m.doc() = "Simplified Limbo (GP only)";
    m.def("make_gp", &make_gp<GP_t>, "Create a GP");
    m.def("make_multi_gp", &make_gp<MultiGP_t>, "Create a Multi GP (multi-dimensional output)");

    py::class_<GP_t>(m, "GP").def("query", &GP_t::query).def("get_log_lik", &GP_t::get_log_lik);
    py::class_<MultiGP_t>(m, "MultiGP").def("query", &MultiGP_t::query);
}