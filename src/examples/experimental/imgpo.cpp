//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#ifdef USE_TBB
#include <tbb/concurrent_hash_map.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#else
#include <map>
#endif

#include <limbo/experimental/acqui/ucb_imgpo.hpp>
#include <limbo/experimental/bayes_opt/imgpo.hpp>
#include <limbo/init/no_init.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>

using namespace limbo;

static constexpr int nb_replicates = 4;

namespace colors {
    static const char* red = "\33[31m";
    static const char* green = "\33[32m";
    static const char* yellow = "\33[33m";
    static const char* reset = "\33[0m";
    static const char* bold = "\33[1m";
}

// support functions
inline double sign(double x)
{
    if (x < 0)
        return -1;
    if (x > 0)
        return 1;
    return 0;
}

inline double hat(double x)
{
    if (x != 0)
        return log(fabs(x));
    return 0;
}

inline double c1(double x)
{
    if (x > 0)
        return 10;
    return 5.5;
}

inline double c2(double x)
{
    if (x > 0)
        return 7.9;
    return 3.1;
}

inline Eigen::VectorXd t_osz(const Eigen::VectorXd& x)
{
    Eigen::VectorXd r = x;
    for (int i = 0; i < x.size(); i++)
        r(i) = sign(x(i)) * exp(hat(x(i)) + 0.049 * sin(c1(x(i)) * hat(x(i))) + sin(c2(x(i)) * hat(x(i))));
    return r;
}

struct Sphere {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::Vector2d opt(0.5, 0.5);
        return tools::make_vector(-(x - opt).squaredNorm());
    }
};

struct Ellipsoid {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::Vector2d opt(0.5, 0.5);
        Eigen::Vector2d z = t_osz(x - opt);
        double r = 0;
        for (size_t i = 0; i < dim_in(); ++i)
            r += std::pow(10, ((double)i) / (dim_in() - 1.0)) * z(i) * z(i) + 1;
        return tools::make_vector(-r);
    }
};

struct Rastrigin {
    BO_PARAM(size_t, dim_in, 4);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        double f = 10 * x.size();
        for (int i = 0; i < x.size(); ++i)
            f += x(i) * x(i) - 10 * cos(2 * M_PI * x(i));
        return tools::make_vector(-f);
    }
};

// see : http://www.sfu.ca/~ssurjano/hart3.html
struct Hartman3 {
    BO_PARAM(size_t, dim_in, 3);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::Matrix<double, 4, 3> a, p;
        a << 3.0, 10, 30, 0.1, 10, 35, 3.0, 10, 30, 0.1, 10, 36;
        p << 0.3689, 0.1170, 0.2673, 0.4699, 0.4387, 0.7470, 0.1091, 0.8732, 0.5547,
            0.0382, 0.5743, 0.8828;
        Eigen::Vector4d alpha;
        alpha << 1.0, 1.2, 3.0, 3.2;

        double res = 0;
        for (int i = 0; i < 4; i++) {
            double s = 0.0f;
            for (size_t j = 0; j < 3; j++) {
                s += a(i, j) * (x(j) - p(i, j)) * (x(j) - p(i, j));
            }
            res += alpha(i) * exp(-s);
        }
        return tools::make_vector(res);
    }
};

// see : http://www.sfu.ca/~ssurjano/hart6.html
struct Hartman6 {
    BO_PARAM(size_t, dim_in, 6);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::Matrix<double, 4, 6> a, p;
        a << 10, 3, 17, 3.5, 1.7, 8, 0.05, 10, 17, 0.1, 8, 14, 3, 3.5, 1.7, 10, 17,
            8, 17, 8, 0.05, 10, 0.1, 14;
        p << 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886, 0.2329, 0.4135, 0.8307,
            0.3736, 0.1004, 0.9991, 0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665,
            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381;

        Eigen::Vector4d alpha;
        alpha << 1.0, 1.2, 3.0, 3.2;

        double res = 0;
        for (int i = 0; i < 4; i++) {
            double s = 0.0f;
            for (size_t j = 0; j < 6; j++) {
                s += a(i, j) * (x(j) - p(i, j)) * (x(j) - p(i, j));
            }
            res += alpha(i) * exp(-s);
        }
        return tools::make_vector(res);
    }
};

// see : http://www.sfu.ca/~ssurjano/goldpr.html
// (with ln, as suggested in Jones et al.)
struct GoldenPrice {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& xx) const
    {
        Eigen::VectorXd x = (4.0 * xx).array() - 2.0;
        double r = (1 + (x(0) + x(1) + 1) * (x(0) + x(1) + 1) * (19 - 14 * x(0) + 3 * x(0) * x(0) - 14 * x(1) + 6 * x(0) * x(1) + 3 * x(1) * x(1))) * (30 + (2 * x(0) - 3 * x(1)) * (2 * x(0) - 3 * x(1)) * (18 - 32 * x(0) + 12 * x(0) * x(0) + 48 * x(1) - 36 * x(0) * x(1) + 27 * x(1) * x(1)));

        return tools::make_vector(-log(r) + 5);
    }
};

struct Params {
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };

    struct bayes_opt_imgpo : public defaults::bayes_opt_imgpo {
    };

    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };

    struct kernel_exp : public defaults::kernel_exp {
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };

    struct acqui_ucb_imgpo : public defaults::acqui_ucb_imgpo {
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0);
    };
};

template <typename T>
void print_res(const T& r)
{
    std::cout << "====== RESULTS ======" << std::endl;
    for (auto x : r) {
        for (auto y : x.second) {
            std::cout << x.first << "\t =>"
                      << " found :" << y.second << " expected " << y.first
                      << std::endl;
        }
        std::vector<std::pair<double, double>>& v = x.second;
        std::sort(v.begin(), v.end(),
            [](const std::pair<double, double>& x1,
                      const std::pair<double, double>& x2) {
                // clang-format off
                return x1.second < x2.second;
                // clang-format on
            });
        double med = v[v.size() / 2].second;
        if (fabs(v[0].first - med) < 0.05)
            std::cout << "[" << colors::green << "OK" << colors::reset << "] ";
        else
            std::cout << "[" << colors::red << "ERROR" << colors::reset << "] ";
        std::cout << colors::yellow << colors::bold << " -- " << x.first
                  << colors::reset << " ";
        std::cout << "Median: " << med << " error :" << fabs(v[0].first - med)
                  << std::endl;
    }
}

bool is_in_argv(int argc, char** argv, const char* needle)
{
    auto it = std::find_if(argv, argv + argc,
        [=](const char* s) {
            // clang-format off
            return strcmp(needle, s) == 0;
            // clang-format on
        });
    return !(it == argv + argc);
}

template <typename T1, typename T2>
void add_to_results(const char* key, T1& map, const T2& p)
{
#ifdef USE_TBB
    typename T1::accessor a;
    if (!map.find(a, key))
        map.insert(a, key);
#else
    typename T1::iterator a;
    a = map.find(key);
    if (a == map.end())
        map[key] = std::vector<std::pair<double, double>>();
#endif
    a->second.push_back(p);
}

int main(int argc, char** argv)
{
    tools::par::init();

#ifdef USE_TBB
    using res_t = tbb::concurrent_hash_map<std::string, std::vector<std::pair<double, double>>>;
#else
    using res_t = std::map<std::string, std::vector<std::pair<double, double>>>;
#endif
    res_t results;

    using kf_t = kernel::Exp<Params>;
    using mean_t = mean::Constant<Params>;
    using model_t = model::GP<Params, kf_t, mean_t>;
    using init_t = init::NoInit<Params>;
    using acqui_t = acqui::experimental::UCB_IMGPO<Params, model_t>;

    using Opt_t = bayes_opt::experimental::IMGPO<Params, modelfun<model_t>, initfun<init_t>, acquifun<acqui_t>>;

    if (!is_in_argv(argc, argv, "--only") || is_in_argv(argc, argv, "sphere"))
        tools::par::replicate(nb_replicates, [&]() {
            // clang-format off
                Opt_t opt;
                opt.optimize(Sphere());
                Eigen::Vector2d s_val(0.5, 0.5);
                double x_opt = FirstElem()(Sphere()(s_val));
                add_to_results("Sphere", results, std::make_pair(x_opt, opt.best_observation()(0)));
            // clang-format on
        });

    if (!is_in_argv(argc, argv, "--only") || is_in_argv(argc, argv, "ellipsoid"))
        tools::par::replicate(nb_replicates, [&]() {
            // clang-format off
                Opt_t opt;
                opt.optimize(Ellipsoid());
                Eigen::Vector2d s_val(0.5, 0.5);
                double x_opt = FirstElem()(Ellipsoid()(s_val));
                add_to_results("Ellipsoid", results, std::make_pair(x_opt, opt.best_observation()(0)));
            // clang-format on
        });

    if (!is_in_argv(argc, argv, "--only") || is_in_argv(argc, argv, "rastrigin"))
        tools::par::replicate(nb_replicates, [&]() {
            // clang-format off
                Opt_t opt;
                opt.optimize(Rastrigin());
                Eigen::Vector4d s_val(0, 0, 0, 0);
                double x_opt = FirstElem()(Rastrigin()(s_val));
                add_to_results("Rastrigin", results, std::make_pair(x_opt, opt.best_observation()(0)));
            // clang-format on
        });

    if (!is_in_argv(argc, argv, "--only") || is_in_argv(argc, argv, "hartman3"))
        tools::par::replicate(nb_replicates, [&]() {
            // clang-format off
                Opt_t opt;
                opt.optimize(Hartman3());
                // double s_max = 3.86278;
                Eigen::Vector3d s_val(0.114614, 0.555549, 0.852547);
                double x_opt = FirstElem()(Hartman3()(s_val));
                add_to_results("Hartman 3", results, std::make_pair(x_opt, opt.best_observation()(0)));
            // clang-format on
        });

    if (!is_in_argv(argc, argv, "--only") || is_in_argv(argc, argv, "hartman6"))
        tools::par::replicate(nb_replicates, [&]() {
            // clang-format off
                Opt_t opt;
                opt.optimize(Hartman6());
                Eigen::Matrix<double, 6, 1> s_val;
                s_val << 0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573;
                //double s_max = 3.32237;
                double x_opt = FirstElem()(Hartman6()(s_val));
                add_to_results("Hartman 6", results, std::make_pair(x_opt, opt.best_observation()(0)));
            // clang-format on
        });

    if (!is_in_argv(argc, argv, "--only") || is_in_argv(argc, argv, "golden_price"))
        tools::par::replicate(nb_replicates, [&]() {
            // clang-format off
                Opt_t opt;
                opt.optimize(GoldenPrice());
                //    double s_max = -log(3);
                Eigen::Vector2d s_val(0.5, 0.25);
                double x_opt = FirstElem()(GoldenPrice()(s_val));
                add_to_results("Golden Price", results, std::make_pair(x_opt, opt.best_observation()(0)));
            // clang-format on
        });

    std::cout << "Benchmark finished." << std::endl;

    print_res(results);
    return 0;
}
