#ifndef LIMBO_OPT_NLOPT_GRAD_HPP
#define LIMBO_OPT_NLOPT_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct opt_nloptgrad {
            BO_PARAM(int, iterations, 500);
        };
    }
    namespace opt {
        template <typename Params, nlopt::algorithm Algorithm>
        struct NLOptGrad {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, bool bounded) const
            {
                nlopt::opt opt(Algorithm, f.param_size());

                opt.set_max_objective(nlopt_func<F>, (void*)&f);

                std::vector<double> x(f.init().size());
                Eigen::VectorXd::Map(&x[0], f.init().size()) = f.init();

                opt.set_maxeval(Params::opt_nloptgrad::iterations());

                if (bounded) {
                    opt.set_lower_bounds(std::vector<double>(f.param_size(), 0));
                    opt.set_upper_bounds(std::vector<double>(f.param_size(), 1));
                }

                double max;

                opt.optimize(x, max);

                return Eigen::VectorXd::Map(x.data(), x.size());
            }

        protected:
            template <typename F>
            static double nlopt_func(const std::vector<double>& x, std::vector<double>& grad, void* my_func_data)
            {
                F* f = (F*)(my_func_data);
                Eigen::VectorXd params = Eigen::VectorXd::Map(x.data(), x.size());
                double v;
                if (!grad.empty()) {
                    Eigen::VectorXd g;
                    auto p = f->utility_and_grad(params);
                    v = std::get<0>(p);
                    g = std::get<1>(p);
                    Eigen::VectorXd::Map(&grad[0], g.size()) = g;
                }
                else {
                    v = f->utility(params);
                }
                return v;
            }
        };
    }
}

#endif
#endif
