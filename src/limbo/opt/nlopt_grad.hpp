#ifndef LIMBO_OPT_NLOPT_GRAD_HPP
#define LIMBO_OPT_NLOPT_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

namespace limbo {
    namespace opt {
        template <typename Params, nlopt::algorithm Algorithm = nlopt::LD_MMA>
        struct NLOptGrad {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f) const
            {
                nlopt::opt opt(Algorithm, f.param_size());

                opt.set_max_objective(nlopt_func<F>, (void*)&f);

                std::vector<double> x(f.init().size());
                Eigen::VectorXd::Map(&x[0], f.init().size()) = f.init();

                opt.set_ftol_rel(Params::nlopt::epsilon());
                opt.set_maxeval(Params::nlopt::iters());

                double min;

                opt.optimize(x, min);

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
