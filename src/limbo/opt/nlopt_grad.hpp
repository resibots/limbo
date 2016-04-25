#ifndef LIMBO_OPT_NLOPT_GRAD_HPP
#define LIMBO_OPT_NLOPT_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/opt/optimizer.hpp>

namespace limbo {
    namespace defaults {
        struct opt_nloptgrad {
            /// @ingroup opt_defaults
            /// number of calls to the optimized function
            BO_PARAM(int, iterations, 500);
            /// @ingroup opt_defaults
            /// nlopt relative and absolute tolerance stopping criteria
            BO_PARAM(double, epsilon, 1e-12);
        };
    }
    namespace opt {
        /**
        @ingroup opt
         Binding to gradient-based NLOpt algorithms.
         See: http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms

         Algorithms:
         - GD_STOGO
         - GD_STOGO_RAND
         - LD_LBFGS_NOCEDAL
         - LD_LBFGS
         - LN_PRAXIS
         - LD_VAR1
         - LD_VAR2
         - LD_TNEWTON
         - LD_TNEWTON_RESTART
         - LD_TNEWTON_PRECOND
         - LD_TNEWTON_PRECOND_RESTART
         - GD_MLSL
         - GN_MLSL_LDS
         - GD_MLSL_LDS
         - LD_MMA
         - LD_AUGLAG
         - LD_AUGLAG_EQ
         - LD_SLSQP
         - LD_CCSAQ

         Parameters :
         - int iterations
        */
        template <typename Params, nlopt::algorithm Algorithm = nlopt::LD_LBFGS>
        struct NLOptGrad {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                int dim = init.size();
                nlopt::opt opt(Algorithm, dim);

                opt.set_max_objective(nlopt_func<F>, (void*)&f);
                opt.set_ftol_rel(Params::opt_nloptgrad::epsilon());	
                opt.set_ftol_abs(Params::opt_nloptgrad::epsilon());

                std::vector<double> x(dim);
                Eigen::VectorXd::Map(&x[0], dim) = init;

                opt.set_maxeval(Params::opt_nloptgrad::iterations());

                if (bounded) {
                    opt.set_lower_bounds(std::vector<double>(dim, 0));
                    opt.set_upper_bounds(std::vector<double>(dim, 1));
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
                    auto r = eval_grad(*f, params);
                    v = opt::fun(r);
                    Eigen::VectorXd g = opt::grad(r);
                    Eigen::VectorXd::Map(&grad[0], g.size()) = g;
                }
                else {
                    v = eval(*f, params);
                }
                return v;
            }
        };
    }
}

#endif
#endif
