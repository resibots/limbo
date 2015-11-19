#ifndef LIMBO_OPT_NLOPT_NO_GRAD_HPP
#define LIMBO_OPT_NLOPT_NO_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

namespace limbo {
    namespace opt {
        template <typename Params, nlopt::algorithm Algorithm = nlopt::LN_COBYLA>
        struct NLOptNoGrad {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f) const
            {
                // Assert that the algorithm is non-gradient
                assert(Algorithm == nlopt::LN_COBYLA || Algorithm == nlopt::LN_BOBYQA || 
                    Algorithm == nlopt::LN_NEWUOA || Algorithm == nlopt::LN_NEWUOA_BOUND || 
                    Algorithm == nlopt::LN_PRAXIS || Algorithm == nlopt::LN_NELDERMEAD ||
                    Algorithm == nlopt::LN_SBPLX);

                nlopt::opt opt(Algorithm, f.param_size());

                opt.set_max_objective(this->nlopt_func<F>, (void*)&f);

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
                double v = f->utility(params);
                return v;
            }
        };
    }
}

#endif
#endif
