#ifndef LIMBO_OPT_CMAES_HPP
#define LIMBO_OPT_CMAES_HPP

#include <vector>
#include <iostream>
#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>
#include <limbo/opt/optimizer.hpp>

#ifndef USE_LIBCMAES
#warning NO libcmaes support
#else

#include <libcmaes/cmaes.h>

namespace limbo {
    namespace defaults {
        struct opt_cmaes {
            BO_PARAM(int, restarts, 1);
            BO_PARAM(double, max_fun_evals, -1);
        };
    }
    namespace opt {
        template <typename Params>
        struct Cmaes {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, double bounded) const
            {
                size_t dim = init.size();

                // wrap the function
                libcmaes::FitFunc f_cmaes = [&](const double* x, const int n) {
		Eigen::Map<const Eigen::VectorXd> m(x, n);
		// remember that our optimizers maximize
		return -eval(f, m);
                };

                if (bounded)
                    return _opt_bounded(f_cmaes, dim, init);
                else
                    return _opt_unbounded(f_cmaes, dim, init);
            }

        private:
            // F is a CMA-ES style function, not our function
            template <typename F>
            Eigen::VectorXd _opt_unbounded(F& f_cmaes, int dim, const Eigen::VectorXd& init) const
            {
                using namespace libcmaes;
                // initial step-size, i.e. estimated initial parameter error.
                double sigma = 0.5;
                std::vector<double> x0(init.data(), init.data() + init.size());

                CMAParameters<> cmaparams(x0, sigma);
                _set_common_params(cmaparams, dim);
                // used by restart I think
                cmaparams.set_x0(-1.0, 1.0);

                // the optimization itself
                CMASolutions cmasols = cmaes<>(f_cmaes, cmaparams);
                return cmasols.get_best_seen_candidate().get_x_dvec();
            }

            // F is a CMA-ES style function, not our function
            template <typename F>
            Eigen::VectorXd _opt_bounded(F& f_cmaes, int dim, const Eigen::VectorXd& init) const
            {
                using namespace libcmaes;
                // create the parameter object
                // boundary_transformation
                double lbounds[dim], ubounds[dim]; // arrays for lower and upper parameter bounds, respectively
                for (int i = 0; i < dim; i++) {
                    lbounds[i] = 0.0;
                    ubounds[i] = 1.0;
                }
                GenoPheno<pwqBoundStrategy> gp(lbounds, ubounds, dim);
                // initial step-size, i.e. estimated initial parameter error.
                // we suppose we are optimizing on [0, 1], but we have no idea where to start
                double sigma = 0.5;
                std::vector<double> x0(init.data(), init.data() + init.size());
                // -1 for automatically decided lambda, 0 is for random seeding of the internal generator.
                CMAParameters<GenoPheno<pwqBoundStrategy>> cmaparams(dim, &x0.front(), sigma, -1, 0, gp);
                _set_common_params(cmaparams, dim);
                // used by restart I think
                cmaparams.set_x0(0, 1.0);

                // the optimization itself
                CMASolutions cmasols = cmaes<GenoPheno<pwqBoundStrategy>>(f_cmaes, cmaparams);
                //cmasols.print(std::cout, 1, gp);
                //to_f_representation
                return gp.pheno(cmasols.get_best_seen_candidate().get_x_dvec());
            }

            template <typename P>
            void _set_common_params(P& cmaparams, int dim) const
            {
                using namespace libcmaes;

                // set multi-threading to true
                cmaparams.set_mt_feval(true);
                // aCMAES should be the best choice
                // [see: https://github.com/beniz/libcmaes/wiki/Practical-hints ]
                // but we want the restart -> aIPOP_CMAES
                cmaparams.set_algo(aIPOP_CMAES);
                cmaparams.set_restarts(Params::opt_cmaes::restarts());
                // if no max fun evals provided, we compute a recommended value
                size_t max_evals = Params::opt_cmaes::max_fun_evals() < 0
                    ? (900.0 * (dim + 3.0) * (dim + 3.0))
                    : Params::opt_cmaes::max_fun_evals();
                cmaparams.set_max_fevals(max_evals);
                // max iteration is here only for security
                cmaparams.set_max_iter(100000);
                // we do not know if what is the actual maximum / minimum of the function
                // therefore we deactivate this stopping criterion
                cmaparams.set_stopping_criteria(FTARGET, false);
            }
        };
    }
}
#endif
#endif
