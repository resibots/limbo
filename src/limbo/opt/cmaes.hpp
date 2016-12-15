//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
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
#ifndef LIMBO_OPT_CMAES_HPP
#define LIMBO_OPT_CMAES_HPP

#include <Eigen/Core>
#include <iostream>
#include <vector>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>

#ifndef USE_LIBCMAES
#warning NO libcmaes support
#else

#include <libcmaes/cmaes.h>

namespace limbo {
    namespace defaults {
        struct opt_cmaes {
            /// @ingroup opt_defaults
            /// number of restarts of CMA-ES
            BO_PARAM(int, restarts, 1);
            /// @ingroup opt_defaults
            /// number of calls to the function to be optimized
            BO_PARAM(double, max_fun_evals, -1);
            /// @ingroup opt_defaults
            /// threshold based on the difference in value of a fixed number
            /// of trials
            BO_PARAM(double, fun_tolerance, -1);
            /// @ingroup opt_defaults
            /// function value target
            BO_PARAM(double, fun_target, -1);
            /// @ingroup opt_defaults
            /// computes initial objective function value
            BO_PARAM(bool, fun_compute_initial, false);
            /// @ingroup opt_defaults
            /// sets the version of cmaes to use
            BO_PARAM(int, variant, aIPOP_CMAES);
            /// @ingroup opt_defaults
            /// defines elitism strategy:
            /// 0 -> no elitism
            /// 1 -> elitism: reinjects the best-ever seen solution
            /// 2 -> initial elitism: reinject x0 as long as it is not improved upon
            /// 3 -> initial elitism on restart: restart if best encountered solution is not the the final
            /// solution and reinjects the best solution until the population has better fitness, in its majority
            BO_PARAM(int, elitism, 0);
            /// @ingroup opt_defaults
            /// enables or disables uncertainty handling
            BO_PARAM(bool, handle_uncertainty, false);
            /// @ingroup opt_defaults
            /// enables or disables verbose mode for cmaes
            BO_PARAM(bool, verbose, false);
        };
    }

    namespace opt {
        /// @ingroup opt
        /// Covariance Matrix Adaptation Evolution Strategy by Hansen et al.
        /// (See: https://www.lri.fr/~hansen/cmaesintro.html)
        /// - our implementation is based on libcmaes (https://github.com/beniz/libcmaes)
        /// - Support bounded and unbounded optimization
        /// - Only available if libcmaes is installed (see the compilation instructions)
        ///
        /// - Parameters :
        ///   - double max_fun_evals
        ///   - int restarts
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
                cmaparams.set_algo(Params::opt_cmaes::variant());
                cmaparams.set_restarts(Params::opt_cmaes::restarts());
                cmaparams.set_elitism(Params::opt_cmaes::elitism());

                // if no max fun evals provided, we compute a recommended value
                size_t max_evals = Params::opt_cmaes::max_fun_evals() < 0
                    ? (900.0 * (dim + 3.0) * (dim + 3.0))
                    : Params::opt_cmaes::max_fun_evals();
                cmaparams.set_max_fevals(max_evals);
                // max iteration is here only for security
                cmaparams.set_max_iter(100000);

                if (Params::opt_cmaes::fun_tolerance() == -1) {
                    // we do not know if what is the actual maximum / minimum of the function
                    // therefore we deactivate this stopping criterion
                    cmaparams.set_stopping_criteria(FTARGET, false);
                }
                else {
                    // the FTARGET criteria also allows us to enable ftolerance
                    cmaparams.set_stopping_criteria(FTARGET, true);
                    cmaparams.set_ftolerance(Params::opt_cmaes::fun_tolerance());
                }

                // we allow to set the ftarget parameter
                if (Params::opt_cmaes::fun_target() != -1) {
                    cmaparams.set_stopping_criteria(FTARGET, true);
                    cmaparams.set_ftarget(-Params::opt_cmaes::fun_target());
                }

                // enable stopping criteria by several equalfunvals and maxfevals
                cmaparams.set_stopping_criteria(EQUALFUNVALS, true);
                cmaparams.set_stopping_criteria(MAXFEVALS, true);

                // enable additional criterias to stop
                cmaparams.set_stopping_criteria(TOLX, true);
                cmaparams.set_stopping_criteria(CONDITIONCOV, true);

                // enable or disable different parameters
                cmaparams.set_initial_fvalue(Params::opt_cmaes::fun_compute_initial());
                cmaparams.set_uh(Params::opt_cmaes::handle_uncertainty());
                cmaparams.set_quiet(!Params::opt_cmaes::verbose());
            }
        };
    }
}
#endif
#endif
