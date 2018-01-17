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
#ifndef LIMBO_OPT_NLOPT_GRAD_HPP
#define LIMBO_OPT_NLOPT_GRAD_HPP

#ifndef USE_NLOPT
#warning No NLOpt
#else
#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct opt_nloptgrad {
            /// @ingroup opt_defaults
            /// number of calls to the optimized function
            BO_PARAM(int, iterations, 500);
            /// @ingroup opt_defaults
            /// tolerance for convergence: stop when an optimization step (or an
            /// estimate of the optimum) changes the objective function value by
            /// less than tol multiplied by the absolute value of the function
            /// value.
            /// IGNORED if negative
            BO_PARAM(double, fun_tolerance, -1);
            /// @ingroup opt_defaults
            /// tolerance for convergence: stop when an optimization step (or an
            /// estimate of the optimum) changes all the parameter values by
            /// less than tol multiplied by the absolute value of the parameter
            /// value.
            /// IGNORED if negative
            BO_PARAM(double, xrel_tolerance, -1);
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
         - LD_VAR1
         - LD_VAR2
         - LD_TNEWTON
         - LD_TNEWTON_RESTART
         - LD_TNEWTON_PRECOND
         - LD_TNEWTON_PRECOND_RESTART
         - GD_MLSL
         - GD_MLSL_LDS
         - LD_MMA
         - LD_AUGLAG
         - LD_AUGLAG_EQ
         - LD_SLSQP
         - LD_CCSAQ

         Parameters :
         - int iterations
         - double fun_tolerance
         - double xrel_tolerance
        */
        template <typename Params, nlopt::algorithm Algorithm = nlopt::LD_LBFGS>
        struct NLOptGrad {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                // Assert that the algorithm is gradient-based
                // TO-DO: Add support for MLSL (Multi-Level Single-Linkage)
                // TO-DO: Add better support for AUGLAG and AUGLAG_EQ
                // clang-format off
                static_assert(Algorithm == nlopt::LD_MMA || Algorithm == nlopt::LD_SLSQP ||
                    Algorithm == nlopt::LD_LBFGS || Algorithm == nlopt::LD_TNEWTON_PRECOND_RESTART ||
                    Algorithm == nlopt::LD_TNEWTON_PRECOND || Algorithm == nlopt::LD_TNEWTON_RESTART ||
                    Algorithm == nlopt::LD_TNEWTON || Algorithm == nlopt::LD_VAR2 ||
                    Algorithm == nlopt::LD_VAR1 || Algorithm == nlopt::GD_STOGO ||
                    Algorithm == nlopt::GD_STOGO_RAND || Algorithm == nlopt::LD_LBFGS_NOCEDAL ||
                    Algorithm == nlopt::LD_AUGLAG || Algorithm == nlopt::LD_AUGLAG_EQ ||
                    Algorithm == nlopt::LD_CCSAQ, "NLOptGrad accepts gradient-based nlopt algorithms only");
                // clang-format on
                int dim = init.size();
                nlopt::opt opt(Algorithm, dim);

                opt.set_max_objective(nlopt_func<F>, (void*)&f);

                std::vector<double> x(dim);
                Eigen::VectorXd::Map(&x[0], dim) = init;

                opt.set_maxeval(Params::opt_nloptgrad::iterations());
                opt.set_ftol_rel(Params::opt_nloptgrad::fun_tolerance());
                opt.set_xtol_rel(Params::opt_nloptgrad::xrel_tolerance());

                if (bounded) {
                    opt.set_lower_bounds(std::vector<double>(dim, 0));
                    opt.set_upper_bounds(std::vector<double>(dim, 1));
                }

                double max;

                try {
                    opt.optimize(x, max);
                }
                catch (nlopt::roundoff_limited& e) {
                    // In theory it's ok to ignore this error
                    std::cerr << "[NLOptGrad]: " << e.what() << std::endl;
                }
                catch (std::invalid_argument& e) {
                    // In theory it's ok to ignore this error
                    std::cerr << "[NLOptGrad]: " << e.what() << std::endl;
                }
                catch (std::runtime_error& e) {
                    // In theory it's ok to ignore this error
                    std::cerr << "[NLOptGrad]: " << e.what() << std::endl;
                }

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
