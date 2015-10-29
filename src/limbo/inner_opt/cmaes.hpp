#ifndef LIMBO_INNER_OPT_CMAES_HPP
#define LIMBO_INNER_OPT_CMAES_HPP

#include <vector>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h> /* free() */
#include <stddef.h> /* NULL */

#include <Eigen/Core>

#include <limbo/tools/parallel.hpp>

#include <cmaes/cmaes_interface.h>
#include <cmaes/boundary_transformation.h>

namespace limbo {

    namespace defaults {
        struct cmaes {
            BO_PARAM(int, nrestarts, 1);
            BO_PARAM(double, max_fun_evals, -1);
        };
    }

    namespace inner_opt {
        template <typename Params>
        struct Cmaes {
            Cmaes() {}

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const AggregatorFunction& afun) const
            {
                return this->operator()(acqui, dim_in,
                    Eigen::VectorXd::Constant(dim_in, 0.5), afun);
            }

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const Eigen::VectorXd& init, const AggregatorFunction& afun) const
            {

                int nrestarts = Params::cmaes::nrestarts();
                double incpopsize = 2;
                cmaes_t evo;
                double* const* pop;
                double* fitvals;
                double fbestever = 0, * xbestever = NULL;
                double fmean;
                int irun, lambda = 0, countevals = 0;
                char const* stop;
                boundary_transformation_t boundaries;
                double lowerBounds[] = {0.0}; // TODO  put this into params?
                double upperBounds[] = {1.006309}; // Allows solution to be pretty close to 1
                int nb_bounds = 1; /* numbers used from lower and upperBounds */

                boundary_transformation_init(&boundaries, lowerBounds, upperBounds,
                    nb_bounds);
                double* x_in_bounds = cmaes_NewDouble(dim_in);
                double init_point[dim_in];
                for (int i = 0; i < dim_in; ++i)
                    init_point[i] = init(i);

                for (irun = 0; irun < nrestarts + 1; ++irun) {

                    fitvals = cmaes_init(&evo, acqui.dim_in(), init_point, NULL, 0, lambda, NULL);

                    evo.countevals = countevals;
                    evo.sp.stopMaxFunEvals = Params::cmaes::max_fun_evals() < 0
                        ? (900.0 * (dim_in + 3.0) * (dim_in + 3.0))
                        : Params::cmaes::max_fun_evals();

                    int pop_size = cmaes_Get(&evo, "popsize");
                    double** all_x_in_bounds = new double* [pop_size];
                    for (int i = 0; i < pop_size; ++i)
                        all_x_in_bounds[i] = cmaes_NewDouble(dim_in);
                    std::vector<Eigen::VectorXd> pop_eigen(pop_size, Eigen::VectorXd(dim_in));

                    while (!(stop = cmaes_TestForTermination(&evo))) {
                        pop = cmaes_SamplePopulation(&evo);
                        tools::par::loop(0, pop_size, [&](int i) {
                                // clang-format off
                                boundary_transformation(&boundaries, pop[i], all_x_in_bounds[i], dim_in);
                                for (int j = 0; j < dim_in; ++j)
                                  pop_eigen[i](j) = all_x_in_bounds[i][j];
                                fitvals[i] = -acqui(pop_eigen[i], afun);
                            // clang-format on
                        });
                        cmaes_UpdateDistribution(&evo, fitvals);
                    }

                    for (int i = 0; i < pop_size; ++i)
                        free(all_x_in_bounds[i]);

                    lambda = incpopsize * cmaes_Get(&evo, "lambda");
                    countevals = cmaes_Get(&evo, "eval");

                    if (irun == 0 || cmaes_Get(&evo, "fbestever") < fbestever) {
                        fbestever = cmaes_Get(&evo, "fbestever");
                        xbestever = cmaes_GetInto(&evo, "xbestever",
                            xbestever); /* alloc mem if needed */
                    }
                    const double* xmean = cmaes_GetPtr(&evo, "xmean");
                    Eigen::VectorXd v(acqui.dim_in());
                    for (int j = 0; j < v.size(); ++j)
                        v(j) = xmean[j];

                    if ((fmean = -acqui(v, afun)) < fbestever) {
                        fbestever = fmean;
                        xbestever = cmaes_GetInto(&evo, "xmean", xbestever);
                    }

                    cmaes_exit(&evo);

                    if (stop) {
                        if (strncmp(stop, "Fitness", 7) == 0 || strncmp(stop, "MaxFunEvals", 11) == 0) {
                            //    printf("stop: %s", stop);
                            break;
                        }
                    }
                }
                boundary_transformation(&boundaries, xbestever, x_in_bounds, dim_in);

                Eigen::VectorXd result = Eigen::VectorXd::Zero(acqui.dim_in());
                for (size_t i = 0; i < acqui.dim_in(); ++i)
                    result(i) = x_in_bounds[i];
                free(xbestever);
                boundary_transformation_exit(&boundaries);

                free(x_in_bounds);

                return result;
            }

        private:
            double* _ar_funvals;
            double* const* _cmaes_pop;
            int _lambda;
        };
    }
}

#endif
