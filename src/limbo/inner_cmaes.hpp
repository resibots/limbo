#ifndef INNER_CMAES_HPP_
#define INNER_CMAES_HPP_


#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <limits>

#include <stdio.h>
#include <stdlib.h> /* free() */
#include <stddef.h> /* NULL */

#include "cmaes/cmaes_interface.h"
#include "cmaes/boundary_transformation.h"


namespace limbo {

  namespace defaults {
    struct cmaes {
      BO_PARAM(int, nrestarts, 10);
      BO_PARAM(double, max_fun_evals, -1);
    };
  }

  namespace inner_optimization {
    template <typename Params>
    struct Cmaes {
      Cmaes() {}
      template <typename AcquisitionFunction>
      Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim) const {
          return this->operator()(acqui, dim, Eigen::VectorXd::Constant(dim, 0.5));
      }
      template <typename AcquisitionFunction>
      Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim, const Eigen::VectorXd& init) const {
        int nrestarts = Params::cmaes::nrestarts();
        double incpopsize = 2;
        cmaes_t evo;
        double *const*pop;
        double *fitvals;
        double fbestever = 0, *xbestever = NULL;
        double fmean;
        int i, irun, lambda = 0, countevals = 0;
        char const * stop;
        boundary_transformation_t boundaries;
        double lowerBounds[] = {0.0}; // TODO  put this into params?
        double upperBounds[] = {1.0};
        int nb_bounds = 1; /* numbers used from lower and upperBounds */

        boundary_transformation_init(&boundaries, lowerBounds, upperBounds, nb_bounds);
        double* x_in_bounds = cmaes_NewDouble(dim);
        double init_point[dim];
        for (int i = 0; i < dim; ++i)
          init_point[i] = init(i);
         for (irun = 0; irun < nrestarts + 1; ++irun) {
          fitvals = cmaes_init(&evo, acqui.dim(), init_point, NULL, 0, lambda, NULL);
          evo.countevals = countevals;
          evo.sp.stopMaxFunEvals =
            Params::cmaes::max_fun_evals() < 0 ?
            (900.0 * (dim + 3.0) * (dim + 3.0))
            : Params::cmaes::max_fun_evals();

          Eigen::VectorXd v(acqui.dim());
          while (!(stop = cmaes_TestForTermination(&evo))) {
            pop = cmaes_SamplePopulation(&evo);
            for (i = 0; i < cmaes_Get(&evo, "popsize"); ++i) {
              boundary_transformation(&boundaries, pop[i], x_in_bounds, dim);

              for (int j = 0; j < v.size(); ++j)
                v(j) = x_in_bounds[j];
              fitvals[i] = -acqui(v);
            }
            cmaes_UpdateDistribution(&evo, fitvals);
          }

          lambda = incpopsize * cmaes_Get(&evo, "lambda");
          countevals = cmaes_Get(&evo, "eval");

          if (irun == 0 || cmaes_Get(&evo, "fbestever") < fbestever) {
            fbestever = cmaes_Get(&evo, "fbestever");
            xbestever = cmaes_GetInto(&evo, "xbestever", xbestever); /* alloc mem if needed */
          }
          const double *xmean = cmaes_GetPtr(&evo, "xmean");
          for (int j = 0; j < v.size(); ++j)
            v(j) = xmean[j];

          if ((fmean =  -acqui(v)) < fbestever) {
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
        boundary_transformation(&boundaries, xbestever, x_in_bounds, dim);

        Eigen::VectorXd result = Eigen::VectorXd::Zero(acqui.dim());
        for (size_t i = 0; i < acqui.dim(); ++i)
          result(i) = x_in_bounds[i];
        free(xbestever);
        boundary_transformation_exit(&boundaries);

        return result;

      }
     private:
      double *_ar_funvals;
      double * const * _cmaes_pop;
      int _lambda;


    };


  }
}
#endif
