#ifndef INNER_OPTIMIZATION_HPP_
#define INNER_OPTIMIZATION_HPP_


#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <limits>

namespace limbo {
  namespace inner_optimization {
    template <typename Params>
    struct ExhaustiveSearch {

      ExhaustiveSearch() {
        nb_pts = Params::exhaustivesearch::nb_pts;
      }

      template <typename AcquisitionFunction>
      Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim) const {
        return explore(0, acqui, Eigen::VectorXd::Constant(dim, 0));
      }

     private:
      // recursive exploration
      template <typename AcquisitionFunction>
      Eigen::VectorXd explore(int dim,
                              const AcquisitionFunction& acqui,
                              const Eigen::VectorXd& current,
                              Eigen::VectorXd& result) const {
        double best_fit = -std::numeric_limits<double>::max();

        Eigen::VectorXd current_result(result.size());
        for (double x = 0; x <= 1; x += 1 / (double)nb_pts) {
          Eigen::VectorXd point = current;
          point[dim] = x;
          double val;
          if (dim == current.size() - 1) {
            val = acqui(point);
            if (val > best_fit) {
              best_fit = val;
              current_result = point;
            }
          } else {
            Eigen::VectorXd temp_result = current_result;
            std::tie(temp_result, val) = explore(dim + 1, acqui, point, temp_result);
            if (val > best_fit) {
              best_fit = val;
              current_result = temp_result;
            }
          }
        }
        return current_result;
      }
      int nb_pts;
    };


  }
}
#endif
