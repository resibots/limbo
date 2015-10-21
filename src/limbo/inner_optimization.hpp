#ifndef INNER_OPTIMIZATION_HPP_
#define INNER_OPTIMIZATION_HPP_

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <limits>

namespace limbo {
    namespace inner_optimization {
        template <typename Params>
        struct Random {
            Random() {}

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const AggregatorFunction&) const
            {
                return Eigen::VectorXd::Random(dim_in);
            }
        };

        template <typename Params>
        struct ExhaustiveSearch {
            ExhaustiveSearch() 
            {
                _step_size = 1.0 / (double)Params::exhaustive_search::nb_pts();
                _upper_lim = 1.0 + _step_size;
            }

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const AggregatorFunction& afun) const
            {
                Eigen::VectorXd result(dim_in);
                _explore(acqui, afun, 0, Eigen::VectorXd::Zero(dim_in), result);
                return result;
            }

        private:
            double _step_size;
            double _upper_lim;

            // recursive exploration
            template <typename AcquisitionFunction, typename AggregatorFunction>
            double _explore(const AcquisitionFunction& acqui, const AggregatorFunction& afun, int curr_dim,
                const Eigen::VectorXd& current_point,
                Eigen::VectorXd& result) const
            {
                double best_fit = -std::numeric_limits<double>::max();

                Eigen::VectorXd current_result(result.size());
                for (double x = 0; x < _upper_lim; x += _step_size) {
                    Eigen::VectorXd new_point = current_point;
                    new_point[curr_dim] = x;
                    double val;
                    if (curr_dim == current_point.size() - 1) {
                        val = acqui(new_point, afun);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = new_point;
                        }
                    }
                    else {
                        Eigen::VectorXd temp_result = current_result;
                        val = _explore(acqui, afun, curr_dim + 1, new_point, temp_result);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = temp_result;
                        }
                    }
                }
                result = current_result;
                return best_fit;
            }
        };
    }
}
#endif
