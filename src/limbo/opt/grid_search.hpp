#ifndef LIMBO_OPT_GRID_SEARCH_HPP
#define LIMBO_OPT_GRID_SEARCH_HPP

#include <limits>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/opt/optimizer.hpp>

namespace limbo {
    namespace defaults {
        struct opt_gridsearch {
            BO_PARAM(int, bins, 5);
        };
    }
    namespace opt {
        template <typename Params>
        struct GridSearch {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                // Grid search does not support unbounded search
                assert(bounded);
                size_t dim = init.size();
                return _inner_search(f, 0, Eigen::VectorXd::Constant(dim, 0.5));
            }

        protected:
            template <typename F>
            Eigen::VectorXd _inner_search(const F& f, size_t depth, const Eigen::VectorXd& current) const
            {
                size_t dim = current.size();
                double step_size = 1.0 / (double)Params::opt_gridsearch::bins();
                double upper_lim = 1.0 + step_size;
                double best_fit = -std::numeric_limits<double>::max();
                Eigen::VectorXd current_result(dim);
                for (double x = 0; x < upper_lim; x += step_size) {
                    Eigen::VectorXd new_point = current;
                    new_point[depth] = x;
                    double val;
                    if (depth == dim - 1) {
                        val = eval(f, new_point);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = new_point;
                        }
                    }
                    else {
                        Eigen::VectorXd temp_result = _inner_search(f, depth + 1, new_point);
                        val = eval(f, temp_result);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = temp_result;
                        }
                    }
                }
                return current_result;
            }
        };
    }
}

#endif
