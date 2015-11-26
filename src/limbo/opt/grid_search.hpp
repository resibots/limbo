#ifndef LIMBO_OPT_GRID_SEARCH_HPP
#define LIMBO_OPT_GRID_SEARCH_HPP

#include <limits>

#include <Eigen/Core>

namespace limbo {
    namespace opt {
        template <typename Params>
        struct GridSearch {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, bool bounded) const
            {
                // Grid search does not support unbounded search
                assert(bounded);
                return _inner_search(f, 0, f.init());
            }

        protected:
            template <typename F>
            Eigen::VectorXd _inner_search(const F& f, int depth, const Eigen::VectorXd& current) const
            {
                double step_size = 1.0 / (double)Params::grid_search::nb_pts();
                double upper_lim = 1.0 + step_size;

                double best_fit = -std::numeric_limits<double>::max();

                Eigen::VectorXd current_result(f.param_size());
                for (double x = 0; x < upper_lim; x += step_size) {
                    Eigen::VectorXd new_point = current;
                    new_point[depth] = x;
                    double val;
                    if (depth == f.param_size() - 1) {
                        val = f.utility(new_point);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = new_point;
                        }
                    }
                    else {
                        Eigen::VectorXd temp_result = _inner_search(f, depth + 1, new_point);
                        val = f.utility(temp_result);
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
