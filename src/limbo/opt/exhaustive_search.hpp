#ifndef LIMBO_OPT_EXHAUSTIVE_SEARCH_HPP
#define LIMBO_OPT_EXHAUSTIVE_SEARCH_HPP

#include <limits>

#include <Eigen/Core>

namespace limbo {
    namespace opt {
        template <typename Params, typename F>
        Eigen::VectorXd exhaustive_search(F& f, int depth = 0)
        {
            double step_size = 1.0 / (double)Params::exhaustive_search::nb_pts();
            double upper_lim = 1.0 + step_size;

            double best_fit = -std::numeric_limits<double>::max();

            Eigen::VectorXd current_result(f.param_size());
            for (double x = 0; x < upper_lim; x += step_size) {
                    Eigen::VectorXd new_point = f.init();
                    new_point[depth] = x;
                    double val;
                    if (depth == f.init().size() - 1) {
                        val = f.utility(new_point);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = new_point;
                        }
                    }
                    else {
                        F f_copy = f;
                        f_copy.set_init(new_point);
                        Eigen::VectorXd temp_result = exhaustive_search<Params>(f_copy, depth + 1);
                        val = f.utility(temp_result);
                        if (val > best_fit) {
                            best_fit = val;
                            current_result = temp_result;
                        }
                    }
                }
                return current_result;
        }
    }
}

#endif
