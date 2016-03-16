#ifndef LIMBO_INIT_GRID_SAMPLING_HPP
#define LIMBO_INIT_GRID_SAMPLING_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct init_gridsampling {
            BO_PARAM(int, bins, 5);
        };
    }
    namespace init {
        template <typename Params>
        struct GridSampling {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                _explore(0, seval, Eigen::VectorXd::Constant(StateFunction::dim_in, 0), opt);
            }

        private:
            // recursively explore all the dimensions
            template <typename StateFunction, typename Opt>
            void _explore(int dim_in, const StateFunction& seval, const Eigen::VectorXd& current,
                Opt& opt) const
            {
                for (double x = 0; x <= 1.0f; x += 1.0f / (double)Params::init_gridsampling::bins()) {
                    Eigen::VectorXd point = current;
                    point[dim_in] = x;
                    if (dim_in == current.size() - 1) {
                        opt.eval_and_add(seval, point);
                    }
                    else {
                        _explore(dim_in + 1, seval, point, opt);
                    }
                }
            }
        };
    }
}
#endif
