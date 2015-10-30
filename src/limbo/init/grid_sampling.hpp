#ifndef LIMBO_INIT_GRID_SAMPLING_HPP
#define LIMBO_INIT_GRID_SAMPLING_HPP

#include <Eigen/Core>

namespace limbo {
    namespace init {
        // params:
        //  -init::nb_bins
        template <typename Params>
        struct GridSampling {
            template <typename F, typename Opt>
            void operator()(const F& feval, Opt& opt) const
            {
                _explore(0, feval, Eigen::VectorXd::Constant(F::dim_in, 0), opt);
            }

        private:
            // recursively explore all the dim_inensions
            template <typename F, typename Opt>
            void _explore(int dim_in, const F& feval, const Eigen::VectorXd& current,
                Opt& opt) const
            {
                for (double x = 0; x <= 1.0f; x += 1.0f / (double)Params::init::nb_bins()) {
                    Eigen::VectorXd point = current;
                    point[dim_in] = x;
                    if (dim_in == current.size() - 1) {
                        opt.add_new_sample(point, feval(point));
                    }
                    else {
                        _explore(dim_in + 1, feval, point, opt);
                    }
                }
            }
        };
    }
}
#endif
