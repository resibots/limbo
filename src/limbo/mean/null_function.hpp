#ifndef LIMBO_MEAN_NULL_FUNCTION_HPP
#define LIMBO_MEAN_NULL_FUNCTION_HPP

#include <Eigen/Core>

namespace limbo {
    namespace mean {
        template <typename Params>
        struct NullFunction {
            NullFunction(size_t dim_out = 1) : _dim_out(dim_out) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Eigen::VectorXd::Zero(_dim_out);
            }

        protected:
            size_t _dim_out;
        };
    }
}

#endif
