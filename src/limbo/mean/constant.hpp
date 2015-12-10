#ifndef LIMBO_MEAN_CONSTANT_HPP
#define LIMBO_MEAN_CONSTANT_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct mean_constant {
            BO_PARAM(double, constant, 1);
        };
    }
    namespace mean {
        template <typename Params>
        struct Constant {
            Constant(size_t dim_out = 1) : _dim_out(dim_out) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Eigen::VectorXd::Constant(_dim_out, Params::mean_constant::constant());
            }

        protected:
            size_t _dim_out;
        };
    }
}

#endif
