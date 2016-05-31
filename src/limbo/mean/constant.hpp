#ifndef LIMBO_MEAN_CONSTANT_HPP
#define LIMBO_MEAN_CONSTANT_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct mean_constant {
            ///@ingroup mean_defaults
            BO_PARAM(double, constant, 1);
        };
    }
    namespace mean {
        /** @ingroup mean
          A constant mean (the traditionnal choice for Bayesian optimization)

          Parameter:
            - ``double constant`` (the value of the constant)
        */
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
