#ifndef MEAN_FUNCTIONS_CONSTANT_HPP_
#define MEAN_FUNCTIONS_CONSTANT_HPP_

namespace limbo {
    namespace mean_fun {
        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct Constant {
            Constant(size_t dim_out = 1) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Params::meanconstant::constant();
            }
        };
    }
}
#endif
