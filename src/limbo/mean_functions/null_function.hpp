#ifndef MEAN_FUNCTIONS_NULL_FUNCTION_HPP_
#define MEAN_FUNCTIONS_NULL_FUNCTION_HPP_

namespace limbo {
    namespace mean_functions {
        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct NullFunction {
            NullFunction(size_t dim_out = 1) : _dim_out(dim_out) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return ObsType::Zero(_dim_out);
            }

        protected:
            size_t _dim_out;
        };
    }
}
#endif
