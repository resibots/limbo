#ifndef MEAN_FUNCTIONS_MEAN_DATA_HPP_
#define MEAN_FUNCTIONS_MEAN_DATA_HPP_

namespace limbo {
    namespace mean_functions {
        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct MeanData {
            MeanData(size_t dim_out = 1) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP& gp) const
            {
                return gp.mean_observation().array();
            }
        };
    }
}
#endif
