#ifndef MEAN_FUNCTIONS_DATA_HPP_
#define MEAN_FUNCTIONS_DATA_HPP_

namespace limbo {
    namespace mean {
        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct Data {
            Data(size_t dim_out = 1) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP& gp) const
            {
                return gp.mean_observation().array();
            }
        };
    }
}
#endif
