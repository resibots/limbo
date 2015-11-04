#ifndef LIMBO_OPT_IMPL_MODEL_NO_OPT_HPP
#define LIMBO_OPT_IMPL_MODEL_NO_OPT_HPP

#include <vector>

#include <Eigen/Core>

namespace limbo {
    namespace opt {
        namespace impl {
            template <typename Params>
            struct ModelNoOpt {
                template <typename Opt>
                void operator()(Opt& opt, const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, double noise,
                    const std::vector<Eigen::VectorXd>& bl_samples = std::vector<Eigen::VectorXd>())
                {
                    opt.compute(samples, observations, noise, bl_samples);
                }
            };
        }
    }
}

#endif
