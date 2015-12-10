#ifndef LIMBO_STAT_GP_ACQUISITIONS_HPP
#define LIMBO_STAT_GP_ACQUISITIONS_HPP

#include <cmath>

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct GPAcquisitions : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled())
                    return;

                this->_create_log_file(bo, "gp_acquisitions.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration mu sigma acquisition" << std::endl;

                Eigen::VectorXd mu;
                double sigma, acqui;

                if (!blacklisted && !bo.samples().empty()) {
                    std::tie(mu, sigma) = bo.model().query(bo.samples().back());
                    acqui = typename BO::acquisition_function_t(bo.model(), bo.current_iteration())(bo.samples().back(), afun);
                }
                else if (!bo.bl_samples().empty()) {
                    std::tie(mu, sigma) = bo.model().query(bo.bl_samples().back());
                    acqui = typename BO::acquisition_function_t(bo.model(), bo.current_iteration())(bo.bl_samples().back(), afun);
                }
                else
                    return;

                (*this->_log_file) << bo.total_iterations() << " " << afun(mu) << " " << sigma << " " << acqui << std::endl;
            }
        };
    }
}

#endif
