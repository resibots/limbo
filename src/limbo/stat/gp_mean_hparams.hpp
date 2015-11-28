#ifndef LIMBO_STAT_GP_MEAN_HPARAMS_HPP
#define LIMBO_STAT_GP_MEAN_HPARAMS_HPP

#include <cmath>

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct GPMeanHParams : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "gp_mean_hparams.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration gp_mean_hparams" << std::endl;

                (*this->_log_file) << bo.total_iterations() << " " << bo.model().mean_function().h_params().transpose() << std::endl;
            }
        };
    }
}

#endif
