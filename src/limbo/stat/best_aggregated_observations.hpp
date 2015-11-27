#ifndef LIMBO_STAT_BEST_AGGREGATED_OBSERVATIONS_HPP
#define LIMBO_STAT_BEST_AGGREGATED_OBSERVATIONS_HPP

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct BestAggregatedObservations : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "best_aggregated_observations.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration best_aggregated_observation" << std::endl;

                if (!blacklisted)
                    (*this->_log_file) << bo.total_iterations() << " " << afun(bo.best_observation(afun)) << std::endl;
            }
        };
    }
}

#endif
