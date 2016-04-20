#ifndef LIMBO_STAT_BEST_OBSERVATIONS_HPP
#define LIMBO_STAT_BEST_OBSERVATIONS_HPP

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        ///@ingroup stat
        /// Write the best observation so far
        ///filename: `best_observations.dat"`
        template <typename Params>
        struct BestObservations : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "best_observations.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration best_observation" << std::endl;

                if (!blacklisted)
                    (*this->_log_file) << bo.total_iterations() << " " << bo.best_observation(afun).transpose() << std::endl;
            }
        };
    }
}

#endif
