#ifndef LIMBO_STAT_GP_LIKELIHOOD_HPP
#define LIMBO_STAT_GP_LIKELIHOOD_HPP

#include <cmath>

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct GPLikelihood : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                this->_create_log_file(bo, "gp_likelihood.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration gp_likelihood" << std::endl;

                (*this->_log_file) << bo.total_iterations() << " " << bo.model().get_lik() << std::endl;
            }
        };
    }
}

#endif
