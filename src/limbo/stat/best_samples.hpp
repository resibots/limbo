#ifndef LIMBO_STAT_BEST_SAMPLES_HPP
#define LIMBO_STAT_BEST_SAMPLES_HPP

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct BestSamples : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled() || bo.samples().empty())
                    return;

                this->_create_log_file(bo, "best_samples.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration best_sample" << std::endl;

                if (!blacklisted)
                    (*this->_log_file) << bo.total_iterations() << " " << bo.best_sample(afun).transpose() << std::endl;
            }
        };
    }
}

#endif
