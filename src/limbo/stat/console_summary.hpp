#ifndef LIMBO_STAT_CONSOLE_SUMMARY_HPP
#define LIMBO_STAT_CONSOLE_SUMMARY_HPP

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct ConsoleSummary : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled() || bo.observations().empty())
                    return;

                std::cout << bo.total_iterations() << " new point: "
                          << (blacklisted ? bo.bl_samples().back()
                                          : bo.samples().back()).transpose();
                if (blacklisted)
                    std::cout << " value: "
                              << "No data, blacklisted";
                else
                    std::cout << " value: " << afun(bo.observations().back());

                std::cout << " best:" << afun(bo.best_observation(afun)) << std::endl;
            }
        };
    }
}

#endif
