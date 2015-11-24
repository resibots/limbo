#ifndef LIMBO_STAT_ACQUISITIONS_HPP
#define LIMBO_STAT_ACQUISITIONS_HPP

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct Acquisitions : public StatBase<Params> {
            Acquisitions() {}

            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction&, bool blacklisted)
            {
                this->_create_log_file(bo, "acquisitions.dat");
                if (bo.dump_enabled() && !blacklisted)
                    (*this->_log_file)
                        << bo.iteration()
                        << " new point: " << bo.samples()[bo.samples().size() - 1].transpose()
                        << " value: "
                        << bo.observations()[bo.observations().size() - 1].transpose()
                        << std::endl;
            }
        };
    }
}

#endif
