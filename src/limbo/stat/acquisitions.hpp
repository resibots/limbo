#ifndef LIMBO_STAT_ACQUISITIONS_HPP
#define LIMBO_STAT_ACQUISITIONS_HPP

#include <fstream>
#include <string>
#include <limbo/stat/stat.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct Acquisitions : public Stat<Params> {
            Acquisitions() {}

            template <typename BO>
            void operator()(const BO& bo, bool blacklisted)
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
