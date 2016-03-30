#ifndef LIMBO_STAT_STAT_BASE_HPP
#define LIMBO_STAT_STAT_BASE_HPP

#include <fstream>
#include <string>

#include <memory>

namespace limbo {
    namespace stat {
        /**
          Base class for statistics

          The only method provided is protected :

          \rst
          .. code-block:: cpp
          
            template <typename BO>
            void _create_log_file(const BO& bo, const std::string& name)

          \endrst
        */
        template <typename Params>
        struct StatBase {
            StatBase() {}

            /// main method (to be written in derived classes)
            template <typename BO>
            void operator()(const BO& bo, bool blacklisted)
            {
                assert(false);
            }

        protected:
            std::shared_ptr<std::ofstream> _log_file;

            template <typename BO>
            void _create_log_file(const BO& bo, const std::string& name)
            {
                if (!_log_file && bo.stats_enabled()) {
                    std::string log = bo.res_dir() + "/" + name;
                    _log_file = std::make_shared<std::ofstream>(log.c_str());
                }
            }
        };
    }
}

#endif
