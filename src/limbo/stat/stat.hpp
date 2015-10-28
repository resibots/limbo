#ifndef LIMBO_STATISTICS_STAT_HPP_
#define LIMBO_STATISTICS_STAT_HPP_

#include <fstream>
#include <string>
#include <boost/shared_ptr.hpp>

namespace limbo {
    namespace stat {
        template <typename Params>
        struct Stat {
            Stat() {}

            template <typename BO>
            void operator()(const BO& bo, bool blacklisted)
            {
                assert(false);
            }

        protected:
            boost::shared_ptr<std::ofstream> _log_file;

            template <typename BO>
            void _create_log_file(const BO& bo, const std::string& name)
            {
                if (!_log_file && bo.dump_enabled()) {
                    std::string log = bo.res_dir() + "/" + name;
                    _log_file = boost::shared_ptr<std::ofstream>(new std::ofstream(log.c_str()));
                }
            }
        };
    }
}

#endif
