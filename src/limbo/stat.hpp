#ifndef LIMBO_STAT_HPP_
#define LIMBO_STAT_HPP_

#include <fstream>
#include <string>
#include <boost/shared_ptr.hpp>

namespace limbo {
  namespace stat {

    template<typename Params>
    struct Stat {
      Stat() {}

      template<typename BO>
      void operator()(const BO& bo) {
        assert(false);
      }

     protected:
      boost::shared_ptr<std::ofstream> _log_file;
      template<typename BO>
      void _create_log_file(const BO& bo, const std::string& name) {
        if (!_log_file && bo.dump_enabled() ) {
          std::string log = bo.res_dir() + "/" + name;
          _log_file = boost::shared_ptr<std::ofstream>(new std::ofstream(log.c_str()));
        }
      }
    };

    template<typename Params>
    struct Acquisitions : public Stat<Params> {
      Acquisitions() {}

      template<typename BO>
      void operator()(const BO& bo) {
        this->_create_log_file(bo, "acquisitions.dat");
        if (bo.dump_enabled())
          (*this->_log_file) << bo.iteration() << " new point: "
                             << bo.samples()[bo.samples().size() - 1].transpose()
                             << " value: " << bo.observations()[bo.observations().size() - 1]
                             << std::endl;

      }
    };


  }
}

#endif
