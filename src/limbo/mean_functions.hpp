#ifndef MEAN_FUNCTIONS_HPP_
#define MEAN_FUNCTIONS_HPP_

#include <fstream>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
namespace limbo {
  namespace defaults {
    struct meanconstant {
      BO_PARAM(double, constant, 0.0);
    };
  }
  namespace mean_functions {
    template<typename Params>
    struct NullFunction {
      NullFunction() {}
      template<typename GP>
      double operator()(const Eigen::VectorXd& v, const GP&)const {
        return 0;
      }
    };

    template<typename Params>
    struct MeanConstant {
      MeanConstant() {
      }
      template<typename GP>
      double operator()(const Eigen::VectorXd& v, const GP&)const {
        return  Params::meanconstant::constant();
      }
    };

    template<typename Params>
    struct MeanData {
      MeanData() {
      }
      template<typename GP>
      double operator()(const Eigen::VectorXd& v, const GP& gp)const {
        return  gp.mean_observation();
      }
    };



    template<typename Params>
    struct MeanArchive {
      MeanArchive() {
        // create and open an archive for input
        std::ifstream ifs(Params::meanarchive::filename());
        assert(ifs.good());
        boost::archive::text_iarchive ia(ifs);
        // read class state from archive
        ia >> _archive;
        std::cout << _archive.size() << " elements loaded in the archive" << std::endl;
      }

      template<typename GP>
      double operator()(const Eigen::VectorXd& v, const GP&)const {
        std::vector<double> key(v.size(), 0);
        for (int i = 0; i < v.size(); i++)
          key[i] = v[i];
        return  _archive.at(key);
      }

     protected:
      struct classcomp {
        bool operator() (const std::vector<double>& lhs, const std::vector<double>& rhs) const {
          assert(lhs.size() == 6 && rhs.size() == 6);
          int i = 0;
          while (i < 5 && lhs[i] == rhs[i])
            i++;
          return lhs[i] < rhs[i];
        }
      };

      std::map<std::vector<double>, double, classcomp> _archive;
    };



  }
}
#endif
