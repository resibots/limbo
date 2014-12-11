#ifndef INIT_FUNCTIONS_HPP_
#define INIT_FUNCTIONS_HPP_


#include <vector>
#include <iostream>
#include <Eigen/Core>


#include <limits>

namespace limbo {
  namespace init_functions {

    // params is here only to make it easy to switch
    // from/to the other init functions
    template<typename Params>
    struct NoInit {
      template<typename F, typename Opt>
      void operator()(const F& f, Opt& opt) const {}
    };

    // initialize in [0,1] !
    // params: init::nb_samples
    template<typename Params>
    struct RandomSampling {
      template<typename F, typename Opt>
      void operator()(const F& feval, Opt& opt) const {
        for (int i = 0; i < Params::init::nb_samples(); i++) {
          Eigen::VectorXd new_sample(F::dim);
          for (size_t i = 0; i < F::dim; i++)
            new_sample[i] = misc::rand<double>(0, 1);
          std::cout << "random sample:" << new_sample.transpose() << std::endl;
          opt.add_new_sample(new_sample, feval(new_sample));
        }
      }
    };


    // params:
    //  -init::nb_bins
    //  - init::nb_samples
    template<typename Params>
    struct RandomSamplingGrid {
      template<typename F, typename Opt>
      void operator()(const F& feval, Opt& opt) const {
        for (int i = 0; i < Params::init::nb_samples(); i++) {
          Eigen::VectorXd new_sample(F::dim);
          for (size_t i = 0; i < F::dim; i++)
            new_sample[i] =
              int(((double) (Params::init::nb_bins() + 1) * rand())
                  / (RAND_MAX + 1.0)) / double(Params::init::nb_bins());
          opt.add_new_sample(new_sample, feval(new_sample));
        }
      }
    };

    // params:
    //  -init::nb_bins
    template<typename Params>
    struct GridSampling {
      template<typename F, typename Opt>
      void operator()(const F& feval, Opt& opt) const {
        _explore(0, feval, Eigen::VectorXd::Constant(F::dim, 0), opt);
      }
     private:
      //recursively explore all the dimensions
      template<typename F, typename Opt>
      void _explore(int dim, const F& feval, const Eigen::VectorXd& current, Opt& opt) const {
        for (double x = 0; x <= 1.0f; x += 1.0f / (double)Params::init::nb_bins()) {
          Eigen::VectorXd point = current;
          point[dim] = x;
          if (dim == current.size() - 1) {
            opt.add_new_sample(point, feval(point));
          } else {
            _explore(dim + 1, feval, point, opt);
          }
        }
      }
    };
  }
}
#endif
