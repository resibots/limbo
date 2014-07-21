#ifndef STOPPING_CRITERION_HPP_
#define STOPPING_CRITERION_HPP_

#include <iostream>
#include <Eigen/Core>
#include <vector>

//USING_PART_OF_NAMESPACE_EIGEN

namespace limbo {
  namespace stopping_criterion {
    template<typename Params>
    struct MaxIterations {
      MaxIterations() {
        iteration = 0;
      }
      template<typename BO>
      bool operator()(const BO& bo) {
        return bo.iteration() <= Params::maxiterations::n_iterations();
      }
     protected:
      int iteration;
    };


    template<typename Params>
    struct MaxPredictedValue {
      MaxPredictedValue() {}

      template<typename BO>
      bool operator()(const BO& bo) {
        typename BO::inneroptimization_t opti;
        Eigen::VectorXd result(bo.dim());
        double val = opti(GPMean<typename BO::meanfunction_t, typename BO::kernelfunction_t>(bo.meanfunction(), bo.kernelfunction(), bo.inverted_kernel(), bo.observations(), bo.samples()), result);

        if ( bo.observations().size() == 0 || bo.observations().maxCoeff() <= Params::maxpredictedvalue::ratio * val)
          return true;
        else {
          std::cout << "stop caused by Max predicted value reached. Thresold: " << Params::maxpredictedvalue::ratio*val << " max observations: " << bo.observations().maxCoeff() << std::endl;
          return false;
        }
      }

     protected:

      template <typename  MeanFunction, typename KernelFunction>
      struct GPMean {
        GPMean(const MeanFunction& mean_function,
               const KernelFunction& kernel_function,
               const Eigen::MatrixXd& inverted_kernel,
               const Eigen::VectorXd& observations,
               const  std::vector<Eigen::VectorXd>& samples):
          _mean_function(mean_function), _kernel_function(kernel_function), _inverted_kernel(inverted_kernel), _observations(observations), _samples(samples) {
        }


        double operator()(const Eigen::VectorXd& v)const {


          if (_samples.size() == 0)
            return (_mean_function(v));

          // compute k
          Eigen::VectorXd k(_samples.size());
          for (int i = 0; i < k.size(); i++)
            k[i] = _kernel_function(_samples[i], v);

          Eigen::VectorXd mean_vector(_samples.size());
          for (int i = 0; i < mean_vector.size(); i++)
            mean_vector[i] = _mean_function(_samples[i]);

          double mu = _mean_function(v) + (k.transpose() * _inverted_kernel * (_observations - mean_vector))[0];

          return mu;
        }


       protected:
        const MeanFunction& _mean_function;
        const KernelFunction& _kernel_function;
        const Eigen::MatrixXd& _inverted_kernel;
        const Eigen::VectorXd& _observations;
        const  std::vector<Eigen::VectorXd>& _samples;


      };



    };



    template<typename BO>
    struct ChainCriteria {
      typedef bool result_type;

      ChainCriteria(const BO& bo): _bo(bo) {}


      template<typename stopping_criterion>
      bool operator()(bool state, stopping_criterion stop)const {
        return state && stop(_bo);
      }

     protected:
      const BO& _bo;
    };




  }
}
#endif
