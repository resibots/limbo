#ifndef LIMBO_OPT_NLOPT_HPP
#define LIMBO_OPT_NLOPT_HPP

#include <Eigen/Core>

#include <vector>

#include <nlopt.hpp>

namespace limbo {
    namespace opt {
        template <typename Params, nlopt::algorithm Algorithm = nlopt::LD_MMA>
        struct NLOpt {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f) const
            {
              nlopt::opt opt(Algorithm, f.param_size());

              opt.set_min_objective(this->nlopt_func, NULL);

              std::vector<double> x(f.init().size());
              for(int i=0;i<f.init().size();i++)
                x[i] = f.init()(i);

              opt.set_ftol_rel(1e-4);

              double min;

              opt.optimize(x, min);

              return Eigen::VectorXd::Map(x.data(), x.size());
            }

          protected:
            static double nlopt_func(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
            {
              Eigen::VectorXd params = Eigen::VectorXd::Map(x.data(), x.size());
              double v = x[0]*x[0]+x[1]*x[1];
              if(!grad.empty())
              {
                Eigen::VectorXd g;
                grad[0] = 2*x[0];
                grad[1] = 2*x[1];
              }
              return v;
            }
        };
    }
}

#endif
