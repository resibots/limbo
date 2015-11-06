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

              opt.set_min_objective(this->nlopt_func<F>, (void*)&f);

              std::vector<double> x(f.init().size());
              for(int i=0;i<f.init().size();i++)
                x[i] = f.init()(i);

              opt.set_ftol_rel(1e-4);

              double min;

              opt.optimize(x, min);

              return Eigen::VectorXd::Map(x.data(), x.size());
            }

          protected:
            template<typename F>
            static double nlopt_func(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data)
            {
              F* f = (F*)(my_func_data);
              Eigen::VectorXd params = Eigen::VectorXd::Map(x.data(), x.size());
              double v;
              if(!grad.empty())
              {
                Eigen::VectorXd g;
                auto p = f->utility_and_grad(params);
                v = std::get<0>(p);
                g = std::get<1>(p);
                for(int i=0;i<g.size();i++)
                  grad[i] = g(i);
              }
              else
              {
                v = f->utility(params);
              }
              return v;
            }
        };
    }
}

#endif
