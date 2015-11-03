#ifndef LIMBO_OPT_INNER_STRUCTS_HPP
#define LIMBO_OPT_INNER_STRUCTS_HPP

#include <iostream>

#include <Eigen/Core>

namespace limbo {
    namespace opt {

        template <typename Params, typename AcquisitionFunction, typename AggregatorFunction>
        struct InnerOptimization
        {
        public:
            InnerOptimization(const AcquisitionFunction& acqui, const AggregatorFunction& afun, const Eigen::VectorXd& init)
            {
                _acqui = std::make_shared<AcquisitionFunction>(acqui);
                _afun = std::make_shared<AggregatorFunction>(afun);
                _init = init;
            }

            double utility(const Eigen::VectorXd& params)
            {
                return (*_acqui)(params, *_afun);
            }

            size_t param_size()
            {
                return _acqui->dim_in();
            }

            Eigen::VectorXd init()
            {
                return _init;
            }

        protected:
            std::shared_ptr<AcquisitionFunction> _acqui;
            std::shared_ptr<AggregatorFunction> _afun;
            Eigen::VectorXd _init;
        };
    }
}

#endif
