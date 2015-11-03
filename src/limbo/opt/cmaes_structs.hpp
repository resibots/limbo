#ifndef LIMBO_OPT_CMAES_STRUCTS_HPP
#define LIMBO_OPT_CMAES_STRUCTS_HPP

#include <vector>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h> /* free() */
#include <stddef.h> /* NULL */

#include <Eigen/Core>

#include <limbo/opt/cmaes.hpp>

namespace limbo {
    namespace opt {

        template <typename Params, typename AcquisitionFunction, typename AggregatorFunction>
        struct CmaesUtility
        {
        public:
            CmaesUtility(const AcquisitionFunction& acqui, const AggregatorFunction& afun, const Eigen::VectorXd& init)
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

        template <typename Params>
        struct Cmaes {
            Cmaes() {}

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const AggregatorFunction& afun) const
            {
                return this->operator()(acqui, dim_in,
                    Eigen::VectorXd::Constant(dim_in, 0.5), afun);
            }

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const Eigen::VectorXd& init, const AggregatorFunction& afun) const
            {
                CmaesUtility<Params, AcquisitionFunction, AggregatorFunction> util(acqui, afun, init);
                return opt::cmaes<Params>(util);
            }
        };
    }
}

#endif
