#ifndef LIMBO_OPT_IMPL_INNER_CMAES_HPP
#define LIMBO_OPT_IMPL_INNER_CMAES_HPP

#include <limbo/opt/cmaes.hpp>
#include <limbo/opt/impl/inner_struct.hpp>

namespace limbo {
	namespace opt {
		namespace impl {
			template <typename Params>
		    struct InnerCmaes {
		        InnerCmaes() {}

		        template <typename AcquisitionFunction, typename AggregatorFunction>
		        Eigen::VectorXd operator()(const AcquisitionFunction& acqui, const AggregatorFunction& afun) const
		        {
		            return this->operator()(acqui, afun, Eigen::VectorXd::Constant(acqui.dim_in(), 0.5));
		        }

		        template <typename AcquisitionFunction, typename AggregatorFunction>
		        Eigen::VectorXd operator()(const AcquisitionFunction& acqui, const AggregatorFunction& afun, const Eigen::VectorXd& init) const
		        {
		            InnerOptimization<Params, AcquisitionFunction, AggregatorFunction> util(acqui, afun, init);
		            return opt::cmaes<Params>(util);
		        }
		    };
		}
	}
}

#endif