#ifndef LIMBO_OPT_IMPL_INNER_EXHAUSTIVE_SEARCH_HPP
#define LIMBO_OPT_IMPL_INNER_EXHAUSTIVE_SEARCH_HPP

#include <limbo/opt/exhaustive_search.hpp>
#include <limbo/opt/impl/inner_struct.hpp>

namespace limbo {
	namespace opt {
		namespace impl {
			template <typename Params>
		    struct InnerExhaustiveSearch {
		        InnerExhaustiveSearch() {}

		        template <typename AcquisitionFunction, typename AggregatorFunction>
	            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, const AggregatorFunction& afun) const
	            {
	                InnerOptimization<Params, AcquisitionFunction, AggregatorFunction> util(acqui, afun, Eigen::VectorXd::Zero(acqui.dim_in()));
	                return opt::exhaustive_search<Params>(util);
	            }
		    };
		}
	}
}


#endif