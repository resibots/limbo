#ifndef LIMBO_OPT_IMPL_INNER_RANDOM_SEARCH_HPP
#define LIMBO_OPT_IMPL_INNER_RANDOM_SEARCH_HPP

#include <limbo/opt/impl/inner_struct.hpp>

namespace limbo {
	namespace opt {
		namespace impl {
			template <typename Params>
		    struct InnerRandom {
		        InnerRandom() {}

		        template <typename AcquisitionFunction, typename AggregatorFunction>
	            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, const AggregatorFunction& afun) const
	            {
	                return (Eigen::VectorXd::Random(acqui.dim_in()).array() + 1) / 2;
	            }
		    };
		}
	}
}


#endif