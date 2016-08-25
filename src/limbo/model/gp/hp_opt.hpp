#ifndef LIMBO_MODEL_GP_HP_OPT_HPP
#define LIMBO_MODEL_GP_HP_OPT_HPP

#include <Eigen/Core>

#include <limbo/opt/parallel_repeater.hpp>
#include <limbo/opt/rprop.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = opt::ParallelRepeater<Params, opt::Rprop<Params>>>
            struct HPOpt {
            public:
                HPOpt() : _called(false) {}
                ~HPOpt()
                {
                    if (!_called) {
                        std::cerr << "'HPOpt' was never called!" << std::endl;
                        assert(false);
                    }
                }

            protected:
                bool _called;
            };
        }
    }
}

#endif
