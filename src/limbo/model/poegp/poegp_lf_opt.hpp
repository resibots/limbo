//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_MODEL_POEGP_POEGP_KERNEL_LF_OPT_HPP
#define LIMBO_MODEL_POEGP_POEGP_KERNEL_LF_OPT_HPP

#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace model {
        namespace poegp {
            ///@ingroup model_opt
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = opt::Rprop<Params>>
            struct POEKernelLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    POEKernelLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.h_params(), false);
                    gp.set_h_params(params);
                    gp.recompute(false);
                }

            protected:
                template <typename GP>
                struct POEKernelLFOptimization {
                public:
                    POEKernelLFOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp_all(this->_original_gp);
                        gp_all.set_h_params(params);
                        gp_all.recompute(false);

                        double lik_all = 0.0;
                        Eigen::VectorXd grad_all = Eigen::VectorXd::Zero(params.size());

                        auto gps = gp_all.get_gps();

                        for (auto gp : gps) {
                            lik_all += gp.compute_log_lik();

                            if (!compute_grad)
                                continue;

                            Eigen::VectorXd grad = gp.compute_kernel_grad_log_lik();
                            grad_all.array() += grad.array();
                        }

                        if (!compute_grad)
                            return opt::no_grad(lik_all);

                        return {lik_all, grad_all};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        } // namespace poegp
    } // namespace model
} // namespace limbo

#endif