//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
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
#ifndef LIMBO_MODEL_GP_KERNEL_LF_OPT_HPP
#define LIMBO_MODEL_GP_KERNEL_LF_OPT_HPP

#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = opt::ParallelRepeater<Params, opt::Rprop<Params>>>
            struct KernelLFOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    KernelLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    auto params = optimizer(optimization, (gp.kernel_function().h_params().array() + 6.0) / 7.0, true);
                    gp.kernel_function().set_h_params(-6.0 + params.array() * 7.0);
                    gp.set_lik(opt::eval(optimization, params));
                    gp.recompute(false);
                }

            protected:
                template <typename GP>
                struct KernelLFOptimization {
                public:
                    KernelLFOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(-6.0 + params.array() * 7.0);

                        gp.recompute(false);

                        size_t n = gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = gp.matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        double a = (gp.obs_mean().transpose() * gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                        if (!compute_grad)
                            return opt::no_grad(lik);

                        // K^{-1} using Cholesky decomposition
                        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);

                        gp.matrixL().template triangularView<Eigen::Lower>().solveInPlace(w);
                        gp.matrixL().template triangularView<Eigen::Lower>().transpose().solveInPlace(w);

                        // alpha * alpha.transpose() - K^{-1}
                        w = gp.alpha() * gp.alpha().transpose() - w;

                        // only compute half of the matrix (symmetrical matrix)
                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
                        for (size_t i = 0; i < n; ++i) {
                            for (size_t j = 0; j <= i; ++j) {
                                Eigen::VectorXd g = gp.kernel_function().grad(gp.samples()[i], gp.samples()[j]);
                                if (i == j)
                                    grad += w(i, j) * g * 0.5;
                                else
                                    grad += w(i, j) * g;
                            }
                        }

                        return {lik, grad};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        }
    }
}

#endif
