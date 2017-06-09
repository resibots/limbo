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
#ifndef LIMBO_MODEL_GP_KERNEL_LOO_OPT_HPP
#define LIMBO_MODEL_GP_KERNEL_LOO_OPT_HPP

#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the kernel only
            template <typename Params, typename Optimizer = opt::ParallelRepeater<Params, opt::Rprop<Params>>>
            struct KernelLooOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    KernelLooOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    Eigen::VectorXd params = optimizer(optimization, gp.kernel_function().h_params(), false);
                    gp.kernel_function().set_h_params(params);
                    gp.recompute(false);
                }

            protected:
                template <typename GP>
                struct KernelLooOptimization {
                public:
                    KernelLooOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params);

                        gp.recompute(false);

                        size_t n = gp.obs_mean().rows();

                        Eigen::MatrixXd l = gp.matrixL();

                        // K^{-1} using Cholesky decomposition
                        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);

                        l.template triangularView<Eigen::Lower>().solveInPlace(w);
                        l.template triangularView<Eigen::Lower>().transpose().solveInPlace(w);

                        // alpha
                        Eigen::MatrixXd alpha = gp.alpha();
                        Eigen::VectorXd inv_diag = w.diagonal().array().inverse();

                        double loo = (((-0.5 * (alpha.array().square().array().colwise() * inv_diag.array())).array().colwise() - 0.5 * inv_diag.array().log().array()) - 0.5 * std::log(2 * M_PI)).colwise().sum().sum(); //LOO.rowwise().sum();

                        // double loo_total = 0.0;
                        // for (int K = 0; K < gp.obs_mean().cols(); K++) {
                        //     double loo_tmp = (-0.5 * alpha.col(K).array().square() / w.diagonal().array() - 0.5 * w.diagonal().array().inverse().log()).sum() - 0.5 * n * std::log(2 * M_PI);
                        //     loo_total += loo_tmp;
                        // }
                        //
                        // std::cout << loo << " vs " << loo_total << std::endl;

                        if (!compute_grad)
                            return opt::no_grad(loo);

                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
                        Eigen::MatrixXd grads = Eigen::MatrixXd::Zero(params.size(), gp.obs_mean().cols());
                        // only compute half of the matrix (symmetrical matrix)
                        // TO-DO: Make it better
                        std::vector<std::vector<Eigen::VectorXd>> full_dk;
                        for (size_t i = 0; i < n; i++) {
                            full_dk.push_back(std::vector<Eigen::VectorXd>());
                            for (size_t j = 0; j <= i; j++)
                                full_dk[i].push_back(gp.kernel_function().grad(gp.samples()[i], gp.samples()[j], i, j));
                            for (size_t j = i + 1; j < n; j++)
                                full_dk[i].push_back(Eigen::VectorXd::Zero(params.size()));
                        }

                        for (size_t i = 0; i < n; i++)
                            for (size_t j = 0; j < i; ++j)
                                full_dk[j][i] = full_dk[i][j];

                        for (int j = 0; j < grad.size(); j++) {
                            Eigen::MatrixXd dKdTheta_j = Eigen::MatrixXd::Zero(n, n);
                            for (size_t i = 0; i < n; i++) {
                                for (size_t k = 0; k < n; k++)
                                    dKdTheta_j(i, k) = full_dk[i][k](j);
                            }
                            Eigen::MatrixXd Zeta_j = w * dKdTheta_j;
                            Eigen::MatrixXd Zeta_j_alpha = Zeta_j * alpha;
                            Eigen::MatrixXd Zeta_j_K = Zeta_j * w;
                            for (size_t i = 0; i < n; i++)
                                grads.row(j).array() += (alpha.row(i).array() * Zeta_j_alpha.row(i).array() - 0.5 * (1. + alpha.row(i).array().square() / w.diagonal()(i)) * Zeta_j_K.diagonal()(i)) / w.diagonal()(i);
                        }

                        grad = grads.rowwise().sum();

                        return {loo, grad};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        }
    }
}

#endif
