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
#ifndef LIMBO_MODEL_POEGP_HPP
#define LIMBO_MODEL_POEGP_HPP

#include <limbo/model.hpp>
#include <limbo/spt/spt.hpp>

namespace limbo {

    namespace defaults {
        struct model_poegp {
            BO_PARAM(int, leaf_size, 100);
            BO_PARAM(double, tau, 0.05);
        };
    } // namespace defaults

    namespace model {
        /// @ingroup model
        /// Gaussian process with product of experts.
        /// A combination of:
        /// - Deisenroth, M.P. and Ng, J.W., 2015. Distributed gaussian processes. arXiv preprint arXiv:1502.02843.
        /// - McFee, B. and Lanckriet, G.R., 2011. Large-scale music similarity search with spatial trees. In ISMIR (pp. 55-60).
        /// for improving the learning and querying performance of GPs
        /// Parameters:
        /// - leaf_size: Max number of samples per leaf of the Spatial Tree
        /// - tau: Percentage of overlap between neighboring (in space) leafs

        template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
        class POEGP {
        public:
            using GP_t = limbo::model::GP<Params, KernelFunction, MeanFunction, limbo::model::gp::NoLFOpt<Params>>;

            POEGP() {}

            POEGP(int dim_in, int dim_out)
            {
                _gps.resize(1);
                _gps[0] = GP_t(dim_in, dim_out);
            }

            void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = true)
            {
                int n = Params::model_poegp::leaf_size();
                int N = samples.size();
                // int leaves = N / n;
                int d = std::ceil(std::log(N / double(n)) / std::log(2.0));

                // std::vector<int> ids(samples.size());
                // for (int i = 0; i < ids.size(); i++)
                //     ids[i] = i;
                // std::random_shuffle(ids.begin(), ids.end());
                //
                // _gps.resize(leaves);
                //
                // limbo::tools::par::loop(0, leaves, [&](size_t i) {
                //     std::vector<Eigen::VectorXd> s, o;
                //     for (int j = 0; j < n; j++) {
                //         s.push_back(samples[ids[i * n + j]]);
                //         o.push_back(observations[ids[i * n + j]]);
                //     }
                //
                //     _gps[i].compute(s, o, false);
                //     // _gps[i].optimize_hyperparams();
                // });
                _samples = samples;

                auto root = spt::make_spt(samples, observations, d, Params::model_poegp::tau());

                auto leaves = spt::get_leaves(root);

                _root = root;
                _leaves = leaves;

                KernelFunction kernel_func(_samples[0].size());
                MeanFunction mean_func(observations[0].size());

                if (_gps.size() > 0) {
                    kernel_func = _gps[0].kernel_function();
                    mean_func = _gps[0].mean_function();
                }

                _gps.resize(leaves.size());
                _gps[0].kernel_function() = kernel_func;
                _gps[0].mean_function() = mean_func;
                _update_kernel_and_mean_functions();

                limbo::tools::par::loop(0, leaves.size(), [&](size_t i) {
                    _gps[i].compute(leaves[i]->points(), leaves[i]->observations(), compute_kernel);
                });

                if (!_h_params.size())
                    _h_params = _gps[0].kernel_function().h_params();
            }

            void set_h_params(const Eigen::VectorXd& params)
            {
                assert(_gps.size());
                // limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                //     _gps[i].kernel_function().set_h_params(params);
                // });
                _gps[0].kernel_function().set_h_params(params);

                _h_params = params;
            }

            Eigen::VectorXd h_params() const
            {
                return _h_params;
            }

            const KernelFunction& kernel_function() const
            {
                assert(_gps.size());
                return _gps[0].kernel_function();
            }

            KernelFunction& kernel_function()
            {
                assert(_gps.size());
                return _gps[0].kernel_function();
            }

            const MeanFunction& mean_function() const
            {
                assert(_gps.size());
                return _gps[0].mean_function();
            }

            MeanFunction& mean_function()
            {
                assert(_gps.size());
                return _gps[0].mean_function();
            }

            ///  recomputes the GP
            void recompute(bool update_obs_mean = true, bool update_full_kernel = true)
            {
                _update_kernel_and_mean_functions();

                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    _gps[i].recompute(update_obs_mean, update_full_kernel);
                });
            }

            std::vector<GP_t> get_gps()
            {
                return _gps;
            }

            /// Do not forget to call this if you use hyper-prameters optimization!!
            void optimize_hyperparams()
            {
                _hp_optimize(*this);
            }

            std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
            {
                // std::vector<int> ids = _find_gps(_root, v);
                // assert(ids.size());
                //
                // if (ids.size() == 1)
                //     return _gps[ids[0]].query(v);

                std::vector<double> mus(_gps.size());
                std::vector<double> sigmas(_gps.size());
                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    Eigen::VectorXd tmu;
                    double ts;
                    std::tie(tmu, ts) = _gps[i].query(v); //_gps[ids[i]].query(v);

                    mus[i] = tmu(0);
                    sigmas[i] = 1.0 / (ts + 1e-12);
                });

                // double multi_mu = std::accumulate(mus.begin(), mus.end(), 1, std::multiplies<double>());
                double multi_sg = std::accumulate(sigmas.begin(), sigmas.end(), 0.0, std::plus<double>());
                double multi_mu = 0.0;
                for (size_t i = 0; i < _gps.size(); i++) {
                    multi_mu += mus[i] * sigmas[i];
                }

                Eigen::VectorXd mu(1);
                mu << (multi_mu / multi_sg);

                return std::make_tuple(mu, 1.0 / multi_sg);
            }

            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                // std::vector<int> ids = _find_gps(_root, v);
                // assert(ids.size());
                //
                // if (ids.size() == 1)
                //     return _gps[ids[0]].mu(v);

                std::vector<double> mus(_gps.size());
                std::vector<double> sigmas(_gps.size());
                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    Eigen::VectorXd tmu;
                    double ts;
                    std::tie(tmu, ts) = _gps[i].query(v); //_gps[ids[i]].query(v);

                    mus[i] = tmu(0);
                    sigmas[i] = 1.0 / (ts + 1e-12);
                });

                double multi_sg = std::accumulate(sigmas.begin(), sigmas.end(), 0.0, std::plus<double>());
                double multi_mu = 0.0;
                for (size_t i = 0; i < _gps.size(); i++) {
                    multi_mu += mus[i] * sigmas[i];
                }

                Eigen::VectorXd mu(1);
                mu << (multi_mu / multi_sg);

                return mu;
            }

            double sigma(const Eigen::VectorXd& v) const
            {
                // std::vector<int> ids = _find_gps(_root, v);
                // assert(ids.size());
                //
                // if (ids.size() == 1)
                //     return _gps[ids[0]].sigma(v);

                std::vector<double> mus(_gps.size());
                std::vector<double> sigmas(_gps.size());
                // for (int i = 0; i < _gps.size(); i++)
                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    Eigen::VectorXd tmu;
                    double ts;
                    std::tie(tmu, ts) = _gps[i].query(v); //_gps[ids[i]].query(v);

                    mus[i] = tmu(0);
                    sigmas[i] = 1.0 / (ts + 1e-12);
                });

                double multi_sg = std::accumulate(sigmas.begin(), sigmas.end(), 0.0, std::plus<double>());

                return multi_sg;
            }

            double compute_log_lik() const
            {
                double lik_all = 0.0;
                for (auto gp : _gps) {
                    lik_all += gp.compute_log_lik();
                }

                return lik_all;
            }

            /// return the list of samples that have been tested so far
            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

        protected:
            std::vector<GP_t> _gps;
            HyperParamsOptimizer _hp_optimize;
            Eigen::VectorXd _h_params;

            std::shared_ptr<spt::SpatialTreeNode> _root;
            std::vector<std::shared_ptr<spt::SpatialTreeNode>> _leaves;
            std::vector<Eigen::VectorXd> _samples;

            void _update_kernel_and_mean_functions()
            {
                assert(_gps.size());

                limbo::tools::par::loop(1, _gps.size(), [&](size_t i) {
                    _gps[i].kernel_function() = _gps[0].kernel_function();
                    _gps[i].mean_function() = _gps[0].mean_function();
                });
            }

            std::vector<int> _find_gps(const std::shared_ptr<spt::SpatialTreeNode>& node, const Eigen::VectorXd& v) const
            {
                std::vector<int> ids;
                if (!node->right() || !node->left()) {
                    int i = 0;
                    for (; i < _leaves.size(); i++)
                        if (node == _leaves[i])
                            break;
                    ids.push_back(i);
                }
                else {
                    std::vector<int> left_ids, right_ids;

                    Eigen::VectorXd split_dir = node->split_dir();
                    double val = split_dir.dot(v);
                    if (val <= node->split_median_left()) {
                        left_ids = _find_gps(node->left(), v);
                    }

                    if (val > node->split_median_right()) {
                        right_ids = _find_gps(node->right(), v);
                    }

                    for (int i = 0; i < left_ids.size(); i++)
                        ids.push_back(left_ids[i]);
                    for (int i = 0; i < right_ids.size(); i++)
                        ids.push_back(right_ids[i]);
                }

                return ids;
            }
        };
    } // namespace model
} // namespace limbo

#endif