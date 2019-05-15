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
            BO_PARAM(int, expert_size, 100);
        };
    } // namespace defaults

    namespace model {
        /// @ingroup model
        /// Gaussian process with product of experts.
        /// - Deisenroth, M.P. and Ng, J.W., 2015. Distributed gaussian processes. arXiv preprint arXiv:1502.02843.
        ///   This implementation is the naive PoEGP: working to incorporate rBCM
        template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
        class POEGP {
        public:
            using GP_t = limbo::model::GP<Params, KernelFunction, MeanFunction, limbo::model::gp::NoLFOpt<Params>>;

            /// useful because the model might be created before knowing anything about the process
            POEGP() {}

            /// useful because the model might be created before having samples
            POEGP(int dim_in, int dim_out)
            {
                _gps.resize(1);
                _gps[0] = GP_t(dim_in, dim_out);
            }

            /// Compute the GP from samples and observations. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = true)
            {
                assert(samples.size() != 0);
                assert(observations.size() != 0);
                assert(samples.size() == observations.size());

                // TO-DO: Make this selection a template
                int n = Params::model_poegp::expert_size();
                int N = samples.size();
                int leaves = std::ceil(N / n);

                std::vector<int> ids(samples.size());
                for (size_t i = 0; i < ids.size(); i++)
                    ids[i] = i;
                std::random_shuffle(ids.begin(), ids.end());

                KernelFunction kernel_func(samples[0].size());
                MeanFunction mean_func(observations[0].size());

                if (_gps.size() > 0) {
                    kernel_func = _gps[0].kernel_function();
                    mean_func = _gps[0].mean_function();
                }

                // save samples and observations
                _samples = samples;
                _observations = observations;

                _gps.resize(leaves);
                _gps[0].kernel_function() = kernel_func;
                _gps[0].mean_function() = mean_func;

                // compute mean observation
                _mean_observation = Eigen::VectorXd::Zero(_observations[0].size());
                for (size_t j = 0; j < _observations.size(); j++)
                    _mean_observation.array() += _observations[j].array();
                _mean_observation.array() /= static_cast<double>(_observations.size());

                // Update kernel and mean functions
                _update_kernel_and_mean_functions();

                limbo::tools::par::loop(0, leaves, [&](size_t i) {
                    std::vector<Eigen::VectorXd> s, o;
                    for (int j = 0; j < n; j++) {
                        if ((i * n + j) >= ids.size()) // TO-DO: Check how to handle these cases
                            break;
                        s.push_back(samples[ids[i * n + j]]);
                        o.push_back(observations[ids[i * n + j]]);
                    }

                    _gps[i].compute(s, o, compute_kernel);
                });
            }

            /// Do not forget to call this if you use hyper-prameters optimization!!
            void optimize_hyperparams()
            {
                _hp_optimize(*this);
            }

            // TO-DO: Add add_sample function

            /// Queries the POEGP and gets the mean and variance
            std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
            {
                std::vector<double> mus(_gps.size());
                std::vector<double> sigmas(_gps.size());
                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    Eigen::VectorXd tmu;
                    double ts;
                    std::tie(tmu, ts) = _gps[i].query(v);

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

                return std::make_tuple(mu, 1.0 / multi_sg);
            }

            /// Queries the POEGP and gets the mean
            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                std::vector<double> mus(_gps.size());
                std::vector<double> sigmas(_gps.size());
                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    Eigen::VectorXd tmu;
                    double ts;
                    std::tie(tmu, ts) = _gps[i].query(v);

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

            /// Queries the POEGP and gets the variance
            double sigma(const Eigen::VectorXd& v) const
            {
                std::vector<double> mus(_gps.size());
                std::vector<double> sigmas(_gps.size());
                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    Eigen::VectorXd tmu;
                    double ts;
                    std::tie(tmu, ts) = _gps[i].query(v);

                    mus[i] = tmu(0);
                    sigmas[i] = 1.0 / (ts + 1e-12);
                });

                double multi_sg = std::accumulate(sigmas.begin(), sigmas.end(), 0.0, std::plus<double>());

                return 1.0 / multi_sg;
            }

            /// return the number of dimensions of the input
            int dim_in() const
            {
                assert(_gps.size());
                return _gps[0].dim_in();
            }

            /// return the number of dimensions of the output
            int dim_out() const
            {
                assert(_gps.size());
                return _gps[0].dim_out();
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

            // TO-DO: Add helper functions: max_observation, mean_observation, mean_vector, obs_mean

            /// return the number of samples used to compute the GP
            int nb_samples() const { return _samples.size(); }

            ///  recomputes the GP
            void recompute(bool update_obs_mean = true, bool update_full_kernel = true)
            {
                assert(!_samples.empty() && _gps.size());

                _update_kernel_and_mean_functions();

                limbo::tools::par::loop(0, _gps.size(), [&](size_t i) {
                    _gps[i].recompute(update_obs_mean, update_full_kernel);
                });
            }

            /// return the mean observation
            Eigen::VectorXd mean_observation() const
            {
                assert(_gps.size());
                return _observations.size() > 0 ? _mean_observation
                                                : Eigen::VectorXd::Zero(dim_out());
            }

            /// return the list of GPs
            std::vector<GP_t> gp_models() const
            {
                return _gps;
            }

            /// return the list of GPs
            std::vector<GP_t>& gp_models()
            {
                return _gps;
            }

            /// compute and return the log likelihood
            double compute_log_lik()
            {
                assert(_gps.size());

                _log_lik = 0.0;
                for (auto gp : _gps) {
                    _log_lik += gp.compute_log_lik();
                }

                return _log_lik;
            }

            /// compute and return the gradient of the log likelihood wrt to the kernel parameters
            Eigen::VectorXd compute_kernel_grad_log_lik()
            {
                assert(_gps.size());

                Eigen::VectorXd grad = _gps[0].compute_kernel_grad_log_lik();
                for (size_t i = 1; i < _gps.size(); i++) {
                    grad.array() += _gps[i].compute_kernel_grad_log_lik().array();
                }

                return grad;
            }

            /// compute and return the gradient of the log likelihood wrt to the mean parameters
            Eigen::VectorXd compute_mean_grad_log_lik()
            {
                assert(_gps.size());

                Eigen::VectorXd grad = _gps[0].compute_mean_grad_log_lik();
                for (size_t i = 1; i < _gps.size(); i++) {
                    grad.array() += _gps[i].compute_mean_grad_log_lik();
                }

                return grad;
            }

            /// return the likelihood (do not compute it -- return last computed)
            double get_log_lik() const { return _log_lik; }

            /// set the log likelihood (e.g. computed from outside)
            void set_log_lik(double log_lik) { _log_lik = log_lik; }

            // TO-DO: Add log LOO-CV and gradients

            /// return the list of samples
            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

            /// return the list of observations
            const std::vector<Eigen::VectorXd>& observations() const
            {
                return _observations;
            }

            /// return the observations (in matrix form)
            /// (NxD), where N is the number of points and D is the dimension output
            Eigen::MatrixXd observations_matrix() const
            {
                Eigen::MatrixXd obs(_observations.size(), dim_out());
                for (size_t i = 0; i < _observations.size(); i++) {
                    obs.row(i) = _observations[i];
                }

                return obs;
            }

            /// save the parameters and the data for the GP to the archive (text or binary)
            template <typename A>
            void save(const std::string& directory) const
            {
                A archive(directory);
                save(archive);
            }

            /// save the parameters and the data for the GP to the archive (text or binary)
            template <typename A>
            void save(const A& archive) const
            {
                // Eigen::VectorXd dims(2);
                // dims << dim_in(), dim_out();
                // archive.save(dims, "dims");

                size_t size = _gps.size();
                Eigen::VectorXd s(1);
                s << size;
                archive.save(s, "size");

                archive.save(_observations, "observations");

                for (size_t i = 0; i < size; i++) {
                    _gps[i].template save<A>(archive.directory() + "/gp_" + std::to_string(i));
                }
            }

            /// load the parameters and the data for the GP from the archive (text or binary)
            /// if recompute is true, we do not read the kernel matrix
            /// but we recompute it given the data and the hyperparameters
            template <typename A>
            void load(const std::string& directory, bool recompute = true)
            {
                A archive(directory);
                load(archive, recompute);
            }

            /// load the parameters and the data for the GP from the archive (text or binary)
            /// if recompute is true, we do not read the kernel matrix
            /// but we recompute it given the data and the hyperparameters
            template <typename A>
            void load(const A& archive, bool recompute = true)
            {
                _observations.clear();
                archive.load(_observations, "observations");

                // Eigen::VectorXd dims;
                // archive.load(dims, "dims");

                Eigen::VectorXd s;
                archive.load(s, "size");
                _gps.resize(static_cast<size_t>(s[0]));

                // recompute mean observation
                _mean_observation = Eigen::VectorXd::Zero(_observations[0].size());
                for (size_t j = 0; j < _observations.size(); j++)
                    _mean_observation.array() += _observations[j].array();
                _mean_observation.array() /= static_cast<double>(_observations.size());

                for (size_t i = 0; i < _gps.size(); i++) {
                    // do not recompute the individual GPs on their own
                    _gps[i].template load<A>(archive.directory() + "/gp_" + std::to_string(i), false);
                }

                if (recompute)
                    this->recompute(true, true);
                else // if we do not wish to recompute, update the kernel and mean
                    _update_kernel_and_mean_functions();
            }

        protected:
            std::vector<GP_t> _gps;
            HyperParamsOptimizer _hp_optimize;
            std::vector<Eigen::VectorXd> _samples, _observations;
            Eigen::VectorXd _mean_observation;
            double _log_lik = 0.;

            void _update_kernel_and_mean_functions()
            {
                assert(_gps.size());

                limbo::tools::par::loop(1, _gps.size(), [&](size_t i) {
                    _gps[i].kernel_function() = _gps[0].kernel_function();
                    _gps[i].mean_function() = _gps[0].mean_function();
                });
            }
        };
    } // namespace model
} // namespace limbo

#endif