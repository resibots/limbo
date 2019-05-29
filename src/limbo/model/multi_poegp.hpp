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
#ifndef LIMBO_MODEL_MULTI_POEGP_HPP
#define LIMBO_MODEL_MULTI_POEGP_HPP

#include <limbo/model/multi_gp.hpp>
#include <limbo/model/poegp.hpp>
#include <limbo/model/poegp/no_split.hpp>

namespace limbo {
    namespace model {
        /// @ingroup model
        /// A wrapper for N-output Gaussian processes where each GP is a product of experts
        /// It is parametrized by:
        /// - GP class
        /// - a kernel function (the same type for all GPs, but can have different parameters)
        /// - a mean function (the same type and parameters for all GPs)
        /// - [optional] an optimizer for the hyper-parameters
        template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>, typename Split = poegp::RandomSplit<Params>>
        class MultiPOEGP : public MultiGP<Params, POEGP, KernelFunction, MeanFunction, HyperParamsOptimizer, poegp::NoSplit<Params>> {
        public:
            // We want POEGPs that do not split the data
            using base_t = MultiGP<Params, POEGP, KernelFunction, MeanFunction, HyperParamsOptimizer, poegp::NoSplit<Params>>;
            using GP_t = typename base_t::GP_t;

            /// useful because the model might be created before knowing anything about the process
            MultiPOEGP() : base_t() {}

            /// useful because the model might be created before having samples
            MultiPOEGP(int dim_in, int dim_out) : base_t(dim_in, dim_out) {}

            /// Compute the GP from samples and observations. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = true)
            {
                // Get parameters
                int n = Params::model_poegp::expert_size();
                size_t N = samples.size();
                size_t K = std::ceil(N / static_cast<double>(n));

                // Split data-once
                std::vector<std::vector<Eigen::VectorXd>> split_samples, split_obs;
                std::tie(split_samples, split_obs) = _split(samples, observations, K);
                assert(split_samples.size() == split_obs.size());
                K = split_samples.size(); // the splitting algorithm might give a different number of partitions than requested

                if (this->_dim_in != samples[0].size()) {
                    this->_dim_in = samples[0].size();
                }

                if (this->_dim_out != observations[0].size()) {
                    this->_dim_out = observations[0].size();
                    this->_mean_function = MeanFunction(this->_dim_out); // the cost of building a functor should be relatively low
                }

                if ((int)this->_gp_models.size() != this->_dim_out) {
                    this->_gp_models.resize(this->_dim_out);
                    for (int i = 0; i < this->_dim_out; i++)
                        this->_gp_models[i] = GP_t(this->_dim_in, 1);
                }

                // save observations
                // TO-DO: Check how can we improve for not saving observations twice (one here and one for each GP)!?
                this->_observations = observations;

                // compute the new observations for the GPs
                std::vector<std::vector<std::vector<Eigen::VectorXd>>> obs(this->_dim_out);
                for (int i = 0; i < this->_dim_out; i++)
                    obs[i].resize(K);

                // compute mean observation
                this->_mean_observation = Eigen::VectorXd::Zero(this->_dim_out);
                for (size_t j = 0; j < this->_observations.size(); j++)
                    this->_mean_observation.array() += this->_observations[j].array();
                this->_mean_observation.array() /= static_cast<double>(this->_observations.size());

                // for (size_t j = 0; j < observations.size(); j++) {
                //     // obs[i].resize(K);
                //     for (size_t k = 0; k < K; k++) {
                //         Eigen::VectorXd mean_vector = this->_mean_function(split_samples[k][j], *this);
                //         assert(mean_vector.size() == this->_dim_out);
                //         for (int i = 0; i < this->_dim_out; i++) {
                //             obs[i][k].push_back(limbo::tools::make_vector(split_obs[k][j][i] - mean_vector[i]));
                //         }
                //     }
                // }
                for (size_t k = 0; k < K; k++) {
                    for (size_t j = 0; j < split_samples[k].size(); j++) {
                        Eigen::VectorXd mean_vector = this->_mean_function(split_samples[k][j], *this);
                        assert(mean_vector.size() == this->_dim_out);
                        for (int i = 0; i < this->_dim_out; i++) {
                            obs[i][k].push_back(limbo::tools::make_vector(split_obs[k][j][i] - mean_vector[i]));
                        }
                    }
                }

                // do the actual computation
                limbo::tools::par::loop(0, this->_dim_out, [&](size_t i) {
                    this->_gp_models[i].compute(split_samples, obs[i], compute_kernel);
                });
            }

        protected:
            Split _split;
        };
    } // namespace model
} // namespace limbo

#endif