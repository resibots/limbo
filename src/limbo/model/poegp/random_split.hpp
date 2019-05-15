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
#ifndef LIMBO_MODEL_POEGP_RANDOM_SPLIT_HPP
#define LIMBO_MODEL_POEGP_RANDOM_SPLIT_HPP

#include <vector>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

// Quick hack for definition of 'I' in <complex.h>
#undef I

namespace limbo {
    namespace model {
        namespace poegp {
            template <typename Params>
            struct RandomSplit {
            public:
                std::pair<std::vector<std::vector<Eigen::VectorXd>>, std::vector<std::vector<Eigen::VectorXd>>> operator()(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, size_t K)
                {
                    std::vector<Eigen::VectorXd> rand_samples = samples, rand_obs = observations;

                    for (size_t i = samples.size() - 1; i > 0; --i) {
                        tools::rgen_int_t rgen(0, i);
                        size_t index = rgen.rand();
                        std::swap(rand_samples[i], rand_samples[index]);
                        std::swap(rand_obs[i], rand_obs[index]);
                    }

                    size_t n = std::ceil(samples.size() / static_cast<double>(K));
                    std::vector<std::vector<Eigen::VectorXd>> split_samples, split_obs;

                    for (size_t i = 0; i < K; i++) {
                        split_samples.push_back(std::vector<Eigen::VectorXd>());
                        split_obs.push_back(std::vector<Eigen::VectorXd>());

                        for (size_t j = 0; j < n; j++) {
                            if ((i * n + j) >= samples.size()) // TO-DO: Check how to handle these cases
                                break;
                            split_samples[i].push_back(rand_samples[i * n + j]);
                            split_obs[i].push_back(rand_obs[i * n + j]);
                        }
                    }

                    return {split_samples, split_obs};
                }
            };
        } // namespace poegp
    } // namespace model
} // namespace limbo

#endif