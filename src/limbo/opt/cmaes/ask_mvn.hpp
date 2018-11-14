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
#ifndef LIMBO_OPT_CMAES_ASK_MVN_HPP
#define LIMBO_OPT_CMAES_ASK_MVN_HPP

#include <Eigen/Core>

#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace opt {
        namespace cmaes {
            template <typename Params>
            struct AskMVN {
            public:
                Eigen::MatrixXd operator()(int lambda, const Eigen::VectorXd& xmean, const Eigen::MatrixXd& B, const Eigen::VectorXd& D, double sigma) const
                {
                    static thread_local limbo::tools::rgen_gauss_t gauss_rng(0., 1.);
                    int dim = xmean.size();

                    Eigen::MatrixXd pop(dim, lambda);
                    for (int i = 0; i < lambda; i++) {
                        Eigen::VectorXd random_D = D.array() * (tools::random_vec(dim, gauss_rng)).array();
                        pop.col(i) = xmean + sigma * B * random_D;
                    }

                    return pop;
                }
            };
        } // namespace cmaes
    } // namespace opt
} // namespace limbo

#endif