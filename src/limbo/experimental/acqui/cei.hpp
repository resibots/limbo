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
#ifndef LIMBO_ACQUI_CEI_HPP
#define LIMBO_ACQUI_CEI_HPP

#include <cmath>
#include <vector>
#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct acqui_cei {
            /// @ingroup acqui_defaults
            BO_PARAM(double, jitter, 0.0);
        };
    }

    namespace experimental {
        namespace acqui {
            template <typename Params, typename Model, typename ConstraintModel>
            class CEI {
            public:
                CEI(const Model& model, const ConstraintModel& constraint_model, int iteration = 0)
                    : _model(model), _constraint_model(constraint_model) {}

                size_t dim_in() const { return _model.dim_in(); }

                size_t dim_out() const { return _model.dim_out(); }

                template <typename AggregatorFunction>
                double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
                {
                    Eigen::VectorXd mu;
                    double sigma_sq;
                    std::tie(mu, sigma_sq) = _model.query(v);
                    double sigma = std::sqrt(sigma_sq);

                    // If \sigma(x) = 0 or we do not have any observation yet we return 0
                    if (sigma < 1e-10 || _model.samples().size() < 1)
                        return 0.0;

                    // Compute constrained EI(x)
                    // First find the best (predicted) observation so far
                    // (We are zeroing infeasible samples subject to the constraint value)
                    std::vector<double> rewards;
                    for (auto s : _model.samples())
                        rewards.push_back(afun(_model.mu(s)));

                    double f_max = *std::max_element(rewards.begin(), rewards.end());
                    // Calculate Z and \Phi(Z) and \phi(Z)
                    double X = afun(mu) - f_max - Params::acqui_cei::jitter();
                    double Z = X / sigma;
                    double phi = std::exp(-0.5 * std::pow(Z, 2.0)) / std::sqrt(2.0 * M_PI);
                    double Phi = 0.5 * std::erfc(-Z / std::sqrt(2));

                    return _pf(v, afun) * (X * Phi + sigma * phi);
                }

            protected:
                const Model& _model;
                const ConstraintModel& _constraint_model;

                template <typename AggregatorFunction>
                double _pf(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
                {
                    Eigen::VectorXd mu;
                    double sigma_sq;
                    std::tie(mu, sigma_sq) = _constraint_model.query(v);
                    double sigma = std::sqrt(sigma_sq);

                    if (sigma < 1e-10 || _constraint_model.samples().size() < 1)
                        return 1.0;

                    double Z = (afun(mu) - 1.0) / sigma;
                    double Phi = 0.5 * std::erfc(-Z / std::sqrt(2));

                    return Phi;
                }
            };
        }
    }
}

#endif
