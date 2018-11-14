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
#ifndef LIMBO_OPT_CMAES_CMAES_LEARNER_HPP
#define LIMBO_OPT_CMAES_CMAES_LEARNER_HPP

#include <Eigen/Core>

namespace limbo {
    namespace opt {
        namespace cmaes {
            template <typename Params>
            struct CmaesLearner {
            public:
                CmaesLearner(const Eigen::VectorXd& xmean, const Eigen::MatrixXd& init_cov, double init_sigma, int lambda)
                {
                    _dim = xmean.size();

                    _old_mean = xmean;
                    _old_cov = init_cov * init_sigma;
                    _old_shape_mat = init_cov;
                    _old_step_size = init_sigma;

                    _lambda = lambda;
                    _mu = std::floor(lambda / 2.); // TO-DO: Make this parameterizable

                    _weights = Eigen::VectorXd(_mu);
                    double log_mu_half = std::log(_mu + 0.5);
                    for (int i = 0; i < _mu; i++) {
                        _weights(i) = log_mu_half - std::log(i + 1);
                    }
                    double w_sum = _weights.sum();

                    _weights /= w_sum;

                    w_sum = _weights.sum();

                    _mu_eff = w_sum * w_sum / (_weights.array().square().sum());

                    // Strategy parameter setting: Adaptation
                    _cc = (4. + _mu_eff / _dim) / (_dim + 4. + 2. * _mu_eff / _dim); // time constant for cumulation for C
                    _cs = (_mu_eff + 2.) / (_dim + _mu_eff + 5.); // t-const for cumulation for sigma control
                    _c1 = 2. / ((_dim + 1.3) * (_dim + 1.3) + _mu_eff); // learning rate for rank-one update of C
                    _cmu = 2. * (_mu_eff - 2. + 1. / _mu_eff) / ((_dim + 2.) * (_dim + 2.) + 2. * _mu_eff / 2.); // and for rank-mu update
                    _damps = 1. + 2. * std::max(0., std::sqrt((_mu_eff - 1.) / (_dim + 1.)) - 1.) + _cs; // damping for sigma

                    // Initialize dynamic (internal) strategy parameters and constants
                    _pc = Eigen::VectorXd::Zero(_dim);
                    _ps = Eigen::VectorXd::Zero(_dim);
                    _chi_N = std::sqrt(_dim) * (1. - 1. / (4. * _dim) + 1. / (21. * _dim * _dim));
                }

                std::tuple<Eigen::VectorXd, Eigen::MatrixXd, double> update_parameters(const Eigen::VectorXd& fitness, const Eigen::MatrixXd& pop, int countevals)
                {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_solver(_old_shape_mat);

                    Eigen::MatrixXd inv_cov_old = es_solver.operatorInverseSqrt();

                    // sort individuals
                    std::vector<int> idx(fitness.size());
                    std::iota(std::begin(idx), std::end(idx), 0);

                    std::sort(std::begin(idx), std::end(idx), [&](int a, int b) {
                        return fitness(b) < fitness(a); // we want descending order --- we are maximizing
                    });

                    Eigen::MatrixXd population(pop.rows(), _mu);
                    for (int i = 0; i < _mu; i++) {
                        population.col(i) = pop.col(idx[i]);
                    }

                    // compute weighted mean
                    Eigen::VectorXd xmean = population * _weights;

                    Eigen::VectorXd diff_xmean = (xmean - _old_mean) / (_old_step_size + _constant);

                    // do the updates
                    _ps = (1. - _cs) * _ps + (std::sqrt(_cs * (2. - _cs) * _mu_eff)) * inv_cov_old * diff_xmean;

                    double hsig = 0.;
                    double norm_ps = _ps.norm();
                    if ((norm_ps / std::sqrt(1. - std::pow(1. - _cs, (2. * countevals / static_cast<double>(_lambda)))) / _chi_N) < (1.4 + 2. / (_dim + 1.))) {
                        hsig = 1.;
                    }

                    _pc = (1. - _cc) * _pc + hsig * std::sqrt(_cc * (2. - _cc) * _mu_eff) * diff_xmean;

                    // maybe vectorize difference computation?
                    Eigen::MatrixXd wdiff = Eigen::MatrixXd::Zero(_old_cov.rows(), _old_cov.cols());
                    for (int i = 0; i < _mu; i++) {
                        Eigen::VectorXd diff = population.col(i) - _old_mean;
                        wdiff += _weights(i) * (diff * diff.transpose());
                    }
                    wdiff /= (_old_step_size * _old_step_size + _constant);

                    Eigen::MatrixXd spc = _pc * _pc.transpose();
                    Eigen::MatrixXd C = (1. - _c1 - _cmu + (1. - hsig) * _c1 * _cc * (2. - _cc)) * _old_shape_mat + _c1 * spc + _cmu * wdiff;

                    // enforce symmetry
                    Eigen::MatrixXd C_triang, C_triang_tr;
                    C_triang = C.triangularView<Eigen::Upper>();
                    C_triang_tr = C.triangularView<Eigen::StrictlyUpper>().transpose();
                    C = C_triang + C_triang_tr;

                    double sigma = _old_step_size * std::exp((_cs / _damps) * (norm_ps / _chi_N - 1.));

                    _old_mean = xmean;
                    _old_cov = sigma * C;
                    _old_shape_mat = C;
                    _old_step_size = sigma;

                    return {xmean, C, sigma};
                }

            protected:
                static constexpr double _constant = 1e-20;
                int _lambda, _mu;
                double _dim, _cc, _cs, _c1, _cmu, _damps, _chi_N, _old_step_size, _mu_eff;
                Eigen::VectorXd _pc, _ps, _weights, _old_mean;
                Eigen::MatrixXd _old_cov, _old_shape_mat;
            };
        } // namespace cmaes
    } // namespace opt
} // namespace limbo

#endif