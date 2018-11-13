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
#ifndef LIMBO_OPT_TR_CMAES_HPP
#define LIMBO_OPT_TR_CMAES_HPP

#include <Eigen/Core>
#include <iostream>
#include <vector>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>

namespace limbo {
    namespace defaults {
        struct opt_tr_cmaes {
            BO_PARAM(int, max_fun_evals, -1);
        };
    } // namespace defaults

    namespace opt {
        /// @ingroup opt
        /// Trust Region CMA-ES
        /// Abdolmaleki, A., Price, B., Lau, N., Reis, L.P. and Neumann, G.
        /// "Deriving and improving CMA-ES with information geometric trust regions."
        /// In Proceedings of the Genetic and Evolutionary Computation Conference (pp. 657-664). 2017.
        template <typename Params, typename Opt>
        struct TRCmaes {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, double bounded) const
            {
                static thread_local limbo::tools::rgen_gauss_t gauss_rng(0., 1.);
                size_t dim = init.size();

                // return -eval(f, m);

                // Algorithm variables
                Eigen::MatrixXd B = Eigen::MatrixXd::Identity(dim, dim); // B defines the coordinate system
                Eigen::VectorXd D = Eigen::VectorXd::Ones(dim); // diagonal D defines the scaling
                Eigen::VectorXd D_square = D.array().square();
                Eigen::MatrixXd C = B * D_square.asDiagonal() * B.transpose(); // covariance matrix C

                double sigma = 1.; // initial sigma
                int lambda = 4 + std::floor(3 * std::log(dim)); // population size, offspring number

                Eigen::VectorXd xmean = init;

                int max_evals = Params::opt_tr_cmaes::max_fun_evals(); // max number of evaluations
                if (max_evals < 1) {
                    max_evals = 100; // just for testing
                }
                int evals = 0;

                TRCmaesLearner learner(dim, xmean, C, sigma, lambda);

                while (evals < max_evals) {
                    // std::cout << "evals: " << evals << std::endl;
                    // std::cout << "xmean: " << xmean.transpose() << std::endl;
                    // std::cout << "C: " << C << std::endl
                    //           << std::endl;
                    // std::cout << "B: " << B << std::endl
                    //           << std::endl;
                    // std::cout << "D: " << D.transpose() << std::endl
                    //           << std::endl;
                    // std::cout << "sigma: " << sigma << std::endl;
                    // std::cin.get();

                    Eigen::MatrixXd pop(dim, lambda);
                    for (int i = 0; i < lambda; i++) {
                        Eigen::VectorXd random_D = D.array() * (tools::random_vec(dim, gauss_rng)).array();
                        pop.col(i) = xmean + B * random_D;
                    }

                    Eigen::VectorXd fitness(lambda);
                    tools::par::loop(0, lambda, [&](size_t i) {
                        fitness(i) = eval(f, pop.col(i));
                    });
                    evals += lambda;

                    std::tie(xmean, C, sigma) = learner.update_parameters(fitness, pop);

                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C * sigma);
                    if (eig.info() != Eigen::Success) {
                        std::cout << "EigenSolver error!" << std::endl;
                        std::cout << C << std::endl
                                  << std::endl;
                        std::cout << sigma << std::endl;
                        break;
                    }
                    B = eig.eigenvectors();
                    D = eig.eigenvalues().cwiseMax(0).cwiseSqrt();
                }

                return xmean;
            }

        protected:
            struct TRCmaesLearner {
            public:
                TRCmaesLearner(int dim, const Eigen::VectorXd& xmean, const Eigen::MatrixXd& init_cov, double init_sigma, int lambda)
                {
                    _dim = dim;

                    // obj.pc = zeros(1, obj.dimParameters);
                    _pc = Eigen::VectorXd::Zero(_dim); // usually close to 1
                    _ps_step_size = Eigen::VectorXd::Zero(_dim);

                    _old_mean = xmean;
                    _old_cov = init_cov * init_sigma;
                    _old_shape_mat = init_cov;
                    _old_step_size = init_sigma;

                    _lambda = lambda;
                    _mu = std::floor(lambda / 2.);

                    _weights = Eigen::VectorXd(_mu);
                    double log_mu_half = std::log(_mu + 0.5);
                    for (int i = 0; i < _mu; i++) {
                        _weights(i) = log_mu_half - std::log(i + 1);
                    }
                    double w_sum = _weights.sum();

                    _weights /= w_sum;

                    w_sum = _weights.sum();

                    _mu_eff = w_sum * w_sum / (_weights.array().square().sum());
                }

                std::tuple<Eigen::VectorXd, Eigen::MatrixXd, double> update_parameters(const Eigen::VectorXd& fitness, const Eigen::MatrixXd& pop)
                {
                    Eigen::MatrixXd matrixL = Eigen::LLT<Eigen::MatrixXd>(_old_shape_mat).matrixL();

                    // compute inverse with cholesky
                    Eigen::MatrixXd inv_old_shape_mat = Eigen::MatrixXd::Identity(_old_shape_mat.rows(), _old_shape_mat.cols());
                    matrixL.template triangularView<Eigen::Lower>().solveInPlace(inv_old_shape_mat);
                    matrixL.template triangularView<Eigen::Lower>().transpose().solveInPlace(inv_old_shape_mat);

                    _inv_old_cov = inv_old_shape_mat / _old_step_size;

                    _logdet_old_shape_mat = 2 * matrixL.diagonal().array().log().sum();

                    // std::cout << "_old_shape_mat: " << _old_shape_mat << std::endl;
                    // std::cout << "_inv_old_cov: " << _inv_old_cov << std::endl;

                    // sort individuals
                    std::vector<int> idx(fitness.size());
                    std::iota(std::begin(idx), std::end(idx), 0);

                    std::sort(std::begin(idx), std::end(idx), [&](int a, int b) {
                        return fitness(b) < fitness(a); // we want descending order --- we are maximizing
                    });

                    _population = Eigen::MatrixXd(pop.rows(), _mu);
                    for (int i = 0; i < _mu; i++) {
                        _population.col(i) = pop.col(idx[i]);
                    }

                    // std::cout << "_population: " << _population << std::endl;

                    // do the updates
                    double dim_double = _dim;
                    _cc = (4. + _mu_eff / dim_double) / (dim_double + 4. + 2. * _mu_eff / dim_double); // time constant for cumulation for C
                    _cs = (_mu_eff + 2.) / (dim_double + _mu_eff + 5.); // t-const for cumulation for sigma control

                    _c1 = (4. * dim_double) / (std::pow(dim_double + 1.3, 2.) + _mu_eff);

                    _epsilon_shape = std::min(0.2, 1.5 * ((_mu_eff + 1. / _mu_eff) / (std::pow(dim_double + 2., 2.) + _mu_eff)));
                    _epsilon_step_size = (_mu_eff * _mu_eff) / (2. * dim_double);

                    Eigen::VectorXd sample_mean = (_population.array().rowwise() * _weights.transpose().array()).rowwise().sum();
                    // std::cout << "sample_mean: " << sample_mean.transpose() << std::endl;

                    // maybe vectorize difference computaiton?
                    Eigen::MatrixXd difference(_population.rows(), _population.cols());
                    for (int i = 0; i < _population.cols(); i++) {
                        difference.col(i) = _population.col(i) - _old_mean;
                    }
                    Eigen::MatrixXd tmp = (difference.array().rowwise() * _weights.transpose().array());
                    _sample_cov = tmp * difference.transpose();
                    // std::cout << "_sample_cov: " << _sample_cov << std::endl;

                    Eigen::VectorXd xmean = sample_mean;

                    double hsig = 1.;
                    // std::cout << "_pc (before): " << _pc.transpose() << std::endl;
                    // std::cout << "_cc: " << _cc << std::endl;
                    // std::cout << "_mu_eff: " << _mu_eff << std::endl;
                    // std::cout << "sqrt1: " << std::sqrt(_cc * (2. - _cc) * _mu_eff) << std::endl;
                    // std::cout << "sqrt2: " << std::sqrt(_old_step_size) << std::endl;
                    _pc = (1. - _cc) * _pc + hsig * std::sqrt(_cc * (2. - _cc) * _mu_eff) * (xmean - _old_mean) / (std::sqrt(_old_step_size) + _constant);

                    _ps_step_size = ((1. - _cs) * _ps_step_size + std::sqrt(_cs * (2. - _cs) * _mu_eff) * (xmean - _old_mean));

                    _rscoff = 1.;
                    _rank1_step_size = _rscoff * (_ps_step_size * _ps_step_size.transpose());
                    _rank1_shape_mat = _c1 * _old_step_size * (_pc * _pc.transpose());
                    // std::cout << "_c1: " << _c1 << std::endl;
                    // std::cout << "_old_step_size: " << _old_step_size << std::endl;
                    // std::cout << "_pc: " << _pc.transpose() << std::endl;
                    // std::cout << "_rank1_shape_mat: " << _rank1_shape_mat << std::endl;

                    Eigen::MatrixXd new_shape = _optimize_shape_dual_function();

                    // TO-DO: Maybe remove the tmp matrices
                    Eigen::MatrixXd new_shape_triang, new_shape_triang_tr;
                    new_shape_triang = new_shape.triangularView<Eigen::Upper>();
                    new_shape_triang_tr = new_shape.triangularView<Eigen::StrictlyUpper>().transpose();
                    new_shape = new_shape_triang + new_shape_triang_tr;

                    // std::cout << "new_shape: " << new_shape << std::endl;

                    double new_step_size = std::max(0., _optimize_step_size_dual_function()); // step-size should always be bigger than zero

                    // std::cout << "new_step_size: " << new_step_size << std::endl;

                    _old_mean = xmean;
                    _old_cov = new_step_size * new_shape;
                    _old_shape_mat = new_shape;
                    _old_step_size = new_step_size;

                    return {xmean, new_shape, new_step_size};
                }

            protected:
                static constexpr double _constant = 1e-20;
                int _dim, _lambda, _mu;
                double _old_step_size, _mu_eff, _logdet_old_shape_mat;
                double _cs, _cc, _c1, _epsilon_shape, _epsilon_step_size, _rscoff;
                Eigen::VectorXd _old_mean, _weights, _pc, _ps_step_size;
                Eigen::MatrixXd _old_cov, _old_shape_mat, _inv_old_cov, _population, _sample_cov, _rank1_step_size, _rank1_shape_mat;

                double _dual_function_shape(const Eigen::VectorXd& params) const
                {
                    double lambda = params(0);
                    Eigen::MatrixXd rank1 = _rank1_shape_mat;
                    Eigen::MatrixXd new_shape = (lambda * _old_shape_mat + ((_sample_cov + rank1) / (_old_step_size + _constant))).array() / (lambda + _c1 + 1.);

                    Eigen::LLT<Eigen::MatrixXd> llt(new_shape);
                    Eigen::MatrixXd matrixL = llt.matrixL();
                    double logdet = 2 * matrixL.diagonal().array().log().sum();

                    Eigen::MatrixXd tmp = _sample_cov + rank1 + lambda * _old_cov;
                    Eigen::MatrixXd sol = llt.solve(tmp);
                    double tr = sol.trace();

                    return -(1. + _c1 + lambda) * logdet - (1. / (_old_step_size + _constant)) * tr + lambda * (2. * _epsilon_shape + _logdet_old_shape_mat + _dim);
                }

                Eigen::MatrixXd _optimize_shape_dual_function() const
                {
                    Eigen::VectorXd params(1);
                    params << 10.; // lambda shape initial

                    Opt opt;
                    params = opt([&](const Eigen::VectorXd& p, bool compute_grad) {
                        assert(!compute_grad);
                        return opt::no_grad(-_dual_function_shape(p));
                    },
                        params, false); // no bounds

                    double lambda = params(0);

                    Eigen::MatrixXd rank1 = _rank1_shape_mat;
                    Eigen::MatrixXd new_shape = (lambda * _old_shape_mat + ((_sample_cov + rank1) / (_old_step_size + _constant))).array() / (lambda + _c1 + 1.);

                    return new_shape;
                }

                double _dual_function_step_size(const Eigen::VectorXd& params) const
                {
                    double dim_double = _dim;
                    double lambda = params(0);
                    Eigen::MatrixXd rank1 = _rank1_step_size;

                    Eigen::LLT<Eigen::MatrixXd> llt(_old_shape_mat);
                    Eigen::MatrixXd matrixL = llt.matrixL();

                    Eigen::MatrixXd tmp = _sample_cov + rank1 + lambda * _old_cov;
                    Eigen::MatrixXd sol = llt.solve(tmp);
                    double tr = sol.trace();

                    double new_step_size = (tr / dim_double) / (1. + lambda + _rscoff);

                    return -(1. + _rscoff + lambda) * (dim_double * std::log(new_step_size + _constant)) - (tr / (new_step_size + _constant)) + lambda * (2. * _epsilon_step_size + dim_double + dim_double * std::log(_old_step_size + _constant));
                }

                double _optimize_step_size_dual_function() const
                {
                    Eigen::VectorXd params(1);
                    params << 1.; // lambda step size initial

                    Opt opt;
                    params = opt([&](const Eigen::VectorXd& p, bool compute_grad) {
                        assert(!compute_grad);
                        return opt::no_grad(-_dual_function_step_size(p));
                    },
                        params, false); // no bounds

                    double dim_double = _dim;
                    double lambda = params(0);

                    Eigen::MatrixXd rank1 = _rank1_step_size;
                    Eigen::LLT<Eigen::MatrixXd> llt(_old_shape_mat);
                    Eigen::MatrixXd matrixL = llt.matrixL();

                    Eigen::MatrixXd tmp = rank1 + _sample_cov + lambda * _old_cov;
                    Eigen::MatrixXd sol = llt.solve(tmp);
                    double tr = sol.trace();

                    return (tr / dim_double) / (1. + lambda + _rscoff);
                }
            };
        };
    } // namespace opt
} // namespace limbo

#endif