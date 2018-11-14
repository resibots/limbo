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
#ifndef LIMBO_OPT_CMA_HPP
#define LIMBO_OPT_CMA_HPP

#include <Eigen/Core>
#include <iostream>
#include <vector>

#include <limbo/opt/cmaes/ask_mvn.hpp>
#include <limbo/opt/cmaes/cmaes_learner.hpp>
#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>

namespace limbo {
    namespace defaults {
        struct opt_cma {
            BO_PARAM(int, max_fun_evals, -1);
        };
    } // namespace defaults

    namespace opt {
        /// @ingroup opt
        /// CMA Evolutionary Strategy
        template <typename Params, typename Ask = cmaes::AskMVN<Params>, typename Learner = cmaes::CmaesLearner<Params>>
        struct CMA {
        public:
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, double bounded) const
            {
                size_t dim = init.size();

                // Algorithm variables
                Eigen::MatrixXd B = Eigen::MatrixXd::Identity(dim, dim); // B defines the coordinate system
                Eigen::VectorXd D = Eigen::VectorXd::Ones(dim); // diagonal D defines the scaling
                Eigen::VectorXd D_square = D.array().square();
                Eigen::MatrixXd C = B * D_square.asDiagonal() * B.transpose(); // covariance matrix C

                double sigma = 1.; // initial sigma
                int lambda = 4 + std::floor(3 * std::log(dim)); // population size, offspring number

                Eigen::VectorXd xmean = init;

                int max_evals = Params::opt_cma::max_fun_evals(); // max number of evaluations
                if (max_evals < 1) {
                    max_evals = 100; // just for testing
                }
                int evals = 0;

                Learner learner(xmean, C, sigma, lambda);
                Ask ask;

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
                    // get population
                    Eigen::MatrixXd pop = ask(lambda, xmean, B, D, sigma);

                    // evaluate fitness
                    Eigen::VectorXd fitness(lambda);
                    tools::par::loop(0, lambda, [&](size_t i) {
                        fitness(i) = eval(f, pop.col(i));
                    });
                    evals += lambda;

                    // std::cout << B * D.asDiagonal().inverse() * B.transpose() << std::endl;

                    // update distribution and step size
                    std::tie(xmean, C, sigma) = learner.update_parameters(fitness, pop, evals);

                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C); // C * sigma
                    B = eig.eigenvectors();
                    D = eig.eigenvalues().cwiseMax(0).cwiseSqrt();
                    // if (eig.info() != Eigen::Success) {
                    //     std::cout << "EigenSolver error!" << std::endl;
                    //     std::cout << C << std::endl
                    //               << std::endl;
                    //     std::cout << sigma << std::endl;
                    //     break;
                    // }
                }

                return xmean;
            }
        };
    } // namespace opt
} // namespace limbo

#endif