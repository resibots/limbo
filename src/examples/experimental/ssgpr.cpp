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
#include <fstream>
#include <limbo/experimental/model/ssgpr.hpp>
#include <limbo/experimental/model/ssgpr/lf_opt.hpp>
#include <limbo/experimental/ssgpr/sparse_spectrum_features.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/null_function.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>

using namespace limbo;

struct Params {
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.1 * 0.1);
        BO_PARAM(bool, optimize_noise, true);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public defaults::opt_rprop {
    };
    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };

    struct opt_cmaes : public defaults::opt_cmaes {
        BO_PARAM(int, restarts, 5);
        BO_PARAM(double, max_fun_evals, 1000);
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
        BO_PARAM(bool, fun_compute_initial, true);
        BO_PARAM(int, variant, aBIPOP_CMAES);
        BO_PARAM(int, elitism, 1);

        BO_PARAM(double, lbound, -6.);
        BO_PARAM(double, ubound, 6.);
    };

    struct sparse_spectrum_features : public defaults::sparse_spectrum_features {
        BO_PARAM(int, nproj, 20);
        BO_PARAM(double, sigma_o, 1.0);
        BO_PARAM(bool, fixed, true);
    };
};

int main(int argc, char** argv)
{
    // our data (1-D inputs, 1-D outputs)
    double bounds = 20.0;
    int N_interp = 200;
    std::vector<Eigen::VectorXd> samples;
    std::vector<Eigen::VectorXd> observations;

    size_t N = 100;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd s = tools::random_vector(1).array() * 2. * bounds - bounds;
        samples.push_back(s);

        // TODO try more complicated function. cosine is trivial for the nonlinear mapping we're using
        observations.push_back(tools::make_vector(std::cos(s(0))));
    }

    // the type of the GP
    using SSGPR_t = limbo::experimental::model::SSGPR<Params, limbo::experimental::SparseSpectrumFeatures<Params>, mean::NullFunction<Params>, limbo::experimental::model::ssgpr::LFOpt<Params> >;

    // 1-D inputs, 1-D outputs
    SSGPR_t ssgp(1, 1);

    // timing variables
    std::chrono::time_point<std::chrono::system_clock> start, now;
    long int elapsed;

    // compute the SSGP in batch
    start = std::chrono::system_clock::now();

    std::cout << "Learning the SSGPR..." << std::endl;
    ssgp.compute(samples, observations, false);
    ssgp.optimize_hyperparams();
    std::cout << "Learned!" << std::endl;

    now = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << elapsed << " milliseconds" << std::endl;
    
    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs("ssgp.dat");
    std::vector<Eigen::VectorXd> x_test;
    std::vector<Eigen::VectorXd> batch_mu; 
    for (int i = 0; i < N_interp; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / double(N_interp)).array() * 2. * bounds - bounds;
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = ssgp.query(v);

        ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        x_test.push_back(v);
        batch_mu.push_back(mu);
    }

    // Compute an incremental ssgp 
    SSGPR_t gp2(1, 1);

    start = std::chrono::system_clock::now();

    std::cout << "Learning the I-SSGPR..." << std::endl;

    // Optimize the hyperparameters for a small set at beginning
    int n_init = 10;
    for (int i = 0; i < n_init; i++) {
        gp2.add_sample(samples[i], observations[i]);
    }
    gp2.optimize_hyperparams();

    // Only incremental updates for rest of data
    for (int i = n_init; i < N; i++) {
        gp2.add_sample(samples[i], observations[i]);
    }
    
    std::cout << "Learned!" << std::endl;
    now = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << elapsed << " milliseconds" << std::endl;

    std::ofstream ofs_inc("issgp.dat");
    std::vector<Eigen::VectorXd> incremental_mu; 
    for (int i = 0; i < N_interp; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / double(N_interp)).array() * 2. * bounds - bounds;
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp2.query(v);

        ofs_inc << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        incremental_mu.push_back(mu);
    }

    // check if results are close to results from batch learning
    // for(int i=0; i<N; i++) {
    //     if(! batch_mu[i].isApprox(incremental_mu[i]), 1e-1) {
    //         std::cout << "Incremental GP outputs don't match batch outputs: " << samples[i] << " " << batch_mu[i] << " "  << incremental_mu[i] << std::endl;
    //     }
    // }

    // an alternative is to optimize the hyper-parameters
    // in that case, we need a kernel with hyper-parameters that are designed to be optimized
    using Kernel2_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::NullFunction<Params>;
    using GP2_t = model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params>>;

    GP2_t gp_ard(1, 1);

    start = std::chrono::system_clock::now();

    std::cout << "Learning the normal GP..." << std::endl;
    gp_ard.compute(samples, observations, false);
    gp_ard.optimize_hyperparams();
    std::cout << "Learned!" << std::endl;
    
    now = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
    std::cout << elapsed << " milliseconds" << std::endl;

    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs_ard("gp_ard.dat");
    std::vector<Eigen::VectorXd> ard_mu; 
    for (int i = 0; i < N_interp; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / double(N_interp)).array() * 2. * bounds - bounds;
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp_ard.query(v);

        ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
        ard_mu.push_back(mu);
    }

    // write the data to a file (useful for plotting)
    std::ofstream ofs_data("data.dat");
    for (size_t i = 0; i < samples.size(); ++i)
        ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;

    // Check MSE
    double mse_ss_batch = 0, mse_ss_inc = 0, mse_full_batch = 0;
    for (int i = 0; i < N_interp; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / double(N_interp)).array() * 2. * bounds - bounds;
        Eigen::VectorXd truth = tools::make_vector(std::cos(v(0)));

        mse_ss_batch += (batch_mu[i] - truth).dot(batch_mu[i] - truth);
        mse_ss_inc += (incremental_mu[i] - truth).dot(incremental_mu[i] - truth);
        mse_full_batch += (ard_mu[i] - truth).dot(ard_mu[i] - truth);
    }
    std::cout << "MSEs - SSGP batch: " << mse_ss_batch/N_interp << ", SSGP incremental: " << mse_ss_inc/N_interp << ", Full GP: " << mse_full_batch/N_interp << std::endl;

    return 0;
}
