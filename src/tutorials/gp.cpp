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
#include <limbo/tools/macros.hpp>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <fstream>

// this tutorials shows how to use a Gaussian process for regression

using namespace limbo;

struct Params {
    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 1.0);
        BO_PARAM(double, l, 0.15);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public defaults::opt_rprop {
    };
    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
};

int main(int argc, char** argv)
{

    // our data (1-D inputs, 1-D outputs)
    std::vector<Eigen::VectorXd> samples = {
        tools::make_vector(0),
        tools::make_vector(0.25),
        tools::make_vector(0.5),
        tools::make_vector(1.0)};
    std::vector<Eigen::VectorXd> observations = {
        tools::make_vector(-1.0),
        tools::make_vector(2.0),
        tools::make_vector(1.0),
        tools::make_vector(3.0)};

    // the type of the GP
    typedef kernel::Exp<Params> Kernel_t;
    typedef mean::Data<Params> Mean_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;

    // 1-D inputs, 1-D outputs
    GP_t gp(1, 1);

    // noise is the same for all the samples (0.01)
    Eigen::VectorXd noises = Eigen::VectorXd::Ones(samples.size()) * 0.01;

    // compute the GP
    gp.compute(samples, observations, noises);

    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs("gp.dat");
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / 100.0);
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp.query(v);
        // an alternative (slower) is to query mu and sigma separately:
        //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
        //  double s2 = gp.sigma(v);
        ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
    }

    // an alternative is to optimize the hyper-paramerers
    // in that case, we need a kernel with hyper-parameters that are designed to be optimized
    typedef kernel::SquaredExpARD<Params> Kernel2_t;
    typedef mean::Data<Params> Mean_t;
    typedef model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params>> GP2_t;

    GP2_t gp_ard(1, 1);
    // do not forget to call the optimization!
    gp_ard.compute(samples, observations, noises, false);
    gp_ard.optimize_hyperparams();

    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs_ard("gp_ard.dat");
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / 100.0);
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp_ard.query(v);
        ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
    }

    // write the data to a file (useful for plotting)
    std::ofstream ofs_data("data.dat");
    for (size_t i = 0; i < samples.size(); ++i)
        ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;
    return 0;
}
