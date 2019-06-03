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
#include <fstream>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/tools.hpp>
#include <limbo/tools/macros.hpp>

#include <limbo/serialize/text_archive.hpp>

// this tutorials shows how to use a Gaussian process for regression

using namespace limbo;

struct Params {
    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 1.0);
        BO_PARAM(double, l, 0.2);
    };
    struct kernel : public defaults::kernel {
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public defaults::opt_rprop {
    };
};

int main(int argc, char** argv)
{

    // our data (1-D inputs, 1-D outputs)
    std::vector<Eigen::VectorXd> samples;
    std::vector<Eigen::VectorXd> observations;

    size_t N = 8;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd s = tools::random_vector(1).array() * 4.0 - 2.0;
        samples.push_back(s);
        observations.push_back(tools::make_vector(std::cos(s(0))));
    }

    // the type of the GP
    using Kernel_t = kernel::Exp<Params>;
    using Mean_t = mean::Data<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t>;

    // 1-D inputs, 1-D outputs
    GP_t gp(1, 1);

    // compute the GP
    gp.compute(samples, observations);

    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs("gp.dat");
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / 100.0).array() * 4.0 - 2.0;
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp.query(v);
        // an alternative (slower) is to query mu and sigma separately:
        //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
        //  double s2 = gp.sigma(v);
        ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
    }

    // an alternative is to optimize the hyper-parameters
    // in that case, we need a kernel with hyper-parameters that are designed to be optimized
    using Kernel2_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::Data<Params>;
    using GP2_t = model::GP<Params, Kernel2_t, Mean_t, model::gp::KernelLFOpt<Params>>;

    GP2_t gp_ard(1, 1);
    // do not forget to call the optimization!
    gp_ard.compute(samples, observations, false);
    gp_ard.optimize_hyperparams();

    // write the predicted data in a file (e.g. to be plotted)
    std::ofstream ofs_ard("gp_ard.dat");
    for (int i = 0; i < 100; ++i) {
        Eigen::VectorXd v = tools::make_vector(i / 100.0).array() * 4.0 - 2.0;
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp_ard.query(v);
        ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
    }

    // write the data to a file (useful for plotting)
    std::ofstream ofs_data("data.dat");
    for (size_t i = 0; i < samples.size(); ++i)
        ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;

    // Sometimes is useful to save an optimized GP
    gp_ard.save<serialize::TextArchive>("myGP");

    // Later we can load -- we need to make sure that the type is identical to the one saved
    gp_ard.load<serialize::TextArchive>("myGP");
    return 0;
}
