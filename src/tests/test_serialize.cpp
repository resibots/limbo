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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_serialize

#include <cstring>
#include <fstream>

// Quick hack for definition of 'I' in <complex.h>
#undef I
#include <boost/test/unit_test.hpp>

#include <limbo/model/gp.hpp>
#include <limbo/serialize/text_archive.hpp>

struct Params {
    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 1.0);
        BO_PARAM(double, l, 0.2);
    };
    struct kernel : public limbo::defaults::kernel {
    };
    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public limbo::defaults::opt_rprop {
    };
    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
    };
};

BOOST_AUTO_TEST_CASE(test_text_archive)
{
    using namespace limbo;

    // our data (3-D inputs, 1-D outputs)
    std::vector<Eigen::VectorXd> samples;
    std::vector<Eigen::VectorXd> observations;

    size_t n = 8;
    for (size_t i = 0; i < n; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        samples.push_back(s);
        observations.push_back(tools::make_vector(std::cos(s(0) * s(1) * s(2))));
    }
    // 3-D inputs, 1-D outputs
    model::GPOpt<Params> gp(3, 1);
    gp.compute(samples, observations);
    gp.optimize_hyperparams();

    // attempt to save
    serialize::TextArchive a1("/tmp/test_model.dat");
    gp.save(a1);

    // attempt to read
    model::GPOpt<Params> gp2(3, 1);
    serialize::TextArchive a2("/tmp/test_model.dat");
    gp2.load(a2);

    // check that the 2 GPs match
    size_t N = 100;

    double diff_mu = 0.;
    double diff_sigma = 0.;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        samples.push_back(s);

        Eigen::VectorXd mu1, mu2;
        double s1, s2;

        std::tie(mu1, s1) = gp.query(s);
        std::tie(mu2, s2) = gp2.query(s);

        diff_mu += std::abs(mu1(0) - mu2(0));
        diff_sigma += std::abs(s1 - s2);
    }

    diff_mu /= double(N);
    diff_sigma /= double(N);

    BOOST_CHECK(diff_mu < 1e-6);
    BOOST_CHECK(diff_sigma < 1e-6);
}

BOOST_AUTO_TEST_CASE(test_bin_archive)
{
}
