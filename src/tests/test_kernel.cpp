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
#define BOOST_TEST_MODULE test_kernel

#include <boost/test/unit_test.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <iostream>

using namespace limbo;
struct Params {
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.0);
    };

    struct kernel_squared_exp_ard {
        BO_DYN_PARAM(int, k); //equivalent to the standard exp ARD
        BO_PARAM(double, sigma_sq, 1);
    };
};

BO_DECLARE_DYN_PARAM(int, Params::kernel_squared_exp_ard, k);

Eigen::VectorXd make_v2(double x1, double x2)
{
    Eigen::VectorXd v2(2);
    v2 << x1, x2;
    return v2;
}

BOOST_AUTO_TEST_CASE(test_kernel_SE_ARD)
{
    Params::kernel_squared_exp_ard::set_k(0);

    kernel::SquaredExpARD<Params> se(2);
    Eigen::VectorXd hp(se.h_params_size());
    hp(0) = 0;
    hp(1) = 0;

    se.set_h_params(hp);

    Eigen::VectorXd v1 = make_v2(1, 1);
    BOOST_CHECK(std::abs(se(v1, v1) - 1) < 1e-6);

    Eigen::VectorXd v2 = make_v2(0, 1);
    double s1 = se(v1, v2);

    BOOST_CHECK(std::abs(s1 - std::exp(-0.5 * (v1.transpose() * v2)[0])) < 1e-5);

    hp(0) = 1;
    se.set_h_params(hp);
    double s2 = se(v1, v2);
    BOOST_CHECK(s1 < s2);

    Params::kernel_squared_exp_ard::set_k(1);
    se = kernel::SquaredExpARD<Params>(2);
    hp = Eigen::VectorXd(se.h_params_size());
    hp(0) = 0;
    hp(1) = 0;
    hp(2) = -std::numeric_limits<double>::max();
    hp(3) = -std::numeric_limits<double>::max();

    se.set_h_params(hp);
    BOOST_CHECK(s1 == se(v1, v2));
}
