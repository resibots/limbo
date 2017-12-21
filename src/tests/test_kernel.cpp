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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_kernel

#include <iostream>

#include <boost/test/unit_test.hpp>

#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/matern_five_halves.hpp>
#include <limbo/kernel/matern_three_halves.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

using namespace limbo;
struct Params {
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.0);
    };

    struct kernel_squared_exp_ard {
        BO_DYN_PARAM(int, k);
        BO_PARAM(double, sigma_sq, 1);
    };

    struct kernel_exp : public defaults::kernel_exp {
    };

    struct kernel_maternthreehalves : public defaults::kernel_maternthreehalves {
    };

    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };
};

struct ParamsNoise {
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.01);
        BO_PARAM(bool, optimize_noise, true);
    };

    struct kernel_squared_exp_ard {
        BO_PARAM(int, k, 0);
        BO_PARAM(double, sigma_sq, 1);
    };

    struct kernel_exp : public defaults::kernel_exp {
    };

    struct kernel_maternthreehalves : public defaults::kernel_maternthreehalves {
    };

    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };
};

BO_DECLARE_DYN_PARAM(int, Params::kernel_squared_exp_ard, k);

Eigen::VectorXd make_v2(double x1, double x2)
{
    Eigen::VectorXd v2(2);
    v2 << x1, x2;
    return v2;
}

// Check gradient via finite differences method
template <typename Kernel>
std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> check_grad(const Kernel& kern, const Eigen::VectorXd& x, const Eigen::VectorXd& x1, const Eigen::VectorXd& x2, double e = 1e-4)
{
    Eigen::VectorXd analytic_result, finite_diff_result;
    Kernel ke = kern;
    ke.set_h_params(x);

    analytic_result = ke.grad(x1, x2);

    finite_diff_result = Eigen::VectorXd::Zero(x.size());
    for (int j = 0; j < x.size(); j++) {
        Eigen::VectorXd test1 = x, test2 = x;
        test1[j] -= e;
        test2[j] += e;
        Kernel k1 = kern;
        k1.set_h_params(test1);
        Kernel k2 = kern;
        k2.set_h_params(test2);
        double res1 = k1(x1, x2);
        double res2 = k2(x1, x2);
        finite_diff_result[j] = (res2 - res1) / (2.0 * e);
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

template <typename Kernel>
void check_kernel(size_t N, size_t K, double e = 1e-6)
{
    Kernel kern(N);

    for (size_t i = 0; i < K; i++) {
        Eigen::VectorXd hp = tools::random_vector(kern.h_params_size()).array() * 6. - 3.;

        double error;
        Eigen::VectorXd analytic, finite_diff;

        Eigen::VectorXd x1 = tools::random_vector(N).array() * 10. - 5.;
        Eigen::VectorXd x2 = tools::random_vector(N).array() * 10. - 5.;

        std::tie(error, analytic, finite_diff) = check_grad(kern, hp, x1, x2, e);
        // std::cout << error << ": " << analytic.transpose() << " vs " << finite_diff.transpose() << std::endl;
        BOOST_CHECK(error < 1e-5);
    }
}

BOOST_AUTO_TEST_CASE(test_grad_exp)
{
    for (int i = 1; i <= 10; i++) {
        check_kernel<kernel::Exp<Params>>(i, 100);
        check_kernel<kernel::Exp<ParamsNoise>>(i, 100);
    }
}

BOOST_AUTO_TEST_CASE(test_grad_matern_three)
{
    for (int i = 1; i <= 10; i++) {
        check_kernel<kernel::MaternThreeHalves<Params>>(i, 100);
        check_kernel<kernel::MaternThreeHalves<ParamsNoise>>(i, 100);
    }
}

BOOST_AUTO_TEST_CASE(test_grad_matern_five)
{
    for (int i = 1; i <= 10; i++) {
        check_kernel<kernel::MaternFiveHalves<Params>>(i, 100);
        check_kernel<kernel::MaternFiveHalves<ParamsNoise>>(i, 100);
    }
}

BOOST_AUTO_TEST_CASE(test_grad_SE_ARD)
{
    Params::kernel_squared_exp_ard::set_k(0);
    for (int i = 1; i <= 10; i++) {
        check_kernel<kernel::SquaredExpARD<Params>>(i, 100);
        check_kernel<kernel::SquaredExpARD<ParamsNoise>>(i, 100);
    }

    Params::kernel_squared_exp_ard::set_k(1);
    for (int i = 1; i <= 10; i++) {
        check_kernel<kernel::SquaredExpARD<Params>>(i, 100);
    }
}

BOOST_AUTO_TEST_CASE(test_kernel_SE_ARD)
{
    Params::kernel_squared_exp_ard::set_k(0);

    kernel::SquaredExpARD<Params> se(2);
    Eigen::VectorXd hp = Eigen::VectorXd::Zero(se.h_params_size());

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
    hp = Eigen::VectorXd::Zero(se.h_params_size());

    se.set_h_params(hp);
    BOOST_CHECK(s1 == se(v1, v2));
}
