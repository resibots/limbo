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
#define BOOST_TEST_MODULE test_mean

#include <iostream>

#include <boost/test/unit_test.hpp>

#include <limbo/mean/constant.hpp>
#include <limbo/mean/function_ard.hpp>
#include <limbo/mean/null_function.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

using namespace limbo;
struct Params {
    struct mean_constant {
        BO_PARAM(double, constant, 1);
    };
};

// Check gradient via finite differences method
template <typename Mean>
std::tuple<double, Eigen::MatrixXd, Eigen::MatrixXd> check_grad(const Mean& mean, const Eigen::VectorXd& x, const Eigen::VectorXd& v, double e = 1e-4)
{
    Eigen::MatrixXd analytic_result, finite_diff_result;
    Mean me = mean;
    me.set_h_params(x);

    analytic_result = me.grad(v, v);

    finite_diff_result = Eigen::MatrixXd::Zero(me(v, v).size(), x.size());
    for (int j = 0; j < x.size(); j++) {
        Eigen::VectorXd test1 = x, test2 = x;
        test1[j] -= e;
        test2[j] += e;
        Mean m1 = mean;
        m1.set_h_params(test1);
        Mean m2 = mean;
        m2.set_h_params(test2);
        Eigen::VectorXd res1 = m1(v, v);
        Eigen::VectorXd res2 = m2(v, v);
        finite_diff_result.col(j) = (res2 - res1).array() / (2.0 * e);
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

template <typename Mean>
void check_mean(size_t dim_in, size_t dim_out, size_t K)
{
    Mean mean(dim_out);

    for (size_t i = 0; i < K; i++) {
        Eigen::VectorXd hp = tools::random_vector(mean.h_params_size()).array() * 10. - 5.;

        double error;
        Eigen::MatrixXd analytic, finite_diff;

        Eigen::VectorXd v = tools::random_vector(dim_in).array() * 10. - 5.;

        std::tie(error, analytic, finite_diff) = check_grad(mean, hp, v);
        // std::cout << error << ": " << analytic << " vs " << finite_diff << std::endl;
        BOOST_CHECK(error < 1e-6);
    }
}

BOOST_AUTO_TEST_CASE(test_mean_constant)
{
    for (int k = 1; k <= 10; k++) {
        for (int i = 1; i <= 10; i++) {
            check_mean<mean::Constant<Params>>(k, i, 100);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_mean_function_ard)
{
    // This test checks the gradients computation of FunctionARD when the base mean function
    // also has tunable parameters
    for (int k = 1; k <= 10; k++) {
        for (int i = 1; i <= 10; i++) {
            check_mean<mean::FunctionARD<Params, mean::Constant<Params>>>(k, i, 100);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_mean_function_ard_dummy)
{
    // This test checks the gradients computation of FunctionARD when the base mean function
    // has no tunable parameters
    for (int k = 1; k <= 10; k++) {
        for (int i = 1; i <= 10; i++) {
            check_mean<mean::FunctionARD<Params, mean::NullFunction<Params>>>(k, i, 100);
        }
    }
}
