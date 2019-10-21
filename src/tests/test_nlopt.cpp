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
#define BOOST_TEST_MODULE test_nlopt

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>
#include <limbo/opt/nlopt_grad.hpp>
#include <limbo/opt/nlopt_no_grad.hpp>

using namespace limbo;

struct Params {
    struct opt_nloptgrad : public defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 200);
    };

    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_PARAM(int, iterations, 200);
    };
};

opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad)
{
    double v = -params(0) * params(0) - params(1) * params(1);
    if (!eval_grad)
        return opt::no_grad(v);
    Eigen::VectorXd grad(2);
    grad(0) = -2 * params(0);
    grad(1) = -2 * params(1);
    return {v, grad};
}

opt::eval_t my_constraint(const Eigen::VectorXd& params, bool eval_grad)
{
    double v = params(0) + 3. * params(1) - 10.;
    if (!eval_grad)
        return opt::no_grad(v);
    Eigen::VectorXd grad(2);
    grad(0) = 1.;
    grad(1) = 3.;
    return {v, grad};
}

opt::eval_t my_inequality_constraint(const Eigen::VectorXd& params, bool eval_grad)
{
    double v = -params(0) - 3. * params(1) + 10.;
    if (!eval_grad)
        return opt::no_grad(v);
    Eigen::VectorXd grad(2);
    grad(0) = -1.;
    grad(1) = -3.;
    return {v, grad};
}

BOOST_AUTO_TEST_CASE(test_nlopt_grad_simple)
{
    opt::NLOptGrad<Params, nlopt::LD_MMA> optimizer;
    Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}

BOOST_AUTO_TEST_CASE(test_nlopt_no_grad_simple)
{
    opt::NLOptNoGrad<Params, nlopt::LN_COBYLA> optimizer;
    Eigen::VectorXd best(2);
    best << 1, 1;
    size_t N = 10;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);
        if (g.norm() < best.norm()) {
            best = g;
        }
    }

    BOOST_CHECK_SMALL(best(0), 0.00000001);
    BOOST_CHECK_SMALL(best(1), 0.00000001);
}

BOOST_AUTO_TEST_CASE(test_nlopt_no_grad_constraint)
{
    opt::NLOptNoGrad<Params, nlopt::LN_COBYLA> optimizer;
    optimizer.initialize(2);
    optimizer.add_equality_constraint(my_constraint);

    Eigen::VectorXd best = tools::random_vector(2).array() * 50.; // some random big value
    Eigen::VectorXd target(2);
    target << 1., 3.;
    size_t N = 10;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);
        if ((g - target).norm() < (best - target).norm()) {
            best = g;
        }
    }

    BOOST_CHECK_SMALL(std::abs(1. - best(0)), 0.000001);
    BOOST_CHECK_SMALL(std::abs(3. - best(1)), 0.000001);
}

BOOST_AUTO_TEST_CASE(test_nlopt_grad_constraint)
{
    opt::NLOptGrad<Params, nlopt::LD_AUGLAG_EQ> optimizer;
    optimizer.initialize(2);
    optimizer.add_inequality_constraint(my_inequality_constraint);

    Eigen::VectorXd best = tools::random_vector(2).array() * 50.; // some random big value
    Eigen::VectorXd target(2);
    target << 1., 3.;
    size_t N = 10;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);
        if ((g - target).norm() < (best - target).norm()) {
            best = g;
        }
    }

    BOOST_CHECK_SMALL(std::abs(1. - best(0)), 0.0001);
    BOOST_CHECK_SMALL(std::abs(3. - best(1)), 0.0001);
}