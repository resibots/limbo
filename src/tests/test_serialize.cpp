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
#define BOOST_TEST_MODULE test_serialize

#include <cstring>
#include <fstream>

#include <boost/test/unit_test.hpp>

#include <limbo/kernel/exp.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/mean/function_ard.hpp>
#include <limbo/mean/null_function.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/mean_lf_opt.hpp>
#include <limbo/model/multi_gp.hpp>
#include <limbo/serialize/binary_archive.hpp>
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

    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 1);
    };
};

// Different parameters in load to test
struct LoadParams {
    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 10.0);
        BO_PARAM(double, l, 1.);
    };
    struct kernel : public limbo::defaults::kernel {
    };
    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
        BO_PARAM(double, sigma_sq, 10.0);
    };
    struct opt_rprop : public limbo::defaults::opt_rprop {
    };

    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 2.);
        BO_PARAM(double, l, 0.1);
    };

    struct mean_constant {
        BO_PARAM(double, constant, -1);
    };
};

double get_diff(double a, double b)
{
    return std::abs(a - b);
}

double get_diff(const Eigen::VectorXd& a, const Eigen::VectorXd& b)
{
    return (a - b).norm();
}

template <typename GP, typename GPLoad, typename Archive>
void test_gp(const std::string& name, bool optimize_hp = true)
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
    GP gp(3, 1);
    gp.compute(samples, observations);
    if (optimize_hp)
        gp.optimize_hyperparams();

    // attempt to save
    Archive a1(name);
    gp.save(a1);
    // We can also save like this
    // gp.template save<Archive>(name);

    // attempt to load -- use only the name
    GPLoad gp2(3, 1);
    gp2.template load<Archive>(name);

    BOOST_CHECK_EQUAL(gp.nb_samples(), gp2.nb_samples());

    // check that the two GPs make the same predictions
    size_t k = 1000;
    for (size_t i = 0; i < k; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        auto v1 = gp.query(s);
        auto v2 = gp2.query(s);
        BOOST_CHECK_SMALL(get_diff(std::get<0>(v1), std::get<0>(v2)), 1e-10);
        BOOST_CHECK_SMALL(get_diff(std::get<1>(v1), std::get<1>(v2)), 1e-10);
    }

    // attempt to load without recomputing
    // and without knowing the dimensions
    GPLoad gp3;
    Archive a3(name);
    gp3.load(a3, false);

    BOOST_CHECK_EQUAL(gp.nb_samples(), gp3.nb_samples());

    // check that the two GPs make the same predictions
    for (size_t i = 0; i < k; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        auto v1 = gp.query(s);
        auto v2 = gp3.query(s);
        BOOST_CHECK_SMALL(get_diff(std::get<0>(v1), std::get<0>(v2)), 1e-10);
        BOOST_CHECK_SMALL(get_diff(std::get<1>(v1), std::get<1>(v2)), 1e-10);
    }
}

BOOST_AUTO_TEST_CASE(test_text_archive)
{
    test_gp<limbo::model::GPOpt<Params>, limbo::model::GPOpt<LoadParams>, limbo::serialize::TextArchive>("/tmp/gp_opt_text");
    test_gp<limbo::model::GPBasic<Params>, limbo::model::GPBasic<LoadParams>, limbo::serialize::TextArchive>("/tmp/gp_basic_text", false);

    using GPMean = limbo::model::GP<Params, limbo::kernel::MaternFiveHalves<Params>, limbo::mean::Constant<Params>, limbo::model::gp::MeanLFOpt<Params>>;
    using GPMeanLoad = limbo::model::GP<LoadParams, limbo::kernel::MaternFiveHalves<LoadParams>, limbo::mean::Constant<LoadParams>, limbo::model::gp::MeanLFOpt<LoadParams>>;
    test_gp<GPMean, GPMeanLoad, limbo::serialize::TextArchive>("/tmp/gp_mean_text");
}

BOOST_AUTO_TEST_CASE(test_bin_archive)
{
    test_gp<limbo::model::GPOpt<Params>, limbo::model::GPOpt<LoadParams>, limbo::serialize::BinaryArchive>("/tmp/gp_opt_bin");
    test_gp<limbo::model::GPBasic<Params>, limbo::model::GPBasic<LoadParams>, limbo::serialize::BinaryArchive>("/tmp/gp_basic_bin", false);

    using GPMean = limbo::model::GP<Params, limbo::kernel::MaternFiveHalves<Params>, limbo::mean::Constant<Params>, limbo::model::gp::MeanLFOpt<Params>>;
    using GPMeanLoad = limbo::model::GP<LoadParams, limbo::kernel::MaternFiveHalves<LoadParams>, limbo::mean::Constant<LoadParams>, limbo::model::gp::MeanLFOpt<LoadParams>>;
    test_gp<GPMean, GPMeanLoad, limbo::serialize::BinaryArchive>("/tmp/gp_mean_bin");
}

BOOST_AUTO_TEST_CASE(test_multi_gp_save)
{
    using GP_Multi_t = limbo::model::MultiGP<Params, limbo::model::GP, limbo::kernel::Exp<Params>, limbo::mean::NullFunction<Params>>;
    test_gp<GP_Multi_t, GP_Multi_t, limbo::serialize::TextArchive>("/tmp/gp_multi_text", false);
}
