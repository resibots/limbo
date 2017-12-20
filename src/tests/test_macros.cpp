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
#define BOOST_TEST_MODULE test_macros

#include <cstring>

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>

struct Params {
    struct test {
        BO_PARAM(double, a, 1);
        BO_DYN_PARAM(int, b);
        BO_PARAM_ARRAY(double, c, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        BO_PARAM_VECTOR(double, d, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        BO_PARAM_STRING(e, "e");
    };
};

BO_DECLARE_DYN_PARAM(int, Params::test, b);

BOOST_AUTO_TEST_CASE(test_macros)
{
    BOOST_CHECK(Params::test::a() == 1.0);

    Params::test::set_b(2);
    BOOST_CHECK(Params::test::b() == 2);
    Params::test::set_b(3);
    BOOST_CHECK(Params::test::b() == 3);

    BOOST_CHECK(__VA_NARG__(1) == 1);
    BOOST_CHECK(__VA_NARG__(10, 11, 12, 13) == 4);

    BOOST_CHECK(Params::test::c_size() == 6);
    BOOST_CHECK(Params::test::c(0) == 1.0);
    BOOST_CHECK(Params::test::c(1) == 2.0);
    BOOST_CHECK(Params::test::c(2) == 3.0);
    BOOST_CHECK(Params::test::c(3) == 4.0);
    BOOST_CHECK(Params::test::c(4) == 5.0);
    BOOST_CHECK(Params::test::c(5) == 6.0);

    BOOST_CHECK(Params::test::d().size() == 6);
    BOOST_CHECK(Params::test::d()(0) == 1.0);
    BOOST_CHECK(Params::test::d()(1) == 2.0);
    BOOST_CHECK(Params::test::d()(2) == 3.0);
    BOOST_CHECK(Params::test::d()(3) == 4.0);
    BOOST_CHECK(Params::test::d()(4) == 5.0);
    BOOST_CHECK(Params::test::d()(5) == 6.0);

    BOOST_CHECK(strcmp(Params::test::e(), "e") == 0);
}
