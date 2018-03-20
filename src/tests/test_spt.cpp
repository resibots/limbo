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
#define BOOST_TEST_MODULE test_spt

#include <boost/test/unit_test.hpp>

#include <limbo/spt/spt.hpp>
#include <limbo/tools/random_generator.hpp>

BOOST_AUTO_TEST_CASE(test_spt)
{
    using namespace limbo;
    // generate data
    std::vector<Eigen::VectorXd> samples, observations;
    size_t N = 200;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd s = tools::random_vector(2);
        samples.push_back(s);
        observations.push_back(s.array() * 2.);
    }

    // test with no overlap
    {
        // create tree
        auto root = spt::make_spt(samples, observations, 2, 0.);
        // get leaves
        auto leaves = spt::get_leaves(root);

        // check if points are split correctly
        size_t n = 0;
        for (size_t i = 0; i < leaves.size(); i++) {
            n += leaves[i]->points().size();
        }
        BOOST_CHECK(N == n);

        // check if no point overlaps other regions
        for (size_t i = 0; i < leaves.size(); i++) {
            auto points = leaves[i]->points();
            for (size_t j = 0; j < leaves.size(); j++) {
                if (i == j)
                    continue;
                for (auto& p : points) {
                    BOOST_CHECK(!spt::in_bounds(leaves[j], p));
                }
            }
        }
    }

    // test with overlap
    {
        // create tree
        auto root = spt::make_spt(samples, observations, 2, 0.1);
        // get leaves
        auto leaves = spt::get_leaves(root);

        // check if points are split correctly
        size_t n = 0;
        for (size_t i = 0; i < leaves.size(); i++) {
            n += leaves[i]->points().size();
        }
        BOOST_CHECK(n > N);

        // check if the overlapping points are as many as they should
        size_t overlapping = 0;
        for (size_t i = 0; i < leaves.size(); i++) {
            auto points = leaves[i]->points();
            for (size_t j = 0; j < leaves.size(); j++) {
                if (i == j)
                    continue;
                for (auto& p : points) {
                    if (spt::in_bounds(leaves[j], p))
                        overlapping++;
                }
            }
        }

        BOOST_CHECK(overlapping == (n - N));
    }
}
