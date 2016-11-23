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

#ifndef LIMBO_TOOLS_RANDOM_GENERATOR_HPP
#define LIMBO_TOOLS_RANDOM_GENERATOR_HPP

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <list>
#include <stdlib.h>
#include <random>
#include <utility>
#include <mutex>
#include <limbo/tools/rand_utils.hpp>

namespace limbo {
    namespace tools {
        /// @ingroup tools
        /// a mt19937-based random generator (mutex-protected)
        ///
        /// usage :
        /// - RandomGenerator<dist<double>>(0.0, 1.0);
        /// - double r = rgen.rand();
        template <typename D>
        class RandomGenerator {
        public:
            using result_type = typename D::result_type;
            RandomGenerator(result_type a, result_type b) : _dist(a, b), _rgen(randutils::auto_seed_128{}.base()) {}
            result_type rand()
            {
                std::lock_guard<std::mutex> lock(_mutex);
                return _dist(_rgen);
            }

        private:
            D _dist;
            std::mt19937 _rgen;
            std::mutex _mutex;
        };

        /// @ingroup tools
        using rdist_double_t = std::uniform_real_distribution<double>;
        /// @ingroup tools
        using rdist_int_t = std::uniform_int_distribution<int>;
        /// @ingroup tools
        using rdist_gauss_t = std::normal_distribution<>;

        /// @ingroup tools
        /// Double random number generator
        using rgen_double_t = RandomGenerator<rdist_double_t>;

        /// @ingroup tools
        /// Double random number generator (gaussian)
        using rgen_gauss_t = RandomGenerator<rdist_gauss_t>;

        ///@ingroup tools
        ///integer random number generator
        using rgen_int_t = RandomGenerator<rdist_int_t>;

        /// @ingroup tools
        /// random vector in [0, 1]
        ///
        /// - this function is thread safe because the random number generator we use is thread-safe
        /// - we use a C++11 random number generator
        Eigen::VectorXd random_vector_bounded(int size)
        {
            static rgen_double_t rgen(0.0, 1.0);
            Eigen::VectorXd res(size);
            for (int i = 0; i < size; ++i)
                res[i] = rgen.rand();
            return res;
        }

        /// @ingroup tools
        /// random vector in R
        ///
        /// - this function is thread safe because the random number generator we use is thread-safe
        /// - we use a C++11 random number generator
        Eigen::VectorXd random_vector_unbounded(int size)
        {
            static rgen_gauss_t rgen(0.0, 10.0);
            Eigen::VectorXd res(size);
            for (int i = 0; i < size; ++i)
                res[i] = rgen.rand();
            return res;
        }

        /// @ingroup tools
        /// random vector wrapper for both bounded and unbounded versions
        Eigen::VectorXd random_vector(int size, bool bounded = true)
        {
            if (bounded)
                return random_vector_bounded(size);
            return random_vector_unbounded(size);
        }
    }
}

#endif
