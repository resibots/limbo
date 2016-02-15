

//| This file is a part of the sferes2 framework.
//| Copyright 2009, ISIR / Universite Pierre et Marie Curie (UPMC)
//| Main contributor(s): Jean-Baptiste Mouret, mouret@isir.fr
//|
//| This software is a computer program whose purpose is to facilitate
//| experiments in evolutionary computation and evolutionary robotics.
//|
//| This software is governed by the CeCILL license under French law
//| and abiding by the rules of distribution of free software.  You
//| can use, modify and/ or redistribute the software under the terms
//| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
//| following URL "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and rights to
//| copy, modify and redistribute granted by the license, users are
//| provided only with a limited warranty and the software's author,
//| the holder of the economic rights, and the successive licensors
//| have only limited liability.
//|
//| In this respect, the user's attention is drawn to the risks
//| associated with loading, using, modifying and/or developing or
//| reproducing the software by the user in light of its specific
//| status of free software, that may mean that it is complicated to
//| manipulate, and that also therefore means that it is reserved for
//| developers and experienced professionals having in-depth computer
//| knowledge. Users are therefore encouraged to load and test the
//| software's suitability as regards their requirements in conditions
//| enabling the security of their systems and/or data to be ensured
//| and, more generally, to use and operate it in the same conditions
//| as regards security.
//|
//| The fact that you are presently reading this means that you have
//| had knowledge of the CeCILL license and that you accept its terms.

#ifndef LIMBO_TOOLS_MATH_HPP
#define LIMBO_TOOLS_MATH_HPP

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <list>
#include <stdlib.h>
#include <random>
#include <utility>
#include <mutex>

namespace limbo {
    namespace tools {
        // get sign of number
        template <typename T>
        inline constexpr int signum(T x, std::false_type is_signed)
        {
            return T(0) < x;
        }

        template <typename T>
        inline constexpr int signum(T x, std::true_type is_signed)
        {
            return (T(0) < x) - (x < T(0));
        }

        template <typename T>
        inline constexpr int signum(T x)
        {
            return signum(x, std::is_signed<T>());
        }

        // inf or NAN checks
        template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
        inline bool is_nan_or_inf(T v)
        {
            return std::isinf(v) || std::isnan(v);
        }

        template <typename T, typename std::enable_if<!std::is_arithmetic<T>::value, int>::type = 0>
        inline bool is_nan_or_inf(const T& v)
        {
            for (int i = 0; i < v.size(); ++i)
                if (std::isinf(v(i)) || std::isnan(v(i)))
                    return true;
            return false;
        }
    }
}

#endif
