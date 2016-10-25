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
#ifndef LIMBO_KERNEL_EXP_HPP
#define LIMBO_KERNEL_EXP_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct kernel_exp {
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 1);
        };
    }
    namespace kernel {
        /**
          @ingroup kernel
          \rst
          Exponential kernel (see :cite:`brochu2010tutorial` p. 9).

          .. math::
              k(v_1, v_2)  = \sigma^2\exp \Big(-\frac{1}{l^2} ||v_1 - v_2||^2\Big)

          Parameters:
            - ``double sigma_sq`` (signal variance)
            - ``double l`` (characteristic length scale)
          \endrst
        */
        template <typename Params>
        struct Exp {
            Exp(size_t dim = 1) {}
            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double _l = Params::kernel_exp::l();
                return Params::kernel_exp::sigma_sq() * (std::exp(-(1 / (2 * std::pow(_l, 2))) * std::pow((v1 - v2).norm(), 2)));
            }
        };
    }
}

#endif
