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
#ifndef LIMBO_MEAN_FUNCTION_ARD_HPP
#define LIMBO_MEAN_FUNCTION_ARD_HPP

#include <Eigen/Core>

namespace limbo {
    namespace mean {

        /// Functor used to optimize the mean function using the maximum likelihood principle
        ///
        /// @see limbo::model::gp::KernelMeanLFOpt, limbo::model::gp::MeanLFOpt
        template <typename Params, typename MeanFunction>
        struct FunctionARD {
            FunctionARD(size_t dim_out = 1)
                : _mean_function(dim_out), _tr(dim_out, dim_out + 1)
            {
                Eigen::VectorXd h = Eigen::VectorXd::Zero(dim_out * (dim_out + 1));
                for (size_t i = 0; i < dim_out; i++)
                    h[i * (dim_out + 2)] = 1;
                this->set_h_params(h);
            }

            size_t h_params_size() const { return _tr.rows() * _tr.cols(); }

            const Eigen::VectorXd& h_params() const { return _h_params; }

            void set_h_params(const Eigen::VectorXd& p)
            {
                _h_params = p;
                for (int c = 0; c < _tr.cols(); c++)
                    for (int r = 0; r < _tr.rows(); r++)
                        _tr(r, c) = p[r * _tr.cols() + c];
            }

            template <typename GP>
            Eigen::MatrixXd grad(const Eigen::VectorXd& x, const GP& gp) const
            {
                Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(_tr.rows(), _h_params.size());
                Eigen::VectorXd m = _mean_function(x, gp);
                for (int i = 0; i < _tr.rows(); i++) {
                    grad.block(i, i * _tr.cols(), 1, _tr.cols() - 1) = m.transpose();
                    grad(i, (i + 1) * _tr.cols() - 1) = 1;
                }
                return grad;
            }

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
            {
                Eigen::VectorXd m = _mean_function(x, gp);
                Eigen::VectorXd m_1(m.size() + 1);
                m_1.head(m.size()) = m;
                m_1[m.size()] = 1;
                return _tr * m_1;
            }

        protected:
            MeanFunction _mean_function;
            Eigen::MatrixXd _tr;
            Eigen::VectorXd _h_params;
        };
    }
}

#endif
