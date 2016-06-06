#ifndef LIMBO_KERNEL_SQUARED_EXP_ARD_HPP
#define LIMBO_KERNEL_SQUARED_EXP_ARD_HPP

#include <Eigen/Core>

namespace limbo {
    namespace defaults {
        struct kernel_squared_exp_ard {
            BO_PARAM(int, k, 0); //equivalent to the standard exp ARD
            BO_PARAM(double, sigma_sq, 1);
        };
    }

    namespace kernel {
        /**
        @ingroup kernel
        \rst

        Squared exponential covariance function with automatic relevance detection (to be used with a likelihood optimizer)
        Computes the squared exponential covariance like this:

        .. math::
            k_{SE}(x, y) = \alpha^2 \exp \Big(-\frac{1}{2}(x-y)^TM(x-y)\Big),

	 with :math:`M = \Lambda\Lambda^T + diag(l_1^{-2}, \dots, l_n^{-2})` being the characteristic length scales and :math:`\alpha` describing the variability of the latent function. The parameters :math:`l_1^2, \dots, l_n^2, \alpha, \Lambda` are expected in this order in the parameter array.

        Reference: :cite:`Rasmussen2006`, p. 106 & :cite:`brochu2010tutorial`, p. 10
        \endrst
        */
        template <typename Params>
        struct SquaredExpARD {
            SquaredExpARD(int dim = 1) : _sf2(0), _ell(dim), _A(dim, Params::kernel_squared_exp_ard::k()), _input_dim(dim)
            {
                //assert(Params::SquaredExpARD::k()<dim);
                Eigen::VectorXd p = Eigen::VectorXd::Zero(_ell.size() + _ell.size() * Params::kernel_squared_exp_ard::k() + 1);
                p.head(_ell.size()) = Eigen::VectorXd::Ones(_ell.size()) * -1;
                this->set_h_params(p);
                _sf2 = Params::kernel_squared_exp_ard::sigma_sq();
            }

            size_t h_params_size() const { return _ell.size() + _ell.size() * Params::kernel_squared_exp_ard::k() + 1; }

            const Eigen::VectorXd& h_params() const { return _h_params; }

            void set_h_params(const Eigen::VectorXd& p)
            {
                _h_params = p;
                for (size_t i = 0; i < _input_dim; ++i)
                    _ell(i) = std::exp(p(i));
                for (size_t j = 0; j < (unsigned int)Params::kernel_squared_exp_ard::k(); ++j)
                    for (size_t i = 0; i < _input_dim; ++i)
                        _A(i, j) = p((j + 1) * _input_dim + i); //can be negative
                // _sf2 = 1; // exp(2 * p(p.size()-1));
            }

            Eigen::VectorXd grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                Eigen::VectorXd grad = Eigen::VectorXd::Zero(this->h_params_size());
                Eigen::MatrixXd K = (_A * _A.transpose());
                K.diagonal() += (Eigen::MatrixXd)(_ell.array().inverse().square());
                double z = ((x1 - x2).transpose() * K * (x1 - x2)).norm();
                double k = _sf2 * std::exp(-0.5 * z);

                grad.head(_input_dim) = (x1 - x2).cwiseQuotient(_ell).array().square() * k;
                Eigen::MatrixXd G = -k * (x1 - x2) * (x1 - x2).transpose() * _A;
                for (size_t j = 0; j < Params::kernel_squared_exp_ard::k(); ++j)
                    grad.segment((1 + j) * _input_dim, _input_dim) = G.col(j);

                grad(this->h_params_size() - 1) = 0; // 2.0 * k;
                return grad;
            }

            double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                assert(x1.size() == _ell.size());
                Eigen::MatrixXd K = (_A * _A.transpose());
                K.diagonal() += (Eigen::MatrixXd)(_ell.array().inverse().square());
                double z = ((x1 - x2).transpose() * K * (x1 - x2)).norm();
                return _sf2 * std::exp(-0.5 * z);
            }

            const Eigen::VectorXd& ell() const { return _ell; }

        protected:
            double _sf2;
            Eigen::VectorXd _ell;
            Eigen::MatrixXd _A;
            size_t _input_dim;
            Eigen::VectorXd _h_params;
        };
    }
}

#endif
