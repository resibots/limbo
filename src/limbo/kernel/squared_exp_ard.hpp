#ifndef LIMBO_KERNEL_SQUARED_EXP_ARD_HPP
#define LIMBO_KERNEL_SQUARED_EXP_ARD_HPP

#include <Eigen/Core>

namespace limbo {
    namespace kernel {
        /**
        @ingroup kernel
        \rst

        Squared exponential covariance function with automatic relevance detection (to be used with a likelihood optimizer)
        Computes the squared exponential covariance like this:

        .. math::
            k_{SE}(x, y) = \alpha^2 \exp(-\frac{1}{2}(x-y)^T\Lambda^{-1}(x-y)),

        with :math:`\Lambda = diag(l_1^2, \dots, l_n^2)` being the characteristic length scales and :math:`\alpha` describing the variability of the latent function. The parameters :math:`l_1^2, \dots, l_n^2, \alpha` are expected in this order in the parameter array.

        \endrst
        */
        template <typename Params>
        struct SquaredExpARD {
            SquaredExpARD(int dim = 1) : _sf2(0), _ell(dim), _input_dim(dim)
            {
                this->set_h_params(Eigen::VectorXd::Ones(_ell.size() + 1) * -1);
            }

            size_t h_params_size() const { return _ell.size() + 1; }

            const Eigen::VectorXd& h_params() const { return _h_params; }

            void set_h_params(const Eigen::VectorXd& p)
            {
                _h_params = p;
                for (size_t i = 0; i < _input_dim; ++i)
                    _ell(i) = std::exp(p(i));
                _sf2 = 1; // exp(2 * p(_input_dim));
            }

            Eigen::VectorXd grad(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                Eigen::VectorXd grad(_input_dim + 1);
                Eigen::VectorXd z = (x1 - x2).cwiseQuotient(_ell).array().square();
                double k = _sf2 * std::exp(-0.5 * z.sum());
                grad.head(_input_dim) = z * k;
                grad(_input_dim) = 0; // 2.0 * k;
                return grad;
            }

            double operator()(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const
            {
                assert(x1.size() == _ell.size());
                double z = (x1 - x2).cwiseQuotient(_ell).squaredNorm();
                return _sf2 * std::exp(-0.5 * z);
            }

            const Eigen::VectorXd& ell() const { return _ell; }

        protected:
            double _sf2;
            Eigen::VectorXd _ell;
            size_t _input_dim;
            Eigen::VectorXd _h_params;
        };
    }
}

#endif
