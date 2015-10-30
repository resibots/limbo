#ifndef LIMBO_MEAN_FUNCTION_ARD_HPP
#define LIMBO_MEAN_FUNCTION_ARD_HPP

#include <Eigen/Core>

namespace limbo {
    namespace mean {
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
