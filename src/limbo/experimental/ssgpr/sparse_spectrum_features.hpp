#ifndef LIMBO_EXPERIMENTAL_SSPGR_SSF_HPP
#define LIMBO_EXPERIMENTAL_SSPGR_SSF_HPP

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <limbo/tools/random_generator.hpp>

namespace limbo {

    namespace defaults {
        struct sparse_spectrum_features {
            BO_PARAM(int, nproj, 20);
            BO_PARAM(double, sigma_o, 0.1);
            BO_PARAM(bool, fixed, true);
        };
    }

    namespace experimental {

        template <typename Params>
        class SparseSpectrumFeatures {
        public:
            SparseSpectrumFeatures() : _sigma_o(Params::sparse_spectrum_features::sigma_o()), _nproj(Params::sparse_spectrum_features::nproj()), _n(-1), _fixed(Params::sparse_spectrum_features::fixed()) {}

            SparseSpectrumFeatures(int dim_in) : _sigma_o(Params::sparse_spectrum_features::sigma_o()), _nproj(Params::sparse_spectrum_features::nproj()), _n(dim_in), _fixed(Params::sparse_spectrum_features::fixed())
            {
                _l = Eigen::VectorXd::Ones(_n);
                reset();
            }

            void reset()
            {
                static thread_local tools::rgen_gauss_t rgen(0.0, 1.0);
                assert(_n > 0);

                // std::cout << "Setting random Wf" << std::endl;
                _Wf = Eigen::MatrixXd::Zero(_nproj, _n);
                // std::cout << "init" << std::endl;
                for (int i = 0; i < _nproj; i++)
                    for (int j = 0; j < _n; j++)
                        _Wf(i, j) = rgen.rand();
                // std::cout << "Wow" << std::endl;

                rescale();
            }

            void rescale()
            {
                assert(_l.size() > 0);

                _W = _Wf;
                Eigen::VectorXd inv = _l.array().inverse();
                for (int i = 0; i < _W.rows(); i++) {
                    // std::cout << _Wf.row(i).size() << " vs " << inv.size() << std::endl;
                    _W.row(i) = _Wf.row(i).array() * inv.transpose().array();
                }
            }

            Eigen::VectorXd params() const
            {
                assert(_n > 0);

                int size = _n + 1;
                if (!_fixed)
                    size += _Wf.size();
                Eigen::VectorXd p(size);
                p(0) = _sigma_o;
                p.segment(1, _n) = _l;

                if (!_fixed)
                    p.tail(_Wf.size()) = Eigen::VectorXd::Map(_Wf.data(), _Wf.size());

                return p;
            }

            void set_params(const Eigen::VectorXd& p)
            {
                assert(_n > 0);

                _sigma_o = p(0);
                _l = p.segment(1, _n);

                if (!_fixed) {
                    Eigen::VectorXd tmp = p.tail(_nproj * _n);
                    _Wf = Eigen::MatrixXd::Map(tmp.data(), _nproj, _n);
                }

                rescale();
            }

            int dim_out() const
            {
                return 2 * _nproj;
            }

            Eigen::VectorXd evaluate(const Eigen::VectorXd& x) const
            {
                assert(_W.size() > 0);

                Eigen::VectorXd tmp = (_W * x);
                double f = _sigma_o / std::sqrt(_nproj);

                Eigen::VectorXd res(2 * tmp.size());
                res.head(tmp.size()) = f * (tmp.array().cos());
                res.tail(tmp.size()) = f * (tmp.array().sin());

                return res;
            }

            // to-do gradient

        protected:
            double _sigma_o;
            Eigen::VectorXd _l;
            int _nproj, _n;
            bool _fixed;

            Eigen::MatrixXd _Wf, _W;
        };
    }
}

#endif