#ifndef MEAN_FUNCTIONS_HPP_
#define MEAN_FUNCTIONS_HPP_

#include <fstream>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
namespace limbo {
    namespace mean_functions {
        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct NullFunction {
            NullFunction(size_t dim_out = 1) : _dim_out(dim_out) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return ObsType::Zero(_dim_out);
            }

        protected:
            size_t _dim_out;
        };

        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct MeanConstant {
            MeanConstant(size_t dim_out = 1) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Params::meanconstant::constant();
            }
        };

        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct MeanData {
            MeanData(size_t dim_out = 1) {}

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP& gp) const
            {
                return gp.mean_observation().array();
            }
        };

        template <typename Params, typename ObsType = Eigen::VectorXd>
        struct MeanArchive {
            MeanArchive(size_t dim_out = 1)
            {
                // create and open an archive for input
                std::ifstream ifs(Params::meanarchive::filename());
                assert(ifs.good());
                boost::archive::text_iarchive ia(ifs);
                // read class state from archive
                ia >> _archive;
                std::cout << _archive.size() << " elements loaded in the archive"
                          << std::endl;
            }

            template <typename GP>
            ObsType operator()(const Eigen::VectorXd& v, const GP&) const
            {
                std::vector<double> key(v.size(), 0);
                for (int i = 0; i < v.size(); i++)
                    key[i] = v[i];
                Eigen::VectorXd res(1);
                res(0) = _archive.at(key);
                return res;
            }

        protected:            
            struct classcomp {
                bool operator()(const std::vector<double>& lhs, const std::vector<double>& rhs) const
                {
                    assert(lhs.size() == 6 && rhs.size() == 6);
                    int i = 0;
                    while (i < 5 && lhs[i] == rhs[i])
                        i++;
                    return lhs[i] < rhs[i];
                }
            };

            std::map<std::vector<double>, double, classcomp> _archive;
        };

        template <typename Params, typename MeanFunction, typename ObsType = Eigen::VectorXd>
        struct MeanFunctionARD {
            MeanFunctionARD(size_t dim_out = 1)
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
            ObsType operator()(const Eigen::VectorXd& x, const GP& gp) const
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
