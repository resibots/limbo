#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

// Quick hack for definition of 'I' in <complex.h>
#undef I
#include <boost/filesystem.hpp>

#include <Eigen/Core>

namespace limbo {
    namespace serialize {

        class BinaryArchive {
        public:
            BinaryArchive(const std::string& dir_name) : _dir_name(dir_name) {}

            /// write an Eigen::Matrix*
            void save(const Eigen::MatrixXd& v, const std::string& object_name) const
            {
                _create_directory();

                std::ofstream out(fname(object_name).c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
                _write_binary(out, v);
                out.close();
            }

            /// write a vector of Eigen::Vector*
            template <typename T>
            void save(const std::vector<T>& v, const std::string& object_name) const
            {
                _create_directory();

                std::stringstream s;

                int size = v.size();
                s.write((char*)(&size), sizeof(int));
                for (auto& x : v) {
                    _write_binary(s, x);
                }

                std::ofstream out(fname(object_name).c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
                out << s.rdbuf();
                out.close();
            }

            /// load an Eigen matrix (or vector)
            template <typename M>
            void load(M& m, const std::string& object_name) const
            {
                std::ifstream in(fname(object_name).c_str(), std::ios::in | std::ios::binary);
                _read_binary(in, m);
                in.close();
            }

            /// load a vector of Eigen::Vector*
            template <typename V>
            void load(std::vector<V>& m_list, const std::string& object_name) const
            {
                m_list.clear();

                std::ifstream in(fname(object_name).c_str(), std::ios::in | std::ios::binary);

                int size;
                in.read((char*)(&size), sizeof(int));

                for (int i = 0; i < size; i++) {
                    V v;
                    _read_binary(in, v);
                    m_list.push_back(v);
                }
                in.close();
                assert(!m_list.empty());
            }

            std::string fname(const std::string& object_name) const
            {
                return _dir_name + "/" + object_name + ".bin";
            }

        protected:
            std::string _dir_name;

            void _create_directory() const
            {
                boost::filesystem::path my_path(_dir_name);
                boost::filesystem::create_directory(my_path);
            }

            template <class Matrix, class Stream>
            void _write_binary(Stream& out, const Matrix& matrix) const
            {
                typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
                out.write((char*)(&rows), sizeof(typename Matrix::Index));
                out.write((char*)(&cols), sizeof(typename Matrix::Index));
                out.write((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
            }

            template <class Matrix, class Stream>
            void _read_binary(Stream& in, Matrix& matrix) const
            {
                typename Matrix::Index rows = 0, cols = 0;
                in.read((char*)(&rows), sizeof(typename Matrix::Index));
                in.read((char*)(&cols), sizeof(typename Matrix::Index));
                matrix.resize(rows, cols);
                in.read((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
            }
        };
    } // namespace serialize
} // namespace limbo