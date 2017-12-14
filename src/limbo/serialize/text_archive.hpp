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

        class TextArchive {
        public:
            TextArchive(const std::string& dir_name) : _dir_name(dir_name),
                                                       _fmt(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "", "") {}

            /// write an Eigen::Matrix*
            void save(const Eigen::MatrixXd& v, const std::string& object_name)
            {
                _create_directory();
                std::ofstream ofs(fname(object_name).c_str());
                ofs << v.format(_fmt) << std::endl;
            }

            /// write a vector of Eigen::Vector*
            template <typename T>
            void save(const std::vector<T>& v, const std::string& object_name)
            {
                _create_directory();
                std::ofstream ofs(fname(object_name).c_str());
                for (auto& x : v) {
                    ofs << x.transpose().format(_fmt) << std::endl;
                }
            }

            /// load an Eigen matrix (or vector)
            template <typename M>
            void load(M& m, const std::string& object_name)
            {
                auto values = _load(object_name);
                m.resize(values.size(), values[0].size());
                for (size_t i = 0; i < values.size(); ++i)
                    for (size_t j = 0; j < values[i].size(); ++j)
                        m(i, j) = values[i][j];
            }

            /// load a vector of Eigen::Vector*
            template <typename V>
            void load(std::vector<V>& m_list, const std::string& object_name)
            {
                m_list.clear();
                auto values = _load(object_name);
                assert(!values.empty());
                for (size_t i = 0; i < values.size(); ++i) {
                    V v(values[i].size());
                    for (size_t j = 0; j < values[i].size(); ++j)
                        v(j) = values[i][j];
                    m_list.push_back(v);
                }
                assert(!m_list.empty());
            }

            std::string fname(const std::string& object_name) const
            {
                return _dir_name + "/" + object_name + ".dat";
            }

        protected:
            std::string _dir_name;
            Eigen::IOFormat _fmt;

            void _create_directory()
            {
                boost::filesystem::path my_path(_dir_name);
                boost::filesystem::create_directory(my_path);
            }

            std::vector<std::vector<double>> _load(const std::string& object_name)
            {
                std::ifstream ifs(fname(object_name).c_str());
                assert(ifs.good() && "file not found");
                std::string line;
                std::vector<std::vector<double>> v;
                while (std::getline(ifs, line)) {
                    std::stringstream line_stream(line);
                    std::string cell;
                    std::vector<double> line;
                    while (std::getline(line_stream, cell, ' '))
                        line.push_back(std::stod(cell));
                    v.push_back(line);
                }
                assert(!v.empty() && "empty file");
                return v;
            }
        };
    } // namespace serialize
} // namespace limbo