#include <iostream>
#include <limbo/limbo.hpp>
#include <chrono>
#include "spt.hpp"

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> gaussian(m, v);
    return gaussian(gen);
}

// struct BlockStruct {
// public:
//     int _r, _c, _sr, _sc;
//
//     BlockStruct(int r, int c, int sr, int sc) : _r(r), _c(c), _sr(sr), _sc(sc) {}
//
//     Eigen::Block<Eigen::MatrixXd> get_block(Eigen::MatrixXd& m)
//     {
//         return Eigen::Block<Eigen::MatrixXd>(m.derived(), _r, _c, _sr, _sc);
//     }
// };
//
// template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
// class STGP {
// public:
//     /// useful because the model might be created before knowing anything about the process
//     STGP() : _dim_in(-1), _dim_out(-1), _N(200) {}
//
//     /// useful because the model might be created  before having samples
//     STGP(int dim_in, int dim_out)
//         : _dim_in(dim_in), _dim_out(dim_out), _N(200), _kernel_function(dim_in), _mean_function(dim_out) {}
//
//     /// Compute the GP from samples and observations. This call needs to be explicit!
//     void compute(const std::vector<Eigen::VectorXd>& samples,
//         const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = true)
//     {
//         assert(samples.size() != 0);
//         assert(observations.size() != 0);
//         assert(samples.size() == observations.size());
//
//         if (_dim_in != samples[0].size()) {
//             _dim_in = samples[0].size();
//             _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low
//         }
//
//         if (_dim_out != observations[0].size()) {
//             _dim_out = observations[0].size();
//             _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low
//         }
//
//         _samples = samples;
//
//         // _observations.resize(observations.size(), _dim_out);
//         // for (int i = 0; i < _observations.rows(); ++i)
//         //     _observations.row(i) = observations[i];
//         //
//         // _mean_observation = _observations.colwise().mean();
//
//         _N = 200;
//         int B = 7;
//         int N = _samples.size();
//         int d = std::ceil(std::log(N / double(_N)) / std::log(2.0));
//
//         _root = spt::make_spt(_samples, observations, d);
//         _leaves = spt::get_leaves(_root);
//         // std::cout << "Regions: " << _leaves.size() << std::endl;
//         _boundaries.clear();
//
//         _observations.resize(observations.size(), _dim_out);
//         int r = 0;
//         for (size_t i = 0; i < _leaves.size(); i++) {
//             for (size_t j = 0; j < _leaves[i]->observations().size(); j++) {
//                 _observations.row(r) = _leaves[i]->observations()[j];
//                 r++;
//             }
//         }
//
//         _mean_observation = _observations.colwise().mean();
//
//         int s = 0;
//         for (size_t i = 0; i < _leaves.size(); i++) {
//             // std::cout << _leaves[i]->points().size() << ((i < _leaves.size() - 1) ? ", " : "");
//             for (int k = 0; k < _leaves[i]->points().size(); k++)
//                 _samples[s++] = _leaves[i]->points()[k];
//             for (size_t j = i + 1; j < _leaves.size(); j++) {
//                 std::vector<Eigen::VectorXd> bp = spt::get_shared_boundaries(_leaves[i], _leaves[j]);
//                 limbo::tools::rgen_int_t rgen(0.0, bp.size() - 1);
//                 // std::cout << i + 1 << ", " << j + 1 << ": " << bp.size() << std::endl;
//                 // ofs02 << bp.size() << std::endl;
//                 if (bp.size() > 0) {
//                     std::vector<Eigen::VectorXd> new_bp;
//                     for (int i = 0; i < B; i++)
//                         new_bp.push_back(bp[rgen.rand()]);
//                     _boundaries.push_back(new_bp);
//                 }
//                 else
//                     _boundaries.push_back(std::vector<Eigen::VectorXd>());
//             }
//         }
//         // std::cout << std::endl;
//         // std::cout << "Boundaries: " << _boundaries.size() << std::endl;
//
//         // // Query point -- just for testing
//         // Eigen::VectorXd v = Eigen::VectorXd::Zero(2);
//         // // query GP
//         // int kk = 0;
//
//         int num = 0;
//
//         // Eigen::VectorXd c_D_star;
//         // for (int i = 0; i < _leaves.size(); i++) {
//         //     // std::cout << _leaves[i]->points().size() << std::endl;
//         //     Eigen::VectorXd k(_leaves[i]->points().size());
//         //     // TO-DO: Fix kernels
//         //     for (int j = 0; j < k.size(); j++)
//         //         k[j] = _kernel_function(_leaves[i]->points()[j], v);
//         //     // c_d_star.push_back(k);
//         //     c_D_star.conservativeResize(c_D_star.size() + k.size());
//         //     c_D_star.tail(k.size()) = k;
//         //     // std::cout << k.size() << std::endl;
//         //     // num += k.size();
//         // }
//         // std::cout << "Whole cD*(k): " << c_D_star.size() << "x1" << std::endl;
//         // // std::cout << c_D_star.transpose() << std::endl;
//
//         int b = 0;
//         // // std::vector<Eigen::VectorXd> c_delta_star;
//         // Eigen::VectorXd c_delta_star(B * _boundaries.size());
//         // for (size_t i = 0; i < _leaves.size(); i++) {
//         //     for (size_t j = i + 1; j < _leaves.size(); j++) {
//         //         if (_boundaries[b].size() == 0) {
//         //             // c_delta_star.push_back(Eigen::VectorXd::Zero(B));
//         //             c_delta_star.segment(b * B, B) = Eigen::VectorXd::Zero(B);
//         //             // std::cout << B << std::endl;
//         //         }
//         //         else {
//         //             Eigen::VectorXd k(_boundaries[b].size());
//         //             // TO-DO: Fix kernels
//         //             for (int kkk = 0; kkk < k.size(); kkk++) {
//         //                 if (kk == i)
//         //                     k[kkk] = _kernel_function(_boundaries[b][kkk], v);
//         //                 else if (kk == j)
//         //                     k[kkk] = -_kernel_function(_boundaries[b][kkk], v);
//         //                 else
//         //                     k[kkk] = 0.0;
//         //             }
//         //             // c_delta_star.push_back(k);
//         //             c_delta_star.segment(b * B, B) = k;
//         //             // std::cout << k.size() << std::endl;
//         //         }
//         //         b++;
//         //     }
//         // }
//         // std::cout << "Whole δ*(k): " << c_delta_star.size() << "x1" << std::endl;
//         // // std::cout << c_delta_star.transpose() << std::endl;
//
//         int num1 = 0;
//         int num2 = 0;
//         b = 0;
//         // std::vector<Eigen::MatrixXd> c_D_D;
//         // auto start = std::chrono::high_resolution_clock::now();
//         _DD_blocks.clear();
//         Eigen::MatrixXd c_D_D;
//         for (size_t i = 0; i < _leaves.size(); i++) {
//             // for (size_t j = i + 1; j < _leaves.size(); j++) {
//             std::vector<Eigen::VectorXd> points = _leaves[i]->points();
//             Eigen::MatrixXd DD(_leaves[i]->points().size(), _leaves[i]->points().size());
//             for (int k1 = 0; k1 < _leaves[i]->points().size(); k1++) {
//                 for (int k2 = 0; k2 <= k1; k2++) {
//                     // TO-DO: Fix kernels
//                     DD(k1, k2) = _kernel_function(points[k1], points[k2]);
//                 }
//             }
//
//             for (int k1 = 0; k1 < _leaves[i]->points().size(); k1++)
//                 for (int k2 = 0; k2 < k1; k2++)
//                     DD(k2, k1) = DD(k1, k2);
//
//             // c_D_D.push_back(DD);
//             int r = c_D_D.rows(), c = c_D_D.cols();
//             c_D_D.conservativeResizeLike(Eigen::MatrixXd::Zero(c_D_D.rows() + DD.rows(), c_D_D.cols() + DD.cols()));
//             c_D_D.block(r, c, DD.rows(), DD.cols()) = DD;
//             // blocks.push_back(c_D_D.block(r, c, DD.rows(), DD.cols()));
//             _DD_blocks.push_back({r, c, DD.rows(), DD.cols()});
//             // std::cout << DD.rows() << " " << DD.cols() << std::endl;
//             // num1 += DD.rows();
//             // num2 += DD.cols();
//             b++;
//             // }
//         }
//         // auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
//         // std::cout << "Time cDD: " << time1 / double(1000.0) << std::endl;
//         // std::cout << "Whole cDD: " << c_D_D.rows() << "x" << c_D_D.cols() << std::endl;
//         // std::cout << c_D_D << std::endl;
//         // Eigen::Block<Eigen::MatrixXd> block = _DD_blocks[0].get_block(c_D_D);
//         // for (int i = 0; i < block.rows(); i++)
//         //     for (int j = 0; j < block.cols(); j++)
//         //         block(i, j) = 4;
//         // block = block.inverse();
//         // std::cout << c_D_D << std::endl;
//
//         // TO-DO: Fix the inversion (use parallel if possible and proper inversion with cholesky)
//         _c_D_D_inverse = c_D_D;
//         for (int i = 0; i < _DD_blocks.size(); i++) {
//             Eigen::Block<Eigen::MatrixXd> block = _DD_blocks[i].get_block(_c_D_D_inverse);
//             // block = block.inverse();
//             block = block.llt().solve(Eigen::MatrixXd::Identity(block.rows(), block.cols()));
//         }
//
//         b = 0;
//         // num1 = 0;
//         // num2 = 0;
//         // std::vector<Eigen::MatrixXd> c_d_D;
//         // auto start3 = std::chrono::high_resolution_clock::now();
//         _c_d_D = Eigen::MatrixXd::Zero(_boundaries.size() * B, _samples.size());
//         for (size_t i = 0; i < _leaves.size(); i++) {
//             for (size_t j = i + 1; j < _leaves.size(); j++) {
//                 // for (int b = 0; b < _boundaries.size(); b++) {
//                 num = 0;
//                 for (int k = 0; k < _leaves.size(); k++) {
//                     std::vector<Eigen::VectorXd> points = _leaves[k]->points();
//                     Eigen::MatrixXd DD = Eigen::MatrixXd::Zero(B, _leaves[k]->points().size());
//                     if (_boundaries[b].size() > 0) {
//                         //     // for (int kkk = 0; kkk < B; kkk++)
//                         //     DD.block(0, 0, B, DD.cols()) = Eigen::MatrixXd::Zero(1, DD.cols());
//                         //     // DD(kkk, s) = 0.0;
//                         // }
//                         // else {
//                         for (int s = 0; s < _leaves[k]->points().size(); s++) {
//                             // TO-DO: Fix kernels
//                             for (int kkk = 0; kkk < B; kkk++) {
//                                 if (k == i)
//                                     DD(kkk, s) = _kernel_function(_boundaries[b][kkk], points[s]);
//                                 else if (k == j)
//                                     DD(kkk, s) = -_kernel_function(_boundaries[b][kkk], points[s]);
//                                 else
//                                     DD(kkk, s) = 0.0;
//                             }
//                         }
//                     }
//                     // c_d_D.push_back(DD);
//                     // std::cout << b* B << "," << num << " -> " << B << "x" << _leaves[k]->points().size() << std::endl;
//                     _c_d_D.block(b * B, num, B, _leaves[k]->points().size()) = DD;
//                     num += _leaves[k]->points().size();
//                 }
//                 // num2 = num; // / _boundaries.size();
//                 // num1 += B;
//                 b++;
//             }
//         }
//         // auto time3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start3).count();
//         // std::cout << "Time cδD: " << time3 / double(1000.0) << std::endl;
//         // std::cout << "Whole cδD: " << _c_d_D.rows() << "x" << _c_d_D.cols() << std::endl;
//         // std::cout << c_d_D << std::endl;
//
//         // std::vector<Eigen::MatrixXd> c_d_d;
//         _dd_blocks.clear();
//         Eigen::MatrixXd c_d_d = Eigen::MatrixXd::Zero(_boundaries.size() * B, _boundaries.size() * B);
//         // num1 = 0;
//         // num2 = 0;
//         int b1 = 0, b2 = 0;
//         // auto start2 = std::chrono::high_resolution_clock::now();
//         // for (int b1 = 0; b1 < _boundaries.size(); b1++)
//         for (size_t i1 = 0; i1 < _leaves.size(); i1++) {
//             for (size_t j1 = i1 + 1; j1 < _leaves.size(); j1++) {
//                 // for (int b2 = 0; b2 < _boundaries.size(); b2++)
//                 b2 = 0;
//                 for (size_t i2 = 0; i2 < _leaves.size(); i2++) {
//                     for (size_t j2 = i2 + 1; j2 < _leaves.size(); j2++) {
//                         // std::cout << b1 << " " << b2 << ": " << i1 << ", " << j1 << " -> " << i2 << ", " << j2 << std::endl;
//                         Eigen::MatrixXd DD(B, B);
//                         for (int k1 = 0; k1 < _boundaries[b1].size(); k1++) {
//                             for (int k2 = 0; k2 < _boundaries[b2].size(); k2++) {
//                                 // TO-DO: fix kernels
//                                 if (i1 == i2 && j1 == j2)
//                                     DD(k1, k2) = 2.0 * _kernel_function(_boundaries[b1][k1], _boundaries[b2][k2]);
//                                 if (i1 == i2 || j1 == j2)
//                                     DD(k1, k2) = _kernel_function(_boundaries[b1][k1], _boundaries[b2][k2]);
//                                 else if (i1 == j2 || i2 == j1)
//                                     DD(k1, k2) = -_kernel_function(_boundaries[b1][k1], _boundaries[b2][k2]);
//                                 else
//                                     DD(k1, k2) = 0.0;
//                             }
//                         }
//
//                         // c_d_d.push_back(DD);
//                         c_d_d.block(b1 * B, b2 * B, B, B) = DD;
//                         _dd_blocks.push_back({b1 * B, b2 * B, B, B});
//                         b2++;
//                     }
//                     // num1 += B;
//                     // num2 += B;
//                 }
//                 b1++;
//             }
//         }
//         // auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start2).count();
//         // std::cout << "Time cdd: " << time2 / double(1000.0) << std::endl;
//
//         // std::cout << "Whole cδδ: " << c_d_d.rows() << "x" << c_d_d.cols() << std::endl;
//         // std::cout << c_d_d << std::endl;
//         // Eigen::Block<Eigen::MatrixXd> block = _dd_blocks[0].get_block(c_d_d);
//         // // for (int i = 0; i < block.rows(); i++)
//         // //     for (int j = 0; j < block.cols(); j++)
//         // //         block(i, j) = 4;
//         // block = block.inverse();
//
//         // TO-DO: Fix the inversion (use parallel if possible and proper inversion with cholesky)
//         _c_d_d_inverse = c_d_d;
//         for (int i = 0; i < _dd_blocks.size(); i++) {
//             Eigen::Block<Eigen::MatrixXd> block = _dd_blocks[i].get_block(_c_d_d_inverse);
//             // block = block.inverse();
//             block = block.llt().solve(Eigen::MatrixXd::Identity(block.rows(), block.cols()));
//         }
//
//         this->_compute_obs_mean();
//     }
//
// protected:
//     int _dim_in;
//     int _dim_out;
//     int _N;
//
//     std::shared_ptr<spt::SPTNode> _root;
//     std::vector<std::shared_ptr<spt::SPTNode>> _leaves;
//     std::vector<std::vector<Eigen::VectorXd>> _boundaries;
//     std::vector<BlockStruct> _DD_blocks, _dd_blocks;
//
//     KernelFunction _kernel_function;
//     MeanFunction _mean_function;
//
//     std::vector<Eigen::VectorXd> _samples;
//     Eigen::MatrixXd _observations;
//     Eigen::MatrixXd _mean_vector;
//     Eigen::MatrixXd _obs_mean;
//
//     Eigen::VectorXd _mean_observation;
//
//     Eigen::MatrixXd _c_d_D;
//     Eigen::MatrixXd _c_d_d_inverse;
//     Eigen::MatrixXd _c_D_D_inverse;
//     Eigen::MatrixXd _matrixL;
//
//     HyperParamsOptimizer _hp_optimize;
//
//     void _compute_obs_mean()
//     {
//         _mean_vector.resize(_samples.size(), _dim_out);
//         for (int i = 0; i < _mean_vector.rows(); i++)
//             _mean_vector.row(i) = _mean_function(_samples[i], *this);
//         _obs_mean = _observations - _mean_vector;
//     }
// };

// namespace Eigen {
//     template <class Matrix>
//     void write_binary(const std::string filename, const Matrix& matrix)
//     {
//         write_binary(filename.c_str(), matrix);
//     }
//     template <class Matrix>
//     void write_binary(const char* filename, const Matrix& matrix)
//     {
//         std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
//         typename Matrix::Index rows = matrix.rows(), cols = matrix.cols();
//         out.write((char*)(&rows), sizeof(typename Matrix::Index));
//         out.write((char*)(&cols), sizeof(typename Matrix::Index));
//         out.write((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
//         out.close();
//     }
//     template <class Matrix>
//     void read_binary(const std::string filename, Matrix& matrix)
//     {
//         read_binary(filename.c_str(), matrix);
//     }
//     template <class Matrix>
//     void read_binary(const char* filename, Matrix& matrix)
//     {
//         std::ifstream in(filename, std::ios::in | std::ios::binary);
//         typename Matrix::Index rows = 0, cols = 0;
//         in.read((char*)(&rows), sizeof(typename Matrix::Index));
//         in.read((char*)(&cols), sizeof(typename Matrix::Index));
//         matrix.resize(rows, cols);
//         in.read((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
//         in.close();
//     }
//     MatrixXd colwise_sig(const MatrixXd& matrix)
//     {
//         VectorXd matrix_mean = matrix.colwise().mean();
//         MatrixXd matrix_std = (matrix - matrix_mean.transpose().replicate(matrix.rows(), 1));
//         matrix_std = matrix_std.array().pow(2);
//         MatrixXd matrix_sum = matrix_std.colwise().sum();
//         matrix_sum *= (1.0 / (matrix.rows() - 1));
//         return matrix_sum.array().sqrt();
//     }
//     double percentile_v(VectorXd vector, int p)
//     {
//         p = p - 1;
//         if (p < 0)
//             p = 0;
//         std::sort(vector.data(), vector.data() + vector.size());
//         return vector(std::floor((p / 100.0) * vector.size()));
//     }
//     VectorXd percentile(const MatrixXd& matrix, int p)
//     {
//         VectorXd result(matrix.cols());
//         for (int i = 0; i < matrix.cols(); i++) {
//             result(i) = percentile_v(matrix.col(i), p);
//         }
//         return result;
//     }
// }
//
// ///optimize the likelihood of the kernel only
// template <typename Params, typename Optimizer = limbo::opt::ParallelRepeater<Params, limbo::opt::Rprop<Params>>>
// struct KernelLFOpt : public limbo::model::gp::HPOpt<Params, Optimizer> {
// public:
//     template <typename GP>
//     void operator()(GP& gp)
//     {
//         this->_called = true;
//         KernelLFOptimization<GP> optimization(gp);
//         Optimizer optimizer;
//         Eigen::VectorXd params = optimizer(optimization, gp.kernel_function().h_params(), false);
//         gp.kernel_function().set_h_params(params);
//         gp.set_lik(limbo::opt::eval(optimization, params));
//         gp.recompute(false);
//         std::cout << "likelihood: " << gp.get_lik() << std::endl;
//     }
//
// protected:
//     template <typename GP>
//     struct KernelLFOptimization {
//     public:
//         KernelLFOptimization(const GP& gp) : _original_gp(gp) {}
//         Eigen::MatrixXd _to_matrix(const std::vector<Eigen::VectorXd>& xs) const
//         {
//             Eigen::MatrixXd result(xs.size(), xs[0].size());
//             for (size_t i = 0; i < (size_t)result.rows(); ++i) {
//                 result.row(i) = xs[i];
//             }
//             return result;
//         }
//         Eigen::MatrixXd _to_matrix(std::vector<Eigen::VectorXd>& xs) const { return _to_matrix(xs); }
//         limbo::opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
//         {
//             GP gp(this->_original_gp);
//             gp.kernel_function().set_h_params(params);
//             gp.recompute(false);
//             size_t n = gp.obs_mean().rows();
//             // --- cholesky ---
//             // see:
//             // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
//             Eigen::MatrixXd l = gp.matrixL();
//             long double det = 2 * l.diagonal().array().log().sum();
//             double a = (gp.obs_mean().transpose() * gp.alpha())
//                            .trace(); // generalization for multi dimensional observation
//             // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
//             double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);
//             // sum(((ll - log(curb.std'))./log(curb.ls)).^p);
//             Eigen::VectorXd p = gp.kernel_function().h_params();
//             Eigen::VectorXd ll = p.segment(0, p.size() - 2); // length scales
//             // Std calculation of samples in logspace
//             Eigen::MatrixXd samples = _to_matrix(gp.samples());
//             Eigen::MatrixXd samples_std = Eigen::colwise_sig(samples).array().log();
//             double snr = std::log(500); // signal to noise threshold
//             double ls = std::log(100); // length scales threshold
//             size_t pp = 30; // penalty power
//             lik -= ((ll - samples_std.transpose()) / ls).array().pow(pp).sum();
//             // f = f + sum(((lsf - lsn)/log(curb.snr)).^p); % signal to noise ratio
//             double lsf = p(p.size() - 2);
//             double lsn = p(p.size() - 1); //std::log(0.01);
//             lik -= std::pow((lsf - lsn) / snr, pp);
//             if (!compute_grad)
//                 return limbo::opt::no_grad(lik);
//             // K^{-1} using Cholesky decomposition
//             Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);
//             gp.matrixL().template triangularView<Eigen::Lower>().solveInPlace(w);
//             gp.matrixL().template triangularView<Eigen::Lower>().transpose().solveInPlace(w);
//             // alpha * alpha.transpose() - K^{-1}
//             w = gp.alpha() * gp.alpha().transpose() - w;
//             // only compute half of the matrix (symmetrical matrix)
//             Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
//             for (size_t i = 0; i < n; ++i) {
//                 for (size_t j = 0; j <= i; ++j) {
//                     Eigen::VectorXd g = gp.kernel_function().grad(gp.samples()[i], gp.samples()[j]);
//                     if (i == j)
//                         grad += w(i, j) * g * 0.5;
//                     else
//                         grad += w(i, j) * g;
//                 }
//             }
//             // Gradient update with penalties
//             /// df(li) += (p * ((ll - log(curb.std')).^(p-1))) / (log(curb.ls)^p);
//             Eigen::VectorXd grad_ll = pp * (ll - samples_std.transpose()).array().pow(pp - 1) / std::pow(ls, pp);
//             grad.segment(0, grad.size() - 2) = grad.segment(0, grad.size() - 2) - grad_ll;
//             /// df(sfi) = df(sfi) + p*(lsf - lsn).^(p-1)/log(curb.snr)^p;
//             double mgrad_v = pp * std::pow((lsf - lsn), pp - 1) / std::pow(snr, pp);
//             grad(grad.size() - 2) = grad(grad.size() - 2) - mgrad_v;
//             // NOTE: This is for the noise calculation
//             // df(end) = df(end) - p * sum((lsf - lsn).^ (p - 1) / log(curb.snr) ^ p);
//             grad(grad.size() - 1) += pp * std::pow((lsf - lsn), pp - 1) / std::pow(snr, pp);
//             return {lik, grad};
//         }
//
//     protected:
//         const GP& _original_gp;
//     };
// };

struct Rastrigin {
    double operator()(const Eigen::VectorXd& x) const
    {
        double f = 10 * x.size();
        for (size_t i = 0; i < x.size(); ++i)
            f += x(i) * x(i) - 10 * cos(2 * M_PI * x(i));
        return f;
    }
};

struct Params {
    struct kernel {
        BO_PARAM(double, noise, 0.01);
        BO_PARAM(bool, optimize_noise, true);
    };

    // struct kernel_exp {
    //     BO_PARAM(double, sigma_sq, 1);
    //     BO_PARAM(double, l, 1);
    // };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public limbo::defaults::opt_parallelrepeater {
        BO_PARAM(int, repeats, 3);
    };
};

template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
class STGP {
public:
    using GP_t = limbo::model::GP<Params, KernelFunction, MeanFunction, HyperParamsOptimizer>;

    void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations)
    {
        int n = 100;
        int N = samples.size();
        int d = std::ceil(std::log(N / double(n)) / std::log(2.0));
        std::cout << "We want tree of depth: " << d << std::endl;

        _root = spt::make_spt(samples, observations, d, 0.1);

        _leaves = spt::get_leaves(_root);

        std::cout << "Regions: " << _leaves.size() << std::endl;

        for (size_t i = 0; i < _leaves.size(); i++)
            std::cout << _leaves[i]->points().size() << " ";
        std::cout << std::endl;

        _gps.resize(_leaves.size());
        // for (size_t i = 0; i < _leaves.size(); i++)
        limbo::tools::par::loop(0, _leaves.size(), [&](size_t i) {
            _gps[i].compute(_leaves[i]->points(), _leaves[i]->observations(), false);
            _gps[i].optimize_hyperparams();
        });
        for (size_t i = 0; i < _leaves.size(); i++) {
            Eigen::VectorXd p = _gps[i].kernel_function().h_params();
            std::cout << i << ": ";
            for (int j = 0; j < p.size() - 2; j++)
                std::cout << std::exp(p(j)) << " ";
            std::cout << std::exp(2 * p(p.size() - 2)) << " " << std::exp(2 * p(p.size() - 1)) << std::endl;
        }
    }

    std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
    {
        return _gps[_find_gp(v)].query(v);
    }

    Eigen::VectorXd mu(const Eigen::VectorXd& v) const
    {
        return _gps[_find_gp(v)].mu(v);
    }

    double sigma(const Eigen::VectorXd& v) const
    {
        return _gps[_find_gp(v)].sigma(v);
    }

protected:
    std::shared_ptr<spt::SPTNode> _root;
    std::vector<std::shared_ptr<spt::SPTNode>> _leaves;
    std::vector<GP_t> _gps;

    int _find_gp(const Eigen::VectorXd& v) const
    {
        auto n = _root;
        while (n->right() && n->left()) {
            Eigen::VectorXd split_dir = n->split_dir();
            double val = split_dir.dot(v);
            if (val <= n->split_median())
                n = n->left();
            else
                n = n->right();
        }
        int i = 0;
        for (; i < _leaves.size(); i++)
            if (n == _leaves[i])
                break;

        return i;
    }
};

int main()
{
    limbo::tools::par::init();
    int N = 1200;
    // int n = 200;
    int N_test = 10000;
    int D = 12;

    Rastrigin func;

    std::vector<Eigen::VectorXd> points, obs;
    for (size_t i = 0; i < N; i++) {
        bool skip = false;
        Eigen::VectorXd p = limbo::tools::random_vector(D).array() * 10.24 - 5.12;
        // Eigen::VectorXd p(D);
        // for (int j = 0; j < D; j++) {
        //     p(j) = gaussian_rand(0.0, 2.0); //std::min(5.12, std::max(-5.12, gaussian_rand(0.0, 2.0)));
        //     if (p(j) > 5.12 || p(j) < -5.12) {
        //         skip = true;
        //         break;
        //     }
        // }
        if (!skip) {
            points.push_back(p);
            Eigen::VectorXd ob(1);
            ob << func(p) + gaussian_rand(0.0, 1.0);
            obs.push_back(ob);
        }
    }

    STGP<Params, limbo::kernel::SquaredExpARD<Params>, limbo::mean::NullFunction<Params>, limbo::model::gp::KernelLFOpt<Params>> gp;

    auto start = std::chrono::high_resolution_clock::now();
    gp.compute(points, obs);
    auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Time: " << time1 / double(1000.0) << std::endl;

    limbo::model::GP<Params, limbo::kernel::SquaredExpARD<Params>, limbo::mean::NullFunction<Params>, limbo::model::gp::KernelLFOpt<Params>> gp_old;
    start = std::chrono::high_resolution_clock::now();
    gp_old.compute(points, obs, false);
    gp_old.optimize_hyperparams();
    Eigen::VectorXd pp = gp_old.kernel_function().h_params();
    std::cout << "old_gp : ";
    for (int j = 0; j < pp.size() - 2; j++)
        std::cout << std::exp(pp(j)) << " ";
    std::cout << std::exp(2 * pp(pp.size() - 2)) << " " << std::exp(2 * pp(pp.size() - 1)) << std::endl;
    time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Time (old gp): " << time1 / double(1000.0) << std::endl;

    std::vector<Eigen::VectorXd> test_points, test_obs, stgp, old_gp;
    std::vector<double> stgp_err, old_gp_err;
    for (size_t i = 0; i < N_test; i++) {
        Eigen::VectorXd p = limbo::tools::random_vector(D).array() * 10.24 - 5.12;
        // Eigen::VectorXd p(D);
        // for (int j = 0; j < D; j++) {
        //     p(j) = gaussian_rand(0.0, 10.0);
        // }
        test_points.push_back(p);

        Eigen::VectorXd ob(1);
        ob << func(p);
        test_obs.push_back(ob);
    }

    start = std::chrono::high_resolution_clock::now();
    double err = 0.0;
    for (size_t i = 0; i < N_test; i++) {
        stgp.push_back(gp.mu(test_points[i]));
        stgp_err.push_back((stgp.back() - test_obs[i]).norm());
        err += stgp_err.back();
    }
    time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Time (query) in ms: " << time1 / double(N_test) << " average mse: " << err / double(N_test) << std::endl;

    start = std::chrono::high_resolution_clock::now();
    err = 0.0;
    for (size_t i = 0; i < N_test; i++) {
        old_gp.push_back(gp_old.mu(test_points[i]));
        old_gp_err.push_back((old_gp.back() - test_obs[i]).norm());
        err += old_gp_err.back();
    }
    time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Time (old_gp-query) in ms: " << time1 / double(N_test) << " average mse: " << err / double(N_test) << std::endl;

    // std::cout << "Saving data..." << std::endl;
    //
    // std::ofstream ofs("gp.dat");
    // std::ofstream ofs_old("gp_old.dat");
    // int pp = 2000;
    // for (int i = 0; i < pp; ++i) {
    //     Eigen::VectorXd v = limbo::tools::make_vector(i / double(pp)).array() * 10.24 - 5.12;
    //     Eigen::VectorXd mu;
    //     double sigma;
    //     std::tie(mu, sigma) = gp.query(v);
    //     // an alternative (slower) is to query mu and sigma separately:
    //     //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
    //     //  double s2 = gp.sigma(v);
    //     ofs << v.transpose() << " " << mu[0] << " " << std::sqrt(sigma) << std::endl;
    //
    //     std::tie(mu, sigma) = gp_old.query(v);
    //     ofs_old << v.transpose() << " " << mu[0] << " " << std::sqrt(sigma) << std::endl;
    // }
    //
    // std::ofstream ofs_data("data.dat");
    // for (size_t i = 0; i < points.size(); ++i)
    //     ofs_data << points[i].transpose() << " " << obs[i].transpose() << std::endl;

    // OLD EXPERIMENTS
    // // std::vector<double> test = {2, 5, 1, 20, -2, 6, 6, 4, 20, -10, -30, 50, 100, -200, -42, 1.5};
    // // for (double p = 0.0; p <= 1.01; p += 0.05)
    // //     std::cout << p << ": " << spt::get_quantile(test, p) << std::endl;
    // // // std::cout << spt::get_median(test) << std::endl;
    // int N = 7000;
    // int n = 200;
    // int D = 2;
    //
    // std::vector<Eigen::VectorXd> points, obs;
    // for (size_t i = 0; i < N; i++) {
    //     // Eigen::VectorXd p = limbo::tools::random_vector(D).array() * 10.0 - 5.0;
    //     Eigen::VectorXd p(D);
    //     for (int j = 0; j < D; j++) {
    //         p(j) = gaussian_rand(0.0, 10.0);
    //     }
    //     points.push_back(p);
    //     Eigen::VectorXd ob(1);
    //     ob << p(0);
    //     obs.push_back(ob);
    // }
    //
    // // STGP<Params, limbo::kernel::Exp<Params>, limbo::mean::NullFunction<Params>> gp;
    // //
    // // auto start = std::chrono::high_resolution_clock::now();
    // // gp.compute(points, obs);
    // // auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    // // std::cout << "Time: " << time1 / double(1000.0) << std::endl;
    // //
    // // limbo::model::GP<Params, limbo::kernel::Exp<Params>, limbo::mean::NullFunction<Params>> gp_old;
    // // start = std::chrono::high_resolution_clock::now();
    // // gp_old.compute(points, obs);
    // // time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    // // std::cout << "Time (old gp): " << time1 / double(1000.0) << std::endl;
    //
    // // Eigen::MatrixXd cov = spt::sample_covariance(points);
    // // std::cout << cov << std::endl;
    //
    // // Eigen::VectorXd dir = spt::get_split_dir(points);
    // // std::cout << dir.transpose() << std::endl;
    //
    // int d = std::ceil(std::log(N / double(n)) / std::log(2.0));
    // std::cout << "We want tree of depth: " << d << std::endl;
    //
    // auto start = std::chrono::high_resolution_clock::now();
    // auto tree = spt::make_spt(points, obs, d);
    // auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    //
    // start = std::chrono::high_resolution_clock::now();
    // auto leaves = get_leaves(tree);
    // auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    //
    // std::cout << "Num of leaves: " << leaves.size() << std::endl;
    // // std::cout << leaves[0]->points().size() << std::endl;
    //
    // start = std::chrono::high_resolution_clock::now();
    // std::ofstream ofs0("bp.dat");
    // std::ofstream ofs02("bp_num.dat");
    // for (size_t i = 0; i < leaves.size(); i++) {
    //     for (size_t j = i + 1; j < leaves.size(); j++) {
    //         std::vector<Eigen::VectorXd> bp = spt::get_shared_boundaries(leaves[i], leaves[j]);
    //         limbo::tools::rgen_int_t rgen(0.0, bp.size() - 1);
    //         // std::cout << i + 1 << ", " << j + 1 << ": " << bp.size() << std::endl;
    //         // ofs02 << bp.size() << std::endl;
    //         int B = 7;
    //         if (bp.size() > 0) {
    //             ofs02 << B << std::endl;
    //             // for (auto b : bp) {
    //             //     ofs0 << b(0) << " " << b(1) << std::endl;
    //             // }
    //             std::vector<Eigen::VectorXd> new_bp;
    //             for (int i = 0; i < B; i++)
    //                 new_bp.push_back(bp[rgen.rand()]);
    //             for (auto b : new_bp)
    //                 ofs0 << b(0) << " " << b(1) << std::endl;
    //         }
    //         else
    //             ofs02 << 0 << std::endl;
    //         // std::cout << b(0) << " " << b(1) << std::endl;
    //         // std::cout << "----------------------" << std::endl;
    //         // std::cin.get();
    //     }
    // }
    // auto time3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    //
    // std::cout << time1 / 1000.0 << " " << time2 / 1000.0 << " " << time3 / 1000.0 << std::endl;
    // std::cout << (time1 + time2 + time3) / 1000.0 << std::endl;
    //
    // std::ofstream ofs("out.dat");
    // std::ofstream ofs2("num.dat");
    // std::ofstream ofs3("split.dat");
    // // std::ofstream ofs4("bounds.dat");
    // for (size_t i = 0; i < leaves.size(); i++) {
    //     // std::cout << i << ":" << std::endl;
    //     ofs2 << leaves[i]->points().size() << std::endl;
    //     for (size_t j = 0; j < leaves[i]->points().size(); j++) {
    //         // std::cout << leaves[i]->points()[j].transpose() << std::endl;
    //         ofs << leaves[i]->points()[j].transpose() << std::endl;
    //     }
    //
    //     // auto c = leaves[i];
    //     // Eigen::VectorXd spv2 = c->split_vector();
    //     // Eigen::VectorXd min_p = c->min();
    //     // Eigen::VectorXd max_p = c->max();
    //     // ofs4 << spv2(0) << " " << spv2(1) << " " << min_p(0) << " " << min_p(1) << " " << max_p(0) << " " << max_p(1) << std::endl;
    //
    //     auto c = leaves[i];
    //     // if (i % 2 == 0) {
    //     size_t k = 0;
    //     auto p = leaves[i]->parent();
    //     while (p) {
    //         Eigen::VectorXd sp = p->split_dir();
    //         Eigen::VectorXd spv = p->split_vector();
    //         Eigen::VectorXd min_p = p->min();
    //         Eigen::VectorXd max_p = p->max();
    //         ofs3 << sp(0) << " " << sp(1) << " " << spv(0) << " " << spv(1) << " " << d - k << " " << min_p(0) << " " << min_p(1) << " " << max_p(0) << " " << max_p(1) << std::endl;
    //
    //         // p = p->parent();
    //         k++;
    //
    //         // auto c = p; //leaves[i];
    //         // Eigen::VectorXd spv2 = c->split_vector();
    //         // Eigen::VectorXd min_p = c->min();
    //         // Eigen::VectorXd max_p = c->max();
    //         // ofs4 << spv2(0) << " " << spv2(1) << " " << min_p(0) << " " << min_p(1) << " " << max_p(0) << " " << max_p(1) << std::endl;
    //         p = p->parent();
    //     }
    //     // }
    //     // std::cout << "------------------------------------------" << std::endl;
    // }
    return 0;
}
