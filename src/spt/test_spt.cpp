#include <iostream>
#include <limbo/limbo.hpp>
#include <chrono>
#include "stgp.hpp"
#include "test_functions.hpp"

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> gaussian(m, v);
    return gaussian(gen);
}

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

template <typename Function>
void benchmark(const std::string& name)
{
    Function func;
    std::vector<int> dims;
    if (func.dims() > 0)
        dims.push_back(func.dims());
    else {
        for (int i = 1; i <= 24; i += 2) {
            dims.push_back(i);
        }
    }

    // int N = 1200;
    // int n = 200;
    int N_test = 10000;

    for (size_t dim = 0; dim < dims.size(); dim++) {
        std::vector<Eigen::VectorXd> bounds = func.bounds();
        bool one_bound = (bounds.size() == 1);
        int D = dims[dim];

        for (int N = 400; N <= 12800; N = 2 * N) {
            std::cout << name << " in dim: " << D << " and # of points: " << N << std::endl;
            std::string file_name = name + "_" + std::to_string(D) + "_" + std::to_string(N);

            std::vector<Eigen::VectorXd> points, obs;
            for (size_t i = 0; i < N; i++) {
                Eigen::VectorXd p = limbo::tools::random_vector(D); //.array() * 10.24 - 5.12;
                if (one_bound)
                    p = p.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
                else {
                    for (int j = 0; j < D; j++) {
                        p(j) = p(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
                    }
                }

                points.push_back(p);
                Eigen::VectorXd ob(1);
                // TO-DO: Put noise according to observations width
                ob << func(p); // + gaussian_rand(0.0, 1.0);
                // std::cout << ob << std::endl;
                obs.push_back(ob);
            }

            Eigen::MatrixXd cov = spt::sample_covariance(obs);
            double sigma = std::sqrt(cov(0, 0)) / 20.0;

            std::cout << "Adding noise of: " << sigma << std::endl;

            for (size_t i = 0; i < N; i++)
                obs[i] = obs[i].array() + gaussian_rand(0.0, sigma);

            std::ofstream ofs_res(file_name + ".dat");

            spt::STGP<Params, limbo::kernel::SquaredExpARD<Params>, limbo::mean::NullFunction<Params>, limbo::model::gp::KernelLFOpt<Params>> gp;

            auto start = std::chrono::high_resolution_clock::now();
            gp.compute(points, obs);
            auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time: " << time1 / double(1000.0) << std::endl;
            ofs_res << time1 / double(1000.0) << std::endl;

            limbo::model::GP<Params, limbo::kernel::SquaredExpARD<Params>, limbo::mean::NullFunction<Params>, limbo::model::gp::KernelLFOpt<Params>> gp_old;
            start = std::chrono::high_resolution_clock::now();
            gp_old.compute(points, obs, false);
            gp_old.optimize_hyperparams();
            Eigen::VectorXd pk = gp_old.kernel_function().h_params();
            std::cout << "old_gp : ";
            for (int j = 0; j < pk.size() - 2; j++)
                std::cout << std::exp(pk(j)) << " ";
            std::cout << std::exp(2 * pk(pk.size() - 2)) << " " << std::exp(2 * pk(pk.size() - 1)) << std::endl;
            time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time (old gp): " << time1 / double(1000.0) << std::endl;
            ofs_res << time1 / double(1000.0) << std::endl;

            std::vector<Eigen::VectorXd> test_points, test_obs, stgp, old_gp;
            std::vector<double> stgp_err, old_gp_err;
            for (size_t i = 0; i < N_test; i++) {
                Eigen::VectorXd p = limbo::tools::random_vector(D);
                if (one_bound)
                    p = p.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
                else {
                    for (int j = 0; j < D; j++) {
                        p(j) = p(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
                    }
                }

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
            ofs_res << time1 / double(N_test) << " " << err / double(N_test) << std::endl;

            start = std::chrono::high_resolution_clock::now();
            err = 0.0;
            for (size_t i = 0; i < N_test; i++) {
                old_gp.push_back(gp_old.mu(test_points[i]));
                old_gp_err.push_back((old_gp.back() - test_obs[i]).norm());
                err += old_gp_err.back();
            }
            time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::cout << "Time (old_gp-query) in ms: " << time1 / double(N_test) << " average mse: " << err / double(N_test) << std::endl;
            ofs_res << time1 / double(N_test) << " " << err / double(N_test) << std::endl;

            std::cout << "Saving data..." << std::endl;

            std::ofstream ofs(file_name + "_gp.dat");
            std::ofstream ofs_old(file_name + "_gp_old.dat");
            std::ofstream ofs_real(file_name + "_real.dat");
            int pp = 4000;
            for (int i = 0; i < pp; ++i) {
                Eigen::VectorXd v = limbo::tools::random_vector(D);
                if (one_bound)
                    v = v.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
                else {
                    for (int j = 0; j < D; j++) {
                        v(j) = v(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
                    }
                }
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = gp.query(v);
                // an alternative (slower) is to query mu and sigma separately:
                //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
                //  double s2 = gp.sigma(v);
                ofs << v.transpose() << " " << mu[0] << " " << std::sqrt(sigma) << std::endl;

                std::tie(mu, sigma) = gp_old.query(v);
                ofs_old << v.transpose() << " " << mu[0] << " " << std::sqrt(sigma) << std::endl;

                double val = func(v);
                ofs_real << v.transpose() << " " << val << " 0" << std::endl;
            }

            std::ofstream ofs_data(file_name + "_data.dat");
            for (size_t i = 0; i < points.size(); ++i)
                ofs_data << points[i].transpose() << " " << obs[i].transpose() << std::endl;

            std::cout << "Data saved...!" << std::endl;
        }
    }
}

int main()
{
    limbo::tools::par::init();

    benchmark<Rastrigin>("rastrigin");
    benchmark<Ackley>("ackley");
    benchmark<Bukin>("bukin");
    benchmark<CrossInTray>("crossintray");
    benchmark<DropWave>("dropwave");
    benchmark<GramacyLee>("gramacylee");
    benchmark<HolderTable>("holdertable");
    benchmark<Levy>("levy");
    benchmark<Schwefel>("schwefel");
    benchmark<SixHumpCamel>("sixhumpcamel");
    benchmark<Hartmann6>("hartmann6");

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
