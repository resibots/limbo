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

struct BlockStruct {
public:
    int _r, _c, _sr, _sc;

    BlockStruct(int r, int c, int sr, int sc) : _r(r), _c(c), _sr(sr), _sc(sc) {}

    Eigen::Block<Eigen::MatrixXd> get_block(Eigen::MatrixXd& m)
    {
        return Eigen::Block<Eigen::MatrixXd>(m.derived(), _r, _c, _sr, _sc);
    }
};

template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
class STGP {
public:
    /// useful because the model might be created before knowing anything about the process
    STGP() : _dim_in(-1), _dim_out(-1), _N(200) {}

    /// useful because the model might be created  before having samples
    STGP(int dim_in, int dim_out)
        : _dim_in(dim_in), _dim_out(dim_out), _N(200), _kernel_function(dim_in), _mean_function(dim_out) {}

    /// Compute the GP from samples and observations. This call needs to be explicit!
    void compute(const std::vector<Eigen::VectorXd>& samples,
        const std::vector<Eigen::VectorXd>& observations, bool compute_kernel = true)
    {
        assert(samples.size() != 0);
        assert(observations.size() != 0);
        assert(samples.size() == observations.size());

        if (_dim_in != samples[0].size()) {
            _dim_in = samples[0].size();
            _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low
        }

        if (_dim_out != observations[0].size()) {
            _dim_out = observations[0].size();
            _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low
        }

        _samples = samples;

        // _observations.resize(observations.size(), _dim_out);
        // for (int i = 0; i < _observations.rows(); ++i)
        //     _observations.row(i) = observations[i];
        //
        // _mean_observation = _observations.colwise().mean();

        _N = 200;
        int B = 7;
        int N = _samples.size();
        int d = std::ceil(std::log(N / double(_N)) / std::log(2.0));

        _root = spt::make_spt(_samples, observations, d);
        _leaves = spt::get_leaves(_root);
        std::cout << "Regions: " << _leaves.size() << std::endl;
        _boundaries.clear();

        _observations.resize(observations.size(), _dim_out);
        int r = 0;
        for (size_t i = 0; i < _leaves.size(); i++) {
            for (size_t j = 0; j < _leaves[i]->observations().size(); j++) {
                _observations.row(r) = _leaves[i]->observations()[j];
                r++;
            }
        }

        _mean_observation = _observations.colwise().mean();

        for (size_t i = 0; i < _leaves.size(); i++) {
            std::cout << _leaves[i]->points().size() << ((i < _leaves.size() - 1) ? ", " : "");
            for (size_t j = i + 1; j < _leaves.size(); j++) {
                std::vector<Eigen::VectorXd> bp = spt::get_shared_boundaries(_leaves[i], _leaves[j]);
                limbo::tools::rgen_int_t rgen(0.0, bp.size() - 1);
                // std::cout << i + 1 << ", " << j + 1 << ": " << bp.size() << std::endl;
                // ofs02 << bp.size() << std::endl;
                if (bp.size() > 0) {
                    std::vector<Eigen::VectorXd> new_bp;
                    for (int i = 0; i < B; i++)
                        new_bp.push_back(bp[rgen.rand()]);
                    _boundaries.push_back(new_bp);
                }
                else
                    _boundaries.push_back(std::vector<Eigen::VectorXd>());
            }
        }
        std::cout << std::endl;
        std::cout << "Boundaries: " << _boundaries.size() << std::endl;

        // Query point -- just for testing
        Eigen::VectorXd v = Eigen::VectorXd::Zero(2);
        // query GP
        int kk = 0;

        int num = 0;

        Eigen::VectorXd c_D_star;
        for (int i = 0; i < _leaves.size(); i++) {
            // std::cout << _leaves[i]->points().size() << std::endl;
            Eigen::VectorXd k(_leaves[i]->points().size());
            // TO-DO: Fix kernels
            for (int j = 0; j < k.size(); j++)
                k[j] = _kernel_function(_leaves[i]->points()[j], v);
            // c_d_star.push_back(k);
            c_D_star.conservativeResize(c_D_star.size() + k.size());
            c_D_star.tail(k.size()) = k;
            // std::cout << k.size() << std::endl;
            // num += k.size();
        }
        std::cout << "Whole cD*(k): " << c_D_star.size() << "x1" << std::endl;
        // std::cout << c_D_star.transpose() << std::endl;

        int b = 0;
        // std::vector<Eigen::VectorXd> c_delta_star;
        Eigen::VectorXd c_delta_star(B * _boundaries.size());
        for (size_t i = 0; i < _leaves.size(); i++) {
            for (size_t j = i + 1; j < _leaves.size(); j++) {
                if (_boundaries[b].size() == 0) {
                    // c_delta_star.push_back(Eigen::VectorXd::Zero(B));
                    c_delta_star.segment(b * B, B) = Eigen::VectorXd::Zero(B);
                    // std::cout << B << std::endl;
                }
                else {
                    Eigen::VectorXd k(_boundaries[b].size());
                    // TO-DO: Fix kernels
                    for (int kkk = 0; kkk < k.size(); kkk++) {
                        if (kk == i)
                            k[kkk] = _kernel_function(_boundaries[b][kkk], v);
                        else if (kk == j)
                            k[kkk] = -_kernel_function(_boundaries[b][kkk], v);
                        else
                            k[kkk] = 0.0;
                    }
                    // c_delta_star.push_back(k);
                    c_delta_star.segment(b * B, B) = k;
                    // std::cout << k.size() << std::endl;
                }
                b++;
            }
        }
        std::cout << "Whole δ*(k): " << c_delta_star.size() << "x1" << std::endl;
        // std::cout << c_delta_star.transpose() << std::endl;

        int num1 = 0;
        int num2 = 0;
        b = 0;
        // std::vector<Eigen::MatrixXd> c_D_D;
        _DD_blocks.clear();
        Eigen::MatrixXd c_D_D;
        for (size_t i = 0; i < _leaves.size(); i++) {
            // for (size_t j = i + 1; j < _leaves.size(); j++) {
            Eigen::MatrixXd DD(_leaves[i]->points().size(), _leaves[i]->points().size());
            for (int k1 = 0; k1 < _leaves[i]->points().size(); k1++) {
                for (int k2 = 0; k2 < _leaves[i]->points().size(); k2++) {
                    // TO-DO: Fix kernels
                    DD(k1, k2) = _kernel_function(_leaves[i]->points()[k1], _leaves[i]->points()[k2]);
                }
            }

            // c_D_D.push_back(DD);
            int r = c_D_D.rows(), c = c_D_D.cols();
            c_D_D.conservativeResizeLike(Eigen::MatrixXd::Zero(c_D_D.rows() + DD.rows(), c_D_D.cols() + DD.cols()));
            c_D_D.block(r, c, DD.rows(), DD.cols()) = DD;
            // blocks.push_back(c_D_D.block(r, c, DD.rows(), DD.cols()));
            _DD_blocks.push_back({r, c, DD.rows(), DD.cols()});
            // std::cout << DD.rows() << " " << DD.cols() << std::endl;
            // num1 += DD.rows();
            // num2 += DD.cols();
            b++;
            // }
        }
        std::cout << "Whole cDD: " << c_D_D.rows() << "x" << c_D_D.cols() << std::endl;
        // std::cout << c_D_D << std::endl;
        // Eigen::Block<Eigen::MatrixXd> block = _DD_blocks[0].get_block(c_D_D);
        // for (int i = 0; i < block.rows(); i++)
        //     for (int j = 0; j < block.cols(); j++)
        //         block(i, j) = 4;
        // block = block.inverse();
        // std::cout << c_D_D << std::endl;

        // TO-DO: Fix the inversion (use parallel if possible and proper inversion with cholesky)
        for (int i = 0; i < _DD_blocks.size(); i++) {
            Eigen::Block<Eigen::MatrixXd> block = _DD_blocks[i].get_block(c_D_D);
            block = block.inverse();
        }

        b = 0;
        // num1 = 0;
        // num2 = 0;
        // std::vector<Eigen::MatrixXd> c_d_D;
        Eigen::MatrixXd c_d_D = Eigen::MatrixXd::Zero(_boundaries.size() * B, _samples.size());
        for (size_t i = 0; i < _leaves.size(); i++) {
            for (size_t j = i + 1; j < _leaves.size(); j++) {
                // for (int b = 0; b < _boundaries.size(); b++) {
                num = 0;
                for (int k = 0; k < _leaves.size(); k++) {
                    Eigen::MatrixXd DD(B, _leaves[k]->points().size());
                    for (int s = 0; s < _leaves[k]->points().size(); s++) {
                        if (_boundaries[b].size() == 0) {
                            for (int kkk = 0; kkk < B; kkk++)
                                DD(kkk, s) = 0.0;
                        }
                        else {
                            // TO-DO: Fix kernels
                            for (int kkk = 0; kkk < B; kkk++) {
                                if (k == i)
                                    DD(kkk, s) = _kernel_function(_boundaries[b][kkk], _leaves[k]->points()[s]);
                                else if (k == j)
                                    DD(kkk, s) = -_kernel_function(_boundaries[b][kkk], _leaves[k]->points()[s]);
                                else
                                    DD(kkk, s) = 0.0;
                            }
                        }
                    }
                    // c_d_D.push_back(DD);
                    // std::cout << b* B << "," << num << " -> " << B << "x" << _leaves[k]->points().size() << std::endl;
                    c_d_D.block(b * B, num, B, _leaves[k]->points().size()) = DD;
                    num += _leaves[k]->points().size();
                }
                // num2 = num; // / _boundaries.size();
                // num1 += B;
                b++;
            }
        }
        std::cout << "Whole cδD: " << c_d_D.rows() << "x" << c_d_D.cols() << std::endl;
        // std::cout << c_d_D << std::endl;

        // std::vector<Eigen::MatrixXd> c_d_d;
        _dd_blocks.clear();
        Eigen::MatrixXd c_d_d = Eigen::MatrixXd::Zero(_boundaries.size() * B, _boundaries.size() * B);
        // num1 = 0;
        // num2 = 0;
        int b1 = 0, b2 = 0;
        // for (int b1 = 0; b1 < _boundaries.size(); b1++)
        for (size_t i1 = 0; i1 < _leaves.size(); i1++) {
            for (size_t j1 = i1 + 1; j1 < _leaves.size(); j1++) {
                // for (int b2 = 0; b2 < _boundaries.size(); b2++)
                b2 = 0;
                for (size_t i2 = 0; i2 < _leaves.size(); i2++) {
                    for (size_t j2 = i2 + 1; j2 < _leaves.size(); j2++) {
                        // std::cout << b1 << " " << b2 << ": " << i1 << ", " << j1 << " -> " << i2 << ", " << j2 << std::endl;
                        Eigen::MatrixXd DD(B, B);
                        for (int k1 = 0; k1 < _boundaries[b1].size(); k1++) {
                            for (int k2 = 0; k2 < _boundaries[b2].size(); k2++) {
                                // TO-DO: fix kernels
                                if (i1 == i2 && j1 == j2)
                                    DD(k1, k2) = 2.0 * _kernel_function(_boundaries[b1][k1], _boundaries[b2][k2]);
                                if (i1 == i2 || j1 == j2)
                                    DD(k1, k2) = _kernel_function(_boundaries[b1][k1], _boundaries[b2][k2]);
                                else if (i1 == j2 || i2 == j1)
                                    DD(k1, k2) = -_kernel_function(_boundaries[b1][k1], _boundaries[b2][k2]);
                                else
                                    DD(k1, k2) = 0.0;
                            }
                        }

                        // c_d_d.push_back(DD);
                        c_d_d.block(b1 * B, b2 * B, B, B) = DD;
                        _dd_blocks.push_back({b1 * B, b2 * B, B, B});
                        b2++;
                    }
                    // num1 += B;
                    // num2 += B;
                }
                b1++;
            }
        }

        std::cout << "Whole cδδ: " << c_d_d.rows() << "x" << c_d_d.cols() << std::endl;
        // std::cout << c_d_d << std::endl;
        // Eigen::Block<Eigen::MatrixXd> block = _dd_blocks[0].get_block(c_d_d);
        // // for (int i = 0; i < block.rows(); i++)
        // //     for (int j = 0; j < block.cols(); j++)
        // //         block(i, j) = 4;
        // block = block.inverse();

        // TO-DO: Fix the inversion (use parallel if possible and proper inversion with cholesky)
        for (int i = 0; i < _dd_blocks.size(); i++) {
            Eigen::Block<Eigen::MatrixXd> block = _dd_blocks[i].get_block(c_D_D);
            block = block.inverse();
        }

        // Eigen::MatrixXd tmp = c_d_d.fullPivLu().solve(Eigen::MatrixXd::Identity(c_d_d.rows(), c_d_d.cols()));
        // std::cout << (c_d_d * tmp) << std::endl;

        // this->_compute_obs_mean();
        // if (compute_kernel)
        //     this->_compute_full_kernel();
    }

protected:
    int _dim_in;
    int _dim_out;
    int _N;

    std::shared_ptr<spt::SPTNode> _root;
    std::vector<std::shared_ptr<spt::SPTNode>> _leaves;
    std::vector<std::vector<Eigen::VectorXd>> _boundaries;
    std::vector<BlockStruct> _DD_blocks, _dd_blocks;

    KernelFunction _kernel_function;
    MeanFunction _mean_function;

    std::vector<Eigen::VectorXd> _samples;
    Eigen::MatrixXd _observations;
    Eigen::MatrixXd _mean_vector;
    Eigen::MatrixXd _obs_mean;

    Eigen::VectorXd _mean_observation;

    HyperParamsOptimizer _hp_optimize;
};

struct Params {
    struct kernel {
        BO_PARAM(double, noise, 0.0001);
        BO_PARAM(bool, optimize_noise, false);
    };

    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };
};

int main()
{
    int N = 1000;
    // int n = 500;
    int D = 2;

    std::vector<Eigen::VectorXd> points, obs;
    for (size_t i = 0; i < N; i++) {
        // Eigen::VectorXd p = limbo::tools::random_vector(D).array() * 10.0 - 5.0;
        Eigen::VectorXd p(D);
        for (int j = 0; j < D; j++) {
            p(j) = gaussian_rand(0.0, 10.0);
        }
        points.push_back(p);
        Eigen::VectorXd ob(1);
        ob << p(0);
        obs.push_back(ob);
    }

    STGP<Params, limbo::kernel::Exp<Params>, limbo::mean::NullFunction<Params>> gp;

    auto start = std::chrono::high_resolution_clock::now();
    gp.compute(points, obs);
    auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Time: " << time1 / double(1000.0) << std::endl;

    // // Eigen::MatrixXd cov = spt::sample_covariance(points);
    // // std::cout << cov << std::endl;
    //
    // Eigen::VectorXd dir = spt::get_split_dir(points);
    // std::cout << dir.transpose() << std::endl;

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
