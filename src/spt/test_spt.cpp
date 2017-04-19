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

int main()
{
    int N = 10000;
    int n = 500;
    int D = 2;

    std::vector<Eigen::VectorXd> points;
    for (size_t i = 0; i < N; i++) {
        // Eigen::VectorXd p = limbo::tools::random_vector(D).array() * 10.0 - 5.0;
        Eigen::VectorXd p(D);
        for (int j = 0; j < D; j++) {
            p(j) = gaussian_rand(0.0, 10.0);
        }
        points.push_back(p);
    }

    // // Eigen::MatrixXd cov = spt::sample_covariance(points);
    // // std::cout << cov << std::endl;
    //
    // Eigen::VectorXd dir = spt::get_split_dir(points);
    // std::cout << dir.transpose() << std::endl;

    int d = std::ceil(std::log(N / double(n)) / std::log(2.0));
    std::cout << "We want tree of depth: " << d << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    auto tree = spt::make_spt(points, d);
    auto time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    start = std::chrono::high_resolution_clock::now();
    auto leaves = get_leaves(tree);
    auto time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    std::cout << "Num of leaves: " << leaves.size() << std::endl;
    // std::cout << leaves[0]->points().size() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::ofstream ofs0("bp.dat");
    std::ofstream ofs02("bp_num.dat");
    for (size_t i = 0; i < leaves.size(); i++) {
        for (size_t j = i + 1; j < leaves.size(); j++) {
            std::vector<Eigen::VectorXd> bp = spt::get_shared_boundaries(leaves[i], leaves[j]);
            // std::cout << i + 1 << ", " << j + 1 << ": " << bp.size() << std::endl;
            ofs02 << bp.size() << std::endl;
            for (auto b : bp)
                ofs0 << b(0) << " " << b(1) << std::endl;
            // std::cout << "----------------------" << std::endl;
        }
    }
    auto time3 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();

    std::cout << time1 / 1000.0 << " " << time2 / 1000.0 << " " << time3 / 1000.0 << std::endl;
    std::cout << (time1 + time2 + time3) / 1000.0 << std::endl;

    std::ofstream ofs("out.dat");
    std::ofstream ofs2("num.dat");
    std::ofstream ofs3("split.dat");
    // std::ofstream ofs4("bounds.dat");
    for (size_t i = 0; i < leaves.size(); i++) {
        // std::cout << i << ":" << std::endl;
        ofs2 << leaves[i]->points().size() << std::endl;
        for (size_t j = 0; j < leaves[i]->points().size(); j++) {
            // std::cout << leaves[i]->points()[j].transpose() << std::endl;
            ofs << leaves[i]->points()[j].transpose() << std::endl;
        }

        // auto c = leaves[i];
        // Eigen::VectorXd spv2 = c->split_vector();
        // Eigen::VectorXd min_p = c->min();
        // Eigen::VectorXd max_p = c->max();
        // ofs4 << spv2(0) << " " << spv2(1) << " " << min_p(0) << " " << min_p(1) << " " << max_p(0) << " " << max_p(1) << std::endl;

        auto c = leaves[i];
        // if (i % 2 == 0) {
        size_t k = 0;
        auto p = leaves[i]->parent();
        while (p) {
            Eigen::VectorXd sp = p->split_dir();
            Eigen::VectorXd spv = p->split_vector();
            Eigen::VectorXd min_p = p->min();
            Eigen::VectorXd max_p = p->max();
            ofs3 << sp(0) << " " << sp(1) << " " << spv(0) << " " << spv(1) << " " << d - k << " " << min_p(0) << " " << min_p(1) << " " << max_p(0) << " " << max_p(1) << std::endl;

            // p = p->parent();
            k++;

            // auto c = p; //leaves[i];
            // Eigen::VectorXd spv2 = c->split_vector();
            // Eigen::VectorXd min_p = c->min();
            // Eigen::VectorXd max_p = c->max();
            // ofs4 << spv2(0) << " " << spv2(1) << " " << min_p(0) << " " << min_p(1) << " " << max_p(0) << " " << max_p(1) << std::endl;
            p = p->parent();
        }
        // }
        // std::cout << "------------------------------------------" << std::endl;
    }
    return 0;
}
