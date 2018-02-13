#include <iostream>
#include <limbo/limbo.hpp>
#include <chrono>
#include <algorithm>
#include <string>
#include <iomanip>
#include <limits>
#include "test_functions.hpp"

using namespace limbo;

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<T> gaussian(m, v);
    return gaussian(gen);
}

@PARAMS

inline Eigen::MatrixXd sample_covariance(const std::vector<Eigen::VectorXd>& points)
{
    assert(points.size());

    // Get the sample means
    Eigen::VectorXd means = Eigen::VectorXd::Zero(points[0].size());

    for (size_t i = 0; i < points.size(); i++) {
        means.array() += points[i].array();
    }

    means = means.array() / double(points.size());

    // Calculate the sample covariance matrix
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(points[0].size(), points[0].size());
    for (size_t i = 0; i < points.size(); i++) {
        cov = cov + points[i] * points[i].transpose();
    }

    cov = (cov.array() - (double(points.size()) * means * means.transpose()).array()) / (double(points.size()) - 1.0);

    return cov;
}

template <typename Function>
void benchmark(const std::string& name, std::vector<int> dimensions, std::vector<int> points)
{
    Function func;
    std::vector<int> dims;
    if (func.dims() > 0)
        dims.push_back(func.dims());
    else {
        dims = dimensions;
    }

    // Whether to add noise or not
    bool add_noise = @NOISE;

    int N_test = 10000;

    std::ofstream ofs_res(name + ".dat");

    for (size_t dim = 0; dim < dims.size(); dim++) {
        std::vector<Eigen::VectorXd> bounds = func.bounds();
        bool one_bound = (bounds.size() == 1);
        int D = dims[dim];

        for (size_t n = 0; n < points.size(); n++) {
            int N = points[n];
            if (N <= 0) {
                std::cerr << "Number of points less or equal to zero!" << std::endl;
                continue;
            }

            // Output number of models
            ofs_res << D << " " << N << " @NMODELS" << std::endl;

            std::cout << name << " in dim: " << D << " and # of points: " << N << std::endl;
            std::string file_name = name + "_" + std::to_string(D) + "_" + std::to_string(N);

            std::vector<Eigen::VectorXd> points, obs;
            for (int i = 0; i < N; i++) {
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
                ob << func(p);
                obs.push_back(ob);
            }

            if (add_noise) {
                Eigen::MatrixXd cov = sample_covariance(obs);
                double sigma = std::sqrt(cov(0, 0)) / 20.0;

                std::cout << "Adding noise of: " << sigma << std::endl;

                for (int i = 0; i < N; i++)
                    obs[i] = obs[i].array() + gaussian_rand(0.0, sigma);
            }

            // Learn the GPs code here
            @GPSLEARN

            // Generation of test data
            std::vector<Eigen::VectorXd> test_points, test_obs;
            for (int i = 0; i < N_test; i++) {
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

            // Prediction of the GPs code here
            @GPSQUERY

            std::cout << "Saving data..." << std::endl;

            // std::ofstream ofs(file_name + "_gp.dat");
            // std::ofstream ofs_real(file_name + "_real.dat");
            // int pp = 4000;
            // for (int i = 0; i < pp; ++i) {
            //     Eigen::VectorXd v = limbo::tools::random_vector(D);
            //     if (one_bound)
            //         v = v.array() * (bounds[0](1) - bounds[0](0)) + bounds[0](0);
            //     else {
            //         for (int j = 0; j < D; j++) {
            //             v(j) = v(j) * (bounds[j](1) - bounds[j](0)) + bounds[j](0);
            //         }
            //     }
            //     Eigen::VectorXd mu;
            //     double sigma;
            //     std::tie(mu, sigma) = gp.query(v);
            //
            //     ofs << v.transpose() << " " << mu[0] << " " << std::sqrt(sigma) << std::endl;
            //
            //     double val = func(v);
            //     ofs_real << v.transpose() << " " << val << " 0" << std::endl;
            // }

            std::ofstream ofs_data(file_name + "_data.dat");
            for (size_t i = 0; i < points.size(); ++i)
                ofs_data << points[i].transpose() << " " << obs[i].transpose() << std::endl;

            std::ofstream ofs_test(file_name + "_test.dat");
            for (size_t i = 0; i < test_points.size(); ++i)
                ofs_test << test_points[i].transpose() << " " << test_obs[i].transpose() << std::endl;

            std::cout << "Data saved...!" << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
@FUNCS
    return 0;
}
