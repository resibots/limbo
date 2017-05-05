#ifndef SPT_STGP_HPP
#define SPT_STGP_HPP

#include <limbo/limbo.hpp>
#include "spt.hpp"

namespace spt {

    namespace defaults {
        struct spt_stgp {
            BO_PARAM(int, leaf_size, 100);
            BO_PARAM(double, tau, 0.2);
            BO_PARAM(bool, global_gp, true);
            BO_PARAM(bool, multi_query, false);
        };
    }

    template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = limbo::model::gp::NoLFOpt<Params>>
    class STGP {
    public:
        using GP_t = limbo::model::GP<Params, KernelFunction, MeanFunction, limbo::model::gp::NoLFOpt<Params>>;
        using GP_Big_t = limbo::model::GP<Params, KernelFunction, MeanFunction, HyperParamsOptimizer>;
        // using GP_t = limbo::model::GP<Params, KernelFunction, MeanFunction, HyperParamsOptimizer>;

        STGP(int dim_in = -1, int dim_out = -1) {}

        void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations)
        {
            int n = Params::spt_stgp::leaf_size();
            int N = samples.size();
            int d = std::ceil(std::log(N / double(n)) / std::log(2.0));
            // std::cout << "We want tree of depth: " << d << std::endl;

            GP_Big_t big_gp;

            if (Params::spt_stgp::global_gp()) {
                std::vector<Eigen::VectorXd> subsamples, subobservations;
                std::vector<int> ids(samples.size());
                for (int i = 0; i < ids.size(); i++)
                    ids[i] = i;
                std::random_shuffle(ids.begin(), ids.end());
                int N_sub = std::min(2 * n, N);
                for (int i = 0; i < N_sub; i++) {
                    int id = ids[i];
                    subsamples.push_back(samples[id]);
                    subobservations.push_back(observations[id]);
                }

                big_gp.compute(subsamples, subobservations, false);
                big_gp.optimize_hyperparams();
                Eigen::VectorXd p = big_gp.kernel_function().h_params();
                std::cout << "big: ";
                for (int j = 0; j < p.size() - 2; j++)
                    std::cout << std::exp(p(j)) << " ";
                std::cout << std::exp(2 * p(p.size() - 2)) << " " << std::exp(2 * p(p.size() - 1)) << std::endl;
            }

            _root = spt::make_spt(samples, observations, d, Params::spt_stgp::tau());

            _leaves = spt::get_leaves(_root);

            // std::cout << "Regions: " << _leaves.size() << std::endl;
            //
            // for (size_t i = 0; i < _leaves.size(); i++)
            //     std::cout << _leaves[i]->points().size() << " ";
            // std::cout << std::endl;

            _gps.resize(_leaves.size());
            // for (size_t i = 0; i < _leaves.size(); i++)
            limbo::tools::par::loop(0, _leaves.size(), [&](size_t i) {
              _gps[i].compute(_leaves[i]->points(), _leaves[i]->observations(), false);
              if (Params::spt_stgp::global_gp()) {
                  _gps[i].kernel_function().set_h_params(big_gp.kernel_function().h_params());
                  _gps[i].recompute(false);
              }
              else {
                  _gps[i].optimize_hyperparams();
              }
            });

            if (!Params::spt_stgp::global_gp()) {
                for (size_t i = 0; i < _leaves.size(); i++) {
                    Eigen::VectorXd p = _gps[i].kernel_function().h_params();
                    std::cout << i << ": ";
                    for (int j = 0; j < p.size() - 2; j++)
                        std::cout << std::exp(p(j)) << " ";
                    std::cout << std::exp(2 * p(p.size() - 2)) << " " << std::exp(2 * p(p.size() - 1)) << std::endl;
                }
            }
        }

        std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
        {
            if (!Params::spt_stgp::multi_query())
                return _gps[_find_gp(v)].query(v);

            std::vector<int> ids = _find_gps(_root, v);
            assert(ids.size());
            Eigen::VectorXd mu = Eigen::VectorXd::Zero(1);
            double sigma = 0.0;
            for (int i = 0; i < ids.size(); i++) {
                Eigen::VectorXd tmu;
                double ts;
                std::tie(tmu, ts) = _gps[ids[i]].query(v);
                mu += tmu;
                sigma += ts;
            }

            mu = mu.array() / double(ids.size());
            sigma /= double(ids.size());

            return std::make_tuple(mu, sigma);
        }

        Eigen::VectorXd mu(const Eigen::VectorXd& v) const
        {
            if (!Params::spt_stgp::multi_query())
                return _gps[_find_gp(v)].mu(v);

            std::vector<int> ids = _find_gps(_root, v);
            assert(ids.size());
            Eigen::VectorXd mu = Eigen::VectorXd::Zero(1);
            for (int i = 0; i < ids.size(); i++) {
                Eigen::VectorXd tmu = _gps[ids[i]].mu(v);
                mu += tmu;
            }

            mu = mu.array() / double(ids.size());

            return mu;
        }

        double sigma(const Eigen::VectorXd& v) const
        {
            if (!Params::spt_stgp::multi_query())
                return _gps[_find_gp(v)].sigma(v);

            std::vector<int> ids = _find_gps(_root, v);
            assert(ids.size());
            double sigma = 0.0;
            for (int i = 0; i < ids.size(); i++) {
                double ts = _gps[ids[i]].sigma(v);
                sigma += ts;
            }

            sigma /= double(ids.size());

            return sigma;
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

        std::vector<int> _find_gps(const std::shared_ptr<spt::SPTNode>& node, const Eigen::VectorXd& v) const
        {
            std::vector<int> ids;
            if (!node->right() || !node->left()) {
                int i = 0;
                for (; i < _leaves.size(); i++)
                    if (node == _leaves[i])
                        break;
                ids.push_back(i);
            }
            else {
                std::vector<int> left_ids, right_ids;

                Eigen::VectorXd split_dir = node->split_dir();
                double val = split_dir.dot(v);
                if (val <= node->split_median_left()) {
                    left_ids = _find_gps(node->left(), v);
                }

                if (val > node->split_median_right()) {
                    right_ids = _find_gps(node->right(), v);
                }

                for (int i = 0; i < left_ids.size(); i++)
                    ids.push_back(left_ids[i]);
                for (int i = 0; i < right_ids.size(); i++)
                    ids.push_back(right_ids[i]);
            }

            return ids;
        }
    };
}

#endif
