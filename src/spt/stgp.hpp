#ifndef SPT_STGP_HPP
#define SPT_STGP_HPP

#include <limbo/limbo.hpp>
#include "spt.hpp"

namespace spt {
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
}

#endif
