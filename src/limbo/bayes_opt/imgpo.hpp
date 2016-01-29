#ifndef LIMBO_BAYES_OPT_IMGPO_HPP
#define LIMBO_BAYES_OPT_IMGPO_HPP

#include <iostream>
#include <algorithm>
#include <iterator>
#include <cmath>

#include <boost/parameter/aux_/void.hpp>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>
#include <limbo/bayes_opt/bo_base.hpp>

namespace limbo {
    namespace bayes_opt {
        struct TreeNode {
            std::vector<Eigen::VectorXd> x_max, x_min, x, f;
            std::vector<bool> leaf, samp;
        };
        // clang-format off
        template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_,
          class A6 = boost::parameter::void_>
        // clang-format on
        class IMGPO : public BoBase<Params, A1, A2, A3, A4, A5, A6> {
        public:
            typedef BoBase<Params, A1, A2, A3, A4, A5, A6> base_t;
            typedef typename base_t::model_t model_t;
            typedef typename base_t::acquisition_function_t acquisition_function_t;
            typedef typename base_t::acqui_optimizer_t acqui_optimizer_t;

            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true)
            {
                this->_init(sfun, afun, reset);
                size_t hh_max = 1000;
                // Init tree
                if (this->_total_iterations == 0)
                    _init_tree(hh_max);

                // Init model
                _model = model_t(StateFunction::dim_in, StateFunction::dim_out);

                // Init root
                _tree[0].x_max.push_back(Eigen::VectorXd::Ones(StateFunction::dim_in));
                _tree[0].x_min.push_back(Eigen::VectorXd::Zero(StateFunction::dim_in));
                _tree[0].x.push_back(Eigen::VectorXd::Ones(StateFunction::dim_in) * 0.5);
                _tree[0].f.push_back(sfun(_tree[0].x[0]));
                _tree[0].leaf.push_back(true);
                _tree[0].samp.push_back(true);

                this->add_new_sample(_tree[0].x[0], _tree[0].f[0]);
                double LB = _tree[0].f[0].maxCoeff();
                double inf = std::numeric_limits<double>::infinity();

                size_t depth_T = 0, M = 1;
                double rho_avg = 0, rho_bar = 0;
                size_t xi_max = 0, XI = 1, t = 0, XI_max = 3, split_n = 0;
                double LB_old = LB;

                while (this->_samples.size() == 0 || !this->_stop(*this, afun)) {
                    std::vector<int> i_max(depth_T + 1, -1);
                    std::vector<double> b_max(depth_T + 1, -inf);
                    size_t h_max = depth_T + 1;
                    double b_hi_max = -inf;
                    t = t + 1;

                    // Steps (i)-(ii)
                    for (size_t h = 0; h <= depth_T; h++) {
                        if (h >= h_max)
                            break;

                        bool gp_label = true;
                        while (gp_label) {
                            for (size_t i = 0; i < _tree[0].x.size(); i++) {
                                if (_tree[h].leaf[i] == true) {
                                    double b_hi = _tree[h].f[i][0];
                                    if (b_hi > b_hi_max) {
                                        b_hi_max = b_hi;
                                        i_max[h] = i;
                                        b_max[h] = b_hi;
                                    }
                                }
                            }
                            if (i_max[h] == -1)
                                break;

                            if (_tree[h].samp[i_max[h]] == true) {
                                gp_label = false;
                            }
                            else {
                                Eigen::VectorXd xxx = _tree[h].x[i_max[h]];

                                auto tmp_sample = sfun(xxx);
                                _tree[h].samp[i_max[h]] = true;
                                this->add_new_sample(xxx, tmp_sample);

                                // N = N+1
                                this->_current_iteration++;
                                this->_total_iterations++;
                            }
                        }
                    }

                    // Steps (iii)
                    for (size_t h = 0; h <= depth_T; h++) {
                        if (h >= h_max)
                            break;
                        if (i_max[h] != -1) {
                            int xi = -1;
                            for (size_t h2 = h + 1; h2 < std::min(depth_T, h + std::min((size_t)std::ceil(XI), XI_max)); h2++) {
                                if (i_max[h2] != -1) {
                                    xi = h2 - h;
                                    break;
                                }
                            }

                            double z_max = -inf;
                            size_t M2 = M;
                            if (xi != -1) {
                                std::vector<TreeNode> tmp_tree = std::vector<TreeNode>(h_max);
                                tmp_tree[h].x_max.push_back(_tree[h].x_max[i_max[h]]);
                                tmp_tree[h].x_min.push_back(_tree[h].x_min[i_max[h]]);
                                tmp_tree[h].x.push_back(_tree[h].x[i_max[h]]);

                                M2 = M;
                                for (size_t h2 = h; h2 < h + xi - 1; h2++) {
                                    for (size_t ii = 0; ii < std::pow(3, h2 - h) - 1; ii++) {
                                        auto xx = tmp_tree[h].x[ii];
                                        Eigen::VectorXd to_split = tmp_tree[h2].x_max[ii].array() - tmp_tree[h2].x_min[ii].array();
                                        size_t tmp, splitd;
                                        to_split.maxCoeff(&splitd, &tmp);
                                        auto x_g = xx, x_d = xx;
                                        x_g(splitd) = (5 * tmp_tree[h2].x_min[ii](splitd) + tmp_tree[h2].x_max[ii](splitd)) / 6.0;
                                        x_d(splitd) = (tmp_tree[h2].x_min[ii](splitd) + 5 * tmp_tree[h2].x_max[ii](splitd)) / 6.0;

                                        // TO-DO: Properly handle bl_samples etc
                                        _model.compute(this->_samples, this->_observations, 0.0, this->_bl_samples);
                                        auto tmp_tuple = _model.query(x_g);
                                        // UCB - nu = 0.05
                                        // sqrt(2*log(pi^2*M^2/(12*nu)))
                                        Eigen::VectorXd m_g = std::get<0>(tmp_tuple);
                                        double s2_g = std::get<1>(tmp_tuple);
                                        double gp_varsigma = std::sqrt(2 * std::log(std::pow(M_PI, 2) * std::pow(M2, 2) / (12 * 0.05)));
                                        z_max = std::max(z_max, m_g[0] + gp_varsigma * sqrt(s2_g));
                                        M2++;

                                        _model.compute(this->_samples, this->_observations, 0.0, this->_bl_samples);
                                        auto tmp_tuple2 = _model.query(x_g);
                                        // UCB - nu = 0.05
                                        // sqrt(2*log(pi^2*M^2/(12*nu)))
                                        Eigen::VectorXd m_d = std::get<0>(tmp_tuple2);
                                        double s2_d = std::get<1>(tmp_tuple2);
                                        double gp_varsigma2 = std::sqrt(2 * std::log(std::pow(M_PI, 2) * std::pow(M2, 2) / (12 * 0.05)));
                                        z_max = std::max(z_max, m_d[0] + gp_varsigma2 * sqrt(s2_d));
                                        M2++;

                                        if (z_max >= b_max[h + xi])
                                            break;

                                        tmp_tree[h2 + 1].x.push_back(x_g);
                                        Eigen::VectorXd newmin = tmp_tree[h2].x_min[ii];
                                        tmp_tree[h2 + 1].x_min.push_back(newmin);
                                        Eigen::VectorXd newmax = tmp_tree[h2].x_max[ii];
                                        newmax(splitd) = (2 * tmp_tree[h2].x_min[ii](splitd) + tmp_tree[h2].x_max[ii](splitd)) / 3.0;
                                        tmp_tree[h2 + 1].x_max.push_back(newmax);

                                        tmp_tree[h2 + 1].x.push_back(x_d);
                                        Eigen::VectorXd newmax2 = tmp_tree[h2].x_max[ii];
                                        tmp_tree[h2 + 1].x_max.push_back(newmax2);
                                        Eigen::VectorXd newmin2 = tmp_tree[h2].x_min[ii];
                                        newmin2(splitd) = (tmp_tree[h2].x_min[ii](splitd) + 2 * tmp_tree[h2].x_max[ii](splitd)) / 3.0;
                                        tmp_tree[h2 + 1].x_min.push_back(newmin2);

                                        tmp_tree[h2 + 1].x.push_back(xx);
                                        Eigen::VectorXd newmin3 = tmp_tree[h2].x_min[ii];
                                        newmin3(splitd) = (2 * tmp_tree[h2].x_min[ii](splitd) + tmp_tree[h2].x_max[ii](splitd)) / 3.0;
                                        tmp_tree[h2 + 1].x_min.push_back(newmin3);
                                        Eigen::VectorXd newmax3 = tmp_tree[h2].x_max[ii];
                                        newmax3(splitd) = (tmp_tree[h2].x_min[ii](splitd) + 2 * tmp_tree[h2].x_max[ii](splitd)) / 3.0;
                                        tmp_tree[h2 + 1].x_max.push_back(newmax3);
                                    }
                                    if (z_max >= b_max[h + xi])
                                        break;
                                }
                            }

                            if (xi != -1 && z_max < b_max[h + xi]) {
                                M = M2;
                                i_max[h] = -1;
                                xi_max = std::max(xi, (int)xi_max);
                            }
                        }
                    }

                    // Steps (iv)-(v)
                    double b_hi_max_2 = -inf, rho_t = 0.0;
                    for (size_t h = 0; h <= depth_T; h++) {
                        if (h >= h_max)
                            break;
                        if (i_max[h] != -1 && b_max[h] > b_hi_max_2) {
                            rho_t += 1.0;
                            depth_T = std::max(depth_T, h + 1);
                            split_n++;
                            _tree[h].leaf[i_max[h]] = 0;

                            auto xx = _tree[h].x[i_max[h]];
                            Eigen::VectorXd to_split = _tree[h].x_max[i_max[h]].array() - _tree[h].x_min[i_max[h]].array();
                            size_t tmp, splitd;
                            to_split.maxCoeff(&splitd, &tmp);
                            auto x_g = xx, x_d = xx;
                            x_g(splitd) = (5 * _tree[h].x_min[i_max[h]](splitd) + _tree[h].x_max[i_max[h]](splitd)) / 6.0;
                            x_d(splitd) = (_tree[h].x_min[i_max[h]](splitd) + 5 * _tree[h].x_max[i_max[h]](splitd)) / 6.0;

                            // left node
                            _tree[h + 1].x.push_back(x_g);
                            // TO-DO: Properly handle bl_samples etc
                            _model.compute(this->_samples, this->_observations, 0.0, this->_bl_samples);
                            auto tmp_tuple = _model.query(x_g);
                            // UCB - nu = 0.05
                            // sqrt(2*log(pi^2*M^2/(12*nu)))
                            Eigen::VectorXd m_g = std::get<0>(tmp_tuple);
                            double s2_g = std::get<1>(tmp_tuple);
                            double gp_varsigma = std::sqrt(2 * std::log(std::pow(M_PI, 2) * std::pow(M, 2) / (12 * 0.05)));
                            double UCB = m_g[0] + (gp_varsigma + 0.2) * sqrt(s2_g);
                            Eigen::VectorXd fsample_g;
                            if (UCB <= LB) {
                                M++;
                                fsample_g = Eigen::VectorXd(1);
                                fsample_g(0) = UCB;
                                _tree[h + 1].samp.push_back(false);
                            }
                            else {
                                fsample_g = sfun(x_g);
                                _tree[h + 1].samp.push_back(true);

                                this->add_new_sample(x_g, fsample_g);

                                b_hi_max_2 = std::max(b_hi_max_2, fsample_g(0));

                                // N = N+1
                                this->_current_iteration++;
                                this->_total_iterations++;
                            }

                            _tree[h + 1].f.push_back(fsample_g);

                            Eigen::VectorXd newmin = _tree[h].x_min[i_max[h]];
                            _tree[h + 1].x_min.push_back(newmin);
                            Eigen::VectorXd newmax = _tree[h].x_max[i_max[h]];
                            newmax(splitd) = (2 * _tree[h].x_min[i_max[h]](splitd) + _tree[h].x_max[i_max[h]](splitd)) / 3.0;
                            _tree[h + 1].x_max.push_back(newmax);
                            _tree[h + 1].leaf.push_back(true);

                            // right node
                            _tree[h + 1].x.push_back(x_d);
                            // TO-DO: Properly handle bl_samples etc
                            _model.compute(this->_samples, this->_observations, 0.0, this->_bl_samples);
                            auto tmp_tuple2 = _model.query(x_d);
                            // UCB - nu = 0.05
                            // sqrt(2*log(pi^2*M^2/(12*nu)))
                            Eigen::VectorXd m_g2 = std::get<0>(tmp_tuple);
                            double s2_g2 = std::get<1>(tmp_tuple);
                            double gp_varsigma2 = std::sqrt(2 * std::log(std::pow(M_PI, 2) * std::pow(M, 2) / (12 * 0.05)));
                            double UCB2 = m_g[0] + (gp_varsigma2 + 0.2) * sqrt(s2_g2);
                            Eigen::VectorXd fsample_d;
                            if (UCB2 <= LB) {
                                M++;
                                fsample_d = Eigen::VectorXd(1);
                                fsample_d(0) = UCB2;
                                _tree[h + 1].samp.push_back(false);
                            }
                            else {
                                fsample_d = sfun(x_d);
                                _tree[h + 1].samp.push_back(true);

                                this->add_new_sample(x_d, fsample_d);

                                b_hi_max_2 = std::max(b_hi_max_2, fsample_d(0));

                                // N = N+1
                                this->_current_iteration++;
                                this->_total_iterations++;
                            }

                            _tree[h + 1].f.push_back(fsample_d);

                            Eigen::VectorXd newmax2 = _tree[h].x_max[i_max[h]];
                            _tree[h + 1].x_max.push_back(newmax2);
                            Eigen::VectorXd newmin2 = _tree[h].x_min[i_max[h]];
                            newmin2(splitd) = (_tree[h].x_min[i_max[h]](splitd) + 2 * _tree[h].x_max[i_max[h]](splitd)) / 3.0;
                            _tree[h + 1].x_min.push_back(newmin2);
                            _tree[h + 1].leaf.push_back(true);

                            // central node
                            _tree[h + 1].x.push_back(xx);
                            _tree[h + 1].f.push_back(_tree[h].f[i_max[h]]);
                            _tree[h + 1].samp.push_back(true);
                            Eigen::VectorXd newmin3 = _tree[h].x_min[i_max[h]];
                            Eigen::VectorXd newmax3 = _tree[h].x_max[i_max[h]];
                            newmin3(splitd) = (2 * _tree[h].x_min[i_max[h]](splitd) + _tree[h].x_max[i_max[h]](splitd)) / 3.0;
                            newmax3(splitd) = (_tree[h].x_min[i_max[h]](splitd) + 2 * _tree[h].x_max[i_max[h]](splitd)) / 3.0;
                            _tree[h + 1].x_min.push_back(newmin3);
                            _tree[h + 1].x_max.push_back(newmax3);
                            _tree[h + 1].leaf.push_back(true);
                        }
                    }

                    // Finalizing iteration
                    rho_avg = (rho_avg * (t - 1) + rho_t) / t;
                    rho_bar = std::max(rho_bar, rho_avg);
                    // update XI
                    if (std::abs(LB_old - LB) < 1e-6)
                        XI = std::max((int)(XI - std::pow(2, -1)), 1);
                    else
                        XI = XI + std::pow(2, 2);
                    LB_old = LB;
                }
            }

            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_observation(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_observations.size());
                std::transform(this->_observations.begin(), this->_observations.end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_observations[std::distance(rewards.begin(), max_e)];
            }

            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_sample(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_observations.size());
                std::transform(this->_observations.begin(), this->_observations.end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_samples[std::distance(rewards.begin(), max_e)];
            }

            const model_t& model() const { return _model; }

        protected:
            void _init_tree(size_t h_max = 1000)
            {
                _tree.clear();
                _tree = std::vector<TreeNode>(h_max);
            }

            model_t _model;
            std::vector<TreeNode> _tree;
        };
    }
}

#endif
