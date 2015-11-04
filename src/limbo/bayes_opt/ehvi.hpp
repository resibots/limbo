#ifndef LIMBO_BAYES_OPT_EHVI_HPP
#define LIMBO_BAYES_OPT_EHVI_HPP

#include <algorithm>

#include <ehvi/ehvi_calculations.h>
#include <ehvi/ehvi_sliceupdate.h>

#include <limbo/bayes_opt/bo_multi.hpp>
#include <limbo/acqui/ehvi.hpp>

namespace limbo {
    namespace bayes_opt {
        template <class Params, class A2 = boost::parameter::void_,
            class A3 = boost::parameter::void_, class A4 = boost::parameter::void_,
            class A5 = boost::parameter::void_, class A6 = boost::parameter::void_>
        class Ehvi : public BoMulti<Params, A2, A3, A4, A5, A6> {
        public:
            typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> pareto_point_t;
            typedef BoBase<Params, A3, A4, A5, A6> base_t;
            typedef typename base_t::model_t model_t;
            typedef typename base_t::acqui_optimizer_t acqui_optimizer_t;

            template <typename EvalFunction>
            void optimize(const EvalFunction& feval, bool reset = true)
            {
                this->_init(feval, reset);

                acqui_optimizer_t inner_opt;

                while (this->_samples.size() == 0 || this->_pursue(*this, FirstElem())) {
                    std::cout.flush();
                    this->template update_pareto_model<EvalFunction::dim>();
                    this->update_pareto_data();

                    // copy in the ehvi structure to compute expected improvement
                    std::deque<individual*> pop;
                    for (auto x : this->pareto_data()) {
                        individual* ind = new individual;
                        ind->f[0] = std::get<1>(x)(0);
                        ind->f[1] = std::get<1>(x)(1);
                        ind->f[2] = 0;
                        pop.push_back(ind);
                    }

                    // optimize ehvi
                    std::cout << "optimizing ehvi (" << this->pareto_data().size() << ")"
                              << std::endl;

                    auto acqui = acqui::Ehvi<Params, model_t>(
                        this->_models, pop,
                        Eigen::Vector3d(Params::ehvi::x_ref(), Params::ehvi::y_ref(), 0));

                    // maximize with CMA-ES
                    typedef std::pair<Eigen::VectorXd, double> pair_t;
                    pair_t init(Eigen::VectorXd::Zero(1), -std::numeric_limits<float>::max());
                    auto body = [&](int i) -> pair_t {
                    // clang-format off
                    auto x = this->pareto_data()[i];
                    Eigen::VectorXd s = inner_opt(acqui, acqui.dim(), std::get<0>(x), FirstElem());
                    double hv = acqui(s);
                    return std::make_pair(s, hv);
                        // clang-format on
                    };
                    auto comp = [](const pair_t& v1, const pair_t& v2) {
                    // clang-format off
                    return v1.second > v2.second;
                        // clang-format on
                    };
                    auto m = tools::par::max(init, this->pareto_data().size(), body, comp);

                    // maximize with NSGA-II
                    auto body2 = [&](int i) {
                    // clang-format off
                    auto x = std::get<0>(this->pareto_model()[i]);
                    float hv = acqui(x);
                    return std::make_pair(x, hv);
                        // clang-format on
                    };
                    auto m2 = tools::par::max(init, this->pareto_model().size(), body2, comp);

                    // take the best
                    std::cout << "best (cmaes):" << m.second << std::endl;
                    std::cout << "best (NSGA-II):" << m2.second << std::endl;
                    if (m2.second > m.second)
                        m = m2;

                    std::cout << "sample selected" << std::endl;
                    Eigen::VectorXd new_sample = m.first;
                    std::cout << "new sample:" << new_sample.transpose() << std::endl;

                    std::cout << "expected improvement: " << acqui(new_sample) << std::endl;
                    std::cout << "expected value: " << this->_models[0].mu(new_sample) << " "
                              << this->_models[1].mu(new_sample) << " "
                              << this->_models[0].sigma(new_sample) << " "
                              << this->_models[1].sigma(new_sample) << std::endl;
                    std::cout << "opt done" << std::endl;

                    // delete pop
                    for (auto x : pop)
                        delete x;

                    // add sample
                    this->add_new_sample(new_sample, feval(new_sample));
                    std::cout
                        << this->_iteration << " | new sample:" << new_sample.transpose()
                        << " => "
                        << this->_observations[this->_observations.size() - 1].transpose()
                        << std::endl;

                    this->_update_stats(*this, false);
                    this->_iteration++;
                }
            }

        protected:
            // former way to deal with the template of RefreshStat
            /*    void _update_stats() {
                boost::fusion::for_each(this->_stat, RefreshStat_f<Nsbo>(*this));
                }*/
        };
    }
}

#endif
