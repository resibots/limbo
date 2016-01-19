#ifndef LIMBO_BAYES_OPT_BO_MULTI_HPP
#define LIMBO_BAYES_OPT_BO_MULTI_HPP
#define VERSION "xxx"

#include <Eigen/Core>

#ifndef USE_SFERES
#warning No sferes
#else
#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/eval/parallel.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/ea/nsga2.hpp>
#endif

#include <limbo/bayes_opt/bo_base.hpp>
#include <limbo/experimental/bayes_opt/pareto.hpp>

namespace limbo {
    namespace experimental {
        namespace bayes_opt {
            namespace multi {
#ifdef USE_SFERES
                struct SferesParams {
                    struct evo_float {
                        typedef sferes::gen::evo_float::mutation_t mutation_t;
                        typedef sferes::gen::evo_float::cross_over_t cross_over_t;
                        SFERES_CONST float cross_rate = 0.5f;
                        SFERES_CONST float mutation_rate = 0.1f;
                        SFERES_CONST float eta_m = 15.0f;
                        SFERES_CONST float eta_c = 10.0f;
                        SFERES_CONST mutation_t mutation_type = sferes::gen::evo_float::polynomial;
                        SFERES_CONST cross_over_t cross_over_type = sferes::gen::evo_float::sbx;
                    };

                    struct pop {
                        SFERES_CONST unsigned size = 100;
                        SFERES_CONST unsigned nb_gen = 1000;
                        SFERES_CONST int dump_period = -1;
                        SFERES_CONST int initial_aleat = 1;
                    };

                    struct parameters {
                        SFERES_CONST float min = 0.0f;
                        SFERES_CONST float max = 1.0f;
                    };
                };

                template <typename M>
                class SferesFit {
                public:
                    SferesFit(const std::vector<M>& models) : _models(models) {}
                    SferesFit() {}

                    const std::vector<float>& objs() const { return _objs; }

                    float obj(size_t i) const { return _objs[i]; }

                    template <typename Indiv>
                    void eval(const Indiv& indiv)
                    {
                        this->_objs.resize(_models.size());
                        Eigen::VectorXd v(indiv.size());
                        for (size_t j = 0; j < indiv.size(); ++j)
                            v[j] = indiv.data(j);
                        // we protect against overestimation because this has some spurious effect
                        for (size_t i = 0; i < _models.size(); ++i)
                            this->_objs[i] = std::min(_models[i].mu(v), _models[i].max_observation());
                    }

                protected:
                    std::vector<M> _models;
                    std::vector<float> _objs;
                };
#endif
            }

            // clang-format off
            template <class Params,
              class A2 = boost::parameter::void_,
              class A3 = boost::parameter::void_,
              class A4 = boost::parameter::void_,
              class A5 = boost::parameter::void_,
              class A6 = boost::parameter::void_>
            // clang-format on
            class BoMulti : public limbo::bayes_opt::BoBase<Params, A2, A3, A4, A5, A6> {
            public:
                typedef limbo::bayes_opt::BoBase<Params, A2, A3, A4, A5, A6> base_t;
                typedef typename base_t::model_t model_t;
                typedef typename base_t::acqui_optimizer_t acqui_optimizer_t;
                typedef typename base_t::acquisition_function_t acquisition_function_t;
                // point, obj, sigma
                typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> pareto_point_t;
                typedef std::vector<pareto_point_t> pareto_t;

                size_t nb_objs() const { return this->_observations[0].size(); }

                const pareto_t& pareto_model() const { return _pareto_model; }

                const pareto_t& pareto_data() const { return _pareto_data; }

                const std::vector<model_t>& models() const { return _models; }

                // will be called at the end of the algo
                void update_pareto_data()
                {
                    std::vector<Eigen::VectorXd> v(this->_samples.size());
                    size_t dim = this->_observations[0].size();
                    std::fill(v.begin(), v.end(), Eigen::VectorXd::Zero(dim));
                    _pareto_data = pareto::pareto_set<1>(
                        _pack_data(this->_samples, this->_observations, v));
                }

                // will be called at the end of the algo
                template <int D>
                void update_pareto_model()
                {
                    std::cout << "updating models...";
                    std::cout.flush();
                    this->_update_models();
                    std::cout << "ok" << std::endl;
#ifdef USE_SFERES

                    typedef sferes::gen::EvoFloat<D, multi::SferesParams> gen_t;
                    typedef sferes::phen::Parameters<gen_t, multi::SferesFit<model_t>, multi::SferesParams> phen_t;
                    typedef sferes::eval::Parallel<multi::SferesParams> eval_t;
                    typedef boost::fusion::vector<> stat_t;
                    typedef sferes::modif::Dummy<> modifier_t;
                    typedef sferes::ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, multi::SferesParams> nsga2_t;

                    // commented to remove a dependency to a particular version of sferes
                    nsga2_t ea;
                    ea.set_fit_proto(multi::SferesFit<model_t>(_models));
                    ea.run();
                    auto pareto_front = ea.pareto_front();
                    tools::par::sort(pareto_front.begin(), pareto_front.end(), sferes::fit::compare_objs_lex());
                    _pareto_model.resize(pareto_front.size());
                    Eigen::VectorXd point(D), objs(nb_objs()), sigma(nb_objs());
                    for (size_t p = 0; p < pareto_front.size(); ++p) {
                        for (size_t i = 0; i < pareto_front[p]->size(); ++i)
                            point(i) = pareto_front[p]->data(i);
                        for (size_t i = 0; i < nb_objs(); ++i) {
                            objs(i) = pareto_front[p]->fit().obj(i);
                            sigma(i) = _models[i].sigma(point);
                        }
                        _pareto_model[p] = std::make_tuple(point, objs, sigma);
                    }
#endif
                }

            protected:
                std::vector<model_t> _models;
                pareto_t _pareto_model;
                pareto_t _pareto_data;

                pareto_t _pack_data(const std::vector<Eigen::VectorXd>& points,
                    const std::vector<Eigen::VectorXd>& objs,
                    const std::vector<Eigen::VectorXd>& sigma) const
                {
                    assert(points.size() == objs.size());
                    assert(sigma.size() == objs.size());
                    pareto_t p(points.size());
                    tools::par::loop(0, p.size(), [&](size_t k) {
                        // clang-format off
                    p[k] = std::make_tuple(points[k], objs[k], sigma[k]);
                        // clang-format on
                    });
                    return p;
                }

                void _update_models()
                {
                    size_t dim = this->_samples[0].size();
                    std::vector<std::vector<double>> uni_obs(nb_objs());
                    for (size_t i = 0; i < this->_observations.size(); ++i)
                        for (int j = 0; j < this->_observations[i].size(); ++j)
                            uni_obs[j].push_back(this->_observations[i][j]);
                    std::vector<model_t> models(nb_objs(), model_t(dim));
                    _models = models;
                    for (size_t i = 0; i < uni_obs.size(); ++i)
                        _models[i].compute(this->_samples, uni_obs[i], 1e-5);
                }
            };
        }
    }
}

#endif
