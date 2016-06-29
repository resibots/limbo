#ifndef LIMBO_STAT_PARETO_FRONT_HPP
#define LIMBO_STAT_PARETO_FRONT_HPP

#include <limbo/limbo.hpp>

namespace limbo {
    namespace experimental {
        namespace stat {
            template <typename F>
            struct ParetoBenchmark {
                template <typename BO, typename AggregatorFunction>
                void operator()(BO& opt, const AggregatorFunction& afun, bool blacklisted)
                {
                    opt.update_pareto_data();
#ifndef NSBO // this is already done is NSBO
                    opt.template update_pareto_model<F::dim_in>();
#endif
                    auto dir = opt.res_dir() + "/";
                    auto p_model = opt.pareto_model();
                    auto p_data = opt.pareto_data();
                    std::string it = std::to_string(opt.current_iteration());
                    std::string model = dir + "pareto_model_" + it + ".dat";
                    std::string model_real = dir + "pareto_model_real_" + it + ".dat";
                    std::string data = dir + "pareto_data_" + it + ".dat";
                    std::string obs_f = dir + "obs_" + it + ".dat";
                    std::ofstream pareto_model(model.c_str()), pareto_data(data.c_str()),
                        pareto_model_real(model_real.c_str()), obs(obs_f.c_str());
                    F f;
                    for (auto x : p_model)
                        pareto_model << std::get<1>(x).transpose() << " "
                                     << std::get<2>(x).transpose() << std::endl;
                    for (auto x : p_model)
                        pareto_model_real << f(std::get<0>(x)).transpose() << " " << std::endl;
                    for (auto x : p_data)
                        pareto_data << std::get<1>(x).transpose() << std::endl;
                    for (size_t i = 0; i < opt.observations().size(); ++i)
                        obs << opt.observations()[i].transpose() << " "
                            << opt.samples()[i].transpose() << std::endl;
                    std::cout << "stats done" << std::endl;
                }
            };
        }
    }
}

#endif
