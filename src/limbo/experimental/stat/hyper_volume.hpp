#ifndef LIMBO_STAT_HYPER_VOLUME_HPP
#define LIMBO_STAT_HYPER_VOLUME_HPP

#include <limbo/stat/stat_base.hpp>
#include <limbo/experimental/tools/pareto.hpp>
#include <hv/hypervol.h>

namespace limbo {
    namespace experimental {
        namespace stat {
            namespace defaults {
                struct stat_hyper_volume {
                    BO_PARAM_ARRAY(double, ref, 10, 10);
                };
            }

            template <typename Params>
            struct HyperVolume : public limbo::stat::StatBase<Params> {
                template <typename BO, typename AggregatorFunction>
                void operator()(const BO& bo, const AggregatorFunction&, bool blacklisted)
                {
                    if (bo.observations().empty())
                        return;
                    if (!bo.stats_enabled())
                        return;
                    // convert the data to C arrays
                    double** data = new double* [bo.observations().size()];
                    for (size_t i = 0; i < bo.observations().size(); ++i) {
                        size_t dim = bo.observations()[i].size();
                        data[i] = new double[dim];
                        for (size_t k = 0; k < dim; ++k)
                            data[i][k] = bo.observations()[i](k) + Params::stat_hyper_volume::ref(k);
                    }
                    // call the hypervolume by Zitzler
                    int noObjectives = bo.observations()[0].size();
                    int redSizeFront1 = FilterNondominatedSet(data, bo.observations().size(), noObjectives);
                    double hv = CalculateHypervolume(data, redSizeFront1, noObjectives);

                    // write
                    this->_create_log_file(bo, "hypervolume.dat");
                    (*this->_log_file) << bo.current_iteration() << "\t" << hv << std::endl;

                    // free data
                    for (size_t i = 0; i < bo.observations().size(); ++i)
                        delete[] data[i];
                    delete[] data;
                }
            };
        }
    }
}

#endif
