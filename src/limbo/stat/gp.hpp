#ifndef LIMBO_STAT_GP_HPP
#define LIMBO_STAT_GP_HPP

#include <cmath>

#include <limbo/stat/stat_base.hpp>

namespace limbo {
    namespace stat {
        /// @ingroup stat
        /// filename: `gp_<iteration>.dat`
        template <typename Params>
        struct GP : public StatBase<Params> {
            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
            {
                if (!bo.stats_enabled())
                    return;
                std::string  fname = bo.res_dir() + "/" +
                  "gp_" + std::to_string(bo.total_iterations()) + ".dat";
                std::ofstream ofs(fname.c_str());
                _explore(0, ofs, bo, afun, Eigen::VectorXd::Constant(bo.model().dim_in(), 0));
            }
          protected:
            // recursively explore all the dimensions
            template <typename BO, typename AggregatorFunction>
            void _explore(int dim_in, std::ofstream& ofs,
                          const BO& bo, const AggregatorFunction& afun,
                          const Eigen::VectorXd& current) const
            {
                for (double x = 0; x <= 1.0f; x += 1.0f / (double)Params::stat_gp::bins()) {
                    Eigen::VectorXd point = current;
                    point[dim_in] = x;
                    if (dim_in == current.size() - 1) {
                        double sigma;
                        Eigen::VectorXd mu;
                        std::tie(mu, sigma) = bo.model().query(point);
                        double acqui = typename BO::acquisition_function_t(bo.model(), bo.current_iteration())(point, afun);
                        ofs << point.transpose() << " "
                            << mu.transpose() << " "
                            << sigma << " "
                            << acqui << std::endl;
                    } else {
                        _explore(dim_in + 1, ofs, bo, afun, point);
                    }
                }
            }
        };
    }
}

#endif
