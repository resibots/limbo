#ifndef LIMBO_ACQUI_EHVI_HPP
#define LIMBO_ACQUI_EHVI_HPP

#include <vector>

#include <Eigen/Core>

#include <ehvi/ehvi_calculations.h>

namespace limbo {
    namespace acqui {
        // only work in 2D for now
        template <typename Params, typename Model>
        class Ehvi {
        public:
            Ehvi(const std::vector<Model>& models, const std::deque<individual*>& pop,
                const Eigen::VectorXd& ref_point)
                : _models(models), _pop(pop), _ref_point(ref_point)
            {
                assert(_models.size() == 2);
            }

            size_t dim() const { return _models[0].dim(); }

            double operator()(const Eigen::VectorXd& v) const
            {
                assert(_models.size() == 2);
                double r[3] = {_ref_point(0), _ref_point(1), _ref_point(2)};
                double mu[3] = {_models[0].mu(v), _models[1].mu(v), 0};
                double s[3] = {_models[0].sigma(v), _models[1].sigma(v), 0};
                double ehvi = ehvi2d(_pop, r, mu, s);
                return ehvi;
            }

        protected:
            const std::vector<Model>& _models;
            const std::deque<individual*>& _pop;
            Eigen::VectorXd _ref_point;
        };
    }
}

#endif
