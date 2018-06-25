//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/function_ard.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_mean_lf_opt.hpp>
#include <limbo/tools/macros.hpp>

using namespace limbo;

struct Params {
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif
    struct opt_rprop : public defaults::opt_rprop {
    };

    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.001);
    };

    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };

    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(int, hp_period, 50);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 5);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };
};

template <typename Params, typename Model>
class UCB_multi {
public:
    UCB_multi(const Model& model, int iteration = 0) : _model(model) {}

    size_t dim_in() const { return _model.dim_in(); }

    size_t dim_out() const { return _model.dim_out(); }

    template <typename AggregatorFunction>
    limbo::opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient) const
    {
        assert(!gradient);
        // double mu, sigma;
        // std::tie(mu, sigma) = _model.query(v);
        // return (mu + Params::ucb::alpha() * sqrt(sigma));

        return limbo::opt::no_grad(std::sqrt(_model.sigma(v)));
    }

protected:
    const Model& _model;
};

template <typename Params>
struct MeanOffset : public mean::BaseMean<Params> {
    MeanOffset(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2; // constant overestimation
        res(1) = 2; // constant overestimation

        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        return res;
    }
};

template <typename Params>
struct MeanRotation : public mean::BaseMean<Params> {
    MeanRotation(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
    {
        Eigen::VectorXd res(2);
        res(0) = 0; // constant overestimation
        res(1) = 0; // constant overestimation
        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        double theta = M_PI / 2;
        Eigen::Matrix2d rot;
        rot(0, 0) = cos(theta);
        rot(0, 1) = -sin(theta);
        rot(1, 0) = sin(theta);
        rot(1, 1) = cos(theta);
        return rot * res;
    }
};

template <typename Params>
struct MeanComplet : public mean::BaseMean<Params> {
    MeanComplet(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2; // constant overestimation
        res(1) = 2; // constant overestimation
        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        double theta = M_PI / 2;
        Eigen::Matrix2d rot;
        rot(0, 0) = cos(theta);
        rot(0, 1) = -sin(theta);
        rot(1, 0) = sin(theta);
        rot(1, 1) = cos(theta);
        return rot * res;
    }
};

struct fit_eval {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 2);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        res(0) = 0;
        res(1) = 0;
        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        return res;
    }
};

int main()
{

    using Kernel_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::FunctionARD<Params, MeanComplet<Params>>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t, model::gp::KernelMeanLFOpt<Params>>;
    using Acqui_t = UCB_multi<Params, GP_t>;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>> opt;
    opt.optimize(fit_eval());

    std::cout << opt.best_observation() << " res  "
              << opt.best_sample().transpose() << std::endl;
    return 0;
}
