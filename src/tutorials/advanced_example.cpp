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
// please see the explanation in the documentation
#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(int, hp_period, 10);
    };
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, true);
    };
    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };
    struct stop_mintolerance {
        BO_PARAM(double, tolerance, -0.1);
    };
    struct acqui_ei {
        BO_PARAM(double, jitter, 0.0);
    };
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct opt_rprop : public defaults::opt_rprop {
    };
    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
    struct opt_cmaes : public defaults::opt_cmaes {
        BO_PARAM(int, restarts, 1);
        BO_PARAM(int, max_fun_evals, -1);
    };
};

Eigen::Vector2d forward_kinematics(const Eigen::VectorXd& x)
{
    Eigen::VectorXd rads = x * 2 * M_PI;

    Eigen::MatrixXd dh_mat(6, 4);

    dh_mat << rads(0), 0, 1, 0,
        rads(1), 0, 1, 0,
        rads(2), 0, 1, 0,
        rads(3), 0, 1, 0,
        rads(4), 0, 1, 0,
        rads(5), 0, 1, 0;

    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity(4, 4);

    for (int i = 0; i < dh_mat.rows(); i++) {
        Eigen::VectorXd dh = dh_mat.row(i);

        Eigen::Matrix4d submat;
        submat << cos(dh(0)), -cos(dh(3)) * sin(dh(0)), sin(dh(3)) * sin(dh(0)), dh(2) * cos(dh(0)),
            sin(dh(0)), cos(dh(3)) * cos(dh(0)), -sin(dh(3)) * cos(dh(0)), dh(2) * sin(dh(0)),
            0, sin(dh(3)), cos(dh(3)), dh(1),
            0, 0, 0, 1;
        mat = mat * submat;
    }

    return (mat * Eigen::Vector4d(0, 0, 0, 1)).head(2);
}

template <typename Params>
struct MeanFWModel : mean::BaseMean<Params> {
    MeanFWModel(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP&) const
    {
        Eigen::VectorXd pos = forward_kinematics(x);
        return pos;
    }
};

template <typename Params>
struct MinTolerance {
    MinTolerance() {}

    template <typename BO, typename AggregatorFunction>
    bool operator()(const BO& bo, const AggregatorFunction& afun)
    {
        return afun(bo.best_observation(afun)) > Params::stop_mintolerance::tolerance();
    }
};

template <typename Params>
struct DistanceToTarget {
    using result_type = double;
    DistanceToTarget(const Eigen::Vector2d& target) : _target(target) {}

    double operator()(const Eigen::VectorXd& x) const
    {
        return -(x - _target).norm();
    }

protected:
    Eigen::Vector2d _target;
};

template <typename Params>
struct eval_func {
    BO_PARAM(size_t, dim_in, 6);
    BO_PARAM(size_t, dim_out, 2);

    eval_func() {}

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd xx = x;
        // blocked joint
        xx(1) = 0;
        Eigen::VectorXd grip_pos = forward_kinematics(xx);
        return grip_pos;
    }
};

int main()
{
    using kernel_t = kernel::SquaredExpARD<Params>;

    using mean_t = MeanFWModel<Params>;

    using gp_opt_t = model::gp::KernelLFOpt<Params>;

    using gp_t = model::GP<Params, kernel_t, mean_t, gp_opt_t>;

    using acqui_t = acqui::EI<Params, gp_t>;
    using acqui_opt_t = opt::Cmaes<Params>;

    using init_t = init::RandomSampling<Params>;

    using stop_t = boost::fusion::vector<stop::MaxIterations<Params>, MinTolerance<Params>>;

    using stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>, stat::Samples<Params>, stat::Observations<Params>, stat::AggregatedObservations<Params>, stat::GPAcquisitions<Params>, stat::BestAggregatedObservations<Params>, stat::GPKernelHParams<Params>>;

    bayes_opt::BOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, acquiopt<acqui_opt_t>, initfun<init_t>, statsfun<stat_t>, stopcrit<stop_t>> boptimizer;
    // Instantiate aggregator
    DistanceToTarget<Params> aggregator({1, 1});
    boptimizer.optimize(eval_func<Params>(), aggregator);
    std::cout << "New target!" << std::endl;
    aggregator = DistanceToTarget<Params>({1.5, 1});
    // Do not forget to pass `false` as the last parameter in `optimize`,
    // so you do not reset the data in boptimizer
    // i.e. keep all the previous data points in the Gaussian Process
    boptimizer.optimize(eval_func<Params>(), aggregator, false);
    return 1;
}
