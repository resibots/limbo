//#define SHOW_TIMER
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#include "limbo/limbo.hpp"

using namespace limbo;

struct Params {

    struct kf_exp {
        BO_PARAM(double, sigma, 1);
    };

    struct kf_maternthreehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct kf_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct meanconstant {
        BO_PARAM_VECTOR(double, constant, 0, 0);
    };

    struct meanarchive {
        BO_PARAM_STRING(filename, "missing");
    };

    struct gp_auto : defaults::gp_auto {
    };

    struct gp_auto_mean : defaults::gp_auto_mean {
    };

    struct ucb : public defaults::ucb {
    };

    struct gp_ucb : public defaults::gp_ucb {
    };

    struct exhaustive_search {
        BO_PARAM(int, nb_pts, 20);
    };

    struct cmaes : public defaults::cmaes {
    };

    struct init {
        BO_PARAM(int, nb_samples, 5);
        BO_PARAM(int, nb_bins, 5);
    };

    struct maxiterations {
        BO_PARAM(int, n_iterations, 20);
    };

    struct maxpredictedvalue : public defaults::maxpredictedvalue {
    };

    struct boptimizer {
        BO_PARAM(double, noise, 0.001);
        BO_PARAM(int, dump_period, 1);
    };
};

struct MeanEval {
    MeanEval(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP&) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2.5 * x(0);
        res(1) = -4.5 * x(1);
        return res;
    }
};

struct Average {
    typedef double result_type;
    double operator()(const Eigen::VectorXd& x) const
    {
        return (x(0) + x(1)) / 2;
    }
};

struct StateEval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        res(0) = 3 * x(0) + 5;
        res(1) = -5 * x(1) + 2;
        return res;
    }
};

int main()
{
    typedef kernel_functions::Exp<Params> Kernel_exp_t;
    typedef kernel_functions::MaternThreeHalfs<Params> Kernel_mat_3_t;
    typedef kernel_functions::MaternFiveHalfs<Params> Kernel_mat_5_t;
    typedef kernel_functions::SquaredExpARD<Params> Kernel_sq_exp_ard_t;

    typedef mean_functions::NullFunction<Params> Mean_null_t;
    typedef mean_functions::MeanConstant<Params> Mean_const_t;
    typedef mean_functions::MeanData<Params> Mean_data_t;
    typedef mean_functions::MeanArchive<Params> Mean_arch_t;
    typedef mean_functions::MeanFunctionARD<Params, MeanEval> Mean_ard_t;

    typedef model::GP<Params, Kernel_exp_t, Mean_null_t> GP_exp_null_t;
    typedef model::GP<Params, Kernel_exp_t, Mean_const_t> GP_exp_const_t;
    typedef model::GP<Params, Kernel_exp_t, Mean_data_t> GP_exp_data_t;
    typedef model::GP<Params, Kernel_exp_t, Mean_arch_t> GP_exp_arch_t;
    typedef model::GP<Params, Kernel_exp_t, Mean_ard_t> GP_exp_ard_t;

    typedef model::GP<Params, Kernel_mat_3_t, Mean_null_t> GP_mat3_null_t;
    typedef model::GP<Params, Kernel_mat_3_t, Mean_const_t> GP_mat3_const_t;
    typedef model::GP<Params, Kernel_mat_3_t, Mean_data_t> GP_mat3_data_t;
    typedef model::GP<Params, Kernel_mat_3_t, Mean_arch_t> GP_mat3_arch_t;
    typedef model::GP<Params, Kernel_mat_3_t, Mean_ard_t> GP_mat3_ard_t;

    typedef model::GP<Params, Kernel_mat_5_t, Mean_null_t> GP_mat5_null_t;
    typedef model::GP<Params, Kernel_mat_5_t, Mean_const_t> GP_mat5_const_t;
    typedef model::GP<Params, Kernel_mat_5_t, Mean_data_t> GP_mat5_data_t;
    typedef model::GP<Params, Kernel_mat_5_t, Mean_arch_t> GP_mat5_arch_t;
    typedef model::GP<Params, Kernel_mat_5_t, Mean_ard_t> GP_mat5_ard_t;

    typedef model::GP<Params, Kernel_sq_exp_ard_t, Mean_null_t> GP_sqexpard_null_t;
    typedef model::GP<Params, Kernel_sq_exp_ard_t, Mean_const_t> GP_sqexpard_const_t;
    typedef model::GP<Params, Kernel_sq_exp_ard_t, Mean_data_t> GP_sqexpard_data_t;
    typedef model::GP<Params, Kernel_sq_exp_ard_t, Mean_arch_t> GP_sqexpard_arch_t;
    typedef model::GP<Params, Kernel_sq_exp_ard_t, Mean_ard_t> GP_sqexpard_ard_t;

    typedef model::GPAuto<Params, Kernel_sq_exp_ard_t, Mean_null_t> GPAuto_sqexpard_null_t;
    typedef model::GPAuto<Params, Kernel_sq_exp_ard_t, Mean_const_t> GPAuto_sqexpard_const_t;
    typedef model::GPAuto<Params, Kernel_sq_exp_ard_t, Mean_data_t> GPAuto_sqexpard_data_t;
    typedef model::GPAuto<Params, Kernel_sq_exp_ard_t, Mean_arch_t> GPAuto_sqexpard_arch_t;
    typedef model::GPAuto<Params, Kernel_sq_exp_ard_t, Mean_ard_t> GPAuto_sqexpard_ard_t;

    typedef model::GPAutoMean<Params, Kernel_sq_exp_ard_t, Mean_ard_t> GPAutoMean_sqexpard_ard_t;

    typedef acquisition_functions::UCB<Params, GP_exp_null_t> Acqui_ucb_GP_exp_null_t;
    typedef acquisition_functions::UCB<Params, GP_exp_const_t> Acqui_ucb_GP_exp_const_t;
    typedef acquisition_functions::UCB<Params, GP_exp_data_t> Acqui_ucb_GP_exp_data_t;
    typedef acquisition_functions::UCB<Params, GP_exp_arch_t> Acqui_ucb_GP_exp_arch_t;
    typedef acquisition_functions::UCB<Params, GP_exp_ard_t> Acqui_ucb_GP_exp_ard_t;

    typedef acquisition_functions::UCB<Params, GP_mat3_null_t> Acqui_ucb_GP_mat3_null_t;
    typedef acquisition_functions::UCB<Params, GP_mat3_const_t> Acqui_ucb_GP_mat3_const_t;
    typedef acquisition_functions::UCB<Params, GP_mat3_data_t> Acqui_ucb_GP_mat3_data_t;
    typedef acquisition_functions::UCB<Params, GP_mat3_arch_t> Acqui_ucb_GP_mat3_arch_t;
    typedef acquisition_functions::UCB<Params, GP_mat3_ard_t> Acqui_ucb_GP_mat3_ard_t;

    typedef acquisition_functions::UCB<Params, GP_mat5_null_t> Acqui_ucb_GP_mat5_null_t;
    typedef acquisition_functions::UCB<Params, GP_mat5_const_t> Acqui_ucb_GP_mat5_const_t;
    typedef acquisition_functions::UCB<Params, GP_mat5_data_t> Acqui_ucb_GP_mat5_data_t;
    typedef acquisition_functions::UCB<Params, GP_mat5_arch_t> Acqui_ucb_GP_mat5_arch_t;
    typedef acquisition_functions::UCB<Params, GP_mat5_ard_t> Acqui_ucb_GP_mat5_ard_t;

    typedef acquisition_functions::UCB<Params, GP_sqexpard_null_t> Acqui_ucb_GP_sqexpard_null_t;
    typedef acquisition_functions::UCB<Params, GP_sqexpard_const_t> Acqui_ucb_GP_sqexpard_const_t;
    typedef acquisition_functions::UCB<Params, GP_sqexpard_data_t> Acqui_ucb_GP_sqexpard_data_t;
    typedef acquisition_functions::UCB<Params, GP_sqexpard_arch_t> Acqui_ucb_GP_sqexpard_arch_t;
    typedef acquisition_functions::UCB<Params, GP_sqexpard_ard_t> Acqui_ucb_GP_sqexpard_ard_t;

    typedef acquisition_functions::UCB<Params, GPAuto_sqexpard_null_t> Acqui_ucb_GPAuto_sqexpard_null_t;
    typedef acquisition_functions::UCB<Params, GPAuto_sqexpard_const_t> Acqui_ucb_GPAuto_sqexpard_const_t;
    typedef acquisition_functions::UCB<Params, GPAuto_sqexpard_data_t> Acqui_ucb_GPAuto_sqexpard_data_t;
    typedef acquisition_functions::UCB<Params, GPAuto_sqexpard_arch_t> Acqui_ucb_GPAuto_sqexpard_arch_t;
    typedef acquisition_functions::UCB<Params, GPAuto_sqexpard_ard_t> Acqui_ucb_GPAuto_sqexpard_ard_t;

    typedef acquisition_functions::UCB<Params, GPAutoMean_sqexpard_ard_t> Acqui_ucb_GPAutoMean_sqexpard_ard_t;

    typedef acquisition_functions::GP_UCB<Params, GP_exp_null_t> Acqui_gp_ucb_GP_exp_null_t;
    typedef acquisition_functions::GP_UCB<Params, GP_exp_const_t> Acqui_gp_ucb_GP_exp_const_t;
    typedef acquisition_functions::GP_UCB<Params, GP_exp_data_t> Acqui_gp_ucb_GP_exp_data_t;
    typedef acquisition_functions::GP_UCB<Params, GP_exp_arch_t> Acqui_gp_ucb_GP_exp_arch_t;
    typedef acquisition_functions::GP_UCB<Params, GP_exp_ard_t> Acqui_gp_ucb_GP_exp_ard_t;

    typedef acquisition_functions::GP_UCB<Params, GP_mat3_null_t> Acqui_gp_ucb_GP_mat3_null_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat3_const_t> Acqui_gp_ucb_GP_mat3_const_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat3_data_t> Acqui_gp_ucb_GP_mat3_data_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat3_arch_t> Acqui_gp_ucb_GP_mat3_arch_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat3_ard_t> Acqui_gp_ucb_GP_mat3_ard_t;

    typedef acquisition_functions::GP_UCB<Params, GP_mat5_null_t> Acqui_gp_ucb_GP_mat5_null_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat5_const_t> Acqui_gp_ucb_GP_mat5_const_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat5_data_t> Acqui_gp_ucb_GP_mat5_data_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat5_arch_t> Acqui_gp_ucb_GP_mat5_arch_t;
    typedef acquisition_functions::GP_UCB<Params, GP_mat5_ard_t> Acqui_gp_ucb_GP_mat5_ard_t;

    typedef acquisition_functions::GP_UCB<Params, GP_sqexpard_null_t> Acqui_gp_ucb_GP_sqexpard_null_t;
    typedef acquisition_functions::GP_UCB<Params, GP_sqexpard_const_t> Acqui_gp_ucb_GP_sqexpard_const_t;
    typedef acquisition_functions::GP_UCB<Params, GP_sqexpard_data_t> Acqui_gp_ucb_GP_sqexpard_data_t;
    typedef acquisition_functions::GP_UCB<Params, GP_sqexpard_arch_t> Acqui_gp_ucb_GP_sqexpard_arch_t;
    typedef acquisition_functions::GP_UCB<Params, GP_sqexpard_ard_t> Acqui_gp_ucb_GP_sqexpard_ard_t;

    typedef acquisition_functions::GP_UCB<Params, GPAuto_sqexpard_null_t> Acqui_gp_ucb_GPAuto_sqexpard_null_t;
    typedef acquisition_functions::GP_UCB<Params, GPAuto_sqexpard_const_t> Acqui_gp_ucb_GPAuto_sqexpard_const_t;
    typedef acquisition_functions::GP_UCB<Params, GPAuto_sqexpard_data_t> Acqui_gp_ucb_GPAuto_sqexpard_data_t;
    typedef acquisition_functions::GP_UCB<Params, GPAuto_sqexpard_arch_t> Acqui_gp_ucb_GPAuto_sqexpard_arch_t;
    typedef acquisition_functions::GP_UCB<Params, GPAuto_sqexpard_ard_t> Acqui_gp_ucb_GPAuto_sqexpard_ard_t;

    typedef acquisition_functions::GP_UCB<Params, GPAutoMean_sqexpard_ard_t> Acqui_gp_ucb_GPAutoMean_sqexpard_ard_t;

    typedef inner_optimization::Random<Params> In_opt_random_t;
    typedef inner_optimization::ExhaustiveSearch<Params> In_opt_ex_search_t;
    typedef inner_optimization::Cmaes<Params> In_opt_cmaes_t;

    typedef init_functions::NoInit<Params> Init_no_init_t;
    typedef init_functions::RandomSampling<Params> Init_random_t;
    typedef init_functions::RandomSamplingGrid<Params> Init_random_grid_init_t;
    typedef init_functions::GridSampling<Params> Init_grid_t;

    typedef boost::fusion::vector<stat::Acquisitions<Params>> Stat_t;
    typedef boost::fusion::vector<stopping_criterion::MaxIterations<Params>, stopping_criterion::MaxPredictedValue<Params>> Stop_t;

    BOptimizer<Params, model_fun<GP_exp_null_t>, acq_fun<Acqui_ucb_GP_exp_null_t>, inneropt_fun<In_opt_random_t>, init_fun<Init_no_init_t>, stat_fun<Stat_t>, stop_fun<Stop_t>> opt_1;
    opt_1.optimize(StateEval());
    opt_1.best_observation();
    opt_1.best_sample();
    opt_1.optimize(StateEval(), Average(), true);
    opt_1.best_observation(Average());
    opt_1.best_sample(Average());

    BOptimizer<Params, model_fun<GP_exp_const_t>, acq_fun<Acqui_ucb_GP_exp_const_t>, inneropt_fun<In_opt_random_t>, init_fun<Init_no_init_t>, stat_fun<Stat_t>, stop_fun<Stop_t>> opt_2;
    opt_2.optimize(StateEval());
    opt_2.best_observation();
    opt_2.best_sample();
    opt_2.optimize(StateEval(), Average(), true);
    opt_2.best_observation(Average());
    opt_2.best_sample(Average());

    BOptimizer<Params, model_fun<GP_exp_data_t>, acq_fun<Acqui_gp_ucb_GP_exp_data_t>, inneropt_fun<In_opt_random_t>, init_fun<Init_no_init_t>, stat_fun<Stat_t>, stop_fun<Stop_t>> opt_3;
    opt_3.optimize(StateEval());
    opt_3.best_observation();
    opt_3.best_sample();
    opt_3.optimize(StateEval(), Average(), true);
    opt_3.best_observation(Average());
    opt_3.best_sample(Average());

    BOptimizer<Params, model_fun<GP_exp_const_t>, acq_fun<Acqui_ucb_GP_exp_const_t>, inneropt_fun<In_opt_ex_search_t>, init_fun<Init_no_init_t>, stat_fun<Stat_t>, stop_fun<Stop_t>> opt_4;
    opt_4.optimize(StateEval());
    opt_4.best_observation();
    opt_4.best_sample();
    opt_4.optimize(StateEval(), Average(), true);
    opt_4.best_observation(Average());
    opt_4.best_sample(Average());

    BOptimizer<Params, model_fun<GPAutoMean_sqexpard_ard_t>, acq_fun<Acqui_gp_ucb_GPAutoMean_sqexpard_ard_t>, inneropt_fun<In_opt_cmaes_t>, init_fun<Init_random_grid_init_t>, stat_fun<Stat_t>, stop_fun<Stop_t>> opt_100;
    opt_100.optimize(StateEval());
    opt_100.best_observation();
    opt_100.best_sample();
    opt_100.optimize(StateEval(), Average(), true);
    opt_100.best_observation(Average());
    opt_100.best_sample(Average());

    return 0;
}
