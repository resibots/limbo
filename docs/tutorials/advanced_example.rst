Advanced Example
====================

Problem Statement
--------------------------------------------

.. figure:: ../pics/arm.svg
   :alt: 6-DOF planar arm
   :target: ../_images/arm.svg

Let's say we have a planar 6-DOF arm manipulator and we want its end-effector to reach a target position (i.e. catch an object). We will use the following:

- The forward kinematic model as prior knowledge (mean of GP).
- The ``Squared exponential covariance function with automatic relevance detection`` as kernel of the GP.
- Optimize the hyperparameters of the kernel using likelihood optimization.
- Custom output statistics.
- Use different optimizer for the acquisition optimization.
- Initialize the GP with random samples.
- Create a custom stopping criterion and using it in the bayesian optimizer.

Basic Layout
-----------------------------------

The file structure should look like this: ::

  limbo
  |-- exp
       |-- arm_example
            +-- wscript
            +-- main.cpp

.. highlight:: c++

The basic layout of your ``main.cpp`` file should look like this: ::

            #include <iostream>
            #include <limbo/bayes_opt/boptimizer.hpp>
            // Here we have to include other needed limbo headers

            using namespace limbo;

            struct Params {
              // Here go the parameters
            };

            template <typename Params>
            struct eval_func {
              static constexpr int dim_in = sample_dimensions;
              static constexpr int dim_out = output_dimensions;
              // Here we define the function evaluation
            };

            int main(int argc, char** argv)
            {
              // Defines, etc.
              bayes_opt::BOptimizer<Params, ...> opt;
              opt.optimize(eval_func<Params, ...>());
              auto val = opt.best_observation();
            }

The ``wscript`` will have the following form:

.. highlight:: python

.. code:: python

    from waflib.Configure import conf

    def options(opt):
        pass

    @conf
    def configure(blah):
        pass

    def build(bld):
        bld(features='cxx cxxprogram',
            source='main.cpp',
            includes='. ../../src',
            target='arm_example',
            uselib='BOOST EIGEN TBB LIBCMAES NLOPT',
            use='limbo')

Adding the forward kinematic model as prior
----------------------------------------------

.. highlight:: c++

To compute the forward kinematics of our simple planar arm we use the following code: ::

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

      for (size_t i = 0; i < dh_mat.rows(); i++) {
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

To make this forward kinematic model useful to our GP, we need to create a mean function: ::

  template <typename Params>
  struct MeanFWModel {
      MeanFWModel(size_t dim_out = 1) {}

      template <typename GP>
      Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP&) const
      {
          Eigen::VectorXd pos = forward_kinematics(x);
          return pos;
      }
  };

Using State-based bayesian optimization
-----------------------------------------

Creating an Aggregator: ::

  template <typename Params>
  struct DistanceToTarget {
    typedef double result_type;
    DistanceToTarget(const Eigen::Vector2d& target) : _target(target) {}

    double operator()(const Eigen::VectorXd& x) const
    {
        return -(x - _target).norm();
    }

  protected:
    Eigen::Vector2d _target;
  };

Here, we are using a very simple aggregator that simply computes the distance between the end-effector and the target position.

Adding custom stop criterion
-------------------------------

When our bayesian optimizer finds a solution that the end-effector of the arm is reasonably close to the target, we want it to stop. We can easily do this by creating our own stopping criterion: ::

  template <typename Params>
  struct MinTolerance {
      MinTolerance() {}

      template <typename BO, typename AggregatorFunction>
      bool operator()(const BO& bo, const AggregatorFunction& afun)
      {
          return afun(bo.best_observation(afun)) > Params::stop_mintolerance::tolerance();
      }
  };

Creating the evaluation function
-----------------------------------------

::

  template <typename Params>
  struct eval_func {
      static constexpr int dim_in = 6;
      static constexpr int dim_out = 2;

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

Creating the experiment
-------------------------------------------------

Creating the GP model
^^^^^^^^^^^^^^^^^^^^^^^

**Kernel alias:** ::

  using kernel_t = kernel::SquaredExpARD<Params>;

**Mean alias:** ::

  using mean_t = MeanFWModel<Params>;

**Likelihood optimization alias:** ::

  using gp_opt_t = model::gp::KernelLFOpt<Params>;

**GP alias:** ::

  using gp_t = model::GP<Params, kernel_t, mean_t, gp_opt_t>;

Acquisition, Initialization and other aliases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Acquisition aliases:** ::

  using acqui_t = acqui::UCB<Params, gp_t>;
  using acqui_opt_t = opt::Cmaes<Params>;

**Initialization alias:** ::

  using init_t = init::RandomSampling<Params>;

**Stopping criteria alias:** ::

  using stop_t = boost::fusion::vector<stop::MaxIterations<Params>, MinTolerance<Params>>;

**Statistics alias:** ::

  using stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>, stat::Samples<Params>, stat::Observations<Params>, stat::AggregatedObservations<Params>, stat::GPAcquisitions<Params>, stat::BestAggregatedObservations<Params>, stat::GPKernelHParams<Params>>;

Setting the parameter structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

  struct Params {
    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.0);
    };
    struct bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, true);
    };
    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };
    struct stop_mintolerance {
          BO_PARAM(double, tolerance, -0.02);
    };
    struct acqui_ucb {
        BO_PARAM(double, alpha, 0.4);
    };
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
    struct opt_rprop : public defaults::opt_rprop {
    };
    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
    struct opt_cmaes {
        BO_PARAM(int, restarts, 1);
        BO_PARAM(int, max_fun_evals, -1);
    };
  };

Creating and running the Bayesian Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your main function, you need to have something like the following: ::

  // includes
  // parameter structure

  int main(int argc, char** argv)
  {
    // aliases
    bayes_opt::BOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, acquiopt<acqui_opt_t>, initfun<init_t>, statsfun<stat_t>, stopcrit<stop_t>> boptimizer;
    // Instantiate aggregator
    DistanceToTarget<Params> aggregator({-1.5, -1.5});
    boptimizer.optimize(eval_func(), aggregator);
    // Adding new target
    aggregator = DistanceToTarget<Params>({3, 1.5});
    boptimizer.optimize(eval_func<Params>(), aggregator);
    // rest of code
  }


Running the experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, from the root of limbo, run a build command, with the additional switch ``--exp arm_example``: ::

    ./waf configure --exp arm_example
    ./waf build --exp arm_example

Then, an executable named ``arm_example`` should be produced under the folder ``build/exp/arm_example``. When running the experiment, you should expect something like the following: ::

  0 new point:    0.131781    0.554741     0.43973    0.448952    0.551195 0.000187801 value: -2.21293 best:-1.24215
  1 new point:  0.125424 0.0152663  0.999616  0.523596  0.832982  0.513717 value: -0.0971545 best:-0.0971545
  2 new point: 7.72149e-05    0.305844    0.707572    0.281713    0.662167    0.594993 value: -2.44114 best:-0.0971545
  3 new point: 0.161434 0.900689 0.999596 0.378527 0.538626 0.576684 value: -0.718139 best:-0.0971545
  4 new point:  0.157492  0.658696 0.0234359  0.344446  0.995436  0.212072 value: -2.0776 best:-0.0971545
  5 new point:  0.902813  0.227166 0.0309232   0.98105  0.543195   0.70282 value: -2.86496 best:-0.0971545
  6 new point: 0.0149745  0.096635 0.0331401  0.567209  0.776161  0.708063 value: -0.864753 best:-0.0971545
  7 new point:  0.0791348 0.00053872   0.633395   0.811948   0.599573  0.0847644 value: -0.00412038 best:-0.00412038
  New target!
  0 new point: 0.00159966   0.968279   0.332128   0.858613   0.999992   0.633525 value: -0.496254 best:-0.494746
  1 new point: 0.000264513     0.99999  0.00765048     0.94105    0.379622    0.999865 value: -0.000907635 best:-0.000907635

Using state-based bayesian optimization, we can transfer what we learned doing one task to learn faster new tasks.

The whole ``main.cpp`` file: ::

  #include <limbo/limbo.hpp>

  using namespace limbo;

  struct Params {
      struct bayes_opt_boptimizer {
          BO_PARAM(double, noise, 0.0);
      };
      struct bayes_opt_bobase {
          BO_PARAM(int, stats_enabled, true);
      };
      struct stop_maxiterations {
          BO_PARAM(int, iterations, 100);
      };
      struct stop_mintolerance {
          BO_PARAM(double, tolerance, -0.02);
      };
      struct acqui_ucb {
          BO_PARAM(double, alpha, 0.4);
      };
      struct init_randomsampling {
          BO_PARAM(int, samples, 10);
      };
      struct opt_rprop : public defaults::opt_rprop {
      };
      struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
      };
      struct opt_cmaes {
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

      for (size_t i = 0; i < dh_mat.rows(); i++) {
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
  struct MeanFWModel {
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
      typedef double result_type;
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
      static constexpr int dim_in = 6;
      static constexpr int dim_out = 2;

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

      using acqui_t = acqui::UCB<Params, gp_t>;
      using acqui_opt_t = opt::Cmaes<Params>;

      using init_t = init::RandomSampling<Params>;

      using stop_t = boost::fusion::vector<stop::MaxIterations<Params>, MinTolerance<Params>>;

      using stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>, stat::Samples<Params>, stat::Observations<Params>, stat::AggregatedObservations<Params>, stat::GPAcquisitions<Params>, stat::BestAggregatedObservations<Params>, stat::GPKernelHParams<Params>>;

      bayes_opt::BOptimizer<Params, modelfun<gp_t>, acquifun<acqui_t>, acquiopt<acqui_opt_t>, initfun<init_t>, statsfun<stat_t>, stopcrit<stop_t>> boptimizer;
      // Instantiate aggregator
      DistanceToTarget<Params> aggregator({1.5, 1.5});
      boptimizer.optimize(eval_func<Params>(), aggregator);
      std::cout << "New target!" << std::endl;
      aggregator = DistanceToTarget<Params>({3, 1.5});
      boptimizer.optimize(eval_func<Params>(), aggregator);
      return 1;
  }
