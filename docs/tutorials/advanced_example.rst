Advanced Example
====================

Problem Statement
--------------------------------------------

.. figure:: ../pics/arm.svg
   :alt: 6-DOF planar arm
   :target: ../_images/arm.svg

Let's say we have a planar 6-DOF arm manipulator and we want its end-effector to reach a target position (i.e. catch an object). Now, imagine that a joint is broken (or not working properly). We want to use Bayesian Optimization to learn a compensatory behavior that will allow our arm to reach the target in spite of damage. Now, assume that once we have learned how to reach one target, we want to reach another (different) one. We will use state-based bayesian optimization to transfer knowledge between the two tasks. We will use the following:

- The forward kinematic model as prior knowledge (mean of GP).
- The ``Squared exponential covariance function with automatic relevance detection`` as kernel of the GP.
- Optimize the hyperparameters of the GP kernel using likelihood optimization.
- Custom output statistics.
- Use different optimizer for the acquisition optimization.
- Initialize the GP with random samples.
- Custom stopping criterion.

Also, we assume that all of our arm's links have **length of 1cm**.

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

To compute the forward kinematics of our simple planar arm we use the following code:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 38-65

To make this forward kinematic model useful to our GP, we need to create a mean function:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 67-77

Using State-based bayesian optimization
-----------------------------------------

Creating an Aggregator:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 90-102


Here, we are using a very simple aggregator that simply computes the distance between the end-effector and the target position.

Adding custom stop criterion
-------------------------------

When our bayesian optimizer finds a solution that the end-effector of the arm is reasonably close to the target, we want it to stop. We can easily do this by creating our own stopping criterion:


.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 79-88

Creating the evaluation function
-----------------------------------------

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 104-119

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
.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 6-36

Creating and running the Bayesian Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your main function, you need to have something like the following:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 121-148
