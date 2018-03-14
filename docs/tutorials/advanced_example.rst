Advanced Example
====================

Problem Statement
--------------------------------------------

.. figure:: ../pics/arm.svg
   :alt: 6-DOF planar arm
   :target: ../_images/arm.svg

Let's say we have a planar 6-DOF arm manipulator and we want its end-effector to reach a target position (i.e. catch an object). Now, imagine that a joint is broken (or not working properly). We want to use Bayesian Optimization to learn a compensatory behavior that will allow our arm to reach the target in spite of damage. Moreover, once we have learned how to reach one target, we want to reach another (different) one. We will use state-based bayesian optimization to transfer knowledge between the two tasks. We will use the following:

- The forward kinematic model as prior knowledge (mean of Gaussian Process).
- The **Squared exponential covariance function with automatic relevance detection** as the kernel function of the GP.
- Likelihood optimization for the hyperparameters of the GP kernel.
- Use Expected Improvement as the acquisition function.
- Use different optimizer for the acquisition optimization.
- Initialize the GP with random samples.
- Custom stopping criterion.

Also, we assume that all of our arm's links are **1cm long**.

Basic Layout
-----------------------------------

The file structure should look like this: ::

  limbo
  |-- exp
       |-- arm_example
            +-- wscript
            +-- main.cpp
  |-- src
  ...

The basic layout of your ``main.cpp`` file should look like this:

.. code-block:: c++

    #include <limbo/limbo.hpp>

    using namespace limbo;

    struct Params {
      // Here go the parameters
    };

    template <typename Params>
    struct eval_func {
      BO_PARAM(size_t, dim_in, sample_dimensions);
      BO_PARAM(size_t, dim_out, output_dimensions);
      // Here we define the evaluation function
    };

    int main(int argc, char** argv)
    {
      // Defines, etc.
      bayes_opt::BOptimizer<Params, ...> opt;
      opt.optimize(eval_func<Params, ...>());
      auto val = opt.best_observation();
    }

The ``wscript`` will have the following form:

.. code-block:: python

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
   :lines: 85-112

To make this forward kinematic model useful to our GP, we need to create a mean function:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 114-124

Using State-based bayesian optimization
-----------------------------------------
See the explanation of the meaning of :ref:`state-based-bo`.

Creating an Aggregator:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 137-149


Here, we are using a very simple aggregator that simply computes the distance between the end-effector and the target position.

Adding custom stop criterion
-------------------------------

When our bayesian optimizer finds a solution that the end-effector of the arm is reasonably close to the target, we want it to stop. We can easily do this by creating our own stopping criterion:


.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 126-135

Creating the evaluation function
-----------------------------------------

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 151-166

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

  using acqui_t = acqui::EI<Params, gp_t>;
  using acqui_opt_t = opt::Cmaes<Params>;

**Initialization alias:** ::

  using init_t = init::RandomSampling<Params>;

**Stopping criteria alias:** ::

  using stop_t = boost::fusion::vector<stop::MaxIterations<Params>, MinTolerance<Params>>;

**Statistics alias:** ::

  using stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>,
    stat::Samples<Params>, stat::Observations<Params>,
    stat::AggregatedObservations<Params>, stat::GPAcquisitions<Params>,
    stat::BestAggregatedObservations<Params>, stat::GPKernelHParams<Params>>;

Setting the parameter structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 50-84

Creating and running the Bayesian Optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In your main function, you need to have something like the following:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 168-199

Running the experiment
^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, from the root of limbo, run a build command, with the additional switch ``--exp arm_example``: ::

   ./waf configure --exp arm_example
   ./waf build --exp arm_example

.. highlight:: none

Then, an executable named ``arm_example`` should be produced under the folder ``build/exp/arm_example``. When running the experiment, you should expect something like the following: ::

  0 new point:  0.99374 0.999401 0.999716 0.532902 0.999337 0.648682 value: -0.73579 best:-0.73579
  1 new point: 0.00924137    0.52356   0.992816   0.591639   0.900581 0.00022477 value: -1.59172 best:-0.73579
  2 new point:  0.999114  0.303668 0.0132791  0.792124  0.391522  0.999149 value: -3.00304 best:-0.73579
  3 new point: 0.990481 0.981046 0.379883 0.999432 0.282599 0.291695 value: -0.824245 best:-0.73579
  4 new point: 0.00771956   0.949676   0.956241  0.0293142   0.244216  0.0891216 value: -2.68335 best:-0.73579
  5 new point:   0.972554 0.00021536   0.998559      0.808   0.346161 0.00134114 value: -3.7039 best:-0.73579
  6 new point: 0.00275069   0.495195 0.00167023   0.994612   0.631628   0.707545 value: -1.97244 best:-0.73579
  7 new point:  0.124159  0.262741  0.303586  0.999707  0.335987 0.0192833 value: -1.28697 best:-0.73579
  8 new point:  0.026011  0.307255  0.101375 0.0195426  0.562741 0.0400001 value: -2.31361 best:-0.73579
  9 new point:    0.153818  0.00117556 9.92801e-05    0.376417     0.18015  0.00215051 value: -1.00521 best:-0.73579
  10 new point:  0.107636  0.710152   0.41314 0.0703153  0.646439  0.606494 value: -0.685661 best:-0.685661
  11 new point: 0.282632 0.794559 0.940368 0.530688 0.113832 0.439228 value: -1.44821 best:-0.685661
  12 new point: 0.0110291  0.266178  0.576008  0.425873  0.120849  0.444479 value: -1.38147 best:-0.685661
  13 new point:  0.109448 0.0548453  0.458707  0.487198  0.739701  0.588758 value: -0.147514 best:-0.147514
  14 new point: 0.0877909 0.0481023  0.837642  0.438223  0.387531  0.649942 value: -0.150097 best:-0.147514
  15 new point: 0.111047 0.633206 0.509962 0.443725 0.359951 0.243446 value: -0.780535 best:-0.147514
  16 new point:  0.0827364   0.186448   0.333666   0.839036   0.536232 0.00476438 value: -0.575265 best:-0.147514
  17 new point:   0.123347 0.00377535   0.554967   0.103699   0.371233   0.233517 value: -0.00982979 best:-0.00982979
  New target!
  18 new point:    0.998604    0.984174    0.999353 0.000200245    0.356422    0.999792 value: -1.38941 best:-0.370924
  19 new point: 0.0189988  0.999921  0.942007  0.665937  0.999451  0.986427 value: -3.16331 best:-0.370924
  20 new point: 0.0688255  0.230665 0.0273747  0.270297  0.980095  0.990872 value: -2.89225 best:-0.370924
  21 new point: 0.997416 0.857708 0.998912 0.984197 0.391337 0.114332 value: -0.883913 best:-0.370924
  22 new point:    0.999982 0.000168928    0.999398    0.336817    0.258304    0.929625 value: -1.19557 best:-0.370924
  23 new point: 0.0700344  0.281842  0.919103 0.0183289  0.074567  0.970264 value: -4.17297 best:-0.370924
  24 new point: 7.78708e-05 2.18076e-06    0.983852     0.99996    0.825274    0.612332 value: -2.57724 best:-0.370924
  25 new point: 0.000101799    0.997473    0.797134    0.994634    0.377403 3.70205e-05 value: -2.33882 best:-0.370924
  26 new point: 0.0467502  0.327915  0.235275  0.966877 0.0363554  0.909477 value: -3.49153 best:-0.370924
  27 new point:    0.999615 0.000340133    0.637717    0.994796  0.00143888    0.464556 value: -2.75311 best:-0.370924
  28 new point:   0.998958 0.00509838   0.474495   0.667517   0.532318   0.520064 value: -0.038073 best:-0.038073


Using state-based bayesian optimization, we can transfer what we learned during one task to achieve faster new tasks.

Full ``main.cpp``:

.. literalinclude:: ../../src/tutorials/advanced_example.cpp
   :language: c++
   :linenos:
   :lines: 47-
