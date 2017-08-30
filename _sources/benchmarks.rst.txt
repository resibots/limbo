Benchmarking with other Bayesian Optimization/Gaussian Processes Libraries
==========================================================================

In this section, we will compared the performance (both in accuracy and speed) of our library to the one of other Bayesian Optimization (BO) and Gaussian Processes (GP) libraries in several test functions.

.. _bench_bayes_opt:

BayesOpt C++ Bayesian Optimization Library
-------------------------------------------

**Last updated: 27/05/2017**

The BayesOpt library and information about it can be found `here <https://bitbucket.org/rmcantin/bayesopt>`_.

Parameters/Setup
~~~~~~~~~~~~~~~~~

Limbo was configured to replicate BayesOpt's default parameters:


+------------------------+---------------------------------------+
|      **Kernel\***      | Matern5 (:math:`\sigma^2 = 1, l = 1`) |
+------------------------+---------------------------------------+
|**Acquisition Function**|       UCB (:math:`\alpha = 0.125`)    |
+------------------------+---------------------------------------+
|   **Initialization**   |     RandomSampling (10 Samples)       |
+------------------------+---------------------------------------+
|    **Mean function**   |         Constant (value of 1)         +
+------------------------+---------------------------------------+
|    **Sample noise**    |                 1e-10                 +
+------------------------+---------------------------------------+
|   **Max iterations**   |                  190                  +
+------------------------+---------------------------------------+

**\*** *When the hyperparameters are optimized, the kernel parameters are learnt.*

The acquisition function is optimized as follows: an outer optimization process uses **DIRECT** for :math:`225d` iterations (where :math:`d` is the input dimension) and the solution found is fed as the initial point to an inner optimization process that uses **BOBYQA** for :math:`25d` iterations.

Results
~~~~~~~~

.. figure:: ./pics/benchmark_limbo_bayes_opt.png
   :alt: Benchmarks vs BayesOpt C++ library
   :target: ./_images/benchmark_limbo_bayes_opt.png

Two configurations are tested: with and without optimization of the hyper-parameters of the Gaussian Process. Each experiment has been replicated 250 times. The median of the data is pictured with a thick dot, while the box represents the first and third quartiles. The most extreme data points are delimited by the whiskers and the outliers are individually depicted as smaller circles. According to the benchmarks we performed (see figure above), Limbo finds solutions with the same level of quality as BayesOpt, within a significantly lower amount of time: for the same accuracy (less than 2.10−3 between the optimized solutions found by Limbo and BayesOpt), Limbo is between 1.47 and 1.76 times faster (median values) than BayesOpt when the hyperparameters are not optimized, and between 2.05 and 2.54 times faster when they are.
