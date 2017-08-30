Bayesian optimization benchmarks
===============================

*August 04, 2017* -- hal01 (24 cores)

- We compare to BayesOpt (https://github.com/rmcantin/bayesopt) 
- Accuracy: lower is better (difference with the optimum)
- Wall time: lower is better

- In each replicate, 10 random samples + 190 function evaluations
- see `src/benchmarks/limbo/bench.cpp` and `src/benchmarks/bayesopt/bench.cpp`

Naming convention
------------------

- limbo_def: default Limbo parameters

- opt_cmaes: use CMA-ES (from libcmaes) to optimize the acquisition function
- opt_direct: use DIRECT (from NLopt) to optimize the acquisition function
- acq_ucb: use UCB for the acquisition function
- acq_ei: use EI for the acquisition function
- hp_opt: use hyper-parameter optimization
- bayesopt_def: same parameters as default parameters in BayesOpt
branin
-----------------

40 replicates 

.. figure:: fig_benchmarks/branin.png

ellipsoid
-----------------

40 replicates 

.. figure:: fig_benchmarks/ellipsoid.png

goldsteinprice
-----------------

40 replicates 

.. figure:: fig_benchmarks/goldsteinprice.png

hartmann3
-----------------

40 replicates 

.. figure:: fig_benchmarks/hartmann3.png

hartmann6
-----------------

40 replicates 

.. figure:: fig_benchmarks/hartmann6.png

rastrigin
-----------------

40 replicates 

.. figure:: fig_benchmarks/rastrigin.png

sixhumpcamel
-----------------

40 replicates 

.. figure:: fig_benchmarks/sixhumpcamel.png

sphere
-----------------

40 replicates 

.. figure:: fig_benchmarks/sphere.png

