This page presents benchmarks in which we compare the Bayesian optimization performance of **Limbo** against BayesOpt (https://github.com/rmcantin/bayesopt , a state-of-the-art Bayesian Optimization library in C++).

Each library is given 200 evaluations (10 random samples + 190 function evaluations) to find the optimum of the hidden function. We compare both the accuracy of the obtained solution (difference with the actual optimum solution) and the time (wall clock time) required by the library to run the optimization process. The results show that while the libraries generate solutions with similar accuracy (they are based on the same algorithm), **Limbo** generates these solutions significantly faster than BayesOpt.

In addition to comparing the performance of the libraries with their default parameter values (and evaluating **Limbo** with the same parameters as BayesOpt, see variant: limbo/bench_bayes_def), we also evaluate the performance of multiple variants of **Limbo**, including different acquisition functions (UCB or EI), different inner-optimizers (CMAES or DIRECT) and whether optimizing or not the hyper-parameters of the model. In all the these comparisons, **Limbo** is faster than BayesOpt (for similar results), even when BayesOpt is not optimizing the hyper-parameters of the Gaussian processes.


Details
-------
- We compare to BayesOpt (https://github.com/rmcantin/bayesopt)
- Accuracy: lower is better (difference with the optimum)
- Wall time: lower is better
- In each replicate, 10 random samples + 190 function evaluations
- see `src/benchmarks/limbo/bench.cpp` and `src/benchmarks/bayesopt/bench.cpp`


