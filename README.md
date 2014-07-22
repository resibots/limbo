limbo
=====

A lightweight framework for Bayesian and model-based optimisation of black-box functions (C++11).

Authors
------
- Antoine Cully (Pierre & Marie University): http://www.isir.upmc.fr/?op=view_profil&lang=fr&id=278
- Jean-Baptiste Mouret (Pierre & Marie University): http://pages.isir.upmc.fr/~mouret/website/

Main features
-------------
- Bayesian optimisation based on Gaussian processes
- Parego (Multi-objective optimization)
- Generic framework (template-based), which allows easy customization for testing original ideas
- Can exploit multicore computers

Main references
---------------

- *General introduction:* Brochu, E., Cora, V. M., & De Freitas, N. (2010). A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. /arXiv preprint arXiv:1012.2599/.

- *Gaussian Processes (GP)*: Rasmussen, C. A, Williams C. K. I. (2006). /Gaussian Processes for Machine Learning./ MIT Press. 

- *Optimizing hyperparameters:* Blum, M., & Riedmiller, M. (2013). Optimization of Gaussian Process Hyperparameters using Rprop. In European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning.

- Parego (Multi-objective optimization): Knowles, J. (2006). ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. /Evolutionary Computation, IEEE Transactions on/, 10(1), 50-66.

- CMA-ES (inner optimization): Auger, A., & Hansen, N. (2005, September). A restart CMA evolution strategy with increasing population size. In /Evolutionary Computation, 2005. The 2005 IEEE Congress on/ (Vol. 2, pp. 1769-1776). IEEE.

Other libraries
---------------
Limbo is a framework for our research that is voluntarily kept small. It is designed to be very fast and flexible, but it does not aim at covering every possible use case for Bayesian optimization.

If you need a more full-featured library, check:
- BayesOpt: http://rmcantin.bitbucket.org/html/
- libGP (no optimization): https://github.com/genix/libGP

