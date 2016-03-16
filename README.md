limbo
=====

A lightweight framework for Bayesian and model-based optimisation of black-box functions (C++11).

Documentation
-------------
Documentation is available here: http://www.resibots.eu/limbo


Authors
------
- Antoine Cully (Imperial College): http://www.isir.upmc.fr/?op=view_profil&lang=fr&id=278
- Jean-Baptiste Mouret (Inria): http://pages.isir.upmc.fr/~mouret/website/
- Konstantinos Chatzilygeroudis (Inria)
- Federico Allocai (Inria)
- 
Main features
-------------
- Bayesian optimisation based on Gaussian processes
- Generic framework (template-based), which allows easy customization for testing original ideas
- Can exploit multicore computers
- Experimental support for some multi-objective algorithms

Main references
---------------

- **General introduction:** Brochu, E., Cora, V. M., & De Freitas, N. (2010). A tutorial on Bayesian optimization of expensive cost functions, with application to active user modeling and hierarchical reinforcement learning. *arXiv preprint arXiv:1012.2599*.

- **Gaussian Processes (GP)**: Rasmussen, C. A, Williams C. K. I. (2006). /Gaussian Processes for Machine Learning./ MIT Press. 

- **Optimizing hyperparameters:** Blum, M., & Riedmiller, M. (2013). Optimization of Gaussian Process Hyperparameters using Rprop. In *European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning*.

- **Parego (Multi-objective optimization):** Knowles, J. (2006). ParEGO: A hybrid algorithm with on-line landscape approximation for expensive multiobjective optimization problems. *Evolutionary Computation, IEEE Transactions on*, 10(1), 50-66.

- **CMA-ES (inner optimization):** Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with increasing population size. In *Evolutionary Computation, 2005. The 2005 IEEE Congress on* (Vol. 2, pp. 1769-1776). IEEE.

- **Expected hypervolume improvement (multi-objective optimization):** Hupkens, I., Emmerich, M. T. M., Deutz A. H. (2014). Faster Computation of Expected Hypervolume Improvement. arXiv: http://arxiv.org/abs/1408.7114


Other libraries
---------------
Limbo is a framework for our research that is voluntarily kept small. It is designed to be very fast and flexible, but it does not aim at covering every possible use case for Bayesian optimization.

If you need a more full-featured library, check:
- BayesOpt: http://rmcantin.bitbucket.org/html/
- libGP (no optimization): https://github.com/mblum/libgp

