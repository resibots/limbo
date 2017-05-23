---
title: 'Limbo: A Fast and Flexible Library for Bayesian Optimization'
tags:
  - Bayesian Optimization
  - Gaussian Processes
  - C++11
authors:
  - name: Antoine Cully
	orcid: 0000-0002-3190-7073
	affiliation: 1
  - name: Konstantinos Chatzilygeroudis
	orcid: 0000-0003-3585-1027
	affiliation: 2, 3, 4
  - name: Federico Allocati
	orcid: 0000-0000-0000-0000
	affiliation: 2, 3, 4
  - name: Jean-Baptiste Mouret
	orcid: 0000-0002-2513-027X
	affiliation: 2, 3, 4
affiliations:
  - name: Personal Robotics Lab, Imperial College London, London, United Kingdom
	index: 1
  - name: Inria Nancy Grand - Est, Villers-lès-Nancy, France
	index: 2
  - name: CNRS, Loria, UMR 7503, Vandœuvre-lès-Nancy, France
	index: 3
  - name: Université de Lorraine, Loria, UMR 7503, Vandœuvre-lès-Nancy, France
	index: 4
formatted_doi:

repository:
  https://github.com/resibots/limbo
archive_doi:

paper_url:

date: 20 Mai 2017
bibliography: paper.bib
output: pdf_document
---

# Summary

Limbo (LIbrary for Model-based Bayesian Optimization) is an open-source C++11 library for Bayesian optimization and Gaussian Processes which is designed to be both highly flexible and very fast. It can be used to optimize functions for which the gradient is unknown, evaluations are expensive, and runtime cost matters (e.g., on embedded systems or robots).
In particular, Bayesian optimization recently attracted a lot of interest for direct policy search in robot learning [@lizotte2007automatic] and online adaptation; for example, it was used to allow a legged robot to learn a new gait after a mechanical damage in about 10-15 trials (2 minutes) [@cully2015robots].

The implementation of Limbo follows a policy-based design [@alexandrescu2001modern] relying on templates, which allows it to be highly flexible without paying the cost induced by classic object-oriented designs [@driesen1996direct] (cost of virtual functions). Benchmarks on standard functions demonstrate that Limbo is about 2 times faster than BayesOpt (another C++ library, [@martinezcantin14a]) for a similar accuracy. In practice, changing one of the components of the algorithms in Limbo (e.g., changing the acquisition function) usually requires changing only a template definition in the source code. This design allows users to rapidly experiment and test new ideas while being as fast as specialized code.

In addition to Bayesian Optimization functions, Limbo implements a high-performing Gaussian Processes (GP) framework including, among other features, incremental Cholesky decomposition for the update of the GP, and automatic adaptation of the hyper-parameters via the optimization of the log-likelihood of the GP.
A special attention has been given to the internal optimization operations that are inherent to Bayesian Optimization and Gaussian processes (e.g., for the optimization of the acquisition function or the for hyper-parameters adaptation). In particular, all these internal optimization steps can be automatically replicated and executed in parallel to mitigate the effects of local optimums. Moreover, optimizers in Limbo are wrappers around standard optimization libraries:

* NLOpt (which provides many local, global, gradient-based, gradient-free algorithms [@nlopt])
* libcmaes (which provides the Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES), and several variants of it [@hansen1996adapting])
* a few other algorithms that are implemented in Limbo (in particular, RPROP [@riedmiller1993direct], which is a gradient-based optimization algorithm)

The library is distributed via a GitHub repository ^[<http://github.com/resibots/limbo>], with an extensive documentation ^[<http://www.resibots.eu/limbo>] containing guides, examples, and tutorials. The code is standard-compliant but it is currently mostly developed for GNU/Linux and Mac OS X with both the GCC and Clang compilers. New contributors can rely on a full API reference, while their developments are checked via a continuous integration platform (automatic unit-testing routines).

# References
