---
title: 'Limbo: A Flexible High-performance Library for Gaussian Processes modeling and Data-Efficient Optimization'
tags:
 - Gaussian Processes
 - Bayesian Optimization
 - C++11
authors:
 - name: Antoine Cully
   orcid: 0000-0002-3190-7073
   affiliation: 1
 - name: Konstantinos Chatzilygeroudis
   orcid: 0000-0003-3585-1027
   affiliation: "2"
 - name: Federico Allocati
   orcid: 0000-0000-0000-0000
   affiliation: "2"
 - name: Jean-Baptiste Mouret
   orcid: 0000-0002-2513-027X
   affiliation: "2"
affiliations:
 - name: Personal Robotics Lab, Imperial College London, London, United Kingdom
   index: 1
 - name: Inria, CNRS, Université de Lorraine, LORIA, Nancy, France
   index: 2
date: 17th of January, 2018
bibliography: paper.bib
output: pdf_document
---

# Summary

Limbo (LIbrary for Model-Based Optimization) is an open-source C++11 library for Gaussian Processes and data-efficient optimization (e.g., Bayesian optimization, see [@shahriari2016taking]) that is designed to be both highly flexible and very fast. It can be used as a state-of-the-art optimization library or to experiment with novel algorithms with "plugin" components. Limbo is currently mostly used for data-efficient policy search in robot learning [@lizotte2007automatic] and online adaptation because computation time matters when using the low-power embedded computers of robots. For example, Limbo was the key library to develop a new algorithm that allows a legged robot to learn a new gait after a mechanical damage in about 10-15 trials (2 minutes) [@cully2015robots], and a 4-DOF manipulator to learn neural networks policies for goal reaching in about 5 trials [@chatzilygeroudis2017].

The implementation of Limbo follows a policy-based design [@alexandrescu2001modern] that leverages C++ templates: this allows it to be highly flexible without the cost induced by classic object-oriented designs [@driesen1996direct] (cost of virtual functions). The regression benchmarks^[<http://www.resibots.eu/limbo/reg_benchmarks.html>] show that the query time of Limbo's Gaussian processes is several orders of magnitude better than the one of GPy (a state-of-the-art Python library for Gaussian processes^[<https://sheffieldml.github.io/GPy/>]) for a similar accuracy (the learning time highly depends on the optimization algorithm chosen to optimize the hyper-parameters). The black-box optimization benchmarks^[<http://www.resibots.eu/limbo/bo_benchmarks.html>] demonstrate that Limbo is about 2 times faster than BayesOpt (a C++ library for data-efficient optimization, [@martinezcantin14a]) for a similar accuracy and data-efficiency. In practice, changing one of the components of the algorithms in Limbo (e.g., changing the acquisition function) usually requires changing only a template definition in the source code. This design allows users to rapidly experiment and test new ideas while keeping the software as fast as specialized code.

Limbo takes advantage of multi-core architectures to parallelize the internal optimization processes (optimization of the acquisition function, optimization of the hyper-parameters of a Gaussian process) and it vectorizes many of the linear algebra operations (via the Eigen 3 library^[<http://eigen.tuxfamily.org/>] and optional bindings to Intel's MKL). To keep the library lightweight, most of the optimizers in Limbo are wrappers around external optimization libraries:

* NLOpt^[<http://ab-initio.mit.edu/nlopt>] (which provides many local, global, gradient-based, gradient-free algorithms)
* libcmaes^[<https://github.com/beniz/libcmaes>] (which provides the Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES), and several variants of it [@hansen1996adapting])
* a few other algorithms that are implemented in Limbo (in particular, RPROP [@riedmiller1993direct], which is a gradient-based optimization algorithm)

The library is distributed under the CeCILL-C license^[<http://www.cecill.info/index.en.html>] via a GitHub repository ^[<http://github.com/resibots/limbo>], with an extensive documentation ^[<http://www.resibots.eu/limbo>] that contains guides, examples, and tutorials. The code is standard-compliant but it is currently mostly developed for GNU/Linux and Mac OS X with both the GCC and Clang compilers. New contributors can rely on a full API reference, while their developments are checked via a continuous integration platform (automatic unit-testing routines).

Limbo is currently used in the ERC project ResiBots^[<http://www.resibots.eu>], which is focused on data-efficient trial-and-error learning for robot damage recovery, and in the H2020 projet PAL^[<http://www.pal4u.eu/>], which uses social robots to help coping with diabetes. It has been instrumental in many scientific publications since 2015 [@cully2015robots] [@chatzilygeroudis2018resetfree] [@tarapore2016] [@chatzilygeroudis2017] [@pautrat2018bayesian] [@chatzilygeroudis2018using]


# Acknowledgments
This work received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (GA no. 637972, project ``ResiBots'') and from the European Commission through the H2020 projects AnDy (GA no. 731540) and PAL (GA no. 643783).

# References
