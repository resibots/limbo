limbo [![Build Status](https://travis-ci.org/resibots/limbo.svg?branch=master)](https://travis-ci.org/resibots/limbo)
=====

A lightweight framework for Bayesian optimization of black-box functions (C++11) and, more generally, for data-efficient optimization. It is designed to be very fast and very flexible.

Documentation
-------------
Documentation is available here: http://www.resibots.eu/limbo

Authors
------
- Antoine Cully (Imperial College): http://www.isir.upmc.fr/?op=view_profil&lang=fr&id=278
- Jean-Baptiste Mouret (Inria): http://pages.isir.upmc.fr/~mouret/website/
- Konstantinos Chatzilygeroudis (Inria)
- Federico Allocati (Inria)

Main features
-------------
- Implementation of the classic algorithms (Bayesian optimization, many kernels, likelihood maximization, etc.)
- Modern C++-11
- Generic framework (template-based / policy-based design), which allows for easy customization, to test novel ideas
- Experimental framework that allows user to easily test variants of experiments, compare treatments, submit jobs to clusters (OAR scheduler), etc.
- High performance (in particular, Limbo can exploit multicore computers via Intel TBB and vectorize some operations via Eigen3)
- Purposely small to be easily maintained and quickly understood

Scientific articles that use Limbo
--------------------------------
Cully, A., Clune, J., Tarapore, D., & Mouret, J. B. (2015). Robots that can adapt like animals. *Nature*, 521(7553), 503-507.
