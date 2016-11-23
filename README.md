limbo [![Build Status](https://travis-ci.org/resibots/limbo.svg?branch=master)](https://travis-ci.org/resibots/limbo)
=====

A lightweight framework for Bayesian optimization of black-box functions (C++11) and, more generally, for data-efficient optimization. It is designed to be very fast and very flexible.

Documentation & Versions
------------------------
The development branch is the [master](https://github.com/resibots/limbo/tree/master) branch. For the latest stable release, check the [release-1.0](https://github.com/resibots/limbo/tree/release-1.0) branch.
Documentation is available at: http://www.resibots.eu/limbo

A short paper that introduces the library is available on arxiv: https://arxiv.org/abs/1611.07343 

Citing Limbo
------------
If you use Limbo in a scientific paper, please cite:
Cully, A., Chatzilygeroudis, K., Allocati, F., and Mouret J.-B., (2016). [Limbo: A Fast and Flexible Library for Bayesian Optimization](https://arxiv.org/abs/1611.07343). *arXiv preprint arXiv:1611.07343*.

In BibTex:
  
    @article{cully_limbo_2016,
        title={Limbo: A Fast and Flexible Library for Bayesian Optimization},
        author={Cully, A. and Chatzilygeroudis, K. and Allocati, F.  and Mouret, J.-B.},
        year={2016},
        journal={arXiv preprint},
        pages={arxiv:1611.07343}
    }


Authors
------
- Antoine Cully (Imperial College): http://www.antoinecully.com
- Jean-Baptiste Mouret (Inria): http://members.loria.fr/JBMouret
- Konstantinos Chatzilygeroudis (Inria): http://costashatz.github.io/
- Federico Allocati (Inria)

Other contributors
-------------------
- Vaios Papaspyros (Inria)
- Roberto Rama (Inria)

Limbo is partly funded by the ResiBots ERC Project (http://www.resibots.eu).


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
- Cully, A., Clune, J., Tarapore, D., & Mouret, J. B. (2015). [Robots that can adapt like animals](http://www.nature.com/nature/journal/v521/n7553/full/nature14422.html). *Nature*, 521(7553), 503-507.
- Tarapore D, Clune J, Cully A, Mouret JB (2016). [How Do Different Encodings Influence the Performance of the MAP-Elites Algorithm?](https://hal.inria.fr/hal-01302658/document). *In Proc. of Genetic and Evolutionary Computation Conference*.
- Chatzilygeroudis, K., Vassiliades, V. and Mouret, J.B. (2016). [Reset-free Trial-and-Error Learning for Data-Efficient Robot Damage Recovery](https://arxiv.org/abs/1610.04213). *arXiv preprint arXiv:1610.04213*.
- Chatzilygeroudis, K., Cully, A. and Mouret, J.B. (2016). [Towards semi-episodic learning for robot damage recovery](https://arxiv.org/abs/1610.01407). *Workshop on AI for Long-Term Autonomy at the IEEE International Conference on Robotics and Automation 2016*.

Research projects that use Limbo
--------------------------------
- Resibots. ERC Starting Grant: http://www.resibots.eu/
- PAL. H2020 EU project: http://www.pal4u.eu/
