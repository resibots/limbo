limbo [![Build Status](https://travis-ci.org/resibots/limbo.svg?branch=master)](https://travis-ci.org/resibots/limbo) [![DOI](http://joss.theoj.org/papers/10.21105/joss.00545/status.svg)](https://doi.org/10.21105/joss.00545)
============

Limbo (LIbrary for Model-Based Optimization) is an open-source C++11 library for Gaussian Processes and data-efficient optimization (e.g., Bayesian optimization) that is designed to be both highly flexible and very fast. It can be used as a state-of-the-art optimization library or to experiment with novel algorithms with "plugin" components.

![logo](./docs/logo/logo_limbo.png)

Documentation & Versions
------------------------
The development branch is the [master](https://github.com/resibots/limbo/tree/master) branch. For the latest stable release, check the [release-2.1](https://github.com/resibots/limbo/tree/release-2.1) branch.
Documentation is available at: http://www.resibots.eu/limbo

Citing Limbo
------------
If you use Limbo in a scientific paper, please cite:


Cully, A., Chatzilygeroudis, K., Allocati, F., and Mouret J.-B., (2018). [Limbo: A Flexible High-performance Library for Gaussian Processes modeling and Data-Efficient Optimization](http://joss.theoj.org/papers/10.21105/joss.00545). *The Journal of Open Source Software*.

In BibTex:

    @article{cully2018limbo,
        title={{Limbo: A Flexible High-performance Library for Gaussian Processes modeling and Data-Efficient Optimization}},
        author={Cully, A. and Chatzilygeroudis, K. and Allocati, F.  and Mouret, J.-B.},
        year={2018},
        journal={{The Journal of Open Source Software}},
        publisher={The Open Journal},
        volume={3},
        number={26},
        pages={545},
        doi={10.21105/joss.00545}
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
- High performance (in particular, Limbo can exploit multi-core computers via Intel TBB and vectorize some operations via [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page))
- Purposely small to be easily maintained and quickly understood

Scientific articles that use Limbo
----------------------------------
- Chatzilygeroudis, K., & Mouret, J. B. (2018). [Using Parameterized Black-Box Priors to Scale Up Model-Based Policy Search for Robotics](https://arxiv.org/pdf/1709.06917). *Proceedings of the International Conference on Robotics and Automation (ICRA)*.
- Pautrat, R., Chatzilygeroudis, K., & Mouret, J.-B. (2018). [Bayesian Optimization with Automatic Prior Selection for Data-Efficient Direct Policy Search](https://arxiv.org/pdf/1709.06919). *Proceedings of the International Conference on Robotics and Automation (ICRA)*.
- Chatzilygeroudis, K., Vassiliades, V. and Mouret, J.-B. (2017). [Reset-free Trial-and-Error Learning for Robot Damage Recovery](https://arxiv.org/abs/1610.04213). *Robotics and Autonomous Systems*.
- Karban P., Pánek D., Mach F. and Doležel, I. (2017). [Calibration of numerical models based on advanced optimization and penalization techniques](https://www.degruyter.com/downloadpdf/j/jee.2017.68.issue-5/jee-2017-0073/jee-2017-0073.pdf). *Journal of Electrical Engineering, 68(5), 396-400*.
- Chatzilygeroudis K., Rama R., Kaushik, R., Goepp, D., Vassiliades, V. and Mouret, J.-B. (2017). [Black-Box Data-efficient Policy Search for Robotics](https://arxiv.org/abs/1703.07261). *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.
- Tarapore, D., Clune, J., Cully, A., and Mouret, J.-B. (2016). [How Do Different Encodings Influence the Performance of the MAP-Elites Algorithm?](https://hal.inria.fr/hal-01302658/document). *In Proc. of Genetic and Evolutionary Computation Conference*.
- Cully, A., Clune, J., Tarapore, D., and Mouret, J.-B. (2015). [Robots that can adapt like animals](http://www.nature.com/nature/journal/v521/n7553/full/nature14422.html). *Nature*, 521(7553), 503-507.
- Chatzilygeroudis, K., Cully, A. and Mouret, J.-B. (2016). [Towards semi-episodic learning for robot damage recovery](https://arxiv.org/abs/1610.01407). *Workshop on AI for Long-Term Autonomy at the IEEE International Conference on Robotics and Automation 2016*.
- Papaspyros, V., Chatzilygeroudis, K., Vassiliades, V., and Mouret, J.-B. (2016). [Safety-Aware Robot Damage Recovery Using Constrained Bayesian Optimization and Simulated Priors](https://arxiv.org/pdf/1611.09419v3). *Workshop on Bayesian Optimization at the Annual Conference on Neural Information Processing Systems (NIPS) 2016.*

Research projects that use Limbo
--------------------------------
- Resibots. ERC Starting Grant: http://www.resibots.eu/
- PAL. H2020 EU project: http://www.pal4u.eu/
