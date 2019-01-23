.. limbo documentation master file, created by
   sphinx-quickstart on Tue Nov 17 14:21:26 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _limbo_doc:

Limbo's documentation
=================================

Limbo (LIbrary for Model-Based Optimization) is an open-source C++11 library for Gaussian Processes and data-efficient optimization (e.g., Bayesian optimization, see :cite:`b-brochu2010tutorial,b-Mockus2013`) that is designed to be both highly flexible and very fast. It can be used as a state-of-the-art optimization library or to experiment with novel algorithms with "plugin" components. Limbo is currently mostly used for data-efficient policy search in robot learning :cite:`b-lizotte2007automatic` and online adaptation because computation time matters when using the low-power embedded computers of robots. For example, Limbo was the key library to develop a new algorithm that allows a legged robot to learn a new gait after a mechanical damage in about 10-15 trials (2 minutes) :cite:`b-cully_robots_2015`, and a 4-DOF manipulator to learn neural networks policies for goal reaching in about 5 trials :cite:`b-chatzilygeroudis2017`.

The implementation of Limbo follows a policy-based design :cite:`b-alexandrescu2001modern` that leverages C++ templates: this allows it to be highly flexible without the cost induced by classic object-oriented designs (cost of virtual functions). `The regression benchmarks <http://www.resibots.eu/limbo/reg_benchmarks.html>`_ show that the query time of Limbo's Gaussian processes is several orders of magnitude better than the one of GPy (a state-of-the-art `Python library for Gaussian processes <https://sheffieldml.github.io/GPy/>`_) for a similar accuracy (the learning time highly depends on the optimization algorithm chosen to optimize the hyper-parameters). The `black-box optimization benchmarks <http://www.resibots.eu/limbo/bo_benchmarks.html>`_ demonstrate that Limbo is about 2 times faster than BayesOpt (a C++ library for data-efficient optimization, :cite:`b-martinezcantin14a`) for a similar accuracy and data-efficiency. In practice, changing one of the components of the algorithms in Limbo (e.g., changing the acquisition function) usually requires changing only a template definition in the source code. This design allows users to rapidly experiment and test new ideas while keeping the software as fast as specialized code.

Limbo takes advantage of multi-core architectures to parallelize the internal optimization processes (optimization of the acquisition function, optimization of the hyper-parameters of a Gaussian process) and it vectorizes many of the linear algebra operations (via the `Eigen 3 library <http://eigen.tuxfamily.org/>`_ and optional bindings to Intel's MKL).

The library is distributed under the `CeCILL-C license <http://www.cecill.info/index.en.html>`_ via a `Github repository <http://github.com/resibots/limbo>`_. The code is standard-compliant but it is currently mostly developed for GNU/Linux and Mac OS X with both the GCC and Clang compilers. New contributors can rely on a full API reference, while their developments are checked via a continuous integration platform (automatic unit-testing routines).

Limbo is currently used in the `ERC project ResiBots <http://www.resibots.eu>`_, which is focused on data-efficient trial-and-error learning for robot damage recovery, and in the `H2020 projet PAL <http://www.pal4u.eu/>`_, which uses social robots to help coping with diabetes. It has been instrumental in many scientific publications since 2015 :cite:`b-cully_robots_2015,b-chatzilygeroudis2018resetfree,b-tarapore2016,b-chatzilygeroudis2017,b-pautrat2018bayesian,b-chatzilygeroudis2018using`.

Limbo shares many ideas with `Sferes2 <http://github.com/sferes2>`_, a similar framework for evolutionary computation.


Contents:
-----------

.. toctree::
   :hidden:
   :caption: Limbo (BO library)

.. toctree::
   :maxdepth: 2

   self
   tutorials/index
   guides/index
   api
   bo
   defaults
   reg_benchmarks
   bo_benchmarks
   faq


.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

-----

.. bibliography:: guides/refs.bib
  :style: plain
  :cited:
  :keyprefix: b-
