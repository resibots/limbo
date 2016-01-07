.. limbo documentation master file, created by
   sphinx-quickstart on Tue Nov 17 14:21:26 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _limbo_doc:

Limbo's documentation
=================================

Limbo is a lightweight framework for Bayesian Optimization, a powerful approach for global optimization of expensive, non-convex functions.

Limbo is primarily designed for *researchers* who need to experiment with novel ideas / algorithms. It is not designed for end-users who need a "black-box" to optimize a function (although Limbo can be used for this).

Limbo has been used in several scientific publications, in particular:

- Cully A, Clune J, Tarapore DT, Mouret J-B. Robots that can adapt like animals. Nature, 2015. 521.7553.

The development of Limbo is funded by the `ERC project ResiBots <http://www.resibots.eu>`_.

Limbo shares many ideas with `Sferes2 <http://github.com/sferes2>`_, a similar framework for evolutionary computation.

Main features
--------------

- Bayesian optimisation based on Gaussian processes
- Generic framework (template-based / policy-based design), which allows easy customization for testing novel ideas
- Programming / experimental framework that allows user to easily test variants of experiments, compare treatments, submit jobs to clusters, etc.
- High performance (in particular, Limbo can exploit multicore computers via Intel TBB and vectorize some operations via Eigen3)
- Purposely small to be easily maintained and quickly understood
- Modern C++-11
- Experimental support for multi-objective optimization


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
   faq


.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`


TODO list
==========

.. todolist::
