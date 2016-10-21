.. limbo documentation master file, created by
   sphinx-quickstart on Tue Nov 17 14:21:26 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _limbo_doc:

Limbo's documentation
=================================

Limbo is a lightweight framework for Bayesian Optimization, a powerful approach for global optimization of expensive, non-convex functions. Github page (to report issues and/or help us to improve the library): `Sferes2 <http://github.com/resibots/limbo>`_

The development of Limbo is funded by the `ERC project ResiBots <http://www.resibots.eu>`_.

Limbo shares many ideas with `Sferes2 <http://github.com/sferes2>`_, a similar framework for evolutionary computation.

Main features
--------------

- Bayesian optimization based on Gaussian processes
- Generic framework (template-based / policy-based design), which allows for easy customization, to test novel ideas
- Programming / experimental framework that allows users to easily test variants of experiments, compare treatments, submit jobs to clusters, etc.
- High performance (in particular, Limbo can exploit multicore computers via Intel TBB and vectorize some operations via Eigen3)
- Purposely small to be easily maintained and quickly understood
- Modern C++-11
- Experimental support for multi-objective optimization and sparse Gaussian processes


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
   defaults
   faq


.. Indices and tables
.. ==================
..
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
