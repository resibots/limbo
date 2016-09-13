.. _gp-tutorial:

Gaussian Process
====================

Limbo relies on our C++-11 implementation of Gaussian processes (See :ref:`gaussian-process` for a short introduction ) which can be useful by itself. This tutorial explains how to create and use a Gaussian Process (GP).

Data
----
We assume that our samples are in a vector called ``samples`` and that our observations are in a vector called ``observations``. Please note that the type of both observations and samples is Eigen::VectorXd (in this example, they are 1-D vectors).


.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 30-40

Basic usage
------------

We first create a basic GP with an Exponential kernel (``kernel::Exp<Params>``) and a mean function equals to the mean of the obsevations (``mean::Data<Params>``). The ``Exp`` kernel needs a few parameters to be defined in a ``Params`` structure:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 14-17

The type of the GP is defined by the following lines:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 43-45

To use the GP, we need :

- to instantiante a ``GP_t`` object
- to call the method ``compute()``.

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 47-54

Here we assume that the noise is the same for all samples and that it is equal to 0.01.

Querying the GP can be achieved in two different ways:

- ``gp.mu(v)`` and ``gp.sigma(v)``, which return the mean and the variance (sigma squared) for the input data point ``v``
- ``std::tie(mu, sigma) = gp.query(v)``, which returns the mean and the variance at the same time.

The second approach is faster because some computations are the same for ``mu`` and ``sigma``.


To visualize the predictions of the GP, we can query it for many points and record the predictions in a file:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 57-67


Hyper-parameter optimization
----------------------------
Most kernel functions have some parameters. It is common in the GP literature to set them by maximizing the log-likelihood of the data knowing the model (see :ref:`gaussian-process` for a description of this concept).

In limbo, only a subset of the kernel functions can have their hyper-parameters optimized. The most common one is ``SquaredExpARD`` (Squared Exponential with Automatic Relevance Determination).

A new GP type is defined as follows:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 71-73

It uses the default values for the parameters of ``SquaredExpARD``:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 19-20

After calling the ``compute()`` method, the hyper-parameters can be optimized by calling the 'optimize_hyperparams()' function. Once the new parameters are found, the GP needs to be recomputed:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 77-79


We can have a look at the difference between the two GPs:

.. figure:: ../pics/gp.png
   :alt: Comparisons of GP
   :target: ../_images/gp.png


This plot is generated using matplotlib:

.. literalinclude:: ../../src/tutorials/plot_gp.py
   :language: python
   :linenos:
