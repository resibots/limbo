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
   :lines: 77-86

Basic usage
------------

We first create a basic GP with an Exponential kernel (``kernel::Exp<Params>``) and a mean function equals to the mean of the observations (``mean::Data<Params>``). The ``Exp`` kernel needs a few parameters to be defined in a ``Params`` structure:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 61-72

The type of the GP is defined by the following lines:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 87-91

To use the GP, we need :

- to instantiante a ``GP_t`` object
- to call the method ``compute()``.

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 92-97

Here we assume that the noise is the same for all samples and that it is equal to 0.01.

Querying the GP can be achieved in two different ways:

- ``gp.mu(v)`` and ``gp.sigma(v)``, which return the mean and the variance (sigma squared) for the input data point ``v``
- ``std::tie(mu, sigma) = gp.query(v)``, which returns the mean and the variance at the same time.

The second approach is faster because some computations are the same for ``mu`` and ``sigma``.


To visualize the predictions of the GP, we can query it for many points and record the predictions in a file:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 99-110


Hyper-parameter optimization
----------------------------
Most kernel functions have some parameters. It is common in the GP literature to set them by maximizing the log-likelihood of the data knowing the model (see :ref:`gaussian-process` for a description of this concept).

In limbo, only a subset of the kernel functions can have their hyper-parameters optimized. The most common one is ``SquaredExpARD`` (Squared Exponential with Automatic Relevance Determination).

A new GP type is defined as follows:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 112-116

It uses the default values for the parameters of ``SquaredExpARD``:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 66-69

After calling the ``compute()`` method, the hyper-parameters can be optimized by calling the ``optimize_hyperparams()`` function. The GP does not need to be recomputed and we pass ``false`` for the last parameter in ``compute()`` as we do not need to compute the kernel matrix again (it will be recomputed in the hyper-parameters optimization).

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 119-121


We can have a look at the difference between the two GPs:

.. figure:: ../pics/gp.png
   :alt: Comparisons of GP
   :target: ../_images/gp.png


This plot is generated using matplotlib:

.. literalinclude:: ../../src/tutorials/plot_gp.py
   :language: python
   :linenos:

Here is the complete ``main.cpp`` file of this tutorial:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :lines: 46-

Saving and Loading
-------------------

We can also save our optimized GP model:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 138-139

This will create a directory called ``myGP`` with several files (the GP data, kernel hyperparameters etc.). If we want a binary format (i.e., more compact), we can replace the ``TextArchive`` by ``BinaryArchive``.

To the load a saved model, we can do the following:

.. literalinclude:: ../../src/tutorials/gp.cpp
   :language: c++
   :linenos:
   :lines: 141-142

Note that we need to have the same kernel and mean function (i.e., the same GP type) as the one used for saving.