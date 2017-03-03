.. _opt-tutorial:

Optimization Sub-API
====================



Limbo uses optimizers in several situations, most notably to optimize hyper-parameters of Gaussian processes and to optimize acquisition functions. Nevertheless, these optimizers might be useful in other contexts. This tutorial briefly explains how to use it.

Optimizers in Limbo are wrappers around:

- NLOpt (which provides many local, global, gradient-based, gradient-free algorithms)
- libcmaes (which provides the Covariance Matrix Adaptation Evolutionary Strategy, that is, CMA-ES)
- a few other algorithms that are implemented in Limbo (in particular, RPROP, which is gradient-based optimization algorithm)


We first need to define a function to be optimized. Here we chose :math:`-(x_1-0.5)^2 - (x_2-0.5)^2`, whose maximum is [0.5, 0.5]:

.. literalinclude:: ../../src/tutorials/opt.cpp
   :language: c++
   :linenos:
   :lines: 74-82


.. warning::

 Limbo optimizers always MAXIMIZE f(x), whereas many libraries MINIMIZE f(x)

The first thing to note is that the functions needs to return an object of type ``eval_t``, which is actually a pair made of a double (:math:`f(x)`) and a vector (the gradient). We need to do so because (1) many fast algorithm use the gradient, (2) the gradient and the function often share some computations, therefore it is often faster to compute both the function value and the gradient at the same time (this is, for instance, the case with the log-likelihood that we optimize to find the hyper-parameters of Gaussian processes).

Thanks to c++11, we can simply return ``{v, grad}`` and an object of type ``eval_t`` will be created. When we do not know how to compute the gradient, we return ``opt::no_grad(v)``, which creates a special object without the gradient information (using boost::optional).

The boolean ``eval_grad`` is true when we need to evaluate the gradient for x, and false otherwise. This is useful because some algorithms do not need the gradient: there is no need to compute this value.

As usual, each algorithm has some parameters (typically the number of iterations to perform). They are defined like the other parameters in Limbo (see :ref:`Parameters <parameters>`):


.. literalinclude:: ../../src/tutorials/opt.cpp
   :language: c++
   :linenos:
   :lines: 51-71

Now we can instantiate our optimizer and call it:

.. literalinclude:: ../../src/tutorials/opt.cpp
   :language: c++
   :linenos:
   :lines: 87-91


We can do the same with a gradient-free optimizer from NLOpt:

.. literalinclude:: ../../src/tutorials/opt.cpp
   :language: c++
   :linenos:
   :lines: 94-99


Or with CMA-ES:

.. literalinclude:: ../../src/tutorials/opt.cpp
   :language: c++
   :linenos:
   :lines: 105-108


See the :ref:`API documentation <opt-api>` for more details.

Here is the full file.

.. literalinclude:: ../../src/tutorials/opt.cpp
   :language: c++
   :linenos:
   :lines: 46-
