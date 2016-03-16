Implementation details & customization of algorithms
====================================================

.. highlight:: c++

Limbo follows a  `policy-based design <https://en.wikipedia.org/wiki/Policy-based_design>`_, which allows users to combine high flexibility (almost every part of Limbo can be substituted by a user-defined part) with high performance (the abstraction do not add any overhead, contrary to classic OOP design). These two features are critical for researchers who want to experiment new ideas in Bayesian optimization. This means that changing a part of limbo (e.g. changing the kernel functions) corresponds to changing a template parameter of the optimizer.

For the parameters of the algorithms themselves (e.g. an epsilon), all the classes in Limbo they are given by a template class (usually called Params in our code, and always the first argument). See :doc:`parameters` for details.

To avoid defining each component of an optimizer manually, Limbo provides sensible defaults. In addition, Limbo relies on `Boost.Parameter <http://www.boost.org/doc/libs/1_60_0/libs/parameter/doc/html/index.html>`  to make it easy to customize a single part. This Boost library allows us to write classes that accept template argument (user-defined custom classes) by name. For instance, to customize the stopping criteria:


::

  using namespace limbo;

  // here stop_t is a user-defined list of stopping criteria
  bayes_opt::BOptimizer<Params, stopcrit<stop_t>> boptimizer;

Or to define a custom acquisition function:

::

  using namespace limbo;

  // here acqui_t is a user-defined acquisition function
  bayes_opt::BOptimizer<Params, acquifun<acqui_t>> boptimizer;

Class Structure
---------------

.. figure:: ../pics/limbo_uml.png
   :alt: UML class diagram
   :target: ../_images/limbo_uml.png

   Click on the image to see it bigger.


Sequence graph
---------------
.. figure:: ../pics/limbo_call_graph.png
   :alt: Sequence diagram
   :target: ../_images/limbo_call_graph.png

   Click on the image to see it bigger.



File Structure
--------------
(see below for a short explanation of the concepts)

::

  src
  +-- limbo:
       +-- acqui: acquisition functions
       |-- bayes_opt: bayesian optimizers
       |-- init: initialization functions
       |-- kernel: kernel functions
       |-- mean: mean functions
       |-- model: models (Gaussian Processes)
       |-- opt: optimizers (Rprop, CMA-ES, etc.)
       |-- stat: statistics (to dump data)
       |-- stop: stopping criteria
       |-- tools: useful macros & small functions
  |-- tests: unit tests
  |-- benchmarks: a few benchmark functions
  |-- examples: a few examples
  |-- cmaes: [external] the CMA-ES library, used for inner optimizations -- from https://www.lri.fr/~hansen/cmaesintro.html
  |-- ehvi: [external] the Expected HyperVolume Improvement, used for Multi-Objective Optimization -- by Iris Hupkens


Each directory in the `limbo` directory corresponds to a namespace with the same name. There is also a file for each directory called "directory.hpp" (e.g. `acqui.hpp`) that includes the whole namespace.


.. _acquisition-guide:

Acquisition Functions (acqui::)
--------------------------------


We can change which ``Acquisition Function`` our ``BOptimizer`` uses, using the ``acquifun`` templated parameter. Every acquisition function takes as template parameters the ``Params`` and a ``Model``.


::

    typedef AcquiName<Params, Model> acqui_t;

    BOptimizer<Params, acquifun<acqui_t>> boptimizer;

The acquisition functions provided by **limbo** are the following (see :ref:`here <acqui-functions>` for more details):

- UCB
    - ``Params::ucb::alpha`` should be available and a float.
    - Model needs to have the following functions implemented:
        - ``int dim_in() const`` - returns the dimension of input data
        - ``int dim_out() const`` - return the dimension of the output data
        - ``std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const`` - returns mean and sigma at the point v
- GP_UCB
    - ``Params::gp_ucb::delta`` should be available and a float.
    - Model needs to have the following functions implemented:
        - ``int dim_in() const`` - returns the dimension of input data
        - ``int dim_out() const`` - return the dimension of the output data
        - ``std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const`` - returns mean and sigma at the point v

Models (model::)
-----------------

We can change which ``Model`` our ``BOptimizer`` uses, using the ``modelfun`` templated parameter. Each model should take as the first template parameter the ``Params`` and could optionally have more.

::

    typedef ModelName<Params, ...> model_t;

    BOptimizer<Params, modelfun<model_t>> boptimizer;

Each model should have implemented the following functions:

- Should have constructor of the form:
    - ``ModelName(int dim_in, int dim_out)``
- ``void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, double noise, const std::vector<Eigen::VectorXd>& bl_samples)``


**limbo** provides only a **Gaussian Process** model for now. See :ref:`gaussian process section of the BO guide <gaussian-process>` for more details.

.. _kernel-guide:

Kernel Functions in GP model (kernel::)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can change which ``Kernel Function`` our ``GP`` uses, using the second template parameter of the GP class. Every kernel function takes as template parameters the ``Params`` and optionally some more.

::

    typedef KernelName<Params> kernel_t;
    typedef GP<Params, kernel_t, ...> gp_t;

    BOptimizer<Params, modelfun<gp_t>> boptimizer;

The kernel functions provided by **limbo** are the following (see :ref:`kernel function section of the BO guide <kernel-functions>` for more details):

- Exp
    - ``Params::kf_exp::sigma`` should be available and a float.
- MaternFiveHalfs
    - ``Params::kf_maternfivehalfs::sigma`` should be available and a float.
    - ``Params::kf_maternfivehalfs::l`` should be available and a float.
- MaternThreeHalfs
    - ``Params::kf_maternthreehalfs::sigma`` should be available and a float.
    - ``Params::kf_maternthreehalfs::l`` should be available and a float.
- SquaredExpARD
    - No params needed
    - Used for kernel's hyperparameters optimization


.. _mean-guide:

Mean Functions in GP model (mean::)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can change which ``Mean Function`` our ``GP`` uses, using the third template parameter of the GP class. Every mean function takes as template parameters the ``Params`` and optionally some more.

::

    typedef MeanName<Params> mean_t;
    typedef GP<Params, ..., mean_t, ...> gp_t;

    BOptimizer<Params, modelfun<gp_t>> boptimizer;

The mean functions provided by **limbo** are the following (see :ref:`the mean function section of the BO guide <mean-functions>` for more details):

- NullFunction
    - No params needed
    - Zero mean
- Constant
    - ``Params::meanconstant::constant`` should be available and a ``Eigen::VectorXd`` with size same as ``GP::dim_out``.
    - Constant mean
- Data
    - GP needs to have the following functions implemented:
        - ``Eigen::VectorXd mean_observation()`` - returns the mean observation
    - Mean of actual data
- FunctionARD
    - No params needed
    - Used for mean's hyperparameters optimization
    - It takes as a template parameter the mean function to use

Optimizers (opt::)
------------------

Statistics (stat::)
-------------------

We can change which ``Statistics`` our ``BOptimizer`` outputs, using the ``statfun`` templated parameter. Every statistic takes as template parameters the ``Params`` and optionally some more. All statistics should inherit from ``StatBase`` class.

::

    typedef StatName<Params> stat_t;

    BOptimizer<Params, statfun<stat_t>> boptimizer;

**limbo** provides only **Acquisitions** statistics for now.
