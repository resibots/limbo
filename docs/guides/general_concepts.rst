General Concepts
======================

Class Structure
---------------

.. figure:: ../pics/limbo_uml.png
   :alt: UML class diagram
   :target: ../_images/limbo_uml.png

   Click on the image to see it bigger.

File Structure
--------------
(see below for a short explanation of the concepts)

::

  src
  +-- benchmarks: a few benchmark functions
  |-- cmaes: the CMA-ES library, used for inner optimizations -- from https://www.lri.fr/~hansen/cmaesintro.html
  |-- ehvi: the Expected HyperVolume Improvement, used for Multi-Objective Optimization
  |-- examples: a few examples
  |-- limbo
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


Each directory in the `limbo` directory corresponds to a namespace with the same name. There is also a file for each directory called "directory.hpp" (e.g. `acqui.hpp`) that includes the whole namespace.


Acquisition Functions
--------------------------------

We can change which ``Acquisition Function`` our ``BOptimizer`` uses, using the ``acquifun`` templated parameter. Every acquisition function takes as template parameters the ``Params`` and a ``Model``.

.. highlight:: c++

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

Models
-----------------

We can change which ``Model`` our ``BOptimizer`` uses, using the ``modelfun`` templated parameter. Each model should take as the first template parameter the ``Params`` and could optionally have more.

::

    typedef ModelName<Params, ...> model_t;

    BOptimizer<Params, modelfun<model_t>> boptimizer;

Each model should have implemented the following functions:

- Should have constructor of the form:
    - ``ModelName(int dim_in, int dim_out)``
- ``void compute(const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, double noise, const std::vector<Eigen::VectorXd>& bl_samples)``


**limbo** provides only a **Gaussian Process** model for now. See :ref:`here <gaussian-process>` for more details.


Kernel Functions in GP model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can change which ``Kernel Function`` our ``GP`` uses, using the second template parameter of the GP class. Every kernel function takes as template parameters the ``Params`` and optionally some more.

::

    typedef KernelName<Params> kernel_t;
    typedef GP<Params, kernel_t, ...> gp_t;

    BOptimizer<Params, modelfun<gp_t>> boptimizer;

The kernel functions provided by **limbo** are the following (see :ref:`here <kernel-functions>` for more details):

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


Mean Functions in GP model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can change which ``Mean Function`` our ``GP`` uses, using the third template parameter of the GP class. Every mean function takes as template parameters the ``Params`` and optionally some more.

::

    typedef MeanName<Params> mean_t;
    typedef GP<Params, ..., mean_t, ...> gp_t;

    BOptimizer<Params, modelfun<gp_t>> boptimizer;

The mean functions provided by **limbo** are the following (see :ref:`here <mean-functions>` for more details):

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

Statistics
-----------------

We can change which ``Statistics`` our ``BOptimizer`` outputs, using the ``statfun`` templated parameter. Every statistic takes as template parameters the ``Params`` and optionally some more. All statistics should inherit from ``StatBase`` class.

::

    typedef StatName<Params> stat_t;

    BOptimizer<Params, statfun<stat_t>> boptimizer;

**limbo** provides only **Acquisitions** statistics for now.

Parameters
-----------

Bayesian Optimization algorithms, acquisition functions, etc. all have many parameters. The traditionnal approach is to use a configuration file (e.g. XML, or json, .ini, ...). However,  each time a developer adds a parameter, some code has to be added to parse the configuration file: there is often more code to parse and check the configuration file than *real code* (that is, code that actually does something). As a result, scientists often either skip this part until they have  "final" version of their code (often, never), or do it in a "quick and dirty way" (e.g. without checking the syntax, without checking that the parameter value is in the right range, etc.).

Put differently, using a configuration file is nice for the user, but not for the developer. Since **limbo** is targeted to scientists who want to *easily* test  new code, we need a way to separate parameters from code that do not require any boilerplate code.

In **limbo**, every class takes a structure name (usually called ``Params``) that contains the parameters. By doing so, we rely on the compiler to check the types, and we require very little work to separate parameters values from algorithms.

From the user's point of view, this looks like this:

::

    struct Params {
      struct ucb {
        BO_PARAM(float, alpha, 0.1);
      };
    };
    // ...
    // ... instantiate an optimizer:
    bayes_opt::BOptimizer<Params> opt;


(do not forget the semi-colons!). This structure says that the value of the parameter ``alpha`` for the class "UCB" is ``0.1``.

In the UCB class, the value can be accessed like this:

::

    float x = Params::ucb::alpha();

No need to write any parsing code!

Many limbo classes provide default parameters. To use them, the parameter sub-structure has to inherit from the default structure:

::

    struct Params {
      struct ucb : public defaults::ucb {
      };
    };

That way, the ``ucb::alpha()`` exists, but it has its default value.


Sometimes, we need to define parameters that can be changed at runtime. In that case, we can use a ``BO_DYN_PARAM`` instead of a ``BO_PARAM``:

::

    struct Params {
      struct ucb {
        BO_DYN_PARAM(float, alpha, 0.1);
      };
    };


However, for dynamic parameters, we need to call ``BO_DECLARE_DYN_PARAM`` in our ``.cpp`` file (typically, just before the main function):

::

    BO_DECLARE_DYN_PARAM(int, Params::ucb, alpha);

**Warning!** Dynamic parameters are not thread-safe! (standard parameters are thread safe and add no overhead -- they are equivalent to writing a constant).

Last, we can also use arrays, vectors, and strings as follows:

::


    struct Params {
        struct test {
            BO_PARAM(double, a, 1);
            BO_DYN_PARAM(int, b);
            BO_PARAM_ARRAY(double, c, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            BO_PARAM_VECTOR(double, d, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            BO_PARAM_STRING(e, "e");
        };
    };
    BO_DECLARE_DYN_PARAM(int, Params::test, b);

All these macros are defined in ``tools/macros.hpp``.
