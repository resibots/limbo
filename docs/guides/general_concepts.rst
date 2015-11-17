General Concepts
======================

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

We can change which ``Kernel Function`` our ``GP`` uses, using the second template parameter of the GP class. Every kernel function takes as template parameters only the ``Params``.

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


