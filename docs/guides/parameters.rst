.. _params-guide:

.. highlight:: c++


Parameters
===========

Bayesian Optimization algorithms, acquisition functions, etc. all have many parameters. The traditionnal approach is to use a configuration file (e.g. XML, or json, .ini, ...). However,  each time a developer adds a parameter, some code has to be added to parse the configuration file: there is often more code to parse and check the configuration file than *real code* (that is, code that actually does something). As a result, scientists often either skip this part until they have  "final" version of their code (often, never), or do it in a "quick and dirty way" (e.g. without checking the syntax, without checking that the parameter value is in the right range, etc.).

Put differently, using a configuration file is nice for the user, but not for the developer. Since **limbo** is targeted to scientists who want to *easily* test  new code, we need a way to separate parameters from code that do not require any boilerplate code.

In **limbo**, every class takes a structure (usually called ``Params``) that contains the parameters. By doing so, we rely on the compiler to check the types, and we require very little work to separate parameters values from algorithms.

From the user's point of view, this looks like this:

::

    struct Params {
      struct acqui_ucb {
        BO_PARAM(double, alpha, 0.1);
      };
    };
    // ...
    // ... instantiate an optimizer:
    bayes_opt::BOptimizer<Params> opt;


(do not forget the semi-colons!). This structure says that the value of the parameter ``alpha`` for the class "UCB" is ``0.1``.

In the UCB class, the value can be accessed like this:

::

    float x = Params::acqui_ucb::alpha();

No need to write any parsing code!

Default parameters
------------------

Many limbo classes provide default parameters. To use them, the parameter sub-structure has to inherit from the default structure:

::

    struct Params {
      struct acqui_ucb : public defaults::acqui_ucb {
      };
    };

That way, the ``acqui_ucb::alpha()`` exists, but with the default value.

Dynamic parameters
------------------

Sometimes, we need to define parameters that can be changed at runtime. In that case, we can use a ``BO_DYN_PARAM`` instead of a ``BO_PARAM``:

::

    struct Params {
      struct acqui_ucb {
        BO_DYN_PARAM(double, alpha);
      };
    };


However, for dynamic parameters, we need to call ``BO_DECLARE_DYN_PARAM`` in our ``.cpp`` file (typically, just before the main function):

::

    BO_DECLARE_DYN_PARAM(int, Params::acqui_ucb, alpha);

.. warning:: Dynamic parameters are not thread-safe! (standard parameters are thread safe and add no overhead -- they are equivalent to writing a constant).

Arrays
------

Last, we can also use arrays, Eigen vectors, and strings as follows::

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

Multiple sets of parameters
---------------------------

In some rare cases, you may have 2 different instances of the same algorithm, for example, if you choose to use the same optimizer for the acquisition optimization and the GP hyperparametrs optimization.  In such cases, you may also wish to set different parameters for each instance. This can be easily accomplished in the following way:

::

    struct Params {
      // All the other parameters
    }

    struct ParamsAcquiOpt {
        struct opt_cmaes {
            BO_PARAM(int, restarts, 10);
            BO_PARAM(int, max_fun_evals, 500);
        };
    };

    struct ParamsGPOpt {
        struct opt_cmaes {
            BO_PARAM(int, restarts, 1);
            BO_PARAM(int, max_fun_evals, 200);
        };
    };

Then, when declaring the types to use:

::

    using Acqui_opt_t = opt::Cmaes<ParamsAcquiOpt>;
    using Gp_opt_t = opt::Cmaes<ParamsGPOpt>;

    using Kernel_t = kernel::MaternFiveHalfs<Params>;
    using Mean_t = mean::Data<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t, model::gp::KernelLFOpt<Params, Gp_opt_t>>;
    using Acqui_t = acqui::UCB<Params, GP_t>;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>, acquiopt<Acqui_opt_t>> opt;
