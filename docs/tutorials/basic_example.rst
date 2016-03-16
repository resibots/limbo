Basic Example
=================================================

Basic Example
----------------------------

Create directories and files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's say we want to create an experiment called "test". The first thing to do is to create the folder ``exp/test`` under the limbo root. Then add two files:

* the ``main.cpp`` file
* a pyhton file called ``wscript``, which will be used by ``waf`` to register the executable for building

The file structure should look like this: ::

  limbo
  |-- exp
       |-- test
            +-- wscript
            +-- main.cpp

Next, copy the following content to the ``wscript`` file: 

.. code:: python

    def options(opt):
        pass


    def build(bld):
        bld(features='cxx cxxprogram',
            source='main.cpp',
            includes='. ../../src',
            target='test',
            uselib='BOOST EIGEN TBB',
            use='limbo') 

For this example, we will optimize a simple function: :math:`-{(5*x - 2.5)}^2 + 5`, using all default values and settings.

.. highlight:: c++

To begin, the ``main`` file has to include the necessary files, and declare the ``Parameter struct``: ::

    #include <iostream>
    #include <limbo/bayes_opt/boptimizer.hpp>

    using namespace limbo;

    struct Params {
        struct bayes_opt_boptimizer {
            BO_PARAM(double, noise, 0.0);
        };

        struct bayes_opt_bobase {
            BO_PARAM(int, stats_enabled, false);
        };

        struct init_randomsampling {
            BO_PARAM(int, samples, 10);
        };

        struct stop_maxiterations {
            BO_PARAM(int, iterations, 40);
        };

        struct acqui_gpucb : public defaults::acqui_gpucb {
        };

        struct opt_gridsearch : public defaults::opt_gridsearch {
        };

        struct opt_rprop : public defaults::opt_rprop {
        };

        struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
        };
    };

Here we are stating that the samples are observed without noise (which makes sense, because we are going to evaluate the function),
that we don't want to output any stats (by setting the dump period to -1), that the model has to be initialized with 10 samples (that will be
selected randomly), and that the optimizer should run for 40 iterations. The rest of the values are taken from the defaults.

Then, we have to define the evaluation function for the optimizer to call: ::

    struct Eval {
        static constexpr size_t dim_in = 1;
        static constexpr size_t dim_out = 1;

        Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
        {
            Eigen::VectorXd res(1);
            res(0) = -((5 * x(0) - 2.5) * (5 * x(0) - 2.5)) + 5;
            return res;
        }
    };

It is required that the evaluation struct has the static members ``dim_in`` and ``dim_out``, specifying the input and output dimension.
Also, it should have the ``operator()`` expecting a ``const Eigen::VectorXd&`` of size ``dim_in``, and return another one, of size ``dim_out``.

With this, we can declare the main function: ::

    int main() {
        bayes_opt::BOptimizer<Params> boptimizer;
        boptimizer.optimize(Eval());
        std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
        return 0;
    }

Finally, from the root of limbo, run a build command, with the additional switch ``--exp test``: ::

    ./waf build --exp test

Then, an executable named ``test`` should be produced under the folder ``build/exp/test``.

Full ``main.cpp``::

    #include <iostream>
    #include <limbo/bayes_opt/boptimizer.hpp>

    using namespace limbo;

    struct Params {
        struct bayes_opt_boptimizer {
            BO_PARAM(double, noise, 0.0);
        };

        struct bayes_opt_bobase {
            BO_PARAM(int, stats_enabled, false);
        };

        struct init_randomsampling {
            BO_PARAM(int, samples, 10);
        };

        struct stop_maxiterations {
            BO_PARAM(int, iterations, 40);
        };

        struct acqui_gpucb : public defaults::acqui_gpucb {
        };

        struct opt_gridsearch : public defaults::opt_gridsearch {
        };

        struct opt_rprop : public defaults::opt_rprop {
        };

        struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
        };
    };

     struct Eval {
        static constexpr size_t dim_in = 1;
        static constexpr size_t dim_out = 1;

        Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
        {
            Eigen::VectorXd res(1);
            res(0) = -((5 * x(0) - 2.5) * (5 * x(0) - 2.5)) + 5;
            return res;
        }
    };

    int main() {
        bayes_opt::BOptimizer<Params> boptimizer;
        boptimizer.optimize(Eval());
        std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
        return 0;
    }