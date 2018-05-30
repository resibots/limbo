.. _basic-example:

Basic Example
=================================================
If you are not familiar with the main concepts of Bayesian Optimization, a quick introduction is available :ref:`here <bayesian_optimization>`.
In this tutorial, we will explain how to create a new experiment in which a simple function ( :math:`-{(5 * x - 2.5)}^2 + 5`) is maximized.

Let's say we want to create an experiment called "myExp". The first thing to do is to create the folder ``exp/myExp`` under the limbo root. Then add two files:

* the ``main.cpp`` file
* a python file called ``wscript``, which will be used by ``waf`` to register the executable for building

The file structure should look like this: ::

  limbo
  |-- exp
       |-- myExp
            +-- wscript
            +-- main.cpp
  |-- src
  ...

Next, copy the following content to the ``wscript`` file:

.. code:: python

   from waflib.Configure import conf

    def options(opt):
        pass


    def build(bld):
        bld(features='cxx cxxprogram',
            source='main.cpp',
            includes='. ../../src',
            target='myExp',
            uselib='BOOST EIGEN TBB LIBCMAES NLOPT',
            use='limbo')

For this example, we will optimize a simple function: :math:`-{(5 * x - 2.5)}^2 + 5`, using all default values and settings. If you did not compile with libcmaes and/or nlopt, remove LIBCMAES and/or NLOPT from 'uselib'.

To begin, the ``main`` file has to include the necessary files:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 48-53

We also need to declare the ``Parameter struct``:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 55-97


Here we are stating that the samples are observed without noise (which makes sense, because we are going to evaluate the function), that we want to output the stats (by setting stats_enabled to `true`), that the model has to be initialized with 10 samples (that will be selected randomly), and that the optimizer should run for 40 iterations. The rest of the values are taken from the defaults. **By default limbo optimizes in** :math:`[0,1]`, but you can optimize without bounds by setting ``BO_PARAM(bool, bounded, false)`` in ``bayes_opt_bobase`` parameters. If you do so, limbo outputs random numbers, wherever needed, sampled from a gaussian centered in zero with a standard deviation of :math:`10`, instead of uniform random numbers in :math:`[0,1]` (in the bounded case). Finally **limbo always maximizes**; this means that you have to update your objective function if you want to minimize.

Then, we have to define the evaluation function for the optimizer to call:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 98-112

It is required that the evaluation struct has the static function members ``dim_in()`` and ``dim_out()``, specifying the input and output dimensions.
Also, it should have the ``operator()`` expecting a ``const Eigen::VectorXd&`` of size ``dim_in()``, and return another one, of size ``dim_out()``.

With this, we can declare the main function:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 114-123

The full ``main.cpp`` can be found `here <../../src/tutorials/basic_example.cpp>`_

Finally, from the root of limbo, run a build command, with the additional switch ``--exp myExp``: ::

    ./waf build --exp myExp

Then, an executable named ``myExp`` should be produced under the folder ``build/exp/myExp``.
When running this executable, you should see something similar to this: 


.. literalinclude:: ./example_run_basic_example/print_test.dat

These lines show the result of each sample evaluation of the :math:`40` iterations (after the random initialization). In particular, we can see that algorithm progressively converges toward the maximum of the function (:math:`5`) and that the maximum found is located at :math:`x = 0.500014`.

Running the executable also created a folder with a name composed of YOUCOMPUTERHOSTNAME-DATE-HOUR-PID. This folder should contain two files: ::
  
  limbo
  |-- YOUCOMPUTERHOSTNAME-DATE-HOUR-PID
     +-- samples.dat
     +-- aggregated_observations.dat 


The file ``samples.dat`` contains the coordinates of the samples that have been evaluated during each iteration, while the file ``aggregated_observations.dat`` contains the corresponding observed values. 

If you want to display the different observations in a graph, you can use the python script ``print_aggregated_observations.py`` (located in ``limbo_root/src/tutorials``).
For instance, from the root of limbo you can run ::
  python src/tutorials/print_aggregated_observations.py YOUCOMPUTERHOSTNAME-DATE-HOUR-PID/aggregated_observations.dat


