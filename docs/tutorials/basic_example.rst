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
  |-- src
  ...

Next, copy the following content to the ``wscript`` file:

.. code:: python

    def options(opt):
        pass


    def build(bld):
        bld(features='cxx cxxprogram',
            source='main.cpp',
            includes='. ../../src',
            target='test',
            uselib='BOOST EIGEN TBB LIBCMAES NLOPT',
            use='limbo')

For this example, we will optimize a simple function: :math:`-{(5 \times x - 2.5)}^2 + 5`, using all default values and settings. If you did not compile with libcmaes and/or nlopt, remove LIBCMAES and/or NLOPT from 'uselib'.

To begin, the ``main`` file has to include the necessary files, and declare the ``Parameter struct``:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 3-57



Here we are stating that the samples are observed without noise (which makes sense, because we are going to evaluate the function), that we want to output the stats (by setting stats_enabled to `true`), that the model has to be initialized with 10 samples (that will be selected randomly), and that the optimizer should run for 40 iterations. The rest of the values are taken from the defaults.

Then, we have to define the evaluation function for the optimizer to call:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 59-72

It is required that the evaluation struct has the static members ``dim_in`` and ``dim_out``, specifying the input and output dimensions.
Also, it should have the ``operator()`` expecting a ``const Eigen::VectorXd&`` of size ``dim_in``, and return another one, of size ``dim_out``.

With this, we can declare the main function:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
   :lines: 74-83


Finally, from the root of limbo, run a build command, with the additional switch ``--exp test``: ::

    ./waf build --exp test

Then, an executable named ``test`` should be produced under the folder ``build/exp/test``.

Full ``main.cpp``:

.. literalinclude:: ../../src/tutorials/basic_example.cpp
   :language: c++
   :linenos:
