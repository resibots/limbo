Installation and Compilation
=================================================

Installation
----------------------------

To install **limbo**, simply clone the source code from https://github.com/resibots/limbo with git, or download it
as a zip.

Dependencies
~~~~~~~~~~~~~

Required
+++++++++++++
* `Boost <http://www.boost.org>`_ , with the following libraries: serialization, filesystem, system, unit_test_framework, program_options, graph and thread
* `Eigen <http://eigen.tuxfamily.org>`_

Optional
+++++++++++++
* `Intel TBB <https://www.threadingbuildingblocks.org>`_ is not mandatory, but highly recommended
* `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ is supported as backend for Eigen. In our experience, it provided best results when compiling with Intel's Compiler
* `Sferes2 <https://github.com/sferes2/sferes2>`_ if you plan to use the multi-objective bayesian optimization algorithms

Compilation
----------------------------

We use  the `waf <https://waf.io>`_  build system, which is provided with the **limbo** source code.

Configuration
~~~~~~~~~~~~~

The first step is to configure your ``waf`` environment. For this, assuming that you are in the root limbo directory, you have to run the command: ::

    ./waf configure

Make sure that the ``waf`` file has execution rights.
If everything is okay, you should expect an output like this: ::

    Setting top to                           : /path/to/limbo
    Setting out to                           : /path/to/limbo/build 
    Checking for 'g++' (c++ compiler)        : /usr/bin/g++ 
    Checking for 'gcc' (c compiler)          : /usr/bin/gcc 
    Checking boost includes                  : 1_55 
    Checking boost libs                      : ok 
    Checking Intel TBB includes              : not found
    Checking for compiler option to support OpenMP : -fopenmp 
    Checking Intel MKL includes                    : not found
    ['-Wall', '-std=c++11', '-O3', '-march=native', '-g']

The actual ouput may differ, depending on your configuration and installed libraries.

``waf`` should automatically detect Intel's TBB and MKL, if they where installed in the default folders, but if it doesn't,
you can use the following command-line options to indicate where they are:

* ``--tbb /path/to/tbb``
* ``--mkl /path/to/mkl``
* ``--sferes /path/to/sferes2``

Note that Sferes2 won't be used unless you specify it's installation folder.
You can also specify a different compiler than the default, setting the environment variables ``CC`` and ``CXX``.

Some examples: ::

    CC=icc CXX=icpc ./waf configure --sferes /path/to/sferes2 --mkl /path/to/mkl --tbb /path/to/tbb


::

   CC=clang-3.6 CXX=clang++-3.6 ./waf configure --sferes /path/to/sferes2

Build
~~~~~~~~~~~~~

The second step is to run the build command:::

    ./waf build

Depending on your compiler, there may be some warnings, but the output should end with the following lines: ::

    execution summary 
      tests that pass 5/5 
        /path/to/limbo/build/src/tests/test_macros
        /path/to/limbo/build/src/tests/test_optimizers
        /path/to/limbo/build/src/tests/test_init_functions
        /path/to/limbo/build/src/tests/test_gp
        /path/to/limbo/build/src/tests/test_boptimizer
