Download and Compilation
=================================================

Download
----------------------------

To get **limbo**, simply clone the source code from https://github.com/resibots/limbo with git, or download it
as a zip.

Dependencies
~~~~~~~~~~~~~

Required
+++++++++++++
* `Boost <http://www.boost.org>`_ , with the following libraries: filesystem, system, unit_test_framework, program_options, and thread; `Boost` is mainly used for the interaction with the system.
* `Eigen 3 <http://eigen.tuxfamily.org>`_, Eigen3 is a highly-efficient, templated-based C++ library for linear algebra.

Optional
+++++++++++++
* `Intel TBB <https://www.threadingbuildingblocks.org>`_ is not mandatory, but highly recommended; TBB is used in Limbo to take advantage of multicore architectures.
* `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ is supported as backend for Eigen. In our experience, it provided best results when compiling with Intel's Compiler (ICC)
* `Sferes2 <https://github.com/sferes2/sferes2>`_ if you plan to use the multi-objective bayesian optimization algorithms (experimental).

Compilation
----------------------------

We use  the `WAF <https://waf.io>`_  build system, which is provided with the **limbo** source code. To know why we use waf (and not CMAKE, SCONS, traditional makefiles, etc.), see the :ref:`FAQ <faq-waf>`.

Configuration
~~~~~~~~~~~~~

The first step is to configure your waf environment. For this, assuming that you are in the main limbo directory, you have to run the command: ::

    ./waf configure

Make sure that the waf file has execution rights.
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

Waf should automatically detect Intel's TBB and MKL, if they where installed in the default folders, but if it doesn't,
you can use the following command-line options to indicate where they are:

* ``--tbb /path/to/tbb``
* ``--mkl /path/to/mkl``
* ``--sferes /path/to/sferes2``

Note that Sferes2 won't be used unless you specify it's installation folder.
You can also specify a different compiler than the default, setting the environment variables ``CC`` and ``CXX``.

A full example: ::

    CC=icc CXX=icpc ./waf configure --sferes ~/sferes2 --mkl ~/intel/mkl --tbb ~/intel/tbb

Build
~~~~~~~~~~~~~

The second step is to run the build command:::

    ./waf build

Depending on your compiler, there may be some warnings, but the output should end with the following lines: ::

    execution summary
      tests that pass 5/5
        /home/fallocat/limbo_git/build/src/tests/test_macros
        /home/fallocat/limbo_git/build/src/tests/test_optimizers
        /home/fallocat/limbo_git/build/src/tests/test_init_functions
        /home/fallocat/limbo_git/build/src/tests/test_gp
        /home/fallocat/limbo_git/build/src/tests/test_boptimizer
