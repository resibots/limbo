.. _compilation-tutorial:

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
* `Boost <http://www.boost.org>`_ , with the following libraries: filesystem, system, unit_test_framework (test), program_options, and thread; `Boost` is mainly used for the interaction with the system.
* `Eigen 3 <http://eigen.tuxfamily.org>`_, Eigen3 is a highly-efficient, templated-based C++ library for linear algebra.

Optional but highly recommended
+++++++++++++++++++++++++++++++++
* `Intel TBB <https://www.threadingbuildingblocks.org>`_ is not mandatory, but highly recommended; TBB is used in Limbo to take advantage of multicore architectures.

* `NLOpt <http://ab-initio.mit.edu/wiki/index.php/NLopt>`_ [mirror: http://members.loria.fr/JBMouret/mirrors/nlopt-2.4.2.tar.gz] with C++ binding: ::

    ./configure --with-cxx --enable-shared --without-python --without-matlab --without-octave
    sudo make install

.. caution::

  The Debian/Unbuntu NLOpt package does NOT come with C++ bindings. Therefore you need to compile NLOpt yourself. The brew package (OSX) comes with C++ bindings (`brew install nlopt`).

* `libcmaes <https://github.com/beniz/libcmaes>`_. We advise you to use our own `fork of libcmaes <https://github.com/resibots/libcmaes>`_ (branch **fix_flags_native**). Make sure that you install with **sudo** or configure the **LD_LIBRARY_PATH** accordingly. Be careful that gtest (which is a dependency of libcmaes) needs to be manually compiled **even if you install it with your package manager** (e.g. apt-get): ::

    sudo apt-get install libgtest-dev
    sudo cd /usr/src/gtest
    sudo mkdir build && cd build
    sudo cmake ..
    sudo make
    sudo cp *.a /usr/lib

Follow the instructions below (you can also have a look `here <https://github.com/resibots/libcmaes#build>`_): ::

    git clone https://github.com/resibots/libcmaes.git
    cd libcmaes
    git checkout fix_flags_native

Configuring with Makefiles: ::

   ./autogen.sh
   ./configure
   make -j4

or CMake: ::

    mkdir build
    cd build
    cmake ..
    make -j4

In addition, you should be careful to configure **libcmaes** to use the same Eigen3 version as what you intend to use with Limbo (configuring with Makefiles): ::

    ./configure --with-eigen3-include=YOUR_DESIRED_DIR/include/eigen3

or (configuring with CMake): ::

    cmake -DEIGEN3_INCLUDE_DIR=YOUR_DESIRED_DIR/include/eigen3 ..

Additionally, you can enable the usage of TBB for parallelization (configuring with Makefiles): ::

    ./configure --enable-tbb

or (configuring with CMake): ::

    cmake -DUSE_TBB=ON -DUSE_OPENMP=OFF ..

Optional
+++++++++++++
* `Intel MKL <https://software.intel.com/en-us/intel-mkl>`_ is supported as backend for Eigen. In our experience, it provided best results when compiling with Intel's Compiler (ICC)
* `LAPACKE/BLAS <http://www.netlib.org/lapack/lapacke.html>`_ is supported as a backend for Eigen (`version>=3.3 <https://eigen.tuxfamily.org/dox/TopicUsingBlasLapack.html>`_). In our experience, it gives high speed-ups with **big** matrices (i.e., more than 1200 dimensions) and hurts a bit the performance with **small** matrices (i.e., less than 800 dimensions). You can enable LAPACKE/BLAS by using the ``--lapacke_blas`` option (if you have Eigen3.3 or later).
* `Sferes2 <https://github.com/sferes2/sferes2>`_ if you plan to use the multi-objective bayesian optimization algorithms (experimental).

Compilation
----------------------------

We use  the `WAF <https://waf.io>`_  build system, which is provided with the **limbo** source code. To know why we use waf (and not CMAKE, SCONS, traditional makefiles, etc.), see the :ref:`FAQ <faq-waf>`.

Like most build systems, it has a configuration and build steps, described bellow.

Configuration
~~~~~~~~~~~~~

.. caution::
  Make sure that the waf file has execution rights.

The first step is to configure your waf environment. For this, assuming that you are in the root directory of  Limbo, you have to run the command: ::

    ./waf configure

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

Waf should automatically detect the libraries if they where installed in the default folders, but if it doesn't,
you can use the following command-line options to indicate where they are:

* ``--libcmaes=/path/to/libcmaes``
* ``--nlopt=/path/to/nlopt``
* ``--tbb=/path/to/tbb``
* ``--mkl=/path/to/mkl``
* ``--sferes=/path/to/sferes2``
* ``--boost-includes /path/to/boost-includes`` [.h]
* ``--boost-libs /path/to/boost-libraries`` [.a, .so, .dynlib]
* ``--eigen /path/to/eigen3``


Note that Sferes2 won't be used unless you specify it's installation folder.
You can also specify a different compiler than the default, setting the environment variables ``CC`` and ``CXX``.

A full example::

    CC=icc CXX=icpc ./waf configure --sferes ~/sferes2 --mkl ~/intel/mkl --tbb ~/intel/tbb

Build
~~~~~~~~~~~~~

The second step is to run the build command::

    ./waf build

Depending on your compiler, there may be some warnings, but the output should end with the following lines: ::

    'build' finished successfully (time in sec)
